# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""PyTorch/TorchScript tensor helpers for LR-TT weights.

compose_lrtt_weights() forms the effective weights W_eff = C + alpha * A @ B
and is used by LRTTSimulatorTile.get_effective_weights(). The other functions
are general tensor helpers that the LR-TT tile does not call:
- pack/unpack of the first K columns or rows
- transpose of chunk buffers
- scatter of rank rows or columns into padded buffers
- AXPY accumulation and chunked outer products
- Kaiming-normal initialization of rank rows and extraction of submatrices
- the projections B @ x and A^T @ d
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def pack_first_k_columns(matrix: Tensor, k: int) -> Tensor:
    """Pack first K columns from matrix.
    
    Args:
        matrix: Input matrix [rows, cols]
        k: Number of columns to extract
        
    Returns:
        Packed matrix [rows, k]
    """
    return matrix[:, :k].contiguous()


def pack_first_k_rows(matrix: Tensor, k: int) -> Tensor:
    """Pack first K rows from matrix.
    
    Args:
        matrix: Input matrix [rows, cols]  
        k: Number of rows to extract
        
    Returns:
        Packed matrix [k, cols]
    """
    return matrix[:k, :].contiguous()


def unpack_to_k_columns(packed: Tensor, target_cols: int, k: int) -> Tensor:
    """Unpack K columns into larger matrix, zero-padding the rest.
    
    Args:
        packed: Packed matrix [rows, k]
        target_cols: Total columns in output
        k: Number of active columns
        
    Returns:
        Unpacked matrix [rows, target_cols] with zeros in columns [k:]
    """
    rows = packed.size(0)
    result = torch.zeros(rows, target_cols, device=packed.device, dtype=packed.dtype)
    result[:, :k] = packed
    return result


def unpack_to_k_rows(packed: Tensor, target_rows: int, k: int) -> Tensor:
    """Unpack K rows into larger matrix, zero-padding the rest.
    
    Args:
        packed: Packed matrix [k, cols]
        target_rows: Total rows in output
        k: Number of active rows
        
    Returns:
        Unpacked matrix [target_rows, cols] with zeros in rows [k:]
    """
    cols = packed.size(1)
    result = torch.zeros(target_rows, cols, device=packed.device, dtype=packed.dtype)
    result[:k, :] = packed
    return result


def scatter_rank_rows_to_padded(
    rank_data: Tensor,     # [rank, cols]
    target_rows: int,      # Total rows in output
    start_row: int = 0     # Starting row index
) -> Tensor:
    """Scatter rank rows into padded buffer at specified position.
    
    Args:
        rank_data: Rank data [rank, cols]
        target_rows: Total rows in output buffer
        start_row: Starting row index to place rank_data
        
    Returns:
        Padded matrix [target_rows, cols] with rank_data at [start_row:start_row+rank]
    """
    rank, cols = rank_data.shape
    result = torch.zeros(target_rows, cols, device=rank_data.device, dtype=rank_data.dtype)
    
    end_row = min(start_row + rank, target_rows)
    actual_rank = end_row - start_row
    
    if actual_rank > 0:
        result[start_row:end_row, :] = rank_data[:actual_rank, :]
        
    return result


def scatter_rank_cols_to_padded(
    rank_data: Tensor,     # [rows, rank] 
    target_cols: int,      # Total cols in output
    start_col: int = 0     # Starting col index
) -> Tensor:
    """Scatter rank columns into padded buffer at specified position.
    
    Args:
        rank_data: Rank data [rows, rank]
        target_cols: Total columns in output buffer
        start_col: Starting column index to place rank_data
        
    Returns:
        Padded matrix [rows, target_cols] with rank_data at [:, start_col:start_col+rank]
    """
    rows, rank = rank_data.shape
    result = torch.zeros(rows, target_cols, device=rank_data.device, dtype=rank_data.dtype)
    
    end_col = min(start_col + rank, target_cols)
    actual_rank = end_col - start_col
    
    if actual_rank > 0:
        result[:, start_col:end_col] = rank_data[:, :actual_rank]
        
    return result


def transpose_chunk_buffer(
    buffer: Tensor,
    chunk_rows: int,
    chunk_cols: int
) -> Tensor:
    """Transpose a chunk buffer efficiently.
    
    Args:
        buffer: Input buffer [chunk_rows, chunk_cols]
        chunk_rows: Expected chunk rows
        chunk_cols: Expected chunk cols
        
    Returns:
        Transposed buffer [chunk_cols, chunk_rows]
    """
    if buffer.size(0) != chunk_rows or buffer.size(1) != chunk_cols:
        # Resize if needed
        resized = torch.zeros(chunk_rows, chunk_cols, device=buffer.device, dtype=buffer.dtype)
        min_rows = min(buffer.size(0), chunk_rows)
        min_cols = min(buffer.size(1), chunk_cols)
        resized[:min_rows, :min_cols] = buffer[:min_rows, :min_cols]
        buffer = resized
        
    return buffer.t().contiguous()


def axpy_accumulate(
    y: Tensor,          # Target matrix to accumulate into
    alpha: float,       # Scale factor
    x: Tensor,          # Source matrix
    in_place: bool = True
) -> Tensor:
    """AXPY operation: y = y + alpha * x (digital accumulation).
    
    Args:
        y: Target matrix to accumulate into
        alpha: Scale factor
        x: Source matrix (same shape as y)
        in_place: Whether to modify y in-place or return new tensor
        
    Returns:
        Result tensor (y if in_place, else new tensor)
    """
    if y.shape != x.shape:
        raise ValueError(f"Shape mismatch: y{y.shape} vs x{x.shape}")
        
    if in_place:
        y.add_(x, alpha=alpha)
        return y
    else:
        return y + alpha * x


def outer_product_chunked(
    left: Tensor,       # [rows, rank_chunk]
    right: Tensor,      # [rank_chunk, cols] 
    target: Optional[Tensor] = None,  # [rows, cols] optional accumulation target
    alpha: float = 1.0,
    in_place: bool = False
) -> Tensor:
    """Chunked outer product: result = alpha * left @ right (+ target).

    Args:
        left: Left matrix [rows, rank_chunk]
        right: Right matrix [rank_chunk, cols]
        target: Optional target to accumulate into [rows, cols]
        alpha: Scale factor
        in_place: Whether to accumulate in-place into target
        
    Returns:
        Result matrix [rows, cols]
    """
    # Compute outer product  
    outer = left @ right
    
    if alpha != 1.0:
        outer = outer * alpha
        
    if target is not None:
        if in_place:
            target.add_(outer)
            return target
        else:
            return target + outer
    else:
        return outer


def kaiming_init_rank_rows(
    full_matrix: Tensor,    # [full_rows, cols] matrix to initialize  
    rank: int,              # Number of rows to initialize
    fan_in: int,            # Fan-in for Kaiming calculation
    gain: float = 1.0,      # Gain factor
    start_row: int = 0      # Starting row index
) -> None:
    """Initialize first rank rows with Kaiming Normal, zero rest.
    
    Args:
        full_matrix: Full matrix to initialize [full_rows, cols]
        rank: Number of rows to initialize with Kaiming
        fan_in: Fan-in for Kaiming std calculation
        gain: Gain factor for std
        start_row: Starting row index for Kaiming initialization
    """
    # Zero the full matrix first
    full_matrix.zero_()
    
    # Kaiming Normal for active rank rows
    if rank > 0:
        std = gain * torch.sqrt(torch.tensor(2.0 / fan_in))
        end_row = min(start_row + rank, full_matrix.size(0))
        actual_rank = end_row - start_row
        
        if actual_rank > 0:
            torch.nn.init.normal_(
                full_matrix[start_row:end_row, :], 
                mean=0.0, 
                std=std.item()
            )


def extract_rank_submatrix(
    full_matrix: Tensor,    # [full_rows, full_cols]
    rank_rows: int,         # Number of rows to extract
    rank_cols: int,         # Number of cols to extract  
    start_row: int = 0,     # Starting row
    start_col: int = 0      # Starting col
) -> Tensor:
    """Extract rank submatrix from full matrix.
    
    Args:
        full_matrix: Full matrix [full_rows, full_cols]
        rank_rows: Number of rows to extract
        rank_cols: Number of columns to extract
        start_row: Starting row index
        start_col: Starting column index
        
    Returns:
        Extracted submatrix [rank_rows, rank_cols]
    """
    end_row = min(start_row + rank_rows, full_matrix.size(0))
    end_col = min(start_col + rank_cols, full_matrix.size(1))
    
    return full_matrix[start_row:end_row, start_col:end_col].contiguous()


def compute_lora_projections(
    A_lr: Tensor,      # [d_size, rank] left factor
    B_lr: Tensor,      # [rank, x_size] right factor  
    x: Tensor,         # [x_size, batch] input
    d: Tensor          # [d_size, batch] error
) -> Tuple[Tensor, Tensor]:
    """Compute LoRA projections for A/B updates.
    
    Args:
        A_lr: A matrix left factor [d_size, rank] 
        B_lr: B matrix right factor [rank, x_size]
        x: Input tensor [x_size, batch]
        d: Error tensor [d_size, batch]
        
    Returns:
        Tuple of (X_B, D_A):
        - X_B: B projection [rank, batch] = B_lr @ x
        - D_A: A projection [rank, batch] = A_lr^T @ d
    """
    # X_B = B_lr @ x: [rank, x_size] @ [x_size, batch] -> [rank, batch]
    X_B = B_lr @ x
    
    # D_A = A_lr^T @ d: [rank, d_size] @ [d_size, batch] -> [rank, batch]
    D_A = A_lr.t() @ d
    
    return X_B, D_A


def compose_lrtt_weights(
    visible_weights: Tensor,    # [d_size, x_size] 
    A_weights: Tensor,          # [d_size, rank]
    B_weights: Tensor,          # [rank, x_size]
    lora_alpha: float = 1.0,
    rank: Optional[int] = None  # Use only first rank components
) -> Tensor:
    """Compose LRTT effective weights: W_eff = W_visible + alpha * A @ B.
    
    Args:
        visible_weights: Main weights [d_size, x_size]
        A_weights: A matrix [d_size, rank_full]
        B_weights: B matrix [rank_full, x_size] 
        lora_alpha: LoRA scaling factor
        rank: Use only first rank components (None = use all)
        
    Returns:
        Effective weights [d_size, x_size]
    """
    if rank is not None:
        # Use only first rank components
        A_active = A_weights[:, :rank]  # [d_size, rank]
        B_active = B_weights[:rank, :]  # [rank, x_size]
    else:
        A_active = A_weights
        B_active = B_weights
        
    # Compose: W_eff = W_visible + alpha * A @ B
    AB_product = A_active @ B_active  # [d_size, x_size]
    W_eff = visible_weights + lora_alpha * AB_product
    
    return W_eff


# TorchScript compilation for performance (optional)
try:
    # Compile key kernels with TorchScript
    pack_first_k_columns = torch.jit.script(pack_first_k_columns)
    pack_first_k_rows = torch.jit.script(pack_first_k_rows)  
    axpy_accumulate = torch.jit.script(axpy_accumulate)
    outer_product_chunked = torch.jit.script(outer_product_chunked)
    compute_lora_projections = torch.jit.script(compute_lora_projections)
    compose_lrtt_weights = torch.jit.script(compose_lrtt_weights)
except Exception:
    # Keep the Python functions if TorchScript compilation is not available
    pass