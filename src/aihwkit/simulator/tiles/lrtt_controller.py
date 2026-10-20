# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""LR-TT controller: update and transfer of the A, B and C tiles of an LR-TT layer.

Runs on top of three aihwkit tiles, A ([d_size, rank]), B ([rank, x_size]) and
the core tile C ([d_size, x_size]), and implements:
- the pulsed updates of A and B from the projections B@x and A^T@d
- the transfer of A @ B into C (rank-wise one-hot read, direct or exact)
- the reinitialization of A and B and the transfer schedule
- the optional forward injection y = C@x + alpha*A@(B@x)

Notation: the code's A ([d_size, rank]) and B ([rank, x_size]) are the factors
written B (M x r) and A (r x N) in the paper; the transferred product A @ B is BA.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict, Any
import math

from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery
from aihwkit.simulator.tiles.floating_point import FloatingPointTile
from aihwkit.simulator.parameters.enums import PulseType


class LRTTController:
    """LR-TT controller for the three tiles A, B and C.

    Tiles:
    - tile_a: A weights [d_size, rank]
    - tile_b: B weights [rank, x_size]
    - tile_c: core (visible) weights C [d_size, x_size]

    Core operations:
    1. reinit(): initialization / reinitialization of A and B (reinit_mode)
    2. ab_weight_update(): pulsed updates of A and B from the projections
    3. ab_weight_transfer(): A@B -> C transfer, then reinit of the transferred ranks
    4. forward_inject(): y = C@x + alpha*A@(B@x) composition
    """

    def __init__(
        self,
        tile_a: AnalogTileWithoutPeriphery,   # A [d_size, rank]
        tile_b: AnalogTileWithoutPeriphery,   # B [rank, x_size]
        tile_c: AnalogTileWithoutPeriphery,   # C (visible) [d_size, x_size]
        d_size: int,
        x_size: int,
        rank: int,
        *,
        transfer_lr: float = 1.0,
        transfer_every: int = 32,
        dynamic_te: bool = False,
        dynamic_te_power: float = 1.0,
        dynamic_te_min: Optional[int] = None,
        dynamic_te_max: Optional[int] = None,
        te_warmup_schedule: Optional[List[int]] = None,
        te_warmup_steps: int = 0,
        units_in_mbatch: bool = False,
        lora_alpha: float = 1.0,
        reinit_gain: float = 1.0,
        reinit_mode: str = "standard",
        a_init_mode: str = "zero",  # "zero" or "kaiming"
        b_init_mode: str = "kaiming",  # "kaiming" or "zero"
        correct_gradient_magnitudes: bool = False,
        fast_lr: float = 1.0,
        scale_transfer_lr: bool = True,
        transfer_fast_lr_ref: str = "geomean",
        rank_chunk: Optional[int] = None,
        ab_bl_mgmt: Optional[Dict[str, Any]] = None,
        transfer_bl_mgmt: Optional[Dict[str, Any]] = None,
        forward_inject: bool = False,
        num_reads: int = 1,
        multi_read_mode: str = "average",
        update_mode: str = "lora",  # "lora" or "reconstruction"
        transfer_method: str = "onehot",  # "onehot", "direct", or "set"
        transfer_rank_schedule: str = "all",  # "all" or "round_robin"
        transfer_ranks_per_step: int = 1,  # ranks per transfer in round_robin mode
        fi_continuous_alpha: bool = False,  # alpha = eff_transfer_lr when FI=True
        device: Optional[torch.device] = None,  # device of the controller tensors
        dtype: torch.dtype = torch.float32      # dtype of the controller tensors
    ):
        """Initialize LR-TT controller.

        Args:
            tile_a: Tile of the A matrix [d_size, rank]
            tile_b: Tile of the B matrix [rank, x_size]
            tile_c: Tile of the core matrix C [d_size, x_size]
            d_size: Output dimension
            x_size: Input dimension
            rank: LoRA rank (must be <= min(d_size, x_size))
            transfer_lr: Transfer learning rate scalar
            transfer_every: Transfer frequency (steps or samples)
            units_in_mbatch: Whether transfer_every counts samples vs steps
            lora_alpha: LoRA scaling factor alpha
            reinit_gain: Kaiming uniform initialization multiplier (1.0 = standard PyTorch default)
            reinit_mode: Treatment of A and B after a transfer:
                        "standard" - A and B reinitialized after every transfer
                        "decay" - A and B initialized once and kept
                        "hybrid" - A reset after every transfer, B initialized once and kept
            a_init_mode, b_init_mode: Initialization of A and B (see PythonLRTTDevice)
            correct_gradient_magnitudes: Divide transfer LR by fast effective LR reference
            fast_lr, scale_transfer_lr, transfer_fast_lr_ref: see PythonLRTTDevice
            rank_chunk: Chunk size for transfer (None = full rank)
            ab_bl_mgmt, transfer_bl_mgmt: stored as attributes; not read by the
                   update or the transfer
            forward_inject: Enable forward injection y = C@x + alpha*A@(B@x)
            dynamic_te, dynamic_te_power, dynamic_te_min, dynamic_te_max,
            te_warmup_schedule, te_warmup_steps: transfer-period schedule
                   (see _update_dynamic_te)
            num_reads: Number of reads per rank during one-hot transfer (default 1)
            multi_read_mode: How to handle multiple reads: 'average' or 'per_read'
            update_mode: A/B update mode: 'lora' (projected update) or
                   'reconstruction' (gradient reconstruction)
            transfer_method: Transfer method: "onehot", "direct", or "set" (exact floating-point update)
            transfer_rank_schedule, transfer_ranks_per_step: ranks per transfer
                   ("all" or "round_robin")
            fi_continuous_alpha: alpha = effective transfer LR with forward injection
            device: Device of the controller tensors (if None: CUDA for CUDA tiles
                   or when CUDA is available, otherwise CPU; LRTTSimulatorTile keeps
                   it in sync with the tiles through set_device())
            dtype: Explicit dtype for tensors
        """
        if rank <= 0 or rank > min(d_size, x_size):
            raise ValueError(f"Invalid rank {rank} for dimensions {d_size}x{x_size}")

        self.tile_a = tile_a
        self.tile_b = tile_b
        self.tile_c = tile_c

        self.d_size = d_size
        self.x_size = x_size
        self.rank = rank

        # LRTT parameters
        self.transfer_lr = transfer_lr

        self.transfer_every = transfer_every
        self.units_in_mbatch = units_in_mbatch
        self.lora_alpha = lora_alpha
        self.reinit_gain = reinit_gain
        self.reinit_mode = reinit_mode
        self.a_init_mode = a_init_mode
        self.b_init_mode = b_init_mode
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.fast_lr = fast_lr
        self.scale_transfer_lr = scale_transfer_lr
        self.transfer_fast_lr_ref = transfer_fast_lr_ref
        self.auto_scale_mode = "none"    # Set from post_init
        self.auto_momentum = 0.99        # Set from post_init
        # EMA state: lazy init (first update creates 0-dim CUDA tensors)
        self.m_x = None
        self.m_d = None
        self.m_xb = None    # separate mode only
        self.m_da = None     # separate mode only
        self._autoscale_ema_initialized = False
        self.gran_a = 1.0    # Set from lrtt_tile.py for separate mode
        self.gran_b = 1.0
        self.last_lr_eff_a = 1.0   # Python float (after .item())
        self.last_lr_eff_b = 1.0
        self._last_lr_sgd = 0.0  # optimizer LR of the last update (0 before the first update)
        self.rank_chunk = rank_chunk or rank
        self.forward_inject_enabled = forward_inject
        self.fi_continuous_alpha = fi_continuous_alpha

        # Dynamic TE state
        self.dynamic_te = dynamic_te
        self.dynamic_te_power = dynamic_te_power
        self.te_base = transfer_every  # TE_0 (immutable)
        self.dynamic_te_min = dynamic_te_min if dynamic_te_min is not None else transfer_every
        self.dynamic_te_max = dynamic_te_max if dynamic_te_max is not None else transfer_every * 10
        self.lr_peak: Optional[float] = None  # Auto-tracked peak LR

        # TE warmup ramp state
        self.te_warmup_schedule = te_warmup_schedule  # e.g. [32, 64, 128, 230]
        self.te_warmup_steps = te_warmup_steps
        self.te_step_counter = 0  # Counts _update_dynamic_te calls

        # If warmup schedule is set, start TE at the first value
        if self.te_warmup_schedule and len(self.te_warmup_schedule) > 0:
            self.transfer_every = self.te_warmup_schedule[0]

        self.num_reads = num_reads
        self.multi_read_mode = multi_read_mode
        self.update_mode = update_mode
        self.transfer_method = transfer_method
        self.transfer_rank_schedule = transfer_rank_schedule
        self.transfer_ranks_per_step = transfer_ranks_per_step
        self._rr_rank_idx: int = 0  # Round-robin pointer: next transfer start rank

        # BL management settings
        self.ab_bl_mgmt = ab_bl_mgmt or {}
        self.transfer_bl_mgmt = transfer_bl_mgmt or {}

        # Diagnostics: record last_transfer_delta at every transfer (off by default;
        # the BERT fine-tuning script enables it)
        self.enable_diagnostics: bool = False

        # Counters and state
        self.transfer_counter = 0
        self._last_m_batch = 1  # Last mini-batch size (for units_in_mbatch)
        self.num_a_updates = 0
        self.num_b_updates = 0
        self.num_transfers = 0
        self.last_transfer_delta = None
        self.last_actual_delta = None

        # Cached buffers for efficiency
        self._x_b_buffer: Optional[Tensor] = None
        self._d_a_buffer: Optional[Tensor] = None
        self._pad_buffer_a: Optional[Tensor] = None
        self._pad_buffer_b: Optional[Tensor] = None

        # C tile bound cache (not read by the transfer methods)
        self._c_bounds_cached: bool = False
        self._c_max_bound: Optional[Tensor] = None  # None means no clipping needed
        self._c_min_bound: Optional[Tensor] = None

        # Device info - infer from tiles if not provided
        if device is None:
            # From the tile type and the CUDA availability
            device = self._infer_device_from_tile()
        self.device = device
        self.dtype = dtype

        # Initialization flags: C is set by the layer (set_weights); A and B by reinit()
        self._c_initialized = True
        self._tiles_initialized = False

        # Transfer options
        self.transfer_micro_steps: int = 1          # M: number of micro-transfer repetitions
        self.transfer_centering: bool = False       # Remove row/column means of the read A columns / B rows
        self.transfer_normalize: bool = False       # Per-rank L2 normalization of the read A columns / B rows

        # Transfer mode: "off", "pilot" (micro-stepped, optional gain calibration) or "sigma_delta"
        self.transfer_mode: str = "off"             # {"pilot", "sigma_delta", "off"}
        self.transfer_gamma_mode: str = "off"       # "pilot": run the pilot gain measurement in transfer_mode "pilot";
                                                    # also used as the mode when transfer_mode is empty
        self.transfer_pilot_frac: float = 1.0/16.0  # Pilot transfer lr = |transfer_lr| * frac

        # Sigma-Delta state for "sigma_delta" mode
        self.sd_quantum: Optional[float] = None     # g: unit quantum for rank-wise pulses (None -> derive per transfer)
        self.sd_acc: Optional[Tensor] = None        # h_k residuals [rank], persistent across transfers

        # --- Read averaging ---
        self.read_n_avg: int = 1                    # Reads averaged per rank (1 = single read)
        self.differential_read: bool = False        # Use +e/-e differential (True) or +e only (False)

        # --- AGC (Automatic Gain Control) settings ---
        self.agc_enabled: bool = False              # Enable AGC for read amplitude optimization
        self.agc_margin: float = 0.85               # Target output bound fraction (avoid clipping)
        self.agc_max_iters: int = 6                 # Max iterations for AGC binary search

        # --- Two-Amplitude differential read settings ---
        self.two_amp_enabled: bool = False          # Enable two-amplitude differential read (odd offset removal)
        self.two_amp_ratio: float = 0.5             # Ratio a1/a2 for two-amplitude method

        # === Reconstruction update parameters (for update_mode='reconstruction') ===
        self.recon_lambda_a: float = 1e-3
        self.recon_lambda_b: float = 1e-3
        self.recon_use_scalar_stabilizer: bool = False
        self.recon_use_exact_gram: bool = False
        self.recon_exact_gram_every: int = 0  # 0 = disabled, N = every N steps
        self.recon_ema_beta: float = 0.9
        self.recon_lr_scale: float = 1.0
        self.recon_clip_norm: float = 10.0
        self.recon_use_clip_norm: bool = False

        # EMA state for scalar stabilizer (sA = ||A||^2/rank, sB = ||B||^2/rank)
        self._ema_sA: float = 0.0
        self._ema_sB: float = 0.0
        self._ema_initialized: bool = False
        self._recon_step_counter: int = 0

        # Transfer one-hot vectors cache
        self._transfer_vec_a: Optional[Tensor] = None

        # === Logging of the A/B update inputs ===
        self.log_ab_scaling: bool = False
        """Enable logging of x,d max values during A/B updates."""

        self.log_ab_scaling_every: int = 10
        """Log x,d max values every N steps (only when log_ab_scaling=True)."""

        self._ab_update_step: int = 0
        """Internal counter for A/B update logging."""

        # === Manual-scaling (hardware) mode flag ===
        self._hardware_mode: Optional[bool] = None
        """Cache for hardware mode detection (use_manual_scaling=True)."""

    @property
    def effective_alpha(self) -> float:
        """Return the forward-injection scaling factor alpha.

        When fi_continuous_alpha is True and forward injection is enabled,
        alpha equals the effective transfer learning rate (transfer_lr, or
        transfer_lr * lr_sgd when scale_transfer_lr is True).
        Otherwise falls back to the static lora_alpha.
        """
        if self.fi_continuous_alpha and self.forward_inject_enabled:
            lr_tr = self.transfer_lr
            if self.scale_transfer_lr:
                lr_tr *= self._last_lr_sgd
            return lr_tr
        return self.lora_alpha

    def _is_hardware_mode(self) -> bool:
        """Check if tiles are in hardware mode (use_manual_scaling=True).

        In hardware mode, learning rate cannot be configured - weight updates
        are purely determined by pulse coincidences and dw_min.

        Returns:
            True if use_manual_scaling=True in tile config, False otherwise.
        """
        if self._hardware_mode is not None:
            return self._hardware_mode

        # Check tile_a's rpu_config for use_manual_scaling
        try:
            if hasattr(self.tile_a, 'rpu_config') and hasattr(self.tile_a.rpu_config, 'update'):
                self._hardware_mode = getattr(self.tile_a.rpu_config.update, 'use_manual_scaling', False)
            else:
                self._hardware_mode = False
        except Exception:
            self._hardware_mode = False

        return self._hardware_mode

    def _ensure_sd_state(self) -> None:
        """Ensure sigma-delta state tensors exist on the right device/dtype."""
        if self.sd_acc is None or self.sd_acc.numel() != self.rank or self.sd_acc.device != self.device:
            self.sd_acc = torch.zeros(self.rank, device=self.device, dtype=self.dtype)

    def _diff_read(self, tile, e: Tensor, amp: float, mode: str, read_n_avg: int) -> Tensor:
        """Differential read with amplitude scaling and averaging.

        Computes: d(amp) = 0.5 * (f(+amp*e) - f(-amp*e)) averaged over read_n_avg readings.

        This removes DC offset and even-order distortions through differential reading,
        and reduces stochastic noise by sqrt(read_n_avg) through averaging.

        Args:
            tile: Analog tile to read from (tile_a or tile_b)
            e: One-hot vector [1, rank]
            amp: Input amplitude scaling factor
            mode: "fwd" for forward pass, "bwd" for backward pass
            read_n_avg: Number of readings to average (noise reduction by sqrt(N))

        Returns:
            Differential read result: 0.5*(f(+amp*e) - f(-amp*e)) averaged
        """
        acc = None
        for _ in range(read_n_avg):
            if mode == "fwd":
                yp = tile.forward(amp * e)
                ym = tile.forward(-amp * e)
            else:  # "bwd"
                yp = tile.backward(amp * e)
                ym = tile.backward(-amp * e)
            d = 0.5 * (yp - ym)
            acc = d if acc is None else (acc + d)
        return acc / float(read_n_avg)

    def _pick_amp_agc(self, tile, e: Tensor, mode: str, margin: float = 0.85, max_iters: int = 6) -> float:
        """Automatic Gain Control: pick amplitude to maximize SNR without clipping.

        Uses binary search to find the largest amplitude that keeps output within
        margin * out_bound, maximizing signal strength while avoiding ADC saturation.

        Args:
            tile: Analog tile to probe
            e: One-hot vector [1, rank]
            mode: "fwd" for forward pass, "bwd" for backward pass
            margin: Target fraction of out_bound (0.85 = 85%)
            max_iters: Maximum binary search iterations

        Returns:
            Optimal amplitude for the tile read operation
        """
        # Get IO parameters from tile config
        # FloatingPointTile doesn't have forward/backward IO parameters
        io = getattr(tile.rpu_config, 'forward' if mode == "fwd" else 'backward', None)
        if io is None:
            return 1.0  # FloatingPointTile: return the default amplitude 1.0
        out_bound = float(getattr(io, "out_bound", 0.0) or 0.0)
        inp_bound = float(getattr(io, "inp_bound", 1.0) or 1.0)

        # Start with maximum allowed input amplitude
        amp = min(1.0, inp_bound)

        # If no output bound defined, just use max input
        if out_bound <= 0.0:
            return amp

        for _ in range(max_iters):
            # Probe with current amplitude
            if mode == "fwd":
                yp = tile.forward(amp * e)
                ym = tile.forward(-amp * e)
            else:
                yp = tile.backward(amp * e)
                ym = tile.backward(-amp * e)

            raw_max = float(torch.max(yp.abs().amax(), ym.abs().amax()).item())

            # Binary search: decrease if clipping, increase if too low
            if raw_max > margin * out_bound and amp > 1e-4:
                amp *= 0.5
                continue
            if raw_max < 0.2 * out_bound and amp < inp_bound:
                amp = min(amp * 2.0, inp_bound)
                continue
            break

        return amp

    def _two_amp_read(self, tile, e: Tensor, mode: str, read_n_avg: int = 8, margin: float = 0.85) -> tuple:
        """Two-amplitude differential read to cancel odd offset.

        The one-hot output model is: d(a) = a*w_k + b_odd
        where w_k is the desired weight column and b_odd is odd-order distortion.

        Using two amplitudes a1, a2:
          d(a1) = a1*w_k + b_odd
          d(a2) = a2*w_k + b_odd
        Solving: w_k = (d(a2) - d(a1)) / (a2 - a1)

        Args:
            tile: Analog tile to read from
            e: One-hot vector [1, rank]
            mode: "fwd" or "bwd"
            read_n_avg: Number of readings per amplitude level (noise reduction by sqrt(N))
            margin: AGC margin for amplitude selection

        Returns:
            (w_hat: estimated weight, b_odd: estimated odd offset, (a1, a2): amplitudes used)
        """
        # Find optimal high amplitude using AGC
        a2 = self._pick_amp_agc(tile, e, mode=mode, margin=margin, max_iters=self.agc_max_iters)

        # Low amplitude is a fraction of high amplitude
        a1 = self.two_amp_ratio * a2

        # Read at both amplitudes
        d1 = self._diff_read(tile, e, amp=a1, mode=mode, read_n_avg=read_n_avg)
        d2 = self._diff_read(tile, e, amp=a2, mode=mode, read_n_avg=read_n_avg)

        # Solve for weight (cancel odd offset)
        denom = a2 - a1
        w_hat = (d2 - d1) / max(denom, 1e-12)

        # Estimate odd offset
        b_odd = d1 - a1 * w_hat

        return w_hat.squeeze(0), b_odd.squeeze(0), (a1, a2)

    def _infer_device_from_tile(self) -> torch.device:
        """Infer the device from the tile type and the CUDA availability.

        LRTTSimulatorTile keeps the device in sync with the tiles through
        set_device() when the tiles are moved.
        """
        # Check tile backend type
        if hasattr(self.tile_c, 'tile'):
            tile_str = str(type(self.tile_c.tile).__name__)
            if 'Cuda' in tile_str or 'CUDA' in tile_str:
                return torch.device('cuda')

        # Default based on CUDA availability
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_tile_device(self) -> torch.device:
        """Get device that tiles expect for operations."""
        # Cached device of the controller
        return self.device

    def _get_tile_dtype(self) -> torch.dtype:
        """Get common dtype from tiles."""
        # Cached dtype of the controller
        return self.dtype

    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure scratch buffers are allocated for given batch size."""
        if (self._x_b_buffer is None or
            self._x_b_buffer.size(-1) != batch_size):

            # Use cached device
            device = self.device

            # Projection buffers
            self._x_b_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype
            )
            self._d_a_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype
            )

            # Padding buffers must match the tile input dimensions:
            # the A tile expects [x_size, batch] inputs,
            # the B tile expects [d_size, batch] errors
            self._x_pad = torch.zeros(
                self.x_size, batch_size, device=device, dtype=self.dtype
            )
            self._d_pad = torch.zeros(
                self.d_size, batch_size, device=device, dtype=self.dtype
            )

    def reinit(self) -> None:
        """Reinit A,B matrices based on reinit_mode, a_init_mode, and b_init_mode.

        Reinit modes (reinit() runs at construction and after every transfer of all ranks):
        - "standard": initialize A and B at every call
        - "decay": initialize A and B at the first call only; afterwards they are
          kept (and decay only through the retention of their devices)
        - "hybrid": reset A at every call; initialize B at the first call only

        A initialization (controlled by a_init_mode):
        - "zero": device reset of all columns (reset value plus reset_std
          noise; zero for floating-point tiles)
        - "kaiming": Kaiming-uniform draw scaled by reinit_gain

        B initialization (controlled by b_init_mode):
        - "kaiming": Kaiming-uniform draw scaled by reinit_gain
        - "zero": B=0
        """
        with torch.no_grad():
            # Helper: set raw weights on inner tile directly (bypasses periphery)
            def _set_raw(tile, w):
                tile.tile.set_weights(w)

            # Helper: reset all columns with the device reset (reset value and
            # reset_std), the model of the capacitor discharge of a 6T1C cell,
            # instead of writing zeros through the update path.
            def _reset_tile(tile):
                if isinstance(tile, FloatingPointTile):
                    # Floating-point tiles have no device reset: set zeros
                    tile.tile.set_weights(torch.zeros_like(tile.tile.get_weights()))
                    return
                tile.tile.reset_columns(0, -1, 1.0)

            def _get_raw_gpu(tile):
                return tile.tile.get_weights().to(self.device)

            if self.reinit_mode == "standard":
                # A matrix: Use a_init_mode
                if self.a_init_mode == "kaiming":
                    A_init = torch.empty(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                    nn.init.kaiming_uniform_(A_init, a=math.sqrt(5))
                    A_init *= self.reinit_gain
                    _set_raw(self.tile_a, A_init)
                else:  # "zero"
                    _reset_tile(self.tile_a)

                # B matrix: Use b_init_mode
                if self.b_init_mode == "zero":
                    B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                else:  # "kaiming"
                    B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                    B_init *= self.reinit_gain
                _set_raw(self.tile_b, B_init)

            elif self.reinit_mode == "decay":
                # Decay mode: initialize at the first call only; A and B are kept
                # after a transfer and decay through the retention of their devices
                if not self._tiles_initialized:
                    # First time: Use a_init_mode for A initialization
                    if self.a_init_mode == "kaiming":
                        A_init = torch.empty(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(A_init, a=math.sqrt(5))
                        A_init *= self.reinit_gain
                        _set_raw(self.tile_a, A_init)
                    else:  # "zero"
                        _reset_tile(self.tile_a)

                    # B matrix: Use b_init_mode
                    if self.b_init_mode == "zero":
                        B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    else:  # "kaiming"
                        B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                        B_init *= self.reinit_gain
                    _set_raw(self.tile_b, B_init)
                # else: A and B are kept

            elif self.reinit_mode == "hybrid":
                # A reset at every call, B initialized at the first call only
                _reset_tile(self.tile_a)

                if not self._tiles_initialized:
                    # First time: Use b_init_mode
                    if self.b_init_mode == "zero":
                        B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    else:  # "kaiming"
                        B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                        B_init *= self.reinit_gain
                    _set_raw(self.tile_b, B_init)
                # else: B is kept

            else:
                raise ValueError(f"Unknown reinit_mode: {self.reinit_mode}. "
                                 f"Must be 'standard', 'decay' or 'hybrid'")

        # Apply device clipping if available
        if hasattr(self.tile_a, 'clip_weights'):
            self.tile_a.clip_weights()
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()

        # Kaiming-uniform initialization of C with forward injection, only if
        # _c_initialized is False (it is set to True at construction)
        if self.forward_inject_enabled and not self._c_initialized:
            # Kaiming uniform init (same as PyTorch nn.Linear default)
            C_init = torch.empty(self.d_size, self.x_size, device=self.device, dtype=self.dtype)
            nn.init.kaiming_uniform_(C_init, a=math.sqrt(5))
            C_init *= self.reinit_gain
            self.tile_c.set_weights(C_init)
            if hasattr(self.tile_c, 'clip_weights'):
                self.tile_c.clip_weights()
            self._c_initialized = True

        # Reset counters
        self.transfer_counter = 0
        self._tiles_initialized = True

    def _reinit_ranks(self, rank_indices: List[int]) -> None:
        """Reinit only the specified ranks of A and B after transfer.

        If all ranks are selected, delegates to full reinit().
        Otherwise, performs per-rank read-modify-write on A columns and B rows.

        Args:
            rank_indices: List of rank indices to reinitialize.
        """
        # All ranks selected -> full reinit()
        if len(rank_indices) == self.rank:
            self.reinit()
            return

        with torch.no_grad():
            def _get_raw(tile):
                return tile.tile.get_weights().to(self.device)

            def _set_raw(tile, w):
                tile.tile.set_weights(w)

            A_full = _get_raw(self.tile_a)  # [d_size, rank]
            B_full = _get_raw(self.tile_b)  # [rank, x_size]

            for k in rank_indices:
                if self.reinit_mode == "standard":
                    # A[:, k] = 0, B[k, :] = Kaiming
                    if self.a_init_mode == "kaiming":
                        col = torch.empty(self.d_size, 1, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(col, a=math.sqrt(5))
                        A_full[:, k] = col.squeeze(1) * self.reinit_gain
                    else:
                        A_full[:, k] = 0.0

                    if self.b_init_mode == "zero":
                        B_full[k, :] = 0.0
                    else:
                        row = torch.empty(1, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(row, a=math.sqrt(5))
                        B_full[k, :] = row.squeeze(0) * self.reinit_gain

                elif self.reinit_mode == "decay":
                    # A and B are kept
                    pass

                elif self.reinit_mode == "hybrid":
                    # A[:, k] = 0, B unchanged
                    A_full[:, k] = 0.0

            _set_raw(self.tile_a, A_full)
            _set_raw(self.tile_b, B_full)

        # Apply device clipping if available
        if hasattr(self.tile_a, 'clip_weights'):
            self.tile_a.clip_weights()
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()

        # Reset counters
        self.transfer_counter = 0

    def ab_weight_update(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """Update A and B with rank-r gradient approximation.

        When update_mode='lora': projected LR-TT update (_ab_weight_update_lora).
        When update_mode='reconstruction': gradient reconstruction
        (_ab_weight_update_reconstruction).

        Args:
            x: Input tensor
            d: Error tensor
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
        # Branch based on update_mode
        if self.update_mode == "reconstruction":
            return self._ab_weight_update_reconstruction(x, d, lr, in_trans, out_trans)
        else:
            return self._ab_weight_update_lora(x, d, lr, in_trans, out_trans)

    def _lazy_init_ema(self, device: torch.device) -> None:
        """Lazily initialize EMA tensors on the correct device."""
        if self._autoscale_ema_initialized:
            return
        self.m_x = torch.tensor(0.0, device=device)
        self.m_d = torch.tensor(0.0, device=device)
        if self.auto_scale_mode == "separate":
            self.m_xb = torch.tensor(0.0, device=device)
            self.m_da = torch.tensor(0.0, device=device)
        self._autoscale_ema_initialized = True

    @staticmethod
    def _update_ema_exact(m_old: Tensor, amax_seq: Tensor, momentum: float) -> Tensor:
        """N sequential per-sample EMA updates in closed form.

        Per-sample EMA of the auto-scale statistics, as in aihwkit's C++
        updateAutoScale (rpu_chopped_transfer_device.cpp):
          - Cold start (m_old <= 0): m = amax_seq[0]
          - Further samples: m = (1-tau) * m + tau * amax  with tau = (1-mu)/N

        Closed-form for N sequential applications:
            m[N] = (1-tau)^N * m_old + tau * sum_k (1-tau)^(N-1-k) * amax_seq[k]

        Args:
            m_old: Previous EMA value (scalar tensor)
            amax_seq: Per-sample amax sequence [N]
            momentum: auto_momentum (e.g., 0.99)
        Returns:
            New EMA value (scalar tensor)
        """
        n = amax_seq.numel()
        if n == 0:
            return m_old
        # All amax values come from .abs().amax() and are >= 0; every sample
        # enters the EMA. Only the cold start is handled separately:
        if float(m_old) <= 0.0:
            # Cold start: m = first value
            m_curr = amax_seq[0]
            if n == 1:
                return m_curr
            rest = amax_seq[1:]
            n_rest = n - 1
        else:
            m_curr = m_old
            rest = amax_seq
            n_rest = n
        tau = (1.0 - momentum) / n  # N = number of samples, as m_batch in the C++ code
        one_minus_tau = 1.0 - tau
        # Decay exponents (newest sample has smallest exponent -> largest weight)
        exponents = torch.arange(n_rest - 1, -1, -1, device=rest.device, dtype=rest.dtype)
        decay = one_minus_tau ** exponents
        return (one_minus_tau ** n_rest) * m_curr + tau * (decay * rest).sum()

    def _update_autoscale_ema(
        self, x: Tensor, d: Tensor, XB: Tensor, DA: Tensor, m_batch: int
    ) -> None:
        """Update EMA statistics and compute lr_eff_a/b for auto_scale_mode.

        Used by _ab_weight_update_lora and by the forward_inject=True hook.
        The EMA is the per-sample EMA of aihwkit's C++ chopped-transfer device
        (updateAutoScale, see _update_ema_exact).

        Args:
            x: Input activations [batch, ..., x_size]
            d: Gradient w.r.t. output [batch, ..., d_size]
            XB: B@x projection [batch, ..., rank]
            DA: A^T@d projection [batch, ..., rank]
            m_batch: Mini-batch size (not used; tau uses the number of amax values)
        """
        self._lazy_init_ema(x.device)

        # Per-sample amax sequences (max over feature dim only, NOT batch-aggregate),
        # as aihwkit's C++ code computes Find_Absolute_Max per outer product.
        x_seq = x.abs().amax(dim=-1).reshape(-1).to(self.m_x.device)
        d_seq = d.abs().amax(dim=-1).reshape(-1).to(self.m_d.device)

        if self.auto_scale_mode == "shared":
            self.m_x = self._update_ema_exact(self.m_x, x_seq, self.auto_momentum)
            self.m_d = self._update_ema_exact(self.m_d, d_seq, self.auto_momentum)
            denom = self.m_x * self.m_d
            fast_lr_t = x.new_tensor(self.fast_lr)
            lr_eff = torch.where(denom > 0, fast_lr_t / denom, fast_lr_t).item()
            self.last_lr_eff_a = lr_eff
            self.last_lr_eff_b = lr_eff
        elif self.auto_scale_mode == "separate":
            xb_seq = XB.abs().amax(dim=-1).reshape(-1).to(self.m_xb.device)
            da_seq = DA.abs().amax(dim=-1).reshape(-1).to(self.m_da.device)
            self.m_x = self._update_ema_exact(self.m_x, x_seq, self.auto_momentum)
            self.m_d = self._update_ema_exact(self.m_d, d_seq, self.auto_momentum)
            self.m_xb = self._update_ema_exact(self.m_xb, xb_seq, self.auto_momentum)
            self.m_da = self._update_ema_exact(self.m_da, da_seq, self.auto_momentum)
            denom_a = self.m_xb * self.m_d
            fa = x.new_tensor(self.fast_lr * self.gran_a)
            self.last_lr_eff_a = torch.where(denom_a > 0, fa / denom_a, fa).item()
            denom_b = self.m_x * self.m_da
            fb = x.new_tensor(self.fast_lr * self.gran_b)
            self.last_lr_eff_b = torch.where(denom_b > 0, fb / denom_b, fb).item()

    def _ab_weight_update_lora(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """Projected LR-TT update of A and B.

        XB = B@x (tile_b forward) and DA = A^T@d (tile_a backward) are read
        from the tiles; A is updated with (XB, d) and B with (x, DA) by the
        pulsed updates of the tiles, with learning rates lr_eff_a / lr_eff_b.
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Projections (analog path, IO resolution controlled per-tile)
        with torch.no_grad():
            XB = self.tile_b.forward(x)     # [batch, rank] = B@X
            DA = self.tile_a.backward(d)    # [batch, rank] = A^T@D

        # 2) lr_eff_a, lr_eff_b via fast_lr + auto_scale
        self._last_lr_sgd = lr
        m_batch = x.shape[0]

        if self._is_hardware_mode():
            self.last_lr_eff_a = self.last_lr_eff_b = 1.0
        elif self.auto_scale_mode == "none":
            self.last_lr_eff_a = self.last_lr_eff_b = self.fast_lr
        else:
            self._update_autoscale_ema(x, d, XB, DA, m_batch)

        lr_eff_a = self.last_lr_eff_a
        lr_eff_b = self.last_lr_eff_b

        # Optional logging of the x,d max values of the A and B updates
        self._ab_update_step += 1
        if self.log_ab_scaling and (self._ab_update_step % self.log_ab_scaling_every == 1):
            # A tile update uses: x=XB (projection), d=d (gradient)
            a_x_max = XB.abs().max().item()
            a_d_max = d.abs().max().item()
            # B tile update uses: x=x (input), d=DA (projection)
            b_x_max = x.abs().max().item()
            b_d_max = DA.abs().max().item()

            print(f"  [AB-Scaling Step {self._ab_update_step}] "
                  f"A: x_max={a_x_max:.4f}, d_max={a_d_max:.4f} | "
                  f"B: x_max={b_x_max:.4f}, d_max={b_d_max:.4f} | "
                  f"lr_eff_a={lr_eff_a:.6f}, lr_eff_b={lr_eff_b:.6f}")

        # 3) dA = -lr_eff_a * D^T @ (B@X) -> tile_a.update(XB, d)
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_eff_a)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1

        # 4) dB = -lr_eff_b * (A^T@D)^T @ X -> tile_b.update(x, DA)
        lr_b_old = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_eff_b)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x, DA)
        else:
            self.tile_b.update(x, DA)
        self.tile_b.set_learning_rate(lr_b_old)
        self.num_b_updates += 1

        # 5) Dynamic TE update (before counter check)
        self._update_dynamic_te(lr)

        # 6) Counter (always += m_batch, matching TikiTaka convention)
        self._last_m_batch = m_batch
        self.transfer_counter += m_batch

    def _ab_weight_update_reconstruction(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """TikiTaka-style gradient reconstruction update for A and B.

        When forward_inject=False, the forward pass uses only C (y = Cx), so A,B
        don't appear in the loss. Instead of using LoRA chain rule, we treat A,B
        as "gradient buffers" and minimize:

            L_rec(A,B) = 1/2 ||AB + G||_F^2 + (lambda_a/2)||A||_F^2 + (lambda_b/2)||B||_F^2

        where G = D^T @ X is the ideal gradient for C. This makes AB ~ -G,
        so that `C += transfer_lr * AB` implements SGD descent: C -= transfer_lr * G.

        Gradients:
            dL_rec/dA = A(BB^T) + GB^T + lambda_a*A
            dL_rec/dB = (A^TA)B + A^TG + lambda_b*B

        Hardware mapping:
            - Hebbian terms (GB^T, A^TG):
                XB = tile_b.forward(X)
                tile_a.update(XB, D)  -> A -= lr * D^T @ XB = A -= lr * GB^T
                DA = tile_a.backward(D)
                tile_b.update(X, DA)  -> B -= lr * DA^T @ X = B -= lr * A^TG

            - Stabilizer terms: Use scalar approximation or exact Gram
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Effective learning rate (no lora_alpha in reconstruction mode)
        lr_rec = lr * self.recon_lr_scale

        # 2) Projections for Hebbian terms
        with torch.no_grad():
            XB = self.tile_b.forward(x)  # [batch, rank]
            DA = self.tile_a.backward(d)  # [batch, rank]

        # 3) Hebbian updates: A -= lr * GB^T, B -= lr * A^TG
        # A update: A -= lr_rec * D^T @ XB = A -= lr_rec * GB^T
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_rec)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1

        # B update: B -= lr_rec * DA^T @ X = B -= lr_rec * A^TG
        lr_b_old = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_rec)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x, DA)
        else:
            self.tile_b.update(x, DA)
        self.tile_b.set_learning_rate(lr_b_old)
        self.num_b_updates += 1

        # 4) Stabilizer terms (prevent ||A@B|| growth)
        self._apply_reconstruction_stabilizer(lr_rec)

        # 5) Optional norm clipping (recon_use_clip_norm)
        if self.recon_use_clip_norm:
            self._clip_ab_norms(max_norm=self.recon_clip_norm)

        # 6) Dynamic TE update (before counter check)
        self._update_dynamic_te(lr)

        # 7) Counter (always += m_batch, matching TikiTaka convention)
        m_batch = x.shape[0]
        self._last_m_batch = m_batch
        self.transfer_counter += m_batch

    def _apply_reconstruction_stabilizer(self, lr_rec: float) -> None:
        """Apply stabilizer terms for reconstruction update.

        Uses forward/backward to read weights and update() to apply decay,
        avoiding expensive get_weights()/set_weights() calls.
        """
        with torch.no_grad():
            device = self.device

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank = torch.eye(self.rank, device=device, dtype=self.dtype)
            A_T = self.tile_a.forward(I_rank)  # [rank, d_size] = A.T

            # Read B using backward: tile_b.backward(I) = I @ B = B
            B_read = self.tile_b.backward(I_rank)  # [rank, x_size] = B

            # Compute norms for scalar stabilizer
            A_norm_sq = torch.sum(A_T ** 2).item()
            B_norm_sq = torch.sum(B_read ** 2).item()
            sA = A_norm_sq / self.rank  # tr(A^TA)/rank
            sB = B_norm_sq / self.rank  # tr(BB^T)/rank

            # Update EMA estimates
            if not self._ema_initialized:
                self._ema_sA = sA
                self._ema_sB = sB
                self._ema_initialized = True
            else:
                beta = self.recon_ema_beta
                self._ema_sA = beta * self._ema_sA + (1 - beta) * sA
                self._ema_sB = beta * self._ema_sB + (1 - beta) * sB

            # Determine whether to use exact Gram this step
            self._recon_step_counter += 1
            use_exact_this_step = self.recon_use_exact_gram or (
                self.recon_exact_gram_every > 0 and
                self._recon_step_counter % self.recon_exact_gram_every == 0
            )

            if use_exact_this_step:
                # Exact Gram matrix stabilizer
                A_rank = A_T.t()  # [d_size, rank]
                B_rank = B_read   # [rank, x_size]

                BBT = B_rank @ B_rank.t()  # [rank, rank]
                ATA = A_rank.t() @ A_rank  # [rank, rank]

                # Combined stabilizer + L2: d_A = A @ BBT + lambda_a * A = A @ (BBT + lambda_a*I)
                BBT_reg = BBT + self.recon_lambda_a * I_rank
                ATA_reg = ATA + self.recon_lambda_b * I_rank

                # Apply A -= lr_rec * A @ BBT_reg using tile.update
                d_A = (A_rank @ BBT_reg).t()  # [rank, d_size]
                lr_old_a = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(lr_rec)
                self.tile_a.update(I_rank, d_A.t())
                self.tile_a.set_learning_rate(lr_old_a)

                # B -= lr_rec * ATA_reg @ B
                lr_old_b = self.tile_b.get_learning_rate()
                self.tile_b.set_learning_rate(lr_rec)
                self.tile_b.update(B_rank, ATA_reg.t())
                self.tile_b.set_learning_rate(lr_old_b)

            elif self.recon_use_scalar_stabilizer:
                # Scalar approximation: BB^T ~ sB*I, A^TA ~ sA*I
                # Exponential decay (always positive, no sign flip risk)
                exp_arg_A = lr_rec * (self._ema_sB + self.recon_lambda_a)
                exp_arg_B = lr_rec * (self._ema_sA + self.recon_lambda_b)

                # Clamp exponent to prevent extreme decay
                exp_arg_A = min(exp_arg_A, 3.0)
                exp_arg_B = min(exp_arg_B, 3.0)

                decay_A = math.exp(-exp_arg_A)
                decay_B = math.exp(-exp_arg_B)

                # Apply A *= decay using tile.update: A -= (1-decay)*A
                c_A = 1.0 - decay_A
                if c_A > 1e-8:
                    lr_old_a = self.tile_a.get_learning_rate()
                    self.tile_a.set_learning_rate(c_A)
                    self.tile_a.update(I_rank, A_T.t())
                    self.tile_a.set_learning_rate(lr_old_a)

                c_B = 1.0 - decay_B
                if c_B > 1e-8:
                    lr_old_b = self.tile_b.get_learning_rate()
                    self.tile_b.set_learning_rate(c_B)
                    self.tile_b.update(B_read, I_rank)
                    self.tile_b.set_learning_rate(lr_old_b)

    def _clip_ab_norms(self, max_norm: float = 10.0) -> None:
        """Scale A and B down to the Frobenius norm max_norm where they exceed it.

        Uses forward/backward to read weights and update() to apply scaling.
        """
        with torch.no_grad():
            device = self.device

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank_a = torch.eye(self.rank, device=device, dtype=self.dtype)
            A_T = self.tile_a.forward(I_rank_a)  # [rank, d_size] = A.T
            A_norm = torch.norm(A_T).item()

            if A_norm > max_norm:
                scale = max_norm / A_norm
                c = 1.0 - scale
                lr_old = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(c)
                self.tile_a.update(I_rank_a, A_T.t())
                self.tile_a.set_learning_rate(lr_old)

            # Read B using backward: tile_b.backward(I) = I @ B = B
            I_rank_b = torch.eye(self.rank, device=device, dtype=self.dtype)
            B_read = self.tile_b.backward(I_rank_b)
            B_norm = torch.norm(B_read).item()

            if B_norm > max_norm:
                scale = max_norm / B_norm
                c = 1.0 - scale
                lr_old = self.tile_b.get_learning_rate()
                self.tile_b.set_learning_rate(c)
                self.tile_b.update(B_read, I_rank_b)
                self.tile_b.set_learning_rate(lr_old)

    def _compute_effective_transfer_lr(self) -> float:
        """Compute effective transfer LR with optional SGD scaling and gradient correction.

        Returns:
            Effective transfer learning rate for this transfer step.
        """
        lr_tr = self.transfer_lr
        if self.scale_transfer_lr:
            lr_tr *= self._last_lr_sgd
        if self.correct_gradient_magnitudes:
            if self.transfer_fast_lr_ref == "A":
                ref = self.last_lr_eff_a
            elif self.transfer_fast_lr_ref == "B":
                ref = self.last_lr_eff_b
            else:  # "geomean"
                ref = math.sqrt(self.last_lr_eff_a * self.last_lr_eff_b)
            if ref > 0.0:
                lr_tr /= ref
        return lr_tr

    def ab_weight_transfer(self, method: Optional[str] = None) -> None:
        """A@B -> C transfer, then reinit of the transferred ranks.

        Transfer: C += lr_tr * (A @ B), with lr_tr from _compute_effective_transfer_lr()

        Supports per-rank round-robin scheduling via transfer_rank_schedule:
        - "all": Transfer all ranks at once (default)
        - "round_robin": Cycle through ranks, transferring transfer_ranks_per_step
          ranks each time. After transfer, only the transferred ranks are reinit'd.

        Args:
            method: Transfer method override.
                   "onehot" - One-hot reading (analog-realistic pulsed update)
                   "direct" - Direct weight access (pulsed update)
                   "set" - Exact floating-point update of C (no pulses)
                   If None, use self.transfer_method setting.
        """
        # Compute effective transfer LR once for all sub-methods
        self._eff_transfer_lr = self._compute_effective_transfer_lr()

        # Use instance setting if not specified
        if method is None:
            method = self.transfer_method

        # Compute rank indices to transfer
        if self.transfer_rank_schedule == "round_robin":
            n = self.transfer_ranks_per_step
            rank_indices = [(self._rr_rank_idx + i) % self.rank for i in range(n)]
        else:
            rank_indices = list(range(self.rank))

        # Dispatch to transfer method
        if method == "onehot":
            self._ab_weight_transfer_onehot(rank_indices)
        elif method == "direct":
            self._ab_weight_transfer_direct(rank_indices)
        elif method == "set":
            self._ab_weight_transfer_set(rank_indices)
        else:
            raise ValueError(f"Unknown transfer method: {method}. Use 'onehot', 'direct', or 'set'.")

        # Reinit only the transferred ranks
        self._reinit_ranks(rank_indices)

        # Advance round-robin pointer
        if self.transfer_rank_schedule == "round_robin":
            self._rr_rank_idx = (self._rr_rank_idx + self.transfer_ranks_per_step) % self.rank

    def _ab_weight_transfer_direct(self, rank_indices: List[int]) -> None:
        """Transfer with direct (digital) access to the A and B weights.

        A and B are taken from get_weights(); C is written by pulsed updates
        in chunks of rank_chunk ranks.

        Args:
            rank_indices: List of rank indices to transfer.
        """
        with torch.no_grad():
            # Get weights (they come in the tile's native device)
            A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
            B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]

            A_lr = A_weights[:, rank_indices]  # [d_size, len(rank_indices)]
            B_lr = B_weights[rank_indices, :]  # [len(rank_indices), x_size]

            # Record intended transfer delta for diagnostics
            if self.enable_diagnostics:
                self.last_transfer_delta = (self._eff_transfer_lr * (A_lr @ B_lr)).detach()

            # Transfer in chunks to manage memory
            lr_eff = abs(self._eff_transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)

            n_sel = len(rank_indices)
            chunk_size = min(self.rank_chunk, n_sel)
            for off in range(0, n_sel, chunk_size):
                end = min(off + chunk_size, n_sel)

                # Pack chunks (keep on same device as tiles)
                D_chunk = A_lr[:, off:end].contiguous()  # [d_size, cur]
                X_chunk = B_lr[off:end, :].contiguous()  # [cur, x_size]

                # Sign rule: PWU computes W += -lr * D @ X^T; the transfer needs W += +transfer_lr * D @ X^T
                if self._eff_transfer_lr > 0:
                    D_chunk = -D_chunk

                # Controller device
                dev = self.device
                X_chunk_d = X_chunk.contiguous().to(dev, non_blocking=True)
                D_chunk_t_d = D_chunk.t().contiguous().to(dev, non_blocking=True)

                # Pulsed update to C tile
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(X_chunk_d, D_chunk_t_d)
                else:
                    self.tile_c.update(X_chunk_d, D_chunk_t_d)

                del X_chunk_d, D_chunk_t_d

            self.tile_c.set_learning_rate(old_lr)
        self.num_transfers += 1

    def _ab_weight_transfer_set(self, rank_indices: List[int]) -> None:
        """Deterministic FP transfer via tile.update() with NoneWithDevice.

        C tile must be constructed with PulseType.NONE_WITH_DEVICE (set in
        lrtt_tile.py create_update_params). This gives pure FP update with
        device weight clipping, avoiding get_weights/set_weights CPU roundtrips.

        Args:
            rank_indices: List of rank indices to transfer.
        """
        with torch.no_grad():
            # Read raw A, B weights and scale factors
            A_raw = self.tile_a.tile.get_weights().to(self.device)
            B_raw = self.tile_b.tile.get_weights().to(self.device)
            alpha_a = self.tile_a.get_scales()
            alpha_b = self.tile_b.get_scales()
            alpha_c = self.tile_c.get_scales()

            A_lr = A_raw[:, rank_indices]
            if alpha_a is not None:
                A_lr = A_lr * alpha_a.view(-1, 1)
            B_lr = B_raw[rank_indices, :]
            if alpha_b is not None:
                B_lr = B_lr * alpha_b.view(-1, 1)

            if self.enable_diagnostics:
                self.last_transfer_delta = (self._eff_transfer_lr * (A_lr @ B_lr)).detach()

            # Undo alpha_c so update writes correct raw delta
            if alpha_c is not None:
                A_adj = A_lr / alpha_c.view(-1, 1)
            else:
                A_adj = A_lr

            # PWU: W += -lr * d^T @ x; the transfer needs W += +lr * A_adj @ B_lr
            # So pass d = -A_adj^T, x = B_lr
            old_lr = self.tile_c.tile.get_learning_rate()
            self.tile_c.tile.set_learning_rate(abs(self._eff_transfer_lr))
            self.tile_c.tile.update(
                B_lr.contiguous(),
                (-A_adj).t().contiguous(),
                False, False, False, False,
            )
            self.tile_c.tile.set_learning_rate(old_lr)

        self.num_transfers += 1

    def _read_ab_onehot_symmetric(self, rank_indices: Optional[List[int]] = None) -> tuple:
        """One-hot read of the selected columns of A and rows of B.

        A column k of A is read with tile_a.forward(e_k), a row k of B with
        tile_b.backward(e_k), using the IO model of the tiles. Modes:
        1. Basic (default): one read per rank, +e only or +e/-e differential
           (differential_read); with read count N > 1, N differential reads
           are averaged
        2. AGC: Automatic gain control of the read amplitude (agc_enabled=True)
        3. Two-Amplitude: offset removal with two amplitudes (two_amp_enabled=True)

        The read count is read_n_avg if it is > 1, otherwise num_reads.

        Returns:
            (A_cols: [d_size, rank], B_rows: [rank, x_size])
        """
        # Ensure one-hot cache exists on correct device
        if self._transfer_vec_a is None or self._transfer_vec_a.device != self.device:
            self._transfer_vec_a = torch.eye(
                self.rank, dtype=self.dtype, device=self.device
            )

        I = self._transfer_vec_a

        # Default: all ranks
        if rank_indices is None:
            rank_indices = list(range(self.rank))

        # Read count: read_n_avg if > 1, otherwise num_reads
        read_count = self.read_n_avg if self.read_n_avg > 1 else self.num_reads

        # Fast path: batch processing (no AGC, no two_amp, single read)
        if read_count == 1 and not self.agc_enabled and not self.two_amp_enabled:
            I_sel = I[rank_indices]  # [len(rank_indices), rank]
            if self.differential_read:
                A_p = self.tile_a.forward(I_sel)    # [n_sel, d_size]
                A_m = self.tile_a.forward(-I_sel)   # [n_sel, d_size]
                B_p = self.tile_b.backward(I_sel)   # [n_sel, x_size]
                B_m = self.tile_b.backward(-I_sel)  # [n_sel, x_size]
                A_cols = (0.5 * (A_p - A_m)).T  # [d_size, n_sel]
                B_rows = 0.5 * (B_p - B_m)      # [n_sel, x_size]
            else:
                A_cols = self.tile_a.forward(I_sel).T   # [d_size, n_sel]
                B_rows = self.tile_b.backward(I_sel)    # [n_sel, x_size]
            return A_cols, B_rows

        # Slow path: per-rank processing (for AGC, two_amp, or multi-read)
        A_cols = []
        B_rows = []

        for k in rank_indices:
            e = I[k].unsqueeze(0)  # [1, rank], +one-hot

            if self.two_amp_enabled:
                # Two-amplitude read to cancel odd offset
                a_k, _, _ = self._two_amp_read(
                    self.tile_a, e, mode="fwd",
                    read_n_avg=read_count, margin=self.agc_margin
                )
                b_k, _, _ = self._two_amp_read(
                    self.tile_b, e, mode="bwd",
                    read_n_avg=read_count, margin=self.agc_margin
                )
            elif self.agc_enabled:
                # AGC mode: find optimal amplitude, then differential read
                amp_a = self._pick_amp_agc(self.tile_a, e, mode="fwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)
                amp_b = self._pick_amp_agc(self.tile_b, e, mode="bwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)

                a_k = self._diff_read(self.tile_a, e, amp=amp_a, mode="fwd", read_n_avg=read_count).squeeze(0)
                b_k = self._diff_read(self.tile_b, e, amp=amp_b, mode="bwd", read_n_avg=read_count).squeeze(0)
            elif read_count == 1:
                if self.differential_read:
                    # Differential read: cancels DC offset (2x tile ops)
                    a_p = self.tile_a.forward(e).squeeze(0)   # [d_size]
                    a_m = self.tile_a.forward(-e).squeeze(0)  # [d_size]
                    b_p = self.tile_b.backward(e).squeeze(0)  # [x_size]
                    b_m = self.tile_b.backward(-e).squeeze(0) # [x_size]
                    a_k = 0.5 * (a_p - a_m)
                    b_k = 0.5 * (b_p - b_m)
                else:
                    # Simple read: faster but includes DC offset (1x tile ops)
                    a_k = self.tile_a.forward(e).squeeze(0)   # [d_size]
                    b_k = self.tile_b.backward(e).squeeze(0)  # [x_size]
            else:
                # Multi-read averaging with amplitude=1.0
                a_k = self._diff_read(self.tile_a, e, amp=1.0, mode="fwd", read_n_avg=read_count).squeeze(0)
                b_k = self._diff_read(self.tile_b, e, amp=1.0, mode="bwd", read_n_avg=read_count).squeeze(0)

            A_cols.append(a_k)
            B_rows.append(b_k)

        A_cols = torch.stack(A_cols, dim=1)  # [d_size, rank]
        B_rows = torch.stack(B_rows, dim=0)  # [rank, x_size]
        return A_cols, B_rows

    def _center_and_normalize(self, A_cols: Tensor, B_rows: Tensor, eps: float = 1e-8) -> tuple:
        """Optional row/column mean removal and per-rank L2 normalization.

        Args:
            A_cols: [d_size, rank]
            B_rows: [rank, x_size]
            eps: Numerical stability epsilon

        Returns:
            (A_cols_processed, B_rows_processed)
        """
        if self.transfer_centering:
            A_cols = A_cols - A_cols.mean(dim=0, keepdim=True)
            B_rows = B_rows - B_rows.mean(dim=1, keepdim=True)

        if self.transfer_normalize:
            for k in range(A_cols.shape[1]):
                ak = A_cols[:, k]
                bk = B_rows[k, :]
                na = ak.norm()
                nb = bk.norm()
                if na > eps:
                    A_cols[:, k] = ak / na
                if nb > eps:
                    B_rows[k, :] = bk / nb

        return A_cols, B_rows

    def _ze_norm2_via_gram(self, A_cols: Tensor, B_rows: Tensor) -> float:
        """||sum_k outer(a_k, b_k)||_F^2 = sum_{i,j} (a_i^T a_j)*(b_i^T b_j).

        Computes Frobenius norm squared of the outer product sum efficiently
        using Gram matrices without materializing the full [d_size, x_size] matrix.
        """
        G_A = A_cols.t() @ A_cols      # [rank, rank]
        G_B = B_rows @ B_rows.t()      # [rank, rank]
        return (G_A * G_B).sum().item()

    def _ab_weight_transfer_onehot(self, rank_indices: List[int]) -> None:
        """One-hot based transfer; dispatches to the off, pilot or sigma_delta variant.

        Transfer modes:
        - "off": rank-wise pulsed transfer without calibration (default)
        - "pilot": micro-stepped transfer with optional pilot gain calibration
        - "sigma_delta": Sigma-delta quantization with residual accumulation

        Args:
            rank_indices: List of rank indices to transfer.
        """
        # Select the mode (transfer_mode first, transfer_gamma_mode as fallback)
        mode = getattr(self, 'transfer_mode', None) or self.transfer_gamma_mode

        if mode == "sigma_delta":
            self._ab_weight_transfer_onehot_sigma_delta(rank_indices)
        elif mode == "pilot":
            self._ab_weight_transfer_onehot_pilot(rank_indices)
        else:  # "off" (default)
            self._ab_weight_transfer_onehot_off(rank_indices)

    def _ab_weight_transfer_onehot_pilot(self, rank_indices: List[int]) -> None:
        """One-hot transfer with micro-steps and optional pilot calibration (transfer_mode='pilot').

        1. One-hot read of A and B (_read_ab_onehot_symmetric)
        2. If transfer_gamma_mode == 'pilot': a pilot transfer with learning rate
           transfer_pilot_frac * |lr_tr| measures the gain gamma of the transfer
           into C (clamped to [0.05, 20]); otherwise gamma = 1
        3. The remaining learning rate gamma * |lr_tr| (minus the pilot rate) is
           written in max(2, transfer_micro_steps) micro-steps, one fewer after a pilot
        """
        with torch.no_grad():
            # 0) Setup: one-hot cache, back up the learning rate of C
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(
                    self.rank, dtype=self.dtype, device=self.device
                )

            old_lr_c = self.tile_c.get_learning_rate()

            try:
                # 1) One-hot read of A and B
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)

                # Optional centering/normalization
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # Record intended transfer delta for diagnostics (digital get_weights for fair comparison)
                if self.enable_diagnostics:
                    A_dg = self.tile_a.get_weights()[0]
                    B_dg = self.tile_b.get_weights()[0]
                    self.last_transfer_delta = (self._eff_transfer_lr * (A_dg[:, rank_indices] @ B_dg[rank_indices, :])).detach()

                # 2) Pilot-based gamma estimation (scalar calibration)
                Z_norm2 = self._ze_norm2_via_gram(A_cols, B_rows) + 1e-12
                pilot_frac = max(1e-4, float(self.transfer_pilot_frac))
                M_total = max(2, int(self.transfer_micro_steps))
                lr_abs = abs(self._eff_transfer_lr)

                # Pilot lr
                lr_pilot = lr_abs * pilot_frac
                gamma = 1.0  # Default gamma

                if self.transfer_gamma_mode == "pilot":
                    # Measure the actual scale with a pilot transfer
                    num_acc = 0.0

                    def _forward_c(x_row):
                        """Forward C tile with input row vector."""
                        y = self.tile_c.forward(x_row.unsqueeze(0))  # [1, d_size]
                        return y.squeeze(0)

                    for ki in range(A_cols.shape[1]):
                        a_k = A_cols[:, ki]
                        b_k = B_rows[ki, :]

                        # Response before the transfer
                        y0 = _forward_c(b_k)

                        # Sign rule: PWU computes W += -lr * D @ X^T
                        D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                        X_k = b_k

                        # One pilot transfer
                        self.tile_c.set_learning_rate(lr_pilot)
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                        else:
                            self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

                        # Measure the change after the transfer
                        y1 = _forward_c(b_k)
                        dy = y1 - y0
                        num_acc += (dy * a_k).sum().item()

                    # gamma = measured change / expected change
                    gamma = num_acc / (lr_pilot * Z_norm2)
                    # Clamp gamma to [0.05, 20]
                    gamma = float(torch.clamp(
                        torch.tensor(gamma, device=self.device), 0.05, 20.0
                    ).item())

                # 3) Main transfer: split the remaining lr over M_rest micro-transfers
                lr_remain = lr_abs * gamma - (lr_pilot if self.transfer_gamma_mode == "pilot" else 0.0)
                lr_remain = max(0.0, lr_remain)
                M_rest = max(1, M_total - 1) if self.transfer_gamma_mode == "pilot" else M_total

                # Transfer scheme depends on multi_read_mode
                num_reads = self.num_reads
                if self.multi_read_mode == "per_read" and num_reads > 1:
                    # per_read mode: transfer after each read (lr divided by num_reads)
                    lr_step = lr_remain / (M_rest * num_reads) if M_rest > 0 else 0.0
                    self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                    I = self._transfer_vec_a
                    for k in rank_indices:
                        e = I[k].unsqueeze(0)

                        for _ in range(num_reads):
                            # Fresh read each time
                            a_p = self.tile_a.forward(e).squeeze(0)
                            a_m = self.tile_a.forward(-e).squeeze(0)
                            b_p = self.tile_b.backward(e).squeeze(0)
                            b_m = self.tile_b.backward(-e).squeeze(0)
                            a_k = 0.5 * (a_p - a_m)
                            b_k = 0.5 * (b_p - b_m)

                            D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                            X_k = b_k

                            # M_rest micro-transfers
                            for _ in range(M_rest):
                                if hasattr(self.tile_c, '_orig_update'):
                                    self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                                else:
                                    self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                else:
                    # average mode (default): use the already averaged A_cols, B_rows
                    lr_step = lr_remain / M_rest if M_rest > 0 else 0.0
                    self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                    for ki in range(A_cols.shape[1]):
                        a_k = A_cols[:, ki]
                        b_k = B_rows[ki, :]
                        D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                        X_k = b_k

                        # M_rest micro-transfers
                        for _ in range(M_rest):
                            if hasattr(self.tile_c, '_orig_update'):
                                self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                            else:
                                self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

                # Print gamma at the first transfer (not when LRTT_SILENT is set)
                import os
                if self.num_transfers < 1 and not os.environ.get("LRTT_SILENT"):
                    pilot_info = f"lr_pilot={lr_pilot:.3e} " if self.transfer_gamma_mode == "pilot" else ""
                    print(f"[LRTT onehot] gamma={gamma:.3f} {pilot_info}"
                          f"lr_remain={lr_remain:.3e} Znorm2={Z_norm2:.3e}")

            finally:
                # 5) Restore the learning rate of C
                self.tile_c.set_learning_rate(old_lr_c)

        self.num_transfers += 1

    def _ab_weight_transfer_onehot_off(self, rank_indices: List[int]) -> None:
        """One-hot transfer without calibration (transfer_mode='off', default).

        Each selected rank k is read as a column a_k of A (tile_a forward) and a
        row b_k of B (tile_b backward), and C receives transfer_micro_steps
        pulsed updates a_k b_k^T, each with learning rate |lr_tr| / transfer_micro_steps.
        """
        with torch.no_grad():
            # 1) Setup: one-hot cache, back up the learning rate of C
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(
                    self.rank, dtype=self.dtype, device=self.device
                )

            old_lr_c = self.tile_c.get_learning_rate()

            try:
                # 2) One-hot read of A and B (through the I/O model of the tiles)
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # 3) Learning rate of each micro-step: |lr_tr| / M
                lr_abs = abs(self._eff_transfer_lr)
                M_total = max(1, int(self.transfer_micro_steps))
                lr_step = lr_abs / M_total
                self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                # Record intended transfer delta for diagnostics (digital get_weights for fair comparison)
                if self.enable_diagnostics:
                    A_dg = self.tile_a.get_weights()[0]
                    B_dg = self.tile_b.get_weights()[0]
                    self.last_transfer_delta = (self._eff_transfer_lr * (A_dg[:, rank_indices] @ B_dg[rank_indices, :])).detach()

                # 4) micro-transfer
                for ki in range(A_cols.shape[1]):
                    a_k = A_cols[:, ki]
                    b_k = B_rows[ki, :]
                    D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                    X_k = b_k

                    for _ in range(M_total):
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                        else:
                            self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

            finally:
                # 5) Restore the learning rate of C
                self.tile_c.set_learning_rate(old_lr_c)

        self.num_transfers += 1

    def _ab_weight_transfer_onehot_sigma_delta(self, rank_indices: List[int]) -> None:
        """One-hot based transfer, sigma-delta mode: per-rank integrator h_k with a fixed quantum g.

        Sigma-Delta quantization:
        - Target scalar per rank: delta_k := |lr_tr| (the same for every rank)
        - First-order sigma-delta: h_k <- h_k + delta_k; n_k <- round(h_k / g); h_k <- h_k - n_k*g
        - Sign rule: tile.update computes W += -lr * D @ X^T, so D=-a_k when lr_tr>0
        - Quantum g: sd_quantum if set, otherwise g := |lr_tr| / transfer_micro_steps
        """
        with torch.no_grad():
            # Setup: one-hot cache, back up the learning rate of C
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(self.rank, dtype=self.dtype, device=self.device)

            old_lr_c = self.tile_c.get_learning_rate()

            try:
                # 1) One-hot read: A_cols[d, r], B_rows[r, x]
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)

                # Optional centering/normalization
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # Record intended transfer delta for diagnostics (digital get_weights for fair comparison)
                if self.enable_diagnostics:
                    A_dg = self.tile_a.get_weights()[0]
                    B_dg = self.tile_b.get_weights()[0]
                    self.last_transfer_delta = (self._eff_transfer_lr * (A_dg[:, rank_indices] @ B_dg[rank_indices, :])).detach()

                # 2) Sigma-delta state and parameters
                self._ensure_sd_state()
                lr_abs = float(abs(self._eff_transfer_lr))
                g = float(self.sd_quantum) if (self.sd_quantum is not None and self.sd_quantum > 0.0) \
                    else max(lr_abs / float(max(1, int(self.transfer_micro_steps))), 1e-12)

                # Target scalar per rank delta_k := |_eff_transfer_lr| (selected ranks only)
                # 3) Sigma-delta integration/quantization: accumulate h_k -> integer pulses n_k, update the residual
                ri = rank_indices  # selected ranks for this transfer step
                self.sd_acc[ri] = self.sd_acc[ri] + lr_abs  # h_k += delta_k
                n_sel_float = self.sd_acc[ri] / g
                n_sel = torch.round(n_sel_float).to(torch.int64)
                self.sd_acc[ri] = self.sd_acc[ri] - n_sel.to(self.dtype) * g

                # 4) Transfer n_k integer pulses into C
                # Sign rule: D=-a_k when _eff_transfer_lr>0
                sign = -1.0 if (self._eff_transfer_lr > 0) else 1.0

                # Unit pulse lr = g
                self.tile_c.set_learning_rate(g)

                nonzero = int((n_sel != 0).sum().item())
                max_rep = int(n_sel.abs().max().item()) if nonzero > 0 else 0

                for col_idx in range(len(ri)):
                    reps = int(n_sel[col_idx].item())
                    if reps == 0:
                        continue

                    a_k = (sign * A_cols[:, col_idx]).unsqueeze(0)  # [1, d]
                    b_k = B_rows[col_idx, :].unsqueeze(0)          # [1, x]

                    # Both signs of reps are supported: for reps<0 the sign is absorbed into D
                    if reps < 0:
                        a_k = -a_k
                        reps = -reps

                    # reps unit updates
                    for _ in range(reps):
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(b_k, a_k)
                        else:
                            self.tile_c.update(b_k, a_k)

                # Print the quantizer state at the first transfer
                if self.num_transfers < 1:  # first transfer only
                    res_max = float(self.sd_acc.abs().max().item())
                    print(f"[sigma_delta transfer] g={g:.3e}, nonzero_ranks={nonzero}, max_reps={max_rep}, "
                          f"residual_max<=g/2? {res_max <= 0.5*g + 1e-12} (res_max={res_max:.3e})")

            finally:
                # Restore the learning rate of C
                self.tile_c.set_learning_rate(old_lr_c)

        self.num_transfers += 1

    def forward_inject(
        self,
        x: Tensor,                    # [x_size, m] or [batch, x_size]
        out_trans: bool = False,
        in_trans: bool = False
    ) -> Tensor:
        """Forward inject: y = C@x + alpha * A@(B@x), alpha = effective_alpha.

        - If forward_inject_enabled=False or rank=0: C only (y = C@x)
        - Otherwise analog forward passes of the three tiles
          (_forward_inject_analog_unified): g = B@x, y_ab = A@g, y_c = C@x,
          y = y_c + alpha*y_ab, also for transposed inputs and outputs

        Args:
            x: Input tensor [x_size, m] or [batch, x_size]
            out_trans: Output transposed flag
            in_trans: Input transposed flag

        Returns:
            Output tensor [d_size, m] or [batch, d_size]
        """
        # Initialize tiles on first forward if needed
        if not self._tiles_initialized:
            self.reinit()

        # Handle disabled forward injection
        if not self.forward_inject_enabled or self.rank == 0:
            return self.tile_c.forward(x, in_trans=in_trans, out_trans=out_trans)

        # Analog forward passes of the three tiles (also for transposed inputs)
        return self._forward_inject_analog_unified(x, in_trans=in_trans, out_trans=out_trans)

    def _forward_inject_digital_fallback(
        self,
        x: Tensor,
        out_trans: bool,
        in_trans: bool
    ) -> Tensor:
        """Forward pass through C with W_eff = C + alpha * A @ B written into C.

        W_eff is composed digitally, written into tile_c for one forward pass,
        and the C weights are restored afterwards. forward_inject() uses
        _forward_inject_analog_unified() instead.
        """
        # get_weights() materializes the full weight tensors
        C_weights = self.tile_c.get_weights()[0]   # [d_size, x_size]
        A_lr = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]        # [rank, x_size]

        # Large intermediate tensor W_eff
        W_eff = C_weights + self.effective_alpha * (A_lr @ B_lr)

        # Set temporary weights and forward
        original_weights = C_weights.clone()
        self.tile_c.set_weights(W_eff)

        try:
            result = self.tile_c.forward(x, bias=False, in_trans=in_trans, out_trans=out_trans)
        finally:
            # Restore original weights
            self.tile_c.set_weights(original_weights)

        return result

    def _forward_inject_analog_hybrid(self, x: Tensor) -> Tensor:
        """Digital forward pass from the stored weights (deterministic).

        Computes y = x @ (C^T + alpha * B^T @ A^T) without tile reads.
        forward_inject() uses _forward_inject_analog_unified() instead.
        """
        # Get component weights directly
        C_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0][:, :self.rank]  # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0][:self.rank, :]  # [rank, x_size]

        # Compute effective weight matrix: W_eff = C^T + alpha * B^T @ A^T
        W_eff = C_weights.t() + self.effective_alpha * (B_weights.t() @ A_weights.t())

        # Ensure same device as input
        W_eff = W_eff.to(x.device)

        # Forward pass: y = x @ W_eff
        result = x @ W_eff  # [batch, x_size] @ [x_size, d_size] = [batch, d_size]

        return result

    def _forward_inject_analog_unified(
        self,
        x: Tensor,
        in_trans: bool,
        out_trans: bool
    ) -> Tensor:
        """Analog forward pass with forward injection.

        Uses the forward passes of the tiles in the order B, A, C, so that the
        IO model (noise, quantization, bounds) of each tile applies and the
        analog optimizer receives the inputs and errors of all three tiles.
        """
        # 1) Normalize input to batch-first
        x_bf = x.t() if in_trans else x  # [batch, x_size]

        # 2) Tile forward passes in the order B -> A -> C
        g = self.tile_b.forward(x_bf)      # [batch, rank]
        y_ab = self.tile_a.forward(g)      # [batch, d_size]
        y_c = self.tile_c.forward(x_bf)    # [batch, d_size]

        # 3) Composition
        y = y_c + self.effective_alpha * y_ab   # [batch, d_size]

        # 4) Output transpose
        return y.t() if out_trans else y

    def _update_dynamic_te(self, lr_current: float) -> None:
        """Update transfer_every based on warmup schedule and LR decay ratio.

        Phase 1 (Warmup ramp): TE steps through te_warmup_schedule in equal intervals.
        Phase 2 (Peak LR): TE fixed at te_base.
        Phase 3 (LR decay): TE increases proportionally to LR ratio.

        Formula (Phase 3): TE(t) = clip(te_min, te_max, round(TE_0 * (lr_peak / lr_current)^p))
        """
        self.te_step_counter += 1

        # Phase 1: Warmup ramp (step-wise schedule)
        if (self.te_warmup_schedule and self.te_warmup_steps > 0
                and self.te_step_counter <= self.te_warmup_steps):
            n_phases = len(self.te_warmup_schedule)
            phase_len = self.te_warmup_steps / n_phases
            phase_idx = min(int((self.te_step_counter - 1) / phase_len), n_phases - 1)
            self.transfer_every = self.te_warmup_schedule[phase_idx]
            # Still track lr_peak during warmup
            if lr_current > 0 and (self.lr_peak is None or lr_current > self.lr_peak):
                self.lr_peak = lr_current
            return

        if not self.dynamic_te:
            self.transfer_every = self.te_base
            return

        if lr_current <= 0:
            return

        # Phase 2: Peak tracking (lr still increasing after warmup)
        if self.lr_peak is None or lr_current > self.lr_peak:
            self.lr_peak = lr_current
            self.transfer_every = self.te_base
            return

        # Phase 3: Decay - increase TE proportionally
        ratio = self.lr_peak / lr_current  # >= 1.0
        te_proposed = round(self.te_base * (ratio ** self.dynamic_te_power))
        te_proposed = max(self.dynamic_te_min, min(self.dynamic_te_max, te_proposed))

        # Monotonic non-decreasing
        self.transfer_every = max(self.transfer_every, te_proposed)

    def should_transfer(self) -> bool:
        """Check if transfer should occur based on counter and schedule.

        Uses the same modulo-based boundary crossing check as aihwkit's
        Tiki-Taka transfer (see transfer.py and rpu_transfer_device.cpp):
        - The counter increases by m_batch per update; reinit() after a
          transfer sets it back to zero.
        - effective_te = transfer_every * m_batch when units_in_mbatch=True.
        - Transfer fires when the current batch crosses an effective_te boundary.
        """
        effective_te = self.transfer_every
        if self.units_in_mbatch:
            effective_te = int(math.ceil(self.transfer_every * self._last_m_batch))
        if effective_te <= 0:
            return False
        rest_count = ((self.transfer_counter - self._last_m_batch) % effective_te) + self._last_m_batch
        return rest_count >= effective_te

    def reset_transfer_counter(self) -> None:
        """Set the transfer counter to zero."""
        self.transfer_counter = 0

    def get_state_dict(self) -> Dict[str, Any]:
        """Get controller state for serialization."""
        return {
            'transfer_counter': self.transfer_counter,
            'num_a_updates': self.num_a_updates,
            'num_b_updates': self.num_b_updates,
            'num_transfers': self.num_transfers,
            'd_size': self.d_size,
            'x_size': self.x_size,
            'rank': self.rank,
            'transfer_lr': self.transfer_lr,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'reinit_mode': self.reinit_mode,
            'forward_inject_enabled': self.forward_inject_enabled,
            'lr_peak': self.lr_peak,
            'te_step_counter': self.te_step_counter,
            'transfer_rank_schedule': self.transfer_rank_schedule,
            'transfer_ranks_per_step': self.transfer_ranks_per_step,
            '_rr_rank_idx': self._rr_rank_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load controller state from serialization."""
        # The key 'forward_inject' is accepted for forward_inject_enabled
        if 'forward_inject' in state_dict and 'forward_inject_enabled' not in state_dict:
            state_dict['forward_inject_enabled'] = state_dict.pop('forward_inject')

        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_device(self, device: torch.device) -> None:
        """Set device and clear buffers for reallocation.

        Args:
            device: Target device (CPU or CUDA)
        """
        self.device = torch.device(device)
        # Clear buffers so they get reallocated on the new device
        self._x_b_buffer = None
        self._d_a_buffer = None
        self._x_pad = None
        self._d_pad = None
        # Clear transfer vectors (one-hot reading)
        self._transfer_vec_a = None
