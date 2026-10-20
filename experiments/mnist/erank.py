#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Entropy effective rank of a matrix.

compute_erank_via_gram returns the entropy effective rank of Roy and Vetterli
(2007), erank(X) = exp(-sum_i p_i log p_i) with p_i = s_i / sum_j s_j over the
singular values s_i of X. The singular values are obtained from the
eigenvalues of the smaller Gram matrix (X @ X.T or X.T @ X).
run_erank_fixed_lr.py applies it to C - C_0 of the LR-TT layer after every epoch.
"""

import torch
from torch import Tensor


def compute_erank_via_gram(X: Tensor, eps: float = 1e-12) -> float:
    """Compute effective rank via Gram matrix eigenvalues.

    More efficient for tall/wide matrices:
        G = X @ X.T (if n < m) or X.T @ X (if m < n)
        l_i = eigenvalues of G, s_i = sqrt(l_i)
        erank = exp(-sum_i p_i log(p_i)) where p_i = s_i / sum_j s_j

    Args:
        X: Input matrix tensor
        eps: Small value for numerical stability

    Returns:
        Effective rank (float)
    """
    if X.numel() == 0:
        return 0.0

    if X.dim() == 1:
        X = X.unsqueeze(0)

    n, m = X.shape

    # Choose smaller dimension for Gram matrix
    if n < m:
        G = X @ X.T
    else:
        G = X.T @ X

    # Get eigenvalues (symmetric matrix)
    try:
        eigenvalues = torch.linalg.eigvalsh(G)
    except RuntimeError:
        return 0.0

    # Filter positive eigenvalues and take sqrt to get singular values
    eigenvalues = eigenvalues[eigenvalues > 1e-20]
    if len(eigenvalues) == 0:
        return 0.0

    S = torch.sqrt(eigenvalues)

    # Normalize singular values (same as SVD-based erank)
    S_sum = S.sum()
    if S_sum < eps:
        return 0.0

    p = S / S_sum

    # Entropy and effective rank
    H = -torch.sum(p * torch.log(p + eps))
    return float(torch.exp(H).item())
