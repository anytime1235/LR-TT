#!/usr/bin/env python
# coding=utf-8
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Plot measured mu ECDF from the sub-pulse diagnostic runs (Fig. 5e).

Reads mu_distribution.json from the single_rpu stochastic run directories
(single_rpu_stoch_{8,10,12,14}b) and plots one ECDF per bit-width of the
measured mu distribution, summed over all diagnosed tiles and steps.

Usage:
    python plot_mu_ecdf.py [--results-dir DIR] [--output PATH]

Requires: the run directories of ../tikitaka/launchers/subpulse_diagnostics.sh,
whose update diagnostics write mu_distribution.json (histogram of the
per-element mu values).
"""

import argparse
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Data loading
# ============================================================================

def load_mu_distribution(run_dir):
    """Load mu_distribution.json from a run directory.

    Returns None if file doesn't exist.
    """
    path = os.path.join(run_dir, "mu_distribution.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_histograms(data):
    """Sum histogram counts across all tiles and steps.

    Args:
        data: Loaded mu_distribution.json dict.

    Returns:
        (bin_edges, total_counts) numpy arrays.
    """
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)

    for h in data["histograms"]:
        total_counts += np.array(h["counts"])

    return bin_edges, total_counts


def counts_to_ecdf(bin_edges, counts):
    """Convert histogram counts to ECDF (x, y) arrays.

    Returns:
        x: right bin edges (mu values)
        y: cumulative fraction P(mu <= x)
    """
    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    if total == 0:
        return bin_edges[1:], np.zeros_like(counts, dtype=float)
    return bin_edges[1:], cumulative / total


def frac_below_threshold(bin_edges, counts, threshold=1.0):
    """Compute fraction of elements with mu < threshold from histogram."""
    total = counts.sum()
    if total == 0:
        return 0.0
    mask = bin_edges[1:] <= threshold
    return counts[mask].sum() / total


# ============================================================================
# Plotting
# ============================================================================

BIT_COLORS = {8: '#d62728', 10: '#ff7f0e', 12: '#2ca02c', 14: '#1f77b4'}


def plot_ecdf_by_bitwidth(ax, base_dir, method_prefix, bits_list, title=None):
    """Plot ECDF curves for different bit-widths of one method.

    Returns dict of {bits: frac_mu_lt_1} for annotation.
    """
    fracs = {}
    for bits in bits_list:
        tag = f"{method_prefix}_{bits}b"
        run_dir = os.path.join(base_dir, tag)
        data = load_mu_distribution(run_dir)
        if data is None:
            continue

        bin_edges, counts = aggregate_histograms(data)
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        fracs[bits] = frac

        ax.plot(x, y, color=BIT_COLORS.get(bits, 'gray'),
                label=f"{bits}b (frac<1={frac:.3f})", linewidth=1.5)

    # mu=1 boundary
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1.05, 0.5, r'$\mu$=1', transform=ax.get_xaxis_transform(),
            fontsize=9, alpha=0.7)

    # Sub-pulse region shading
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1e-6,
               1.0, alpha=0.05, color='red')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu$ (expected pulse count per element)')
    ax.set_ylabel(r'P($\mu \leq$ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)

    return fracs


# ============================================================================
# Main
# ============================================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Plot measured mu ECDF")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(here, "..", "tikitaka", "results",
                                             "subpulse_diagnostics"),
                        help="Base directory containing run subdirectories "
                             "(default: ../tikitaka/results/subpulse_diagnostics, "
                             "the default RESULTS_DIR of "
                             "launchers/subpulse_diagnostics.sh)")
    parser.add_argument("--output", type=str,
                        default="./results/mu_ecdf/mu_distribution_measured_ecdf.png",
                        help="Output plot path")
    args = parser.parse_args()

    base_dir = args.results_dir

    # Check if any mu_distribution.json exists
    found = False
    for tag in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
        if os.path.exists(os.path.join(base_dir, tag, "mu_distribution.json")):
            found = True
            break

    if not found:
        print("=" * 60)
        print("ERROR: No mu_distribution.json found in any run directory.")
        print(f"  Searched: {base_dir}/*/mu_distribution.json")
        print()
        print("Run the sub-pulse diagnostics first; their update diagnostics")
        print("write the mu histograms.")
        print()
        print("Run:")
        print("  cd experiments/bert_squad/tikitaka")
        print("  bash launchers/subpulse_diagnostics.sh")
        print("=" * 60)
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # ECDF per bit-width
    plot_ecdf_by_bitwidth(
        ax, base_dir, "single_rpu_stoch", [8, 10, 12, 14],
        title=r'Measured $\mu$ ECDF - single_rpu stochastic (all steps)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
