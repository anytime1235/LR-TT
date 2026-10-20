#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Sub-pulse diagnostic figure (mu ECDF and update quality)

Reads the run directories written by ../tikitaka/launchers/subpulse_diagnostics.sh
(single_rpu stochastic runs at 8, 10, 12 and 14 bit: mu_distribution.json and
update_diagnostics_summary.json).

subpulse_overview.png (2 panels):
  (a) mu ECDF by bit-width (stochastic 8/10/12/14b)
  (b) Update quality vs bit-width: cosine similarity + recovery ratio
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Paths --
HERE = os.path.dirname(os.path.abspath(__file__))
_cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_cli.add_argument("--results-dir", type=str,
                  default=os.path.join(HERE, "..", "tikitaka", "results",
                                       "subpulse_diagnostics"),
                  help="output directory of launchers/subpulse_diagnostics.sh")
_cli.add_argument("--out-dir", type=str, default="./results/subpulse_figures")
_args = _cli.parse_args()
BASE = _args.results_dir
OUT_DIR = _args.out_dir

# -- Style --
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BIT_COLORS = {6: "#9467bd", 8: "#d62728", 10: "#ff7f0e", 12: "#2ca02c", 14: "#1f77b4"}


# ===================================================================
#  Data loaders
# ===================================================================

def load_diagnostics_summary(tag):
    path = os.path.join(BASE, tag, "update_diagnostics_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_mu_distribution(tag):
    path = os.path.join(BASE, tag, "mu_distribution.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_metrics(tag):
    """Mean cosine similarity and final recovery ratio over the diagnosed tiles."""
    diag = load_diagnostics_summary(tag)
    if diag is None:
        return None
    layers = list(diag.keys())
    return {
        "cosine": np.mean([diag[l]["mean_cosine_sim"] for l in layers]),
        "recovery": np.mean([diag[l]["final_recovery_ratio"] for l in layers]),
    }


def mu_ecdf(data, step=None, tile_filter=None):
    """Compute ECDF from mu_distribution.json."""
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)
    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        if tile_filter is not None and not tile_filter(h["tile_name"]):
            continue
        total_counts += np.array(h["counts"])
    cumulative = np.cumsum(total_counts)
    total = cumulative[-1]
    if total == 0:
        return bin_edges[1:], np.zeros_like(total_counts)
    return bin_edges[1:], cumulative / total


def frac_below(data, threshold=1.0, step=None):
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)
    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        total_counts += np.array(h["counts"])
    total = total_counts.sum()
    if total == 0:
        return 0.0
    mask = bin_edges[1:] <= threshold
    return total_counts[mask].sum() / total


# ===================================================================
#  OVERVIEW FIGURE: 2 panels
# ===================================================================

def plot_overview():
    fig, (ax_ecdf, ax_quality) = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- (a) mu ECDF by bit-width --
    bits_list = [8, 10, 12, 14]
    for bits in bits_list:
        data = load_mu_distribution(f"single_rpu_stoch_{bits}b")
        if data is None:
            continue
        x, y = mu_ecdf(data)
        frac = frac_below(data, 1.0)
        ax_ecdf.plot(x, y, color=BIT_COLORS[bits], linewidth=1.8,
                     label=rf"{bits}b  (P($\mu$<1)={frac:.3f})")

    # mu=1 boundary
    ax_ecdf.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_ecdf.axvspan(1e-6, 1.0, alpha=0.06, color="red")
    ax_ecdf.text(0.08, 0.55, "sub-pulse\nregion",
                 transform=ax_ecdf.transAxes, fontsize=9, color="#c0392b",
                 fontstyle="italic", alpha=0.8)
    ax_ecdf.text(1.3, 0.5, r"$\mu$ = 1", fontsize=9, alpha=0.6,
                 rotation=90, va="center")

    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel(r"$\mu$  (expected pulse count per element)")
    ax_ecdf.set_ylabel(r"P($\mu \leq$ x)")
    ax_ecdf.set_xlim(1e-5, 1e3)
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.legend(loc="lower right", framealpha=0.9)
    ax_ecdf.grid(True, alpha=0.25)
    ax_ecdf.set_title(r"(a)  $\mu$ ECDF - Single RPU Stochastic", fontweight="bold")

    # -- (b) Update quality vs bit-width --
    cosines, recoveries, bits_plot = [], [], []
    for bits in bits_list:
        m = aggregate_metrics(f"single_rpu_stoch_{bits}b")
        if m is None:
            continue
        bits_plot.append(bits)
        cosines.append(m["cosine"])
        recoveries.append(m["recovery"])

    # Left y-axis: cosine similarity
    color_cos = "#2980b9"
    ax_quality.set_xlabel("Bit-width")
    ax_quality.set_ylabel("Cosine Similarity (direction fidelity)", color=color_cos)
    ln1 = ax_quality.plot(bits_plot, cosines, "o-", color=color_cos, linewidth=2,
                          markersize=8, label="Cosine Similarity", zorder=3)
    ax_quality.tick_params(axis="y", labelcolor=color_cos)
    ax_quality.set_ylim(0, 0.35)
    ax_quality.axhline(1.0, color=color_cos, linestyle=":", alpha=0.3)

    # Right y-axis: recovery ratio (log)
    ax_r = ax_quality.twinx()
    color_rec = "#e74c3c"
    ax_r.set_ylabel("Recovery Ratio  (||actual|| / ||target||)", color=color_rec)
    ln2 = ax_r.plot(bits_plot, recoveries, "s--", color=color_rec, linewidth=2,
                    markersize=8, label="Recovery Ratio", zorder=3)
    ax_r.tick_params(axis="y", labelcolor=color_rec)
    ax_r.set_yscale("log")
    ax_r.set_ylim(1, 100)
    ax_r.axhline(1.0, color=color_rec, linestyle=":", alpha=0.3)

    # Ideal annotations
    ax_quality.annotate("ideal: cos -> 1", xy=(13.5, 0.32), fontsize=7.5,
                        color=color_cos, alpha=0.6, fontstyle="italic")
    ax_r.annotate("ideal: recovery -> 1", xy=(12.5, 1.3), fontsize=7.5,
                  color=color_rec, alpha=0.6, fontstyle="italic")

    ax_quality.set_xticks(bits_plot)
    ax_quality.grid(True, alpha=0.25)
    ax_quality.set_title("(b)  Update Quality vs Bit-width (Stochastic)",
                         fontweight="bold")

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax_quality.legend(lines, labels, loc="center right", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "subpulse_overview.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Overview figure saved: {out_path}")


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_overview()
    print("Done.")
