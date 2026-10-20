# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""plot_qzr_odr_maps.py: QZR / ODR heat maps of Fig. 5a and 5b (1x2 layout).

Panels:
  (a) QZR_nonzero heatmap (12 layer x 6 sublayer)
  (b) ODR heatmap (log10 scale, 12x6)

Input: metrics_paper_A_rootcause_summary.csv written by backward_diagnostics.py
(default --data-dir: ./results/backward_diagnostics, the default --out-dir of
backward_diagnostics.py).

Usage:
  python plot_qzr_odr_maps.py [--data-dir DIR] [--out-dir DIR] [--run-tag v3]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="QZR / ODR maps - AIMC BERT-base")
parser.add_argument("--data-dir", type=str, default="./results/backward_diagnostics",
                    help="directory holding metrics_paper_A_rootcause_summary.csv "
                         "(default: ./results/backward_diagnostics, the default "
                         "--out-dir of backward_diagnostics.py)")
parser.add_argument("--out-dir", type=str, default="./results/qzr_odr_maps")
parser.add_argument("--run-tag", type=str, default="v3",
                    help="suffix of the output file names (default: v3, the "
                         "default --run-tag of backward_diagnostics.py)")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR = args.out_dir
RUN_TAG = args.run_tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 12
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]

CSV_A_SUMMARY = os.path.join(DATA_DIR, "metrics_paper_A_rootcause_summary.csv")

# ---------------------------------------------------------------------------
# Publication RC params
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_mat(df, col):
    """Pivot DataFrame into (12, 6) ndarray [layer x sublayer]."""
    mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
    for _, row in df.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        if sl in SUBLAYER_ORDER and li < N_LAYERS:
            mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
    return mat


def annotate_heatmap(ax, mat, vmin, vmax, fmt=".3f", fontsize=7):
    """Write cell values on heatmap with auto-contrast text color."""
    for li in range(mat.shape[0]):
        for si in range(mat.shape[1]):
            val = mat[li, si]
            if np.isnan(val):
                continue
            txt = f"{val:{fmt}}"
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            color = "white" if norm > 0.6 or norm < 0.15 else "black"
            ax.text(si, li, txt, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")


def _save(fig, out_dir, basename, run_tag):
    """Save figure as PDF + PNG."""
    for ext in ["pdf", "png"]:
        path = os.path.join(out_dir, f"{basename}_{run_tag}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def panel_a_qzr(ax, df_a):
    """(a) QZR_nonzero heatmap - plasma, 0-0.5."""
    mat = to_mat(df_a, "QZR_nonzero")
    vmin, vmax = 0.0, 0.5
    im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title("(a) Quantization Zero Rate (QZR)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("QZR (fraction rounded to 0)")
    annotate_heatmap(ax, mat, vmin, vmax)
    return im


def panel_b_odr(ax, df_a):
    """(b) ODR heatmap - log10, hot_r, auto range."""
    odr_mat = to_mat(df_a, "ODR")
    log_odr = np.log10(np.clip(odr_mat, 1e-3, None))
    vmin, vmax = float(np.nanmin(log_odr)), float(np.nanmax(log_odr))
    im = ax.imshow(log_odr, aspect="auto", cmap="hot_r", origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title(r"(b) Outlier Dominance Ratio (log$_{10}$ ODR)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"log$_{10}$(ODR)")
    annotate_heatmap(ax, log_odr, vmin, vmax, fmt=".1f")
    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt.rcParams.update(RCPARAMS)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CSV ...")
    df_a = pd.read_csv(CSV_A_SUMMARY)
    print(f"  A_summary: {len(df_a)} rows")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    panel_a_qzr(axes[0], df_a)
    panel_b_odr(axes[1], df_a)

    plt.tight_layout()
    _save(fig, OUT_DIR, "qzr_odr_maps", RUN_TAG)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
