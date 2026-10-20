#!/bin/bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Sub-pulse update diagnostics (Fig. 5e): single_rpu with stochastic pulses at
# 8, 10, 12 and 14 bit, 4 runs of 128 steps; seed 42, batch 48, perfect I/O
# (--io-bits 0), desired_bl 31, noise-free ConstantStep device with
# dw_min = 2 / 2^bits and w_max 1.0, QKVO on analog tiles (--target-layers
# attention); the other settings are the defaults of paper_experiment.py
# (analog_lr 0.016).
# Every step is diagnosed: --diag-update-exact writes the expected pulse count
# per element (mu) and the realized versus target update
# (update_diagnostics.csv, update_diagnostics_summary.json,
# mu_distribution.json with the mu histograms per tile and step,
# pq_distribution.json). The mu ECDFs of Fig. 5e are plotted by
# ../diagnostics/plot_mu_ecdf.py and plot_subpulse_figures.py.
# Output: RESULTS_DIR (default results/subpulse_diagnostics, relative to the
# tikitaka directory), one directory single_rpu_stoch_<bits>b per run and the
# log single_rpu_stoch.log.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/subpulse_diagnostics}"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Sub-pulse update diagnostics (128 steps per run) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 128 --batch-size 48 \
  --diag-update-exact --diag-steps 128 --log-every 1"

# single_rpu stochastic bits={8,10,12,14}
run_single_rpu_stoch() {
    for BITS in 8 10 12 14; do
        local TAG="single_rpu_stoch_${BITS}b"
        echo "[GPU $GPU] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
            $COMMON_FLAGS \
            --method single_rpu --pulse-type stochastic --n-bits $BITS \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU $GPU] DONE  $TAG $(date)"
    done
}

run_single_rpu_stoch 2>&1 | tee -a "$RESULTS_DIR/single_rpu_stoch.log"
echo ""
echo "=== Sub-pulse update diagnostics complete: $(date) ==="

# Summary table: per run, the mean over the diagnosed tiles of the per-tile
# values in update_diagnostics_summary.json.
$PYTHON - "$RESULTS_DIR" << 'PYEOF'
import json, os, sys

base = sys.argv[1]
cols = [("mean_eff_mu_p50", "mu_p50"), ("mean_eff_frac_mu_lt_1", "frac<1"),
        ("mean_cosine_sim", "cosine"), ("final_recovery_ratio", "recovery")]
print(f"\n{'Tag':<24} | {'tiles':>5} | " + " | ".join(f"{h:>9}" for _, h in cols))
print("-" * 78)
for bits in [8, 10, 12, 14]:
    tag = f"single_rpu_stoch_{bits}b"
    path = os.path.join(base, tag, "update_diagnostics_summary.json")
    if not os.path.exists(path):
        print(f"{tag:<24} | {'-':>5} | no update_diagnostics_summary.json")
        continue
    with open(path) as f:
        tiles = json.load(f)
    n = max(len(tiles), 1)
    vals = [sum(t[k] for t in tiles.values()) / n for k, _ in cols]
    print(f"{tag:<24} | {len(tiles):>5} | " + " | ".join(f"{v:>9.4g}" for v in vals))
PYEOF
