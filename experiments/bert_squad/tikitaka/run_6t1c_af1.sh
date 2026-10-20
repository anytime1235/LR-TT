#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) TPE study with a 6T1C auxiliary tile, asymmetry only
# (r_AF = 1, r_noise = 0), QKVO trained.
#
# Study ttv1_6t1c_af1_qkvo, SIXT1C_N_TRIALS (default 50) trials. Its best trial
# is the anchor of the af1nz0 sweeps of run_af_noise_tuned_sweep.sh (Fig. 6c,
# Tiki-Taka hyperparameters tuned at r_AF = 1; Supp. Fig. S8).
#
# Device
#   A (fast/aux)  LinearStepDevice with the 6T1C-calibrated asymmetry
#                   gamma_up   = -0.1678 * r_AF
#                   gamma_down =  0.1410 * r_AF      (r_AF = 1 here)
#                 step dw_min = 0.001981 * w_max_fast (6T1C step, about 1010
#                 levels over [-w_max_fast, w_max_fast]); --ls-noise-ratio 0.0
#                 switches cycle-to-cycle noise and device-to-device variation
#                 off, so the asymmetry is the only non-ideality. The 6T1C noise
#                 is added in run_6t1c_full.sh.
#   C (slow/core) ConstantStepDevice, 10 bit, w_max 1.0
#   At r_AF = 0 (and r_noise = 0) the LinearStep tile updates like a
#   ConstantStep tile with the same dw_min.
#
# Search space (identical in run_6t1c_full.sh)
#   fast_lr        log [0.01, 5.0]     pulse-count learning rate of the A tile
#   w_max_fast     log [0.01, 10.0]    range and step of the A tile together
#                                      (the level count stays the same); for a
#                                      LinearStep tile it also sets where the
#                                      asymmetric slope acts
#   transfer_lr    log [0.005, 5.0]
#   classifier_lr  log [0.001, 0.004]  (ln_lr tied); acts on the digital head
#                                      and LayerNorm only
#   transfer_every 1 (not searched)
#   analog_lr      pinned; inert for ttv1, kept > 0 (lr == 0.0 marks a frozen
#                  tile, which holds the FFN tiles still)
#   Trial 0 starts from the best trial of ttv1_flr_qkvo (ConstantStep A tile;
#   run_allanalog_flr.sh) with w_max_fast 1.0.
#
# All other settings are TTV1_COMMON (sweep_common.sh) with only the fast-tile
# device replaced by linear_step: IO 8 bit / out_noise 0.04 / AbsMax /
# Iterative, --target-layers all_train_attention, gamma 0, desired_bl 31,
# 5 epochs, patience 2, batch 48, seed 42, warmup_ratio 0.039566,
# min_lr_rate 0.0.
#
# usage:  GPU=0 bash run_6t1c_af1.sh
# Output: $RESULTS_ROOT/tiki_6t1c_af1_20260806/ (db/, af1/)
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"

BASE="$RESULTS/tiki_6t1c_af1_20260806"
GPU="${GPU:-0}"
NT="${SIXT1C_N_TRIALS:-50}"

AF_RATIO=1.0          # 6T1C asymmetry
LS_NOISE=0.0          # asymmetry only; the 6T1C noise is run_6t1c_full.sh

# trial 0: best trial of ttv1_flr_qkvo (ConstantStep A tile)
A_FLR=1.4449830545981817
A_TLR=0.008554780925054559
A_CLR=0.002362382049063794
A_WMF=1.0
A_TE=1

FLR_RANGE_6="0.01 5.0"
WMF_RANGE_6="0.01 10.0"
TLR_RANGE_6="0.005 5.0"
CLR_RANGE_6="0.001 0.004"

# TTV1_COMMON carries "--device-type constant_step --device-type-slow constant_step".
# Swap only the fast-tile token; the slow tile stays constant_step. The pattern
# has a trailing space so it cannot match "--device-type-slow".
COMMON_6T1C="${TTV1_COMMON/--device-type constant_step /--device-type linear_step }"

mkdir -p "$BASE/db"
cd "$PAPER"

OUT="$BASE/af1"; mkdir -p "$OUT"
echo "=== GPU$GPU  6T1C A tile, AF ratio $AF_RATIO, noise $LS_NOISE, C=ConstantStep10b, $NT trials -> $OUT ==="
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
  "$PYTHON" paper_experiment.py $COMMON_6T1C \
    --target-layers all_train_attention \
    --dw-min 0.001981 --n-bits-slow 10 --w-max 1.0 \
    --ls-gamma-up-ratio "$AF_RATIO" --ls-gamma-down-ratio "$AF_RATIO" \
    --ls-noise-ratio "$LS_NOISE" \
    --analog-lr $ANCHOR_ALR --alr-range $ANCHOR_ALR $ANCHOR_ALR \
    --classifier-lr $A_CLR --clr-range $CLR_RANGE_6 \
    --transfer-lr $A_TLR --tlr-range $TLR_RANGE_6 \
    --transfer-every $A_TE --te-choices "$A_TE" \
    --fast-lr $A_FLR --flr-range $FLR_RANGE_6 \
    --w-max-fast $A_WMF --wmf-range $WMF_RANGE_6 \
    --n-trials "$NT" --tpe-startup-trials 10 \
    --study-name "ttv1_6t1c_af1_qkvo" --db-dir "$BASE/db" \
    --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
echo "=== done (exit $?)  log=$OUT/run.log"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|IO\[forward\|Effective dw_min' $OUT/run.log"
