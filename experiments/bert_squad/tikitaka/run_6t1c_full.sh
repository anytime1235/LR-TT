#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) TPE study with the full 6T1C auxiliary-tile model,
# asymmetry and noise (r_AF = 1, r_noise = 1), QKVO trained.
#
# Study ttv1_6t1c_full_qkvo, FULL6T1C_N_TRIALS (default 50) trials. Its best
# trial is the anchor of the af1nz1 sweep of run_af_noise_tuned_sweep.sh
# (Fig. 6c, Tiki-Taka hyperparameters tuned at r_noise = 1; Supp. Fig. S8).
#
# Same study design as run_6t1c_af1.sh (search space, anchor, all other
# settings); the only difference is --ls-noise-ratio 1.0:
#       af1   r_AF = 1, r_noise = 0   asymmetry only        (run_6t1c_af1.sh)
#       full  r_AF = 1, r_noise = 1   asymmetry and noise   (this script)
#
# Device
#   A (fast/aux)  LinearStepDevice, gamma_up -0.1678, gamma_down 0.1410
#                 (r_AF = 1), step dw_min = 0.001981 * w_max_fast, and with
#                 --ls-noise-ratio 1.0 the 6T1C-calibrated noise and variation
#                 used for the LR-TT 6T1C device:
#                   dw_min_std 0.3     dw_min_dtod 0.1    up_down_dtod 0.01
#                   w_max_dtod 0.05    w_min_dtod 0.05
#                   gamma_up_dtod 0.05 gamma_down_dtod 0.05
#                 (write_noise_std 0, mult_noise False)
#   C (slow/core) ConstantStepDevice, 10 bit, w_max 1.0
#
# Search space (as in run_6t1c_af1.sh)
#   fast_lr        log [0.01, 5.0]
#   w_max_fast     log [0.01, 10.0]
#   transfer_lr    log [0.005, 5.0]
#   classifier_lr  log [0.001, 0.004]  (ln_lr tied)
#   transfer_every 1 (not searched)
#   analog_lr      pinned (inert for ttv1, kept > 0)
#   Trial 0 starts from the best trial of ttv1_flr_qkvo (ConstantStep A tile)
#   with w_max_fast 1.0.
#
# All other settings are TTV1_COMMON (sweep_common.sh): IO 8 bit / out_noise
# 0.04 / AbsMax / Iterative, --target-layers all_train_attention, gamma 0,
# desired_bl 31, 5 epochs, patience 2, batch 48, seed 42, warmup_ratio
# 0.039566, min_lr_rate 0.0.
#
# usage:  GPU=0 bash run_6t1c_full.sh
# Output: $RESULTS_ROOT/tiki_6t1c_full_20260806/ (db/, full/)
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"

BASE="$RESULTS/tiki_6t1c_full_20260806"
GPU="${GPU:-0}"
NT="${FULL6T1C_N_TRIALS:-50}"

AF_RATIO=1.0          # 6T1C asymmetry
LS_NOISE=1.0          # 6T1C noise and variation (run_6t1c_af1.sh: 0.0)

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

# Swap only the fast-tile device token; the trailing space stops it matching
# "--device-type-slow", so the C tile stays ConstantStep.
COMMON_6T1C="${TTV1_COMMON/--device-type constant_step /--device-type linear_step }"

mkdir -p "$BASE/db"
cd "$PAPER"

OUT="$BASE/full"; mkdir -p "$OUT"
echo "=== GPU$GPU  full 6T1C (asymmetry + noise), AF=$AF_RATIO noise=$LS_NOISE, C=ConstantStep10b, $NT trials -> $OUT ==="
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
    --study-name "ttv1_6t1c_full_qkvo" --db-dir "$BASE/db" \
    --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
echo "=== done (exit $?)  log=$OUT/run.log"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|Effective dw_min' $OUT/run.log"
