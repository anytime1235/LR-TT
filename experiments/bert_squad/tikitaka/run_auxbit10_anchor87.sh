#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) TPE study at auxiliary and core tile 10 bit with w_max_fast
# searched, QKVO trained (Fig. 5h, 10-bit point; Fig. 6a, Tiki-Taka QKVO; its
# best trial is the r_AF = 0, r_noise = 0 anchor of run_af_noise_tuned_sweep.sh,
# Fig. 6c and Supp. Fig. S8).
#
# Study ttv1_auxbit10b_anchor87, A87_N_TRIALS (default 50) trials. Trial 0
# starts from the best trial of ttv1_flr_qkvo (run_allanalog_flr.sh) with
# w_max_fast 1.0:
#     fast_lr 1.4449830545981817   transfer_lr 0.008554780925054559
#     transfer_every 1             classifier_lr 0.002362382049063794
# The best trial (fast_lr 1.7069, transfer_lr 0.017218, transfer_every 6,
# w_max_fast 2.3834, classifier_lr 0.0031826) is the af0nz0 anchor of
# run_af_noise_tuned_sweep.sh.
#
# Search space
#   fast_lr        log [0.05, 5.0]     pulse-count learning rate of the A tile
#   transfer_lr    log [0.002, 1.0]
#   w_max_fast     log [0.01, 10.0]    range of the A tile; dw_min_fast =
#                                      w_max_fast / 2^9 (1024 levels)
#   classifier_lr  log [0.001, 0.004]  (ln_lr tied); acts on the digital head
#                                      and LayerNorm only
#   transfer_every categorical {1, 2, 3, 6}
#   analog_lr      pinned; inert for ttv1, kept > 0 (lr == 0.0 marks a frozen
#                  tile, which holds the FFN tiles still)
#
# All other settings are TTV1_COMMON (sweep_common.sh): --target-layers
# all_train_attention, IO 8 bit / out_noise 0.04 / AbsMax / Iterative,
# ConstantStep A and C tiles, gamma 0, desired_bl 31, 5 epochs, patience 2,
# batch 48, seed 42, warmup_ratio 0.039566, min_lr_rate 0.0.
#
# usage:  GPU=0 bash run_auxbit10_anchor87.sh
# Output: $RESULTS_ROOT/tiki_auxbit10_anchor87_20260806/ (db/, aux10b/)
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"

BASE="$RESULTS/tiki_auxbit10_anchor87_20260806"
GPU="${GPU:-0}"
NT="${A87_N_TRIALS:-50}"
AUX_BITS=10
CORE_BITS=10

# trial 0: best trial of ttv1_flr_qkvo (run_allanalog_flr.sh)
A_FLR=1.4449830545981817
A_TLR=0.008554780925054559
A_CLR=0.002362382049063794
A_WMF=1.0
A_TE=1

FLR_RANGE_A="0.05 5.0"
TLR_RANGE_A="0.002 1.0"
WMF_RANGE_A="0.01 10.0"
CLR_RANGE_A="0.001 0.004"
TE_CHOICES_A="1,2,3,6"

mkdir -p "$BASE/db"
cd "$PAPER"

OUT="$BASE/aux10b"; mkdir -p "$OUT"
echo "=== GPU$GPU aux 10b, trial 0 = best trial of ttv1_flr_qkvo, $NT trials -> $OUT ==="
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
  "$PYTHON" paper_experiment.py $TTV1_COMMON \
    --target-layers all_train_attention \
    --n-bits "$AUX_BITS" --n-bits-slow "$CORE_BITS" --w-max 1.0 \
    --analog-lr $ANCHOR_ALR --alr-range $ANCHOR_ALR $ANCHOR_ALR \
    --classifier-lr $A_CLR --clr-range $CLR_RANGE_A \
    --transfer-lr $A_TLR --tlr-range $TLR_RANGE_A \
    --transfer-every $A_TE --te-choices "$TE_CHOICES_A" \
    --fast-lr $A_FLR --flr-range $FLR_RANGE_A \
    --w-max-fast $A_WMF --wmf-range $WMF_RANGE_A \
    --n-trials "$NT" --tpe-startup-trials 10 \
    --study-name "ttv1_auxbit10b_anchor87" --db-dir "$BASE/db" \
    --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
echo "=== done (exit $?)  log=$OUT/run.log"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]' $OUT/run.log"
