#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) TPE study with all 72 encoder linear layers trained
# (--target-layers all), fast_lr searched (Fig. 6a, Tiki-Taka ALL).
#
# Study ttv1_flr_all, ALL_N_TRIALS (default 20) trials, --tpe-startup-trials 5.
# Trial 0 starts from the best trial of ttv1_flr_qkvo (run_allanalog_flr.sh):
#     fast_lr 1.4449830545981817   transfer_lr 0.008554780925054559
#     transfer_every 1             classifier_lr 0.002362382049063794
# The Tiki-Taka ALL point of Fig. 6a uses the best trial of this study
# (fast_lr 0.5604, transfer_lr 0.015402, transfer_every 3,
# classifier_lr 0.0056464).
#
# Search space
#   fast_lr        log [0.05, 5.0]     pulse-count learning rate of the A tile
#   classifier_lr  log [0.001, 0.008]  (ln_lr tied)
#   transfer_lr    log [0.005, 1.0]
#   transfer_every categorical {1, 2, 3}
#   analog_lr      pinned (inert for ttv1, where fast_lr is the pulse-count
#                  learning rate of the A tile)
#
# Substrate `all`: all 72 encoder linear layers on analog tiles with IO, all of
# them trained (same tile count as all_train_*). Other settings from
# TTV1_COMMON (sweep_common.sh): auxiliary and core tile 10 bit ConstantStep,
# w_max 1.0, IO 8 bit / out_noise 0.04 / out_bound 12.0 / AbsMax / Iterative,
# gamma 0, 5 epochs, patience 2, batch 48, seed 42, warmup_ratio 0.039566,
# min_lr_rate 0.0.
#
# usage:  GPU=0 bash run_all_flr.sh
# Output: $RESULTS_ROOT/tiki_all_flr_20260803/ (db/, all/)
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"

BASE="$RESULTS/tiki_all_flr_20260803"
GPU="${GPU:-0}"
# ALL_N_TRIALS (default 20) sets the number of trials of this study.
N_TRIALS="${ALL_N_TRIALS:-20}"

# trial 0: best trial of ttv1_flr_qkvo (run_allanalog_flr.sh)
A_FLR=1.4449830545981817
A_TLR=0.008554780925054559
A_CLR=0.002362382049063794
A_TE=1

FLR_RANGE_ALL="0.05 5.0"
CLR_RANGE_ALL="0.001 0.008"
TLR_RANGE_ALL="0.005 1.0"
TE_CHOICES_ALL="1,2,3"

mkdir -p "$BASE/db"
cd "$PAPER"

OUT="$BASE/all"; mkdir -p "$OUT"
echo "=== GPU$GPU: ttv1 \`all\` with fast_lr swept, $N_TRIALS trials, anchor = best trial of ttv1_flr_qkvo (flr $A_FLR) -> $OUT ==="
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
  "$PYTHON" paper_experiment.py $TTV1_COMMON \
    --target-layers all \
    --n-bits 10 --n-bits-slow 10 --w-max 1.0 \
    --analog-lr $ANCHOR_ALR --alr-range $ANCHOR_ALR $ANCHOR_ALR \
    --classifier-lr $A_CLR --clr-range $CLR_RANGE_ALL \
    --transfer-lr $A_TLR --tlr-range $TLR_RANGE_ALL \
    --transfer-every $A_TE --te-choices $TE_CHOICES_ALL \
    --fast-lr $A_FLR --flr-range $FLR_RANGE_ALL \
    --n-trials "$N_TRIALS" --tpe-startup-trials 5 \
    --study-name "ttv1_flr_all" --db-dir "$BASE/db" \
    --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
echo "=== done (exit $?)  log=$OUT/run.log"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|Analog layers\|Trains analog\|IO\[forward' $OUT/run.log"
