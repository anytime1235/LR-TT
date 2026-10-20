#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Weight-bit sweep for direct SGD and Tiki-Taka, seed 42 (Fig. 5g, Single RPU
# and Tiki-Taka curves).
#
#   phase 1  direct SGD (single_rpu)  --n-bits       6,7,8,9,10
#   phase 2  Tiki-Taka (ttv1)         --n-bits-slow  6,7,8,9,10
#                                     (auxiliary tile 10 bit)
#   5 + 5 = 10 fixed-mode runs in sequence, about 2 h each.
#
# The hyperparameters of each method are those of the best trial of its TPE
# study, so only the weight bit width varies.
#
#   direct SGD  best trial of run_directsgd.sh (study sgd_10b_allanalog_qkvo)
#       analog_lr 0.00993747327721246   classifier_lr 0.00236681394122394
#       (single_rpu has one tile: --n-bits sets its resolution and analog_lr
#        is its pulse-count learning rate; there is no fast_lr)
#
#   Tiki-Taka   best trial of run_allanalog_flr.sh (study ttv1_flr_qkvo)
#       fast_lr 1.4449830545981817     transfer_lr 0.008554780925054559
#       transfer_every 1               classifier_lr 0.002362382049063794
#       analog_lr 0.011861648510064505  (inert for ttv1, kept > 0, see below)
#
# --ln-lr is passed explicitly for both methods: fixed mode defaults ln_lr to
# analog_lr, and the TPE studies tie it to classifier_lr.
#
# --analog-lr for ttv1 stays > 0. The TransferCompound uses fast_lr as the
# pulse-count learning rate of the auxiliary tile, so analog_lr does not reach
# a tile; lr 0.0 marks a frozen tile, which holds the FFN tiles still.
#
# Every run trains all 5 epochs: no early stopping (--patience 0), no Optuna
# pruning in fixed mode, and --max-steps 0 (off).
#
# Bit width and dw_min (2 / 2^bits):
#     6b 0.03125   7b 0.015625   8b 0.0078125   9b 0.00390625   10b 0.001953125
#
# Substrate for both: --target-layers all_train_attention (all 72 encoder
# linear layers analog with IO; QKVO trains, FFN stays at lr 0). IO 8 bit
# (res 1/254), out_noise 0.04, out_bound 12.0, AbsMax, Iterative; ConstantStep
# devices without device noise; 5 epochs, batch 48, warmup_ratio 0.039566,
# min_lr_rate 0.0.
#
# usage:  bash run_bitseed.sh
#         GPU=1 BIT_LIST="6 10" bash run_bitseed.sh   # BIT_LIST default 6 7 8 9 10
#         (SEEDS sets the seeds, default 42)
# Output: $RESULTS_ROOT/bitseed_20260803/sgd/b<BB>_s42/ and
#         $RESULTS_ROOT/bitseed_20260803/ttv1/core<BB>_s42/
# ============================================================================
set -uo pipefail

PYTHON="${PYTHON:-python3}"
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_ROOT:-$PAPER/results}"
BASE="$RESULTS/bitseed_20260803"
GPU="${GPU:-0}"

BIT_LIST="${BIT_LIST:-6 7 8 9 10}"
SEEDS="${SEEDS:-42}"
AUX_BITS=10

# --- direct SGD: best trial of run_directsgd.sh ---
S_ALR=0.00993747327721246
S_CLR=0.00236681394122394

# --- Tiki-Taka: best trial of ttv1_flr_qkvo (run_allanalog_flr.sh) ---
T_ALR=0.011861648510064505
T_CLR=0.002362382049063794
T_FLR=1.4449830545981817
T_TLR=0.008554780925054559
T_TE=1

COMMON="--mode fixed --target-layers all_train_attention \
  --epochs 5 --patience 0 --batch-size 48 --grad-accum-steps 1 \
  --io-bits 8 --out-noise 0.04 --noise-management abs_max \
  --w-max 1.0 --desired-bl 31 \
  --warmup-ratio 0.039566 --min-lr-rate 0.0"

mkdir -p "$BASE"
cd "$PAPER"

echo "=== [$(date '+%F %T')] PHASE 1: direct SGD (single_rpu) ==="
for BITS in $BIT_LIST; do
  for SEED in $SEEDS; do
    OUT="$BASE/sgd/b$(printf '%02d' "$BITS")_s${SEED}"; mkdir -p "$OUT"
    echo "=== [$(date '+%F %T')] START sgd ${BITS}b seed=$SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
      "$PYTHON" paper_experiment.py $COMMON \
        --method single_rpu --seed "$SEED" \
        --device-type constant_step --pulse-type stochastic \
        --n-bits "$BITS" \
        --analog-lr "$S_ALR" --classifier-lr "$S_CLR" --ln-lr "$S_CLR" \
        --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
    echo "=== [$(date '+%F %T')] DONE  sgd ${BITS}b seed=$SEED (exit $?)"
  done
done

echo "=== [$(date '+%F %T')] PHASE 2: TikiTaka (ttv1), core swept, aux ${AUX_BITS}b ==="
for BITS in $BIT_LIST; do
  for SEED in $SEEDS; do
    OUT="$BASE/ttv1/core$(printf '%02d' "$BITS")_s${SEED}"; mkdir -p "$OUT"
    echo "=== [$(date '+%F %T')] START ttv1 core=${BITS}b aux=${AUX_BITS}b seed=$SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
      "$PYTHON" paper_experiment.py $COMMON \
        --method ttv1 --seed "$SEED" \
        --gamma 0.0 --with-reset-prob 0 \
        --device-type constant_step --device-type-slow constant_step \
        --scale-transfer-lr false --units-in-mbatch true \
        --n-bits "$AUX_BITS" --n-bits-slow "$BITS" \
        --analog-lr "$T_ALR" --classifier-lr "$T_CLR" --ln-lr "$T_CLR" \
        --fast-lr "$T_FLR" --transfer-lr "$T_TLR" --transfer-every "$T_TE" \
        --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
    echo "=== [$(date '+%F %T')] DONE  ttv1 core=${BITS}b seed=$SEED (exit $?)"
  done
done

echo "=== [$(date '+%F %T')] ALL BIT x SEED RUNS COMPLETE ==="
