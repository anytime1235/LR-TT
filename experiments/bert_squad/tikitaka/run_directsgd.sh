#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# TPE search for the direct-SGD baseline (single_rpu, one analog tile per layer),
# 10 bit, QKVO trained on the all_train_attention substrate.
#
# Study sgd_10b_allanalog_qkvo, N_TRIALS (default 50) trials. Its best trial
# sets the direct-SGD hyperparameters of run_bitseed.sh (Fig. 5g, Single RPU).
#
# Search space
#   analog_lr      log [0.002, 1.0]    the pulse-count learning rate of the tile
#   classifier_lr  log [0.0005, 0.02]  (ln_lr tied)
#   Trial 0: analog_lr 0.016, classifier_lr 0.003.
# single_rpu builds a SingleRPUConfig without TransferCompound, so there is no
# fast_lr and analog_lr acts on the tile directly. transfer_lr and
# transfer_every are searched only for ttv1 in paper_experiment.py.
#
# w_max is +-1.0: the single tile holds the pretrained BERT weights (mapped with
# weight_scaling_omega 1.0). With --n-bits 10 this gives dw_min 1.953e-3, the
# step of the core tile in the ttv1 studies.
#
# Substrate, IO and schedule as in the ttv1 studies: --target-layers
# all_train_attention, IO 8 bit, out_noise 0.04 (build_single_rpu_config
# applies it to the forward and backward paths), AbsMax, Iterative;
# ConstantStep device, stochastic pulses, desired_bl 31; 5 epochs, patience 2,
# batch 48, seed 42, warmup_ratio 0.039566, min_lr_rate 0.0.
#
# usage:  GPU=0 bash run_directsgd.sh               # BITS (default 10) sets --n-bits
# Output: $RESULTS_ROOT/directsgd_allanalog_20260802/ (db/, sgd10b_tpe/)
# ============================================================================
set -uo pipefail
PYTHON="${PYTHON:-python3}"
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_ROOT:-$PAPER/results}"
N_TRIALS="${N_TRIALS:-50}"
WARMUP_RATIO=0.039566
MIN_LR_RATE=0.0

BASE="$RESULTS/directsgd_allanalog_20260802"
GPU="${GPU:-0}"
BITS="${BITS:-10}"
STUDY="sgd_${BITS}b_allanalog_qkvo"

# Trial 0 (anchor) and search bands.
ANCHOR_SGD_ALR=0.016
ALR_RANGE="0.002 1.0"
ANCHOR_CLR=0.003
CLR_RANGE="0.0005 0.02"

COMMON="--mode tpe --method single_rpu --seed 42 \
  --target-layers all_train_attention \
  --epochs 5 --patience 2 --batch-size 48 --grad-accum-steps 1 \
  --device-type constant_step --pulse-type stochastic --desired-bl 31 \
  --n-bits $BITS --w-max 1.0 \
  --io-bits 8 --out-noise 0.04 --noise-management abs_max \
  --warmup-ratio $WARMUP_RATIO --min-lr-rate $MIN_LR_RATE"

cd "$PAPER"
OUT="$BASE/sgd${BITS}b_tpe"; mkdir -p "$OUT" "$BASE/db"
echo "=== [$(date '+%F %T')] START direct-SGD ${BITS}b  ${N_TRIALS} trials  GPU$GPU -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
  "$PYTHON" paper_experiment.py $COMMON \
    --analog-lr "$ANCHOR_SGD_ALR" --alr-range $ALR_RANGE \
    --classifier-lr "$ANCHOR_CLR" --clr-range $CLR_RANGE \
    --n-trials "$N_TRIALS" --tpe-startup-trials 10 \
    --study-name "$STUDY" --db-dir "$BASE/db" \
    --output-dir "$OUT" >> "$OUT/run.log" 2>&1 < /dev/null
echo "=== [$(date '+%F %T')] DONE direct-SGD ${BITS}b (exit $?)"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|Trains analog\|IO\[forward' $OUT/run.log"
