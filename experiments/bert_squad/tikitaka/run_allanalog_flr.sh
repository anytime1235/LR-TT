#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) TPE studies with fast_lr searched, QKVO or FFN trained, all
# 72 encoder linear layers on analog tiles (Fig. 5g, Tiki-Taka hyperparameters;
# Fig. 6a, Tiki-Taka FFN).
#
# qkvo  --target-layers all_train_attention  (QKVO trains, FFN frozen on analog)
#       study ttv1_flr_qkvo; its best trial is the Tiki-Taka setting of
#       run_bitseed.sh (Fig. 5g) and trial 0 of run_auxbit10_anchor87.sh,
#       run_all_flr.sh, run_6t1c_af1.sh and run_6t1c_full.sh.
# ffn   --target-layers all_train_ffn        (FFN trains, QKVO frozen on analog)
#       study ttv1_flr_ffn (Fig. 6a, Tiki-Taka FFN).
#
# Auxiliary and core tile 10 bit, w_max 1.0 (w_max_fast is not searched).
# Search space (TTV1_SEARCH of sweep_common.sh), N_TRIALS (default 50) trials
# per study:
#     fast_lr        log [0.001, 10.0]   pulse-count learning rate of the A tile
#     transfer_lr    log [0.005, 0.5]
#     classifier_lr  log [0.001, 0.008]  (ln_lr tied)
#     transfer_every categorical {1, 2, 3, 6}
#     analog_lr      pinned to one value (inert for ttv1)
#   Trial 0 is the anchor of sweep_common.sh (fast_lr 0.1, transfer_every 2).
# Other settings: TTV1_COMMON of sweep_common.sh.
#
# usage:  GPU=0 bash run_allanalog_flr.sh            # qkvo, then ffn
#         GPU=1 bash run_allanalog_flr.sh ffn        # one condition
# Output: $RESULTS_ROOT/tiki_allanalog_flr_20260802/ (db/, qkvo/, ffn/)
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"

BASE="$RESULTS/tiki_allanalog_flr_20260802"
GPU="${GPU:-0}"
CONDITIONS="${1:-qkvo ffn}"
mkdir -p "$BASE/db"
cd "$PAPER"

run_study() {
  local NAME=$1 TARGET=$2
  local OUT="$BASE/$NAME"; mkdir -p "$OUT"
  echo "=== [$(date '+%F %T')] START flr_$NAME  GPU$GPU  target=$TARGET  trials=$N_TRIALS  log=$OUT/run.log"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
    "$PYTHON" paper_experiment.py $TTV1_COMMON $TTV1_SEARCH \
      --target-layers "$TARGET" \
      --n-bits 10 --n-bits-slow 10 --w-max 1.0 \
      --study-name "ttv1_flr_${NAME}" --db-dir "$BASE/db" \
      --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
  echo "=== [$(date '+%F %T')] DONE  flr_$NAME (exit $?)"
}

for C in $CONDITIONS; do
  case "$C" in
    qkvo) run_study qkvo all_train_attention ;;
    ffn)  run_study ffn  all_train_ffn ;;
    *)    echo "unknown condition '$C' (expected qkvo or ffn)" >&2; exit 2 ;;
  esac
done

echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|Trains analog\|IO\[forward' $BASE/qkvo/run.log"
