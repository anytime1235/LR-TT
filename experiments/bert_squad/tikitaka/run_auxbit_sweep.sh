#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) auxiliary-tile bit width x w_max_fast studies, QKVO trained
# (Fig. 5h: the 12- and 14-bit points, and trial 0 of the 6-bit study of
# run_auxbit_refine20.sh).
#
# One TPE study per auxiliary (fast, A) tile bit width B in BITS (default
# 6 12 14), N_TRIALS (default 50) trials each, on the all_train_attention
# substrate: all 72 encoder linear layers on analog tiles with IO, the 48
# attention layers (192 tiles) train and the 24 FFN layers (288 tiles) stay
# frozen on analog tiles. The core (slow, C) tile is 10 bit, w_max 1.0, in
# every study.
#
#   6 bit   ttv1_auxbit06b_flr_qkvo; its best trial is trial 0 of the 6-bit
#           study of run_auxbit_refine20.sh (Fig. 5h, 6 bit)
#   12 bit  ttv1_auxbit12b_flr_qkvo (Fig. 5h, 12 bit)
#   14 bit  ttv1_auxbit14b_flr_qkvo (Fig. 5h, 14 bit)
#
# Range of the auxiliary tile
#   The auxiliary tile keeps 2^B levels for any --w-max-fast, so
#       w_max = w_max_fast,  w_min = -w_max_fast,
#       dw_min_fast = 2 * w_max_fast / 2^B = w_max_fast / 2^(B-1).
#   This is the (multilevel, dw_min) parameterization of the LR-TT A/B tiles,
#   w_max = 2^multilevel * dw_min / 2, with B in the role of multilevel. The
#   auxiliary tile starts at 0 and does not enter the forward weight (gamma 0),
#   so w_max_fast may exceed the +-1.0 range of the core tile. Trial 0 uses
#   w_max_fast 1.0, clamped into the band.
#
# Search space (per-bit bands in bands_for() below; the other bands and the
# anchor from sweep_common.sh)
#                    6 bit                          12 and 14 bit
#     fast_lr        log [0.001, 10.0]              log [0.001, 5.0]
#     transfer_lr    log [0.005, 0.5]               log [0.005, 5.0]
#     w_max_fast     log [DW_LO, DW_HI] * 2^(B-1)   log [DW_LO * 2^(B-1), 10.0]
#                    = [2.56e-4, 0.096]
#     classifier_lr  log [0.001, 0.008]  (ln_lr tied)
#     transfer_every categorical {1, 2, 3, 6}
#     analog_lr      pinned to one value (inert for ttv1)
#   DW_LO = 8e-6 and DW_HI = 3e-3 are dw_min_fast values; with
#   w_max_fast = dw_min_fast * 2^(B-1) they give the w_max_fast bands above.
#   Trial 0 is the anchor of sweep_common.sh (fast_lr 0.1, transfer_every 2).
#
# Parallel runs
#   One instance runs the bit widths of BITS in turn; before a study it creates
#   the marker directory claims/bit<B>, so that several instances, each in its
#   own shell on its own GPU, run different bit widths:
#       GPU=0 bash run_auxbit_sweep.sh
#       GPU=1 bash run_auxbit_sweep.sh
#   Each study runs until it has N_TRIALS finished trials (COMPLETE, PRUNED or
#   FAIL); a resumed study runs only the remaining trials. To resume a bit
#   width after an interrupted run, remove its claims/bit<B> first.
#
# usage:  GPU=0 bash run_auxbit_sweep.sh         # foreground
#         BITS="12 14" bash run_auxbit_sweep.sh  # subset
#         bash run_auxbit_sweep.sh --status      # finished trials per bit width
# Output: $RESULTS_ROOT/tiki_auxbit_flr_20260802/ (db/, aux<BB>b/, claims/)
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"
BASE="$RESULTS/tiki_auxbit_flr_20260802"
BITS="${BITS:-6 12 14}"
GPU="${GPU:-0}"
WMF=1.0

# dw_min_fast values that set the w_max_fast bands of the header (times
# 2^(B-1); upper edge 10.0 for 12 and 14 bit).
DW_LO=8e-6
DW_HI=3e-3

COMMON="$TTV1_COMMON --target-layers all_train_attention --n-bits-slow 10 --w-max 1.0"

study_of() { printf 'ttv1_auxbit%02db_flr_qkvo' "$1"; }

# Search bands of bit width $1: B_FLR (fast_lr), B_TLR (transfer_lr) and
# B_WLO, B_WHI (w_max_fast).
bands_for() {
  case "$1" in
    6)     B_FLR="$FLR_RANGE"; B_TLR="$TLR_RANGE"
           B_WHI=$("$PYTHON" -c "print($DW_HI*2**($1-1))") ;;
    12|14) B_FLR="0.001 5.0"; B_TLR="0.005 5.0"; B_WHI=10.0 ;;
    *)     echo "  no search bands for bit width $1 (expected 6, 12 or 14)" >&2; return 1 ;;
  esac
  B_WLO=$("$PYTHON" -c "print($DW_LO*2**($1-1))")
}

finished_count() {
  "$PYTHON" - "$1" <<'PYEOF' 2>/dev/null || echo 0
import sqlite3, sys, os
if not os.path.exists(sys.argv[1]): print(0); raise SystemExit
c = sqlite3.connect(sys.argv[1])
print(c.execute("select count(*) from trials where state in "
                "('COMPLETE','PRUNED','FAIL')").fetchone()[0])
PYEOF
}

# ------------------------------------------------------------------- status
if [[ "${1:-}" == "--status" ]]; then
  printf '%-6s %-10s %-24s %s\n' BIT STATE STUDY FINISHED
  for B in $BITS; do
    S=$(study_of "$B"); ST=free
    [[ -d "$BASE/claims/bit$B" ]] && ST=claimed
    [[ -f "$BASE/claims/bit$B/done" ]] && ST=DONE
    printf '%-6s %-10s %-24s %s/%s\n' "$B" "$ST" "$S" "$(finished_count "$BASE/db/$S.db")" "$N_TRIALS"
  done
  exit 0
fi

# ------------------------------------------------------------------- run
# Take the bit widths one at a time; a bit width with a marker is skipped.
mkdir -p "$BASE/db" "$BASE/claims"
cd "$PAPER"
echo "=== [$(date '+%F %T')] GPU$GPU, bit widths: $BITS"

for B in $BITS; do
  bands_for "$B" || continue
  CLAIM="$BASE/claims/bit$B"
  mkdir "$CLAIM" 2>/dev/null || { echo "  bit $B has a marker (claims/bit$B), skip"; continue; }
  echo "GPU$GPU $(date '+%F %T')" > "$CLAIM/owner"

  S=$(study_of "$B")
  # Zero-padded to match the study name and so aux06b..aux14b sort correctly.
  OUT="$BASE/$(printf 'aux%02db' "$B")"; mkdir -p "$OUT"

  # Optuna runs n_trials additional trials on a loaded study -> ask for the rest.
  DONE=$(finished_count "$BASE/db/$S.db")
  LEFT=$(( N_TRIALS - DONE ))
  if (( LEFT <= 0 )); then
    echo "  bit $B already has $DONE/$N_TRIALS finished trials -> done"
    touch "$CLAIM/done"; continue
  fi

  echo "=== [$(date '+%F %T')] GPU$GPU START aux ${B}b  ($DONE done, $LEFT to go)"
  DWLO_ACT=$("$PYTHON" -c "print((2/2**$B)*$B_WLO)")
  DWHI_ACT=$("$PYTHON" -c "print((2/2**$B)*$B_WHI)")
  echo "        fast_lr [$B_FLR]  transfer_lr [$B_TLR]"
  echo "        w_max_fast band [$B_WLO, $B_WHI]  -> dw_min_fast [$DWLO_ACT, $DWHI_ACT]"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
    "$PYTHON" paper_experiment.py $COMMON \
      --n-bits "$B" --w-max-fast "$WMF" \
      --analog-lr $ANCHOR_ALR --alr-range $ANCHOR_ALR $ANCHOR_ALR \
      --classifier-lr $ANCHOR_CLR --clr-range $CLR_RANGE \
      --transfer-lr $ANCHOR_TLR --tlr-range $B_TLR \
      --transfer-every $ANCHOR_TE --te-choices $TE_CHOICES \
      --fast-lr $ANCHOR_FLR --flr-range $B_FLR \
      --wmf-range "$B_WLO" "$B_WHI" \
      --n-trials "$LEFT" --tpe-startup-trials 10 \
      --study-name "$S" --db-dir "$BASE/db" \
      --output-dir "$OUT" >> "$OUT/run.log" 2>&1 < /dev/null
  RC=$?
  echo "=== [$(date '+%F %T')] GPU$GPU DONE aux ${B}b (exit $RC)"
  (( RC == 0 )) && touch "$CLAIM/done"
done

echo "=== [$(date '+%F %T')] GPU$GPU: finished"
echo "status : bash $0 --status"
echo "settings: grep -a '\[SPACE\]\|\[ANCHOR\]\|Trains analog\|IO\[forward' $BASE/aux*/run.log"
