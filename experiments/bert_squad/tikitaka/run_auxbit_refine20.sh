#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# TPE studies with narrow search bands for auxiliary-tile bit widths 6 and 8,
# each followed by an export (Fig. 5h, 6- and 8-bit points).
#
# One study per bit width (ttv1_auxbit<BB>b_refine20), N_TRIALS (default 20)
# trials, --tpe-startup-trials 4. Trial 0 starts from the parameter set listed
# in cfg_for() for its bit width (values rounded to five significant digits;
# for 6 bit the best trial of the 6-bit study of run_auxbit_sweep.sh), and
# each study searches the bands listed there, which contain the start point;
# transfer_every is limited to {2, 3} for 6 bit and {1, 2, 3} for 8 bit.
#
# analog_lr is pinned to one value: it is inert for ttv1, where fast_lr is the
# pulse-count learning rate of the auxiliary tile.
#
# Settings as in run_auxbit_sweep.sh: all_train_attention (all 72 encoder
# linear layers analog, the 192 attention tiles train, the 288 FFN tiles stay
# frozen on analog tiles), core tile 10 bit / +-1.0, IO 8 bit / out_noise 0.04
# / out_bound 12.0 / AbsMax / Iterative, gamma 0, with_reset_prob 0,
# scale_transfer_lr false, units_in_mbatch true, desired_bl 31, warmup_ratio
# 0.039566, min_lr_rate 0.0, 5 epochs, patience 2, batch 48, seed 42, 9225
# steps.
#
# Parallel runs and export
#   As in run_auxbit_sweep.sh, the marker directories claims/bit<B> let several
#   instances run different bit widths in parallel, each on its own GPU
#   (GPU=<n> bash run_auxbit_refine20.sh). When a bit width has N_TRIALS
#   finished trials, its study and run directory are exported with
#   tools/export_sweep.py to
#       $EXPORTS/ttv1_auxbit<BB>b_refine20_<date>/   (EXPORTS_DIR, default
#                                                    $RESULTS_ROOT/exports)
#
# usage:  GPU=0 bash run_auxbit_refine20.sh        # foreground; BITS default "6 8"
#         bash run_auxbit_refine20.sh --status     # finished trials per bit width
# Output: $RESULTS_ROOT/tiki_auxbit_refine20_20260809/ (db/, aux<BB>b/, claims/)
# ============================================================================
set -uo pipefail

PYTHON="${PYTHON:-python3}"
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_ROOT:-$PAPER/results}"
EXPORTS="${EXPORTS_DIR:-$RESULTS/exports}"
BASE="$RESULTS/tiki_auxbit_refine20_20260809"
BITS="${BITS:-6 8}"
N_TRIALS="${N_TRIALS:-20}"
GPU="${GPU:-0}"

ANCHOR_ALR=0.011861648510064505   # pinned, inert for ttv1

# per-bit:  anchor_flr anchor_tlr anchor_clr anchor_wmf anchor_te te_choices
#           flr_lo flr_hi  tlr_lo tlr_hi  clr_lo clr_hi  wmf_lo wmf_hi
cfg_for() {
  case "$1" in
    6)  A_FLR=0.059064; A_TLR=0.43827;  A_CLR=0.0045935; A_WMF=0.093992; A_TE=2; TE="2,3"
        FLR="0.019101 0.16305";  TLR="0.12603 1.9975";  CLR="0.0022968 0.015979"; WMF="0.028992 0.384" ;;
    8)  A_FLR=0.4523;   A_TLR=0.30762;  A_CLR=0.00116;   A_WMF=0.081231; A_TE=2; TE="1,2,3"
        FLR="0.12324 0.90459";   TLR="0.15381 0.94701"; CLR="0.00057242 0.0048428"; WMF="0.015777 0.16246" ;;
    *)  echo "  no anchor and bands for bit width $1 (expected 6 or 8)" >&2; return 1 ;;
  esac
}

study_of() { printf 'ttv1_auxbit%02db_refine20' "$1"; }
dir_of()   { printf 'aux%02db' "$1"; }

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
  printf '%-5s %-8s %-28s %s\n' BIT STATE STUDY FINISHED
  for B in $BITS; do
    S=$(study_of "$B"); ST=free
    [[ -d "$BASE/claims/bit$B" ]] && ST=claimed
    [[ -f "$BASE/claims/bit$B/done" ]] && ST=DONE
    printf '%-5s %-8s %-28s %s/%s\n' "$B" "$ST" "$S" "$(finished_count "$BASE/db/$S.db")" "$N_TRIALS"
  done
  exit 0
fi

# ------------------------------------------------------------------- run
mkdir -p "$BASE/db" "$BASE/claims"
cd "$PAPER"
echo "=== [$(date '+%F %T')] GPU$GPU, bit widths: $BITS"

for B in $BITS; do
  cfg_for "$B" || continue
  CLAIM="$BASE/claims/bit$B"
  mkdir "$CLAIM" 2>/dev/null || { echo "  bit $B has a marker (claims/bit$B), skip"; continue; }
  echo "GPU$GPU $(date '+%F %T')" > "$CLAIM/owner"

  S=$(study_of "$B"); OUT="$BASE/$(dir_of "$B")"; mkdir -p "$OUT"
  DONE=$(finished_count "$BASE/db/$S.db"); LEFT=$(( N_TRIALS - DONE ))
  if (( LEFT > 0 )); then
    echo "=== [$(date '+%F %T')] GPU$GPU START ${B}b  ($DONE done, $LEFT to go)"
    echo "      anchor: fast_lr=$A_FLR tlr=$A_TLR clr=$A_CLR wmf=$A_WMF te=$A_TE"
    echo "      bands : flr[$FLR] tlr[$TLR] clr[$CLR] wmf[$WMF] te{$TE}"
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
      "$PYTHON" paper_experiment.py \
        --mode tpe --method ttv1 --seed 42 --target-layers all_train_attention \
        --epochs 5 --patience 2 --batch-size 48 --grad-accum-steps 1 \
        --gamma 0.0 --with-reset-prob 0 \
        --device-type constant_step --device-type-slow constant_step \
        --n-bits "$B" --n-bits-slow 10 --w-max 1.0 \
        --io-bits 8 --out-noise 0.04 --noise-management abs_max \
        --scale-transfer-lr false --units-in-mbatch true --desired-bl 31 \
        --warmup-ratio 0.039566 --min-lr-rate 0.0 \
        --analog-lr "$ANCHOR_ALR" --alr-range "$ANCHOR_ALR" "$ANCHOR_ALR" \
        --fast-lr "$A_FLR"       --flr-range $FLR \
        --transfer-lr "$A_TLR"   --tlr-range $TLR \
        --classifier-lr "$A_CLR" --clr-range $CLR \
        --w-max-fast "$A_WMF"    --wmf-range $WMF \
        --transfer-every "$A_TE" --te-choices "$TE" \
        --n-trials "$LEFT" --tpe-startup-trials 4 \
        --study-name "$S" --db-dir "$BASE/db" \
        --output-dir "$OUT" >> "$OUT/run.log" 2>&1 < /dev/null
    RC=$?
    echo "=== [$(date '+%F %T')] GPU$GPU DONE ${B}b (exit $RC)"
    (( RC == 0 )) || { echo "  ${B}b exited with $RC; claims/bit$B is kept (remove it to resume)"; continue; }
  else
    echo "  bit $B already has $DONE/$N_TRIALS -> skip to export"
  fi
  touch "$CLAIM/done"

  # ---- auto-export this bit (isolated: only this study's db + run dir) ----
  STAGE="$BASE/export_stage/$(dir_of "$B")"
  rm -rf "$STAGE"; mkdir -p "$STAGE/db"
  cp "$BASE/db/$S.db" "$STAGE/db/" 2>/dev/null
  cp -r "$OUT" "$STAGE/$(dir_of "$B")" 2>/dev/null
  DEST="$EXPORTS/ttv1_auxbit$(printf '%02d' "$B")b_refine20_$(date '+%Y%m%d_%H%M')"
  if "$PYTHON" "$PAPER/tools/export_sweep.py" "$STAGE" "$DEST" "auxbit$(printf '%02d' "$B")b_refine20"; then
    echo "=== [$(date '+%F %T')] EXPORTED ${B}b -> $DEST"
  else
    echo "  export of ${B}b failed; tools/export_sweep.py can be run on the study directory"
  fi
  rm -rf "$STAGE"
done

echo "=== [$(date '+%F %T')] GPU$GPU: finished"
echo "status : bash $0 --status"
