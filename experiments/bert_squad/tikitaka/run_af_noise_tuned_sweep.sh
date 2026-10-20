#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Tiki-Taka (ttv1) sweeps along one auxiliary-device axis with frozen
# hyperparameters (Fig. 6c, Tiki-Taka bars; Supp. Fig. S8, Tiki-Taka curves).
#
#   usage:  GPU=0 bash run_af_noise_tuned_sweep.sh <axis> <anchor>
#
#   axis   anchor
#   af     af0nz0   asymmetry scale r_AF in AF_LIST at r_noise 0
#   af     af1nz0   asymmetry scale r_AF in AF_LIST at r_noise 0
#   noise  af1nz0   noise scale r_noise in NOISE_LIST at r_AF 1
#   noise  af1nz1   noise scale r_noise in NOISE_LIST at r_AF 1
#   AF_LIST and NOISE_LIST default to 0 0.5 1 2 3 5 10.
#
# Every point is one fixed-mode run (5 epochs, --patience 0) with the
# hyperparameters of the anchor, the best trial of a TPE study:
#   af0nz0  ttv1_auxbit10b_anchor87 (r_AF 0, r_noise 0, ConstantStep A tile;
#           run_auxbit10_anchor87.sh)
#   af1nz0  ttv1_6t1c_af1_qkvo (r_AF 1, r_noise 0; run_6t1c_af1.sh)
#   af1nz1  ttv1_6t1c_full_qkvo (r_AF 1, r_noise 1; run_6t1c_full.sh)
# Their hyperparameters are listed below (A_*) and are used by default.
# transfer_every comes with the anchor: 6 for af0nz0, 1 for the two 6T1C
# anchors, whose studies do not search it.
#
# Anchor from a study database (optional)
#   With SOURCE_ROOT set, the anchor is instead the best COMPLETE trial in the
#   database of the source study under $SOURCE_ROOT (the RESULTS_ROOT of the
#   launcher that ran the study), read when the sweep starts:
#     SOURCE_ROOT=$PWD/results GPU=0 bash run_af_noise_tuned_sweep.sh af af0nz0
#
# Output: $RESULTS_ROOT/tiki_af_noise_tuned_20260809/<anchor>/af<r_AF>_nz<r_noise>/
# ('p' for the decimal point), plus <anchor>/PROVENANCE.txt, which records the
# anchor and the swept values of every launch. Both axes of one anchor write
# into the same directory, and a point that already has summary.json is
# skipped, so the shared point (r_AF 1, r_noise 0) of af1nz0 runs once.
#
# Settings
#   A (fast/aux)  LinearStep, dw_min 0.001981 (6T1C step, scaled by w_max_fast;
#                 the ConstantStep studies use the 10-bit step 0.001953125),
#                 gamma_up -0.1678 * r_AF / gamma_down 0.1410 * r_AF
#   C (slow/core) ConstantStep 10 bit, w_max 1.0
#   mult_noise    False. r_noise scales dw_min_std (0.3), dw_min_dtod (0.1),
#                 up_down_dtod (0.01), gamma_*_dtod (0.05) and w_max/w_min_dtod
#                 (0.05); write_noise_std is 0 at every ratio.
#   substrate     --target-layers all_train_attention (all 72 encoder linear
#                 layers analog with IO; the 48 QKVO layers train, the 24 FFN
#                 layers stay at lr 0), gamma 0, desired_bl 31, units_in_mbatch,
#                 scale_transfer_lr false, with_reset_prob 0
#   IO            8 bit DAC/ADC (inp_res = out_res = 1/254) and out_noise 0.04
#                 on forward, backward and transfer read, out_bound 12.0,
#                 AbsMax / Iterative (TTV1_COMMON, see sweep_common.sh)
#   schedule      5 epochs, batch 48, seed 42, warmup_ratio 0.039566,
#                 min_lr_rate 0.0, no early stopping (--patience 0)
#
# --ln-lr is passed explicitly: fixed mode defaults ln_lr to analog_lr, and
# the TPE studies tie it to classifier_lr.
#
# --analog-lr is pinned and inert for ttv1 (fast_lr is the pulse-count learning
# rate of the A tile) and stays > 0: lr 0.0 marks a frozen tile, which holds
# the FFN tiles still.
# ============================================================================
set -uo pipefail

source "$(dirname "$0")/sweep_common.sh"

AXIS="${1:-}"
ANCHOR="${2:-}"
case "$AXIS:$ANCHOR" in
  af:af0nz0|af:af1nz0|noise:af1nz0|noise:af1nz1) ;;
  *) echo "usage: bash $0 <axis> <anchor>, one of: af af0nz0 | af af1nz0 | noise af1nz0 | noise af1nz1" >&2
     exit 2 ;;
esac

BASE="$RESULTS/tiki_af_noise_tuned_20260809"
GPU="${GPU:-0}"
OUTROOT="$BASE/$ANCHOR"
DW_6T1C=0.001981

case "$ANCHOR" in
  af0nz0)
    SRC_DIR="tiki_auxbit10_anchor87_20260806"; SRC_STUDY="ttv1_auxbit10b_anchor87"
    SRC_NAME="ttv1_auxbit10b_anchor87 (r_AF 0, r_noise 0)"
    A_TRIAL=17
    A_FLR=1.7069407266871666;  A_TLR=0.017218421224856464
    A_CLR=0.003182634554466481; A_WMF=2.383427746418432; A_TE=6 ;;
  af1nz0)
    SRC_DIR="tiki_6t1c_af1_20260806"; SRC_STUDY="ttv1_6t1c_af1_qkvo"
    SRC_NAME="ttv1_6t1c_af1_qkvo (r_AF 1, r_noise 0)"
    A_TRIAL=45
    A_FLR=1.8513492339217332;  A_TLR=0.0377668494167043
    A_CLR=0.0030133866428141325; A_WMF=3.9437048493573035; A_TE=1 ;;
  af1nz1)
    SRC_DIR="tiki_6t1c_full_20260806"; SRC_STUDY="ttv1_6t1c_full_qkvo"
    SRC_NAME="ttv1_6t1c_full_qkvo (r_AF 1, r_noise 1)"
    A_TRIAL=17
    A_FLR=0.3183078689916978;  A_TLR=0.07855558108792836
    A_CLR=0.0018933047005479915; A_WMF=1.002134432824943; A_TE=1 ;;
esac

if [[ "$AXIS" == "af" ]]; then
  AF_LIST="${AF_LIST:-0 0.5 1 2 3 5 10}"; NOISE_LIST="0"
else
  AF_LIST="1"; NOISE_LIST="${NOISE_LIST:-0 0.5 1 2 3 5 10}"
fi

# fixed mode, LinearStep A tile, no early stopping; everything else is
# TTV1_COMMON (sweep_common.sh).
COMMON_T="${TTV1_COMMON/--device-type constant_step /--device-type linear_step }"
COMMON_T="${COMMON_T/--mode tpe /--mode fixed }"
COMMON_T="${COMMON_T/--patience 2 /--patience 0 }"

read_best() {   # read_best <db>  ->  flr tlr te wmf clr f1 trial_number
  "$PYTHON" - "$1" <<'PYEOF'
import sqlite3, json, sys, os
db = sys.argv[1]
if not os.path.exists(db):
    raise SystemExit(1)
c = sqlite3.connect(db)
P = {}
for tid, pn, pv, dj in c.execute(
        "select trial_id,param_name,param_value,distribution_json from trial_params"):
    d = json.loads(dj)
    P.setdefault(tid, {})[pn] = (d["attributes"]["choices"][int(round(pv))]
                                 if d["name"] == "CategoricalDistribution" else pv)
r = c.execute("""select t.trial_id,t.number,v.value from trials t
                 join trial_values v on v.trial_id=t.trial_id
                 where t.state='COMPLETE' order by v.value desc limit 1""").fetchone()
if not r:
    raise SystemExit(1)
p = P[r[0]]
need = ("fast_lr", "transfer_lr", "transfer_every", "w_max_fast", "classifier_lr")
if any(k not in p for k in need):
    raise SystemExit(1)
print(repr(p["fast_lr"]), repr(p["transfer_lr"]), int(round(p["transfer_every"])),
      repr(p["w_max_fast"]), repr(p["classifier_lr"]), round(r[2], 4), r[1])
PYEOF
}

run_one() {   # run_one <tag> <af> <noise>
  local TAG=$1 AF=$2 NZ=$3
  local OUT="$OUTROOT/$TAG"; mkdir -p "$OUT"
  if [[ -f "$OUT/summary.json" ]]; then
    echo "=== [$(date '+%F %T')] SKIP $TAG (already complete)"; return 0
  fi
  echo "=== [$(date '+%F %T')] START $TAG  AF=$AF  noise=$NZ -> $OUT"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 \
    "$PYTHON" paper_experiment.py $COMMON_T \
      --target-layers all_train_attention \
      --dw-min "$DW_6T1C" --n-bits-slow 10 --w-max 1.0 \
      --ls-gamma-up-ratio "$AF" --ls-gamma-down-ratio "$AF" \
      --ls-noise-ratio "$NZ" \
      --analog-lr "$ANCHOR_ALR" --classifier-lr "$T_CLR" --ln-lr "$T_CLR" \
      --fast-lr "$T_FLR" --transfer-lr "$T_TLR" --transfer-every "$T_TE" \
      --w-max-fast "$T_WMF" \
      --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null
  echo "=== [$(date '+%F %T')] DONE  $TAG (exit $?)"
}

cd "$PAPER"
mkdir -p "$OUTROOT"

# Anchor: the trial listed above, or with SOURCE_ROOT the best COMPLETE trial
# of the source study database.
T_FLR=$A_FLR; T_TLR=$A_TLR; T_TE=$A_TE; T_WMF=$A_WMF; T_CLR=$A_CLR
SRC_TRIAL=$A_TRIAL; SRC_F1=""
ANCHOR_FROM="hyperparameters listed in run_af_noise_tuned_sweep.sh"
if [[ -n "${SOURCE_ROOT:-}" ]]; then
  SRC_DB="$SOURCE_ROOT/$SRC_DIR/db/$SRC_STUDY.db"
  if ! BEST="$(read_best "$SRC_DB")"; then
    echo "cannot read a COMPLETE trial with the anchor hyperparameters from $SRC_DB" >&2
    exit 2
  fi
  read -r T_FLR T_TLR T_TE T_WMF T_CLR SRC_F1 SRC_TRIAL <<<"$BEST"
  ANCHOR_FROM="best COMPLETE trial of $SRC_DB"
fi
echo "=== [$(date '+%F %T')] anchor $ANCHOR: $SRC_NAME trial #$SRC_TRIAL${SRC_F1:+, F1 $SRC_F1}"
echo "    from the $ANCHOR_FROM"

{
  echo "anchor_id      : $ANCHOR"
  echo "source_study   : $SRC_NAME"
  echo "source_trial   : #$SRC_TRIAL${SRC_F1:+   F1 $SRC_F1}"
  echo "anchor_from    : $ANCHOR_FROM"
  echo "launch         : $(date '+%F %T')   axis=$AXIS"
  echo "fast_lr        : $T_FLR"
  echo "transfer_lr    : $T_TLR"
  echo "transfer_every : $T_TE"
  echo "w_max_fast     : $T_WMF"
  echo "classifier_lr  : $T_CLR   (ln_lr identical)"
  echo "analog_lr      : $ANCHOR_ALR   (pinned, inert for ttv1)"
  echo "dw_min (A)     : $DW_6T1C   (6T1C measured)"
  echo "axis=$AXIS     : AF={$AF_LIST}  noise={$NOISE_LIST}"
} >> "$OUTROOT/PROVENANCE.txt"
tail -13 "$OUTROOT/PROVENANCE.txt"

echo "=== [$(date '+%F %T')] SWEEP axis=$AXIS anchor=$ANCHOR : AF={$AF_LIST} noise={$NOISE_LIST}"
for AF in $AF_LIST; do
  for NZ in $NOISE_LIST; do
    run_one "af$(printf '%s' "$AF" | tr '.' 'p')_nz$(printf '%s' "$NZ" | tr '.' 'p')" "$AF" "$NZ"
  done
done
echo "=== [$(date '+%F %T')] ALL RUNS COMPLETE (axis=$AXIS anchor=$ANCHOR) ==="
