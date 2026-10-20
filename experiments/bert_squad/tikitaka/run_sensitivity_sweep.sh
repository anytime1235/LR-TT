#!/bin/bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# I/O-precision probe (Fig. 5d): FFN1 at 5 and 6 bit, all other sublayer groups
# at 8 bit. Each run trains for 4 epochs; the FFN1 5- and 6-bit bars of Fig. 5d
# are the F1 after the first epoch ("Epoch 1 F1" in the log, epoch-1 row of
# training_log.csv). The other bars of Fig. 5d come from the one-epoch launchers
# run_qkv_individual_sensitivity.sh, run_sensitivity_sweep_1ep.sh and
# run_sensitivity_sweep_QKVO_1ep.sh.
# Settings shared by the probe launchers (see README.md): ideal device (FP32
# weight update), all 72 encoder linear layers analog, IO 8 bit with AbsMax
# noise management and no output noise; --per-layer-bits sets the forward and
# backward DAC/ADC resolution of the probed group in all 12 layers.
# Output: $RESULTS_ROOT/io_precision_probe/sens_FFN1_<bits>b/ and a log of the
# same name (RESULTS_ROOT default: results/ in this directory).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
OUT_BASE="${RESULTS_ROOT:-$SCRIPT_DIR/results}/io_precision_probe"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"

echo "============================================"
echo "I/O-precision probe: FFN1 at 5 and 6 bit, 4 epochs"
echo "Started: $(date)"
echo "============================================"

# FFN1: 5b, 6b
for BITS_VAL in 5 6; do
  BITS="Q=8,K=8,V=8,O=8,FFN1=${BITS_VAL},FFN2=8"
  DIR="${OUT_BASE}/sens_FFN1_${BITS_VAL}b"
  echo "[$(date +%H:%M)] FFN1=${BITS_VAL}b, rest=8b -> $DIR"
  mkdir -p "$DIR"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_FFN1_${BITS_VAL}b.log"
  echo "[$(date +%H:%M)] Done: FFN1=${BITS_VAL}b"
  echo ""
done

echo "============================================"
echo "[$(date +%H:%M)] All runs completed"
echo "============================================"

# Summary: F1 after the first epoch (the value shown in Fig. 5d)
echo ""
echo "=== SUMMARY ==="
for B in 5 6; do
  LOG="${OUT_BASE}/sens_FFN1_${B}b.log"
  RESULT=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
  echo "  FFN1=${B}b: $RESULT"
done
