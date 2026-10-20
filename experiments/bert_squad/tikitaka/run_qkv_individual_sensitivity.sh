#!/bin/bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# I/O-precision probe (Fig. 5d): Q, K and V each lowered to 4, 5, 6 and 7 bit
# while all other sublayer groups stay at 8 bit; one epoch per run, 12 runs.
# Settings shared by the probe launchers (see README.md): ideal device (FP32
# weight update), all 72 encoder linear layers analog, IO 8 bit with AbsMax
# noise management and no output noise; --per-layer-bits sets the forward and
# backward DAC/ADC resolution of the probed group in all 12 layers.
# Output: $RESULTS_ROOT/io_precision_probe/sens_<group>_<bits>b_1ep/ and a log
# of the same name (RESULTS_ROOT default: results/ in this directory).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
OUT_BASE="${RESULTS_ROOT:-$SCRIPT_DIR/results}/io_precision_probe"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 1 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"

echo "============================================"
echo "Q/K/V Individual Sensitivity Sweep"
echo "  base=8b, one sublayer->{4,5,6,7}b, 1ep"
echo "  Total: 12 runs"
echo "============================================"

RUN=0
for SUBLAYER in Q K V; do
  for BIT in 4 5 6 7; do
    RUN=$((RUN + 1))

    case $SUBLAYER in
      Q) BITS="Q=${BIT},K=8,V=8,O=8,FFN1=8,FFN2=8" ;;
      K) BITS="Q=8,K=${BIT},V=8,O=8,FFN1=8,FFN2=8" ;;
      V) BITS="Q=8,K=8,V=${BIT},O=8,FFN1=8,FFN2=8" ;;
    esac

    DIR="${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep"
    echo "[$(date +%H:%M)] [${RUN}/12] ${SUBLAYER}=${BIT}b, rest=8b -> $DIR"
    mkdir -p "$DIR"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" paper_experiment.py $COMMON \
      --io-bits 8 \
      --per-layer-bits "$BITS" \
      --output-dir "$DIR" \
      2>&1 | tee "${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep.log"
    echo "[$(date +%H:%M)] Done: ${SUBLAYER}=${BIT}b"
    echo ""
  done
done

echo "============================================"
echo "[$(date +%H:%M)] All 12 runs completed"
echo "============================================"

echo ""
echo "=== SUMMARY ==="
for SUBLAYER in Q K V; do
  for BIT in 4 5 6 7; do
    LOG="${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep.log"
    F1=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
    echo "  ${SUBLAYER}=${BIT}b: $F1"
  done
done
