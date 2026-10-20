#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
# ============================================================================
# Settings shared by the Tiki-Taka (ttv1) launchers; sourced, not run.
#
# Sourced by run_allanalog_flr.sh, run_all_flr.sh, run_auxbit_sweep.sh,
# run_auxbit10_anchor87.sh, run_6t1c_af1.sh, run_6t1c_full.sh and
# run_af_noise_tuned_sweep.sh. They all start from TTV1_COMMON; a launcher
# that needs other search bands or another anchor passes its own.
#
# Learning rates of ttv1
#   The auxiliary (A) tile uses fast_lr as its pulse-count learning rate in
#   place of the optimizer learning rate whenever fast_lr > 0 (the aihwkit
#   default is 1.0), and with --scale-transfer-lr false the transfer uses the
#   constant transfer_lr. The ttv1 studies therefore search fast_lr and pin
#   analog_lr to a single value ('--alr-range X X'); analog_lr stays > 0,
#   because lr 0.0 marks a frozen tile in the optimizer. For single_rpu (one
#   tile, no TransferCompound) analog_lr is the pulse-count learning rate.
#
# Schedule
#   warmup_ratio 0.039566: linear warm-up over int(9225 * 0.039566) = 364 of
#   the 9225 training steps (5 epochs of batch 48), then linear decay to zero
#   (min_lr_rate 0.0). For ttv1 the schedule acts on the classifier and
#   LayerNorm parameters.
#
# Substrate and IO (TTV1_COMMON)
#   All 72 encoder linear layers on analog tiles (--target-layers
#   all_train_*): one group trains, the other stays on its analog tiles with
#   lr 0.0. 8-bit DAC/ADC and out_noise 0.04 on the forward, backward and
#   transfer-read paths, out_bound 12.0, AbsMax noise management and
#   Iterative bound management (aihwkit applies bound management in the
#   forward pass only and runs the transfer read without noise management and
#   bound management). ConstantStep devices without device-to-device variation.
#   5 epochs, patience 2, batch 48, seed 42.
# ============================================================================

# --- schedule --------------------------------------------------------------
WARMUP_RATIO=0.039566
MIN_LR_RATE=0.0

# --- search bands ----------------------------------------------------------
# Used by TTV1_SEARCH (run_allanalog_flr.sh: ttv1_flr_qkvo, ttv1_flr_ffn) and
# by the 6-bit study of run_auxbit_sweep.sh; run_auxbit_sweep.sh sets its own
# fast_lr and transfer_lr bands for 12 and 14 bit.
CLR_RANGE="0.001 0.008"
TLR_RANGE="0.005 0.5"
TE_CHOICES="1,2,3,6"
FLR_RANGE="0.001 10.0"

# --- anchor: trial 0 of the studies of run_allanalog_flr.sh and
# run_auxbit_sweep.sh
ANCHOR_ALR=0.011861648510064505     # pinned, inert for ttv1
ANCHOR_CLR=0.0027785544526136414
ANCHOR_TLR=0.017138245546095852
ANCHOR_TE=2
ANCHOR_FLR=0.1

# Paths are relative to this directory; the launchers cd here before calling
# paper_experiment.py. Studies are written under $RESULTS (RESULTS_ROOT, default
# ./results in this directory).
PYTHON="${PYTHON:-python3}"
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_ROOT:-$PAPER/results}"
N_TRIALS="${N_TRIALS:-50}"

# Shared by every ttv1 launcher.
TTV1_COMMON="--mode tpe --method ttv1 --seed 42 \
  --epochs 5 --patience 2 --batch-size 48 --grad-accum-steps 1 \
  --gamma 0.0 --with-reset-prob 0 \
  --device-type constant_step --device-type-slow constant_step \
  --io-bits 8 --out-noise 0.04 --noise-management abs_max \
  --scale-transfer-lr false --units-in-mbatch true \
  --desired-bl 31 --warmup-ratio $WARMUP_RATIO --min-lr-rate $MIN_LR_RATE"

# analog_lr pinned to a point + fast_lr swept.
TTV1_SEARCH="--analog-lr $ANCHOR_ALR --alr-range $ANCHOR_ALR $ANCHOR_ALR \
  --classifier-lr $ANCHOR_CLR --clr-range $CLR_RANGE \
  --transfer-lr $ANCHOR_TLR --tlr-range $TLR_RANGE \
  --transfer-every $ANCHOR_TE --te-choices $TE_CHOICES \
  --fast-lr $ANCHOR_FLR --flr-range $FLR_RANGE \
  --n-trials $N_TRIALS --tpe-startup-trials 10"
