# -*- coding: utf-8 -*-
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""BERT-base + SQuAD v1.1 fine-tuning with LR-TT (Low-Rank Tiki-Taka) analog layers.

Single-run training and diagnostic script. The Q/K/V/O projections of every encoder layer
become LR-TT analog layers (low-rank A/B tiles plus the C tile), the remaining encoder
layers are frozen analog tiles and the QA head stays digital.

The run is configured by the module constants below. Their defaults are the QKVO operating
point of the paper (rank 32, rank-wise stochastic (onehot) transfer, 8-bit DAC/ADC with
out_noise 0.04, ideal constant-step A/B/C devices). A JSON file passed with --config
overrides any of them:

    python fine_bert_squad_lrtt.py --config configs/rank_sweep_onehot/fine_r32.json
    python fine_bert_squad_lrtt.py --config configs/fig6c/gamma_r3p0.json --print-config

Outputs (best-F1 weights, diagnostic JSON and plots) go to --output-dir, by default
results/BERT_SQUAD_LRTT_FINE next to this script. See README_fine.md.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import re
import string
import math
import gc
import argparse
import collections

import json

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports (the source tree of this repository is used when the package is not installed)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401 (per-micro-batch tile.update, LRTT A/B snapshot)

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice, IdealDevice, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================
# Every UPPER_CASE name of this section can be overridden by --config FILE.json (a JSON
# object {name: value}); --print-config prints the effective values.

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths. RESULTS is the output directory (--output-dir); WEIGHT_PATH is derived from it.
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "BERT_SQUAD_LRTT_FINE")
WEIGHT_NAME = "fine_bert_squad_lrtt_model_weight.pth"  # best-F1 state dict, saved under RESULTS
WEIGHT_PATH = os.path.join(RESULTS, WEIGHT_NAME)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384

# Training
N_EPOCHS = 5
SCHEDULE_EPOCHS = 5  # LR schedule horizon in epochs (> N_EPOCHS: first N_EPOCHS of a longer schedule)
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.0011082903245468508  # QKVO operating point (rank 32) of the Optuna study (README_optuna.md)
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 5  # epochs without F1 improvement; 5 = no early stop within 5 epochs
TRAIN_LOSS_EARLY_STOP_PATIENCE = 5  # epochs without train-loss improvement; 5 = no early stop within 5 epochs
TRAIN_LOSS_THRESHOLD = 1.5  # Once train loss drops below this, rely on metric-based early stop only

# Scheduler
WARMUP_STEPS = 365
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogAdam"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters (QKVO operating point, rank 32)
LRTT_RANK = 32
TRANSFER_EVERY = 8
TRANSFER_LR = 0.0007873259241041807
FAST_LR = 0.6757723272066278
AUTO_SCALE_MODE = "none"  # Auto-scale mode: "none", "shared", or "separate"
CORRECT_GRADIENT_MAGNITUDES = False  # Correct transfer magnitude by dividing by effective A/B LR
REINIT_MODE = "decay"
REINIT_GAIN = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
C_DW_MIN = 0.001953         # C tile dw_min (10bit)
C_DESIRED_BL = 31           # C tile desired_bl (relevant for onehot/direct transfer)
AB_DW_MIN = 0.0001231800451741122  # A/B tile dw_min (abml=10: w_max = 0.0631)
AB_DESIRED_BL = 31          # A/B tile desired_bl
AB_MULTILEVEL = 10        # w_max = 2^AB_MULTILEVEL * AB_DW_MIN / 2
AB_RESET_STD = 1e-30      # reset (capacitor discharge) std of the A/B devices

# Device selection
AB_DEVICE = "constantstepideal"  # "6t1c", "linearstep", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "fp", "ideal"
A_DEVICE = None  # Optional override for the A tile device. None = use AB_DEVICE.
B_DEVICE = None  # Optional override for the B tile device. None = use AB_DEVICE.
C_DEVICE = "constantstepideal"   # "softboundsideal", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "ideal"
# Asymmetry and variation of the "linearstepideal" A/B device (Fig. 6c, Supp. Fig. S8/S9
# sweeps). All zero = ideal device. The 6T1C reference values are gamma_up=-0.1678,
# gamma_down=0.1410, dw_min_dtod=0.1, dw_min_std=0.3, up_down_dtod=0.01, w_max_dtod=0.05,
# w_min_dtod=0.05, gamma_up_dtod=0.05, gamma_down_dtod=0.05 (and reset_std=0.01, see
# AB_RESET_STD); a sweep point scales them by its asymmetry or noise ratio.
AB_LINEARSTEP_PARAMS = {
    "gamma_up": 0.0, "gamma_down": 0.0,
    "dw_min_dtod": 0.0, "dw_min_std": 0.0,
    "up_down_dtod": 0.0, "w_max_dtod": 0.0, "w_min_dtod": 0.0,
    "gamma_up_dtod": 0.0, "gamma_down_dtod": 0.0,
}

# IO / noise options
IO_NOISE = True             # If False, disable out_noise (resolution kept)
FORWARD_INJECT = False       # If True, add the auxiliary A*B product to the forward pass (W_eff = C + alpha*A*B)
FI_CONTINUOUS_ALPHA = False  # If True, use continuous alpha for forward injection
IS_PERFECT = False          # If True, forward/backward use ideal FP matmul (no ADC/DAC/noise)
NO_QUANT = False            # If True, disable DAC/ADC quantization (inp_res/out_res = -1)
DAC_BITS = 8             # DAC (inp_res) bits. None=keep aihwkit default (~7-bit); N: res=1/(2**N-2)
ADC_BITS = 8             # ADC (out_res) bits. None=keep aihwkit default (~9-bit); N: res=1/(2**N-2)
OUT_NOISE = 0.04            # out_noise of the forward and backward passes

# Per-module IO bit override (None = use the module-default DAC_BITS/ADC_BITS).
# Set ADC(out_res)/DAC(inp_res) bits per module type, e.g. give FFN higher precision
# than attention. The bits are applied when the model is converted, one conversion pass per
# bit group (the tiles take their IO settings at creation). Example FFN 10-bit ADC: set
# 'intermediate'/'output.dense' adc=10. With all entries None the model is converted in a
# single pass.
LAYER_IO_BITS = {
    'query':            {'dac': None, 'adc': None},
    'key':              {'dac': None, 'adc': None},
    'value':            {'dac': None, 'adc': None},
    'attention.output': {'dac': None, 'adc': None},   # attention output (O) projection
    'intermediate':     {'dac': None, 'adc': None},   # FFN1 (pre-GELU)
    'output.dense':     {'dac': None, 'adc': None},   # FFN2 (pre-LayerNorm)
}
AB_WEIGHT_SCALING_OMEGA = 0.0  # A/B tile weight scaling omega

# Pulse type
AB_PULSE_TYPE = "default"  # "default", "none", "none_with_device", "stochastic_compressed", "mean_count", "deterministic_implicit"

# Transfer options
NO_TRANSFER = False         # If True, disable transfer (set transfer_every to infinity)
NO_SCALE_TRANSFER_LR = True  # If True, constant transfer_lr (scale_transfer_lr=False: not multiplied by the optimizer LR)
TRANSFER_RANK_SCHEDULE = "all"  # "all" or "round_robin"
TRANSFER_RANKS_PER_STEP = 1

# 6T1C Retention parameters
TAU_SEC = 0.0  # 0 = no decay, >0 = retention time constant

# Dynamic TE (transfer every) parameters
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# LoRA target options: which layers have trainable A/B tiles (LORA_TARGET_MODULES below)
# - none: no LRTT layers
# - qkv: only query, key, value
# - qkvo: query, key, value and the attention output projection (default)
# - ffn: FFN layers (intermediate.dense, output.dense)
# - all: all encoder linear layers
AB_IO_PERFECT = False  # If True, A/B tiles fully ideal (no out_noise/ADC/DAC)
LEARN_OUT_SCALING = False  # If True, C tile out_scaling is trainable
LORA_TARGET = "qkvo"  # default
HEAD_LAYER = "train"  # "train" or "freeze" for qa_outputs layer
ENCODER_ANALOG = True  # If True, non-LRTT encoder layers become frozen analog instead of digital
HEAD_ANALOG = False  # If True, qa_outputs becomes frozen analog instead of digital
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT
    "qonly": ["query"],  # Query only (12 layers)
    "konly": ["key"],  # Key only (12 layers)
    "vonly": ["value"],  # Value only (12 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (36 layers)
    "qkvo": ["query", "key", "value", "attention.output"],  # Q/K/V + attention output (48 layers)
    "ffn": (["intermediate", "output.dense"], ["attention"]),  # FFN only (24 layers)
    "dense": ["dense"],  # All layers with "dense" (excludes qkv) (36 layers)
    "allnobn": None,  # Same as all for BERT (no bottleneck) (72 layers)
    "all": None,  # None means all encoder layers (no filtering) (72 layers)
}

# Diagnostic
ENABLE_DIAGNOSTIC = True   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 5            # 0 = all epochs, N = first N epochs only
DIAG_STAMP = None          # tag of the diagnostic output files; None = "te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"
DIAG_FULL_C = False        # also record the effective rank of the assembled (unsplit) C tile at transfer steps

# Per-layer tile selection. Reuses LORA_TARGET_MODULES vocabulary.
#
# Shorthand string:
#   "first_last"  -> first & last LRTT-converted layer x LORA_TARGET modules (default)
#   "all"         -> every LRTT-converted layer x LORA_TARGET modules
#
# Explicit dict {layer_idx: module_selector}, module_selector being:
#   "match"             - inherit LORA_TARGET (= this layer's LRTT-converted modules)
#   key from LORA_TARGET_MODULES - "qkvo"|"qkv"|"qonly"|"konly"|"vonly"|"ffn"|"dense"|"all"
#   list of module names - e.g. ["query", "key"]
#
# Examples:
#   {0: "qkvo", 11: "qkvo"}                       # layer 0,11 x {q,k,v,o}
#   {0: "match", 6: "match", 11: "match"}         # layers 0,6,11 (each = LORA_TARGET)
#   {0: "qonly", 11: "qkvo"}                      # heterogeneous per layer
#   {0: ["query","key"], 11: "match"}             # explicit module names
#
# Intersection rule: actual monitored set = LRTT-converted AND selected layers AND selected modules.
# Empty intersection: warning + skip.
DIAG_TILES = "first_last"

# Rate limits (steps; <=0 means every step)
ERANK_RATE_LIMIT_STEPS = 0    # erank/SVD cost gate
HIST_RATE_STEPS = 1844        # weight + signal histogram cadence (= 1 epoch @ effective bs=48)

# Per-group on/off. Each tracked tile records only the enabled groups.
DIAG_GROUPS = {
    "g1_norms":        True,   # ||A||, ||B||, ||AB||, ||C_eff||
    "g2_minmax":       True,   # A/B/C_eff signed min/max
    "g3_mean":         True,   # signed mean(A/B/C_eff)
    "g3b_mean_abs":    True,   # mean(|A|), mean(|B|), mean(|C_eff|)
    "g3c_weight_hist": False,  # hist(A/B/C_eff) every HIST_RATE_STEPS; costly, default OFF
    "g4_deltas":       True,   # ||delta_A||, ||delta_B||, ||delta_C_eff||, ||delta_AB||
    "g5a_erank_ab":    True,   # erank A, B, AB; sigma1_AB
    "g5b_erank_c":     True,   # erank C_eff, C_delta; sigma1_C_*
    "g6a_cells":       False,  # individual cell values (default OFF)
    "g6b_cell_deltas": False,  # individual cell deltas (default OFF)
    "g7_cosines":      True,   # transfer-event cosines + norm_G_accum etc.
    "g8_signal_abs":   True,   # xa/xb/da/db abs mean+max
    "g10_signal_hist": False,  # xa/xb/da/db hist every HIST_RATE_STEPS; default OFF
    "g11a_xc_dc_abs":  True,   # xc/dc abs mean+max (transfer events)
    "g11c_xc_dc_hist": False,  # xc/dc hist every HIST_RATE_STEPS; default OFF
    "g11d_xfer_meta":  True,   # transfer_lr_c, transfer_n_calls
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

GRAD_ACCUM_STEPS = 1  # micro-batches per optimizer step (BATCH_SIZE is the effective batch size)

# WandB
WANDB_PROJECT = "bert-squad-lrtt-fine"
os.environ["WANDB_MODE"] = "offline"

# Names of the constants above (the --config keys); the derived ones cannot be overridden.
CONSTANT_NAMES = [k for k in globals() if k.isupper() and not k.startswith("_")]
DERIVED_CONSTANTS = ("USE_CUDA", "DEVICE", "WEIGHT_PATH")


# =============================================================================
# Command line
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="BERT-base + SQuAD v1.1 fine-tuning with LR-TT analog layers.")
    parser.add_argument("--config", metavar="FILE.json",
                        help="JSON object {constant: value} overriding the module constants")
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective constants as JSON and exit")
    parser.add_argument("--output-dir", metavar="DIR",
                        help=f"directory for the weights, diagnostic JSON and plots (default: {RESULTS})")
    return parser.parse_args(argv)


def apply_overrides(args):
    """Apply --config and --output-dir to the module constants and recompute the derived ones."""
    global RESULTS, WEIGHT_PATH
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        bad = sorted(k for k in overrides if k not in CONSTANT_NAMES or k in DERIVED_CONSTANTS)
        if bad:
            raise KeyError(f"{args.config}: not overridable: {', '.join(bad)}")
        globals().update(overrides)
    if args.output_dir:
        RESULTS = args.output_dir
    WEIGHT_PATH = os.path.join(RESULTS, WEIGHT_NAME)


def effective_config():
    """The current values of all module constants, in definition order."""
    return {k: globals()[k] for k in CONSTANT_NAMES}



# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=None, dw_min=None, multilevel=None, device_name=None,
                      reset_std=0.01):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c              - LinearStepDevice with the 6T1C parameters: asymmetry (gamma_up,
                            gamma_down), device-to-device and pulse-to-pulse variation,
                            write_noise_std 0
        linearstep        - LinearStepDevice with default params (no nonlinearity, default noise)
        linearstepideal   - LinearStepDevice with asymmetry/variation from AB_LINEARSTEP_PARAMS (zero = ideal)
        constantstep      - ConstantStepDevice with default params (constant step, default noise)
        constantstepideal - ConstantStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep6t1cgamma - LinearStepDevice with all noise/dtod=0 but gamma_up/gamma_down from 6t1c
        fp                - FloatingPointDevice (perfect, no quantization/bounds)
        ideal             - IdealDevice

    If multilevel (or AB_MULTILEVEL) is set to an int > 0, the branches 6t1c,
    linearstepideal, constantstepideal and constantstep6t1cgamma use
    w_max = 2^multilevel * dw_min / 2 and w_min = -w_max instead of +/-1.0
    (linearstep and constantstep keep the bounds of the device class).

    reset_std: std of the reset (capacitor-discharge) operation; 0.01 is the reset
    noise of the 6T1C device model. Applied to all pulsed-device branches.
    """
    if tau_sec is None:
        tau_sec = TAU_SEC
    if dw_min is None:
        dw_min = AB_DW_MIN

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    if multilevel is None:
        multilevel = AB_MULTILEVEL
    if multilevel is not None and multilevel > 0:
        w_max = (2 ** multilevel) * dw_min / 2.0
    else:
        w_max = 1.0
    w_min = -w_max

    name = device_name if device_name is not None else AB_DEVICE
    if name == "fp":
        return FloatingPointDevice()
    if name == "ideal":
        return IdealDevice()
    if name == "linearstep":
        return LinearStepDevice(
            dw_min=dw_min, lifetime=lifetime,
            reset_std=reset_std, reset_dtod=0.0,
        )
    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            write_noise_std=0.0, reset_std=reset_std, reset_dtod=0.0,
            up_down=0.0, mult_noise=False,
            lifetime=lifetime,
            **AB_LINEARSTEP_PARAMS,
        )
    if name == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min, lifetime=lifetime,
            reset_std=reset_std, reset_dtod=0.0,
        )
    if name == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            reset_std=reset_std, reset_dtod=0.0, up_down=0.0,
            lifetime=lifetime,
        )
    if name == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )

    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0, w_max=w_max, w_min=w_min,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_std=reset_std, reset_dtod=0.0,
    )


def _create_c_device(dw_min=None, reset_std=0.0):
    """Create device for C tile based on C_DEVICE setting.

    reset_std: std of the reset operation. Default 0.0 (deterministic). Applied to
    all pulsed-device branches; ignored for ideal/floating-point.
    """
    if dw_min is None:
        dw_min = C_DW_MIN

    if C_DEVICE == "ideal":
        return IdealDevice()
    if C_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=reset_std, reset_dtod=0.0,
            up_down=0.0, mult_noise=False,
        )
    if C_DEVICE == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min,
            reset_std=reset_std, reset_dtod=0.0,
        )
    if C_DEVICE == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            reset_std=reset_std, reset_dtod=0.0, up_down=0.0,
        )
    if C_DEVICE == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
        )
    # Default: softboundsideal
    return SoftBoundsDevice(
        dw_min=dw_min,
        w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, reset_std=reset_std, reset_dtod=0.0,
        mult_noise=False,
    )


def _bits_to_res(bits):
    """Convert a converter bit-count to aihwkit resolution (1/steps).

    None / <2: no override (keep current inp_res/out_res: aihwkit default or NO_QUANT).
    N>=2:      1/(2**N - 2)  (N-bit signed converter).
    """
    if bits is None or bits < 2:
        return None
    return 1.0 / (2 ** bits - 2)


def _apply_quant_bits(rpu_config, dac_bits, adc_bits):
    """Override forward/backward DAC (inp_res) and ADC (out_res) from bit-counts.

    Applied in BOTH create_lrtt_config and the create_frozen_analog_config standalone
    branch so every analog path (LRTT layers AND none-mode frozen layers) gets identical
    quantization. The derived frozen branch inherits via deepcopy of forward/backward.
    Has no effect under is_perfect (aihwkit ignores all IOParameters then).
    """
    dac_res = _bits_to_res(dac_bits)
    if dac_res is not None:
        rpu_config.forward.inp_res = dac_res
        rpu_config.backward.inp_res = dac_res
    adc_res = _bits_to_res(adc_bits)
    if adc_res is not None:
        rpu_config.forward.out_res = adc_res
        rpu_config.backward.out_res = adc_res
    return rpu_config


def _module_io_key(name):
    """Map an encoder Linear module name to its LAYER_IO_BITS key (None if unmatched)."""
    if 'attention.self.query' in name:
        return 'query'
    if 'attention.self.key' in name:
        return 'key'
    if 'attention.self.value' in name:
        return 'value'
    if 'attention.output.dense' in name:
        return 'attention.output'
    if 'intermediate.dense' in name:
        return 'intermediate'
    if name.endswith('output.dense'):  # FFN2 (attention.output already handled above)
        return 'output.dense'
    return None


def _resolve_io_bits(name, default_dac, default_adc):
    """Per-module (dac, adc) bits from LAYER_IO_BITS, falling back to defaults when None."""
    b = LAYER_IO_BITS.get(_module_io_key(name) or '', {})
    dac, adc = b.get('dac'), b.get('adc')
    return (dac if dac is not None else default_dac,
            adc if adc is not None else default_adc)


def _any_layer_io_override():
    """True if any LAYER_IO_BITS entry sets a non-default (non-None) dac/adc bit-count."""
    return any((b.get('dac') is not None or b.get('adc') is not None)
               for b in LAYER_IO_BITS.values())


def create_frozen_analog_config(lrtt_config=None, out_noise=0.0, dac_bits=None, adc_bits=None):
    """Create analog config for non-LRTT encoder layers (frozen analog).

    If lrtt_config is provided, derived from its C tile settings.
    Otherwise, creates a standalone config with default C tile settings.
    """
    from copy import deepcopy
    if lrtt_config is not None:
        rpu_config = SingleRPUConfig(
            device=deepcopy(lrtt_config.device.unit_cell_devices[2]),
        )
        rpu_config.mapping = deepcopy(lrtt_config.device.mapping_c)
        rpu_config.forward = deepcopy(lrtt_config.forward)
        rpu_config.backward = deepcopy(lrtt_config.backward)
        # Apply (possibly FFN-specific) bits, overriding the inherited LRTT quantization.
        # No-op when dac_bits/adc_bits are None (default).
        _apply_quant_bits(rpu_config, dac_bits, adc_bits)
    else:
        rpu_config = SingleRPUConfig(device=_create_c_device())
        rpu_config.mapping = MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=LEARN_OUT_SCALING,
            out_scaling_columnwise=True,
        )
        rpu_config.forward.out_noise = out_noise
        rpu_config.backward.out_noise = out_noise
        rpu_config.forward.is_perfect = IS_PERFECT
        rpu_config.backward.is_perfect = IS_PERFECT
        if NO_QUANT:
            rpu_config.forward.inp_res = -1
            rpu_config.forward.out_res = -1
            rpu_config.backward.inp_res = -1
            rpu_config.backward.out_res = -1
        _apply_quant_bits(rpu_config, DAC_BITS, ADC_BITS)
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    # A and B tile devices: independently overridable via A_DEVICE / B_DEVICE.
    # When both are None, both A and B use AB_DEVICE.
    a_device = _create_ab_device(device_name=A_DEVICE, reset_std=AB_RESET_STD)
    b_device = _create_ab_device(device_name=B_DEVICE, reset_std=AB_RESET_STD)
    c_device = _create_c_device()

    # With multilevel A/B devices the B initialization scales with their w_max.
    if AB_MULTILEVEL is not None and AB_MULTILEVEL > 0:
        ab_w_max = (2 ** AB_MULTILEVEL) * AB_DW_MIN / 2.0
        b_reinit_gain = REINIT_GAIN * ab_w_max
    else:
        b_reinit_gain = REINIT_GAIN

    te = TRANSFER_EVERY if not NO_TRANSFER else 10 ** 9
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=1.0,
        fast_lr=(FAST_LR if not NO_TRANSFER else 1e-30),
        reinit_gain=b_reinit_gain,
        reinit_mode=REINIT_MODE,
        unit_cell_devices=[a_device, b_device, c_device],
        train_c_bias=False,
        mapping_ab=MappingParameter(
            weight_scaling_omega=AB_WEIGHT_SCALING_OMEGA,
            learn_out_scaling=False,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=LEARN_OUT_SCALING,
            out_scaling_columnwise=True,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = FORWARD_INJECT
    device_config.fi_continuous_alpha = FI_CONTINUOUS_ALPHA
    device_config.ab_io_perfect = AB_IO_PERFECT
    device_config.c_desired_bl = C_DESIRED_BL
    device_config.auto_scale_mode = AUTO_SCALE_MODE
    device_config.correct_gradient_magnitudes = CORRECT_GRADIENT_MAGNITUDES
    device_config.scale_transfer_lr = not NO_SCALE_TRANSFER_LR
    device_config.ab_pulse_type = AB_PULSE_TYPE
    device_config.transfer_rank_schedule = TRANSFER_RANK_SCHEDULE
    device_config.transfer_ranks_per_step = TRANSFER_RANKS_PER_STEP

    # Dynamic TE: increase TE as LR decays
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.max_input_size = 0 if IS_PERFECT else 512
    rpu_config.mapping.max_output_size = 0 if IS_PERFECT else 512
    rpu_config.update.desired_bl = AB_DESIRED_BL

    out_noise = OUT_NOISE if IO_NOISE else 0.0
    rpu_config.forward.out_noise = out_noise
    rpu_config.backward.out_noise = out_noise
    rpu_config.forward.is_perfect = IS_PERFECT
    rpu_config.backward.is_perfect = IS_PERFECT
    if NO_QUANT:
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.backward.inp_res = -1
        rpu_config.backward.out_res = -1
    _apply_quant_bits(rpu_config, DAC_BITS, ADC_BITS)

    if BACKWARD_OUT_BOUND != 12.0:
        rpu_config.backward.out_bound = BACKWARD_OUT_BOUND

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    """List all linear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_analog_layers(model):
    """Count analog layers in the model."""
    from aihwkit.nn import AnalogLinear
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def get_lrtt_target_module_names(lora_target):
    """Get module name patterns for LRTT conversion based on lora_target.

    Returns list of substrings that identify which encoder layers should be LRTT.
    Returns [] for none (no LRTT layers).
    """
    if lora_target == "none":
        return []  # Empty = no layers converted to LRTT
    elif lora_target == "qonly":
        return ["query"]  # Query only (12 layers)
    elif lora_target == "konly":
        return ["key"]  # Key only (12 layers)
    elif lora_target == "vonly":
        return ["value"]  # Value only (12 layers)
    elif lora_target == "qkv":
        return ["query", "key", "value"]  # Q/K/V (36 layers)
    elif lora_target == "qkvo":
        return ["query", "key", "value", "attention.output"]  # Q/K/V + attention output (48 layers)
    elif lora_target == "ffn":
        return (["intermediate", "output.dense"], ["attention"])  # FFN only (24 layers)
    elif lora_target == "dense":
        return ["dense"]  # All layers with "dense" in name (excludes qkv) (36 layers)
    elif lora_target == "allnobn":
        return None  # Same as all for BERT (no bottleneck) (72 layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, qa_outputs)
        return None  # None means all encoder layers (72 layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model():
    """Create BERT QA model with selective LRTT analog layers.

    Architecture:
        - LRTT target layers (LORA_TARGET) -> LRTT analog
        - Non-target encoder layers -> FROZEN (analog with ENCODER_ANALOG, else digital)
        - qa_outputs -> digital, TRAINABLE (weight + bias) with HEAD_LAYER == "train"
        - LayerNorm -> digital, TRAINABLE
        - Embeddings -> digital, FROZEN

    LoRA target options (LORA_TARGET):
        - qkv: Q/K/V layers -> LRTT analog (36 layers)
        - qkvo: Q/K/V and attention output layers -> LRTT analog (48 layers)
        - ffn: FFN layers -> LRTT analog (24 layers)
        - all: all encoder layers -> LRTT analog (72 layers)

    LRTT layers have:
        - A/B tiles: TRAINABLE (pulsed updates)
        - C tile: pretrained weights, changed only by transfers
        - out_scaling: TRAINABLE with LEARN_OUT_SCALING (off by default)
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # qa_outputs is always digital
        if "qa_outputs" in layer_name:
            return False
        # Must be in encoder for other layers
        if "encoder" not in layer_name:
            return False
        # If lrtt_patterns is None (all mode), all encoder layers are targets
        if lrtt_patterns is None:
            return True
        if isinstance(lrtt_patterns, tuple):
            include, exclude = lrtt_patterns
            included = True if include is None else any(p in layer_name for p in include)
            return included and not any(p in layer_name for p in exclude)
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude qa_outputs (always digital)
    exclude_modules.append("qa_outputs")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # none: no LRTT conversion (frozen analog layers may still follow in Step 1.5)
        num_analog = 0
    else:
        lrtt_config = create_lrtt_config()

        # Convert to analog with exclusions (only LRTT targets get converted).
        # Per-module IO bits: with LAYER_IO_BITS overrides, convert each bit-group in its
        # own pass (build-time bits; a post-conversion override does not propagate to tiles).
        if not _any_layer_io_override():
            model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)
        else:
            import copy as _copy
            _groups = {}
            for _n in all_linear_names:
                if is_lrtt_target(_n):
                    _key = _resolve_io_bits(_n, DAC_BITS, ADC_BITS)
                    _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = _copy.deepcopy(lrtt_config)
                _apply_quant_bits(_cfg, _gd, _ga)
                _excl = [_n for _n in all_linear_names if _n not in _names]
                model = convert_to_analog(model, _cfg, exclude_modules=_excl, inplace=True)  # inplace: model may already hold LRTT tiles (copy breaks ids + CUDA placement)

        # Count analog layers
        num_analog = count_analog_layers(model)

    # Step 1.5: Convert remaining encoder layers to frozen analog (if enabled)
    frozen_analog_count = 0
    any_frozen_analog = (ENCODER_ANALOG and LORA_TARGET not in ("all", "allnobn")) or HEAD_ANALOG
    if any_frozen_analog:
        # Collect existing tile IDs (LRTT sub-tiles) before frozen conversion
        existing_tile_ids = set()
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    existing_tile_ids.add(id(tile))

        frozen_exclude = []
        if not HEAD_ANALOG:
            frozen_exclude.append("qa_outputs")
        if not ENCODER_ANALOG or LORA_TARGET in ("all", "allnobn"):
            for name in all_linear_names:
                if "encoder" in name:
                    frozen_exclude.append(name)
        _lc = lrtt_config if LORA_TARGET != "none" else None
        if not _any_layer_io_override():
            frozen_config = create_frozen_analog_config(_lc)
            # inplace=True keeps the LRTT tiles (and the ids collected above) as they are;
            # the default inplace=False deep-copies the model, which replaces the tile
            # objects and rebuilds the LRTT inner tiles on the CPU.
            model = convert_to_analog(model, frozen_config, exclude_modules=frozen_exclude,
                                      inplace=True)
        else:
            # Per-module IO bits for frozen-analog (FFN) layers: group by resolved bits.
            _ftargets = [_n for _n in all_linear_names
                         if _n not in frozen_exclude and not is_lrtt_target(_n)]
            _groups = {}
            for _n in _ftargets:
                _key = _resolve_io_bits(_n, DAC_BITS, ADC_BITS)
                _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = create_frozen_analog_config(_lc, dac_bits=_gd, adc_bits=_ga)
                _excl = [_n for _n in all_linear_names if _n not in _names]
                model = convert_to_analog(model, _cfg, exclude_modules=_excl, inplace=True)  # inplace: model may already hold LRTT tiles (copy breaks ids + CUDA placement)
        frozen_analog_count = count_analog_layers(model) - num_analog

        # Hook frozen analog tile updates to no-op (prevent optimizer from modifying weights).
        # AnalogSGD/Adam calls tile.update() on ALL analog tiles unconditionally;
        # LRTT tiles are already hooked in lrtt_tile.py, but frozen analog tiles need this.
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None

        # Bypass AnalogFunction.apply() for frozen tiles to avoid activation/gradient
        # saving overhead. AnalogFunction saves input 3x (autograd ctx, analog_input list,
        # analog_grad_output list) for weight updates that never happen on frozen tiles.
        # This custom Function matches LRTT C tile behavior (direct tile call).
        import types
        class _FrozenAnalogFwd(torch.autograd.Function):
            @staticmethod
            @no_grad()
            def forward(ctx, analog_tile, x_input, is_test):
                ctx.analog_tile = analog_tile
                ctx.saved_analog_tensors = [x_input]
                out = analog_tile.joint_forward(x_input, is_test, ctx)
                ctx.save_for_backward(*ctx.saved_analog_tensors)
                ctx.saved_analog_tensors = []
                return out
            @staticmethod
            @no_grad()
            def backward(ctx, grad_output):
                ctx.saved_analog_tensors = list(ctx.saved_tensors)
                grad_input = ctx.analog_tile.backward(grad_output, ctx)
                ctx.saved_analog_tensors = []
                return None, grad_input, None

        def _frozen_analog_forward(self, x_input, tensor_view=None):
            out = _FrozenAnalogFwd.apply(self, x_input, not self.training)
            if tensor_view is None:
                tensor_view = self.get_tensor_view(out.dim())
            out = self.apply_out_scaling(out, tensor_view)
            if self.digital_bias:
                return out + self.bias.view(*tensor_view)
            return out

        for mod_name, m in model.named_modules():
            if isinstance(m, AnalogLinear):
                # LRTT layers are recognized by module name as well as by tile id, so they
                # are never frozen, even if a conversion pass replaced the tile objects.
                if is_lrtt_target(mod_name):
                    continue
                for tile in m.analog_tiles():
                    if id(tile) not in existing_tile_ids:
                        # Head analog tiles remain trainable (weight + bias)
                        if HEAD_ANALOG and "qa_outputs" in mod_name:
                            continue
                        tile.update = _frozen_noop_update
                        tile.forward = types.MethodType(_frozen_analog_forward, tile)

        # Check at build time that no LRTT tile received the frozen no-op update.
        _locked_lrtt = 0
        for _mn, _mm in model.named_modules():
            if isinstance(_mm, AnalogLinear) and is_lrtt_target(_mn):
                for _t in _mm.analog_tiles():
                    if getattr(_t.update, "__name__", "") == "_frozen_noop_update":
                        _locked_lrtt += 1
        assert _locked_lrtt == 0, (
            f"{_locked_lrtt} LRTT tiles were frozen-hooked (update no-op). "
            "This disables LRTT learning entirely; check convert_to_analog inplace/copy."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 2: Set requires_grad
    # - LRTT layers: A/B TRAINABLE; C tile bias and out scaling as set by lrtt_tile.py
    #   (train_c_bias, mapping_c.learn_out_scaling)
    # - qa_outputs: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - LayerNorm: TRAINABLE
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = not NO_TRANSFER
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "out_scaling_alpha" in name:
            pass  # Frozen analog out scaling: a trainable parameter only with LEARN_OUT_SCALING
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params = count_parameters(model)

    print(f"\nCreated BERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, fast_lr={FAST_LR}, auto_scale={AUTO_SCALE_MODE}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    try:
        return model.to(DEVICE)
    except Exception:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

    # Use full dataset if EVAL_SUBSET_SIZE == 0, otherwise subset
    if EVAL_SUBSET_SIZE > 0:
        eval_examples = raw_datasets["validation"].select(
            range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"])))
        )
    else:
        eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]

        inputs["example_id"] = [
            examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))
        ]

        return inputs

    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    # Use full dataset if TRAIN_SUBSET_SIZE == 0, otherwise subset
    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        collate_fn=collator, num_workers=2,
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
    """Post-process SQuAD predictions. Extracts best answer spans."""
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                    })

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]

    return all_predictions


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    """Evaluate SQuAD model using official metric. Returns (F1, EM)."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

    # Pad to max_length so all batches produce same-sized logits for np.concatenate
    collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH)

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(
        eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn,
        num_workers=2
    )

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Diagnostic Helpers
# =============================================================================

def _make_cell_indices(shape, n=10):
    """Generate n evenly-spaced cell indices for a weight matrix of given shape."""
    rows, cols = shape
    indices = []
    for i in range(n):
        r = min(int(i * rows / n), rows - 1)
        c = min(int(i * cols / n), cols - 1)
        indices.append((r, c))
    return indices


def _expand_lora_target(key):
    """Resolve LORA_TARGET_MODULES[key] to a list of module-name substrings.
    None matches all modules. Tuple form (include, exclude) returns the include list."""
    if key not in LORA_TARGET_MODULES:
        raise ValueError(f"Unknown LORA target key: {key!r}")
    val = LORA_TARGET_MODULES[key]
    if val is None:
        return None
    if isinstance(val, tuple):
        return list(val[0])
    return list(val)


def _short_tile_name(full_name):
    """Convert e.g. 'bert.encoder.layer.0.attention.self.query.analog_module' to 'L0.attention.self.query'."""
    import re
    m = re.search(r"layer\.(\d+)\.(.+?)(?:\.analog_module)?$", full_name)
    return f"L{m.group(1)}.{m.group(2)}" if m else full_name


def resolve_diag_tiles(model, diag_tiles_cfg, lora_target_key):
    """Resolve DIAG_TILES config to ordered dict {short_name: (full_name, module)}.

    diag_tiles_cfg :
        "first_last"           - first & last LRTT-converted tile (default)
        "all"                  - every LRTT-converted tile
        {layer_idx: selector}  - per-layer selector, where selector is:
            "match"            - inherit LORA_TARGET (this layer's LRTT modules)
            key from LORA_TARGET_MODULES - "qkvo"|"qkv"|"qonly"|"konly"|"vonly"|"ffn"|"dense"|"all"
            list of module name substrings - e.g. ["query", "key"]

    Returns ordered dict; empty set: warning printed, returns {}.
    """
    all_lrtt = [(n, m) for n, m in model.named_modules() if hasattr(m, 'controller')]
    if not all_lrtt:
        raise RuntimeError("No LRTT tile found in model")

    if diag_tiles_cfg == "first_last":
        if len(all_lrtt) == 1:
            return {"only": all_lrtt[0]}
        return {"first": all_lrtt[0], "last": all_lrtt[-1]}
    if diag_tiles_cfg == "all":
        return {_short_tile_name(n): (n, m) for n, m in all_lrtt}

    if not isinstance(diag_tiles_cfg, dict):
        raise ValueError(
            f"DIAG_TILES must be 'first_last' | 'all' | dict, got {diag_tiles_cfg!r}")

    out = {}
    for layer_idx, mod_sel in diag_tiles_cfg.items():
        if mod_sel == "match":
            pats = _expand_lora_target(lora_target_key)
        elif isinstance(mod_sel, str):
            pats = _expand_lora_target(mod_sel)
        elif isinstance(mod_sel, (list, tuple)):
            pats = list(mod_sel)
        else:
            raise ValueError(
                f"Invalid module selector for layer {layer_idx}: {mod_sel!r}")
        layer_str = f"layer.{layer_idx}."
        for n, m in all_lrtt:
            if layer_str in n and (pats is None or any(p in n for p in pats)):
                out[_short_tile_name(n)] = (n, m)
    if not out:
        print(f"WARNING: DIAG_TILES resolved to empty set "
              f"(LORA_TARGET={lora_target_key!r}, DIAG_TILES={diag_tiles_cfg!r})")
    return out


def sample_cells(weight_matrix, cell_indices):
    values = []
    for r, c in cell_indices:
        if r < weight_matrix.shape[0] and c < weight_matrix.shape[1]:
            values.append(weight_matrix[r, c].item())
        else:
            values.append(0.0)
    return values


def snapshot_weights(tile):
    """Pre-step snapshot of (A, B, C_eff). C_raw is not tracked; all deltas
    use C_eff (the scale of the transfer)."""
    return (
        tile.tile_a.get_weights()[0].clone().detach(),
        tile.tile_b.get_weights()[0].clone().detach(),
        tile.tile_c.get_weights()[0].clone().detach(),
    )


def collect_tile_diagnostics(tile, A_before, B_before, C_eff_before, step,
                             prev_num_transfers, A_ci, B_ci, C_ci,
                             A_pre_transfer=None, C_initial_eff=None,
                             compute_erank=True, compute_weight_hist=False,
                             groups=None):
    """Thin orchestrator: dispatches to per-group helpers based on `groups` flags."""
    if groups is None:
        groups = DIAG_GROUPS
    controller = tile.controller
    A = A_pre_transfer if A_pre_transfer is not None else tile.tile_a.get_weights()[0]
    B = tile.tile_b.get_weights()[0]
    C_eff = tile.tile_c.get_weights()[0]
    AB = A @ B

    num_transfers = controller.num_transfers
    record = {
        "step": step,
        "transfer_counter": controller.transfer_counter,
        "num_transfers": num_transfers,
        "is_transfer": num_transfers > prev_num_transfers,
    }
    if groups.get("g1_norms"):
        record.update(_diag_g1_norms(A, B, AB, C_eff))
    if groups.get("g2_minmax"):
        record.update(_diag_g2_minmax(A, B, C_eff))
    if groups.get("g3_mean"):
        record.update(_diag_g3_mean_signed(A, B, C_eff))
    if groups.get("g3b_mean_abs"):
        record.update(_diag_g3b_mean_abs(A, B, C_eff))
    if groups.get("g3c_weight_hist") and compute_weight_hist:
        record.update(_diag_g3c_weight_hist(A, B, C_eff))
    if groups.get("g4_deltas"):
        AB_before = (A_before @ B_before) if (A_before is not None and B_before is not None) else None
        record.update(_diag_g4_deltas(A, B, AB, C_eff,
                                      A_before, B_before, AB_before, C_eff_before))
    if compute_erank:
        if groups.get("g5a_erank_ab"):
            record.update(_diag_g5a_erank_ab(A, B, AB))
        if groups.get("g5b_erank_c"):
            record.update(_diag_g5b_erank_c(C_eff, C_initial_eff))
    if groups.get("g6a_cells"):
        record.update(_diag_g6a_cells(A, B, C_eff, A_ci, B_ci, C_ci))
    if groups.get("g6b_cell_deltas"):
        record.update(_diag_g6b_cell_deltas(A, B, C_eff,
                                            A_before, B_before, C_eff_before,
                                            A_ci, B_ci, C_ci))
    return record, num_transfers


def _svd_stats(M):
    """Return (effective_rank, sigma_1). sigma_1 is the largest singular value
    of M (unfiltered; 0.0 if M is empty)."""
    s = torch.linalg.svdvals(M.float().cuda())
    if len(s) == 0:
        return 0.0, 0.0
    sigma1 = s[0].item()
    s_filt = s[s > 1e-10]
    if len(s_filt) == 0:
        return 0.0, sigma1
    p = s_filt / s_filt.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.exp().item(), sigma1


# -----------------------------------------------------------------------------
# Per-group diagnostic helpers
# -----------------------------------------------------------------------------

def _diag_g1_norms(A, B, AB, C_eff):
    return {
        "norm_A":     torch.norm(A).item(),
        "norm_B":     torch.norm(B).item(),
        "norm_AB":    torch.norm(AB).item(),
        "norm_C_eff": torch.norm(C_eff).item(),
    }


def _diag_g2_minmax(A, B, C_eff):
    return {
        "A_min": A.min().item(),         "A_max": A.max().item(),
        "B_min": B.min().item(),         "B_max": B.max().item(),
        "C_eff_min": C_eff.min().item(), "C_eff_max": C_eff.max().item(),
    }


def _diag_g3_mean_signed(A, B, C_eff):
    return {
        "mean_A":     A.mean().item(),
        "mean_B":     B.mean().item(),
        "mean_C_eff": C_eff.mean().item(),
    }


def _diag_g3b_mean_abs(A, B, C_eff):
    return {
        "mean_abs_A":     A.abs().mean().item(),
        "mean_abs_B":     B.abs().mean().item(),
        "mean_abs_C_eff": C_eff.abs().mean().item(),
    }


def _weight_hist_one(t, bins=50):
    """Histogram of a weight tensor over its actual [min, max] range."""
    flat = t.flatten().float()
    lo = flat.min().item()
    hi = flat.max().item()
    if hi > lo:
        counts = torch.histc(flat, bins=bins, min=lo, max=hi).tolist()
        return {"counts": counts, "min": lo, "max": hi}
    return {"counts": [float(flat.numel())] + [0.0] * (bins - 1),
            "min": lo, "max": lo + 1.0}


def _diag_g3c_weight_hist(A, B, C_eff, bins=50):
    """Caller must rate-limit before invoking."""
    return {
        "hist_A":     _weight_hist_one(A, bins=bins),
        "hist_B":     _weight_hist_one(B, bins=bins),
        "hist_C_eff": _weight_hist_one(C_eff, bins=bins),
    }


def _diag_g4_deltas(A, B, AB, C_eff,
                    A_before, B_before, AB_before, C_eff_before):
    return {
        "delta_A":     torch.norm(A - A_before).item()         if A_before is not None     else 0.0,
        "delta_B":     torch.norm(B - B_before).item()         if B_before is not None     else 0.0,
        "delta_AB":    torch.norm(AB - AB_before).item()       if AB_before is not None    else 0.0,
        "delta_C_eff": torch.norm(C_eff - C_eff_before).item() if C_eff_before is not None else 0.0,
    }


def _diag_g5a_erank_ab(A, B, AB):
    er_A, _      = _svd_stats(A)
    er_B, _      = _svd_stats(B)
    er_AB, sg_AB = _svd_stats(AB)
    return {
        "erank_A": er_A, "erank_B": er_B,
        "erank_AB": er_AB, "sigma1_AB": sg_AB,
    }


def _assemble_full_C(arr_mod):
    """Concatenate the C sub-tiles of a split AnalogLinear into the full layer weight.

    With IS_PERFECT=False the 768x768 layers are mapped onto 2x2 sub-tiles of 384x384
    (max_input/output_size=512; array[i][j] = input block i, output block j), so a single
    sub-tile caps the effective rank at 384. The assembled matrix gives the effective rank
    of the whole layer (DIAG_FULL_C).
    """
    n_in = len(arr_mod.array)
    n_out = len(arr_mod.array[0])
    rows = []
    for j in range(n_out):
        cols = [arr_mod.array[i][j].tile_c.get_weights()[0] for i in range(n_in)]
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows, dim=0)


def _diag_g5b_erank_c(C_eff, C_initial_eff):
    er_C,  sg_C  = _svd_stats(C_eff)
    if C_initial_eff is not None:
        er_dC, sg_dC = _svd_stats(C_eff - C_initial_eff)
    else:
        er_dC, sg_dC = 0.0, 0.0
    return {
        "erank_C_eff":   er_C,  "sigma1_C_eff":   sg_C,
        "erank_C_delta": er_dC, "sigma1_C_delta": sg_dC,
    }


def _diag_g6a_cells(A, B, C_eff, A_ci, B_ci, C_ci):
    return {
        "A_cells":     sample_cells(A,     A_ci),
        "B_cells":     sample_cells(B,     B_ci),
        "C_eff_cells": sample_cells(C_eff, C_ci),
    }


def _diag_g6b_cell_deltas(A, B, C_eff,
                          A_before, B_before, C_eff_before,
                          A_ci, B_ci, C_ci):
    out = {}
    if A_before is not None:
        out["A_cell_deltas"]     = sample_cells(A - A_before,         A_ci)
    if B_before is not None:
        out["B_cell_deltas"]     = sample_cells(B - B_before,         B_ci)
    if C_eff_before is not None:
        out["C_eff_cell_deltas"] = sample_cells(C_eff - C_eff_before, C_ci)
    return out


def _cos_sim(a, b):
    na, nb = torch.norm(a).item(), torch.norm(b).item()
    if na > 1e-10 and nb > 1e-10:
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
    return 0.0


def make_diagnostic_plots(log_data, output_path, tile_label="",
                          A_ci=None, B_ci=None, C_ci=None):
    """Create 6x2 diagnostic plot for one tile (C quantities as C_eff)."""
    steps = [r["step"] for r in log_data]
    norm_A = [r.get("norm_A", 0) for r in log_data]
    norm_B = [r.get("norm_B", 0) for r in log_data]
    norm_C_eff = [r.get("norm_C_eff", 0) for r in log_data]
    norm_AB = [r.get("norm_AB", 0) for r in log_data]
    losses = [r.get("loss", 0.0) for r in log_data]
    transfer_steps = [r["step"] for r in log_data if r["is_transfer"]]

    has_cells = "A_cells" in log_data[0]
    if has_cells:
        n_cells = len(log_data[0]["A_cells"])
        A_w = [[r["A_cells"][i] for r in log_data] for i in range(n_cells)]
        B_w = [[r["B_cells"][i] for r in log_data] for i in range(n_cells)]
        C_w = [[r["C_eff_cells"][i] for r in log_data] for i in range(len(log_data[0]["C_eff_cells"]))]
        A_g = [[r.get("A_cell_deltas", [0]*n_cells)[i] for r in log_data] for i in range(n_cells)]
        B_g = [[r.get("B_cell_deltas", [0]*n_cells)[i] for r in log_data] for i in range(n_cells)]
        n_c = len(log_data[0]["C_eff_cells"])
        C_g = [[r.get("C_eff_cell_deltas", [0]*n_c)[i] for r in log_data] for i in range(n_c)]
        a_ci = A_ci or [(i, 0) for i in range(n_cells)]
        b_ci = B_ci or [(0, i) for i in range(n_cells)]
        c_ci = C_ci or [(i, i) for i in range(n_c)]

    fig, axes = plt.subplots(6, 2, figsize=(18, 34))
    fig.suptitle(f"LRTT Diagnostic - {tile_label}" if tile_label else "LRTT Diagnostic",
                 fontsize=14, y=1.01)

    def tl(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    A_mins = [r.get("A_min", r.get("A_eff_min", 0)) for r in log_data]
    A_maxs = [r.get("A_max", r.get("A_eff_max", 0)) for r in log_data]
    B_mins = [r.get("B_min", r.get("B_eff_min", 0)) for r in log_data]
    B_maxs = [r.get("B_max", r.get("B_eff_max", 0)) for r in log_data]
    C_eff_mins = [r.get("C_eff_min", 0) for r in log_data]
    C_eff_maxs = [r.get("C_eff_max", 0) for r in log_data]

    # (0,0) A, B norms + min/max
    ax = axes[0, 0]
    ax.plot(steps, norm_A, label="||A||", alpha=0.8)
    ax.plot(steps, norm_B, label="||B||", alpha=0.8)
    ax.plot(steps, norm_AB, label="||A@B||", alpha=0.6, linestyle="--")
    ax_mm = ax.twinx()
    ax_mm.plot(steps, A_maxs, label="A max", color="red", alpha=0.5, linewidth=0.7, linestyle=":")
    ax_mm.plot(steps, A_mins, label="A min", color="red", alpha=0.5, linewidth=0.7, linestyle="--")
    ax_mm.plot(steps, B_maxs, label="B max", color="blue", alpha=0.5, linewidth=0.7, linestyle=":")
    ax_mm.plot(steps, B_mins, label="B min", color="blue", alpha=0.5, linewidth=0.7, linestyle="--")
    ax_mm.set_ylabel("min/max", fontsize=8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("A, B, AB Norms + min/max (red = transfer)")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax_mm.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (0,1) ||C_eff|| + delta_C_eff
    ax = axes[0, 1]
    ax.plot(steps, norm_C_eff, label="||C_eff||", color="green", alpha=0.8)
    delta_C = [r.get("delta_C_eff", 0) for r in log_data]
    ax2 = ax.twinx()
    ax2.plot(steps, delta_C, label="delta_C_eff", color="orange", alpha=0.8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("||C_eff||", color="green")
    ax2.set_ylabel("delta_C_eff", color="orange")
    ax.set_title("C Norm (eff) + delta_C_eff")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="upper left", fontsize=6); ax.grid(True, alpha=0.3)

    # (1,0) C_eff min/max
    ax = axes[1, 0]
    ax.plot(steps, C_eff_maxs, label="C_eff max", color="purple", alpha=0.8, linewidth=1.0)
    ax.plot(steps, C_eff_mins, label="C_eff min", color="purple", alpha=0.8, linewidth=1.0, linestyle="--")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(y=-1.0, color="gray", linestyle=":", alpha=0.4)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
    ax.set_title("C_eff weight min/max")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (1,1) Effective rank + sigma_1 (twin axis)
    er_key = "erank_C_eff" if any(r.get("erank_C_eff") is not None for r in log_data) else "erank_C"
    er_steps = [r["step"] for r in log_data if r.get(er_key) is not None]
    erank_C = [r[er_key] for r in log_data if r.get(er_key) is not None]
    erd_steps = [r["step"] for r in log_data if r.get("erank_C_delta") is not None]
    erank_C_delta = [r["erank_C_delta"] for r in log_data if r.get("erank_C_delta") is not None]
    ax = axes[1, 1]
    if er_steps:
        ax.plot(er_steps, erank_C, label="erank(C_eff)", color="green", alpha=0.8, linewidth=1.0, marker='.', markersize=3)
    if erd_steps:
        ax.plot(erd_steps, erank_C_delta, label="erank(C - C_init)", color="blue", alpha=0.8, linewidth=1.0, marker='.', markersize=3)
    # sigma_1 overlay (twin axis); collapse signal
    sg1_steps = [r["step"] for r in log_data if r.get("sigma1_C_eff") is not None]
    sg1_C   = [r["sigma1_C_eff"]   for r in log_data if r.get("sigma1_C_eff") is not None]
    sg1d_steps = [r["step"] for r in log_data if r.get("sigma1_C_delta") is not None]
    sg1_dC  = [r["sigma1_C_delta"] for r in log_data if r.get("sigma1_C_delta") is not None]
    if sg1_steps or sg1d_steps:
        axs = ax.twinx()
        if sg1_steps:
            axs.plot(sg1_steps, sg1_C, label="sigma_1(C_eff)", color="orange", alpha=0.6, linewidth=0.8, linestyle="--", marker='.', markersize=2)
        if sg1d_steps:
            axs.plot(sg1d_steps, sg1_dC, label="sigma_1(C - C_init)", color="red", alpha=0.6, linewidth=0.8, linestyle="--", marker='.', markersize=2)
        axs.set_ylabel("sigma_1", fontsize=8)
        l1, la1 = ax.get_legend_handles_labels(); l2, la2 = axs.get_legend_handles_labels()
        ax.legend(l1+l2, la1+la2, fontsize=6, loc="upper right")
    else:
        ax.legend(fontsize=7)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank of C_eff + sigma_1 (twin axis)")
    ax.grid(True, alpha=0.3)

    # (2,0)-(4,1) cell weights/deltas (skipped if g6a_cells off)
    if has_cells:
        for row, (ws, gs, ci, nm) in enumerate(
                [(A_w, A_g, a_ci, "A"), (B_w, B_g, b_ci, "B"), (C_w, C_g, c_ci, "C_eff")], start=2):
            ax = axes[row, 0]
            for i, s in enumerate(ws):
                r, c = ci[i]; ax.plot(steps, s, label=f"{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
            tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Weight")
            ax.set_title(f"{nm} cells: weights"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
            ax = axes[row, 1]
            for i, s in enumerate(gs):
                r, c = ci[i]; ax.plot(steps, s, label=f"d{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
            tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Delta")
            ax.set_title(f"{nm} cells: delta"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
    else:
        for row in range(2, 5):
            for col in range(2):
                axes[row, col].text(0.5, 0.5, "g6a_cells disabled", transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=10, color="gray")
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    # (5,0) G_accum norm (line) + tlr*AB and dC norms at transfers (scatter) + loss
    nG = [max(r.get("norm_G_accum", 1e-10), 1e-10) for r in log_data]
    t_steps_dC = [r["step"] for r in log_data if r["is_transfer"]]
    t_norms_dC = [max(r.get("norm_dC_step", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    t_norms_tlr = [max(r.get("norm_tlrAB", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    ax = axes[5, 0]
    ax.semilogy(steps, nG, label="||G_accum||", color="red", alpha=0.8, linewidth=0.8)
    if t_steps_dC:
        ax.semilogy(t_steps_dC, t_norms_tlr, '^', label="||tlr*A@B|| @T", color="green",
                     markersize=5, alpha=0.9, zorder=5)
        ax.semilogy(t_steps_dC, t_norms_dC, 'o', label="||delta_C|| @T", color="blue",
                     markersize=5, alpha=0.9, zorder=5)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)")
    axl = ax.twinx(); axl.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl.set_ylabel("Loss", color="gray")
    lm, llm = ax.get_legend_handles_labels(); ll, lll = axl.get_legend_handles_labels()
    ax.legend(lm+ll, llm+lll, fontsize=7, loc="upper right")
    ax.set_title("||G_accum|| vs ||tlr*A@B|| + ||delta_C|| at transfers + Loss"); ax.grid(True, alpha=0.3)

    # (5,1) cosines at transfer steps only (G_accum is reset per transfer)
    t_cTG = [r.get("cos_tlrAB_G", 0) for r in log_data if r["is_transfer"]]
    t_cDG = [r.get("cos_dC_G", 0) for r in log_data if r["is_transfer"]]
    t_cDT = [r.get("cos_dC_tlrAB", 0) for r in log_data if r["is_transfer"]]
    ax = axes[5, 1]
    if t_steps_dC:
        ax.scatter(t_steps_dC, t_cTG, label="cos(tlr*AB, G) @T", color="green",
                   s=25, alpha=0.9, zorder=5, marker="^")
        ax.scatter(t_steps_dC, t_cDG, label="cos(dC, G) @T", color="blue",
                   s=25, alpha=0.9, zorder=5, marker="o")
        ax.scatter(t_steps_dC, t_cDT, label="cos(dC, tlr*AB) @T", color="purple",
                   s=25, alpha=0.9, zorder=5, marker="s")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    tl(ax); ax.set_ylabel("Cosine Similarity"); ax.set_ylim(-1.1, 1.1)
    axl2 = ax.twinx(); axl2.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl2.set_ylabel("Loss", color="gray")
    lc, llc = ax.get_legend_handles_labels(); ll2, lll2 = axl2.get_legend_handles_labels()
    ax.legend(lc+ll2, llc+lll2, fontsize=6, loc="lower left")
    ax.set_xlabel("Step"); ax.set_title("Cosines: dC vs G, tlr*AB vs G, dC vs tlr*AB + Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def make_xd_diagnostic_plots(log_data, output_path, tile_label=""):
    """Create x/d distribution diagnostic plots: abs mean/max + histograms."""
    if not log_data:
        return
    steps = [r['step'] for r in log_data]

    xd_keys = [
        ('xa', 'tile_a input (XB = x B^T)'),
        ('da', 'tile_a grad (raw gradient)'),
        ('xb', 'tile_b input (raw x)'),
        ('db', 'tile_b grad (DA = A^T d)'),
        ('xc', 'tile_c transfer input (B weights)'),
        ('dc', 'tile_c transfer grad (A weights)'),
    ]

    fig, axes = plt.subplots(6, 2, figsize=(18, 27))
    fig.suptitle(f'x/d Distribution - {tile_label}', fontsize=13, y=0.99)

    for row, (prefix, desc) in enumerate(xd_keys):
        is_transfer_key = prefix in ('xc', 'dc')

        # For xc/dc, filter to transfer steps only
        if is_transfer_key:
            plot_data = [r for r in log_data if r.get('is_transfer') and f'{prefix}_abs_max' in r]
            plot_steps = [r['step'] for r in plot_data]
        else:
            plot_data = log_data
            plot_steps = steps

        # --- Left: abs mean/max over time ---
        ax = axes[row, 0]
        if plot_data:
            mean_vals = [r.get(f'{prefix}_abs_mean', 0) for r in plot_data]
            max_vals = [r.get(f'{prefix}_abs_max', 0) for r in plot_data]
            ax.plot(plot_steps, max_vals, 'r-', linewidth=0.8, label='abs max', alpha=0.8)
            ax.plot(plot_steps, mean_vals, 'g-', linewidth=0.8, label='abs mean', alpha=0.9)
        ax.set_title(f'|{prefix}| abs mean/max - {desc}')
        ax.set_ylabel(f'|{prefix}|')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        if row == len(xd_keys) - 1:
            ax.set_xlabel('Step')

        # --- Right: histograms at sampled time points ---
        ax = axes[row, 1]
        # xc/dc histograms are in 'xc_dc_hist', xa/da/xb/db in 'xd_hist'
        hist_key = 'xc_dc_hist' if is_transfer_key else 'xd_hist'
        hist_steps = [r for r in plot_data if hist_key in r and prefix in r[hist_key]]
        if hist_steps:
            n_hist = len(hist_steps)
            sample_idx = [0, n_hist // 3, 2 * n_hist // 3, n_hist - 1]
            sample_idx = sorted(set(min(i, n_hist - 1) for i in sample_idx))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sample_idx)))
            for ci, idx in enumerate(sample_idx):
                h = hist_steps[idx][hist_key][prefix]
                counts = h['counts']
                bin_edges = np.linspace(h['min'], h['max'], len(counts) + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                total = sum(counts)
                if total > 0:
                    normed = [c / total for c in counts]
                    ax.plot(bin_centers, normed, color=colors[ci], linewidth=1.0,
                            label=f'step {hist_steps[idx]["step"]}', alpha=0.8)
            ax.set_title(f'|{prefix}| distribution')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No histogram data', transform=ax.transAxes, ha='center')
        if row == len(xd_keys) - 1:
            ax.set_xlabel(f'|{prefix}|')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def make_weight_dynamics_plots(log_data, output_path, tile_label=""):
    """Weight distribution evolution: mean signed, mean abs, deltas, erank, sigma_1, loss."""
    if not log_data:
        return
    steps = [r["step"] for r in log_data]
    transfer_steps = [r["step"] for r in log_data if r["is_transfer"]]
    losses = [r.get("loss", 0.0) for r in log_data]

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"Weight Dynamics - {tile_label}" if tile_label else "Weight Dynamics",
                 fontsize=14, y=1.0)

    def tl(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    def _line(ax, key, label, color, log_y=False):
        vals = [r.get(key) for r in log_data]
        if not any(v is not None and v != 0 for v in vals):
            return False
        xs = [s for s, v in zip(steps, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if log_y:
            ax.semilogy(xs, ys, label=label, color=color, alpha=0.85, linewidth=0.9)
        else:
            ax.plot(xs, ys, label=label, color=color, alpha=0.85, linewidth=0.9)
        return True

    # (0,0) Signed mean
    ax = axes[0, 0]
    _line(ax, "mean_A", "mean(A)", "C0")
    _line(ax, "mean_B", "mean(B)", "C1")
    _line(ax, "mean_C_eff", "mean(C_eff)", "C2")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Signed mean")
    ax.set_title("Signed mean of weights"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (0,1) Mean absolute value
    ax = axes[0, 1]
    _line(ax, "mean_abs_A", "mean(|A|)", "C0")
    _line(ax, "mean_abs_B", "mean(|B|)", "C1")
    _line(ax, "mean_abs_C_eff", "mean(|C_eff|)", "C2")
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("mean(|w|)")
    ax.set_title("Mean absolute value of weights"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,0) Delta norms: A, B, AB (log)
    ax = axes[1, 0]
    _line(ax, "delta_A",  "||delta_A||",  "C0", log_y=True)
    _line(ax, "delta_B",  "||delta_B||",  "C1", log_y=True)
    _line(ax, "delta_AB", "||delta_AB||", "C3", log_y=True)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Delta norm (log)")
    ax.set_title("Per-step delta norms (A, B, AB)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,1) Delta C_eff (separate panel; transfer-driven, different scale)
    ax = axes[1, 1]
    _line(ax, "delta_C_eff", "||delta_C_eff||", "C2")
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("||delta_C_eff||")
    ax.set_title("delta_C_eff (transfer-induced)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,0) Erank: A, B, AB
    ax = axes[2, 0]
    for key, lbl, color in [("erank_A", "erank(A)", "C0"),
                            ("erank_B", "erank(B)", "C1"),
                            ("erank_AB", "erank(AB)", "C3")]:
        recs = [(r["step"], r.get(key)) for r in log_data if r.get(key) is not None]
        if recs:
            xs, ys = zip(*recs)
            ax.plot(xs, ys, label=lbl, color=color, alpha=0.85,
                    linewidth=0.9, marker=".", markersize=3)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank of A, B, AB"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,1) sigma_1: AB, C_eff, C_delta
    ax = axes[2, 1]
    for key, lbl, color in [("sigma1_AB",      "sigma_1(AB)",      "C3"),
                            ("sigma1_C_eff",   "sigma_1(C_eff)",   "C2"),
                            ("sigma1_C_delta", "sigma_1(C - C_init)", "C4")]:
        recs = [(r["step"], r.get(key)) for r in log_data if r.get(key) is not None]
        if recs:
            xs, ys = zip(*recs)
            ax.plot(xs, ys, label=lbl, color=color, alpha=0.85,
                    linewidth=0.9, marker=".", markersize=3)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("sigma_1")
    ax.set_title("Largest singular values"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (3,0) Loss
    ax = axes[3, 0]
    ax.plot(steps, losses, label="loss", color="gray", alpha=0.9)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Training loss"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (3,1) reserved
    axes[3, 1].text(0.5, 0.5, "(reserved)", transform=axes[3, 1].transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
    axes[3, 1].set_xticks([]); axes[3, 1].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def make_weight_hist_plots(log_data, output_path, tile_label=""):
    """Weight histogram evolution overlay (hist_A/B/C_eff). Skips when g3c not enabled."""
    if not log_data:
        return
    hist_recs = [r for r in log_data
                 if all(k in r for k in ("hist_A", "hist_B", "hist_C_eff"))]
    if not hist_recs:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f"Weight Histogram Evolution - {tile_label}" if tile_label
                 else "Weight Histogram Evolution", fontsize=14, y=0.995)

    n_hist = len(hist_recs)
    sample_idx = sorted(set(min(i, n_hist - 1) for i in
                            [0, n_hist // 4, n_hist // 2, 3 * n_hist // 4, n_hist - 1]))
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(sample_idx)))

    for ax, key, label in [(axes[0], "hist_A", "A"),
                           (axes[1], "hist_B", "B"),
                           (axes[2], "hist_C_eff", "C_eff")]:
        for ci, idx in enumerate(sample_idx):
            h = hist_recs[idx][key]
            counts = h["counts"]
            edges = np.linspace(h["min"], h["max"], len(counts) + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            total = sum(counts)
            if total > 0:
                normed = [c / total for c in counts]
                ax.plot(centers, normed, color=colors[ci], linewidth=1.0,
                        label=f"step {hist_recs[idx]['step']}", alpha=0.85)
        ax.set_xlabel(f"{label} weight value")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} weight distribution over time ({n_hist} hist samples)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create optimizer. Uses Analog optimizers when model has analog tiles (LRTT or frozen analog)."""
    if LORA_TARGET == "none" and not ENCODER_ANALOG and not HEAD_ANALOG:
        # None mode (no analog tiles): use standard PyTorch optimizers
        if OPTIMIZER == "AnalogSGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    else:
        # Analog optimizers: required for LRTT tiles and frozen analog tiles
        # (AnalogSGD/Adam calls analog_ctx.reset() to prevent memory leak)
        if OPTIMIZER == "AnalogSGD":
            optimizer = AnalogSGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = AnalogAdam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        optimizer.regroup_param_groups()
        optimizer._grad_accum_steps = GRAD_ACCUM_STEPS

    return optimizer


def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main
# =============================================================================

def main():
    """Train BERT with LRTT on SQuAD."""
    os.makedirs(RESULTS, exist_ok=True)
    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"bert_lrtt_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": "SQuAD",
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "fast_lr": FAST_LR, "auto_scale_mode": AUTO_SCALE_MODE,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "lora_target": LORA_TARGET,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    schedule_ep = SCHEDULE_EPOCHS if SCHEDULE_EPOCHS > 0 else N_EPOCHS
    num_training_steps = len(train_loader) * schedule_ep // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # =========================================================================
    # Diagnostic setup (skipped if ENABLE_DIAGNOSTIC=False)
    # =========================================================================
    diag_state = {}
    A_CI = B_CI = C_CI = []
    A_shape = B_shape = C_shape = ()

    if ENABLE_DIAGNOSTIC:
        # Resolve cell-indices from any one selected tile's shape (all share shape family)
        _resolved = resolve_diag_tiles(model, DIAG_TILES, LORA_TARGET)
        if _resolved:
            _ref_tile = next(iter(_resolved.values()))[1]
            A_shape = tuple(_ref_tile.tile_a.get_weights()[0].shape)
            B_shape = tuple(_ref_tile.tile_b.get_weights()[0].shape)
            C_shape = tuple(_ref_tile.tile_c.get_weights()[0].shape)
            A_CI = _make_cell_indices(A_shape)
            B_CI = _make_cell_indices(B_shape)
            C_CI = _make_cell_indices(C_shape)

        for _sn, (_fn, _mod) in _resolved.items():
            _mod.controller.enable_diagnostics = True
            diag_state[_sn] = {
                "tile": _mod, "full_name": _fn,
                "gc": {}, "log": [],
                "prev_nt": 0, "last_erank_step": -10**9,
                "whist_count": 0,
                "C_initial_eff": _mod.tile_c.get_weights()[0].clone().detach(),
                "A_CI": A_CI, "B_CI": B_CI, "C_CI": C_CI,
            }
            if DIAG_FULL_C:
                _arr = model.get_submodule(_fn.rsplit('.array.', 1)[0]) if '.array.' in _fn else None
                diag_state[_sn]["arr"] = _arr
                diag_state[_sn]["C_full_init"] = (
                    _assemble_full_C(_arr).clone().detach() if _arr is not None else None)
                if _arr is not None:
                    print(f"  [full-C] {_sn}: array {len(_arr.array)}x{len(_arr.array[0])} "
                          f"-> assembled {tuple(diag_state[_sn]['C_full_init'].shape)}")
        print(f"\nDiag tiles (DIAG_TILES={DIAG_TILES!r}): {len(diag_state)} tile(s); shapes A{A_shape} B{B_shape} C{C_shape}")
        for _sn in diag_state:
            print(f"  - {_sn}: {diag_state[_sn]['full_name']}")
        print(f"Diag epochs: {'all' if DIAG_EPOCHS == 0 else f'first {DIAG_EPOCHS}'}")

        def _install_hook(diag_tile, device, gc_dict):
            d_size, x_size = diag_tile.tile_c.get_weights()[0].shape
            gc_dict['G_accum'] = torch.zeros(d_size, x_size, device=device)
            gc_dict['active'] = True
            ctrl = diag_tile.controller

            def _abs_stats(t):
                """Return (mean, max) of |t| as floats."""
                a = t.abs()
                return a.mean().item(), a.max().item()

            def _capture_common(x_b, d_a, x_a, d_b):
                """Accumulate G and record AB + x/d stats after both tiles update.

                Args:
                    x_b: tile_b's x input (raw batch input)
                    d_a: tile_a's d input (raw gradient, alpha already removed in FI)
                    x_a: tile_a's x input (XB = B x in FI; same as x_b projected in NFI)
                    d_b: tile_b's d input (DA = A^T d)
                """
                with torch.no_grad():
                    x_2d = x_b.reshape(-1, x_b.shape[-1])
                    d_2d = d_a.reshape(-1, d_a.shape[-1])
                    # G_accum += d^T x over the rows of this update (x_b and d_a
                    # belong to the same micro-batch; min() keeps the product
                    # defined for inputs with different row counts)
                    min_batch = min(d_2d.shape[0], x_2d.shape[0])
                    gc_dict['G_accum'] = gc_dict['G_accum'] + d_2d[:min_batch].t() @ x_2d[:min_batch]
                    A = diag_tile.tile_a.get_weights()[0].to(device)
                    B = diag_tile.tile_b.get_weights()[0].to(device)
                    AB = A @ B
                    gc_dict['AB_matrix'] = AB.clone()
                    gc_dict['norm_AB_pre'] = torch.norm(AB).item()
                    gc_dict['norm_G_accum'] = torch.norm(gc_dict['G_accum']).item()
                    AB_flat = AB.flatten(); G_flat = gc_dict['G_accum'].flatten()
                    gc_dict['cos_AB_G'] = (torch.nn.functional.cosine_similarity(
                        AB_flat.unsqueeze(0), G_flat.unsqueeze(0)).item()
                        if gc_dict['norm_AB_pre'] > 1e-10 and gc_dict['norm_G_accum'] > 1e-10 else 0.0)
                    # Temporal coherence of G: cosine to the G_accum of the previous step
                    _G_norm = gc_dict['norm_G_accum']
                    if gc_dict.get('G_prev_flat') is not None and _G_norm > 1e-10:
                        _Gp = gc_dict['G_prev_flat']
                        _Gp_norm = gc_dict.get('G_prev_norm', 0.0)
                        if _Gp_norm > 1e-10:
                            gc_dict['cos_G_prev'] = (G_flat @ _Gp).item() / (_G_norm * _Gp_norm)
                        else:
                            gc_dict['cos_G_prev'] = 0.0
                    else:
                        gc_dict['cos_G_prev'] = 0.0
                    gc_dict['G_prev_flat'] = G_flat.detach().clone()
                    gc_dict['G_prev_norm'] = _G_norm
                    # sigma_1(G) and dominance ratio sigma_1 / ||G||_F (top-mode participation)
                    if _G_norm > 1e-10:
                        _svals = torch.linalg.svdvals(gc_dict['G_accum'].float())
                        _s1 = _svals[0].item()
                        gc_dict['sigma1_G'] = _s1
                        gc_dict['sigma1_G_ratio'] = _s1 / _G_norm
                    else:
                        gc_dict['sigma1_G'] = 0.0
                        gc_dict['sigma1_G_ratio'] = 0.0
                    # tile_a sees (XB, d_raw); tile_b sees (x_raw, DA)
                    gc_dict['xa_abs_mean'], gc_dict['xa_abs_max'] = _abs_stats(x_a.to(device))
                    gc_dict['da_abs_mean'], gc_dict['da_abs_max'] = _abs_stats(d_a.to(device))
                    gc_dict['xb_abs_mean'], gc_dict['xb_abs_max'] = _abs_stats(x_b.to(device))
                    gc_dict['db_abs_mean'], gc_dict['db_abs_max'] = _abs_stats(d_b.to(device))
                    # Signal histogram (rate-limited by HIST_RATE_STEPS, gated by g10)
                    gc_dict['_capture_count'] = gc_dict.get('_capture_count', 0) + 1
                    _do_hist = (DIAG_GROUPS.get('g10_signal_hist', False)
                                and HIST_RATE_STEPS > 0
                                and (gc_dict['_capture_count'] - 1) % HIST_RATE_STEPS == 0)
                    if _do_hist:
                        _hists = {}
                        for _prefix, _t in [('xa', x_a), ('da', d_a), ('xb', x_b), ('db', d_b)]:
                            _flat = _t.to(device).abs().flatten().float()
                            _max_val = _flat.max().item()
                            if _max_val > 0:
                                _counts = torch.histc(_flat, bins=50, min=0, max=_max_val).tolist()
                                _hists[_prefix] = {'counts': _counts, 'min': 0.0, 'max': _max_val}
                            else:
                                _hists[_prefix] = {'counts': [float(_flat.numel())] + [0.0]*49, 'min': 0.0, 'max': 1.0}
                        gc_dict['_last_hist'] = _hists
                        gc_dict['_hist_ready'] = True
                    else:
                        gc_dict['_hist_ready'] = False

            # Transfer diagnostic: capture x/d going into C tile
            # - direct/onehot: hook tile_c._orig_update / tile_c.update
            # - set mode: tile_c.tile.update is C-ext (read-only), so we
            #   read A/B weights before transfer (= the x/d that set passes)
            gc_dict['_transfer_xc_all'] = []
            gc_dict['_transfer_dc_all'] = []
            gc_dict['_transfer_lr_c'] = 0.0
            gc_dict['_in_transfer'] = False

            _orig_transfer = ctrl.ab_weight_transfer

            # Hook outer tile_c update (used by direct/onehot)
            _orig_outer_update = getattr(diag_tile.tile_c, '_orig_update',
                                         diag_tile.tile_c.update)

            def _capture_c_update_outer(x_input, d_input, *args, **kwargs):
                if gc_dict.get('_in_transfer'):
                    gc_dict['_transfer_xc_all'].append(x_input.detach().clone())
                    gc_dict['_transfer_dc_all'].append(d_input.detach().clone())
                    gc_dict['_transfer_lr_c'] = diag_tile.tile_c.get_learning_rate()
                return _orig_outer_update(x_input, d_input, *args, **kwargs)

            def _compute_transfer_stats(gc_dict):
                """Compute xc/dc stats from accumulated tensors."""
                with torch.no_grad():
                    xc_list = gc_dict['_transfer_xc_all']
                    dc_list = gc_dict['_transfer_dc_all']
                    if xc_list:
                        xc_cat = torch.cat(xc_list, dim=0).to(device)
                        dc_cat = torch.cat(dc_list, dim=0).to(device)
                        gc_dict['xc_abs_mean'], gc_dict['xc_abs_max'] = _abs_stats(xc_cat)
                        gc_dict['dc_abs_mean'], gc_dict['dc_abs_max'] = _abs_stats(dc_cat)
                        # Transfer histogram (rate-limited by HIST_RATE_STEPS in transfer-call units, gated by g11c)
                        gc_dict['_transfer_hist_count'] = gc_dict.get('_transfer_hist_count', 0) + 1
                        _do_xfer_hist = (DIAG_GROUPS.get('g11c_xc_dc_hist', False)
                                         and HIST_RATE_STEPS > 0
                                         and (gc_dict['_transfer_hist_count'] - 1) % HIST_RATE_STEPS == 0)
                        if _do_xfer_hist:
                            _hists_c = {}
                            for _prefix, _t in [('xc', xc_cat), ('dc', dc_cat)]:
                                _flat = _t.abs().flatten().float()
                                _max_val = _flat.max().item()
                                if _max_val > 0:
                                    _counts = torch.histc(_flat, bins=50, min=0, max=_max_val).tolist()
                                    _hists_c[_prefix] = {'counts': _counts, 'min': 0.0, 'max': _max_val}
                                else:
                                    _hists_c[_prefix] = {'counts': [float(_flat.numel())] + [0.0]*49, 'min': 0.0, 'max': 1.0}
                            gc_dict['_transfer_hist'] = _hists_c
                            gc_dict['_transfer_hist_ready'] = True
                        else:
                            gc_dict['_transfer_hist_ready'] = False
                        gc_dict['transfer_lr_c'] = gc_dict['_transfer_lr_c']
                        gc_dict['transfer_n_calls'] = len(xc_list)
                        del xc_cat, dc_cat
                    gc_dict['_transfer_xc_all'] = []
                    gc_dict['_transfer_dc_all'] = []

            def hooked_transfer(method=None):
                if not gc_dict.get('active'):
                    _orig_transfer(method=method)
                    return

                # Capture A weights before transfer (will be reinit'd after)
                with torch.no_grad():
                    gc_dict['_A_pre_transfer'] = diag_tile.tile_a.get_weights()[0].clone().detach()

                _method = method if method is not None else ctrl.transfer_method
                gc_dict['_transfer_xc_all'] = []
                gc_dict['_transfer_dc_all'] = []
                gc_dict['_in_transfer'] = True

                # For set mode: C-ext tile.update is read-only, so read A/B
                # weights before transfer to reconstruct the x/d values
                if _method == "set":
                    with torch.no_grad():
                        A_raw = diag_tile.tile_a.tile.get_weights().to(device)
                        B_raw = diag_tile.tile_b.tile.get_weights().to(device)
                        alpha_a = diag_tile.tile_a.get_scales()
                        alpha_b = diag_tile.tile_b.get_scales()
                        alpha_c = diag_tile.tile_c.get_scales()
                        A_val = A_raw * alpha_a.view(-1, 1) if alpha_a is not None else A_raw
                        B_val = B_raw * alpha_b.view(-1, 1) if alpha_b is not None else B_raw
                        A_adj = A_val / alpha_c.view(-1, 1) if alpha_c is not None else A_val
                        # set mode passes: x=B_val [rank, x_size], d=(-A_adj).t() [rank, d_size]
                        gc_dict['_transfer_xc_all'].append(B_val.detach())
                        gc_dict['_transfer_dc_all'].append((-A_adj).t().detach())
                        eff_lr = ctrl._compute_effective_transfer_lr()
                        gc_dict['_transfer_lr_c'] = abs(eff_lr)

                _orig_transfer(method=method)
                gc_dict['_in_transfer'] = False
                _compute_transfer_stats(gc_dict)

            # Install hooks
            if hasattr(diag_tile.tile_c, '_orig_update'):
                diag_tile.tile_c._orig_update = _capture_c_update_outer
            else:
                diag_tile.tile_c.update = _capture_c_update_outer
            ctrl.ab_weight_transfer = hooked_transfer

            if ctrl.forward_inject_enabled:
                # FI mode: ab_weight_update is never called.
                # Hook tile_b._orig_update which fires last (after tile_a._orig_update).
                # At call time:
                #   x_input  = ctrl._fi_b_x = raw x         (tile_b's x)
                #   d_input  = ctrl._fi_b_d = DA = A^T d    (tile_b's d)
                #   ctrl._fi_a_x = XB = B x                 (tile_a's x, still cached)
                #   ctrl._fi_a_d = raw d (alpha removed)     (tile_a's d, still cached)
                # del happens after both _orig_update calls, so ctrl._fi_a_* are alive here.
                original_b_update = diag_tile.tile_b._orig_update

                def hooked_b(x_input, d_input, *args, **kwargs):
                    result = original_b_update(x_input, d_input, *args, **kwargs)
                    if gc_dict.get('active'):
                        _capture_common(
                            x_b=x_input,       # raw x
                            d_a=ctrl._fi_a_d,  # raw d (alpha already removed)
                            x_a=ctrl._fi_a_x,  # XB = B x
                            d_b=d_input,       # DA = A^T d
                        )
                    return result

                diag_tile.tile_b._orig_update = hooked_b
            else:
                # NFI mode: tile_c.update triggers ab_weight_update(x, d, lr).
                # x = raw batch input, d = raw gradient.
                # Tile_a physically sees (XB=B x, d); tile_b physically sees (x, DA=A^T d).
                # Compute XB and DA here so stats match FI mode.
                original_fn = ctrl.ab_weight_update

                def hooked(x, d, lr, **kwargs):
                    if gc_dict.get('active'):
                        with torch.no_grad():
                            A_w = diag_tile.tile_a.get_weights()[0].to(device)
                            B_w = diag_tile.tile_b.get_weights()[0].to(device)
                            x_dev = x.to(device); d_dev = d.to(device)
                            x_2d = x_dev.reshape(-1, x_dev.shape[-1])
                            d_2d = d_dev.reshape(-1, d_dev.shape[-1])
                            XB = x_2d @ B_w.t()   # [batch, rank]
                            DA = d_2d @ A_w        # [batch, rank]
                            gc_dict['_nfi_XB'] = XB
                            gc_dict['_nfi_DA'] = DA
                            gc_dict['_nfi_x'] = x_2d
                            gc_dict['_nfi_d'] = d_2d
                    result = original_fn(x, d, lr, **kwargs)
                    if gc_dict.get('active'):
                        with torch.no_grad():
                            XB = gc_dict.pop('_nfi_XB')
                            DA = gc_dict.pop('_nfi_DA')
                            x_2d = gc_dict.pop('_nfi_x')
                            d_2d = gc_dict.pop('_nfi_d')
                            _capture_common(x_b=x_2d, d_a=d_2d, x_a=XB, d_b=DA)
                    return result

                ctrl.ab_weight_update = hooked

        # Install hooks on every tile in diag_state.
        for _sn, _s in diag_state.items():
            _install_hook(_s["tile"], DEVICE, _s["gc"])
        print(f"Gradient tracking hooks installed on {len(diag_state)} tile(s)")

    # Initial evaluation
    init_f1, init_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
    wandb.log({"epoch": 0, "eval/f1": init_f1, "eval/em": init_em})
    print(f"Initial eval: F1={init_f1:.2f}, EM={init_em:.2f}")

    # Training loop
    best_f1 = init_f1
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history = []  # per-epoch F1, loss, etc.
    best_train_loss = float('inf')
    train_loss_no_improvement = 0
    global_step = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad()
        for micro_step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_positions = batch['start_positions'].to(DEVICE)
            end_positions = batch['end_positions'].to(DEVICE)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
                if diag_active:
                    snaps = {sn: snapshot_weights(s["tile"]) for sn, s in diag_state.items()}

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if diag_active:
                    for sn, s in diag_state.items():
                        tile = s["tile"]
                        gcd = s["gc"]
                        A_bef, B_bef, Ceff_bef = snaps[sn]
                        A_pre = gcd.pop('_A_pre_transfer', None)
                        _is_xfer = tile.controller.num_transfers > s["prev_nt"]
                        _do_erank = (ERANK_RATE_LIMIT_STEPS <= 0) or (
                            _is_xfer and (global_step - s["last_erank_step"]) >= ERANK_RATE_LIMIT_STEPS
                        )
                        if _do_erank:
                            s["last_erank_step"] = global_step
                        s["whist_count"] += 1
                        _do_whist = (HIST_RATE_STEPS > 0
                                     and (s["whist_count"] - 1) % HIST_RATE_STEPS == 0)
                        rec, s["prev_nt"] = collect_tile_diagnostics(
                            tile, A_bef, B_bef, Ceff_bef,
                            global_step, s["prev_nt"], s["A_CI"], s["B_CI"], s["C_CI"],
                            A_pre_transfer=A_pre, C_initial_eff=s["C_initial_eff"],
                            compute_erank=_do_erank, compute_weight_hist=_do_whist)
                        rec["loss"] = loss.item() * GRAD_ACCUM_STEPS

                        # Hook-based fields (only populated for tiles with hooks installed)
                        rec["norm_G_accum"] = gcd.get('norm_G_accum', 0.0)
                        rec["norm_AB_pre"] = gcd.get('norm_AB_pre', 0.0)
                        rec["cos_AB_G"] = gcd.get('cos_AB_G', 0.0)
                        # Temporal coherence of G (cosine to the previous step) and sigma_1(G)
                        rec["cos_G_prev"]      = gcd.get('cos_G_prev', 0.0)
                        rec["sigma1_G"]        = gcd.get('sigma1_G', 0.0)
                        rec["sigma1_G_ratio"]  = gcd.get('sigma1_G_ratio', 0.0)
                        # tile_a sees (XB, d_raw); tile_b sees (x_raw, DA)
                        for _pf in ("xa", "xb", "da", "db"):
                            rec[f"{_pf}_abs_mean"] = gcd.get(f"{_pf}_abs_mean", 0.0)
                            rec[f"{_pf}_abs_max"]  = gcd.get(f"{_pf}_abs_max",  0.0)
                        if gcd.get('_hist_ready'):
                            rec['xd_hist'] = gcd['_last_hist']

                        # Effective rank of the assembled (unsplit) C at transfer steps (DIAG_FULL_C)
                        if rec["is_transfer"] and s.get("arr") is not None:
                            with torch.no_grad():
                                _Cf = _assemble_full_C(s["arr"])
                                _er_f, _sg_f = _svd_stats(_Cf)
                                _er_fd, _sg_fd = _svd_stats(_Cf - s["C_full_init"])
                            rec["erank_C_full"] = _er_f
                            rec["sigma1_C_full"] = _sg_f
                            rec["erank_C_full_delta"] = _er_fd
                            rec["sigma1_C_full_delta"] = _sg_fd

                        # Transfer-event-only fields
                        if rec["is_transfer"]:
                            for _pf in ("xc", "dc"):
                                rec[f"{_pf}_abs_mean"] = gcd.get(f"{_pf}_abs_mean", 0.0)
                                rec[f"{_pf}_abs_max"]  = gcd.get(f"{_pf}_abs_max",  0.0)
                            rec['transfer_lr_c']    = gcd.get('transfer_lr_c', 0.0)
                            rec['transfer_n_calls'] = gcd.get('transfer_n_calls', 0)
                            if gcd.get('_transfer_hist_ready') and gcd.get('_transfer_hist'):
                                rec['xc_dc_hist'] = gcd['_transfer_hist']

                            with torch.no_grad():
                                C_eff_after = tile.tile_c.get_weights()[0].to(DEVICE)
                                C_eff_bef   = Ceff_bef.to(DEVICE)
                                delta_C_mat = C_eff_after - C_eff_bef
                                ctrl_delta = tile.controller.last_transfer_delta
                                tlr_AB = ctrl_delta.to(DEVICE) if ctrl_delta is not None else torch.zeros_like(delta_C_mat)
                                dC_f = delta_C_mat.flatten()
                                G_f  = gcd.get('G_accum', torch.zeros_like(delta_C_mat)).flatten()
                                tlr_f = tlr_AB.flatten()
                                rec["cos_dC_G"]      = _cos_sim(dC_f, G_f)
                                rec["cos_tlrAB_G"]   = _cos_sim(tlr_f, G_f)
                                rec["cos_dC_tlrAB"]  = _cos_sim(dC_f, tlr_f)
                                rec["norm_dC_step"]  = torch.norm(delta_C_mat).item()
                                rec["norm_tlrAB"]    = torch.norm(tlr_AB).item()

                            if 'G_accum' in gcd:
                                gcd['G_accum'] = torch.zeros_like(gcd['G_accum'])

                        s["log"].append(rec)

                        # Override snap A with pre-transfer values for next step's A_before
                        A_pre_now = gcd.get('_A_pre_transfer')
                        if A_pre_now is not None:
                            snaps[sn] = (A_pre_now,) + snaps[sn][1:]

                    # Progress bar (first/last preferred when DIAG_TILES="first_last")
                    _first_rec = diag_state["first"]["log"][-1] if "first" in diag_state else None
                    _last_rec  = diag_state["last"]["log"][-1]  if "last"  in diag_state else None
                    tag = ""
                    if _first_rec and _first_rec.get("is_transfer"): tag += " [T1]"
                    if _last_rec  and _last_rec.get("is_transfer"):  tag += " [T2]"
                    _ref = _first_rec or _last_rec or next(iter(s["log"][-1] for s in diag_state.values()), {})
                    pbar.set_postfix_str(
                        f"loss={loss.item() * GRAD_ACCUM_STEPS:.4f} "
                        f"||A||={_ref.get('norm_A', 0):.3f} "
                        f"T1={_first_rec.get('num_transfers', 0) if _first_rec else 0} "
                        f"T2={_last_rec.get('num_transfers', 0) if _last_rec else 0}{tag}")
                else:
                    pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            num_batches += 1

        # Deactivate hooks after DIAG_EPOCHS
        if ENABLE_DIAGNOSTIC and DIAG_EPOCHS > 0 and epoch == DIAG_EPOCHS:
            for _s in diag_state.values():
                _s["gc"]["active"] = False
            print(f"Diagnostic collection stopped after epoch {epoch}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/f1": eval_f1, "eval/em": eval_em,
            "learning_rate": current_lr,
        })

        epoch_history.append({"epoch": epoch, "f1": eval_f1, "em": eval_em, "train_loss": train_loss, "lr": current_lr})

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        train_loss_improved = ""
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            train_loss_no_improvement = 0
            train_loss_improved = " (down)"
        else:
            train_loss_no_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train loss: {train_loss:.4f}{train_loss_improved} | "
            f"F1 {eval_f1:.2f}% | EM {eval_em:.2f}% | "
            f"Best F1 {best_f1:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if best_train_loss > TRAIN_LOSS_THRESHOLD and train_loss_no_improvement >= TRAIN_LOSS_EARLY_STOP_PATIENCE:
            tqdm.write(f"Train loss early stop at epoch {epoch} "
                       f"(train_loss={train_loss:.4f} > {TRAIN_LOSS_THRESHOLD}, no improvement for {train_loss_no_improvement} epochs)")
            break

        if best_train_loss <= TRAIN_LOSS_THRESHOLD and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest F1: {best_f1:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Save diagnostic outputs
    # =========================================================================
    _any_log = any(s["log"] for s in diag_state.values()) if ENABLE_DIAGNOSTIC and diag_state else False
    if ENABLE_DIAGNOSTIC and _any_log:
        stamp = DIAG_STAMP or f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"
        # Summary print
        diag_steps_total = max(len(s["log"]) for s in diag_state.values())
        _xfer_counts = {sn: sum(1 for r in s["log"] if r["is_transfer"]) for sn, s in diag_state.items()}
        print(f"\nDiag: {diag_steps_total}/{global_step} steps, "
              f"transfers={ {sn: c for sn, c in _xfer_counts.items()} }")

        # Build per-tile dict for JSON
        tiles_out = {}
        for sn, s in diag_state.items():
            xfer_steps = [r["step"] for r in s["log"] if r["is_transfer"]]
            tiles_out[sn] = {
                "name": s["full_name"],
                "A_shape": list(A_shape), "B_shape": list(B_shape), "C_shape": list(C_shape),
                "A_cell_indices": s["A_CI"], "B_cell_indices": s["B_CI"], "C_cell_indices": s["C_CI"],
                "total_transfers": len(xfer_steps), "transfer_steps": xfer_steps,
                "steps": s["log"],
            }

        # Output JSON: "tiles" dict, plus first_tile/last_tile aliases
        output = {
            "config": {
                "learning_rate": LEARNING_RATE, "transfer_lr": TRANSFER_LR,
                "transfer_every": TRANSFER_EVERY, "lrtt_rank": LRTT_RANK,
                "fast_lr": FAST_LR, "auto_scale_mode": AUTO_SCALE_MODE, "reinit_mode": REINIT_MODE,
                "transfer_method": TRANSFER_METHOD, "optimizer": OPTIMIZER,
                "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
                "diag_epochs": DIAG_EPOCHS,
                "diag_tiles": DIAG_TILES if isinstance(DIAG_TILES, str) else {str(k): v for k, v in DIAG_TILES.items()},
                "diag_groups": dict(DIAG_GROUPS),
            },
            "best_f1": best_f1, "best_epoch": best_epoch,
            "epoch_history": epoch_history,
            "total_steps": global_step, "diag_steps": diag_steps_total,
            "tiles": tiles_out,
        }
        if "first" in tiles_out: output["first_tile"] = tiles_out["first"]
        if "last"  in tiles_out: output["last_tile"]  = tiles_out["last"]

        json_path = os.path.join(RESULTS, f"squad_diagnostic_log_{stamp}.json")
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {json_path}")

        # Per-tile plots
        for sn, s in diag_state.items():
            label = f"{sn} tile ({s['full_name']})"
            make_diagnostic_plots(s["log"],
                os.path.join(RESULTS, f"squad_diag_{sn}_{stamp}.png"),
                tile_label=label,
                A_ci=s["A_CI"], B_ci=s["B_CI"], C_ci=s["C_CI"])
            make_weight_dynamics_plots(s["log"],
                os.path.join(RESULTS, f"squad_weight_dyn_{sn}_{stamp}.png"),
                tile_label=label)
            make_weight_hist_plots(s["log"],
                os.path.join(RESULTS, f"squad_weight_hist_{sn}_{stamp}.png"),
                tile_label=label)
            make_xd_diagnostic_plots(s["log"],
                os.path.join(RESULTS, f"squad_diag_xd_{sn}_{stamp}.png"),
                tile_label=label)

        # Per-epoch plots (first and last tile only, when present)
        steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
        diag_ep = DIAG_EPOCHS if DIAG_EPOCHS > 0 else N_EPOCHS
        for ep in range(1, diag_ep + 1):
            s0, s1 = (ep-1)*steps_per_epoch, ep*steps_per_epoch
            for sn in ("first", "last"):
                if sn not in diag_state:
                    continue
                ss = diag_state[sn]
                slc = ss["log"][s0:s1]
                if not slc:
                    break
                make_diagnostic_plots(slc,
                    os.path.join(RESULTS, f"squad_diag_{sn}_{stamp}_ep{ep}.png"),
                    tile_label=f"{sn} tile ({ss['full_name']}) - Epoch {ep}",
                    A_ci=ss["A_CI"], B_ci=ss["B_CI"], C_ci=ss["C_CI"])

    # Memory cleanup
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared")

    wandb.finish()


if __name__ == "__main__":
    _args = parse_args()
    apply_overrides(_args)
    if _args.print_config:
        print(json.dumps(effective_config(), indent=2, default=str))
        sys.exit(0)
    main()
