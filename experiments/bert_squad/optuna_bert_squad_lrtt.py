# -*- coding: utf-8 -*-
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""Optuna hyperparameter search for LR-TT fine-tuning of BERT-base on SQuAD v1.1.

The defaults reproduce the common configuration of the paper's BERT-base
studies: AnalogAdam, batch size 48, 5 epochs, linear warmup over 365 steps
then linear decay, no weight decay, maximum sequence length 384, rank 32,
rank-wise stochastic transfer (onehot) with a constant transfer learning
rate, no auxiliary forward contribution, A and B retained after transfer
(reinit mode decay), constant-step ideal A/B and C devices with 10-bit
multi-level updates, 8-bit DAC/ADC with output noise 0.04, and non-adapted
encoder layers mapped as frozen analog tiles. README_optuna.md lists the
search spaces and the parameter sets of the paper.

Usage:
    python optuna_bert_squad_lrtt.py --lora-target qkvo --n-trials 50   # QKVO study (rank 32)
    python optuna_bert_squad_lrtt.py --lora-target ffn                   # FFN study
    python optuna_bert_squad_lrtt.py --lora-target all                   # all-layer study
    python optuna_bert_squad_lrtt.py --lora-target qkvo --rank-sweep     # rank sweep (r = 1..64)
    python optuna_bert_squad_lrtt.py --ab-device constantstep6t1cgamma --search-space wide
                                                                         # study A (asymmetric A/B device)
    python optuna_bert_squad_lrtt.py --ab-device 6t1c                    # study B (full 6T1C device model)
    python optuna_bert_squad_lrtt.py --lora-target qkvo --print-config   # resolved configuration, no training
    python optuna_bert_squad_lrtt.py --lora-target qkvo --visualize      # plots of an existing study

Trials are stored in an Optuna journal file, <results-dir>/optuna_<study>.log,
whose study name is derived from the configuration; launching the same
configuration again (also from several processes) extends the study. Fixed
hyperparameters are recorded as single-value distributions. Every trial trains
with early stopping and returns its best validation F1. A trial that runs out
of GPU memory is enqueued again with a larger gradient-accumulation factor
(the batch of 48 split into more micro-batches) and the process restarts
itself.

Options are listed by --help; flags that are on by default have a --no-...
form (for example --no-encoder-analog). Settings without a flag are edited in
the script: DYNAMIC_TE, TE_WARMUP_STEPS, REINIT_GAIN, LAYER_IO_BITS (per-module
DAC/ADC bits), TRAIN_SUBSET_SIZE and EVAL_SUBSET_SIZE.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import re
import string
import math
import json
import argparse
import gc
import collections

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna_integration import BoTorchSampler
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports: prefer the in-tree package (../../src) over an installed copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401  (per-micro-batch tile.update + LRTT A/B snapshot)

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice, IdealDevice, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter

from collections import Counter


# =============================================================================
# ConfigAwareBoTorchSampler
# =============================================================================

class ConfigAwareBoTorchSampler(BoTorchSampler):
    """BoTorchSampler that respects OPT_CONFIG and avoids duplicate running trials.

    Contextual GP over a dynamic search space (instead of Optuna's
    IntersectionSearchSpace):
    - GP search space = bounding box of (recorded values union current suggest
      range) per parameter, so trials recorded with other suggest ranges of the
      same study also enter the GP.
    - Acquisition is constrained to the *current* suggest region: width>0 params
      via optimize_acqf bounds, width-0 (fixed) params via fixed_features. A
      parameter that is fixed in the current configuration but varies among the
      recorded trials (e.g. ab_multilevel) stays in the GP as a context
      dimension, so observations from other settings contribute conditionally.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('candidates_func', self._contextual_candidates_func)
        super().__init__(*args, **kwargs)
        self._acq_ranges = {}    # name -> (low, high): current suggest range (width > 0)
        self._fixed_values = {}  # name -> value: fixed (width 0) in the current configuration
        self._gp_space = {}      # name -> GP bounding-box distribution

    def infer_relative_search_space(self, study, trial):
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")
        trials = [t for t in study.get_trials(deepcopy=False)
                  if t.state in (TrialState.COMPLETE, TrialState.RUNNING) and t.distributions]
        completed = [t for t in trials if t.state == TrialState.COMPLETE]
        if not completed:
            return {}
        newest = max(trials, key=lambda t: t.number)
        common = set(completed[0].params)
        for t in completed[1:]:
            common &= set(t.params)
        space, acq_ranges, fixed = {}, {}, {}
        for name in sorted(common & set(newest.distributions)):
            cur = newest.distributions[name]
            if isinstance(cur, CategoricalDistribution):
                continue  # keep categorical params on the independent path
            vals = [t.params[name] for t in completed]
            lo, hi = min(min(vals), cur.low), max(max(vals), cur.high)
            if lo == hi:
                continue  # true constant: never swept, nothing for the GP
            log = cur.log and lo > 0
            if isinstance(cur, IntDistribution):
                space[name] = IntDistribution(int(lo), int(hi), log=log)
            else:
                space[name] = FloatDistribution(float(lo), float(hi), log=log)
            if cur.single():
                fixed[name] = cur.low
            else:
                acq_ranges[name] = (cur.low, cur.high)
        self._gp_space, self._acq_ranges, self._fixed_values = space, acq_ranges, fixed
        return space

    def _contextual_candidates_func(self, train_x, train_obj, train_con, bounds, pending_x):
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from optuna._transform import _SearchSpaceTransform

        train_x = normalize(train_x, bounds=bounds)
        model = SingleTaskGP(train_x, train_obj,
                             outcome_transform=Standardize(m=train_obj.size(-1)))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        # Running trials enter as X_pending: the acquisition marginalizes over their
        # unknown outcomes, so its argmax moves away from points that other processes
        # are evaluating (processes that see the same completed trials would
        # otherwise propose near-identical trials).
        pending = None
        if pending_x is not None and len(pending_x):
            pending = normalize(pending_x, bounds=bounds)
        acqf = qLogExpectedImprovement(
            model=model, best_f=train_obj.max(), X_pending=pending,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )

        def to_unit(name, value, col):
            # map a raw value to the [0,1] coordinate of its GP-box column
            tval = float(_SearchSpaceTransform({name: self._gp_space[name]})
                         .transform({name: value})[0])
            lo, hi = float(bounds[0, col]), float(bounds[1, col])
            return min(1.0, max(0.0, (tval - lo) / (hi - lo)))

        acq_bounds = torch.zeros_like(bounds)
        acq_bounds[1] = 1.0
        fixed_features = {}
        for col, name in enumerate(self._gp_space):
            if name in self._fixed_values:
                fixed_features[col] = to_unit(name, self._fixed_values[name], col)
            else:
                a_lo, a_hi = self._acq_ranges[name]
                acq_bounds[0, col] = to_unit(name, a_lo, col)
                acq_bounds[1, col] = to_unit(name, a_hi, col)

        candidates, _ = optimize_acqf(
            acq_function=acqf, bounds=acq_bounds, q=1,
            num_restarts=10, raw_samples=512,
            fixed_features=fixed_features or None,
            options={"batch_limit": 5, "maxiter": 200},
        )
        print(f"[GP] contextual qLogEI: {train_x.size(0)} completed trials, "
              f"{train_x.size(1)}D, pending={0 if pending is None else len(pending)} "
              f"(fixed: {sorted(self._fixed_values)})", flush=True)
        return unnormalize(candidates.detach(), bounds=bounds)

    def _is_almost_identical_to_running(self, params, study, threshold=0.01):
        """Check if params are almost identical to any running trial (within 1%)."""
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        for rt in running_trials:
            identical = True
            for key, val in params.items():
                if key in rt.params and isinstance(val, (int, float)):
                    rt_val = rt.params[key]
                    if rt_val != 0:
                        if abs(val - rt_val) / abs(rt_val) > threshold:
                            identical = False
                            break
                    elif val != 0:
                        identical = False
                        break
            if identical:
                return True
        return False

    def _add_jitter(self, params, search_space, jitter_ratio=0.1):
        """Add small jitter to params while keeping within search space bounds."""
        import random
        jittered = dict(params)
        for key, val in params.items():
            if key in search_space and isinstance(val, float):
                dist = search_space[key]
                if hasattr(dist, 'low') and hasattr(dist, 'high'):
                    # Add +/-10% jitter
                    jitter = val * random.uniform(-jitter_ratio, jitter_ratio)
                    new_val = val + jitter
                    # Clip to the current suggest range (the GP box may be wider
                    # than what suggest() accepts; out-of-range values would fall
                    # back to independent sampling in Trial._suggest)
                    lo, hi = self._acq_ranges.get(key, (dist.low, dist.high))
                    new_val = max(lo, min(hi, new_val))
                    jittered[key] = new_val
        return jittered

    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)

        # If almost identical to running trial, add jitter to GP params
        if self._is_almost_identical_to_running(params, study):
            params = self._add_jitter(params, search_space)

        # Force reinit_mode if fixed in config
        if OPT_CONFIG['reinit_mode'] is not None and 'reinit_mode' in params:
            params['reinit_mode'] = OPT_CONFIG['reinit_mode']
        # Force optimizer if fixed in config
        if 'optimizer' in params:
            params['optimizer'] = OPT_CONFIG['optimizer']

        # Snap width-0 (fixed) params to their exact values: the normalize ->
        # unnormalize round-trip drifts by ~1e-12, which would fail
        # Trial._suggest's containment check against the single-value distribution
        for key, val in self._fixed_values.items():
            if key in params:
                params[key] = val
        # Same for boundary proposals on free params: clamp to the current
        # suggest range so a ~1e-12 overshoot doesn't demote the param to
        # independent (random) sampling
        for key, (lo, hi) in self._acq_ranges.items():
            if key in params:
                params[key] = min(max(params[key], lo), hi)
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == 'reinit_mode' and OPT_CONFIG['reinit_mode'] is not None:
            return OPT_CONFIG['reinit_mode']
        if param_name == 'optimizer':
            return OPT_CONFIG['optimizer']
        return super().sample_independent(study, trial, param_name, param_distribution)


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths (--results-dir): journal, best-params file and plots of every study
RESULTS = os.path.join(os.getcwd(), "results", "optuna_bert_squad_lrtt")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384

# Training defaults
N_EPOCHS = 5
BATCH_SIZE = 48
GRAD_ACCUM_STEPS = 1
_DEFAULT_GRAD_ACCUM = 1  # --grad-accum-steps; objective() sets GRAD_ACCUM_STEPS per trial (a re-queued trial carries its factor in user_attrs)
EVAL_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 2
TRAIN_LOSS_EARLY_STOP_PATIENCE = 1  # Stop if train loss doesn't improve for this many epochs
TRAIN_LOSS_THRESHOLD = 1.5  # Once train loss drops below this, rely on metric-based early stop only

# Scheduler
WARMUP_STEPS = 365

# Dynamic TE
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Fixed LRTT parameters
REINIT_GAIN = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
AB_DEVICE = "constantstepideal"  # "6t1c", "linearstep", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "fp", or "ideal"
A_DEVICE = None  # Optional override for A tile device. None -> use AB_DEVICE for both A and B.
B_DEVICE = None  # Optional override for B tile device. None -> use AB_DEVICE for both A and B.
# Per-tile split sweep toggles. When True, sweep a_X and b_X independently; when
# False, sweep a single ab_X applied to both A and B tiles (all paper studies).
SPLIT_AB_PARAMS = {
    'dw_min': False,
    'multilevel': False,
    'tau_sec': False,
    'reset_std': False,
    'desired_bl': False,
    'weight_scaling_omega': False,
}

# Per-module IO bit override (edited here; None = use the dac_bits/adc_bits of the trial).
# Set ADC(out_res)/DAC(inp_res) bits per module TYPE - e.g. give FFN higher precision
# than attention. The bits are applied when the model is converted, one conversion pass per
# bit group, to the forward and backward inp_res (DAC) / out_res (ADC) of the matched tiles.
# Example FFN 10-bit ADC: set 'intermediate'/'output.dense' adc=10.
LAYER_IO_BITS = {
    'query':            {'dac': None, 'adc': None},
    'key':              {'dac': None, 'adc': None},
    'value':            {'dac': None, 'adc': None},
    'attention.output': {'dac': None, 'adc': None},   # attention output (O) projection
    'intermediate':     {'dac': None, 'adc': None},   # FFN1 (pre-GELU)
    'output.dense':     {'dac': None, 'adc': None},   # FFN2 (pre-LayerNorm)
}
C_DEVICE = "constantstepideal"  # "softboundsideal", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", or "ideal"
IO_NOISE = True  # If False, disable out_noise (resolution kept)
FORWARD_INJECT = False  # If True, add the auxiliary A*B product to the forward pass (W_eff = C + alpha*A*B)
IS_PERFECT = False  # If True, forward/backward use ideal FP matmul (no ADC/DAC/noise)
NO_QUANT = False  # If True, disable DAC/ADC quantization (inp_res/out_res -> -1)
ENCODER_ANALOG = True  # If True, non-LRTT encoder layers become frozen analog instead of digital
HEAD_ANALOG = False  # If True, qa_outputs -> frozen analog instead of digital
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)

# Fixed device and IO values of the paper studies (CLI-overridable). Each is
# recorded in the journal as a single-value distribution.
AB_MULTILEVEL_BITS = 10  # A/B multi-level bits: w_max - w_min = 2^bits * ab_dw_min (--ab-bits)
C_DW_MIN = 0.001953  # C tile dw_min, 10-bit core (--c-dw-min)
AB_RESET_STD = None  # A/B reset_std; None = 0.01 for the 6t1c device, 1e-30 otherwise (--ab-reset-std)
OUT_NOISE = 0.04  # output noise of the analog forward and backward passes (--out-noise)
DAC_BITS = 8  # --dac-bits
ADC_BITS = 8  # --adc-bits

# Suggest ranges (low, high), all log-uniform, of the paper's studies;
# README_optuna.md lists the studies that use each set.
SEARCH_SPACES = {
    # FFN, ALL and study B (full 6T1C device model); contains the 'qkvo-r32' and 'rank' ranges
    'paper': {'learning_rate': (8e-5, 2e-2), 'transfer_lr': (8e-6, 5e-2),
              'transfer_every': (1, 500), 'fast_lr': (9e-3, 4.0), 'ab_dw_min': (7e-6, 5e-3)},
    # QKVO at rank 32 (QKVO operating point)
    'qkvo-r32': {'learning_rate': (8e-5, 2e-2), 'transfer_lr': (1e-4, 3e-2),
                 'transfer_every': (1, 80), 'fast_lr': (7e-2, 2.0), 'ab_dw_min': (3e-5, 6e-4)},
    # study A (asymmetric A/B device; AF = 1 anchor)
    'wide': {'learning_rate': (1e-4, 2e-2), 'transfer_lr': (6e-6, 2e-2),
             'transfer_every': (1, 700), 'fast_lr': (9e-3, 6.0), 'ab_dw_min': (5e-6, 4e-2)},
    # QKVO rank sweep (--rank-sweep, rank_exp 0..6)
    'rank': {'learning_rate': (7e-4, 9e-3), 'transfer_lr': (1e-4, 3e-2),
             'transfer_every': (1, 80), 'fast_lr': (3e-2, 0.9), 'ab_dw_min': (2e-5, 7e-4)},
}
SEARCH_SPACE = 'paper'  # --search-space
RANK = 32  # --rank: LR-TT rank (power of two); rank_exp = log2(RANK) is recorded as a fixed param
RANK_SWEEP = False  # --rank-sweep: search rank_exp in 0..6 instead

# LoRA target options: which layers have trainable A/B tiles (LORA_TARGET_MODULES below)
# - none: no LRTT layers
# - qkv: only query, key, value
# - qkvo: query, key, value and the attention output projection (default)
# - ffn: FFN layers (intermediate.dense, output.dense)
# - all: all encoder linear layers
LORA_TARGET = "qkvo"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze) - qa_outputs for SQuAD
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT
    "qonly": ["query"],  # Query only (12 layers)
    "konly": ["key"],  # Key only (12 layers)
    "vonly": ["value"],  # Value only (12 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (36 layers)
    "qkvo": ["query", "key", "value", "attention.output"],  # Q/K/V + attention output (48 layers)
    "ffn": (["intermediate", "output.dense"], ["attention"]),  # FFN only (24 layers)
    "dense": ["dense"],  # All layers with "dense" (excludes qkv) (36 layers)
    "allnobn": None,  # Same as all (no bottleneck in BERT) (72 layers)
    "all": None,  # None means all encoder layers (no filtering) (72 layers)
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogAdam',
    'tune_wd': False,        # weight_decay = 0 (fixed)
    'tune_momentum': False,  # SGD momentum = 0 (AnalogSGD only; AnalogAdam keeps betas (0.9, 0.999))
    'tune_nesterov': False,  # no Nesterov momentum (AnalogSGD only)
    'reinit_mode': 'decay',  # fixed reinit mode, or None = tune among standard/decay/hybrid
    'no_transfer': False,   # If True, disable transfer (transfer_every = inf)
    'ab_io_perfect': False,  # If True, A/B tiles fully ideal (no out_noise/ADC/DAC)
    'learn_out_scaling': False,  # If True, C tile out_scaling is trainable
    'auto_scale_mode': 'none',
    'correct_gradient_magnitudes': False,
    'transfer_rank_schedule': 'all',
    'transfer_ranks_per_step': 1,
    'scale_transfer_lr': False,  # If True, the transfer LR follows the optimizer LR schedule
    'fi_continuous_alpha': False,
    'ab_pulse_type': 'default',
    'ab_multilevel': True,  # If True, A/B w_max = 2^(AB_MULTILEVEL_BITS-1) * ab_dw_min; else w_max = 1.0
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config.

    The device, out-scaling, encoder, transfer-LR and multi-level markers are
    added when a setting differs from a reference configuration (6t1c A/B
    device, softboundsideal C device, trainable out scaling, scaled transfer
    LR, ...); this scheme gives the study names of the paper's studies.
    """
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    # Add reinit_mode if fixed
    if OPT_CONFIG['reinit_mode'] is not None:
        suffix += f"_{OPT_CONFIG['reinit_mode']}"

    if not OPT_CONFIG['tune_wd']:
        suffix += "_nowd"
    if not OPT_CONFIG['tune_momentum']:
        suffix += "_nomom"
    if not OPT_CONFIG['tune_nesterov']:
        suffix += "_nonest"

    # Always add transfer method
    suffix += f"_{TRANSFER_METHOD}"

    # Separate A and B devices are encoded as _a<A>_b<B>; a shared device as
    # _<AB_DEVICE>, which is omitted for 6t1c.
    if (A_DEVICE is not None or B_DEVICE is not None) and (A_DEVICE != B_DEVICE or (A_DEVICE or B_DEVICE) != AB_DEVICE):
        a_dev = (A_DEVICE if A_DEVICE is not None else AB_DEVICE).replace('-', '')
        b_dev = (B_DEVICE if B_DEVICE is not None else AB_DEVICE).replace('-', '')
        suffix += f"_a{a_dev}_b{b_dev}"
    elif AB_DEVICE != "6t1c":
        suffix += f"_{AB_DEVICE.replace('-', '')}"

    if C_DEVICE != "softboundsideal":
        suffix += f"_c{C_DEVICE}"

    if not IO_NOISE:
        suffix += "_noio"

    if FORWARD_INJECT:
        suffix += "_fwinj"

    if IS_PERFECT:
        suffix += "_perfect"
    if NO_QUANT:
        suffix += "_noquant"

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

    if OPT_CONFIG.get('ab_io_perfect', False):
        suffix += "_abperf"

    if not OPT_CONFIG.get('learn_out_scaling', True):
        suffix += "_noos"

    if ENCODER_ANALOG:
        suffix += "_encanalog"

    if HEAD_ANALOG:
        suffix += "_headanalog"

    if BACKWARD_OUT_BOUND != 12.0:
        suffix += f"_bob{BACKWARD_OUT_BOUND:g}"

    if OPT_CONFIG.get('auto_scale_mode', 'none') != 'none':
        suffix += f"_as-{OPT_CONFIG['auto_scale_mode']}"
    if OPT_CONFIG.get('correct_gradient_magnitudes', False):
        suffix += "_cgm"
    if OPT_CONFIG.get('transfer_rank_schedule', 'all') != 'all':
        suffix += f"_trs-{OPT_CONFIG['transfer_rank_schedule']}-{OPT_CONFIG['transfer_ranks_per_step']}"
    if not OPT_CONFIG.get('scale_transfer_lr', True):
        suffix += "_no-stlr"
    if OPT_CONFIG.get('fi_continuous_alpha', False):
        suffix += "_fica"
    if OPT_CONFIG.get('ab_pulse_type', 'default') != 'default':
        suffix += f"_abpt-{OPT_CONFIG['ab_pulse_type']}"
    if OPT_CONFIG.get('ab_multilevel', False):
        suffix += "_abml"

    # Per-module IO bits marker (LAYER_IO_BITS overrides) - keeps differing per-layer
    # bit configs in separate studies. e.g. FFN 10-bit ADC -> _lio-f1a10-f2a10.
    if _any_layer_io_override():
        _abbr = {'query': 'q', 'key': 'k', 'value': 'v', 'attention.output': 'o',
                 'intermediate': 'f1', 'output.dense': 'f2'}
        _parts = []
        for _k in ['query', 'key', 'value', 'attention.output', 'intermediate', 'output.dense']:
            _b = LAYER_IO_BITS.get(_k, {})
            _d, _a = _b.get('dac'), _b.get('adc')
            if _d is not None or _a is not None:
                _parts.append(_abbr[_k] + (f"d{_d}" if _d is not None else "")
                              + (f"a{_a}" if _a is not None else ""))
        if _parts:
            suffix += "_lio-" + "-".join(_parts)

    # Split mode markers (only when at least one ab param is split per-tile).
    split_keys = sorted(k for k, v in SPLIT_AB_PARAMS.items() if v)
    if split_keys:
        suffix += "_split-" + "-".join(split_keys)

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    # Add epoch count
    suffix += f"_{N_EPOCHS}ep"

    return suffix


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=0.0, dw_min=0.001981, multilevel=None, device_name=None,
                      reset_std=0.01):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c              - LinearStepDevice with the 6T1C parameters: asymmetry (gamma_up,
                            gamma_down), device-to-device and pulse-to-pulse variation,
                            write_noise_std 0
        linearstep        - LinearStepDevice with default params (no nonlinearity, default noise)
        linearstepideal   - LinearStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep      - ConstantStepDevice with default params (constant step, default noise)
        constantstepideal - ConstantStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep6t1cgamma - LinearStepDevice with all noise/dtod=0 but gamma_up/gamma_down from 6t1c
        fp                - FloatingPointDevice (perfect, no quantization/bounds)
        ideal             - IdealDevice

    Args:
        tau_sec: Retention time constant. If 0, lifetime=0 (no decay).
        dw_min: Minimum weight update step size for the device.
        multilevel: If set, w_max - w_min = (2 ** multilevel) * dw_min (symmetric) for
                    6t1c, linearstepideal, constantstepideal and constantstep6t1cgamma.
                    Default (None) keeps w_max=1, w_min=-1.
        device_name: Optional override for the device type. None -> use AB_DEVICE.
                     Used to instantiate A and B tiles independently when A_DEVICE
                     or B_DEVICE differs from AB_DEVICE.
        reset_std: sigma for the reset (capacitor-discharge) operation. Default 0.01,
                   the reset noise of the 6T1C device model. Applied to all pulsed-device
                   branches; ignored for non-pulsed device classes (fp / ideal).

    Note: reset_dtod is hardcoded to 0.0 (PulsedDevice default) on every pulsed-device
    branch - written explicitly for readability, not configurable.
    """
    name = device_name if device_name is not None else AB_DEVICE

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    # Compute symmetric w_max from multilevel
    if multilevel is not None and multilevel > 0:
        w_max = (2 ** multilevel) * dw_min / 2.0
    else:
        w_max = 1.0
    w_min = -w_max

    if name == "fp":
        return FloatingPointDevice()
    if name == "ideal":
        return IdealDevice()
    if name == "linearstep":
        return LinearStepDevice(
            dw_min=dw_min,
            lifetime=lifetime,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
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
    if name == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min,
            lifetime=lifetime,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if name == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
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

    # Default: 6t1c (full noise)
    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0,
        w_max=w_max,
        w_min=w_min,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_std=reset_std,
        reset_dtod=0.0,
    )


def _create_c_device(dw_min=0.001, reset_std=0.0):
    """Create device for C tile.

    Args:
        dw_min: Minimum weight update step size for the device.
        reset_std: sigma for the reset operation. Default 0.0 (deterministic reset). Applied
                   to all pulsed-device branches; ignored for ideal/floating-point.
    """
    if C_DEVICE == "ideal":
        return IdealDevice()
    if C_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
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
    if C_DEVICE == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if C_DEVICE == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
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
    return SoftBoundsDevice(
        dw_min=dw_min,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        reset_std=reset_std,
        reset_dtod=0.0,
        mult_noise=False,  # No multiplicative noise for C tile
    )


def _bits_to_res(bits):
    """Convert a converter bit-count to aihwkit resolution (1/steps).

    None / <2  -> no override (keep current inp_res/out_res: aihwkit default or NO_QUANT).
    N>=2       -> 1/(2**N - 2)  (N-bit signed converter).
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
        # Apply (possibly FFN-specific) bits, overriding the inherited qkvo quantization.
        # No-op when dac_bits/adc_bits match the inherited values (default path).
        _apply_quant_bits(rpu_config, dac_bits, adc_bits)
    else:
        rpu_config = SingleRPUConfig(device=_create_c_device())
        rpu_config.mapping = MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=OPT_CONFIG.get('learn_out_scaling', True),
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
        _apply_quant_bits(rpu_config, dac_bits, adc_bits)
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config(rank, transfer_every, transfer_lr, fast_lr, reinit_mode,
                       c_dw_min=0.001, c_desired_bl=None, out_noise=0.0,
                       auto_scale_mode='none', correct_gradient_magnitudes=False,
                       transfer_rank_schedule='all', transfer_ranks_per_step=1,
                       scale_transfer_lr=True,
                       fi_continuous_alpha=False,
                       ab_pulse_type='default',
                       a_tau_sec=0.0, b_tau_sec=0.0,
                       a_dw_min=0.001981, b_dw_min=0.001981,
                       a_multilevel=None, b_multilevel=None,
                       a_reset_std=0.01, b_reset_std=0.01,
                       ab_desired_bl=31,
                       a_desired_bl=None, b_desired_bl=None,
                       ab_weight_scaling_omega=0.0,
                       a_weight_scaling_omega=None, b_weight_scaling_omega=None,
                       lora_alpha=1.0,
                       dac_bits=None, adc_bits=None):
    """Create LRTT RPU configuration for analog layers.

    Per-tile parameters (each can differ between A and B):
      - tau_sec, dw_min, multilevel, reset_std: applied to LinearStepDevice instances
      - desired_bl: applied via PythonLRTTDevice.{a,b}_desired_bl override
      - weight_scaling_omega: applied via PythonLRTTDevice.{mapping_a, mapping_b}
        overrides of mapping_ab

    a_desired_bl / b_desired_bl: when None, both inherit ab_desired_bl (the rpu.update
    level value). When set, override per tile.
    a_weight_scaling_omega / b_weight_scaling_omega: when None, both inherit
    ab_weight_scaling_omega (the mapping_ab level value). When set, build mapping_a /
    mapping_b override objects.
    """
    # A and B tile devices: independently overridable via A_DEVICE / B_DEVICE.
    # When both are None, both A and B use AB_DEVICE (the default).
    a_device = _create_ab_device(tau_sec=a_tau_sec, dw_min=a_dw_min, multilevel=a_multilevel,
                                 device_name=A_DEVICE, reset_std=a_reset_std)
    b_device = _create_ab_device(tau_sec=b_tau_sec, dw_min=b_dw_min, multilevel=b_multilevel,
                                 device_name=B_DEVICE, reset_std=b_reset_std)
    c_device = _create_c_device(dw_min=c_dw_min)

    # Scale B initialization to the w_max/w_min bounds of the B device.
    # Default w_max=1.0; with multilevel, w_max = 2^(multilevel-1) * b_dw_min.
    # reinit_gain multiplies B_init (and A_init, but A is "zero" by default).
    if b_multilevel is not None and b_multilevel > 0:
        b_w_max = (2 ** b_multilevel) * b_dw_min / 2.0
        b_reinit_gain = REINIT_GAIN * b_w_max
    else:
        b_reinit_gain = REINIT_GAIN

    te = transfer_every

    # Build mapping_a / mapping_b overrides only when per-tile omega differs from shared.
    def _build_mapping(omega):
        return MappingParameter(
            weight_scaling_omega=omega,
            learn_out_scaling=False,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        )

    mapping_ab_obj = _build_mapping(ab_weight_scaling_omega)
    mapping_a_obj = _build_mapping(a_weight_scaling_omega) if a_weight_scaling_omega is not None else None
    mapping_b_obj = _build_mapping(b_weight_scaling_omega) if b_weight_scaling_omega is not None else None

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=lora_alpha,
        fast_lr=fast_lr,
        reinit_gain=b_reinit_gain,
        reinit_mode=reinit_mode,
        unit_cell_devices=[a_device, b_device, c_device],
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=mapping_ab_obj,
        mapping_a=mapping_a_obj,
        mapping_b=mapping_b_obj,
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=OPT_CONFIG.get('learn_out_scaling', True),
            out_scaling_columnwise=True,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = FORWARD_INJECT
    device_config.ab_io_perfect = OPT_CONFIG.get('ab_io_perfect', False)
    device_config.auto_scale_mode = auto_scale_mode
    device_config.correct_gradient_magnitudes = correct_gradient_magnitudes
    device_config.transfer_rank_schedule = transfer_rank_schedule
    device_config.transfer_ranks_per_step = transfer_ranks_per_step
    device_config.scale_transfer_lr = scale_transfer_lr
    device_config.fi_continuous_alpha = fi_continuous_alpha
    device_config.ab_pulse_type = ab_pulse_type
    if c_desired_bl is not None:
        device_config.c_desired_bl = c_desired_bl
    if a_desired_bl is not None:
        device_config.a_desired_bl = a_desired_bl
    if b_desired_bl is not None:
        device_config.b_desired_bl = b_desired_bl

    # Dynamic TE
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.max_input_size = 0 if IS_PERFECT else 512
    rpu_config.mapping.max_output_size = 0 if IS_PERFECT else 512

    rpu_config.update.desired_bl = ab_desired_bl

    rpu_config.forward.out_noise = out_noise
    rpu_config.backward.out_noise = out_noise
    rpu_config.forward.is_perfect = IS_PERFECT
    rpu_config.backward.is_perfect = IS_PERFECT
    if NO_QUANT:
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.backward.inp_res = -1
        rpu_config.backward.out_res = -1
    _apply_quant_bits(rpu_config, dac_bits, adc_bits)

    if BACKWARD_OUT_BOUND != 12.0:
        rpu_config.backward.out_bound = BACKWARD_OUT_BOUND

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


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
        return None  # Same as all (no bottleneck in BERT) (72 layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, qa_outputs)
        return None  # None means all encoder layers (72 layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create BERT QA model with selective LRTT analog layers.

    Architecture:
        - LRTT target layers (--lora-target) -> LRTT analog
        - Non-target encoder layers -> FROZEN (analog with --encoder-analog, the default;
          digital with --no-encoder-analog)
        - qa_outputs -> digital, TRAINABLE (weight + bias) with --head-layer train
        - LayerNorm -> digital, TRAINABLE
        - Embeddings -> digital, FROZEN

    LoRA target options (--lora-target):
        - qkv: Q/K/V layers -> LRTT analog (36 layers)
        - qkvo: Q/K/V and attention output layers -> LRTT analog (48 layers)
        - ffn: FFN layers -> LRTT analog (24 layers)
        - all: all encoder layers -> LRTT analog (72 layers)

    LRTT layers have:
        - A/B tiles: TRAINABLE (pulsed updates)
        - C tile: pretrained weights, changed only by transfers
        - out_scaling: TRAINABLE with --learn-out-scaling (off by default)
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
        analog_count = 0
    else:
        te = int(params["transfer_every"])
        lrtt_config = create_lrtt_config(
            rank=int(params["rank"]),
            transfer_every=te,
            transfer_lr=params["transfer_lr"],
            fast_lr=params["fast_lr"],
            reinit_mode=params["reinit_mode"],
            a_tau_sec=params["a_tau_sec"],
            b_tau_sec=params["b_tau_sec"],
            a_dw_min=params["a_dw_min"],
            b_dw_min=params["b_dw_min"],
            ab_desired_bl=params["ab_desired_bl"],
            a_desired_bl=params["a_desired_bl"],
            b_desired_bl=params["b_desired_bl"],
            a_multilevel=params["a_multilevel"],
            b_multilevel=params["b_multilevel"],
            a_reset_std=params["a_reset_std"],
            b_reset_std=params["b_reset_std"],
            c_dw_min=params["c_dw_min"],
            c_desired_bl=params["c_desired_bl"],
            out_noise=params["out_noise"],
            ab_weight_scaling_omega=params["ab_weight_scaling_omega"],
            a_weight_scaling_omega=params["a_weight_scaling_omega"],
            b_weight_scaling_omega=params["b_weight_scaling_omega"],
            auto_scale_mode=OPT_CONFIG['auto_scale_mode'],
            correct_gradient_magnitudes=OPT_CONFIG['correct_gradient_magnitudes'],
            transfer_rank_schedule=OPT_CONFIG['transfer_rank_schedule'],
            transfer_ranks_per_step=OPT_CONFIG['transfer_ranks_per_step'],
            scale_transfer_lr=OPT_CONFIG['scale_transfer_lr'],
            fi_continuous_alpha=OPT_CONFIG['fi_continuous_alpha'],
            lora_alpha=params["lora_alpha"],
            ab_pulse_type=OPT_CONFIG['ab_pulse_type'],
            dac_bits=params.get("dac_bits"),
            adc_bits=params.get("adc_bits"),
        )

        # Convert to analog with exclusions (only LRTT targets get converted).
        # Per-module IO bits: with LAYER_IO_BITS overrides, convert each bit-group in its
        # own pass (build-time bits - post-conversion override does NOT propagate to tiles).
        if not _any_layer_io_override():
            model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)
        else:
            import copy as _copy
            _groups = {}
            for _n in all_linear_names:
                if is_lrtt_target(_n):
                    _key = _resolve_io_bits(_n, params.get("dac_bits"), params.get("adc_bits"))
                    _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = _copy.deepcopy(lrtt_config)
                _apply_quant_bits(_cfg, _gd, _ga)
                _excl = [_n for _n in all_linear_names if _n not in _names]
                # inplace=True: from the 2nd group on, the model already holds LRTT tiles
                # from earlier groups; the default copy would rebuild them on CPU.
                model = convert_to_analog(model, _cfg, exclude_modules=_excl, inplace=True)

        # Count analog layers
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # Step 1.5: Convert remaining encoder layers to frozen analog (if enabled)
    # Already-converted LRTT layers (AnalogLinear) are naturally skipped by convert_to_analog
    frozen_analog_count = 0
    # In BERT "allnobn" is identical to "all" (no bottleneck layers): every encoder layer
    # is already an LRTT layer, so both skip the frozen-conversion pass.
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
            frozen_config = create_frozen_analog_config(
                _lc, out_noise=params.get("out_noise", 0.0),
                dac_bits=params.get("dac_bits"), adc_bits=params.get("adc_bits"),
            )
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
                _key = _resolve_io_bits(_n, params.get("dac_bits"), params.get("adc_bits"))
                _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = create_frozen_analog_config(
                    _lc, out_noise=params.get("out_noise", 0.0), dac_bits=_gd, adc_bits=_ga,
                )
                _excl = [_n for _n in all_linear_names if _n not in _names]
                # inplace=True required (see comment above): the model already holds LRTT
                # tiles here; a copy would break tile ids and CUDA placement.
                model = convert_to_analog(model, _cfg, exclude_modules=_excl, inplace=True)
        frozen_analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - analog_count

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
            "This disables LRTT learning entirely - check convert_to_analog inplace/copy."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LRTT Analog layers: {analog_count}, Frozen analog layers: {frozen_analog_count}")
    print(f"  Total params: {total_params:,}, Trainable (before grad set): {trainable_before:,}")

    # Step 2: Set requires_grad
    # - LRTT layers: A/B TRAINABLE; C tile bias and out scaling as set by lrtt_tile.py
    #   (train_c_bias, mapping_c.learn_out_scaling)
    # - qa_outputs: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - LayerNorm: TRAINABLE
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = not OPT_CONFIG['no_transfer']
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "out_scaling_alpha" in name:
            pass  # Frozen analog out scaling: a trainable parameter only with --learn-out-scaling
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ln_params = sum(p.numel() for n, p in model.named_parameters() if "LayerNorm" in n and p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,} (LayerNorm: {ln_params:,})")
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


# Per-trial gradient accumulation: train loaders of other micro-batch sizes, keyed by size.
_TRAIN_LOADER_CACHE = {}


def _loader_for_grad_accum(base_loader, ga):
    """Return a train DataLoader with batch_size == BATCH_SIZE // ga.

    Built over base_loader.dataset with the same arguments as the train loader of
    load_data. Returns base_loader itself when its batch_size already matches (the
    factor of --grad-accum-steps); other sizes are cached.
    """
    micro_bs = BATCH_SIZE // ga
    if base_loader.batch_size == micro_bs:
        return base_loader
    if micro_bs not in _TRAIN_LOADER_CACHE:
        _TRAIN_LOADER_CACHE[micro_bs] = DataLoader(
            base_loader.dataset, batch_size=micro_bs, shuffle=True,
            collate_fn=base_loader.collate_fn, num_workers=2,
            generator=torch.Generator().manual_seed(SEED)
        )
    return _TRAIN_LOADER_CACHE[micro_bs]


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
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
    """Evaluate SQuAD model. Returns (F1, EM)."""
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
# Scheduler
# =============================================================================

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
# Optuna Objective
# =============================================================================

def _ab_reset_std():
    """A/B reset_std: --ab-reset-std if given, else 0.01 for the 6t1c device and 1e-30 otherwise."""
    if AB_RESET_STD is not None:
        return AB_RESET_STD
    return 0.01 if AB_DEVICE == "6t1c" else 1e-30


def suggest_params(trial):
    """Draw the trial's hyperparameters.

    Returns (hp, params): hp holds the optimizer settings, params the LRTT,
    device and IO settings consumed by create_model(). Ranges come from
    SEARCH_SPACES[SEARCH_SPACE]; fixed values are suggested as single-value
    distributions so that they are recorded in the journal.
    """
    space = SEARCH_SPACES[SEARCH_SPACE]
    learning_rate = trial.suggest_float('learning_rate', *space['learning_rate'], log=True)

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1        # fixed (unused without transfer)
        transfer_every = 999999999
        rank_exp = 2             # fixed (A=0 init, no effect)
        rank = 4
        fast_lr = 1e-30          # ~0: without transfer the A/B tiles never reach C or the
                                 # forward pass, so the A/B pulse update is reduced to ~0 pulses
                                 # (faster, independent of ab_dw_min). PythonLRTTDevice
                                 # requires fast_lr > 0.
        a_tau_sec = b_tau_sec = 0.0    # fixed
    else:
        transfer_lr = trial.suggest_float('transfer_lr', *space['transfer_lr'], log=True)
        transfer_every = trial.suggest_int('transfer_every', *space['transfer_every'], log=True)
        if RANK_SWEEP:
            rank_exp = trial.suggest_int('rank_exp', 0, 6)
        else:
            rank_exp = trial.suggest_int('rank_exp', int(math.log2(RANK)), int(math.log2(RANK)))
        rank = 2 ** rank_exp
        fast_lr = trial.suggest_float('fast_lr', *space['fast_lr'], log=True)
        # tau_sec -> device.lifetime is per-tile splittable. Default shared.
        if SPLIT_AB_PARAMS.get('tau_sec', False):
            a_tau_sec = trial.suggest_float('a_tau_sec', 0, 0, log=False)
            b_tau_sec = trial.suggest_float('b_tau_sec', 0, 0, log=False)
        else:
            tau_sec = trial.suggest_float('tau_sec', 0, 0, log=False)  # 0 = no decay
            a_tau_sec = b_tau_sec = tau_sec

    # A/B device params
    # dw_min: per-tile splittable. SPLIT_AB_PARAMS['dw_min']=False sweeps a single
    # ab_dw_min and applies it to both; True sweeps a_dw_min and b_dw_min independently.
    if SPLIT_AB_PARAMS.get('dw_min', False):
        a_dw_min = trial.suggest_float('a_dw_min', *space['ab_dw_min'], log=True)
        b_dw_min = trial.suggest_float('b_dw_min', *space['ab_dw_min'], log=True)
    else:
        ab_dw_min = trial.suggest_float('ab_dw_min', *space['ab_dw_min'], log=True)
        a_dw_min = b_dw_min = ab_dw_min

    # desired_bl: per-tile splittable (a_desired_bl / b_desired_bl override the
    # rpu.update level desired_bl in aihwkit's create_update_params).
    if SPLIT_AB_PARAMS.get('desired_bl', False):
        a_desired_bl = trial.suggest_int('a_desired_bl', 31, 31)
        b_desired_bl = trial.suggest_int('b_desired_bl', 31, 31)
        ab_desired_bl = a_desired_bl  # placeholder for rpu.update.desired_bl base; A/B override
    else:
        ab_desired_bl = trial.suggest_int('ab_desired_bl', 31, 31)            # default: 31
        a_desired_bl = b_desired_bl = None  # None -> A and B inherit ab_desired_bl base

    # multilevel: per-tile splittable. w_max-w_min = 2^multilevel * dw_min (symmetric).
    # Enabled by default (--ab-multilevel), with the B init scaled accordingly;
    # --no-ab-multilevel keeps w_max=1.0.
    if OPT_CONFIG.get('ab_multilevel', False):
        if SPLIT_AB_PARAMS.get('multilevel', False):
            a_multilevel = trial.suggest_int('a_multilevel', AB_MULTILEVEL_BITS, AB_MULTILEVEL_BITS)
            b_multilevel = trial.suggest_int('b_multilevel', AB_MULTILEVEL_BITS, AB_MULTILEVEL_BITS)
        else:
            ab_multilevel = trial.suggest_int('ab_multilevel', AB_MULTILEVEL_BITS, AB_MULTILEVEL_BITS)
            a_multilevel = b_multilevel = ab_multilevel
    else:
        a_multilevel = b_multilevel = None

    # reset_std: per-tile splittable (noise of the device reset).
    reset_std = _ab_reset_std()
    if SPLIT_AB_PARAMS.get('reset_std', False):
        a_reset_std = trial.suggest_float('a_reset_std', reset_std, reset_std, log=True)
        b_reset_std = trial.suggest_float('b_reset_std', reset_std, reset_std, log=True)
    else:
        ab_reset_std = trial.suggest_float('ab_reset_std', reset_std, reset_std, log=True)
        a_reset_std = b_reset_std = ab_reset_std

    # C tile pulsed transfer params (only meaningful for onehot/direct)
    if TRANSFER_METHOD in ("onehot", "direct") and not OPT_CONFIG['no_transfer']:
        c_dw_min = trial.suggest_float('c_dw_min', C_DW_MIN, C_DW_MIN, log=True)
        c_desired_bl = trial.suggest_int('c_desired_bl', 31, 31)
    else:
        c_dw_min = 0.001   # default (unused for "set")
        c_desired_bl = None

    # IO non-idealities: fixed values recorded per trial (--out-noise, --dac-bits, --adc-bits).
    # --is-perfect makes the analog read ideal (all IO ignored) so nothing is recorded;
    # --no-io-noise / --no-quant disable output noise / DAC-ADC quantization individually.
    if IS_PERFECT or not IO_NOISE:
        out_noise = 0.0
    else:
        out_noise = trial.suggest_float('out_noise', OUT_NOISE, OUT_NOISE)
    if IS_PERFECT or NO_QUANT:
        dac_bits = 0
        adc_bits = 0
    else:
        dac_bits = trial.suggest_int('dac_bits', DAC_BITS, DAC_BITS)
        adc_bits = trial.suggest_int('adc_bits', ADC_BITS, ADC_BITS)
    # weight_scaling_omega: per-tile splittable via mapping_a / mapping_b override of
    # mapping_ab in aihwkit. Default shared.
    if SPLIT_AB_PARAMS.get('weight_scaling_omega', False):
        a_weight_scaling_omega = trial.suggest_float('a_weight_scaling_omega', 0.0, 0.0)
        b_weight_scaling_omega = trial.suggest_float('b_weight_scaling_omega', 0.0, 0.0)
        ab_weight_scaling_omega = a_weight_scaling_omega  # placeholder for the shared value
    else:
        ab_weight_scaling_omega = trial.suggest_float('ab_weight_scaling_omega', 0.0, 0.0)
        a_weight_scaling_omega = b_weight_scaling_omega = ab_weight_scaling_omega

    # lora_alpha: only sweep when forward_inject is ON and fi_continuous_alpha is OFF
    if FORWARD_INJECT and not OPT_CONFIG.get('fi_continuous_alpha', False):
        lora_alpha = trial.suggest_float('lora_alpha', 0.1, 10.0, log=True)
    else:
        lora_alpha = 1.0

    min_lr_rate = trial.suggest_float('min_lr_rate', 0.0, 0.0)

    # weight_decay: tune or fix to 0
    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    else:
        weight_decay = 0.0

    # momentum (AnalogSGD only): 0.9 with --momentum, 0.0 with --no-momentum (default)
    if OPT_CONFIG['tune_momentum']:
        momentum = 0.9
    else:
        momentum = 0.0

    # nesterov (AnalogSGD only): with --nesterov and momentum > 0; off by default
    if OPT_CONFIG['tune_nesterov'] and momentum > 0:
        nesterov = True
    else:
        nesterov = False

    # reinit_mode: use fixed value if set, otherwise tune
    if OPT_CONFIG['reinit_mode'] is not None:
        reinit_mode = OPT_CONFIG['reinit_mode']
    else:
        reinit_mode = trial.suggest_categorical('reinit_mode', ['standard', 'decay', 'hybrid'])

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    hp = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": nesterov,
        "min_lr_rate": min_lr_rate,
        "optimizer": optimizer_name,
    }
    params = {
        "rank": rank,
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "fast_lr": fast_lr,
        "reinit_mode": reinit_mode,
        "a_tau_sec": a_tau_sec,
        "b_tau_sec": b_tau_sec,
        "a_dw_min": a_dw_min,
        "b_dw_min": b_dw_min,
        "ab_desired_bl": ab_desired_bl,
        "a_desired_bl": a_desired_bl,
        "b_desired_bl": b_desired_bl,
        "a_multilevel": a_multilevel,
        "b_multilevel": b_multilevel,
        "a_reset_std": a_reset_std,
        "b_reset_std": b_reset_std,
        "c_dw_min": c_dw_min,
        "c_desired_bl": c_desired_bl,
        "out_noise": out_noise,
        "dac_bits": dac_bits,
        "adc_bits": adc_bits,
        "ab_weight_scaling_omega": ab_weight_scaling_omega,
        "a_weight_scaling_omega": a_weight_scaling_omega,
        "b_weight_scaling_omega": b_weight_scaling_omega,
        "lora_alpha": lora_alpha,
    }
    return hp, params


def objective(trial, train_loader, eval_features, eval_examples, tokenizer):
    # Per-trial gradient accumulation: a trial enqueued again after an out-of-memory error
    # carries its factor in user_attrs ("retry_grad_accum", set via enqueue_trial), so any
    # process that runs it uses that factor. Other trials use --grad-accum-steps and the
    # base train loader.
    global GRAD_ACCUM_STEPS
    GRAD_ACCUM_STEPS = int(trial.user_attrs.get("retry_grad_accum", _DEFAULT_GRAD_ACCUM))
    train_loader = _loader_for_grad_accum(train_loader, GRAD_ACCUM_STEPS)
    trial.set_user_attr("grad_accum_steps", GRAD_ACCUM_STEPS)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hp, params = suggest_params(trial)
    learning_rate, weight_decay = hp["learning_rate"], hp["weight_decay"]
    momentum, nesterov = hp["momentum"], hp["nesterov"]
    min_lr_rate, optimizer_name = hp["min_lr_rate"], hp["optimizer"]
    p = params

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  rank={p['rank']}, transfer_every={p['transfer_every']}, transfer_lr={p['transfer_lr']:.4e}")
    print(f"  fast_lr={p['fast_lr']:.2e}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, reinit_mode={p['reinit_mode']}")
    if p['a_tau_sec'] == p['b_tau_sec']:
        print(f"  tau_sec={p['a_tau_sec']:.1f}, optimizer={optimizer_name}, min_lr_rate={min_lr_rate:.4f}")
    else:
        print(f"  a_tau_sec={p['a_tau_sec']:.1f}, b_tau_sec={p['b_tau_sec']:.1f}, optimizer={optimizer_name}, min_lr_rate={min_lr_rate:.4f}")
    if p['a_dw_min'] == p['b_dw_min']:
        _ml_str = f", multilevel={p['a_multilevel']}" if p['a_multilevel'] is not None else ", multilevel=off"
        print(f"  ab_dw_min={p['a_dw_min']:.4e}, ab_desired_bl={p['ab_desired_bl']}{_ml_str}")
    else:
        print(f"  a_dw_min={p['a_dw_min']:.4e}, b_dw_min={p['b_dw_min']:.4e}, ab_desired_bl={p['ab_desired_bl']}, multilevel a/b={p['a_multilevel']}/{p['b_multilevel']}")
    print(f"  a_reset_std={p['a_reset_std']:.4e}, b_reset_std={p['b_reset_std']:.4e}")
    if TRANSFER_METHOD in ("onehot", "direct"):
        print(f"  c_dw_min={p['c_dw_min']:.4e}, c_desired_bl={p['c_desired_bl']}")
    print(f"  IO: out_noise={p['out_noise']:g}, dac_bits={p['dac_bits']}, adc_bits={p['adc_bits']}, is_perfect={IS_PERFECT}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

        if LORA_TARGET == "none" and not ENCODER_ANALOG and not HEAD_ANALOG:
            # None mode (no analog tiles): use standard PyTorch optimizers
            if optimizer_name == "AnalogSGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
        else:
            # Analog optimizers: required for LRTT tiles and frozen analog tiles
            # (AnalogSGD/Adam calls analog_ctx.reset() to prevent memory leak)
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = AnalogAdam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
            optimizer.regroup_param_groups()
            optimizer._grad_accum_steps = GRAD_ACCUM_STEPS
            trial.set_user_attr("grad_accum_steps", GRAD_ACCUM_STEPS)

        num_training_steps = len(train_loader) * N_EPOCHS // GRAD_ACCUM_STEPS
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_f1 = 0.0
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        train_loss_no_improvement = 0

        _nan_detected = False
        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
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
                if torch.isnan(loss):
                    tqdm.write(f"[Trial {trial.number}] NaN loss at epoch {epoch}. Aborting.")
                    _nan_detected = True
                    break
                loss.backward()

                if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * GRAD_ACCUM_STEPS
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            if _nan_detected:
                break

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)

            improved = ""
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            train_loss_improved = ""
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                train_loss_no_improvement = 0
                train_loss_improved = " (lower)"
            else:
                train_loss_no_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}% | Best F1: {best_f1:6.2f}% | "
                  f"Train loss: {train_loss:.4f}{train_loss_improved} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_f1, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            if best_train_loss > TRAIN_LOSS_THRESHOLD and train_loss_no_improvement >= TRAIN_LOSS_EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Train loss early stop at epoch {epoch} "
                          f"(train_loss={train_loss:.4f} > {TRAIN_LOSS_THRESHOLD}, no improvement for {train_loss_no_improvement} epochs)")
                break

            if best_train_loss <= TRAIN_LOSS_THRESHOLD and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best F1: {best_f1:.2f}%")
        print(f"{'='*70}\n")
        return best_f1

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        # Delete training loop vars to release model refs for C++ destructor
        try:
            del outputs
        except NameError:
            pass
        try:
            del loss
        except NameError:
            pass
        try:
            del pbar
        except NameError:
            pass
        try:
            del batch
        except NameError:
            pass
        try:
            del input_ids
        except NameError:
            pass
        try:
            del attention_mask
        except NameError:
            pass
        try:
            del start_positions
        except NameError:
            pass
        try:
            del end_positions
        except NameError:
            pass
        # Delete in reverse dependency order: scheduler -> optimizer -> model
        # optimizer holds references to analog tiles via param_groups
        if 'scheduler' in dir():
            del scheduler
        if 'optimizer' in dir():
            del optimizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tqdm.write(f"[Trial {trial.number}] GPU cache cleared")


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs F1."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    f1_scores = [t.value for t in complete_trials]

    # Optimization history
    axes[0].scatter(trial_numbers, f1_scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(f1_scores[:i+1]) for i in range(len(f1_scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('F1 (%)')
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    # LR vs F1
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, f1_scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('F1 (%)')
    axes[2].set_title('Learning Rate vs F1')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        f1_scores = [t.value for t in complete_trials]
        print(f"Best F1: {max(f1_scores):.2f}%, Mean F1: {sum(f1_scores)/len(f1_scores):.2f}%")
        print(f"Best params: {study.best_params}")


class _SpaceRecorder:
    """Stand-in for an Optuna trial that records the suggest calls (--print-config)."""

    def __init__(self):
        self.space = {}

    def suggest_float(self, name, low, high, log=False, step=None):
        self.space[name] = (low, high, log)
        return low

    def suggest_int(self, name, low, high, log=False, step=1):
        self.space[name] = (int(low), int(high), log)
        return int(low)

    def suggest_categorical(self, name, choices):
        self.space[name] = (list(choices),)
        return choices[0]


def print_config(study_name):
    """Print the resolved configuration and the search space of one trial."""
    print(f"study_name: {study_name}")
    print(f"journal: {os.path.join(RESULTS, f'optuna_{study_name}.log')}")
    print(f"model: {MODEL_NAME}  max_seq_length: {MAX_SEQ_LENGTH}  seed: {SEED}")
    print(f"batch_size: {BATCH_SIZE}  grad_accum_steps: {GRAD_ACCUM_STEPS}  epochs: {N_EPOCHS}  "
          f"warmup_steps: {WARMUP_STEPS}  early_stop_patience: {EARLY_STOP_PATIENCE}")
    print(f"transfer_method: {TRANSFER_METHOD}  ab_device: {AB_DEVICE}  a_device: {A_DEVICE}  "
          f"b_device: {B_DEVICE}  c_device: {C_DEVICE}")
    print(f"lora_target: {LORA_TARGET}  head_layer: {HEAD_LAYER}  encoder_analog: {ENCODER_ANALOG}  "
          f"head_analog: {HEAD_ANALOG}")
    print(f"io_noise: {IO_NOISE}  forward_inject: {FORWARD_INJECT}  is_perfect: {IS_PERFECT}  "
          f"no_quant: {NO_QUANT}  backward_out_bound: {BACKWARD_OUT_BOUND}")
    print(f"rank: {'1..64 (rank_exp 0..6)' if RANK_SWEEP else RANK}  search_space: {SEARCH_SPACE}")
    for key, value in OPT_CONFIG.items():
        print(f"{key}: {value}")
    print(f"split_ab_params: {sorted(k for k, v in SPLIT_AB_PARAMS.items() if v) or 'none'}")
    if _any_layer_io_override():
        print(f"layer_io_bits: {LAYER_IO_BITS}")
    recorder = _SpaceRecorder()
    suggest_params(recorder)
    print("search space (fixed values are recorded as single-value distributions):")
    for name, spec in recorder.space.items():
        if len(spec) == 1:
            print(f"  {name}: categorical {spec[0]}")
        elif spec[0] == spec[1]:
            print(f"  {name}: {spec[0]}")
        else:
            print(f"  {name}: [{spec[0]}, {spec[1]}]{' log' if spec[2] else ''}")


# =============================================================================
# Main
# =============================================================================

def main():
    global RESULTS, BATCH_SIZE, GRAD_ACCUM_STEPS, N_EPOCHS, WARMUP_STEPS, TRANSFER_METHOD, AB_DEVICE, A_DEVICE, B_DEVICE, C_DEVICE, IO_NOISE, FORWARD_INJECT, IS_PERFECT, NO_QUANT, LORA_TARGET, HEAD_LAYER, ENCODER_ANALOG, HEAD_ANALOG, BACKWARD_OUT_BOUND, _DEFAULT_GRAD_ACCUM
    global SEARCH_SPACE, RANK, RANK_SWEEP, AB_MULTILEVEL_BITS, C_DW_MIN, AB_RESET_STD, OUT_NOISE, DAC_BITS, ADC_BITS

    ab_devices = ['6t1c', 'linearstep', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'fp', 'ideal']
    parser = argparse.ArgumentParser(
        description="Optuna search for LR-TT fine-tuning of BERT-base on SQuAD v1.1 "
                    "(defaults: the paper's common configuration)")

    def add_switch(on, off, dest, default, help_on, help_off):
        """--on / --off flag pair writing one boolean destination."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f'--{on}', dest=dest, action='store_true',
                           help=help_on + (' (default)' if default else ''))
        group.add_argument(f'--{off}', dest=dest, action='store_false',
                           help=help_off + ('' if default else ' (default)'))
        parser.set_defaults(**{dest: default})

    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: derived from the configuration)')
    parser.add_argument('--results-dir', type=str, default=RESULTS,
                        help='Directory for the journal, best-params file and plots '
                             '(default: ./results/optuna_bert_squad_lrtt)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true',
                        help='Summarize and plot an existing study, then exit')
    parser.add_argument('--print-config', action='store_true',
                        help='Print the resolved configuration and search space, then exit')
    parser.add_argument('--optimizer', type=str, default=OPT_CONFIG['optimizer'],
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help=f"Optimizer type (default: {OPT_CONFIG['optimizer']})")
    add_switch('tune-wd', 'no-wd', 'tune_wd', OPT_CONFIG['tune_wd'],
               'Tune the weight decay (1e-7..1e-2, log)', 'Fix the weight decay to 0')
    add_switch('momentum', 'no-momentum', 'tune_momentum', OPT_CONFIG['tune_momentum'],
               'Momentum 0.9 (SGD only)', 'Momentum 0')
    add_switch('nesterov', 'no-nesterov', 'tune_nesterov', OPT_CONFIG['tune_nesterov'],
               'Nesterov momentum (SGD only)', 'No Nesterov momentum')
    parser.add_argument('--reinit-mode', type=str, default=OPT_CONFIG['reinit_mode'],
                        choices=['tune', 'standard', 'decay', 'hybrid'],
                        help=f"Post-transfer reinit mode, or 'tune' to search among "
                             f"standard/decay/hybrid (default: {OPT_CONFIG['reinit_mode']})")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--grad-accum-steps', type=int, default=GRAD_ACCUM_STEPS,
                        help=f'Gradient accumulation steps (default: {GRAD_ACCUM_STEPS})')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                        help=f'LR warmup steps (default: {WARMUP_STEPS})')
    parser.add_argument('--transfer-method', type=str, default=TRANSFER_METHOD,
                        choices=['onehot', 'direct', 'set'],
                        help=f'Transfer method (default: {TRANSFER_METHOD})')
    parser.add_argument('--ab-device', type=str, default=AB_DEVICE, choices=ab_devices,
                        help=f'A/B tile device type (default: {AB_DEVICE}). Used for both A and B unless --a-device or --b-device overrides.')
    parser.add_argument('--a-device', type=str, default=None, choices=ab_devices,
                        help='Override A tile device only (default: same as --ab-device)')
    parser.add_argument('--b-device', type=str, default=None, choices=ab_devices,
                        help='Override B tile device only (default: same as --ab-device)')
    parser.add_argument('--c-device', type=str, default=C_DEVICE,
                        choices=['softboundsideal', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'ideal'],
                        help=f'C tile device type (default: {C_DEVICE})')
    parser.add_argument('--rank', type=int, default=RANK,
                        help=f'LR-TT rank, a power of two (default: {RANK})')
    parser.add_argument('--rank-sweep', action='store_true',
                        help="Search rank_exp in 0..6 (rank 1..64) with the ranges of the rank "
                             "sweep (--search-space rank) instead of fixing --rank")
    parser.add_argument('--search-space', type=str, default=None, choices=sorted(SEARCH_SPACES),
                        help="Suggest ranges of the paper's studies, listed in README_optuna.md "
                             "(default: 'rank' with --rank-sweep, else 'paper')")
    parser.add_argument('--ab-bits', type=int, default=AB_MULTILEVEL_BITS,
                        help=f'A/B multi-level bits, w_max - w_min = 2^bits * ab_dw_min (default: {AB_MULTILEVEL_BITS})')
    parser.add_argument('--c-dw-min', type=float, default=C_DW_MIN,
                        help=f'C tile dw_min (default: {C_DW_MIN}, 10-bit core)')
    parser.add_argument('--ab-reset-std', type=float, default=AB_RESET_STD,
                        help='A/B reset_std (default: 0.01 for --ab-device 6t1c, 1e-30 otherwise)')
    parser.add_argument('--out-noise', type=float, default=OUT_NOISE,
                        help=f'Output noise of the analog forward/backward passes (default: {OUT_NOISE})')
    parser.add_argument('--dac-bits', type=int, default=DAC_BITS,
                        help=f'DAC (input) resolution in bits (default: {DAC_BITS})')
    parser.add_argument('--adc-bits', type=int, default=ADC_BITS,
                        help=f'ADC (output) resolution in bits (default: {ADC_BITS})')
    add_switch('io-noise', 'no-io-noise', 'io_noise', IO_NOISE,
               'Apply the IO output noise', 'Disable IO out_noise (resolution kept)')
    parser.add_argument('--forward-inject', action='store_true',
                        help='Add the auxiliary A*B product to the forward pass (W_eff = C + alpha*A*B)')
    parser.add_argument('--is-perfect', action='store_true',
                        help='Use ideal FP forward/backward (no ADC/DAC/noise)')
    parser.add_argument('--no-quant', action='store_true',
                        help='Disable DAC/ADC quantization (inp_res/out_res)')
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--ab-io-perfect', action='store_true',
                        help='Make A/B tiles fully ideal (no out_noise/ADC/DAC) - digital adapter model')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'qkvo', 'ffn', 'dense', 'allnobn', 'all'],
                        help=f'LR-TT target layers (default: {LORA_TARGET})')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='qa_outputs layer: train or freeze (default: train)')
    add_switch('encoder-analog', 'no-encoder-analog', 'encoder_analog', ENCODER_ANALOG,
               'Map non-LRTT encoder layers to frozen analog tiles', 'Keep non-LRTT encoder layers digital')
    add_switch('head-analog', 'no-head-analog', 'head_analog', HEAD_ANALOG,
               'Map qa_outputs to a frozen analog tile', 'Keep qa_outputs digital')
    parser.add_argument('--backward-out-bound', type=float, default=BACKWARD_OUT_BOUND,
                        help=f'Backward pass output bound (default: {BACKWARD_OUT_BOUND})')
    add_switch('learn-out-scaling', 'no-learn-out-scaling', 'learn_out_scaling', OPT_CONFIG['learn_out_scaling'],
               'Trainable out_scaling on the C tile', 'Fixed out_scaling on the C tile')
    parser.add_argument('--auto-scale-mode', type=str, default='none',
                        choices=['none', 'shared', 'separate'],
                        help='Auto-scale mode for A/B LR normalization (default: none)')
    parser.add_argument('--correct-gradient-magnitudes', action='store_true',
                        help='Correct transfer magnitude by dividing by effective A/B LR')
    parser.add_argument('--transfer-rank-schedule', type=str, default='all',
                        choices=['all', 'round_robin'],
                        help='Transfer rank schedule (default: all)')
    parser.add_argument('--transfer-ranks-per-step', type=int, default=1,
                        help='Number of ranks per transfer step in round_robin mode (default: 1)')
    add_switch('scale-transfer-lr', 'no-scale-transfer-lr', 'scale_transfer_lr', OPT_CONFIG['scale_transfer_lr'],
               'Scale the transfer LR by the scheduled optimizer LR', 'Constant transfer LR')
    add_switch('ab-multilevel', 'no-ab-multilevel', 'ab_multilevel', OPT_CONFIG['ab_multilevel'],
               'A/B w_max - w_min = 2^(--ab-bits) * ab_dw_min (B init scales accordingly)', 'A/B w_max = 1.0')
    parser.add_argument('--ab-pulse-type', type=str, default='default',
                        choices=['default', 'none', 'none_with_device', 'stochastic_compressed', 'mean_count', 'deterministic_implicit'],
                        help='Pulse type for A/B tile updates (default: use RPUConfig default)')
    parser.add_argument('--fi-continuous-alpha', action='store_true',
                        help='Use transfer LR as forward-injection alpha (continuity condition)')
    args = parser.parse_args()
    if args.rank < 1 or args.rank & (args.rank - 1):
        parser.error('--rank must be a power of two')

    # Update global config
    RESULTS = os.path.abspath(args.results_dir)
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    _DEFAULT_GRAD_ACCUM = GRAD_ACCUM_STEPS  # per-trial default ga (non-retry trials)
    N_EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    A_DEVICE = args.a_device
    B_DEVICE = args.b_device
    C_DEVICE = args.c_device
    RANK = args.rank
    RANK_SWEEP = args.rank_sweep
    SEARCH_SPACE = args.search_space or ('rank' if RANK_SWEEP else 'paper')
    AB_MULTILEVEL_BITS = args.ab_bits
    C_DW_MIN = args.c_dw_min
    AB_RESET_STD = args.ab_reset_std
    OUT_NOISE = args.out_noise
    DAC_BITS = args.dac_bits
    ADC_BITS = args.adc_bits
    IO_NOISE = args.io_noise
    FORWARD_INJECT = args.forward_inject
    IS_PERFECT = args.is_perfect
    NO_QUANT = args.no_quant
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = None if args.reinit_mode == 'tune' else args.reinit_mode
    OPT_CONFIG['tune_wd'] = args.tune_wd
    OPT_CONFIG['tune_momentum'] = args.tune_momentum
    OPT_CONFIG['tune_nesterov'] = args.tune_nesterov
    OPT_CONFIG['no_transfer'] = args.no_transfer
    OPT_CONFIG['ab_io_perfect'] = args.ab_io_perfect
    OPT_CONFIG['learn_out_scaling'] = args.learn_out_scaling
    OPT_CONFIG['auto_scale_mode'] = args.auto_scale_mode
    OPT_CONFIG['correct_gradient_magnitudes'] = args.correct_gradient_magnitudes
    OPT_CONFIG['transfer_rank_schedule'] = args.transfer_rank_schedule
    OPT_CONFIG['transfer_ranks_per_step'] = args.transfer_ranks_per_step
    OPT_CONFIG['scale_transfer_lr'] = args.scale_transfer_lr
    OPT_CONFIG['fi_continuous_alpha'] = args.fi_continuous_alpha
    OPT_CONFIG['ab_pulse_type'] = args.ab_pulse_type
    OPT_CONFIG['ab_multilevel'] = args.ab_multilevel
    ENCODER_ANALOG = args.encoder_analog
    HEAD_ANALOG = args.head_analog
    BACKWARD_OUT_BOUND = args.backward_out_bound

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"bert_squad_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    if args.print_config:
        print_config(study_name)
        return

    os.makedirs(RESULTS, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  Study: {study_name}")
    print(f"  Log  : {RESULTS}/optuna_{study_name}.log")
    print(f"{'='*70}\n")

    storage = JournalStorage(JournalFileBackend(f"{RESULTS}/optuna_{study_name}.log"))

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(n_startup_trials=10,
                                          consider_running_trials=True),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    # Report whether the study is new or extends an existing journal
    existing_trials = study.trials
    n_complete = sum(1 for t in existing_trials if t.state == TrialState.COMPLETE)
    if not existing_trials:
        print(f"[WARNING] NEW STUDY created (no existing trials). "
              f"Verify the study name above matches your intent.\n")
    else:
        print(f"Loaded existing study: {len(existing_trials)} trials total, {n_complete} complete.\n")

    # Trials left in the RUNNING state by a terminated process are marked FAIL. Only the
    # first process of a study in a results directory does this; it creates the lock file
    # _zombie_cleanup_<study>.lock, so later processes leave the RUNNING trials of other
    # processes untouched.
    zombie_lock = os.path.join(RESULTS, f"_zombie_cleanup_{study_name}.lock")
    if not os.path.exists(zombie_lock):
        n_zombie = 0
        for t in existing_trials:
            if t.state == TrialState.RUNNING:
                study.tell(t.number, state=TrialState.FAIL)
                n_zombie += 1
        if n_zombie > 0:
            print(f"[Cleanup] Marked {n_zombie} zombie RUNNING trial(s) as FAIL.")
        # Create lock so subsequent workers skip cleanup
        with open(zombie_lock, 'w') as f:
            f.write(str(os.getpid()))

    print(f"Study: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    initial_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

    def _restart_with_remaining(remaining):
        """execv to thin wrapper (no CUDA) that spawns child + forwards Ctrl+C."""
        new_argv = list(sys.argv)
        for i, arg in enumerate(new_argv):
            if arg == '--n-trials' and i + 1 < len(new_argv):
                new_argv[i + 1] = str(remaining)
                break
        child_cmd = [sys.executable] + new_argv
        wrapper = (
            'import subprocess,signal,sys\n'
            f'p=subprocess.Popen({child_cmd!r})\n'
            'def _fwd(s,f):\n'
            ' try: p.send_signal(s)\n'
            ' except: pass\n'
            'signal.signal(signal.SIGINT,_fwd)\n'
            'sys.exit(p.wait())\n'
        )
        os.execv(sys.executable, [sys.executable, '-c', wrapper])

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_features, eval_examples, tokenizer),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        current_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = max(1, args.n_trials - (current_complete - initial_complete))
        print(f"\n[OOM Recovery] Restarting process for {remaining} remaining trials...")
        _restart_with_remaining(remaining)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    # Save best params
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "best_f1": study.best_value,
                "best_params": study.best_params,
            }, f, indent=2)
        print(f"Best params saved to: {best_params_file}")

    # Save all trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -1, reverse=True)

    all_trials_file = os.path.join(RESULTS, "all_trials.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(BaseException):
    """Raised to trigger process restart after OOM.

    Inherits from BaseException (not Exception) so that
    study.optimize(catch=(Exception,)) does not swallow it.
    """
    pass


def _oom_restart_callback(study, trial):
    """Optuna callback: on OOM, enqueue a retry carrying its ga (user_attrs) and restart process."""
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        is_oom = any(k in err.lower() for k in ("out of memory", "cublas", "cudacachingallocator"))
        is_cuda_assert = any(k in err.lower() for k in ("nvml", "internal assert failed")) and not is_oom
        # ga the failed trial actually ran with (per-trial, set in objective()), not this process's global.
        trial_ga = int(trial.user_attrs.get("grad_accum_steps",
                                            trial.user_attrs.get("retry_grad_accum", GRAD_ACCUM_STEPS)))
        if is_oom:
            # Pick the next divisor of BATCH_SIZE larger than the trial's ga
            # so that micro_bs = BATCH_SIZE // new_grad_accum is always exact.
            divisors = sorted(d for d in range(1, BATCH_SIZE + 1) if BATCH_SIZE % d == 0)
            larger = [d for d in divisors if d > trial_ga]
            if not larger:
                print(f"\n[OOM Recovery] Cannot reduce micro-batch below 1 "
                      f"(BATCH_SIZE={BATCH_SIZE}, already at max GRAD_ACCUM={trial_ga}). "
                      f"Skipping retry.")
                return
            new_grad_accum = larger[0]
            micro_bs = BATCH_SIZE // new_grad_accum

            # The factor is stored with the enqueued trial, so the process that runs it
            # uses this factor (objective() reads retry_grad_accum).
            study.enqueue_trial(dict(trial.params),
                                user_attrs={"retry_grad_accum": new_grad_accum, "retry_of": trial.number})

            print(f"\n[OOM Recovery] Trial {trial.number} OOM. "
                  f"Enqueued retry with grad_accum={new_grad_accum}, micro_bs={micro_bs}; restarting process.")
            raise _OOMRestart()
        elif is_cuda_assert:
            # Other CUDA runtime error (NVML or internal assert): enqueue the trial again
            # with the same factor and continue in a new process.
            study.enqueue_trial(dict(trial.params),
                                user_attrs={"retry_grad_accum": trial_ga, "retry_of": trial.number})
            print(f"\n[Retry] Trial {trial.number}: CUDA error, enqueued again "
                  f"(ga={trial_ga}); restarting the process.")
            raise _OOMRestart()


if __name__ == "__main__":
    main()
