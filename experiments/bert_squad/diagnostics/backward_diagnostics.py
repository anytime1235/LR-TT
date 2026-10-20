# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""backward_diagnostics.py: backward-DAC diagnostics of BERT-base on SQuAD v1.1

Captures the backward gradients of the Q/K/V/O/FFN1/FFN2 analog layers
(DAC 7 bit, ADC 9 bit; learning rate 0) and writes Figure A, the per-layer
QZR/ODR root cause of the 6 sublayers (2x3 panels). Its summary CSV
(metrics_paper_A_rootcause_summary.csv) holds the QZR and ODR values of
Fig. 5a and 5b, drawn by plot_qzr_odr_maps.py.

Details:
  - zero_thresh = inp_res = 1/(2^b-2): elements with |dy|/absmax below it
    round to zero
  - step_size = 2 * inp_bound * res_ratio (as in AIHWKit UniformQuantize)
  - l2_retention, rel_l2_error, clip_rate_scaled metrics
  - ratio reservoir sampling (200k cap) to bound memory use
  - CSV outputs: summary and per-step files

Usage:
  python backward_diagnostics.py                               # full run
  python backward_diagnostics.py --n-step 5 --batch-size 2 --out-dir ./results/smoke
"""

import argparse
import gc
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, set_seed
from transformers import default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="Backward-DAC diagnostics of BERT-base at DAC 7 / ADC 9 bit (Figure A: QZR/ODR of Fig. 5a and 5b)")
parser.add_argument("--n-step",       type=int, default=200,
                    help="diagnostic steps (default: 200)")
parser.add_argument("--batch-size",   type=int, default=8)
parser.add_argument("--out-dir",      type=str, default="./results/backward_diagnostics")
parser.add_argument("--run-tag",      type=str, default="v3",
                    help="label written to the run_tag column of the CSVs (default: v3)")
args = parser.parse_args()

# =============================================================================
# Constants & Paths
# =============================================================================

N_STEP       = args.n_step
BATCH_SIZE   = args.batch_size
RUN_TAG      = args.run_tag

OUT_DIR = args.out_dir

MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
SEED           = 42
INP_BOUND      = 1.0
# DAC 7 / ADC 9 bits (inp_res = 1/126, out_res = 1/510)
DAC_BITS       = 7
ADC_BITS       = 9
N_LAYERS       = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output files
CSV_A_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_summary.csv")
CSV_A_STEPS   = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_steps.csv")
FIG_A = os.path.join(OUT_DIR, "fig_paper_A_rootcause_qkvo_ffn.png")

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, "
      f"BATCH={BATCH_SIZE}, RUN_TAG={RUN_TAG}")
print(f"[Config] OUT_DIR={OUT_DIR}")

# =============================================================================
# Layer Name Utilities
# =============================================================================

# attention.output.dense must appear before output.dense to prevent collision
_LAYER_RE = re.compile(
    r"encoder\.layer\.(\d+)\."
    r"(attention\.self\.query|attention\.self\.key|attention\.self\.value"
    r"|attention\.output\.dense|intermediate\.dense|output\.dense)"
)
_SUBLAYER_MAP = {
    "attention.self.query":   "Q",
    "attention.self.key":     "K",
    "attention.self.value":   "V",
    "attention.output.dense": "O",
    "intermediate.dense":     "FFN1",
    "output.dense":           "FFN2",
}
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]


def parse_layer_name(name: str):
    """Returns (layer_idx: int, sublayer: str) or None."""
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


def _layer_names(model):
    """Split encoder linear layers into target (QKVO) and nontarget (FFN)."""
    always_digital = ["qa_outputs", "pooler"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    target = [
        n for n in all_linear
        if "encoder" in n and "attention" in n
        and not any(d in n for d in always_digital)
    ]
    nontarget = [
        n for n in all_linear
        if "encoder" in n and n not in target
        and not any(d in n for d in always_digital)
    ]
    return target, nontarget, all_linear


# =============================================================================
# RPU Config (adc_bits parameterized)
# =============================================================================

def create_rpu_config(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS, adc_bits=ADC_BITS):
    """SingleRPU with noise-free SoftBoundsDevice. inp_res = 1/(2^dac_bits - 2).

    No output noise; digital bias and column-wise weight scaling (omega 1.0).
    """
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import SoftBoundsDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType

    device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )
    rpu = SingleRPUConfig(device=device)
    for io in [rpu.forward, rpu.backward]:
        io.inp_bound        = INP_BOUND
        io.inp_res          = 1.0 / (2**dac_bits - 2)
        io.out_bound        = 12.0
        io.out_res          = 1.0 / (2**adc_bits - 2)
        io.noise_management = NoiseManagementType.ABS_MAX
        io.out_noise        = 0.0
        io.inp_sto_round    = sto_round
    rpu.backward.nm_thres               = nm_thres
    rpu.mapping.digital_bias            = True
    rpu.mapping.weight_scaling_omega    = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu


# =============================================================================
# Model Creation
# =============================================================================

def create_model(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS, adc_bits=ADC_BITS):
    """BERT-base 2-pass analog conversion.

    Pass 1: Q/K/V/O (target) - analog, weight updates enabled (lr=0 so noop)
    Pass 2: FFN (nontarget)  - analog, tile.update = _noop (frozen)

    All AnalogContext.requires_grad = True -> backward flows through FFN tiles
    -> register_full_backward_hook fires on FFN layers too.
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    target, nontarget, all_linear = _layer_names(model)

    # Pass 1: target (QKVO)
    rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                            dac_bits=dac_bits, adc_bits=adc_bits)
    model = convert_to_analog(
        model, rpu,
        exclude_modules=[n for n in all_linear if n not in target]
    )

    # Pass 2: nontarget (FFN) - same config, update will be nooped
    nt_rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                               dac_bits=dac_bits, adc_bits=adc_bits)
    model = convert_to_analog(
        model, nt_rpu,
        exclude_modules=[n for n in all_linear if n not in nontarget]
    )

    # Freeze nontarget tile updates
    def _noop(x, d, *a, **kw):
        return None

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target:
            for tile in m.analog_tiles():
                tile.update = _noop

    # Gradient flow: disable all, then re-enable AnalogContext + qa_outputs
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)

    n_t   = sum(1 for n, m in model.named_modules()
                if isinstance(m, AnalogLinear) and n in target)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles - target(QKVO):{n_t}, frozen(FFN):{n_all - n_t}, "
          f"nm_thres={nm_thres:.4g}, sto_round={sto_round}, "
          f"dac={dac_bits}b, adc={adc_bits}b")
    return model.to(DEVICE)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(tokenizer, n_step, batch_size):
    """SQuAD v1.1 - first n_step batches. Seed-fixed for reproducibility."""

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map     = inputs.pop("overflow_to_sample_mapping")
        answers        = examples["answers"]
        sp, ep = [], []
        for i, offset in enumerate(offset_mapping):
            ans = answers[sample_map[i]]
            if not ans["answer_start"]:
                sp.append(0); ep.append(0); continue
            sc = ans["answer_start"][0]
            ec = sc + len(ans["text"][0])
            seq = inputs.sequence_ids(i)
            idx = 0
            while seq[idx] != 1:
                idx += 1
            cs = idx
            while idx < len(seq) and seq[idx] == 1:
                idx += 1
            ce = idx - 1
            if offset[cs][0] > ec or offset[ce][1] < sc:
                sp.append(0); ep.append(0)
            else:
                idx = cs
                while idx <= ce and offset[idx][0] <= sc:
                    idx += 1
                sp.append(idx - 1)
                idx = ce
                while idx >= cs and offset[idx][1] >= ec:
                    idx -= 1
                ep.append(idx + 1)
        inputs["start_positions"] = sp
        inputs["end_positions"]   = ep
        return inputs

    raw = load_dataset("squad")
    tok = raw["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw["train"].column_names
    )
    n = min(n_step * batch_size, len(tok))
    subset = tok.shuffle(seed=SEED).select(range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"  Dataset: {n} samples -> {len(loader)} batches")
    return loader


# =============================================================================
# MaskBuffer
# =============================================================================

class MaskBuffer:
    """Captures attention_mask from each forward pass for padding exclusion."""

    def __init__(self):
        self.val = None

    def register(self, model):
        def _hook(mod, args, kwargs):
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                self.val = kwargs["attention_mask"].bool().cpu()
        model.register_forward_pre_hook(_hook, with_kwargs=True)


# =============================================================================
# LayerStats - AIHWKit-consistent quantization + extended metrics
# =============================================================================

class LayerStats:
    """Per-layer backward gradient statistics accumulator.

    zero_thresh = inp_res = 1/(2^b - 2)
    step_size = 2 * INP_BOUND * res_ratio  (AIHWKit UniformQuantize consistent)
    nm_thres > 0 -> alpha = min(absmax, nm_thres) simulating tile nm_thres cap.
    """

    def __init__(self, name: str, layer_idx: int, sublayer: str,
                 inp_res: float, nm_thres: float = 0.0,
                 store_sweep: bool = True):
        self.name        = name
        self.layer_idx   = layer_idx
        self.sublayer    = sublayer
        self.inp_res     = inp_res
        self.nm_thres    = nm_thres
        self.zero_thresh = inp_res   # threshold: ratio < zero_thresh -> quant 0
        self.eps         = 1e-8

        # AIHWKit-consistent quantization parameters
        self.res_ratio  = inp_res if inp_res <= 1.0 else 1.0 / inp_res
        self.step_size  = 2.0 * INP_BOUND * self.res_ratio

        # per-step scalar lists (length grows with steps)
        self.ezr_steps       = []
        self.qzr_all_steps   = []
        self.qzr_nz_steps    = []
        self.odr_steps       = []
        self.cosine_steps    = []
        self.ratio_q50_steps = []
        self.ratio_q90_steps = []
        self.ratio_q99_steps = []

        # clipping and absmax quantiles
        self.p_clip_steps      = []   # P(|delta| > INP_BOUND) element-wise
        self.absmax_q50_steps  = []   # 50th percentile of per-vector absmax
        self.absmax_q90_steps  = []   # 90th
        self.absmax_q99_steps  = []   # 99th
        self.absmax_q999_steps = []   # 99.9th

        # l2 fidelity and clip-rate metrics
        self.l2_retention_steps      = []
        self.rel_l2_error_steps      = []
        self.clip_rate_scaled_steps  = []
        self.n_vec_steps             = []

        # raw per-vector absmax buffer for ECDF (always collected)
        self._absmax_buf = []

        # ratio reservoir sampling
        self._store_sweep = store_sweep
        self._ratio_reservoir_max = 200_000
        self._ratio_reservoir = []
        self._ratio_reservoir_count = 0

        # EZR and QZR_all with padding positions included
        self.ezr_pad_steps     = []
        self.qzr_all_pad_steps = []

    def update(self, dy: torch.Tensor, mask=None):
        """Called from backward hook. dy = grad_output[0]."""
        with torch.no_grad():
            dy = dy.detach().float()
            # padding-included
            dy_flat_pad = dy.reshape(-1, dy.shape[-1])
            self._compute_pad(dy_flat_pad)
            # mask-excluded
            if mask is not None and dy.dim() == 3:
                B, S, D = dy.shape
                if mask.shape == (B, S):
                    mask_dev = mask.to(dy.device)
                    dy_real  = dy[mask_dev]   # (N_real, D)
                else:
                    dy_real = dy_flat_pad
            else:
                dy_real = dy_flat_pad
            self._compute_main(dy_real)

    def _compute_pad(self, dy_flat):
        abs_dy  = dy_flat.abs()
        self.ezr_pad_steps.append((abs_dy == 0).float().mean().item())
        absmax_v = abs_dy.max(dim=1).values.clamp(min=self.eps).unsqueeze(1)
        scaled   = dy_flat / absmax_v
        self.qzr_all_pad_steps.append(
            (scaled.abs() < self.zero_thresh).float().mean().item()
        )

    def _compute_main(self, dy_real: torch.Tensor):
        """Core computation on real-token (mask-excluded) vectors."""
        abs_dy = dy_real.abs()
        N, D   = abs_dy.shape
        if N == 0:
            for lst in [self.ezr_steps, self.qzr_all_steps, self.qzr_nz_steps]:
                lst.append(0.0)
            self.odr_steps.append(1.0)
            self.cosine_steps.append(1.0)
            for lst in [self.ratio_q50_steps, self.ratio_q90_steps, self.ratio_q99_steps]:
                lst.append(0.0)
            self.p_clip_steps.append(0.0)
            for lst in [self.absmax_q50_steps, self.absmax_q90_steps,
                        self.absmax_q99_steps, self.absmax_q999_steps]:
                lst.append(0.0)
            self.clip_rate_scaled_steps.append(0.0)
            self.l2_retention_steps.append(1.0)
            self.rel_l2_error_steps.append(0.0)
            self.n_vec_steps.append(0)
            return

        # EZR
        self.ezr_steps.append((abs_dy == 0).float().mean().item())

        # per-vector absmax (uncapped, for ECDF + reservoir)
        absmax_v = abs_dy.max(dim=1).values   # (N,)
        self._absmax_buf.append(absmax_v.cpu().float().numpy())

        # alpha: cap by nm_thres if >0
        if self.nm_thres > 0:
            alpha = absmax_v.clamp(max=self.nm_thres).clamp(min=self.eps).unsqueeze(1)
        else:
            alpha = absmax_v.clamp(min=self.eps).unsqueeze(1)

        # ratio = |dy| / alpha
        ratio = abs_dy / alpha   # (N, D)

        # QZR_all and QZR_nonzero
        zero_mask = (abs_dy == 0)
        nz_mask   = ~zero_mask
        self.qzr_all_steps.append((ratio < self.zero_thresh).float().mean().item())
        if nz_mask.any():
            self.qzr_nz_steps.append(
                (ratio[nz_mask] < self.zero_thresh).float().mean().item()
            )
        else:
            self.qzr_nz_steps.append(0.0)

        # ODR: absmax / median per vector
        absmed_v = abs_dy.median(dim=1).values.clamp(min=self.eps)
        self.odr_steps.append((absmax_v / absmed_v).mean().item())

        # p_clip: P(|delta| > INP_BOUND) element-wise
        self.p_clip_steps.append((abs_dy > INP_BOUND).float().mean().item())

        # absmax quantiles across N vectors
        absmax_sorted = absmax_v.sort().values
        n = len(absmax_sorted)
        self.absmax_q50_steps.append(absmax_sorted[max(0, int(0.50 * n) - 1)].item())
        self.absmax_q90_steps.append(absmax_sorted[max(0, int(0.90 * n) - 1)].item())
        self.absmax_q99_steps.append(absmax_sorted[max(0, int(0.99 * n) - 1)].item())
        self.absmax_q999_steps.append(absmax_sorted[min(int(0.999 * n), n - 1)].item())

        # Cosine sim: FP32 dy vs DAC-quantized dy (AIHWKit-consistent step)
        scaled = dy_real / alpha * INP_BOUND                        # [-INP_BOUND, +INP_BOUND]
        scaled_q = (scaled / self.step_size).round() * self.step_size
        self.clip_rate_scaled_steps.append(
            (scaled.abs() > INP_BOUND).float().mean().item()
        )
        scaled_q = scaled_q.clamp(-INP_BOUND, INP_BOUND)
        dy_q = scaled_q * alpha / INP_BOUND
        self.cosine_steps.append(
            F.cosine_similarity(dy_real, dy_q, dim=1).mean().item()
        )

        # l2_retention and rel_l2_error
        dy_norm   = dy_real.norm(dim=1).clamp(min=self.eps)
        dy_q_norm = dy_q.norm(dim=1)
        self.l2_retention_steps.append((dy_q_norm / dy_norm).mean().item())
        err_norm = (dy_real - dy_q).norm(dim=1)
        self.rel_l2_error_steps.append((err_norm / dy_norm).mean().item())

        # n_vec
        self.n_vec_steps.append(N)

        # ratio quantiles (always using uncapped absmax)
        ratio_orig = abs_dy / absmax_v.clamp(min=self.eps).unsqueeze(1)
        ratio_np   = ratio_orig.reshape(-1).cpu().float().numpy()
        self.ratio_q50_steps.append(float(np.quantile(ratio_np, 0.50)))
        self.ratio_q90_steps.append(float(np.quantile(ratio_np, 0.90)))
        self.ratio_q99_steps.append(float(np.quantile(ratio_np, 0.99)))

        # ratio reservoir sampling
        if self._store_sweep:
            ratio_flat = ratio_orig.reshape(-1).cpu().float().numpy()
            n_elem = len(ratio_flat)
            sample_size = min(4096, n_elem)
            if n_elem <= sample_size:
                sampled = ratio_flat
            else:
                rng = np.random.default_rng(seed=self._ratio_reservoir_count)
                sampled = ratio_flat[rng.choice(n_elem, sample_size, replace=False)]
            self._ratio_reservoir.append(sampled)
            self._ratio_reservoir_count += n_elem
            # compact if over max
            total = sum(len(a) for a in self._ratio_reservoir)
            if total > self._ratio_reservoir_max:
                all_s = np.concatenate(self._ratio_reservoir)
                keep = np.random.default_rng(42).choice(
                    len(all_s), self._ratio_reservoir_max, replace=False
                )
                self._ratio_reservoir = [all_s[keep]]

    def absmax_array(self) -> np.ndarray:
        """Full absmax distribution for ECDF plots."""
        return np.concatenate(self._absmax_buf) if self._absmax_buf else np.array([])

    def ratio_reservoir_array(self) -> np.ndarray:
        """Sampled ratio distribution for CDF plots."""
        return np.concatenate(self._ratio_reservoir) if self._ratio_reservoir else np.array([])

    def summary(self, label: str = "baseline", dac_bits: int = DAC_BITS,
                adc_bits: int = ADC_BITS, figure_id: str = "",
                run_tag: str = "", sto_round: bool = False) -> dict:
        def _m(lst): return float(np.mean(lst)) if lst else float("nan")
        return {
            "figure_id":        figure_id,
            "run_tag":          run_tag,
            "layer_name":       self.name,
            "layer_idx":        self.layer_idx,
            "sublayer":         self.sublayer,
            "variant":          label,
            "dac_bits":         dac_bits,
            "adc_bits":         adc_bits,
            "inp_bound":        INP_BOUND,
            "inp_res":          self.inp_res,
            "res_ratio":        self.res_ratio,
            "step_size":        self.step_size,
            "nm_thres":         self.nm_thres,
            "sto_round":        sto_round,
            "EZR":              _m(self.ezr_steps),
            "QZR_all":          _m(self.qzr_all_steps),
            "QZR_nonzero":      _m(self.qzr_nz_steps),
            "ODR":              _m(self.odr_steps),
            "cosine_sim":       _m(self.cosine_steps),
            "l2_retention":     _m(self.l2_retention_steps),
            "rel_l2_error":     _m(self.rel_l2_error_steps),
            "clip_rate_scaled": _m(self.clip_rate_scaled_steps),
            "ratio_q50":        _m(self.ratio_q50_steps),
            "ratio_q90":        _m(self.ratio_q90_steps),
            "ratio_q99":        _m(self.ratio_q99_steps),
            "p_clip":           _m(self.p_clip_steps),
            "absmax_q50":       _m(self.absmax_q50_steps),
            "absmax_q90":       _m(self.absmax_q90_steps),
            "absmax_q99":       _m(self.absmax_q99_steps),
            "absmax_q999":      _m(self.absmax_q999_steps),
            "EZR_pad":          _m(self.ezr_pad_steps),
            "QZR_all_pad":      _m(self.qzr_all_pad_steps),
        }

    def step_records(self, label: str = "baseline", dac_bits: int = DAC_BITS,
                     adc_bits: int = ADC_BITS, figure_id: str = "",
                     run_tag: str = "", sto_round: bool = False) -> list:
        """Per-step records for steps CSV."""
        n = len(self.cosine_steps)
        records = []
        for i in range(n):
            def _g(lst, idx, default=float("nan")):
                return lst[idx] if idx < len(lst) else default
            rec = {
                "figure_id":        figure_id,
                "run_tag":          run_tag,
                "layer_name":       self.name,
                "layer_idx":        self.layer_idx,
                "sublayer":         self.sublayer,
                "variant":          label,
                "dac_bits":         dac_bits,
                "adc_bits":         adc_bits,
                "inp_bound":        INP_BOUND,
                "inp_res":          self.inp_res,
                "res_ratio":        self.res_ratio,
                "step_size":        self.step_size,
                "nm_thres":         self.nm_thres,
                "sto_round":        sto_round,
                "step_idx":         i,
                "n_vec":            _g(self.n_vec_steps, i, 0),
                "EZR":              _g(self.ezr_steps, i),
                "QZR_all":          _g(self.qzr_all_steps, i),
                "QZR_nonzero":      _g(self.qzr_nz_steps, i),
                "ODR":              _g(self.odr_steps, i),
                "cosine_sim":       _g(self.cosine_steps, i),
                "l2_retention":     _g(self.l2_retention_steps, i),
                "rel_l2_error":     _g(self.rel_l2_error_steps, i),
                "clip_rate_scaled": _g(self.clip_rate_scaled_steps, i),
                "ratio_q50":        _g(self.ratio_q50_steps, i),
                "ratio_q90":        _g(self.ratio_q90_steps, i),
                "ratio_q99":        _g(self.ratio_q99_steps, i),
                "p_clip":           _g(self.p_clip_steps, i),
                "absmax_q50":       _g(self.absmax_q50_steps, i),
                "absmax_q90":       _g(self.absmax_q90_steps, i),
                "absmax_q99":       _g(self.absmax_q99_steps, i),
                "absmax_q999":      _g(self.absmax_q999_steps, i),
                "EZR_pad":          _g(self.ezr_pad_steps, i),
                "QZR_all_pad":      _g(self.qzr_all_pad_steps, i),
            }
            records.append(rec)
        return records


# =============================================================================
# Hook Registration
# =============================================================================

def register_hooks(model, mask_buf: MaskBuffer, inp_res: float,
                   nm_thres: float = 0.0,
                   store_sweep: bool = True) -> tuple:
    """Register full backward hooks on all AnalogLinear matching the regex.

    Includes both QKVO (target) and FFN1/FFN2 (nontarget frozen) layers.
    """
    from aihwkit.nn import AnalogLinear

    stats_dict, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed
        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer,
                           inp_res=inp_res, nm_thres=nm_thres,
                           store_sweep=store_sweep)
        stats_dict[name] = stats

        def make_hook(s, mb):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0], mask=mb.val)
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats, mask_buf)))

    sublayers_found = sorted(set(s.sublayer for s in stats_dict.values()))
    print(f"[Hook] {len(stats_dict)} hooks, sublayers={sublayers_found}, "
          f"inp_res={inp_res:.6f}, nm_thres={nm_thres:.4g}")
    return stats_dict, handles


# =============================================================================
# NPZ Save - raw absmax for ECDF
# =============================================================================

def save_absmax_npz(stats_dict: dict, filepath: str):
    """Save raw per-vector absmax arrays for all layers to .npz.
    Keys: 'L{idx}_{sublayer}' e.g. 'L0_K', 'L5_FFN1'.
    """
    data = {}
    for name, s in stats_dict.items():
        key = f"L{s.layer_idx}_{s.sublayer}"
        arr = s.absmax_array()
        if len(arr) > 0:
            data[key] = arr
    np.savez_compressed(filepath, **data)
    n_keys = len(data)
    total  = sum(len(v) for v in data.values())
    print(f"  absmax npz -> {filepath}  ({n_keys} layers, {total:,} values)")


# =============================================================================
# CSV Validation
# =============================================================================

def validate_csv(filepath, required_columns, min_rows, critical_columns=None):
    """Validate CSV: existence, columns, row count, no NaN in critical cols."""
    assert os.path.exists(filepath), f"[CSV FAIL] Missing: {filepath}"
    df = pd.read_csv(filepath)
    for col in required_columns:
        assert col in df.columns, f"[CSV FAIL] Missing column '{col}' in {filepath}"
    assert len(df) >= min_rows, (
        f"[CSV FAIL] {filepath}: {len(df)} rows < min {min_rows}"
    )
    if critical_columns:
        for col in critical_columns:
            if col in df.columns:
                n_nan = df[col].isna().sum()
                assert n_nan == 0, (
                    f"[CSV FAIL] {filepath}: {n_nan} NaN in critical column '{col}'"
                )
    print(f"  [CSV OK] {filepath} ({len(df)} rows, {len(df.columns)} cols)")


# =============================================================================
# Diagnostic Run
# =============================================================================

def run_diagnostic(model, loader, n_step: int, desc: str = "Diag"):
    """n_step forward+backward passes. lr=0 -> no weight change."""
    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(SEED)

    for step, batch in enumerate(tqdm(loader, total=n_step, desc=desc)):
        if step >= n_step:
            break
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            start_positions=batch["start_positions"].to(DEVICE),
            end_positions=batch["end_positions"].to(DEVICE),
        )
        outputs.loss.backward()
        optimizer.step()   # lr=0 -> no weight change; flushes tile grad buffers


# =============================================================================
# Root Cause Analytics (for Figure A)
# =============================================================================

def compute_rootcause(stats_dict: dict, dac_bits: int = DAC_BITS,
                      adc_bits: int = ADC_BITS, figure_id: str = "A",
                      run_tag: str = ""):
    """Run consistency checks and compute auto-diagnosis verdict."""
    zero_thresh = 1.0 / (2**dac_bits - 2)

    # Consistency check
    for s in stats_dict.values():
        qzr_nz = np.mean(s.qzr_nz_steps) if s.qzr_nz_steps else 0.0
        odr    = np.array(s.odr_steps) if s.odr_steps else np.array([1.0])
        thresh = (2**dac_bits - 1) / 2
        if qzr_nz > 0.5 and np.median(odr) < thresh:
            print(f"  [WARN] {s.sublayer} L{s.layer_idx}: "
                  f"QZR_nz={qzr_nz:.3f}>0.5 but ODR_p50={np.median(odr):.1f}<{thresh:.1f}")

    rows = [s.summary("baseline", dac_bits=dac_bits, adc_bits=adc_bits,
                       figure_id=figure_id, run_tag=run_tag)
            for s in stats_dict.values()]
    df   = (pd.DataFrame(rows)
              .sort_values(["layer_idx", "sublayer"])
              .reset_index(drop=True))

    kv  = df[df["sublayer"].isin(["K", "V"])]
    qo  = df[df["sublayer"].isin(["Q", "O"])]
    ffn = df[df["sublayer"].isin(["FFN1", "FFN2"])]

    ezr_kv    = kv["EZR"].mean()
    qzr_nz_kv = kv["QZR_nonzero"].mean()
    cos_kv    = kv["cosine_sim"].mean()
    r50_kv    = kv["ratio_q50"].mean()

    print(f"\n[Root Cause] zero_thresh={zero_thresh:.6f} (1/{2**dac_bits-2})")
    print(f"  K/V  EZR={ezr_kv:.4f}, QZR_nz={qzr_nz_kv:.4f}, cosine={cos_kv:.4f}, "
          f"ratio_q50={r50_kv:.6f}")
    print(f"  Q/O  EZR={qo['EZR'].mean():.4f}, QZR_nz={qo['QZR_nonzero'].mean():.4f}")
    if len(ffn) > 0:
        print(f"  FFN  EZR={ffn['EZR'].mean():.4f}, "
              f"QZR_nz={ffn['QZR_nonzero'].mean():.4f}")

    structural = (ezr_kv > 0.3 and (kv["QZR_all"].mean() - ezr_kv) < 0.1)
    bulk_tiny  = (ezr_kv < 0.1 and qzr_nz_kv > 0.5 and r50_kv < zero_thresh)
    if structural and bulk_tiny:
        verdict = f"mixed: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} (both contribute)"
    elif structural:
        verdict = "structural exact-zero dominated (mask / attention sparsity)"
    elif bulk_tiny:
        verdict = "bulk tiny / outlier-dominant (scale issue)"
    else:
        verdict = (f"atypical: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} "
                   f"ratio_q50={r50_kv:.6f}")
    print(f"  [verdict] K/V: {verdict}")
    return df, verdict


# =============================================================================
# Figure A - QKVO+FFN Root Cause (2x3, 6 sublayers)
# =============================================================================

def figure_A(df: pd.DataFrame, verdict: str, stats_dict: dict,
             dac_bits: int = DAC_BITS, n_step: int = N_STEP):
    zero_thresh = 1.0 / (2**dac_bits - 2)

    def to_mat(col):
        mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
        for _, row in df.iterrows():
            li = int(row["layer_idx"])
            sl = row["sublayer"]
            if sl in SUBLAYER_ORDER and li < N_LAYERS:
                mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
        return mat

    qzr_nz_mat = to_mat("QZR_nonzero")
    ezr_mat    = to_mat("EZR")
    l2r_mat    = to_mat("l2_retention")

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        f"Figure A - QKVO+FFN Backward Root Cause Diagnosis "
        f"(BERT-base, {n_step} steps x batch {BATCH_SIZE})\n"
        f"DAC={dac_bits}-bit, zero_thresh=1/{2**dac_bits-2}={zero_thresh:.5f}, "
        f"6 sublayers (QKVO+FFN1+FFN2)",
        fontsize=10, y=1.01
    )

    def heatmap(ax, mat, title, cmap, vmin=0.0, vmax=1.0, label=""):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SUBLAYER_ORDER)))
        ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)

    # [0,0] QZR_nonzero heatmap
    heatmap(axes[0, 0], qzr_nz_mat,
            "(a) QZR_nonzero (mask-excluded)\nrange [0,1]: high = bulk-tiny/outlier-dom",
            "plasma", vmin=0.0, vmax=1.0, label="QZR_nonzero")

    # [0,1] EZR heatmap
    heatmap(axes[0, 1], ezr_mat,
            "(b) EZR - Exact Zero Ratio (mask-excluded)\nhigh = structural sparsity",
            "YlOrRd", vmin=0.0, vmax=1.0, label="EZR")

    # [0,2] l2_retention heatmap
    heatmap(axes[0, 2], l2r_mat,
            "(c) L2 Retention ||dy_q|| / ||dy||\n(mask-excluded, higher = better)",
            "RdYlGn", vmin=0.0, vmax=1.0, label="l2_retention")

    # [1,0] Bar: ratio_q50 per sublayer + zero_thresh line
    ax = axes[1, 0]
    sublayer_means = {
        sl: df[df["sublayer"] == sl]["ratio_q50"].mean() if len(df[df["sublayer"] == sl]) > 0
        else 0.0
        for sl in SUBLAYER_ORDER
    }
    colors6 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD", "#8C564B"]
    bars = ax.bar(SUBLAYER_ORDER,
                  [sublayer_means[sl] for sl in SUBLAYER_ORDER],
                  color=colors6, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh=1/{2**dac_bits-2}={zero_thresh:.5f}")
    for bar, sl in zip(bars, SUBLAYER_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + zero_thresh * 0.05,
                f"{sublayer_means[sl]:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("ratio_q50 = q50(|dy|/absmax)")
    ax.set_title("(d) Median ratio per sublayer (mask-excluded)\n"
                 "ratio < zero_thresh -> quantized to zero", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # [1,1] CDF: worst 3 K layers by QZR_nonzero (from ratio_reservoir)
    ax = axes[1, 1]
    k_df   = df[df["sublayer"] == "K"].sort_values("QZR_nonzero", ascending=False)
    worst3 = k_df.head(3)
    colors3 = ["#e41a1c", "#377eb8", "#4daf4a"]
    for (_, row), col in zip(worst3.iterrows(), colors3):
        lname  = row["layer_name"]
        li     = int(row["layer_idx"])
        stats  = stats_dict.get(lname)
        if stats is None:
            continue
        all_ratios = stats.ratio_reservoir_array()
        if len(all_ratios) == 0:
            continue
        sorted_r = np.sort(all_ratios)
        cdf      = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        idx = np.linspace(0, len(sorted_r) - 1, min(2000, len(sorted_r)), dtype=int)
        ax.plot(sorted_r[idx], cdf[idx], color=col, lw=1.5,
                label=f"K L{li}  QZR_nz={row['QZR_nonzero']:.3f}")
    ax.axvline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh={zero_thresh:.5f}")
    ax.set_xlabel("|dy|/absmax ratio"); ax.set_ylabel("CDF")
    ax.set_title("(e) CDF of |dy|/absmax - worst 3 K layers\n"
                 "left of threshold -> quantized to zero", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # [1,2] Text panel: auto-diagnosis summary
    ax = axes[1, 2]
    ax.axis("off")
    kv  = df[df["sublayer"].isin(["K", "V"])]
    ffn = df[df["sublayer"].isin(["FFN1", "FFN2"])]
    lines = [
        "Auto-Diagnosis Summary",
        "=" * 34,
        f"DAC bits:     {dac_bits}",
        f"inp_res:      1/{2**dac_bits-2} = {zero_thresh:.6f}",
        f"step_size:    {2.0 * INP_BOUND / (2**dac_bits - 2):.6f}",
        f"zero_thresh:  {zero_thresh:.6f}",
        "",
        f"K/V EZR:      {kv['EZR'].mean():.4f}",
        f"K/V QZR_all:  {kv['QZR_all'].mean():.4f}",
        f"K/V QZR_nz:   {kv['QZR_nonzero'].mean():.4f}",
        f"K/V cosine:   {kv['cosine_sim'].mean():.4f}",
        f"K/V l2_ret:   {kv['l2_retention'].mean():.4f}",
        f"K/V r_q50:    {kv['ratio_q50'].mean():.6f}",
        "",
    ]
    if len(ffn) > 0:
        lines += [
            f"FFN EZR:      {ffn['EZR'].mean():.4f}",
            f"FFN QZR_nz:   {ffn['QZR_nonzero'].mean():.4f}",
            f"FFN cosine:   {ffn['cosine_sim'].mean():.4f}",
            f"FFN l2_ret:   {ffn['l2_retention'].mean():.4f}",
            "",
        ]
    lines.append("Verdict (K/V):")
    # word-wrap verdict
    words, line, wrapped = verdict.split(" "), "", []
    for w in words:
        if len(line) + len(w) + 1 > 28:
            wrapped.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        wrapped.append(line)
    lines += wrapped
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(f) Diagnosis Panel", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_A, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure A -> {FIG_A}")


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # [1] Data                                                           #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading data ...")
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader     = load_data(tokenizer, n_step=N_STEP, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------ #
    # [2] Diagnostic run (DAC 7 / ADC 9 bit)                             #
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Baseline 7-bit, {N_STEP} steps ...")
    inp_res_7  = 1.0 / (2**DAC_BITS - 2)   # 1/126
    mask_buf_a = MaskBuffer()
    model_a    = create_model(nm_thres=0.0, sto_round=False,
                              dac_bits=DAC_BITS, adc_bits=ADC_BITS)
    mask_buf_a.register(model_a)
    stats_baseline, handles_a = register_hooks(model_a, mask_buf_a,
                                               inp_res=inp_res_7)
    run_diagnostic(model_a, loader, n_step=N_STEP, desc="A-baseline")
    for h in handles_a:
        h.remove()
    del model_a
    torch.cuda.empty_cache()
    gc.collect()

    # Save the raw per-vector absmax of every layer
    save_absmax_npz(stats_baseline,
                    os.path.join(OUT_DIR, "absmax_raw_A_baseline_7b.npz"))

    # ------------------------------------------------------------------ #
    # [3] Figure A - root cause analysis + figure                        #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Computing root cause -> Figure A ...")
    df_a, verdict_a = compute_rootcause(stats_baseline, dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="A",
                                        run_tag=RUN_TAG)
    df_a.to_csv(CSV_A_SUMMARY, index=False)
    print(f"  -> {CSV_A_SUMMARY}")

    # Steps CSV
    all_a_step_records = []
    for s in stats_baseline.values():
        all_a_step_records.extend(s.step_records(
            "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
            figure_id="A", run_tag=RUN_TAG))
    pd.DataFrame(all_a_step_records).to_csv(CSV_A_STEPS, index=False)
    print(f"  -> {CSV_A_STEPS} ({len(all_a_step_records)} rows)")

    figure_A(df_a, verdict_a, stats_baseline, dac_bits=DAC_BITS, n_step=N_STEP)

    del stats_baseline
    gc.collect()

    # ------------------------------------------------------------------ #
    # [4] CSV Validation + Summary                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[4/4] Validation + Summary")
    print("=" * 60)

    COMMON_COLS = ["figure_id", "run_tag", "variant", "dac_bits", "adc_bits",
                   "inp_bound", "inp_res", "res_ratio", "step_size", "nm_thres",
                   "layer_idx", "sublayer"]
    METRIC_COLS = ["EZR", "QZR_all", "QZR_nonzero", "ODR", "cosine_sim",
                   "l2_retention", "rel_l2_error", "clip_rate_scaled",
                   "ratio_q50", "ratio_q90", "ratio_q99",
                   "absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999"]
    STEP_EXTRA  = ["step_idx", "n_vec"]

    validate_csv(CSV_A_SUMMARY, COMMON_COLS + METRIC_COLS, min_rows=72,
                 critical_columns=["QZR_nonzero", "l2_retention"])
    validate_csv(CSV_A_STEPS, COMMON_COLS + STEP_EXTRA + METRIC_COLS,
                 min_rows=72 * N_STEP,
                 critical_columns=["QZR_nonzero"])

    # File listing
    for f in [CSV_A_SUMMARY, CSV_A_STEPS, FIG_A]:
        status = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{status}] {f}")

if __name__ == "__main__":
    main()
