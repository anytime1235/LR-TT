#!/usr/bin/env python
# coding=utf-8
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Experiment driver for the analog baselines: BERT-base fine-tuning on SQuAD v1.1.

Methods: single_rpu (one analog tile per layer, direct SGD), ttv1 (Tiki-Taka v1,
TransferCompound) and ideal (IdealDevice, FP32 weight update).
Modes: fixed (one run), grid (Optuna GridSampler), tpe (Optuna TPE search).

Notes:
  - With --io-bits > 0 the TTv1 transfer read (transfer_forward) gets the same
    IO settings as forward and backward; --io-bits 0 makes every path perfect.
  - classifier_lr applies to every method (separate param groups for the
    analog tiles, the classifier and LayerNorm).
  - Devices are ConstantStepDevice by default (--device-type selects
    LinearStep etc.).

Usage:
    python paper_experiment.py --mode fixed --method single_rpu \\
        --output-dir results/single_rpu --seed 42
"""

import argparse
import os
import gc
import csv
import json
import re
import string
import collections
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.optim.context import AnalogContext
from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin

from rpu_configs import get_config, dw_min_for_bits, PULSE_TYPE_MAP, DW_MIN_14BIT, io_res_from_bits
from update_diagnostics import UpdateDiagnostics


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
EVAL_BATCH_SIZE = 256


# ============================================================================
# Layer classification
# ============================================================================

def _classify_encoder_layer(layer_name):
    if 'attention' in layer_name:
        return 'attention'
    return 'ffn'


TARGET_LAYERS = "attention"  # "attention", "ffn", "all"


def is_target_layer(layer_name):
    """Check if encoder layer is a target for analog conversion."""
    always_digital = ["qa_outputs", "pooler"]
    if any(d in layer_name for d in always_digital):
        return False
    if "encoder" not in layer_name:
        return False
    if TARGET_LAYERS in ("all", "all_train_attention", "all_train_ffn"):
        return True
    return _classify_encoder_layer(layer_name) == TARGET_LAYERS


def get_layer_subtype(name):
    if "attention" not in name:
        return None
    if "query" in name:
        return "query"
    elif "key" in name:
        return "key"
    elif "value" in name:
        return "value"
    elif "dense" in name:
        return "dense"
    return None


# ============================================================================
# Per-layer mixed precision IO
# ============================================================================

# Sublayer name mapping: module name pattern -> sublayer key
_SUBLAYER_PATTERNS = {
    "intermediate.dense": "FFN1",
    "output.dense": "FFN2",    # catches attention.output.dense and encoder.layer.X.output.dense
    "attention.self.query": "Q",
    "attention.self.key": "K",
    "attention.self.value": "V",
    "attention.output.dense": "O",
}


def _get_sublayer_key(module_name):
    """Map a module name to (layer_idx, sublayer_key) or None."""
    import re
    m = re.search(r"encoder\.layer\.(\d+)\.", module_name)
    if not m:
        return None
    layer_idx = int(m.group(1))
    # Match longest pattern first to avoid "output.dense" matching before "attention.output.dense"
    for pattern in sorted(_SUBLAYER_PATTERNS, key=len, reverse=True):
        if pattern in module_name:
            return (layer_idx, _SUBLAYER_PATTERNS[pattern])
    return None


def _parse_per_layer_bits(spec_str):
    """Parse per-layer-bits spec string into {(layer_idx, sublayer): bits}.

    Format: "sublayer:bits,sublayer:bits;sublayer:bits,..." where layers are
    separated by semicolons (layer 0 first) and sublayers by commas.

    Shorthand: "FFN1=12,FFN2=6,K=10,O=6,Q=8,V=8" applies same to all layers.
    Per-layer: "L0:FFN1=12,K=8;L1:FFN1=10,K=10" for layer-specific.

    Or JSON file path for complex configs.
    """
    import json
    result = {}

    # Try JSON file first
    if spec_str.endswith(".json"):
        with open(spec_str) as f:
            data = json.load(f)
        # Expected format: {"0": {"FFN1": 12, "K": 10, ...}, "1": {...}, ...}
        for li_str, sl_dict in data.items():
            for sl, bits in sl_dict.items():
                result[(int(li_str), sl)] = int(bits)
        return result

    # Shorthand: "FFN1=12,FFN2=6,K=10,O=6,Q=8,V=8" -> all layers same
    if "L" not in spec_str and ";" not in spec_str:
        parts = spec_str.split(",")
        for part in parts:
            sl, bits = part.strip().split("=")
            for li in range(12):
                result[(li, sl.strip())] = int(bits)
        return result

    # Per-layer: "L0:FFN1=12,K=8;L1:FFN1=10"
    for layer_spec in spec_str.split(";"):
        layer_spec = layer_spec.strip()
        if not layer_spec:
            continue
        layer_part, rest = layer_spec.split(":", 1)
        li = int(layer_part.strip().replace("L", ""))
        for part in rest.split(","):
            sl, bits = part.strip().split("=")
            result[(li, sl.strip())] = int(bits)
    return result


def _make_specific_rpu_config_fun(plb_dict, base_config):
    """Create a specific_rpu_config_fun for convert_to_analog.

    This applies per-layer IO bit overrides at tile creation time,
    which is the only way to change IO resolution in aihwkit
    (modifying tile.rpu_config after creation has no effect).

    Args:
        plb_dict: {(layer_idx, sublayer_key): bits} from _parse_per_layer_bits.
        base_config: The base RPU config (used for non-overridden modules).
    """
    from copy import deepcopy
    count_holder = [0]  # mutable counter

    def specific_fn(module_name, module, rpu_config):
        key = _get_sublayer_key(module_name)
        if key is not None and key in plb_dict:
            bits = plb_dict[key]
            cfg = deepcopy(rpu_config)
            res = io_res_from_bits(bits)
            cfg.forward.inp_res = res
            cfg.forward.out_res = res
            cfg.backward.inp_res = res
            cfg.backward.out_res = res
            count_holder[0] += 1
            return cfg
        return rpu_config

    return specific_fn, count_holder


# ============================================================================
# SQuAD data loading
# ============================================================================

def load_data(tokenizer, batch_size, seed=42):
    """Load and tokenize SQuAD v1.1 dataset."""
    raw_datasets = load_dataset("squad")
    eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
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
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
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
    train_subset = tokenized_train.shuffle(seed=seed)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(seed)
    )

    return train_loader, tokenized_eval, eval_examples


# ============================================================================
# SQuAD evaluation
# ============================================================================

def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits,
                                  n_best_size=20, max_answer_length=30):
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
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None):
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


# ============================================================================
# C-tile-only evaluation helpers (TTv1)
# ============================================================================

def zero_fast_tiles(model):
    """Zero out A tile (hidden_weights_0) for C-only eval. Returns saved state."""
    saved = {}
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        for i, tile in enumerate(module.analog_tiles()):
            key = f"{name}.{i}"
            hidden = tile.get_hidden_parameters()
            if "hidden_weights_0" in hidden:
                saved[key] = hidden["hidden_weights_0"].clone()
                hidden["hidden_weights_0"] = torch.zeros_like(saved[key])
                tile.set_hidden_parameters(hidden)
    return saved


def restore_fast_tiles(model, saved):
    """Restore previously saved A tile weights after C-only eval."""
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        for i, tile in enumerate(module.analog_tiles()):
            key = f"{name}.{i}"
            if key in saved:
                hidden = tile.get_hidden_parameters()
                hidden["hidden_weights_0"] = saved[key]
                tile.set_hidden_parameters(hidden)


def evaluate_model(model, eval_features, eval_examples, device):
    """Evaluate SQuAD model. Returns (F1, EM)."""
    model.eval()
    all_start_logits = []
    all_end_logits = []

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(
        eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn
    )

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)
    return results["f1"], results["exact_match"]


# ============================================================================
# Scheduler
# ============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_rate=0.0):
    """Linear schedule with warmup applied to all param groups."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = max(0.0, float(current_step - num_warmup_steps)) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda] * len(optimizer.param_groups)
    )


# ============================================================================
# LR helpers
# ============================================================================

def get_current_analog_lr(optimizer):
    """Get current LR for analog (AnalogContext) param groups.

    After regroup_param_groups(), analog groups are at the end.
    param_groups[0]['lr'] would return classifier_lr, not analog_lr.
    Without a trained analog group it returns the LR of the first param group.
    """
    analog_lrs = []
    for pg in optimizer.param_groups:
        # Frozen tiles sit in lr=0.0 groups purely so the optimizer clears their
        # activation/gradient buffers; averaging them in would understate the
        # LR that the trained tiles actually run at.
        if pg.get("analog_frozen"):
            continue
        if any(isinstance(p, AnalogContext) for p in pg["params"]):
            analog_lrs.append(pg["lr"])
    if not analog_lrs:
        if optimizer.param_groups:
            return float(optimizer.param_groups[0]["lr"])
        return 0.0
    return float(sum(analog_lrs) / len(analog_lrs))


def get_current_classifier_lr(optimizer):
    """Get current LR for classifier (qa_outputs) param group."""
    for pg in optimizer.param_groups:
        if not any(isinstance(p, AnalogContext) for p in pg["params"]):
            # First non-analog group is classifier
            return float(pg["lr"])
    return 0.0


def get_current_ln_lr(optimizer):
    """Get current LR for LayerNorm param group.

    After regroup, digital groups remain in original order:
    classifier first, then other (LayerNorm).
    """
    non_analog = []
    for pg in optimizer.param_groups:
        if not any(isinstance(p, AnalogContext) for p in pg["params"]):
            non_analog.append(pg)
    # classifier=first, ln=second (if exists)
    if len(non_analog) >= 2:
        return float(non_analog[1]["lr"])
    elif len(non_analog) == 1:
        return float(non_analog[0]["lr"])
    return 0.0


# ============================================================================
# Model creation
# ============================================================================

def create_model(args, device_str="cuda"):
    """Create BERT-base QA model with selective analog conversion.

    Returns:
        (model, rpu_config)
    """
    device = torch.device(device_str)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Reinitialize qa_outputs with fixed seed
    if hasattr(model, 'qa_outputs'):
        torch.manual_seed(args.seed)
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)

    # Classify layers
    all_linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    target_layers = [n for n in all_linear_names if is_target_layer(n)]
    exclude = [n for n in all_linear_names if n not in target_layers]

    # Build the RPU config and convert to analog
    method_kwargs = _build_method_kwargs(args)
    rpu_config = get_config(args.method, **method_kwargs)

    print(f"  Method: {args.method}")
    print(f"  Config type: {type(rpu_config).__name__}")
    print(f"  Forward perfect: {rpu_config.forward.is_perfect}")
    print(f"  Backward perfect: {rpu_config.backward.is_perfect}")
    if args.io_bits > 0:
        io_res = io_res_from_bits(args.io_bits)
        print(f"  IO bits: {args.io_bits} (DAC/ADC res={io_res:.6f})")
    else:
        print(f"  IO bits: perfect (no quantization)")
    if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "transfer_forward"):
        print(f"  transfer_forward.is_perfect: {rpu_config.device.transfer_forward.is_perfect}")
    # Resolved IO per path - logged explicitly so a run is self-describing
    # (out_noise in particular is not recoverable from --io-bits alone).
    for _tag, _io in [("forward", rpu_config.forward), ("backward", rpu_config.backward)] + (
        [("transfer", rpu_config.device.transfer_forward)]
        if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "transfer_forward") else []
    ):
        if _io.is_perfect:
            print(f"  IO[{_tag:8s}]: perfect")
        else:
            print(f"  IO[{_tag:8s}]: inp_res={_io.inp_res:.6f} out_res={_io.out_res:.6f} "
                  f"out_noise={_io.out_noise} out_bound={_io.out_bound} "
                  f"NM={_io.noise_management.value} BM={_io.bound_management.value}")
    if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "gamma"):
        print(f"  gamma: {rpu_config.device.gamma}")
    print(f"  Target layers: {len(target_layers)}, Excluded: {len(exclude)}")

    # Convert to analog (with per-layer IO bit override if specified)
    specific_fn = None
    plb = None
    if args.per_layer_bits:
        plb = _parse_per_layer_bits(args.per_layer_bits)
        specific_fn, count_holder = _make_specific_rpu_config_fun(plb, rpu_config)

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude,
                              specific_rpu_config_fun=specific_fn)

    analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog layers: {analog_count}")

    if plb is not None:
        print(f"  [per-layer-bits] Applied to {count_holder[0]} modules at creation time")
        avg_bits = sum(plb.values()) / len(plb)
        print(f"  Per-layer IO bits: {len(plb)} tiles, avg={avg_bits:.2f}b")

    # Set requires_grad
    # all_train_<group>: every encoder linear is analog (same device and IO for
    # all of them), but only <group> receives weight updates, so the analog
    # substrate is that of `all` and only the trained group differs.
    train_only = {"all_train_attention": "attention",
                  "all_train_ffn": "ffn"}.get(TARGET_LAYERS)
    frozen_tile_count = 0
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            if train_only is not None and _classify_encoder_layer(name) != train_only:
                param.requires_grad = False
                frozen_tile_count += 1
            else:
                param.requires_grad = True
        elif "qa_outputs" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total:,}")
    if train_only is not None:
        frozen_group = "ffn" if train_only == "attention" else "attention"
        print(f"  Trains analog group: {train_only}")
        print(f"  Frozen {frozen_group} tiles: {frozen_tile_count} (analog + IO, but no weight update)")

    return model.to(device), rpu_config


def _resolved_io_dict(rpu_config):
    """Snapshot the configured IO settings of every analog path of the built config.

    Stored as `io_resolved` in config.json: forward, backward and (for transfer
    compounds) transfer_forward, with resolution, out_noise, out_bound and the
    noise and bound management, which the top-level `io_*` fields do not fully
    describe.
    """
    if rpu_config is None:
        return None

    def snap(io):
        if io.is_perfect:
            return {"is_perfect": True}
        return {
            "is_perfect": False,
            "inp_res": io.inp_res,
            "out_res": io.out_res,
            "inp_noise": io.inp_noise,
            "out_noise": io.out_noise,
            "out_bound": io.out_bound,
            "noise_management": io.noise_management.value,
            "bound_management": io.bound_management.value,
        }

    out = {"forward": snap(rpu_config.forward), "backward": snap(rpu_config.backward)}
    if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "transfer_forward"):
        out["transfer_forward"] = snap(rpu_config.device.transfer_forward)
    return out


def _build_method_kwargs(args):
    """Build kwargs dict for get_config() from CLI args."""
    kwargs = {}

    # Common
    if args.dw_min is not None:
        kwargs["dw_min"] = args.dw_min
    elif args.n_bits is not None:
        kwargs["dw_min"] = dw_min_for_bits(args.n_bits)
    kwargs["count_pulses"] = args.count_pulses

    # IO bit precision (0 = perfect)
    kwargs["io_bits"] = args.io_bits
    kwargs["noise_management"] = args.noise_management
    kwargs["out_noise"] = args.out_noise

    # Resolve omega: default 1.0, but auto-set to 0.0 if w_max < 1.0
    omega = args.omega if args.omega is not None else (0.0 if args.w_max < 1.0 else 1.0)

    if args.method == "single_rpu":
        pt = PULSE_TYPE_MAP.get(args.pulse_type, PULSE_TYPE_MAP["stochastic"])
        kwargs["pulse_type"] = pt
        kwargs["desired_bl"] = args.desired_bl
        kwargs["w_max"] = args.w_max
        kwargs["omega"] = omega
        kwargs["device_type"] = args.device_type
        kwargs["ls_gamma_up_ratio"] = args.ls_gamma_up_ratio
        kwargs["ls_gamma_down_ratio"] = args.ls_gamma_down_ratio
        kwargs["ls_noise_ratio"] = args.ls_noise_ratio
        kwargs["ls_gamma_up"] = args.ls_gamma_up
        kwargs["ls_gamma_down"] = args.ls_gamma_down
        kwargs["ls_dw_min_std"] = args.ls_dw_min_std
        kwargs["ls_dw_min_dtod"] = args.ls_dw_min_dtod

    elif args.method == "ttv1":
        kwargs["gamma"] = args.gamma
        kwargs["desired_bl"] = args.desired_bl
        kwargs["device_type"] = args.device_type
        kwargs["ls_gamma_up_ratio"] = args.ls_gamma_up_ratio
        kwargs["ls_gamma_down_ratio"] = args.ls_gamma_down_ratio
        kwargs["ls_noise_ratio"] = args.ls_noise_ratio
        kwargs["ls_gamma_up"] = args.ls_gamma_up
        kwargs["ls_gamma_down"] = args.ls_gamma_down
        kwargs["ls_dw_min_std"] = args.ls_dw_min_std
        kwargs["ls_dw_min_dtod"] = args.ls_dw_min_dtod
        kwargs["device_type_slow"] = args.device_type_slow
        kwargs["ls_gamma_up_slow"] = args.ls_gamma_up_slow
        kwargs["ls_gamma_down_slow"] = args.ls_gamma_down_slow
        kwargs["ls_noise_ratio_slow"] = args.ls_noise_ratio_slow
        if args.transfer_every is not None:
            kwargs["transfer_every"] = args.transfer_every
        if args.units_in_mbatch is not None:
            kwargs["units_in_mbatch"] = (args.units_in_mbatch.lower() == "true")
        if args.fast_lr is not None:
            kwargs["fast_lr"] = args.fast_lr
        if args.transfer_lr is not None:
            kwargs["transfer_lr"] = args.transfer_lr
        if args.scale_transfer_lr is not None:
            kwargs["scale_transfer_lr"] = (args.scale_transfer_lr.lower() == "true")
        if args.n_reads_per_transfer is not None:
            kwargs["n_reads_per_transfer"] = args.n_reads_per_transfer
        if args.with_reset_prob is not None:
            kwargs["with_reset_prob"] = args.with_reset_prob
        if args.transfer_bl is not None:
            kwargs["transfer_bl"] = args.transfer_bl
        if args.n_bits_slow is not None:
            kwargs["dw_min_slow"] = dw_min_for_bits(args.n_bits_slow)
        if args.w_max_fast is not None:
            kwargs["w_max_fast"] = args.w_max_fast

    # TTv1 pulse type overrides
    if args.method == "ttv1":
        if args.ttv1_fast_pulse_type is not None:
            kwargs["fast_pulse_type"] = PULSE_TYPE_MAP[args.ttv1_fast_pulse_type]
        if args.ttv1_transfer_pulse_type is not None:
            kwargs["transfer_pulse_type"] = PULSE_TYPE_MAP[args.ttv1_transfer_pulse_type]

    return kwargs


# ============================================================================
# Optimizer creation (classifier_lr for all methods)
# ============================================================================

def create_optimizer(model, args):
    """Create optimizer with 3 param groups (AnalogAdam with regroup)."""
    analog_p = []
    frozen_analog_p = []
    classifier_p = []
    other_p = []

    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext) and not param.requires_grad:
            # Frozen analog tiles are still handed to the optimizer, in a group
            # with lr 0.0. AnalogFunction.backward stores the layer input and
            # grad_output in analog_ctx for every tile, frozen ones included,
            # because gradients flow through them to the preceding layers; the
            # optimizer clears these buffers after each step. For lr == 0.0
            # aihwkit resets the context and skips the weight update.
            frozen_analog_p.append(param)
            continue
        if not param.requires_grad:
            continue
        if isinstance(param, AnalogContext):
            analog_p.append(param)
        elif "qa_outputs" in name:
            classifier_p.append(param)
        else:
            other_p.append(param)

    ln_lr = args.ln_lr

    param_groups = [
        {"params": analog_p, "lr": args.analog_lr},
        {"params": classifier_p, "lr": args.classifier_lr},
        {"params": other_p, "lr": ln_lr},  # LayerNorm LR (default=analog_lr)
    ]
    if frozen_analog_p:
        param_groups.append({"params": frozen_analog_p, "lr": 0.0})

    print(f"  Param groups before regroup: analog={len(analog_p)}, "
          f"classifier={len(classifier_p)}, other(LN)={len(other_p)}, "
          f"frozen_analog={len(frozen_analog_p)}")
    print(f"  LRs: analog={args.analog_lr}, classifier={args.classifier_lr}, ln={ln_lr}")

    optimizer = AnalogAdam(param_groups, lr=args.analog_lr)
    optimizer.regroup_param_groups()

    # regroup_param_groups() moves every AnalogContext into a new single-param
    # group via add_param_group(), and these groups take defaults["lr"]
    # (= analog_lr), not the lr of the source group. The frozen tiles are
    # therefore set back to lr 0.0 here, after regroup and before the LR
    # scheduler is built: LambdaLR stores each group's lr as initial_lr and
    # returns initial_lr * lambda, so 0.0 stays 0.0 for the whole run.
    if frozen_analog_p:
        frozen_ids = {id(p) for p in frozen_analog_p}
        n_zeroed = 0
        for pg in optimizer.param_groups:
            if any(id(p) in frozen_ids for p in pg["params"]):
                pg["lr"] = 0.0
                pg["analog_frozen"] = True
                n_zeroed += 1
        print(f"  Frozen analog groups pinned to lr=0.0: {n_zeroed}")

    # Verify analog LR
    analog_lr = get_current_analog_lr(optimizer)
    classifier_lr = get_current_classifier_lr(optimizer)
    print(f"  After regroup: analog_lr={analog_lr:.6f}, classifier_lr={classifier_lr:.6f}")

    return optimizer


# ============================================================================
# Training (fixed mode)
# ============================================================================

def train_fixed(args):
    """Single training run with fixed hyperparameters."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    # Resolve effective dw_min
    if args.dw_min is not None:
        effective_dw_min = args.dw_min
    elif args.n_bits is not None:
        effective_dw_min = dw_min_for_bits(args.n_bits, w_max=args.w_max)
    else:
        effective_dw_min = DW_MIN_14BIT * args.w_max

    # For TTv1 with --w-max-fast the fast tile's dw_min is effective_dw_min *
    # w_max_fast (printed below).
    effective_dw_min_fast = effective_dw_min
    if args.method == "ttv1" and args.w_max_fast is not None:
        effective_dw_min_fast = effective_dw_min * args.w_max_fast
        # effective_dw_min comes from --n-bits (fast/aux tile); the slow/core
        # tile is set by --n-bits-slow (see _build_method_kwargs,
        # kwargs["dw_min_slow"]), so each is printed from its own source.
        _slow_bits = args.n_bits_slow if args.n_bits_slow is not None else args.n_bits
        _dw_slow = (dw_min_for_bits(_slow_bits) if _slow_bits is not None
                    else effective_dw_min)
        print(f"Effective dw_min (slow/core, {_slow_bits}b, w_max=+-1.0): {_dw_slow:.6e}")
        print(f"Effective dw_min (fast/aux, {args.n_bits}b, w_max=+-{args.w_max_fast:g}): "
              f"{effective_dw_min_fast:.6e}")
    else:
        print(f"Effective dw_min: {effective_dw_min:.6e}")

    # Model
    print("\n=== Creating model ===")
    model, rpu_config = create_model(args, device_str=str(device))

    # Data
    print("\n=== Loading data ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer, args.batch_size, seed=args.seed)
    print(f"Train batches: {len(train_loader)}")

    # Optimizer
    print("\n=== Creating optimizer ===")
    optimizer = create_optimizer(model, args)

    # Scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    num_training_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        num_training_steps = min(num_training_steps, args.max_steps)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_min_lr(
        optimizer, num_warmup_steps, num_training_steps, min_lr_rate=args.min_lr_rate
    )

    # Digital params for grad clipping
    digital_params = [p for p in model.parameters()
                      if not isinstance(p, AnalogContext) and p.requires_grad]

    # Diagnostics: UpdateDiagnostics (--diag-update-exact)
    diag = None
    if args.diag_update_exact:
        diag_w_max = args.w_max
        if args.method == "ttv1" and args.w_max_fast is not None:
            diag_w_max = args.w_max_fast  # track fast tile range
        # For ttv1, fast_lr is the pulse-count learning rate of the fast tile in
        # place of analog_lr (TransferCompound.fast_lr).
        diag_lr = args.analog_lr
        if args.method == "ttv1" and args.fast_lr is not None:
            diag_lr = args.fast_lr
        diag = UpdateDiagnostics(model, effective_dw_min, layer_set=args.diag_layer_set,
                                 method=args.method, lr=diag_lr, device_w_max=diag_w_max,
                                 desired_bl=args.desired_bl, um_grad_scale=1.0)
        print(f"Update diagnostics enabled ({len(diag.tile_registry)} tiles, "
              f"device_w_max={diag_w_max}, desired_bl={args.desired_bl})")

    # Output setup
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    fieldnames = ["step", "epoch", "loss", "analog_lr", "classifier_lr", "ln_lr",
                  "f1", "em", "f1_c_only", "em_c_only", "wall_time_s"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Save config
    _save_config(args, rpu_config, effective_dw_min, num_training_steps)

    # C-only eval: only for TTv1 with gamma > 0
    do_c_only = args.eval_c_only and args.method == "ttv1" and args.gamma is not None and args.gamma > 0

    # Initial eval
    if args.max_steps <= 0 or args.max_steps > 100:
        print("\n=== Initial evaluation ===")
        f1, em = evaluate_model(model, eval_features, eval_examples, device)
        print(f"Initial F1: {f1:.2f}, EM: {em:.2f}")
        if do_c_only:
            saved_a = zero_fast_tiles(model)
            f1_c, em_c = evaluate_model(model, eval_features, eval_examples, device)
            restore_fast_tiles(model, saved_a)
            print(f"Initial C-only F1: {f1_c:.2f}, EM: {em_c:.2f}")
    else:
        f1, em = 0.0, 0.0
        print(f"\n[TEST MODE] Skipping initial eval (max_steps={args.max_steps})")

    # Training loop
    print("\n=== Training ===")
    global_step = 0
    best_f1 = f1
    epochs_no_improve = 0
    _optuna_trial = getattr(args, "_optuna_trial", None)
    f1_c, em_c = 0.0, 0.0
    stop_training = False
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if stop_training:
            break
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if stop_training:
                break

            # Forward
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            # DIAG: grad_accum microbatch handling
            if args.grad_accum_steps > 1 and diag is not None:
                target_step = global_step + 1
                _will_diag = (args.diag_at_steps_set is not None and target_step in args.diag_at_steps_set) or \
                             (args.diag_at_steps_set is None and (args.diag_steps == 0 or target_step <= args.diag_steps))

                if _will_diag:
                    if batch_idx % args.grad_accum_steps == 0:
                        diag.snapshot_weights_before()
                    diag.accumulate_microbatch(model)

            # Grad accum > 1: the analog tiles are updated after every
            # micro-batch (tile.update() per micro-batch, then analog_ctx is
            # reset), so analog_ctx holds the (x, d) tensors of one
            # micro-batch at a time.
            if args.grad_accum_steps > 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if not isinstance(p, AnalogContext):
                            continue
                        if not p.requires_grad:
                            p.reset()
                            continue
                        if p.use_torch_update or not p.has_gradient():
                            continue
                        analog_tile = p.analog_tile
                        runtime = analog_tile.get_runtime()
                        if p.use_indexed:
                            for x_i, d_i in zip(p.analog_input, p.analog_grad_output):
                                analog_tile.update_indexed(
                                    x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                    d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                                )
                        else:
                            x_input = torch.cat(p.analog_input, axis=-1 if analog_tile.in_trans else 0)
                            d_input = torch.cat(p.analog_grad_output, axis=-1 if analog_tile.out_trans else 0)
                            analog_tile.update(
                                x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                                d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                            )
                        p.reset()

            # Only step on accumulation boundary
            if (batch_idx + 1) % args.grad_accum_steps != 0:
                continue

            global_step += 1
            step_loss = loss.item() * args.grad_accum_steps

            # DIAG: determine if this step should be diagnosed
            def _should_diag(step):
                if args.diag_at_steps_set is not None:
                    return step in args.diag_at_steps_set
                return args.diag_steps == 0 or step <= args.diag_steps

            # DIAG (update diagnostics): capture before step (while analog_ctx still has x, d)
            do_diag = (diag is not None and _should_diag(global_step))
            if do_diag:
                diag.snapshot_before_step(model, optimizer)

            # Digital grad clip
            if digital_params:
                torch.nn.utils.clip_grad_norm_(digital_params, max_norm=1.0)

            # Scheduler step
            scheduler.step()

            # Sync analog tile LR with scheduler
            for pg in optimizer.param_groups:
                for p in pg['params']:
                    if isinstance(p, AnalogContext):
                        p.analog_tile.set_learning_rate(pg['lr'])

            # Optimizer step: when grad_accum > 1, analog updates already
            # done above; only run digital Adam + post_update_step.
            if args.grad_accum_steps > 1:
                super(AnalogOptimizerMixin, optimizer).step()
                for pg in optimizer.param_groups:
                    for p in pg['params']:
                        if isinstance(p, AnalogContext) and p.requires_grad:
                            p.analog_tile.post_update_step()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # DIAG (update diagnostics): capture after step
            if do_diag:
                diag.snapshot_after_step(model, global_step)

            total_loss += step_loss
            num_batches += 1

            # Loss divergence check
            if not np.isfinite(step_loss) or step_loss > 1e8:
                print(f"[WARNING] Loss diverged at step {global_step} (loss={step_loss:.2e})")
                stop_training = True
                break

            current_analog_lr = get_current_analog_lr(optimizer)
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{current_analog_lr:.2e}")

            # Logging
            if global_step % args.log_every == 0:
                wall_time = time.time() - start_time
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": f"{step_loss:.6f}",
                    "analog_lr": f"{current_analog_lr:.6e}",
                    "classifier_lr": f"{get_current_classifier_lr(optimizer):.6e}",
                    "ln_lr": f"{get_current_ln_lr(optimizer):.6e}",
                    "f1": "",
                    "em": "",
                    "f1_c_only": "",
                    "em_c_only": "",
                    "wall_time_s": f"{wall_time:.1f}",
                }
                writer.writerow(row)
                csv_file.flush()

            # Max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"\n[MAX_STEPS] Reached {args.max_steps} steps, stopping.")
                stop_training = True
                break

        # Epoch-end evaluation
        if not stop_training or args.max_steps <= 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

            f1, em = evaluate_model(model, eval_features, eval_examples, device)
            print(f"Epoch {epoch} F1: {f1:.2f}, EM: {em:.2f}")

            # C-only evaluation
            f1_c, em_c = 0.0, 0.0
            if do_c_only:
                saved_a = zero_fast_tiles(model)
                f1_c, em_c = evaluate_model(model, eval_features, eval_examples, device)
                restore_fast_tiles(model, saved_a)
                print(f"Epoch {epoch} C-only F1: {f1_c:.2f}, EM: {em_c:.2f}  (delta: {f1 - f1_c:+.2f})")

            wall_time = time.time() - start_time
            eval_row = {
                "step": global_step, "epoch": epoch,
                "loss": f"{avg_loss:.6f}", "f1": f"{f1:.2f}", "em": f"{em:.2f}",
                "f1_c_only": f"{f1_c:.2f}" if do_c_only else "",
                "em_c_only": f"{em_c:.2f}" if do_c_only else "",
                "analog_lr": f"{get_current_analog_lr(optimizer):.6e}",
                "classifier_lr": f"{get_current_classifier_lr(optimizer):.6e}",
                "ln_lr": f"{get_current_ln_lr(optimizer):.6e}",
                "wall_time_s": f"{wall_time:.1f}",
            }
            writer.writerow(eval_row)
            csv_file.flush()

            if f1 > best_f1:
                best_f1 = f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Optuna pruning: report intermediate F1 and prune unpromising trials
            if _optuna_trial is not None:
                import optuna
                _optuna_trial.report(f1, epoch)
                if _optuna_trial.should_prune():
                    print(f"[PRUNE] Trial pruned at epoch {epoch} (F1={f1:.2f}, best={best_f1:.2f})")
                    csv_file.close()
                    raise optuna.TrialPruned()

            # Early stopping (patience in epochs; 0 = disabled)
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[EARLY STOP] No F1 improvement for {args.patience} epoch(s); "
                      f"stopping at epoch {epoch} (best={best_f1:.2f}).")
                break

    csv_file.close()

    # Save diagnostics
    if diag is not None:
        diag.save(args.output_dir)

    # Save summary
    summary = {
        "method": args.method,
        "seed": args.seed,
        "results": {
            "best_f1": best_f1,
            "final_f1": f1,
            "final_em": em,
            "total_steps": global_step,
            "wall_time_s": time.time() - start_time,
        },
    }
    if do_c_only:
        summary["results"]["final_f1_c_only"] = f1_c
        summary["results"]["final_em_c_only"] = em_c
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {args.method}, steps={global_step}, best_f1={best_f1:.2f}, "
          f"final_f1={f1:.2f}, em={em:.2f}")
    if do_c_only:
        print(f"C-only: final_f1={f1_c:.2f}, em={em_c:.2f}, delta_f1={f1 - f1_c:+.2f}")
    print(f"Output: {args.output_dir}")

    return best_f1


# ============================================================================
# Grid / TPE mode
# ============================================================================

def build_method_grid(args):
    """Build grid search space for Optuna GridSampler. Only method-relevant params."""
    if args.method == "single_rpu":
        return {
            "pulse_type": ["stochastic", "deterministic", "mean_count",
                           "none_with_device"],
            "desired_bl": [31, 100],
        }
    elif args.method == "ttv1":
        return {
            "gamma": [0.0, 0.1],
            "transfer_every": [1, 24, 2400],
            "units_in_mbatch": ["true", "false"],
        }
    elif args.method == "ideal":
        return {}
    return {}


def train_grid(args):
    """Grid search mode using Optuna GridSampler."""
    import optuna

    space = build_method_grid(args)
    if not space:
        print("No grid params for this method, running single fixed trial.")
        return train_fixed(args)

    # Compute total grid size
    from itertools import product as iterproduct
    grid_size = 1
    for v in space.values():
        grid_size *= len(v)
    n_trials = args.n_trials if args.n_trials > 0 else grid_size

    sampler = optuna.samplers.GridSampler(space)
    study_name = args.study_name or f"paper_{args.method}_grid"
    storage = None
    if args.db_dir:
        os.makedirs(args.db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(args.db_dir, study_name)}.db"

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        # Sample params from grid
        trial_args = argparse.Namespace(**vars(args))

        if args.method == "single_rpu":
            trial_args.pulse_type = trial.suggest_categorical("pulse_type", space["pulse_type"])
            trial_args.desired_bl = trial.suggest_categorical("desired_bl", space["desired_bl"])
        elif args.method == "ttv1":
            trial_args.gamma = trial.suggest_categorical("gamma", space["gamma"])
            trial_args.transfer_every = trial.suggest_categorical("transfer_every", space["transfer_every"])
            trial_args.units_in_mbatch = trial.suggest_categorical("units_in_mbatch", space["units_in_mbatch"])

        # Unique output dir per trial
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
        trial_args.output_dir = trial_dir

        try:
            f1 = train_fixed(trial_args)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            # Cleanup between trials
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return f1

    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")


def train_tpe(args):
    """TPE search anchored on the passed-in hyperparameters.

    The CLI values are the per-target best-known starting point (trial 0). Bands
    come from --alr-range/--clr-range/--tlr-range/--te-choices/--flr-range/
    --wmf-range, or else from the anchor (analog_lr/classifier_lr in [x/3, 3x] log,
    transfer_lr in [x/10, min(1, 10x)], transfer_every over {0.25,0.5,1,2,4}x);
    fast_lr and w_max_fast are swept only with a range, ttv1 only. gamma /
    with_reset_prob / device / bits / IO are not searched. MedianPruner stops weak
    trials.
    """
    import optuna

    study_name = args.study_name or f"paper_{args.method}_tpe"
    storage = None
    if args.db_dir:
        os.makedirs(args.db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(args.db_dir, study_name)}.db"

    # TPESampler defaults to n_startup_trials=10 (10 random trials before the TPE
    # model engages). For short runs (n_trials ~10-20) that leaves the search
    # entirely or mostly random, so allow lowering it. Default None = stock behaviour.
    tpe_kwargs = {}
    if getattr(args, "tpe_startup_trials", None) is not None:
        tpe_kwargs["n_startup_trials"] = args.tpe_startup_trials
    sampler = optuna.samplers.TPESampler(seed=args.seed, **tpe_kwargs)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # --- Anchor (where trial 0 starts) and search band (how far trials roam) ---
    # The anchor is always the passed-in HP. The band is either given explicitly
    # via --alr-range/--clr-range/--tlr-range/--te-choices, or derived from the
    # anchor as a multiplicative window around it.
    anchor_alr = args.analog_lr
    anchor_clr = args.classifier_lr

    if args.alr_range is not None:
        alr_lo, alr_hi = sorted(args.alr_range)
    else:
        alr_lo, alr_hi = anchor_alr / 3.0, anchor_alr * 3.0
    if args.clr_range is not None:
        clr_lo, clr_hi = sorted(args.clr_range)
    else:
        clr_lo, clr_hi = anchor_clr / 3.0, anchor_clr * 3.0

    te_choices = None
    # transfer_lr is the only learning path to the core (slow/C) tile when gamma=0,
    # so it must stay > 0 and is a primary search dim.
    sweep_tlr = (args.method == "ttv1" and args.transfer_lr is not None
                 and args.transfer_lr > 0)
    if sweep_tlr:
        if args.tlr_range is not None:
            tlr_lo, tlr_hi = sorted(args.tlr_range)
        else:
            tlr_lo, tlr_hi = args.transfer_lr / 10.0, min(1.0, args.transfer_lr * 10.0)
    if args.method == "ttv1":
        if args.te_choices is not None:
            te_choices = sorted({max(1, int(round(float(x))))
                                 for x in args.te_choices.split(",") if x.strip()})
        elif args.transfer_every is not None:
            te0 = int(round(args.transfer_every))
            te_choices = sorted({max(1, int(round(te0 * m))) for m in (0.25, 0.5, 1, 2, 4)})

    # w_max_fast: the aux/fast tile's weight range. Level count stays 2**n_bits, so
    # sweeping this sweeps dw_min_fast = (2/2**n_bits) * w_max_fast - the LRTT A/B
    # (multilevel, dw_min) parameterisation with n_bits playing multilevel's role.
    # Swept only with --wmf-range; otherwise the fast tile keeps --w-max-fast
    # (+-1.0 when unset).
    sweep_wmf = args.method == "ttv1" and args.wmf_range is not None
    if sweep_wmf:
        wmf_lo, wmf_hi = sorted(args.wmf_range)

    # fast_lr is the A/aux tile's SGD learning rate: the TransferCompound uses it
    # in place of the optimizer LR whenever fast_lr > 0 (aihwkit default 1.0), so
    # --analog-lr does not reach a TransferCompound tile and fast_lr is the knob
    # that tunes the aux tile's step size.
    sweep_flr = args.method == "ttv1" and args.flr_range is not None
    if sweep_flr:
        flr_lo, flr_hi = sorted(args.flr_range)

    # Enqueue the best-known HP as the very first trial (only on a fresh study).
    # Optuna rejects an enqueued value outside the distribution, so clamp the
    # anchor into the band and snap transfer_every to the nearest offered choice.
    if len(study.trials) == 0:
        def _clamp(v, lo, hi, name):
            c = min(max(v, lo), hi)
            if c != v:
                print(f"[ANCHOR] {name} {v:g} outside [{lo:g}, {hi:g}] -> clamped to {c:g}")
            return c

        anchor = {
            "analog_lr": _clamp(anchor_alr, alr_lo, alr_hi, "analog_lr"),
            "classifier_lr": _clamp(anchor_clr, clr_lo, clr_hi, "classifier_lr"),
        }
        if sweep_tlr:
            anchor["transfer_lr"] = _clamp(args.transfer_lr, tlr_lo, tlr_hi, "transfer_lr")
        if te_choices is not None and args.transfer_every is not None:
            te0 = int(round(args.transfer_every))
            te_snap = min(te_choices, key=lambda x: abs(x - te0))
            if te_snap != te0:
                print(f"[ANCHOR] transfer_every {te0} not in {te_choices} -> snapped to {te_snap}")
            anchor["transfer_every"] = te_snap
        if sweep_wmf:
            # Unset --w-max-fast means a fast-tile range of +-1.0, which is then
            # the anchor.
            anchor["w_max_fast"] = _clamp(
                args.w_max_fast if args.w_max_fast is not None else 1.0,
                wmf_lo, wmf_hi, "w_max_fast")
        if sweep_flr:
            anchor["fast_lr"] = _clamp(
                args.fast_lr if args.fast_lr is not None else 1.0,
                flr_lo, flr_hi, "fast_lr")
        study.enqueue_trial(anchor)
        print(f"[ANCHOR] Enqueued starting trial: {anchor}")

    print(f"[SPACE] analog_lr     log-uniform [{alr_lo:g}, {alr_hi:g}]")
    print(f"[SPACE] classifier_lr log-uniform [{clr_lo:g}, {clr_hi:g}]  (ln_lr tied)")
    if sweep_tlr:
        print(f"[SPACE] transfer_lr   log-uniform [{tlr_lo:g}, {tlr_hi:g}]")
    if te_choices is not None:
        print(f"[SPACE] transfer_every categorical {te_choices}")
    if sweep_flr:
        print(f"[SPACE] fast_lr       log-uniform [{flr_lo:g}, {flr_hi:g}]  "
              f"(the aux/A tile SGD LR; analog_lr cannot reach it)")
    if sweep_wmf:
        nb = args.n_bits if args.n_bits is not None else 14
        print(f"[SPACE] w_max_fast    log-uniform [{wmf_lo:g}, {wmf_hi:g}]  "
              f"(fast tile: {2 ** nb} levels fixed, dw_min_fast = {2 / 2 ** nb:g} * w_max_fast "
              f"-> [{2 / 2 ** nb * wmf_lo:g}, {2 / 2 ** nb * wmf_hi:g}])")

    def objective(trial):
        trial_args = argparse.Namespace(**vars(args))
        trial_args._optuna_trial = trial

        trial_args.analog_lr = trial.suggest_float("analog_lr", alr_lo, alr_hi, log=True)
        trial_args.classifier_lr = trial.suggest_float("classifier_lr", clr_lo, clr_hi, log=True)
        trial_args.ln_lr = trial_args.classifier_lr  # keep LayerNorm LR tied to classifier LR

        if sweep_tlr:
            trial_args.transfer_lr = trial.suggest_float("transfer_lr", tlr_lo, tlr_hi, log=True)
        if te_choices is not None:
            trial_args.transfer_every = float(
                trial.suggest_categorical("transfer_every", te_choices))
        if sweep_flr:
            trial_args.fast_lr = trial.suggest_float("fast_lr", flr_lo, flr_hi, log=True)
        if sweep_wmf:
            trial_args.w_max_fast = trial.suggest_float(
                "w_max_fast", wmf_lo, wmf_hi, log=True)

        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
        trial_args.output_dir = trial_dir

        try:
            f1 = train_fixed(trial_args)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return f1

    n_trials = args.n_trials if args.n_trials > 0 else 20
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")


# ============================================================================
# Config dump
# ============================================================================

def _save_config(args, rpu_config, effective_dw_min, num_training_steps):
    """Save full config as JSON."""
    config = {
        "method": args.method,
        "target_layers": args.target_layers,
        "seed": args.seed,
        "model": MODEL_NAME,
        "task": "squad_v1.1",
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "analog_lr": args.analog_lr,
        "classifier_lr": args.classifier_lr,
        "ln_lr": args.ln_lr,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_rate": args.min_lr_rate,
        "dw_min": effective_dw_min,
        "n_bits": args.n_bits,
        "n_bits_slow": args.n_bits_slow,
        "desired_bl": args.desired_bl,
        "count_pulses": args.count_pulses,
        "num_training_steps": num_training_steps,
        "rpu_config_type": type(rpu_config).__name__,
        "io_bits": args.io_bits,
        "io_perfect": (args.io_bits == 0),
        "io_res": io_res_from_bits(args.io_bits) if args.io_bits > 0 else None,
        "noise_management": args.noise_management,
        # Fixed label; the configured bound management of every path is in
        # io_resolved.
        "bound_management": "none" if args.io_bits > 0 else "n/a",
        # Per-path IO, read back from the built config.
        "io_resolved": _resolved_io_dict(rpu_config),
        "learn_out_scaling": False,
        "w_max": args.w_max,
        "omega": args.omega,
    }

    if args.method == "single_rpu":
        config["pulse_type"] = args.pulse_type
        config["device_type"] = args.device_type
        if args.device_type != "constant_step":
            config["ls_gamma_up_ratio"] = args.ls_gamma_up_ratio
            config["ls_gamma_down_ratio"] = args.ls_gamma_down_ratio
            config["ls_noise_ratio"] = args.ls_noise_ratio
            config["ls_gamma_up"] = args.ls_gamma_up
            config["ls_gamma_down"] = args.ls_gamma_down
            if args.ls_dw_min_std is not None:
                config["ls_dw_min_std"] = args.ls_dw_min_std
            if args.ls_dw_min_dtod is not None:
                config["ls_dw_min_dtod"] = args.ls_dw_min_dtod

    if args.method == "ttv1":
        config["gamma"] = args.gamma
        config["transfer_every"] = args.transfer_every
        config["units_in_mbatch"] = args.units_in_mbatch
        config["fast_lr"] = args.fast_lr
        config["transfer_lr"] = args.transfer_lr
        config["scale_transfer_lr"] = args.scale_transfer_lr
        config["n_reads_per_transfer"] = args.n_reads_per_transfer
        config["with_reset_prob"] = args.with_reset_prob
        config["transfer_bl"] = args.transfer_bl
        config["ttv1_mode"] = args.ttv1_mode
        config["ttv1_fast_pulse_type"] = args.ttv1_fast_pulse_type
        config["ttv1_transfer_pulse_type"] = args.ttv1_transfer_pulse_type
        config["w_max_fast"] = args.w_max_fast
        # Fast-tile device. Recorded next to device_type_slow so that a 6T1C
        # (linear_step) A tile is distinguishable from a constant_step one in
        # config.json, together with the asymmetry and noise ratios that define
        # the device.
        config["device_type"] = args.device_type
        if args.device_type != "constant_step":
            config["ls_gamma_up_ratio"] = args.ls_gamma_up_ratio
            config["ls_gamma_down_ratio"] = args.ls_gamma_down_ratio
            config["ls_noise_ratio"] = args.ls_noise_ratio
            config["ls_gamma_up"] = args.ls_gamma_up
            config["ls_gamma_down"] = args.ls_gamma_down
        if args.ls_dw_min_std is not None:
            config["ls_dw_min_std"] = args.ls_dw_min_std
        if args.ls_dw_min_dtod is not None:
            config["ls_dw_min_dtod"] = args.ls_dw_min_dtod
        config["device_type_slow"] = args.device_type_slow
        if args.device_type_slow != "constant_step":
            config["ls_gamma_up_slow"] = args.ls_gamma_up_slow
            config["ls_gamma_down_slow"] = args.ls_gamma_down_slow
            config["ls_noise_ratio_slow"] = args.ls_noise_ratio_slow

    config["eval_c_only"] = args.eval_c_only
    config["diag_update_exact"] = args.diag_update_exact
    config["diag_steps"] = args.diag_steps

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper experiment: BERT-base + SQuAD v1.1 analog fine-tuning")

    # Mode
    parser.add_argument("--mode", type=str, default="fixed",
                        choices=["fixed", "grid", "tpe"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_rpu", "ttv1", "ideal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-layers", type=str, default="attention",
                        choices=["attention", "ffn", "all", "all_train_attention",
                                 "all_train_ffn"],
                        help="Which encoder layers to convert: attention(QKVO), ffn, all, "
                             "all_train_attention (convert all to analog but only train "
                             "attention), all_train_ffn (convert all to analog but only "
                             "train ffn). The all_train_* variants keep every encoder "
                             "linear on the same analog device + IO and vary only which "
                             "group gets weight updates; plain attention/ffn keep the "
                             "other group in digital FP32, without device or IO model.")
    parser.add_argument("--output-dir", type=str, required=True)

    # Training
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--analog-lr", type=float, default=0.016)
    parser.add_argument("--classifier-lr", type=float, default=0.003)
    parser.add_argument("--ln-lr", type=float, default=None,
                        help="LayerNorm LR (default: same as --analog-lr; in tpe "
                             "mode every trial ties it to classifier_lr)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-rate", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stopping patience in epochs (0 = disabled). "
                             "Stops a run/trial after N consecutive epochs without "
                             "best-F1 improvement.")

    # SingleRPU
    parser.add_argument("--pulse-type", type=str, default="stochastic",
                        choices=list(PULSE_TYPE_MAP.keys()))
    parser.add_argument("--desired-bl", type=int, default=31)
    parser.add_argument("--dw-min", type=float, default=None)
    parser.add_argument("--n-bits", type=int, default=None)
    parser.add_argument("--n-bits-slow", type=int, default=None,
                        help="TTv1 slow tile bit-width (default: same as --n-bits)")
    parser.add_argument("--w-max", type=float, default=1.0,
                        help="Device weight bound [-w_max, w_max]. Default 1.0. "
                             "For single_rpu: sets tile w_max directly.")
    parser.add_argument("--omega", type=float, default=None,
                        help="weight_scaling_omega. Default: 1.0 for w_max=1.0, "
                             "0.0 for w_max<1.0 (disable scaling so raw weights go to tile).")
    parser.add_argument("--io-bits", type=int, default=0,
                        help="DAC/ADC bit precision of the forward, backward and "
                             "(ttv1) transfer-read IO. "
                             "0 = perfect (no quantization). "
                             "Typical values: 4, 6, 8, 10, 12. "
                             "ADC=DAC on every path.")
    parser.add_argument("--noise-management", type=str, default="abs_max",
                        choices=["abs_max", "none"],
                        help="IO noise management: abs_max (scale input by max abs) or none.")
    parser.add_argument("--out-noise", type=float, default=0.0,
                        help="Analog output (ADC) noise std on every IO path, used by "
                             "single_rpu and ttv1 (the other methods run with 0.0). "
                             "Default 0.0. Only active when --io-bits > 0. e.g. 0.04.")
    parser.add_argument("--per-layer-bits", type=str, default=None,
                        help="Per-layer IO bit allocation. "
                             "Shorthand: 'FFN1=12,FFN2=6,K=10,O=6,Q=8,V=8' (all layers same). "
                             "Per-layer: 'L0:FFN1=12,K=8;L1:FFN1=10,K=10'. "
                             "Or JSON file path. Sets the forward and backward DAC/ADC "
                             "resolution of the matched layers in place of --io-bits "
                             "(use with --io-bits > 0).")

    # TTv1
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--transfer-every", type=float, default=None)
    parser.add_argument("--units-in-mbatch", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--no-uim", action="store_true",
                        help="Shorthand for --units-in-mbatch false")
    parser.add_argument("--fast-lr", type=float, default=None)
    parser.add_argument("--transfer-lr", type=float, default=None)
    parser.add_argument("--scale-transfer-lr", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--n-reads-per-transfer", type=int, default=None)
    parser.add_argument("--with-reset-prob", type=float, default=None)
    parser.add_argument("--transfer-bl", type=int, default=None)
    parser.add_argument("--w-max-fast", type=float, default=None,
                        help="TTv1 fast tile weight bound [-w_max_fast, w_max_fast]. "
                             "The fast-tile step scales with it (dw_min * w_max_fast, "
                             "i.e. 2*w_max_fast/2^n_bits with --n-bits), so the level "
                             "count stays the same. The fast tile starts at 0.")

    # TTv1 mode presets
    parser.add_argument("--ttv1-mode", type=str, default=None,
                        choices=["hidden_buffer", "residual_lane", "residual_lane_noreset"])

    # TTv1 pulse controls
    parser.add_argument("--ttv1-fast-pulse-type", type=str, default=None,
                        choices=list(PULSE_TYPE_MAP.keys()))
    parser.add_argument("--ttv1-transfer-pulse-type", type=str, default=None,
                        choices=list(PULSE_TYPE_MAP.keys()))

    # Device type (LinearStep support)
    parser.add_argument("--device-type", type=str, default="constant_step",
                        choices=["constant_step", "linear_step", "exp_step", "soft_bounds"],
                        help="Analog device model: constant_step (ideal), linear_step (ECRAM/6T1C), exp_step (RRAM-ES), or soft_bounds (RRAM-HfO2)")
    parser.add_argument("--ls-gamma-up-ratio", type=float, default=1.0,
                        help="LinearStep gamma_up scale (1.0 = 6T1C measured -0.1678)")
    parser.add_argument("--ls-gamma-down-ratio", type=float, default=1.0,
                        help="LinearStep gamma_down scale (1.0 = 6T1C measured 0.1410)")
    parser.add_argument("--ls-noise-ratio", type=float, default=0.0,
                        help="LinearStep noise scale (0=noise-free, 1.0=6T1C measured)")
    parser.add_argument("--ls-gamma-up", type=float, default=None,
                        help="Absolute gamma_up value (overrides ratio mode). E.g. 0.1153 for EcRam")
    parser.add_argument("--ls-gamma-down", type=float, default=None,
                        help="Absolute gamma_down value (overrides ratio mode). E.g. 0.5085 for EcRam")
    parser.add_argument("--ls-dw-min-std", type=float, default=None,
                        help="Override dw_min_std (cycle-to-cycle noise) of the ttv1 "
                             "LinearStep fast tile, in place of the --ls-noise-ratio scaling.")
    parser.add_argument("--ls-dw-min-dtod", type=float, default=None,
                        help="Override dw_min_dtod (device-to-device) of the ttv1 "
                             "LinearStep fast tile; the other dtod parameters keep their "
                             "--ls-noise-ratio scaling.")

    # Slow tile device type (TTv1)
    parser.add_argument("--device-type-slow", type=str, default="constant_step",
                        choices=["constant_step", "linear_step"],
                        help="Slow tile device model (default: constant_step)")
    parser.add_argument("--ls-gamma-up-slow", type=float, default=None,
                        help="Slow tile absolute gamma_up (LinearStep)")
    parser.add_argument("--ls-gamma-down-slow", type=float, default=None,
                        help="Slow tile absolute gamma_down (LinearStep)")
    parser.add_argument("--ls-noise-ratio-slow", type=float, default=0.0,
                        help="Slow tile noise scale (0=noise-free)")

    # Evaluation
    parser.add_argument("--eval-c-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate with C tile only (slow weights) in addition to A+C. "
                             "Only applies to TTv1 with gamma>0. Default: enabled. "
                             "Use --no-eval-c-only to disable.")

    # Control
    parser.add_argument("--count-pulses", action="store_true")

    # Diagnostics
    parser.add_argument("--diag-update-exact", action="store_true")
    parser.add_argument("--diag-steps", type=int, default=0,
                        help="Record diagnostics for first N steps (0=all logged steps)")
    parser.add_argument("--diag-at-steps", type=str, default=None,
                        help="Comma-separated list of specific steps to diagnose (e.g. 1,16,32)")
    parser.add_argument("--diag-layer-set", type=str, default=None,
                        help="Comma-separated encoder layer indices to diagnose (e.g. 0,5,11). Default: all layers.")

    # Grid/TPE
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--db-dir", type=str, default=None)
    parser.add_argument("--tpe-startup-trials", type=int, default=None,
                        help="TPESampler n_startup_trials (random trials before the "
                             "TPE model engages). Default None = Optuna's default 10. "
                             "A lower value (e.g. 4) suits studies of 10-20 trials.")

    # --- Explicit TPE search ranges (mode=tpe) ---------------------------------
    # Without these the band is derived from the anchor (alr/clr: anchor/3..anchor*3,
    # transfer_lr: anchor/10..anchor*10, transfer_every: {0.25,0.5,1,2,4}x anchor),
    # which ties how wide the search is to where it starts. Passing an explicit
    # range decouples the two: anchor = where trial 0 starts, range = how far it roams.
    parser.add_argument("--alr-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit analog_lr search range (log-uniform).")
    parser.add_argument("--clr-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit classifier_lr search range (log-uniform). "
                             "ln_lr stays tied to classifier_lr.")
    parser.add_argument("--tlr-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit transfer_lr search range (log-uniform).")
    parser.add_argument("--te-choices", type=str, default=None,
                        help="Explicit transfer_every choices, comma-separated ints "
                             "(e.g. '1,2,3,6,12,22,45,90').")
    parser.add_argument("--flr-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit fast_lr search range (log-uniform, ttv1 only). "
                             "fast_lr is the aux/A tile's SGD learning rate: the "
                             "TransferCompound uses it in place of the optimizer LR "
                             "whenever fast_lr > 0 (aihwkit default 1.0), so "
                             "--analog-lr does not act on the ttv1 tiles and fast_lr "
                             "sets the aux tile's step size. "
                             "Pin analog_lr with '--alr-range X X' when using this.")
    parser.add_argument("--wmf-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit w_max_fast search range (log-uniform, ttv1 only). "
                             "The fast/aux tile keeps 2**n_bits levels regardless of "
                             "w_max_fast; the knob is the range, so dw_min_fast = "
                             "(2/2**n_bits) * w_max_fast. Together with --n-bits this "
                             "reproduces the LRTT A/B parameterisation "
                             "w_max = 2**multilevel * dw_min / 2 (w_min = -w_max): "
                             "n_bits <-> multilevel, w_max_fast <-> w_max. "
                             "Omit to leave the fast tile at the --w-max-fast value "
                             "(or +-1.0 when that is unset).")

    args = parser.parse_args()

    # Parse --diag-at-steps into a set
    args.diag_at_steps_set = None
    if args.diag_at_steps:
        args.diag_at_steps_set = set(int(s.strip()) for s in args.diag_at_steps.split(","))

    # Parse --diag-layer-set into a set
    if args.diag_layer_set is not None:
        args.diag_layer_set = set(int(s.strip()) for s in args.diag_layer_set.split(","))

    # Handle --no-uim shorthand
    if args.no_uim:
        args.units_in_mbatch = "false"

    # Default ln_lr to analog_lr if not specified
    if args.ln_lr is None:
        args.ln_lr = args.analog_lr

    # TTv1 mode presets - apply defaults only where user didn't override
    TTv1_PRESETS = {
        "hidden_buffer":         {"gamma": 0.0, "with_reset_prob": 1.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
        "residual_lane":         {"gamma": 1.0, "with_reset_prob": 1.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
        "residual_lane_noreset": {"gamma": 1.0, "with_reset_prob": 0.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
    }

    if args.ttv1_mode is not None and args.method == "ttv1":
        preset = TTv1_PRESETS[args.ttv1_mode]
        for key, value in preset.items():
            if getattr(args, key) is None:
                setattr(args, key, value)

    # Default gamma to 0.0 if still None
    if args.gamma is None:
        args.gamma = 0.0

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    global TARGET_LAYERS
    args = parse_args()
    TARGET_LAYERS = args.target_layers

    print(f"{'='*60}")
    print(f"Paper Experiment: {args.method} (mode={args.mode})")
    print(f"{'='*60}")
    print(f"Target layers: {TARGET_LAYERS}")

    if args.mode == "fixed":
        train_fixed(args)
    elif args.mode == "grid":
        train_grid(args)
    elif args.mode == "tpe":
        train_tpe(args)


if __name__ == "__main__":
    main()
