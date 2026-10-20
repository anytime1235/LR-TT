# -*- coding: utf-8 -*-
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""Digital LoRA baseline for BERT-base on SQuAD v1.1 (no aihwkit).

LoRA (rank 32, alpha 64 by default) applied to target layers (qkvo/ffn/all):
    - LoRA A/B matrices + qa_outputs + LayerNorm = trainable
    - Base Linear weights + bias + Embeddings = frozen
Optuna searches the learning rate (TPE sampler, seed 42, median pruner).

Settings shared with the LR-TT studies:
    - batch_size=48, epochs=5, warmup_steps=365, maximum sequence length 384
    - linear LR decay to zero, no weight decay
Optimizer: Adam with betas (0, 0.999). Training (with gradient scaling) and
evaluation run under fp16 autocast.

Usage:
    python optuna_bert_squad_digital_lora.py --n-trials 10 --lora-target qkvo
    python optuna_bert_squad_digital_lora.py --n-trials 10 --lora-target ffn
    python optuna_bert_squad_digital_lora.py --n-trials 10 --lora-target all
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import re
import math
import string
import argparse
import collections
from collections import Counter

import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
import evaluate


# =============================================================================
# Constants
# =============================================================================
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
EPOCHS = 5
WARMUP_STEPS = 365
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_RANK = 32
LORA_ALPHA = 64  # scaling = alpha / rank = 2.0 (paper studies: r32a64)

# Same target layers as the LR-TT scripts
LORA_TARGETS = {
    "qkvo": (["query", "key", "value", "attention.output"], None),
    "ffn":  (["intermediate", "output.dense"], ["attention"]),
    "all":  (None, None),
}


# =============================================================================
# LoRA
# =============================================================================

class LoRALinear(nn.Module):
    """Standard LoRA: base Linear frozen + trainable low-rank A/B delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays at zero -> LoRA output starts at zero (standard init)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def _layer_matches_target(name: str, includes, excludes):
    """Same layer selection as get_lrtt_target_module_names of the LR-TT scripts."""
    if "qa_outputs" in name:
        return False
    if "encoder" not in name:
        return False
    if includes is None:
        return True
    # excludes take precedence
    if excludes is not None:
        if any(ex in name for ex in excludes):
            return False
    return any(inc in name for inc in includes)


def inject_lora(model: nn.Module, lora_target: str, rank: int, alpha: int):
    includes, excludes = LORA_TARGETS[lora_target]
    replaced_names = []

    def _recurse(module, prefix):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _layer_matches_target(full_name, includes, excludes):
                setattr(module, child_name, LoRALinear(child, rank, alpha))
                replaced_names.append(full_name)
            else:
                _recurse(child, full_name)

    _recurse(model, "")
    return replaced_names


def set_trainable(model: nn.Module):
    """LoRA A/B + qa_outputs + LayerNorm trainable; everything else frozen."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif "qa_outputs" in name:
            param.requires_grad = True
        elif "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


# =============================================================================
# Data (same preprocessing as the LR-TT scripts)
# =============================================================================

def load_data(tokenizer):
    raw_datasets = load_dataset("squad")
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
        start_positions, end_positions = [], []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0); end_positions.append(0); continue
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
                start_positions.append(0); end_positions.append(0)
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
# Eval
# =============================================================================

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


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
        prelim = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for si in start_indexes:
                for ei in end_indexes:
                    if (si >= len(offset_mapping) or ei >= len(offset_mapping)
                            or offset_mapping[si] is None or offset_mapping[ei] is None):
                        continue
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    prelim.append({
                        "offsets": (offset_mapping[si][0], offset_mapping[ei][1]),
                        "score": start_logits[si] + end_logits[ei],
                    })
        predictions = sorted(prelim, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best = predictions[0]
            start_char, end_char = best["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]
    return all_predictions


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    model.eval()
    all_start, all_end = [], []
    collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH)

    def squad_collate(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = collator(features)
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             collate_fn=squad_collate, num_workers=2)

    with no_grad(), autocast(dtype=torch.float16):
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start.append(outputs.start_logits.float().cpu().numpy())
            all_end.append(outputs.end_logits.float().cpu().numpy())
    model.train()
    all_start = np.concatenate(all_start, axis=0)
    all_end = np.concatenate(all_end, axis=0)
    predictions = postprocess_squad_predictions(eval_examples, eval_features, all_start, all_end)
    formatted = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted, references=references)
    return results["f1"], results["exact_match"]


# =============================================================================
# Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Objective
# =============================================================================

def objective(trial, lora_target, train_loader, eval_features, eval_examples, tokenizer):
    set_seed(SEED)

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Build model: LoRA injection + trainability
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    replaced = inject_lora(model, lora_target, LORA_RANK, LORA_ALPHA)
    set_trainable(model)
    model = model.to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("lora_layers_replaced", len(replaced))
    trial.set_user_attr("trainable_params", n_trainable)
    trial.set_user_attr("total_params", n_total)

    # Adam with beta1 = 0 (no first-moment momentum), no weight decay
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.0, 0.999), eps=1e-8, weight_decay=0.0)

    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_min_lr(optimizer, WARMUP_STEPS, total_steps, min_lr_rate=0.0)

    scaler = GradScaler()

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_positions"].to(DEVICE)
            end_positions = batch["end_positions"].to(DEVICE)

            with autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item() * GRAD_ACCUM_STEPS
            train_count += 1

        avg_loss = train_loss_sum / max(1, train_count)
        trial.set_user_attr(f"train_loss_epoch_{epoch+1}", avg_loss)

        f1, em = evaluate_model(model, eval_features, eval_examples, tokenizer)
        trial.set_user_attr(f"em_epoch_{epoch+1}", em)
        best_f1 = max(best_f1, f1)

        trial.report(f1, epoch + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_f1


# =============================================================================
# Main
# =============================================================================

def main():
    global LORA_RANK, LORA_ALPHA  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description="Optuna learning-rate search for digital LoRA fine-tuning of BERT-base "
                    "on SQuAD v1.1 (journal: ./results/optuna_bert_squad_digital_lora/<study>.log)")
    parser.add_argument("--lora-target", type=str, required=True, choices=["qkvo", "ffn", "all"],
                        help="Sublayers that receive LoRA adapters")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of trials to run (default: 10)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Study name (default: derived from batch size, rank, alpha and target)")
    parser.add_argument("--rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA,
                        help=f"LoRA alpha; the adapter output is scaled by alpha / rank (default: {LORA_ALPHA})")
    args = parser.parse_args()

    LORA_RANK = args.rank
    LORA_ALPHA = args.alpha

    study_name = args.study_name or (
        f"bert_squad_digital_lora_bs{BATCH_SIZE}_adam_nowd_nomom_nonest_"
        f"r{args.rank}a{args.alpha}_{args.lora_target}_{EPOCHS}ep"
    )
    results_dir = os.path.join(os.getcwd(), "results", "optuna_bert_squad_digital_lora")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{study_name}.log")

    print(f"Study: {study_name}")
    print(f"Log:   {log_path}")
    print(f"Target: {args.lora_target}  rank={args.rank}  alpha={args.alpha}  scale={args.alpha/args.rank:.3f}")
    print(f"Device: {DEVICE}  |  bs={BATCH_SIZE}  epochs={EPOCHS}  warmup={WARMUP_STEPS}")

    # Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train features: {len(train_loader.dataset):,}  Eval features: {len(eval_features):,}")

    storage = JournalStorage(JournalFileBackend(log_path))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=3),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, args.lora_target, train_loader, eval_features, eval_examples, tokenizer),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print("\n===== Best =====")
    print(f"F1: {study.best_value:.4f}")
    print(f"Params: {study.best_params}")


if __name__ == "__main__":
    main()
