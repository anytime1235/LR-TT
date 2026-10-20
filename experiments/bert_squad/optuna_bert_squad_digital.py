# -*- coding: utf-8 -*-
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""Digital full fine-tuning baseline for BERT-base on SQuAD v1.1 (no aihwkit).

The linear layers of the target sublayers (qkvo/ffn/all), the QA head
(qa_outputs) and the LayerNorm parameters are fine-tuned in FP32; all other
parameters are frozen. Optuna searches the learning rate (TPE sampler, seed 42).

Settings shared with the LR-TT studies:
    - batch_size=48, epochs=5, warmup_steps=365, maximum sequence length 384
    - linear LR schedule decaying to zero, no weight decay
Optimizer: AdamW with betas (0, 0.999) and weight decay 0, i.e. Adam with beta1 = 0.

Usage:
    python optuna_bert_squad_digital.py --n-trials 10 --lora-target qkvo
    python optuna_bert_squad_digital.py --n-trials 10 --lora-target ffn
    python optuna_bert_squad_digital.py --n-trials 10 --lora-target all
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import re
import string
import math
import gc
import collections
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import numpy as np

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from optuna.samplers import TPESampler

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
import evaluate
from collections import Counter


# =============================================================================
# Config
# =============================================================================
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384

# Training (batch size, epochs and schedule of the LR-TT studies)
N_EPOCHS = 5
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
WARMUP_STEPS = 365
WEIGHT_DECAY = 0.0
BETA1 = 0.0           # no first-moment momentum
BETA2 = 0.999
MIN_LR_RATE = 0.0

EARLY_STOP_PATIENCE = 2
TRAIN_LOSS_EARLY_STOP_PATIENCE = 1
TRAIN_LOSS_THRESHOLD = 1.5

LORA_TARGET_PATTERNS = {
    "qkvo": ["query", "key", "value", "attention.output"],
    "ffn":  ["intermediate", "output.dense"],  # FFN = intermediate + output.dense (non-attention)
    "all":  None,  # all encoder linear layers
}

RESULTS = os.path.join(os.getcwd(), "results", "optuna_bert_squad_digital")


# =============================================================================
# Model setup
# =============================================================================
def get_target_linear_names(model, lora_target):
    """Return list of parameter name prefixes whose Linear layers should be trainable."""
    if lora_target == "all":
        # All encoder linear layers
        names = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "encoder" in name:
                names.append(name)
        return names

    patterns = LORA_TARGET_PATTERNS[lora_target]
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "encoder" not in name:
            continue
        if lora_target == "qkvo":
            # query, key, value, attention.output.dense
            if any(p in name for p in ["attention.self.query", "attention.self.key",
                                       "attention.self.value", "attention.output.dense"]):
                names.append(name)
        elif lora_target == "ffn":
            # intermediate.dense, output.dense (but NOT attention.output.dense)
            if ("intermediate.dense" in name) or ("output.dense" in name and "attention" not in name):
                names.append(name)
    return names


def setup_model(lora_target):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    target_names = get_target_linear_names(model, lora_target)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze target linears
    for name, mod in model.named_modules():
        if name in target_names and isinstance(mod, nn.Linear):
            for p in mod.parameters():
                p.requires_grad = True

    # Unfreeze qa_outputs (head) and LayerNorm
    for name, p in model.named_parameters():
        if "qa_outputs" in name or "LayerNorm" in name:
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Target '{lora_target}': {len(target_names)} linear layers")
    print(f"  Trainable: {n_train:,} / {n_total:,}")
    return model


# =============================================================================
# Data
# =============================================================================
def load_data(tokenizer):
    raw = load_dataset("squad")

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
            while sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0); end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
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
        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]
        return inputs

    tokenized_train = raw["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw["train"].column_names,
    )
    train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = raw["validation"].map(
        preprocess_eval, batched=True,
        remove_columns=raw["validation"].column_names,
    )

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=2,
        generator=torch.Generator().manual_seed(SEED),
    )
    return train_loader, tokenized_eval, raw["validation"]


# =============================================================================
# Eval
# =============================================================================
def normalize_answer(s):
    def rm_arts(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def wsf(t): return ' '.join(t.split())
    def rm_punc(t): return ''.join(ch for ch in t if ch not in set(string.punctuation))
    return wsf(rm_arts(rm_punc(s.lower())))


def postprocess(examples, features, all_start_logits, all_end_logits,
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
            for si in start_indexes:
                for ei in end_indexes:
                    if (si >= len(offset_mapping) or ei >= len(offset_mapping) or
                        offset_mapping[si] is None or offset_mapping[ei] is None):
                        continue
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[si][0], offset_mapping[ei][1]),
                        "score": start_logits[si] + end_logits[ei],
                    })
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            start_char, end_char = predictions[0]["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]
    return all_predictions


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    model.eval()
    all_s, all_e = [], []
    collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH)

    def eval_collate(features):
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
        collate_fn=eval_collate, num_workers=2,
    )
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_s.append(out.start_logits.cpu().numpy())
            all_e.append(out.end_logits.cpu().numpy())
    model.train()
    all_s = np.concatenate(all_s, axis=0)
    all_e = np.concatenate(all_e, axis=0)
    predictions = postprocess(eval_examples, eval_features, all_s, all_e)
    formatted = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    m = evaluate.load("squad")
    r = m.compute(predictions=formatted, references=refs)
    return r["f1"], r["exact_match"]


# =============================================================================
# Scheduler (as in the LR-TT scripts: linear warmup, linear decay to min_lr_rate)
# =============================================================================
def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Objective
# =============================================================================
LORA_TARGET = None  # set via argparse


def objective(trial, train_loader, eval_features, eval_examples, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameter: learning rate only (other settings fixed, see the module docstring)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    trial.set_user_attr("lora_target", LORA_TARGET)
    trial.set_user_attr("batch_size", BATCH_SIZE)
    trial.set_user_attr("epochs", N_EPOCHS)
    trial.set_user_attr("warmup_steps", WARMUP_STEPS)

    print(f"\n{'='*60}\nTrial {trial.number}\n{'='*60}")
    print(f"  lr={learning_rate:.3e}, target={LORA_TARGET}")

    set_seed(SEED)
    model = setup_model(LORA_TARGET)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate,
        betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY,
    )

    num_training_steps = len(train_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer, WARMUP_STEPS, num_training_steps, MIN_LR_RATE,
    )

    best_f1 = 0.0
    best_em = 0.0
    no_improve = 0
    prev_train_loss = float('inf')
    loss_over_threshold_count = 0

    try:
        for epoch in range(1, N_EPOCHS + 1):
            total_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Ep{epoch}", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(1, n_batches)
            f1, em = evaluate_model(model, eval_features, eval_examples, tokenizer)
            trial.report(f1, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", avg_loss)
            trial.set_user_attr(f"f1_epoch_{epoch}", f1)
            trial.set_user_attr(f"em_epoch_{epoch}", em)

            improved = f1 > best_f1
            if improved:
                best_f1 = f1
                best_em = em
                no_improve = 0
            else:
                no_improve += 1

            cur_lr = optimizer.param_groups[0]['lr']
            mark = '(lower)' if avg_loss < prev_train_loss else ''
            star = '*' if improved else ''
            print(f"[Ep{epoch}] Train loss: {avg_loss:.4f} {mark} | F1: {f1:.2f}% | EM: {em:.2f}% | "
                  f"Best F1: {best_f1:.2f}% | LR: {cur_lr:.2e} | No imp: {no_improve}/{EARLY_STOP_PATIENCE} {star}")

            # Train loss early stop
            if avg_loss > TRAIN_LOSS_THRESHOLD:
                loss_over_threshold_count += 1
                if loss_over_threshold_count >= TRAIN_LOSS_EARLY_STOP_PATIENCE and epoch >= 2:
                    print(f"  Early stop: train loss stuck above {TRAIN_LOSS_THRESHOLD}")
                    break
            else:
                loss_over_threshold_count = 0

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stop: F1 no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

            prev_train_loss = avg_loss

            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trial.set_user_attr("best_em", best_em)
    return best_f1


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optuna learning-rate search for digital full fine-tuning of BERT-base "
                    "on SQuAD v1.1 (journal: ./results/optuna_bert_squad_digital/<study>.log)")
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials to run (default: 10)')
    parser.add_argument('--lora-target', type=str, default='qkvo',
                        choices=['qkvo', 'ffn', 'all'],
                        help='Fine-tuned sublayers (default: qkvo)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: bert_squad_digital_<target>_5ep)')
    args = parser.parse_args()

    global LORA_TARGET
    LORA_TARGET = args.lora_target

    study_name = args.study_name or f"bert_squad_digital_{LORA_TARGET}_{N_EPOCHS}ep"
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, f"{study_name}.log")
    print(f"Study: {study_name}\nLog: {log_path}")

    storage = JournalStorage(JournalFileBackend(log_path))
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction='maximize', load_if_exists=True,
        sampler=TPESampler(seed=SEED),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)

    study.optimize(
        lambda t: objective(t, train_loader, eval_features, eval_examples, tokenizer),
        n_trials=args.n_trials,
    )

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete:
        best = max(complete, key=lambda t: t.value)
        print(f"\nBest: F1={best.value:.4f}, lr={best.params['learning_rate']:.3e}")


if __name__ == "__main__":
    main()
