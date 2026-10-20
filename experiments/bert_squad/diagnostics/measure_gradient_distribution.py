# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Measure the backward-gradient element distributions of all sublayer types.

Uses paper_experiment.py create_model() with method=ideal, target_layers=all,
IO perfect. Fixed LR (no warmup/scheduler). Captures gradient histograms
(log-scale bins) for Q, K, V, O, FFN1, FFN2 over full 1 epoch.

Histogram approach: 500 log-spaced bins from 10^-7 to 10^-2 for the nonzero
elements, plus underflow and overflow bins, accumulated per step.
Memory: ~288KB total regardless of epoch length.

Outputs (in --out-dir): grad_histograms.npz (per-layer histograms),
grad_summary.csv (summary statistics every 50 steps), grad_raw_samples.npz
(raw gradient elements of the first 5 steps, at most 100k per tensor) and
grad_pdf_log_all.png (probability density of log10|g| per sublayer type,
pooled over the 12 layers; Fig. 5c).
"""
import os
import sys
# paper_experiment.py and its local imports live in ../tikitaka
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tikitaka"))

import re
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, set_seed, default_data_collator

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext

# Import from paper_experiment directly
from paper_experiment import (
    MODEL_NAME, create_model, load_data, create_optimizer,
    is_target_layer, get_layer_subtype, TARGET_LAYERS,
)

# -- Config --
SEED = 42
BATCH_SIZES = [16, 8]  # tried in order; a CUDA error at one batch size starts the epoch at the next
ANALOG_LR = 0.0357
CLASSIFIER_LR = 0.00076
LN_LR = 0.00076
NUM_STEPS = 0  # 0 = full 1 epoch
_cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_cli.add_argument("--out-dir", type=str, default="./results/gradient_distribution")
OUT_DIR = _cli.parse_args().out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# Histogram config: 500 log-spaced bins covering 10^-7 to 10^-2 (5 decades, 100 bins/decade)
HIST_LOG_MIN = -7.0
HIST_LOG_MAX = -2.0
HIST_N_BINS = 500
HIST_BIN_EDGES = np.linspace(HIST_LOG_MIN, HIST_LOG_MAX, HIST_N_BINS + 1)
# Extra bins for underflow (<10^-7) and overflow (>10^-2)

# Log summary stats every N steps
LOG_SUMMARY_EVERY = 50
# Save raw gradient samples for first N steps (for detailed per-element plots)
RAW_SAMPLE_STEPS = 5

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -- Sublayer classification --
def get_sublayer(name):
    if 'query' in name: return 'Q'
    if 'key' in name: return 'K'
    if 'value' in name: return 'V'
    if 'attention.output.dense' in name: return 'O'
    if 'intermediate.dense' in name: return 'FFN1'
    if '.output.dense' in name and 'attention' not in name: return 'FFN2'
    return 'other'

def get_layer_idx(name):
    m = re.search(r'layer\.(\d+)', name)
    return int(m.group(1)) if m else -1


# -- Create args namespace matching paper_experiment expectations --
args = argparse.Namespace(
    method="ideal",
    target_layers="all",
    seed=SEED,
    model="bert-base-uncased",
    task="squad_v1.1",
    batch_size=BATCH_SIZES[0],
    grad_accum_steps=1,
    epochs=1,
    max_steps=0,
    analog_lr=ANALOG_LR,
    classifier_lr=CLASSIFIER_LR,
    ln_lr=LN_LR,
    warmup_ratio=0.0,
    min_lr_rate=1.0,
    # Ideal: no bits needed
    n_bits=None,
    dw_min=None,
    desired_bl=31,
    count_pulses=False,
    # No IO quantization (perfect)
    io_bits=0,
    noise_management="abs_max",
    out_noise=0.0,
    learn_out_scaling=False,
    per_layer_bits=None,
    # Not used for ideal
    gamma=1.0,
    transfer_every=None,
    units_in_mbatch=None,
    fast_lr=None,
    transfer_lr=None,
    scale_transfer_lr=None,
    n_reads_per_transfer=None,
    with_reset_prob=None,
    transfer_bl=None,
    ttv1_mode=None,
    ttv1_fast_pulse_type=None,
    ttv1_transfer_pulse_type=None,
    n_bits_slow=None,
    pulse_type="stochastic",
    device_type="constant_step",
    ls_gamma_up_ratio=1.0,
    ls_gamma_down_ratio=1.0,
    ls_noise_ratio=0.0,
    ls_gamma_up=None,
    ls_gamma_down=None,
    # Diagnostics off
    diag_update_exact=False,
    diag_steps=0,
    output_dir=OUT_DIR,
    mode="fixed",
    # Remaining arguments read by paper_experiment.py
    w_max=1.0,
    omega=None,
    w_max_fast=None,
    no_uim=False,
    log_every=20,
    diag_at_steps=None,
    diag_layer_set=None,
    n_trials=0,
    study_name=None,
    db_dir=None,
)

# create_model reads the module-level TARGET_LAYERS: convert all 72 layers
import paper_experiment
paper_experiment.TARGET_LAYERS = "all"

# -- Model --
print("\n=== Creating model (ideal, all layers, IO perfect) ===")
model, rpu_config = create_model(args, device_str=str(device))

# -- Data (loaded later after batch size determined) --
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -- Optimizer (fixed LR, no scheduler) --
print("\n=== Creating optimizer (fixed LR, no scheduler) ===")
optimizer = create_optimizer(model, args)
print(f"  LR: analog={ANALOG_LR}, classifier={CLASSIFIER_LR}, ln={LN_LR}")

# -- Register backward hooks with histogram accumulation --
print("\n=== Registering gradient hooks ===")

# Per-layer histogram accumulator (GPU tensors, no CPU transfer during training)
grad_histograms = {}      # name -> torch.LongTensor on GPU, shape (HIST_N_BINS + 2,)
grad_summary = {}          # name -> list of per-step summary dicts
grad_raw_samples = {}      # name -> list of CPU numpy arrays (first few steps only)
grad_step_counts = {}      # name -> int
current_step = [0]

# Pre-compute bin edges on GPU for fast histogram
_bin_edges_gpu = torch.linspace(HIST_LOG_MIN, HIST_LOG_MAX, HIST_N_BINS + 1, device=device, dtype=torch.float32)

def make_bwd_hook(name):
    sub = get_sublayer(name)
    layer = get_layer_idx(name)

    def hook(module, grad_input, grad_output):
        g = grad_output[0].detach().float().flatten()  # stays on GPU
        step = current_step[0]

        abs_g = g.abs()
        nz_mask = abs_g > 0
        nz = abs_g[nz_mask]

        # -- Accumulate histogram on GPU (every step) --
        if name not in grad_histograms:
            grad_histograms[name] = torch.zeros(HIST_N_BINS + 2, dtype=torch.long, device=device)
            grad_step_counts[name] = 0

        if nz.numel() > 0:
            log_nz = torch.log10(nz)
            # bucketize with 501 edges returns indices 0..501:
            #   0 = < HIST_LOG_MIN (underflow)  -> array[0]
            #   1..500 = regular bins            -> array[1..500]
            #   501 = >= HIST_LOG_MAX (overflow) -> array[501]
            # Maps directly to array layout [underflow, bin0..bin499, overflow] (size 502)
            bin_idx = torch.bucketize(log_nz, _bin_edges_gpu)
            grad_histograms[name].scatter_add_(0, bin_idx.long(), torch.ones_like(bin_idx, dtype=torch.long))
        grad_step_counts[name] += 1

        # -- Summary stats on GPU (every LOG_SUMMARY_EVERY steps) --
        if step % LOG_SUMMARY_EVERY == 0:
            if name not in grad_summary:
                grad_summary[name] = []
            n_elem = g.numel()
            n_nz = nz.numel()
            # Sort-based percentile on GPU (torch.quantile() accepts at most 2^24 elements)
            sorted_abs, _ = abs_g.sort()
            n = sorted_abs.numel()
            def _pct(p):
                idx = min(int(p / 100.0 * n), n - 1)
                return sorted_abs[idx].item()
            grad_summary[name].append({
                'step': step, 'sublayer': sub, 'layer': layer,
                'absmax': abs_g.max().item(),
                'mean_abs': abs_g.mean().item(),
                'std': g.std().item(),
                'norm': g.norm().item(),
                'n_elements': n_elem,
                'n_nonzero': n_nz,
                'q001': _pct(0.1),
                'q01': _pct(1),
                'q10': _pct(10),
                'q50': _pct(50),
                'q90': _pct(90),
                'q99': _pct(99),
                'q999': _pct(99.9),
            })

        # -- Raw samples (first RAW_SAMPLE_STEPS steps, only here we go to CPU) --
        if step < RAW_SAMPLE_STEPS:
            if name not in grad_raw_samples:
                grad_raw_samples[name] = []
            g_cpu = g.cpu().numpy()
            if len(g_cpu) > 100000:
                idx = np.random.choice(len(g_cpu), 100000, replace=False)
                grad_raw_samples[name].append(g_cpu[idx])
            else:
                grad_raw_samples[name].append(g_cpu.copy())

    return hook

hooks = []
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        sub = get_sublayer(name)
        if sub != 'other':
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))
print(f"  Registered {len(hooks)} hooks (all 72 layers: Q/K/V/O/FFN1/FFN2 x 12)")

# -- Training loop; after a CUDA error the epoch starts at the next entry of BATCH_SIZES --
import gc

def run_epoch(batch_size):
    """Attempt 1 epoch with given batch_size. Returns (step, success)."""
    global grad_histograms, grad_summary, grad_raw_samples, grad_step_counts

    print(f"\n=== Loading data (batch_size={batch_size}) ===")
    train_loader, _, _ = load_data(tokenizer, batch_size, seed=SEED)
    total_steps = len(train_loader)
    if NUM_STEPS > 0:
        total_steps = min(total_steps, NUM_STEPS)
    print(f"  Train batches: {len(train_loader)}, steps: {total_steps}")

    print(f"\n=== Running {total_steps} steps (1 epoch, no scheduler, batch={batch_size}) ===")
    print(f"  Histogram: every step ({HIST_N_BINS} log bins, 10^{HIST_LOG_MIN}~10^{HIST_LOG_MAX})")
    print(f"  Summary stats: every {LOG_SUMMARY_EVERY} steps")
    print(f"  Raw samples: first {RAW_SAMPLE_STEPS} steps")
    start_time = time.time()

    # Reset accumulators
    grad_histograms.clear()
    grad_summary.clear()
    grad_raw_samples.clear()
    grad_step_counts.clear()

    model.train()
    step = 0
    for batch in train_loader:
        if NUM_STEPS > 0 and step >= NUM_STEPS:
            break
        current_step[0] = step

        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "caching" in err_msg or "nvml" in err_msg or "cuda" in err_msg:
                print(f"\n  CUDA error at step {step} with batch_size={batch_size}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                return step, False
            raise

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step}/{total_steps}, Loss: {loss.item():.4f}, {elapsed:.0f}s")
        step += 1

    elapsed = time.time() - start_time
    print(f"  Completed {step} steps in {elapsed:.0f}s")
    return step, True

# Try batch sizes in order
used_batch_size = None
step = 0
for bs in BATCH_SIZES:
    print(f"\n{'='*60}")
    print(f"  Trying batch_size={bs}")
    print(f"{'='*60}")
    step, success = run_epoch(bs)
    if success:
        used_batch_size = bs
        break
    else:
        print(f"  Moving to the next batch size ...")
        torch.cuda.empty_cache()
        gc.collect()

if used_batch_size is None:
    print("ERROR: CUDA error at every batch size in BATCH_SIZES. Aborting.")
    sys.exit(1)

print(f"\n  Successfully completed with batch_size={used_batch_size}")

for h in hooks:
    h.remove()

# -- Save data --
print(f"\n=== Saving results ===")

import csv

# 1. Summary stats CSV
with open(os.path.join(OUT_DIR, "grad_summary.csv"), 'w', newline='') as f:
    fields = ['name', 'step', 'sublayer', 'layer', 'absmax', 'mean_abs', 'std', 'norm',
              'n_elements', 'n_nonzero', 'q001', 'q01', 'q10', 'q50', 'q90', 'q99', 'q999']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for name, stats_list in grad_summary.items():
        for s in stats_list:
            s['name'] = name
            w.writerow(s)
print(f"  Saved grad_summary.csv")

# 2. Per-layer histograms as npz
hist_dict = {'bin_edges': HIST_BIN_EDGES}
for name, counts in grad_histograms.items():
    sub = get_sublayer(name)
    layer = get_layer_idx(name)
    hist_dict[f"{sub}_L{layer}"] = counts.cpu().numpy()
    hist_dict[f"{sub}_L{layer}_nsteps"] = np.array([grad_step_counts[name]])
np.savez_compressed(os.path.join(OUT_DIR, "grad_histograms.npz"), **hist_dict)
print(f"  Saved grad_histograms.npz ({len(grad_histograms)} layers)")

# Convert GPU histograms to numpy for plotting
for name in grad_histograms:
    if isinstance(grad_histograms[name], torch.Tensor):
        grad_histograms[name] = grad_histograms[name].cpu().numpy()

# 3. Raw samples (first few steps)
raw_dict = {}
for name, samples in grad_raw_samples.items():
    sub = get_sublayer(name)
    layer = get_layer_idx(name)
    for i, s in enumerate(samples):
        raw_dict[f"{sub}_L{layer}_step{i}"] = s
np.savez_compressed(os.path.join(OUT_DIR, "grad_raw_samples.npz"), **raw_dict)
print(f"  Saved grad_raw_samples.npz ({len(raw_dict)} arrays)")

# -- Plot --
print("\n=== Generating plot ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sublayer_types = ['Q', 'K', 'V', 'O', 'FFN1', 'FFN2']
colors = {'Q': '#2196F3', 'K': '#4CAF50', 'V': '#FF9800', 'O': '#F44336',
          'FFN1': '#9C27B0', 'FFN2': '#795548'}

bin_centers = (HIST_BIN_EDGES[:-1] + HIST_BIN_EDGES[1:]) / 2
bin_width = HIST_BIN_EDGES[1] - HIST_BIN_EDGES[0]

# Aggregate histograms by sublayer type (pool all 12 layers)
pooled_hist = {}
pooled_total = {}
for name, counts in grad_histograms.items():
    sub = get_sublayer(name)
    if sub not in pooled_hist:
        pooled_hist[sub] = np.zeros_like(counts)
        pooled_total[sub] = 0
    pooled_hist[sub] += counts
    pooled_total[sub] += counts.sum()

# -- Plot: log|g| PDF of all sublayer types (from accumulated histograms; Fig. 5c) --
fig, ax = plt.subplots(figsize=(14, 8))
for sub in sublayer_types:
    if sub not in pooled_hist: continue
    counts = pooled_hist[sub][1:-1]  # exclude under/overflow
    total = pooled_total[sub]
    density = counts / (total * bin_width) if total > 0 else counts
    underflow = pooled_hist[sub][0]
    overflow = pooled_hist[sub][-1]
    ax.plot(bin_centers, density, color=colors[sub], linewidth=1.5,
            label=f'{sub} (N={total:,}, <$10^{{-7}}$:{underflow:,}, >$10^{{-2}}$:{overflow:,})')

ax.set_xlabel(r'$\log_{10}$(|gradient element|)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Gradient Element Distribution - All Sublayer Types (1 epoch, {step} steps)\n'
             'IdealDevice, all 72 layers analog, IO perfect, BERT-base SQuAD',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_pdf_log_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_pdf_log_all.png")

print(f"\nAll results saved to {OUT_DIR}/")
print("Done.")
