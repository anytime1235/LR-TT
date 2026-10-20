# Optuna hyperparameter searches for BERT-base on SQuAD v1.1

| Script | Purpose |
|---|---|
| `optuna_bert_squad_lrtt.py` | LR-TT studies (QKVO, FFN, all-layer, rank sweep, A/B bit-width sweep, device studies A and B) |
| `optuna_bert_squad_digital.py` | digital full fine-tuning baseline (learning rate only) |
| `optuna_bert_squad_digital_lora.py` | digital LoRA baseline (rank 32, alpha 64, learning rate only) |

Each script stores the trials of a study in an Optuna journal file, which
`optuna.load_study(study_name=<study>, storage=JournalStorage(JournalFileBackend(<file>)))`
loads. Requirements beyond the package: `optuna`, `optuna-integration` with `botorch`,
`transformers`, `datasets`, `evaluate`, `matplotlib`. The checkpoint and the dataset are
downloaded from the Hugging Face hub on first use.

## LR-TT studies

```bash
cd experiments/bert_squad
python optuna_bert_squad_lrtt.py --lora-target qkvo --print-config   # study name, settings, search space
python optuna_bert_squad_lrtt.py --lora-target qkvo --n-trials 50    # run trials of the QKVO study
python optuna_bert_squad_lrtt.py --lora-target qkvo --visualize      # summary and plots of a study
```

The defaults are the configuration shared by all BERT-base studies of the paper (`--help` lists
the flags): AnalogAdam with betas (0.9, 0.999) and no weight decay, batch 48, 5 epochs, 365
warm-up steps then linear decay to 0, maximum sequence length 384, rank 32, rank-wise
stochastic transfer (`onehot`) with a constant transfer learning rate, no auxiliary forward
path (`W_vis = C`), A and B retained after transfer (`--reinit-mode decay`),
`constantstepideal` A/B and C devices with 10-bit multi-level A/B updates (`--ab-bits`) and
`c_dw_min` 0.001953 (10 bit), C out scaling not trained, 8-bit DAC/ADC with output noise 0.04
on the forward and backward passes, non-adapted encoder layers as frozen analog tiles, and the
QA head and LayerNorm trained digitally.

The study name is derived from the configuration. The journal is
`<results-dir>/optuna_<study>.log` (`--results-dir`, default
`./results/optuna_bert_squad_lrtt`); launching the same configuration again, also from several
processes, extends the study. At the end of a run the script also writes
`best_params_<study>.json`, `all_trials.json` and `visualization.png` to the results directory
(`--visualize` writes only the plot). The sampler is a BoTorch Gaussian-process sampler (10
random start-up trials, qLogEI acquisition). Each trial trains with early stopping (patience 2
epochs on F1, 1 epoch on a training loss above 1.5) and returns the best validation F1. A trial
that runs out of GPU memory is enqueued again with the batch of 48 split into more
micro-batches (`--grad-accum-steps` sets the starting factor, default 1); the factor is
recorded with the trial as `grad_accum_steps`. With micro-batches,
`aihwkit.optim.lrtt_grad_accum_patch` updates the A/B tiles once per micro-batch and projects
every micro-batch with the A/B weights from the start of the optimizer step.

| Study | Command | Paper |
|---|---|---|
| QKVO | `--lora-target qkvo` | Figs. 5g, 5h, 6a, 6b, 6c (left); Supp. Figs. S8a, S9a |
| FFN | `--lora-target ffn` | Fig. 6a |
| ALL | `--lora-target all` | Fig. 6a |
| A: asymmetric A/B device (6T1C asymmetry factor, no noise) | `--ab-device constantstep6t1cgamma --search-space wide` | AF = 1 anchor (`README_fine.md`) |
| B: full 6T1C device model (asymmetry, noise, variation, `reset_std` 0.01) | `--ab-device 6t1c` | Fig. 6c (right); noise = 1 anchor (`README_fine.md`) |

The QKVO study also holds the rank sweep (`--rank-sweep`, or `--rank N` for one rank), the A/B
bit-width sweep (`--ab-bits N`) and the C bit-width points (`c_dw_min`); these settings do not
change the study name. With `--ab-device 6t1c` the A/B `reset_std` defaults to 0.01, the reset
noise of the 6T1C device model (1e-30 for the other devices).

### Parameter sets of the paper

`lr` = peak learning rate, `tlr` = transfer learning rate, `te` = transfer interval, `flr` =
fast (A/B) learning-rate scale, `dw` = `ab_dw_min`, `ga` = recorded `grad_accum_steps`; the
sets of the table use `ab_multilevel` = 10 and `c_dw_min` = 0.001953.

| Parameter set | Study | r | lr | tlr | te | flr | dw | ga |
|---|---|---|---|---|---|---|---|---|
| QKVO operating point (rank 32) | QKVO | 32 | 1.108e-3 | 7.873e-4 | 8 | 0.676 | 1.232e-4 | 1 |
| FFN best trial | FFN | 32 | 1.985e-3 | 2.766e-4 | 5 | 0.360 | 2.708e-4 | 2 |
| ALL best trial | ALL | 32 | 1.300e-3 | 6.487e-4 | 8 | 0.394 | 1.319e-4 | 2 |
| AF = 1 anchor | A | 32 | 3.812e-3 | 1.808e-4 | 1 | 0.300 | 1.953e-3 | 1 |
| noise = 1 anchor (`ab_reset_std` 0.01) | B | 32 | 1.636e-3 | 1.825e-4 | 24 | 0.444 | 4.533e-3 | 1 |

- best trial at each rank r (Fig. 6b): the hyperparameters of the configs
  `figS9/fullC_r<r>.json` and `rank_sweep_set/fine_r<r>_set.json` (`README_fine.md`); `ga` = 2
  at r = 1 and 1 at the other ranks; r = 32 is the operating point;
- A/B bit-width points (Fig. 5h): the best rank-32 trials with `ab_multilevel` = 6, 8, 12 and
  14, hyperparameters in Supplementary Table S5; `ga` = 2 for the A/B 8-bit point, 1 otherwise;
- C bit-width points (Fig. 5g): the operating point with `c_dw_min` = 2/2^k, k = 6 to 9; `ga` = 1.

The 10-bit point of both bit-width sweeps is the operating point. A parameter set is added to a
loaded study with `enqueue_trial`; the next trial started by a process running the study takes
it. For example, the C 6-bit point:

```python
study.enqueue_trial({"learning_rate": 1.108e-3, "transfer_lr": 7.873e-4, "transfer_every": 8,
                     "rank_exp": 5, "fast_lr": 0.676, "ab_dw_min": 1.232e-4, "c_dw_min": 0.03125})
```

### Search spaces

All ranges are log-uniform except `rank_exp`, a uniform integer. `--search-space` selects a set
of `SEARCH_SPACES`. `--rank-sweep` searches `rank_exp` in 0..6 (with the `rank` set unless
`--search-space` is given); otherwise `rank_exp` = log2 of `--rank`.

| Set | `learning_rate` | `transfer_lr` | `transfer_every` | `fast_lr` | `ab_dw_min` | `rank_exp` | Used by |
|---|---|---|---|---|---|---|---|
| `paper` (default) | 8e-5 .. 2e-2 | 8e-6 .. 5e-2 | 1 .. 500 | 9e-3 .. 4 | 7e-6 .. 5e-3 | 5 | FFN, ALL, study B |
| `qkvo-r32` | 8e-5 .. 2e-2 | 1e-4 .. 3e-2 | 1 .. 80 | 7e-2 .. 2 | 3e-5 .. 6e-4 | 5 | QKVO at rank 32 (operating point) |
| `wide` | 1e-4 .. 2e-2 | 6e-6 .. 2e-2 | 1 .. 700 | 9e-3 .. 6 | 5e-6 .. 4e-2 | 5 | study A |
| `rank` | 7e-4 .. 9e-3 | 1e-4 .. 3e-2 | 1 .. 80 | 3e-2 .. 0.9 | 2e-5 .. 7e-4 | 0 .. 6 | QKVO rank sweep |

The journal also records the single-valued parameters `tau_sec` 0, `ab_desired_bl` 31,
`c_desired_bl` 31 and `ab_weight_scaling_omega` 0.

## Digital baselines

`optuna_bert_squad_digital.py` (full fine-tuning) and `optuna_bert_squad_digital_lora.py`
(LoRA, rank 32 and alpha 64 by default: `--rank`, `--alpha`) take `--lora-target {qkvo,ffn,all}`
and `--n-trials` (default 10). Both search only the learning rate (log-uniform in 1e-5 .. 1e-3
for full fine-tuning, 1e-5 .. 1e-2 for LoRA) with a TPE sampler (seed 42), with the batch size,
epochs, warmup and linear decay of the LR-TT studies, no weight decay and Adam with betas
(0, 0.999) (`torch.optim.AdamW` with weight decay 0 for full fine-tuning). The linear layers of
the target sublayers (full fine-tuning) or the LoRA factors on them, the QA head and the
LayerNorm parameters are trained. Full fine-tuning runs in FP32 with gradient clipping at norm 1
and stops a trial after two epochs without F1 improvement or when the training loss of the
second or a later epoch is above 1.5; LoRA trains with fp16 autocast. Optuna's median pruner
stops trials early. The journals go to `./results/optuna_bert_squad_digital/<study>.log` and
`./results/optuna_bert_squad_digital_lora/<study>.log` below the current directory.

| `--lora-target` | Full fine-tuning learning rate (Fig. 6a) | LoRA learning rate (Fig. 6a) |
|---|---|---|
| `qkvo` | 1.509e-4 | 3.047e-4 |
| `ffn` | 8.501e-5 | 4.196e-4 |
| `all` | 5.094e-5 | 4.196e-4 |
