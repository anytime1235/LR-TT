# Tiki-Taka and direct-SGD baselines for BERT-base on SQuAD v1.1

`paper_experiment.py` trains the analog baselines, Tiki-Taka with a full-size
auxiliary tile (`--method ttv1`, "TT"), single-tile direct SGD (`single_rpu`,
"Single RPU") and the ideal-device I/O-precision probe (`ideal`), as one run
(`--mode fixed`) or an Optuna TPE study (`--mode tpe`); `--help` lists every
option. Its analog configurations (`rpu_configs.py`) use upstream aihwkit
components only. Requirements: the CUDA build of this repository and
`requirements-paper.txt`. A 5-epoch run takes about two hours on one GPU, a
TPE study several days.

## Running

```bash
cd experiments/bert_squad/tikitaka
GPU=0 bash run_bitseed.sh                       # foreground, CUDA device 0
GPU=1 BITS="12 14" bash run_auxbit_sweep.sh     # a subset of the bit widths
bash run_auxbit_sweep.sh --status               # finished trials per bit width
python tools/export_sweep.py <sweep_root> <out_dir> [label]   # runs.csv, epoch_metrics.csv, runs.json

# one fixed TT run with the af0nz0 hyperparameters (TT QKVO of Figs. 5h (10 bit) and 6a)
python paper_experiment.py --mode fixed --method ttv1 --seed 42 \
    --target-layers all_train_attention --epochs 5 --batch-size 48 \
    --gamma 0.0 --with-reset-prob 0 --units-in-mbatch true --scale-transfer-lr false \
    --device-type constant_step --device-type-slow constant_step \
    --n-bits 10 --n-bits-slow 10 --io-bits 8 --out-noise 0.04 --noise-management abs_max \
    --desired-bl 31 --warmup-ratio 0.039566 --min-lr-rate 0.0 \
    --analog-lr 0.0118616 --fast-lr 1.7069 --transfer-lr 0.0172184 --transfer-every 6 \
    --w-max-fast 2.3834 --classifier-lr 0.0031826 --ln-lr 0.0031826 \
    --output-dir results/ttv1_qkvo_af0nz0
```

Launchers take `GPU` (default 0), `PYTHON` and `RESULTS_ROOT` (default
`results/` in this directory); their headers list the other variables, the
search space, the settings and the output directory. A TPE study writes
`db/<study>.db` and a `trial_XXXX/` directory per trial; a run directory holds
`config.json` (applied settings, I/O per path in `io_resolved`),
`training_log.csv` and `summary.json`. Instances of `run_auxbit_sweep.sh` or
`run_auxbit_refine20.sh` on several GPUs take different bit widths (markers
`claims/bit<B>`).

| Launcher | Runs | Paper |
|---|---|---|
| `run_qkv_individual_sensitivity.sh`, `run_sensitivity_sweep*.sh` | I/O-precision probe | Fig. 5d |
| `launchers/subpulse_diagnostics.sh` | `single_rpu` at 8, 10, 12, 14 bit with update diagnostics | Fig. 5e |
| `run_directsgd.sh` | TPE, direct SGD, 10 bit | Fig. 5g (Single RPU hyperparameters) |
| `run_allanalog_flr.sh [qkvo\|ffn]` | TPE, TT, QKVO or FFN trained | Fig. 5g (TT hyperparameters), 6a (TT FFN) |
| `run_bitseed.sh` | fixed runs, direct SGD and TT at weight bits 6-10 | Fig. 5g |
| `run_auxbit_sweep.sh` | TPE per auxiliary bit width 6, 12, 14 | Fig. 5h (12, 14 bit) |
| `run_auxbit_refine20.sh` | TPE per auxiliary bit width 6, 8, then export | Fig. 5h (6, 8 bit) |
| `run_auxbit10_anchor87.sh` | TPE, auxiliary bit width 10 | Fig. 5h (10 bit), 6a (TT QKVO) |
| `run_all_flr.sh` | TPE, TT, all 72 layers trained | Fig. 6a (TT ALL) |
| `run_6t1c_af1.sh`, `run_6t1c_full.sh` | TPE, 6T1C auxiliary tile, (r_AF, r_noise) = (1, 0), (1, 1) | anchors for Fig. 6c and S8 |
| `run_af_noise_tuned_sweep.sh <axis> <anchor>` | fixed runs along r_AF or r_noise | Fig. 6c, Supp. Fig. S8 |

## Settings

- TT and direct SGD, except where a launcher row or a section below states
  otherwise: all 72 encoder linear layers analog (untrained layers at
  learning rate 0), 8-bit DAC/ADC, output noise 0.04, noise-free
  ConstantStep devices (TT core 10 bit), batch 48 without gradient
  accumulation, 5 epochs, warmup_ratio 0.039566, min_lr_rate 0.0
  (`sweep_common.sh`).
- `ttv1`: `fast_lr` is the pulse-count learning rate of the auxiliary tile
  and the transfer uses the constant `transfer_lr`, so the learning-rate
  schedule acts on the classifier and LayerNorm only; `analog_lr` does not
  act on the tiles and is pinned to 0.011862 (learning rate 0 marks a frozen
  tile). `w_max_fast` scales the range and step of the auxiliary tile.
- `ln_lr` is tied to `classifier_lr` in `tpe` mode and defaults to
  `analog_lr` in `fixed` mode; pass `--ln-lr` when running a TPE trial in
  fixed mode.
- The resolution and output noise apply to the forward, backward and
  transfer-read paths; aihwkit applies bound management in the forward pass
  only and reads the transfer without noise management and bound management.

## Hyperparameters

Starting points (trial 0) and fixed-run hyperparameters, rounded (exact
values in the launchers); `ln_lr` = `classifier_lr`:

| Set | transfer_every | fast_lr | transfer_lr | w_max_fast | classifier_lr | Used by |
|---|---|---|---|---|---|---|
| search start | 2 | 0.1 | 0.017138 | 1.0 (clamped into the band) | 0.0027786 | `run_allanalog_flr.sh`, `run_auxbit_sweep.sh` |
| best trial of `ttv1_flr_qkvo` (Fig. 5g) | 1 | 1.4450 | 0.0085548 | 1.0 | 0.0023624 | `run_bitseed.sh`, `run_auxbit10_anchor87.sh`, `run_all_flr.sh`, `run_6t1c_*.sh` |
| aux 6-bit start | 2 | 0.059064 | 0.43827 | 0.093992 | 0.0045935 | `run_auxbit_refine20.sh` |
| aux 8-bit start | 2 | 0.4523 | 0.30762 | 0.081231 | 0.00116 | `run_auxbit_refine20.sh` |
| anchor `af0nz0` | 6 | 1.7069 | 0.017218 | 2.3834 | 0.0031826 | TT QKVO of Figs. 5h (10 bit) and 6a; sweeps |
| anchor `af1nz0` | 1 | 1.8513 | 0.037767 | 3.9437 | 0.0030134 | sweeps |
| anchor `af1nz1` | 1 | 0.31831 | 0.078556 | 1.0021 | 0.0018933 | sweeps |

The anchors are the best trials of `run_auxbit10_anchor87.sh`,
`run_6t1c_af1.sh` and `run_6t1c_full.sh`. Direct SGD starts at `analog_lr`
0.016, `classifier_lr` 0.003; `run_bitseed.sh` uses its best trial, 0.0099375
and 0.0023668. The best trials of the other TT studies are listed in the
per-figure parameter table of the Supplementary Information.

## I/O-precision probe (Fig. 5d)

`--method ideal` (FP32 weight update), `--target-layers all`, `--io-bits 8`,
AbsMax noise management, no output noise, `analog_lr` 0.0357,
`classifier_lr` = `ln_lr` 0.00076, batch 24 with `--grad-accum-steps 2`,
warmup_ratio 0.05 (the default of `paper_experiment.py`), min_lr_rate 0.05,
seed 42. `--per-layer-bits` lowers the forward and backward DAC/ADC
resolution of one sublayer group in all 12 layers: Q, K, V, O and QKVO at
4-7 bit, FFN1 at 4 and 7, FFN2 at 4-7 bit in one-epoch runs, FFN1 at 5 and 6
bit in 4-epoch runs (`run_sensitivity_sweep.sh`). Fig. 5d shows the F1 after
the first epoch relative to an 8-bit reference run, the same command without
`--per-layer-bits`. Each run writes
`io_precision_probe/sens_<group>_<bits>b[_1ep]/` and a log of the same name.

## Sub-pulse update diagnostics (Fig. 5e)

`launchers/subpulse_diagnostics.sh` runs `single_rpu` with stochastic pulses
for 128 steps per bit width (batch 48, perfect I/O, desired_bl 31, noise-free
ConstantStep device with dw_min = 2 / 2^bits, QKVO analog, `analog_lr`
0.016). It writes `update_diagnostics.csv`, `update_diagnostics_summary.json`,
`mu_distribution.json` (expected pulse count per element, mu) and
`pq_distribution.json` to `RESULTS_DIR/single_rpu_stoch_<bits>b/` (default
`results/subpulse_diagnostics`), read by `../diagnostics/plot_mu_ecdf.py`
and `plot_subpulse_figures.py`.

## Asymmetry and noise sweeps (Fig. 6c, Supp. Fig. S8)

`run_af_noise_tuned_sweep.sh <axis> <anchor>` (`af af0nz0`, `af af1nz0`,
`noise af1nz0`, `noise af1nz1`) trains 5-epoch fixed runs with the anchor
hyperparameters at r_AF or r_noise in 0, 0.5, 1, 2, 3, 5, 10 (`AF_LIST`,
`NOISE_LIST`), at r_noise 0 or r_AF 1 respectively. The auxiliary tile is the
6T1C `LinearStepDevice` with step 0.001981 x `w_max_fast`, also at r_AF = 0;
r_AF multiplies gamma_up = -0.1678 and gamma_down = 0.1410, r_noise
multiplies dw_min_std 0.3, dw_min_dtod 0.1, up_down_dtod 0.01 and the other
dtod parameters 0.05. `tools/collect_af_noise.py` writes `SWEEP.csv` (Supp.
Fig. S8: `best_f1`), `PLOT1_af_axis.csv` and `PLOT2_noise_axis.csv`. The TT
bars of Fig. 6c are sweep points: `af0nz0` at r_AF 0 and 1 and `af1nz0` at
r_AF 1 (left), `af1nz0` at r_noise 0 and 1 and `af1nz1` at r_noise 1 (right).
