# Backward-gradient, I/O-precision and sub-pulse diagnostics (Fig. 5a-e)

Scripts behind the diagnostic panels of Fig. 5 (BERT-base, SQuAD v1.1; all 72
encoder linear layers on analog tiles, except in the Fig. 5e runs, which map
only Q/K/V/O).

| Script or launcher | Paper | Default output directory |
|---|---|---|
| `backward_diagnostics.py`, then `plot_qzr_odr_maps.py` | Fig. 5a,b | `./results/backward_diagnostics`, `./results/qzr_odr_maps` |
| `measure_gradient_distribution.py` | Fig. 5c | `./results/gradient_distribution` |
| `../tikitaka/run_qkv_individual_sensitivity.sh`, `run_sensitivity_sweep_1ep.sh`, `run_sensitivity_sweep_QKVO_1ep.sh`, `run_sensitivity_sweep.sh` | Fig. 5d | `../tikitaka/results/io_precision_probe` |
| `../tikitaka/launchers/subpulse_diagnostics.sh`, then `plot_mu_ecdf.py`, `plot_subpulse_figures.py` | Fig. 5e | `../tikitaka/results/subpulse_diagnostics`, `./results/mu_ecdf`, `./results/subpulse_figures` |
| `precision_allocation.py` | - | `./results/precision_allocation` |

Requirements: aihwkit from this repository (CUDA build), `torch`,
`transformers`, `datasets`, `evaluate`, `pandas`, `matplotlib`, `tqdm`; the
runs need a GPU. `measure_gradient_distribution.py` imports
`../tikitaka/paper_experiment.py`. The scripts here write to `--out-dir`
(`--output` for `plot_mu_ecdf.py`), the launchers to `RESULTS_ROOT` or
`RESULTS_DIR`; the plotting scripts read the default output directories of the
runs they plot (`--data-dir`, `--results-dir`).

```bash
cd experiments/bert_squad/diagnostics
python backward_diagnostics.py && python plot_qzr_odr_maps.py        # Fig. 5a,b
python measure_gradient_distribution.py                               # Fig. 5c
bash ../tikitaka/launchers/subpulse_diagnostics.sh                    # Fig. 5e runs
python plot_mu_ecdf.py && python plot_subpulse_figures.py             # Fig. 5e
python backward_diagnostics.py --n-step 5 --batch-size 2 --out-dir ./results/smoke   # short test run
python precision_allocation.py --figures A --run-sensitivity-allocation   # see below
```

**Fig. 5a,b.** `backward_diagnostics.py`: 200 steps of batch 8 at learning
rate 0 (SQuAD v1.1 training features, seed-42 shuffle, maximum length 384,
stride 128); pretrained `bert-base-uncased` (`qa_outputs` and the pooler
digital); noise-free `SoftBoundsDevice` (`dw_min` 0.001, weights in [-1, 1])
with column-wise weight scaling (omega 1.0); DAC 7 bit (`inp_res = 1/126`),
ADC 9 bit (`out_res = 1/510`), AbsMax noise management, deterministic
rounding, no output noise. `QZR_nonzero` (`QZR_all`) is the fraction of the
nonzero (all) backward-gradient elements of a step with
`|delta| / absmax < inp_res`, which the backward DAC rounds to zero, and `ODR`
the per-vector `absmax / max(median, 1e-8)`, both averaged over the steps
(padding removed). Outputs: `metrics_paper_A_rootcause_summary.csv` (one row
per layer and sublayer), `metrics_paper_A_rootcause_steps.csv`,
`absmax_raw_A_baseline_7b.npz`, an overview figure, and from
`plot_qzr_odr_maps.py` `qzr_odr_maps_<run_tag>.{pdf,png}` (ODR as log10).

**Fig. 5c.** `measure_gradient_distribution.py`: one training epoch of the
model of `paper_experiment.create_model` (`--method ideal`, perfect I/O) at
constant learning rates (analog 0.0357, classifier and LayerNorm 0.00076, seed
42), batch size 16 (`BATCH_SIZES = [16, 8]`: after a CUDA error the epoch
starts at the next batch size). Outputs: per-layer histograms of `log10|g|` of
the nonzero gradient elements (500 bins over [1e-7, 1e-2] plus underflow and
overflow, `grad_histograms.npz`), `grad_summary.csv`, `grad_raw_samples.npz`
and `grad_pdf_log_all.png` (density per sublayer type Q, K, V, O, FFN1, FFN2).

**Fig. 5d.** The probe launchers train with an ideal device (FP32 weight
update), all layers analog, 8-bit I/O except for the probed sublayer group (Q,
K, V, O, QKVO, FFN1 or FFN2) at 4-7 bit, no output noise, batch 24 with 2
gradient-accumulation steps, analog learning rate 0.0357 and seed 42. Each bar
is the F1 after the first epoch (`run_sensitivity_sweep.sh` trains 4 epochs),
relative to the run with every group at 8 bit.

**Fig. 5e.** `subpulse_diagnostics.sh`: `single_rpu` with stochastic pulses at
8, 10, 12 and 14 bit, 128 steps each (batch 48, seed 42, Q/K/V/O analog,
perfect I/O, `desired_bl` 31, noise-free ConstantStep device with
`dw_min = 2 / 2^bits`); each run directory holds the histograms of the expected
pulse count `mu` per tile and step (`mu_distribution.json`, bin edges 0 and
1,000 log-spaced values from 1e-6 to 1e4) and
`update_diagnostics_summary.json`. `plot_mu_ecdf.py` draws the `mu` ECDF per
bit width (`mu_distribution_measured_ecdf.png`); `plot_subpulse_figures.py`
adds the cosine similarity between realized and target update and the recovery
ratio (`subpulse_overview.png`).

**`precision_allocation.py`** runs the Fig. 5a,b metrics with an 8-bit DAC/ADC
baseline (`inp_res = 1/254`), an I/O-resolution sweep at 4, 6, 8, 10 and 12
bit, and a per-tile precision allocation for target average bit widths
(`--run-sensitivity-allocation`, `--offline-allocate`; maps in the
`--per-layer-bits` format of `paper_experiment.py`); `--help` lists the
options.
