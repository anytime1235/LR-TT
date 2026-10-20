# Fixed-configuration BERT-base SQuAD fine-tuning runs

`fine_bert_squad_lrtt.py` fine-tunes `bert-base-uncased` on SQuAD v1.1 with LR-TT analog
Q/K/V/O projections (the other encoder linear layers are frozen analog tiles, the QA head is
digital) and optionally records tile diagnostics (norms, effective ranks, cosine
similarities, per-cell trajectories). `configs/` holds one JSON file per run.

## Configuration

The script is configured by the module constants at the top of the file. Their defaults are
the QKVO operating point (rank 32) of `README_optuna.md`: rank 32, transfer every 8 steps,
rank-wise stochastic transfer (`onehot`) with a constant transfer learning rate, 10-bit
`constantstepideal` A/B and C devices, 8-bit DAC/ADC with output noise 0.04, AnalogAdam,
batch 48, 5 epochs, 365 warm-up steps, seed 42. Every run trains all five epochs (the
early-stopping patience is 5 epochs) and keeps the weights of its best epoch.

- `--config FILE.json` overrides constants with a JSON object `{"CONSTANT": value}`;
  constants that are not listed keep their defaults. Without `--config` the script runs the
  operating point.
- `--print-config` prints the effective constants as JSON and exits before any data or
  model is loaded.
- `--output-dir DIR` sets the output directory (default `results/BERT_SQUAD_LRTT_FINE` next
  to the script).

`GRAD_ACCUM_STEPS` splits the batch of 48 into micro-batches to reduce GPU memory;
`aihwkit.optim.lrtt_grad_accum_patch` then updates the A/B tiles once per micro-batch and
projects every micro-batch with the A/B weights from the start of the optimizer step. Every
config sets `GRAD_ACCUM_STEPS = 1`.

Requirements: the package (see the top-level README), the packages of
`requirements-paper.txt` and `wandb`. The checkpoint and the dataset are downloaded from the
Hugging Face hub on first use.

## Running

```bash
cd experiments/bert_squad
python fine_bert_squad_lrtt.py --config configs/rank_sweep_onehot/fine_r32.json --print-config
python fine_bert_squad_lrtt.py --config configs/rank_sweep_onehot/fine_r32.json --output-dir results/fine_r32
GPU=0 ./run_fine_configs.sh rank_sweep_set        # one group, one run after the other
GPU=0 ./run_fine_configs.sh figS9/fullC_r32.json  # single configs (path relative to configs/)
GPU=0 ./run_fine_configs.sh                       # every config
```

`run_fine_configs.sh` runs the configs in the foreground on the GPU given by `GPU` (default
0) and stops at the first run that exits with an error. One run takes about 2 to 3 h on one
NVIDIA B200 GPU with the diagnostics off and up to 8.5 h with them on. GPU training is not
bitwise reproducible.

Outputs, in `--output-dir` (`run_fine_configs.sh`: `$OUT_DIR/<group>/`, `OUT_DIR` default
`results/BERT_SQUAD_LRTT_FINE`):

- `WEIGHT_NAME`: the state dict of the best-F1 epoch;
- with `ENABLE_DIAGNOSTIC`: the diagnostic log `squad_diagnostic_log_<stamp>.json` (per-step
  tile records and the epoch history) and the plots `squad_*_<stamp>*.png`; `<stamp>` is
  `DIAG_STAMP` or, by default, `te<tau>_r<rank>_<transfer>`;
- with `run_fine_configs.sh`: `runlog_<name>.txt`, the console output with the F1 and EM of
  every epoch.

wandb runs in offline mode and writes `./wandb` in the working directory.

## Parameter sets and device sweeps

The configs use these parameter sets of the Optuna studies (`README_optuna.md`): the QKVO
operating point (rank 32; the defaults of the script), the best trial at each rank (r = 1 to
64), the AF = 1 anchor (best trial of study A, asymmetric A/B device; |w|max 1.0) and the
noise = 1 anchor (best trial of study B, full 6T1C device model; |w|max 2.32).

The asymmetry and noise sweeps put the A/B tiles on the `linearstepideal` device, whose
asymmetry and variation terms are set by `AB_LINEARSTEP_PARAMS`. Relative to the 6T1C
reference values (gamma_up -0.1678, gamma_down 0.141; dw_min_dtod 0.1, dw_min_std 0.3,
up_down_dtod 0.01, w_max_dtod 0.05, w_min_dtod 0.05, gamma_up_dtod 0.05, gamma_down_dtod
0.05):

- the asymmetry-factor scale r_AF multiplies the two gamma terms;
- the noise/variation scale r_noise multiplies the seven variation terms and sets
  `AB_RESET_STD` = 0.01 x r_noise (1e-30 at r_noise = 0).

The auxiliary weight bound is |w|max = 2^AB_MULTILEVEL x AB_DW_MIN / 2. The weight-bound
sweeps scale `AB_DW_MIN` with |w|max, so the 10-bit level count stays the same. In the file
names, `g`, `gamma_r` and `Agam_r` give r_AF, `noi_r` and `noiseB_r` give r_noise, and `p` is
the decimal point; `w{k}x` multiplies the |w|max of the parameter set by k (0.063 for the
operating point, 1.0 for the AF = 1 anchor), `S9a_*_wabs2` uses |w|max 2.0 and
`S9b_*_wabs{k}a` uses |w|max k x 0.063.

## Config groups

| Configs | Parameter set; what varies | Paper panels |
|---|---|---|
| `rank_sweep_onehot/fine_r32` | operating point; diagnostics of the first and the last LR-TT tile | Fig. 5i, Supp. Fig. S10 |
| `rank_sweep_set/*` | best trial at each rank, r = 1 to 64; ideal transfer (`TRANSFER_METHOD = "set"`: C receives the exact low-rank update) | Fig. 6b (ideal transfer) |
| `figS9/diag12_r32` | operating point; 12 monitored tiles with per-cell recording | Fig. 5f, Supp. Fig. S7 |
| `figS9/fullC_r*` | best trial at each rank, r = 1 to 64; effective rank of the assembled C | Supp. Fig. S11; Fig. 6d (r = 1, 4, 16, 64) |
| `figS9/g*_w*x` | operating point; r_AF = 0 to 5, \|w\|max = 0.063 to 1.01, no noise | Supp. Fig. S9a; Supp. Fig. S8a (opt. at r_AF = 0; \|w\|max 0.063) |
| `s9ext/S9a_*` | operating point; r_AF = 0 to 5, \|w\|max = 2.0, no noise | Supp. Fig. S9a |
| `s9ext/S9b_*`, `t86/S9b_*` | AF = 1 anchor; r_AF = 0 to 5, \|w\|max = 0.063 to 0.252 (`s9ext`), 0.5 and 2.0 (`t86`), no noise | Supp. Fig. S9b |
| `t86/Agam_*` | AF = 1 anchor; r_AF = 0 to 10, no noise | Supp. Fig. S8a (opt. at r_AF = 1); Supp. Fig. S9b (\|w\|max 1.0 column) |
| `t86/Anoi_*` | AF = 1 anchor at r_AF = 1; r_noise = 0.5 to 10 | Supp. Fig. S8b (opt. at r_noise = 0; its r_noise = 0 point is `t86/Agam_r1`) |
| `fig6c/gamma_r*` | operating point; r_AF = 3 and 10, no noise | Supp. Fig. S8a (opt. at r_AF = 0) |
| `fig6c2/Bnoi_*` | noise = 1 anchor at r_AF = 1; r_noise = 0 to 10 | Supp. Fig. S8b (opt. at r_noise = 1) |
| `figS9/noiseB_*` | operating point at r_AF = 1; r_noise = 0.1 to 10 | Supp. Section S8, practical heuristic (4) |

Some points of these panels are Optuna trials (`README_optuna.md`) rather than configs: the
rank-wise transfer series of Fig. 6b (best trial at each rank) and the r_AF = 0, |w|max 0.063
point of Supp. Figs. S8a and S9a (QKVO operating point). The LR-TT bars of Fig. 6c are, in the
left panel, the QKVO operating point (opt@0, eval@0), `figS9/g1p0_w1x` (opt@0, eval@1) and
`t86/Agam_r1` (opt@1, eval@1), and in the right panel, `t86/Agam_r1` (opt@0, eval@0),
`t86/Anoi_r1` (opt@0, eval@1) and the noise = 1 anchor (opt@1, eval@1).

## Diagnostics

`rank_sweep_onehot/fine_r32` records the default tiles (`DIAG_TILES = "first_last"`), the
layer 0 query (Fig. 5i) and the layer 11 attention-output (Supp. Fig. S10) projections.
`figS9/diag12_r32` records the `array.0.0` sub-tile of Q/K/V/O in layers 0, 6 and 11 with
per-cell recording (`g6a_cells`) and without the SVD groups. With the analog I/O model
(`IS_PERFECT = False`) a 768x768 layer is mapped onto 2x2 sub-tiles of 384x384, and the
default diagnostics see one sub-tile; `figS9/fullC_r*` set `DIAG_FULL_C`, which assembles the
four sub-tiles and records `erank_C_full` and `erank_C_full_delta` at every transfer step, and
`ERANK_RATE_LIMIT_STEPS = 1`, which limits the per-sub-tile SVD diagnostics to transfer steps.
In the code and in the diagnostic logs, tile `A` is the M x r factor and tile `B` the r x N
factor; in the notation of the paper these are B and A.
