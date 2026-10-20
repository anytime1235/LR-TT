# MNIST on-chip scratch training (Fig. 4a-c, Supplementary Fig. S6)

Analog-tile simulations of LR-TT on MNIST with a 784-256-10 MLP (Supplementary
Section S5): the first layer is an LR-TT block (auxiliary `A`/`B` tiles on the
6T1C device model, core tile `C`), followed by ReLU, a floating-point
classifier (`FloatingPointRPUConfig`) and `LogSoftmax` with NLL loss.

Notation: the code writes the LR-TT product as `A @ B` (`A` is `[256, rank]`,
`B` is `[rank, 784]`), so code `A` is the paper's `B` and vice versa; the
transfer interval tau is `TE` (mini-batches). `--mode` (`reinit_mode`) selects
the post-transfer rule of Supplementary Section S5: `decay` ("w/o reset", `A`
and `B` kept) or `hybrid` ("w/ reset", code `A` reset to zero after every
transfer). `no_multnoise` in a script name refers to the A/B device setting
`mult_noise=False`, which all scripts use.

Requirements: the `aihwkit` of this repository (CUDA build), `torch`,
`torchvision`, and `optuna` for the searches. Run the scripts from this
directory; they use `cuda:0` (select the GPU with `CUDA_VISIBLE_DEVICES`) and
take `--data-dir` (default `./data`, MNIST is downloaded if missing) and
`--results-dir` (default `./results`). `erank.py` (entropy effective rank) and
`lrtt_training_utils.py` (`sync_fast_lr`, `reinit_b_gaussian`) are helpers.

## Common training setting

- Batch size 64, 30 epochs, test accuracy after every epoch; `AnalogSGD` with
  `StepLR(step_size=10, gamma=0.5)`; `sync_fast_lr` sets the A/B update rate to
  the scheduled `lr` at the start of every epoch; constant transfer rate `tlr`
  (`transfer_lr = tlr`, `scale_transfer_lr=False`).
- `PythonLRTTDevice` with `transfer_every=TE`, `units_in_mbatch=True`,
  `lora_alpha=1.0`, `reinit_gain=1.0`, core weight scaling omega 0.6,
  `forward_inject=False` (forward through `C` only), one-hot transfer of all
  ranks. Initial state: `A = 0`, `B` Gaussian with standard deviation
  `sqrt(2 / 784)` (`reinit_b_gaussian`); the digital bias keeps its initial
  value.
- A/B tiles (6T1C `LinearStepDevice`): `dw_min 0.001981`, bounds +-1,
  `gamma_up -0.1678`, `gamma_down 0.1410` (slope magnitudes are used),
  `mult_noise False`, `dw_min_std 0.3`, `dw_min_dtod 0.1`, `up_down_dtod 0.01`,
  `w_max_dtod = w_min_dtod 0.05`, `gamma_up_dtod = gamma_down_dtod 0.05`, no
  write noise, `lifetime` 46505 (decay `1 - 1/46505` per mini-batch),
  `lifetime_dtod 0.1`, exact zero reset.
- Core tile (`LinearStepDevice`): `dw_min 0.001`, bounds +-1,
  `gamma_up = gamma_down = 0`, no step or bound variation and no noise; the
  other parameters keep the `LinearStepDevice` defaults
  (`gamma_up_dtod = gamma_down_dtod 0.05`, no retention decay).
- I/O: 7-bit DAC (`inp_res 1/126`), 9-bit ADC (`out_res 1/510`), bounds 1 and
  12, no output noise; `desired_bl 31`, stochastic compressed pulses with update
  and bit-length management.

## Experiments

**Fig. 4a (decay), Supp. Fig. S6a (hybrid), S6d (hybrid minus decay).**
`hp_search_full_grid_no_multnoise.py` searches each cell of
`rank in {1, 4, 8, 16, 32, 64}` x `TE in {1, 10, 50, 100, 500, 1000}` with
`TPESampler(seed=42)`, 30 trials (`--n-trials`), `lr` in `[0.01, 1.5]` and
`tlr` in `[1e-5, 1.0]` (log-uniform); the first trial is the cell's pair in
`SEED_HPS` (`lr 0.1`, `tlr 0.001` for `TE = 1`), evaluated as given. A trial
(seed 42) runs up to 30 epochs and stops after 5 epochs without improvement or,
from epoch 5 on, below 50 %; its value is the best epoch accuracy, the cell
value the best trial. Fig. 4a and S6a show `100 - best_acc`.

**Fig. 4b.** `sweep_ctile_bits_gamma.py` (decay, rank 8, `TE = 10`,
`lr = 0.11577`) sets the core tile to `dw_min = 2 / 2^b_w` for `b_w` 5-10
(`--bits`) and `gamma_up = gamma_down = AF^C` for `AF^C` 0, 0.5, 1, 2, 5, 10
(`--gammas`) and searches `tlr` per cell (`TPESampler(seed=42)`, 20 trials,
`[1e-5, 1.0]` log-uniform, first trial from `SEED_TLRS`) with the same early
stopping. The panel shows `100 - best_acc`.

**Fig. 4c (decay), Supp. Fig. S6b,c,e (hybrid).** `run_erank_fixed_lr.py`
trains each cell once (seed 42, 30 epochs) at `lr = 0.3` and
`tlr = 0.010 / rank` (decay) or `0.025 / rank^1.5` (hybrid) and records per
epoch the entropy effective rank of `C - C_0` (Supplementary Section S5), the
test accuracy and the transfer count. Fig. 4c and S6b show box plots over tau
and rank, S6c the hybrid runs at rank 8 per epoch and S6e hybrid minus decay;
S6b,c,e use `TE` 10 to 1000.

## Running

```bash
python run_erank_fixed_lr.py --mode decay --ranks 8 --tes 10   # quick start: one cell of Fig. 4c
python run_erank_fixed_lr.py --mode decay                       # Fig. 4c
python run_erank_fixed_lr.py --mode hybrid                      # Supp. Fig. S6b,c,e
python hp_search_full_grid_no_multnoise.py --mode decay         # Fig. 4a
python hp_search_full_grid_no_multnoise.py --mode hybrid        # Supp. Fig. S6a,d
python sweep_ctile_bits_gamma.py                                # Fig. 4b
# one cell of the search (about 30 min), and one trial with given values
python hp_search_full_grid_no_multnoise.py --mode decay --cell 8 10
python hp_search_full_grid_no_multnoise.py --mode decay --cell 8 10 --lr 0.3 --tlr 0.00125
```

The quick start (rank 8, transfer every 10 mini-batches) trains for 30 epochs
and reports the test accuracy and the effective rank of the core-tile update
per epoch (about 97 % test accuracy after 30 epochs; about two minutes on a
GPU).

## Outputs

Under `--results-dir`:

- `erank_fixed_lr_{mode}/erank_C.csv`, one row per run and epoch with the
  columns `mode, rank, te, epoch, erank_dC, val_acc, num_transfers` (`val_acc`
  is the test accuracy), and `summary.json` (settings and the final values of
  every run);
- `hp_search_{mode}_no_multnoise/results_partial.json` (written after every
  cell; a restarted search skips the cells found there), `results_final.json`
  (per cell `best_acc`, `best_lr`, `best_tlr`, `all_trials`) and
  `fixed_trials.json` (runs with `--lr`/`--tlr`);
- `ctile_bits_gamma_sweep/results_partial.json`, `results_final.json` (per cell
  `bits`, `dw_min`, `gamma`, `best_acc`, `best_tlr`, `all_trials`).
