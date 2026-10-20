# Regression experiments (Fig. 4d,e; Supplementary Section S4, Fig. S5)

`regression_lrtt_scratch_decay_sw.py` trains LR-TT from scratch on a simulated
9x9 linear regression `y = T x + noise` (`T` uniform in `[-0.8, 0.8]`, label
noise std 0.02, ternary inputs {0, 0.5, 1}, 200 training and 200 held-out
samples) and records the trajectory of the core tile `C`, which alone forms
the forward pass. The code writes the update as `A @ B` (`A` is
`[out, rank]`), so code `A` is the paper's `B` and vice versa.

Requirements: the `aihwkit` of this repository (a CPU build is sufficient),
`torch`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`,
`wandb`. Run from this directory; a visible GPU is used, otherwise the CPU.

## Settings

Defaults of `ScratchExperimentConfig`:

| Setting | Default |
| --- | --- |
| rank, transfer | `lrtt_rank 4`, transfer every 116 updates, one-hot rank-wise transfer of all ranks, bit length 10 (`c_desired_bl`), rate `lora_alpha 12.89` x `lrtt_lr` |
| A/B tiles | 6T1C `LinearStepDevice`: `dw_min 0.001981`, bounds +-1, `gamma_up -0.1678`, `gamma_down 0.1410`, `mult_noise True`, `dw_min_std 0.3`, device-to-device variation `dw_min_dtod 0.1`, `up_down_dtod 0.01`, bounds and slopes 0.05; retention lifetime 24.89 updates (4 % decay per update); `A` zero and `B` Kaiming at the start, both kept after a transfer (`reinit_mode='decay'`); stochastic compressed pulses, `desired_bl 10`, update management |
| core tile `C` | `SoftBoundsReferenceDevice`: `dw_min 0.000804`, bounds +-1, `up_down_dtod 0.01`, no other variation or noise; initialized to 0; column-wise weight scaling with learnable output scales (default `mapping_c`), trained by SGD at `lrtt_lr` |
| training | ideal I/O (`is_perfect=True`); `AnalogSGD`, `lrtt_lr 0.2852`, batch size 1, gradient-norm clipping 2.0; at most 2000 epochs, early stopping on the held-out MSE in epochs with a transfer, patience 7 such epochs |

## Fig. 4d,e

`run_multi_param_experiments()` trains one model per configuration (a `label`
plus `ScratchExperimentConfig` overrides) on the same data and fits one PCA
basis to all trajectories of the call, so both panels are run together:

```python
import os
from regression_lrtt_scratch_decay_sw import run_multi_param_experiments

os.makedirs('results/lrtt_scratch_decay', exist_ok=True)
fig4d = [{'label': f'r={r}, tau=1', 'lrtt_rank': r, 'lrtt_transfer_every': 1}
         for r in (1, 2, 4)]                                  # tau = 1, r = 1, 2, 4
fig4e = [{'label': f'r=1, tau={t}', 'lrtt_rank': 1, 'lrtt_transfer_every': t}
         for t in (10, 100)]                                  # r = 1, tau = 1 (above), 10, 100
run_multi_param_experiments(fig4d + fig4e, complexity_level='medium', seed=42)
```

## Supp. Fig. S5

With `save_figures = True` in `ScratchExperimentConfig`, the `__main__` block
sweeps the slopes of the core tile (S5a, `c_gamma_*`) and of the A/B tiles
(S5b, `ab_gamma_*`) over the 6T1C fit and `gamma_up = -g`, `gamma_down = g`,
g = 1, 2, 5, the other tile at the 6T1C fit. Both sweeps use rank 4, a transfer
every 100 updates, `lora_alpha 0.43`, `lrtt_lr 0.00879`, a `LinearStepDevice`
core (`dw_min 0.000804`, `mult_noise True`, `dw_min_std 0.3`,
`dw_min_dtod 0.1`, `up_down_dtod 0.01`, slope variation 0.05), non-ideal I/O
(7-bit inputs, 9-bit outputs, output noise 0.06) and no retention decay.

```bash
# after setting save_figures = True in ScratchExperimentConfig
mkdir -p results/lrtt_scratch_decay
python regression_lrtt_scratch_decay_sw.py --no-wandb   # [--results-dir DIR]
```

## Outputs

Each call of `run_multi_param_experiments()` writes to the results directory
(default `results/lrtt_scratch_decay`, which has to exist):
`raw_data_<complexity>_seed<seed>_<timestamp>.pkl` (per configuration the
step-wise `C`, transfer flags, losses and norms, the final MSE and R^2 and the
configuration), `combined_{loss,norms,r2}_*.png` and
`loss_surface_pca_<timestamp>.svg` (the `C` trajectories on the first two
principal components over the training-set MSE, transfer events marked).
