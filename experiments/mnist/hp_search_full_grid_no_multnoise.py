#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Per-cell TPE search over (lr, tlr) on the rank x TE grid, mult_noise=False.

6 ranks (1, 4, 8, 16, 32, 64) x 6 TEs (1, 10, 50, 100, 500, 1000) = 36 cells,
30 trials each. The cells are searched one after the other; the result of
every cell is saved to results_partial.json, and a restarted run skips the
cells found there.

Usage:
  python hp_search_full_grid_no_multnoise.py --mode hybrid
  python hp_search_full_grid_no_multnoise.py --mode decay
  # one cell, full TPE search
  python hp_search_full_grid_no_multnoise.py --mode decay --cell 8 10
  # one cell, single trial with given hyperparameters (no search)
  python hp_search_full_grid_no_multnoise.py --mode decay --cell 8 10 --lr 0.3 --tlr 0.00125
"""

import os; os.environ["LRTT_SILENT"] = "1"
import argparse, math, torch, torch.nn as nn, json, optuna
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, MappingParameter
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from lrtt_training_utils import sync_fast_lr, reinit_b_gaussian

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LIFETIME = 46505
N_TRIALS = 30
SEED = 42

RANKS = [1, 4, 8, 16, 32, 64]
TES = [1, 10, 50, 100, 500, 1000]

# (lr, tlr) of the first (enqueued) trial of each cell, for both modes; a cell
# without an entry (TE = 1) starts from lr = 0.1, tlr = 0.001 (see main)
SEED_HPS = {
    (1,10): (0.1908, 0.04726), (1,50): (0.1551, 0.003145), (1,100): (0.1218, 0.04350),
    (1,500): (0.2560, 0.06478), (1,1000): (0.6782, 0.003058),
    (4,10): (0.3890, 0.001048), (4,50): (0.3418, 0.003911), (4,100): (0.3418, 0.003911),
    (4,500): (0.2412, 0.04950), (4,1000): (0.6852, 0.6249),
    (8,10): (0.0502, 0.01097), (8,50): (0.7011, 0.004174), (8,100): (0.1719, 0.002665),
    (8,500): (0.1502, 0.02262), (8,1000): (0.3198, 1.2764),
    (16,10): (0.1407, 0.08573), (16,50): (0.1872, 0.002161), (16,100): (0.1872, 0.002161),
    (16,500): (0.2335, 0.004402), (16,1000): (0.8820, 0.008597),
    (32,10): (0.1699, 0.001749), (32,50): (0.9934, 0.001219), (32,100): (0.9934, 0.001219),
    (32,500): (0.6024, 0.001898), (32,1000): (0.006297, 0.1306),
    (64,10): (0.2520, 0.001336), (64,50): (0.2520, 0.001336), (64,100): (0.2520, 0.001336),
    (64,500): (0.2520, 0.001336), (64,1000): (0.3187, 0.006835),
}

TAU_SEC = 46505.0
dt_batch_sec = -TAU_SEC * math.log(1 - 1.0/LIFETIME)
AB_LIFETIME = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_loader = val_loader = None


def load_data(data_dir):
    train_loader = DataLoader(
        datasets.MNIST(data_dir, download=True, train=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        datasets.MNIST(data_dir, download=True, train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def create_model(rank, te, tlr, reinit_mode):
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1, reset=0.0, reset_dtod=0.0,
        reset_std=0.0,  # exact A=0 reset
    )
    c_device = LinearStepDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )
    # Transfer every te mini-batches with the constant transfer rate tlr; core
    # weight scaling omega 0.6 (README.md, "Common training setting")
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=te, units_in_mbatch=True,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode=reinit_mode, scale_transfer_lr=False,
        mapping_c=MappingParameter(weight_scaling_omega=0.6),
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    reinit_b_gaussian(model, reinit_gain=1.0)
    return model


def run_trial(rank, te, lr, tlr, reinit_mode, verbose=False):
    torch.manual_seed(SEED)
    model = create_model(rank, te, tlr, reinit_mode)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        sync_fast_lr(model, optimizer.param_groups[0]["lr"])
        model.train()
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                target = target.to(DEVICE, non_blocking=True)
                correct += model(data).argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        acc = 100.0 * correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if verbose:
            print(f"  epoch {epoch:2d}: acc={acc:.2f}%  best={best_acc:.2f}%")
        if epoch >= 5 and best_acc < 50.0:
            break
        if patience >= EARLY_STOP_PATIENCE:
            break

    del model
    torch.cuda.empty_cache()
    return best_acc


def main():
    global train_loader, val_loader
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["hybrid", "decay"])
    parser.add_argument("--data-dir", default="./data", help="MNIST root (downloaded if missing)")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--cell", type=int, nargs=2, metavar=("RANK", "TE"),
                        help="run one (rank, transfer_every) cell instead of the full grid")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="TPE trials per cell")
    parser.add_argument("--lr", type=float, help="with --tlr: run one trial with these values, no search")
    parser.add_argument("--tlr", type=float)
    args = parser.parse_args()
    if (args.lr is None) != (args.tlr is None):
        parser.error("--lr and --tlr must be given together")

    reinit_mode = args.mode
    n_trials = args.n_trials
    cells = [tuple(args.cell)] if args.cell else [(r, t) for r in RANKS for t in TES]
    output_dir = os.path.join(args.results_dir, f"hp_search_{args.mode}_no_multnoise")
    os.makedirs(output_dir, exist_ok=True)
    train_loader, val_loader = load_data(args.data_dir)

    if args.lr is not None:
        fixed = []
        for rank, te in cells:
            print(f"rank={rank}, TE={te}, lr={args.lr:.6f}, tlr={args.tlr:.6f} ({args.mode})")
            acc = run_trial(rank, te, args.lr, args.tlr, reinit_mode, verbose=True)
            print(f"  best_acc={acc:.2f}%")
            fixed.append({'rank': rank, 'te': te, 'mode': args.mode, 'mult_noise': False,
                          'lr': args.lr, 'tlr': args.tlr, 'best_acc': round(acc, 2)})
        with open(f"{output_dir}/fixed_trials.json", 'w') as f:
            json.dump(fixed, f, indent=2)
        return

    total = len(cells)
    print("=" * 70)
    print(f"FULL GRID HP SEARCH (mult_noise=False, {args.mode})")
    print(f"Cells: {cells}")
    print(f"Total: {total} combinations x {n_trials} trials = {total*n_trials}")
    print("=" * 70)

    # Load existing partial results
    all_results = []
    partial_path = f"{output_dir}/results_partial.json"
    done_set = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        done_set = {(r['rank'], r['te']) for r in all_results}
        print(f"Loaded {len(all_results)} existing results, skipping those.")

    cell_idx = 0
    for rank, te in cells:
        cell_idx += 1
        if (rank, te) in done_set:
            print(f"[{cell_idx}/{total}] rank={rank}, TE={te} -- SKIP (already done)")
            continue

        seed_lr, seed_tlr = SEED_HPS.get((rank, te), (0.1, 0.001))
        print(f"\n[{cell_idx}/{total}] rank={rank}, TE={te} "
              f"(seed lr={seed_lr:.4f}, tlr={seed_tlr:.6f})")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        study.enqueue_trial({"lr": seed_lr, "tlr": seed_tlr})

        def objective(trial, _rank=rank, _te=te, _mode=reinit_mode):
            lr = trial.suggest_float("lr", 0.01, 1.5, log=True)
            tlr = trial.suggest_float("tlr", 1e-5, 1.0, log=True)
            return run_trial(_rank, _te, lr, tlr, _mode)

        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        print(f"  Best: {best.value:.2f}% "
              f"(lr={best.params['lr']:.4f}, tlr={best.params['tlr']:.6f})")

        result = {
            'rank': rank,
            'te': te,
            'mode': args.mode,
            'mult_noise': False,
            'best_acc': round(best.value, 2),
            'best_lr': best.params['lr'],
            'best_tlr': best.params['tlr'],
            'n_trials': n_trials,
            'all_trials': [
                {'lr': t.params['lr'], 'tlr': t.params['tlr'],
                 'acc': round(t.value, 2)}
                for t in study.trials
            ],
        }
        all_results.append(result)

        with open(partial_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    with open(f"{output_dir}/results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print heatmap
    print("\n" + "=" * 70)
    print(f"HEATMAP: Best Accuracy (rank x TE), mult_noise=False, {args.mode}")
    print("=" * 70)
    header = f"{'rank':>6s}" + "".join(f"  TE={te:<5d}" for te in TES)
    print(header)
    print("-" * len(header))
    for rank in RANKS:
        row = f"{rank:>6d}"
        for te in TES:
            match = [r for r in all_results if r['rank'] == rank and r['te'] == te]
            if match:
                row += f"  {match[0]['best_acc']:>6.2f}"
            else:
                row += "     N/A"
        print(row)

    print(f"\nResults: {output_dir}/results_final.json")


if __name__ == "__main__":
    main()
