#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""C-tile bit x gamma 2D sweep for LRTT.

6 bit levels x 6 gamma levels = 36 cells, 20 Optuna trials each (tlr search).
All cells: rank=8, TE=10, lr=0.11577, lifetime=46505, omega=0.6 (decay mode).
A/B tiles: 6T1C-calibrated device. C-tile: LinearStepDevice with
dw_min = 2 / 2^bits and gamma_up = gamma_down = gamma.

Usage:
  python sweep_ctile_bits_gamma.py
  python sweep_ctile_bits_gamma.py --bits 8 --gammas 0.0 --n-trials 5   # one cell
"""

import os; os.environ["LRTT_SILENT"] = "1"
import argparse, math, torch, torch.nn as nn, json, optuna, time
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
N_TRIALS = 20
LR = 0.11577

RANK = 8
TE = 10
LIFETIME = 46505

BITS = [5, 6, 7, 8, 9, 10]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# tlr of the first (enqueued) trial for each bit level
SEED_TLRS = {
    10: 0.000234,
    9:  0.000467,
    8:  0.000935,
    7:  0.000332,
    6:  0.000516,
    5:  0.000225,
}

TAU_SEC = 46505.0
dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / LIFETIME)
AB_LIFETIME = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = val_loader = None


def load_data(data_dir):
    train_loader = DataLoader(
        datasets.MNIST(data_dir, download=True, train=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        datasets.MNIST(data_dir, download=True, train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def create_model(dw_min, gamma, tlr):
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
        dw_min=dw_min, w_max=1.0, w_min=-1.0,
        gamma_up=gamma, gamma_down=gamma,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )
    # Transfer every TE mini-batches with the constant transfer rate tlr; core
    # weight scaling omega 0.6 (README.md, "Common training setting")
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=TE, units_in_mbatch=True,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="decay", scale_transfer_lr=False,
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


def run_trial(dw_min, gamma, tlr):
    model = create_model(dw_min, gamma, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
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
    parser.add_argument("--data-dir", default="./data", help="MNIST root (downloaded if missing)")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--bits", type=int, nargs="+", default=BITS, help="C-tile bit levels")
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS, help="C-tile gamma values")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="TPE trials per cell")
    args = parser.parse_args()

    bits_list, gammas, n_trials = args.bits, args.gammas, args.n_trials
    output_dir = os.path.join(args.results_dir, "ctile_bits_gamma_sweep")
    os.makedirs(output_dir, exist_ok=True)
    train_loader, val_loader = load_data(args.data_dir)

    total_cells = len(bits_list) * len(gammas)
    total_trials = total_cells * n_trials

    print("=" * 70)
    print("C-TILE BIT x GAMMA 2D SWEEP (LRTT, tlr Optuna search)")
    print("=" * 70)
    print(f"Bits: {bits_list}")
    print(f"Gammas: {gammas}")
    print(f"Cells: {total_cells}, Trials/cell: {n_trials}, Total trials: {total_trials}")
    print(f"lr={LR} (fixed), rank={RANK}, TE={TE}, lifetime={LIFETIME}, omega=0.6")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}, max epochs={EPOCHS}")
    print()

    all_results = []
    cell_idx = 0
    t_start = time.time()

    for bits in bits_list:
        dw_min = 2.0 / (2**bits)
        seed_tlr = SEED_TLRS[bits]

        for gamma in gammas:
            cell_idx += 1
            elapsed = time.time() - t_start
            if cell_idx > 1:
                avg_per_cell = elapsed / (cell_idx - 1)
                remaining = avg_per_cell * (total_cells - cell_idx + 1)
                eta_str = f", ETA ~{remaining/3600:.1f}h"
            else:
                eta_str = ""

            print(f"[{cell_idx}/{total_cells}] bits={bits}, gamma={gamma} "
                  f"(dw_min={dw_min:.6f}){eta_str}")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42))
            study.enqueue_trial({"tlr": seed_tlr})

            def objective(trial, _dw=dw_min, _g=gamma):
                tlr = trial.suggest_float("tlr", 1e-5, 1.0, log=True)
                return run_trial(_dw, _g, tlr)

            study.optimize(objective, n_trials=n_trials)

            best = study.best_trial
            print(f"  Best: {best.value:.2f}% (tlr={best.params['tlr']:.6e})")

            result = {
                'bits': bits,
                'dw_min': dw_min,
                'gamma': gamma,
                'best_acc': round(best.value, 2),
                'best_tlr': best.params['tlr'],
                'all_trials': [{'tlr': t.params['tlr'], 'acc': round(t.value, 2)}
                               for t in study.trials],
            }
            all_results.append(result)

            # Save partial results after each cell
            with open(f"{output_dir}/results_partial.json", 'w') as f:
                json.dump(all_results, f, indent=2)

    # Save final
    with open(f"{output_dir}/results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/3600:.1f}h")

    # Print heatmap summary
    print("\n" + "=" * 70)
    print("HEATMAP: Best Accuracy (bits x gamma)")
    print("=" * 70)
    header = f"{'bits':>6s}" + "".join(f"  g={g:<5.1f}" for g in gammas)
    print(header)
    print("-" * len(header))
    for bits in bits_list:
        row = f"{bits:>6d}"
        for gamma in gammas:
            match = [r for r in all_results if r['bits'] == bits and r['gamma'] == gamma]
            if match:
                row += f"  {match[0]['best_acc']:>6.2f}"
            else:
                row += "     N/A"
        print(row)

    print(f"\nResults: {output_dir}/results_final.json")


if __name__ == "__main__":
    main()
