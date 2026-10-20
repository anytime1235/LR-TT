#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Effective rank of C - C_0 for every rank x TE cell at a common lr and a rank-scaled tlr.

Each cell is trained once (seed 42, 30 epochs); erank(C - C_0), the test
accuracy and the transfer count are recorded after every epoch.

Hyperparameters:
  - lr  = 0.3 for all cells
  - tlr = 0.010 / rank (decay) or 0.025 / rank^1.5 (hybrid), see TLR_RULES

Usage:
  python run_erank_fixed_lr.py --mode decay
  python run_erank_fixed_lr.py --mode hybrid
  python run_erank_fixed_lr.py --mode decay --ranks 8 --tes 10   # one cell

Output: {results-dir}/erank_fixed_lr_{mode}/ (erank_C.csv, summary.json)
"""
import os; os.environ["LRTT_SILENT"] = "1"
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, csv, json, math, time, torch, torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, MappingParameter
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from erank import compute_erank_via_gram as compute_erank
from lrtt_training_utils import sync_fast_lr, reinit_b_gaussian

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42
LR = 0.3  # lr of every cell

# Transfer learning rate of each mode, tlr = alpha / rank^beta
#   decay:  tlr = 0.010 / rank
#   hybrid: tlr = 0.025 / rank^1.5
TLR_RULES = {
    'decay':  {'alpha': 0.010, 'beta': 1.0},
    'hybrid': {'alpha': 0.025, 'beta': 1.5},
}

LIFETIME = 46505
TAU_SEC = 46505.0
dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / LIFETIME)
AB_LIFETIME = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))

RANKS = [1, 4, 8, 16, 32, 64]
TES = [1, 10, 50, 100, 500, 1000]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


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
        reset_std=0.0)  # exact A=0 reset
    c_device = LinearStepDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0, up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0)
    # Transfer every te mini-batches with the constant transfer rate tlr; core
    # weight scaling omega 0.6 (README.md, "Common training setting")
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=te, units_in_mbatch=True,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode=reinit_mode, scale_transfer_lr=False,
        mapping_c=MappingParameter(weight_scaling_omega=0.6),
        unit_cell_devices=[ab_device, ab_device, c_device])
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


def get_lrtt_tiles(model):
    tiles = {}
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            tile = module.analog_module
            if hasattr(tile, 'controller') and hasattr(tile, 'get_lrtt_component_weights'):
                tiles[name] = tile
    return tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['decay', 'hybrid'], default='decay',
                        help='reinit mode: decay or hybrid (reset)')
    parser.add_argument('--data-dir', default='./data', help='MNIST root (downloaded if missing)')
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--ranks', type=int, nargs='+', default=RANKS)
    parser.add_argument('--tes', type=int, nargs='+', default=TES)
    args = parser.parse_args()

    reinit_mode = args.mode
    ranks, tes = args.ranks, args.tes
    output_dir = Path(args.results_dir) / f"erank_fixed_lr_{reinit_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = load_data(args.data_dir)

    csv_path = output_dir / "erank_C.csv"
    csv_f = open(csv_path, 'w', newline='')
    csv_w = csv.writer(csv_f)
    csv_w.writerow(['mode', 'rank', 'te', 'epoch', 'erank_dC', 'val_acc', 'num_transfers'])

    total = len(ranks) * len(tes)
    run_idx = 0
    all_results = []

    print(f"{'='*60}")
    print(f"ERANK FIXED LR EXPERIMENT")
    print(f"  mode={reinit_mode}")
    tlr_rule = TLR_RULES[reinit_mode]
    print(f"  lr={LR} (fixed), tlr={tlr_rule['alpha']}/rank^{tlr_rule['beta']}")
    print(f"  TEs={tes}, Ranks={ranks}")
    print(f"  {total} runs, EPOCHS={EPOCHS}, SEED={SEED}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}")

    for te in tes:
        for rank in ranks:
            run_idx += 1
            tlr_rule = TLR_RULES[reinit_mode]
            tlr = tlr_rule['alpha'] / rank ** tlr_rule['beta']
            print(f"\n[{run_idx}/{total}] rank={rank}, TE={te}, lr={LR}, tlr={tlr:.10f}")

            torch.manual_seed(SEED)
            model = create_model(rank, te, tlr, reinit_mode)
            optimizer = AnalogSGD(model.parameters(), lr=LR)
            optimizer.regroup_param_groups(model)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            criterion = nn.NLLLoss()

            base_C = {}
            for name, tile in get_lrtt_tiles(model).items():
                C, A, B = tile.get_lrtt_component_weights()
                base_C[name] = C.detach().cpu().clone()

            t0 = time.time()
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
                correct = total_cnt = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                        target = target.to(DEVICE, non_blocking=True)
                        correct += model(data).argmax(1).eq(target).sum().item()
                        total_cnt += target.size(0)
                val_acc = 100.0 * correct / total_cnt
                scheduler.step()

                for name, tile in get_lrtt_tiles(model).items():
                    C, A, B = tile.get_lrtt_component_weights()
                    dC = C.detach().cpu() - base_C[name]
                    er = compute_erank(dC)
                    nt = tile.controller.num_transfers

                csv_w.writerow([reinit_mode, rank, te, epoch, f'{er:.6f}', f'{val_acc:.2f}', nt])

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:2d}: acc={val_acc:.2f}%, erank={er:.1f}, transfers={nt}")

            wall = time.time() - t0
            csv_f.flush()
            print(f"  FINAL: acc={val_acc:.2f}%, erank={er:.1f}, transfers={nt}, wall={wall:.0f}s")
            all_results.append({
                'rank': rank, 'te': te, 'lr': LR, 'tlr': tlr,
                'mode': reinit_mode,
                'final_acc': val_acc, 'final_erank': er,
                'num_transfers': nt, 'wall_time_sec': wall})

            with open(output_dir / "summary.json", 'w') as f:
                json.dump({'completed': run_idx, 'total': total,
                           'mode': reinit_mode,
                           'lr': LR,
                           'tlr_alpha': tlr_rule['alpha'],
                           'tlr_beta': tlr_rule['beta'],
                           'tlr_rule': f"{tlr_rule['alpha']}/rank^{tlr_rule['beta']}",
                           'epochs': EPOCHS, 'seed': SEED,
                           'results': all_results}, f, indent=2)

            del model; torch.cuda.empty_cache()

    csv_f.close()
    print(f"\n{'='*60}")
    print(f"SUMMARY ({reinit_mode})")
    for r in all_results:
        print(f"  TE={r['te']:4d} rank={r['rank']:2d}: erank={r['final_erank']:.1f}, acc={r['final_acc']:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
