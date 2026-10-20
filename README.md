# LR-TT: Low-Rank Tiki-Taka

Simulation code, experiment scripts and run configurations for

> Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
> *Low-Rank Tiki-Taka: Memory-efficient transformer on-chip training for
> analog in-memory computing* (submitted, 2026).

LR-TT replaces the full-size auxiliary matrix of the Tiki-Taka analog
training algorithm with two low-rank auxiliary tiles, `A` (r x N) and `B`
(M x r). Projected gradient updates accumulate on the auxiliary tiles by
stochastic pulses and are consolidated into the core tile `C` by a rank-wise
one-hot transfer every `tau` steps. The repository contains the simulator
implementation of this scheme, the BERT-base/SQuAD, MNIST and regression
experiments of the paper, the Tiki-Taka and digital baselines, and the
system-level latency/energy model.

In the simulator code the two factors are named the other way round:
`tile_a` holds the M x r factor (`B` above) and `tile_b` the r x N factor
(`A` above), and the transferred product BA appears as `A @ B`.

## Relationship to aihwkit

This repository is a modified version of the IBM Analog Hardware Acceleration
Kit, [aihwkit](https://github.com/IBM/aihwkit) v1.0.0 (git `e1037c2a`), which
IBM releases under the MIT License. The analog tiles, device models, the
C++/CUDA RPU engine and the PyTorch bindings are IBM's work. LR-TT adds a
Python implementation of the low-rank auxiliary update and transfer
(`src/aihwkit/simulator/{configs,tiles}/lrtt_*.py`,
`src/aihwkit/optim/lrtt_grad_accum_patch.py`,
`src/aihwkit/linalg/lrtt_kernels.py`), three fixed-scaling pulsed-update
fields in the C++/CUDA update path, and the experiment code under
`experiments/`. The complete list of added, modified and removed files is
in `CHANGES_FROM_AIHWKIT.md`; attribution and license details are in
`NOTICE.md` and `LICENSE`. The package keeps the `aihwkit` import name, so
it cannot be installed side by side with an unmodified aihwkit in the same
environment.

If you use the simulator, please also cite aihwkit (Rasch et al., AICAS 2021,
see `CITATION.cff`). This project is not affiliated with or endorsed by IBM.

## System requirements

* Linux x86-64 (tested on Ubuntu 24.04 with Python 3.12, PyTorch 2.7 and
  CUDA 12.8). The CUDA build needs an NVIDIA GPU (compute capability 7.5 or
  newer) and CUDA toolkit >= 11.
* Python 3.10-3.12, PyTorch >= 2.3 with a matching CUDA toolkit, CMake >= 3.18,
  Ninja, a C++17 compiler, OpenBLAS (`libopenblas-dev`), `scikit-build`,
  `pybind11`, `mypy` (stub generation during the build).
* Python packages for the experiments: `requirements-paper.txt` (the versions
  the paper's numbers were produced with).
* GPU memory: the BERT-base runs use a batch of 48 at sequence length 384 on a
  single GPU (about 40 GB). The batch can be split into micro-batches with
  gradient accumulation; the Optuna script does this for a trial that runs out
  of memory and records the factor in the journal (see
  `experiments/bert_squad/README_optuna.md`). MNIST and regression runs need
  less than 2 GB.

## Installation

```bash
git clone https://github.com/anytime1235/LR-TT.git
cd LR-TT

sudo apt install build-essential cmake ninja-build python3-dev libopenblas-dev
pip install torch torchvision          # pick the wheel matching your CUDA toolkit
pip install scikit-build pybind11 mypy
pip install -r requirements-paper.txt

export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.')
export CMAKE_ARGS="-GNinja -DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_BUILD_TYPE=Release"
USE_CUDA=1 pip install -e . --no-build-isolation
USE_CUDA=1 python setup.py build_ext --inplace
```

For a CPU-only build drop the `CMAKE_ARGS` line and use `USE_CUDA=0`. The
CUDA build compiles about 80 translation units and takes 10-20 minutes on a
16-core machine (`-j` is controlled by `CMAKE_BUILD_PARALLEL_LEVEL`; keep it
small on machines with less than 32 GB RAM). Check the installation with

```bash
python -c "import aihwkit; from aihwkit.simulator.rpu_base import cuda; \
from aihwkit.simulator.configs import PythonLRTTDevice, PythonLRTTRPUConfig; \
print(aihwkit.__version__, cuda.is_compiled())"
python -m pytest tests/test_lrtt_smoke.py
```

## Quick start

`tests/test_lrtt_smoke.py` builds a small LR-TT analog layer and trains it for
six steps, including three transfers; it runs in seconds on CPU or GPU. A full
small-scale experiment is one cell of the MNIST scratch-training runs of Fig. 4c:

```bash
cd experiments/mnist
python run_erank_fixed_lr.py --mode decay --ranks 8 --tes 10 --data-dir ./data
```

which trains the 784-256-10 MLP with the first layer mapped to an LR-TT block
(rank 8, transfer every 10 mini-batches, 6T1C-calibrated auxiliary device) for
30 epochs and reports the test accuracy and the effective rank of the core-tile
update per epoch (about 97 % test accuracy after 30 epochs; about two minutes
on a GPU). See `experiments/mnist/README.md` for the exact options.

## Experiments

`experiments/` holds the code and run configurations of the paper's experiments:

| Directory | Contents | Paper |
|---|---|---|
| `experiments/bert_squad/` | BERT-base fine-tuning on SQuAD v1.1: scripts, launchers and run configurations of the LR-TT fixed-configuration runs and Optuna searches, the Tiki-Taka and digital baselines, and the backward-gradient, I/O-precision and sub-pulse diagnostics | Figs. 5, 6; Supp. Section S6, Figs. S7-S11 |
| `experiments/mnist/` | MNIST scratch training with an LR-TT first layer | Fig. 4a-c; Supp. Section S5, Fig. S6 |
| `experiments/regression/` | linear-regression simulations of the core-state trajectories (9x9 problem) | Fig. 4d,e; Supp. Section S4, Fig. S5 |
| `experiments/system_model/` | closed-form latency, energy and storage model | Figs. 2b, 7; Supp. Section S7, Fig. S12b,c, Table S1 |

Each directory has a README with usage, settings and outputs.
`experiments/bert_squad/README.md` points to the READMEs of its parts: the
LR-TT fixed-configuration runs (`README_fine.md`), the Optuna searches of
LR-TT and of the digital baselines (`README_optuna.md`), the Tiki-Taka and
single-tile direct-SGD baselines (`tikitaka/README.md`) and the diagnostics
(`diagnostics/README.md`).

## Repository layout

```
src/aihwkit/              simulator package (aihwkit 1.0.0 + LR-TT modules)
src/rpucuda/              C++/CUDA RPU engine (aihwkit + fixed-scaling update fields)
tests/                    aihwkit test-suite and the LR-TT smoke test
experiments/bert_squad/   LR-TT, Tiki-Taka and digital BERT-base/SQuAD experiments, diagnostics
experiments/mnist/        MNIST scratch-training sweeps (Fig. 4a-c, Supp. Fig. S6)
experiments/regression/   9x9 core-trajectory regressions (Fig. 4d,e, Supp. Fig. S5)
experiments/system_model/ closed-form latency/energy/storage model (Fig. 2b, 7, Supp. Fig. S12b,c, Table S1)
```

## Citation

See `CITATION.cff`.

## License

MIT License. Copyright (c) 2025-2026 Young Woon Cho, Jaeweon Park, Jongun Won
and Sangbum Kim (Seoul National University) for the LR-TT additions and
modifications; Copyright 2020-2024 IBM for aihwkit. See `LICENSE` and
`NOTICE.md`.
