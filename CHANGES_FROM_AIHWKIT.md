# Changes from aihwkit 1.0.0

This repository is based on the IBM Analog Hardware Acceleration Kit (aihwkit)
version 1.0.0, git commit `e1037c2a` of <https://github.com/IBM/aihwkit>. The
lists below name every file that was added, modified, removed or renamed
relative to that commit. All other files under `src/`, `tests/` and `cmake/`,
and `Makefile`, `pyproject.toml`, `setup.cfg`, `requirements.txt` and
`requirements-dev.txt`, are unchanged.

## Added

| File | Content |
|---|---|
| `src/aihwkit/simulator/configs/lrtt_python.py` | `PythonLRTTDevice`: parameters of an LR-TT block (rank, transfer period, rate and method, reinitialization, forward injection, devices of the `A`, `B` and `C` tiles) |
| `src/aihwkit/simulator/configs/lrtt_rpu_config.py` | `PythonLRTTRPUConfig` based on `SingleRPUConfig`, exported by `aihwkit.simulator.configs` and used by the BERT-base scripts |
| `src/aihwkit/simulator/configs/lrtt_config.py` | `PythonLRTTRPUConfig` based on `IOManagedRPUConfig`, used by the MNIST and regression scripts, and configuration helpers (`lrtt_idealized_config()` and others) |
| `src/aihwkit/simulator/tiles/lrtt_tile.py` | `LRTTSimulatorTile`: simulator tile built from the `A`, `B` and `C` analog tiles |
| `src/aihwkit/simulator/tiles/lrtt_controller.py` | `LRTTController`: projected pulsed updates of `A` and `B`, transfer into `C` (rank-wise one-hot, direct or exact), reinitialization and forward injection |
| `src/aihwkit/optim/lrtt_grad_accum_patch.py` | gradient accumulation for LR-TT tiles, applied on import: replaces `AnalogOptimizerMixin.step()` of aihwkit's analog optimizers (the inputs and errors stored for an LR-TT tile are applied per gradient-accumulation group; other analog tiles with several stored entries are updated once per entry) and `LRTTController._ab_weight_update_lora()`, and adds `AnalogOptimizerMixin._pad_and_cat()`; imported by the BERT-base LR-TT scripts |
| `src/aihwkit/linalg/lrtt_kernels.py` | PyTorch tensor helpers; `compose_lrtt_weights()` forms `C + alpha * A @ B` for `LRTTSimulatorTile.get_effective_weights()` |
| `tests/test_lrtt_smoke.py` | smoke test of an LR-TT layer on CPU and CUDA |
| `experiments/` | experiment code and run configurations of the paper |
| `NOTICE.md`, `CITATION.cff`, `requirements-paper.txt`, `CHANGES_FROM_AIHWKIT.md` | attribution, citation metadata, experiment requirements, this list |

## Modified

The C++/CUDA and Python changes add three fields to the pulsed-update
parameters: `use_manual_scaling`, `manual_x_scaling` and `manual_d_scaling`.
With `use_manual_scaling=True` an update uses the bit length `desired_BL` and
the fixed pulse-probability scales `manual_x_scaling` (input `x`) and
`manual_d_scaling` (error `d`) instead of scales derived from the learning
rate and the input and error magnitudes. The default (`False`) keeps the
behaviour of aihwkit.

| File | Change |
|---|---|
| `src/rpucuda/rpu_pulsed_meta_parameter.h`, `src/rpucuda/rpu_pulsed_meta_parameter.cpp` | the three fields; fixed bit length and scales in `performUpdateManagement()` |
| `src/rpucuda/dense_bit_line_maker.cpp`, `src/rpucuda/sparse_bit_line_maker.cpp` | the update-management path is also taken when `use_manual_scaling` is set |
| `src/rpucuda/cuda/bit_line_maker.cu` | CUDA path: update-management buffers enabled, fixed scales from `performUpdateManagement()`, neutral per-sample scale and pulse-count values |
| `src/rpucuda/cuda/update_management_helper.h`, `src/rpucuda/cuda/update_management_helper.cu` | `setNeutralValues()`: per-sample scales set to 1 and pulse counts to the bit length |
| `src/aihwkit/simulator/rpu_base_src/rpu_base_devices.cpp` | Python bindings of the three fields |
| `src/aihwkit/simulator/parameters/training.py` | the three fields in `UpdateParameters` |
| `src/aihwkit/simulator/configs/__init__.py` | exports `PythonLRTTDevice`, `PythonLRTTRPUConfig` and the configuration helpers of `lrtt_config.py` |
| `CMakeLists.txt` | compute capability 9.0 added to the default `RPU_CUDA_ARCHITECTURES` |
| `setup.py` | package description, URL and authors; `paper` extra |
| `MANIFEST.in` | `LICENSE`, `NOTICE.md`, `CITATION.cff` and `CHANGES_FROM_AIHWKIT.md` included |
| `.gitignore`, `README.md` | rewritten for this repository |

## Removed

`docs/`, `examples/`, `notebooks/`, `.github/`, `CHANGELOG.md`,
`CUDA.Dockerfile`, `requirements-examples.txt`, `.black`, `.clang-format`,
`.pylintrc`, `.readthedocs.yaml`, `.whitesource`.

## Renamed

`LICENSE.txt` is `LICENSE`. The license text is unchanged; the copyright line
of the LR-TT authors is added above the IBM copyright line, and a line naming
aihwkit and its URL follows the IBM line.
