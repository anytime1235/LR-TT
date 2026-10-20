# Notice

This repository is a modified version of the IBM Analog Hardware Acceleration
Kit (aihwkit), version 1.0.0, git commit `e1037c2a` of
<https://github.com/IBM/aihwkit> (Copyright 2020-2024 IBM, MIT License). The
simulator infrastructure -- analog tiles, device models, the C++/CUDA RPU
engine and the PyTorch bindings -- is IBM's work and is redistributed here
unchanged except for the files listed in `CHANGES_FROM_AIHWKIT.md`.

The Low-Rank Tiki-Taka (LR-TT) training algorithm, its Python implementation,
the small C++/CUDA changes that support fixed-scaling pulsed updates, and all
experiment, figure and analysis scripts were written by Young Woon Cho, Jaeweon
Park, Jongun Won and Sangbum Kim (Seoul National University). These additions
and modifications are Copyright (c) 2025-2026 by the authors and are released
under the same MIT License; the full license text, including IBM's original
copyright notice, is in `LICENSE`.

Files taken from aihwkit keep their original IBM copyright header. Files that
were modified carry an additional "Modifications" line; new files carry the
authors' header. The LR-TT modules that started from aihwkit file templates
(`lrtt_config.py`, `lrtt_python.py`, `lrtt_controller.py`, `lrtt_tile.py` and
`lrtt_kernels.py`) keep the IBM header with the Modifications line.

This project is not affiliated with or endorsed by IBM.
