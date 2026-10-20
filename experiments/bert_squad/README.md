# BERT-base on SQuAD v1.1

Fine-tuning experiments of Fig. 5, Fig. 6 and Supplementary Figs. S7-S11.
Each part has its own README:

| Part | Content | README |
|---|---|---|
| `optuna_bert_squad_lrtt.py`, `optuna_bert_squad_digital.py`, `optuna_bert_squad_digital_lora.py` | Optuna searches of LR-TT (QKVO, FFN and all-layer mapping, rank and A/B bit-width sweeps, asymmetric and full 6T1C A/B devices) and of the digital full fine-tuning and LoRA baselines: search spaces and the parameter sets of the paper runs | `README_optuna.md` |
| `fine_bert_squad_lrtt.py`, `configs/`, `run_fine_configs.sh` | fixed-configuration LR-TT runs with tile diagnostics (operating point, ideal-transfer rank sweep, effective rank, asymmetry, noise and weight-bound sweeps) and the figure panels of each config group | `README_fine.md` |
| `tikitaka/` | Tiki-Taka and single-tile direct-SGD baselines, I/O-precision probes and sub-pulse update runs | `tikitaka/README.md` |
| `diagnostics/` | backward-gradient, I/O-precision and sub-pulse diagnostics (Fig. 5a-e) | `diagnostics/README.md` |
