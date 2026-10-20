# System-level latency, energy and storage model

`lrtt_cost_model.py` is the closed-form model behind the system-level analysis
of the paper (Methods, "System-level latency, energy, and storage analysis";
Supplementary Section S7). It produces the values of Fig. 7a-l, the energy,
cell, tile and latency entries of the summary table in Fig. 2b, Supplementary
Table S1 and Supplementary Fig. S12b,c. It needs only `numpy` (`matplotlib`
for `--plots`).

```bash
python lrtt_cost_model.py                 # nominal point: printed tables, CSV/JSON in ./output
python lrtt_cost_model.py --plots         # additionally draws output/fig7_model.png
python lrtt_cost_model.py --help          # every model parameter is a flag
```

All quantities are per token. `--target` selects the auxiliary mapping of the
nominal auxiliary-path table (default `qkvo`, Fig. 7a-d), `--system-target`
that of the full-system decomposition (default `all`, Fig. 7k,l and Table S1).
The Fig. 2b table uses the Q/K/V/O mapping at `r = 32`, `BL = 31` and the
reduced (ADC-less) read window for the LR-TT latency; its F1 entries come from
`experiments/bert_squad/`.

## Model

The model implements the counting rules and equations of Methods
("System-level latency, energy, and storage analysis") and Supplementary
Section S7 for the BERT-base encoder (`N_blk = 12`, `D = 768`, `D_ff = 3072`,
`H = 12`; `n_core = 4` core stages per block: Q/K/V grouped, O, FFN1, FFN2).
The targets `qkvo` (Q, K, V, O, 768 x 768), `ffn` (FFN1 3072 x 768, FFN2
768 x 3072) and `all` map 48, 24 and 72 matrices; the full system always
includes the core path of all 72 matrices. LR-TT factors of the same shape are
packed along the rank, `floor(T/r)` per tile row (A) or column (B) for tile
size `T`; a transfer event every `transfer_every` steps is amortized over
`B * S * transfer_every` token positions. Fig. 7i uses
`chi(BL) = t_upd / (t_proj + t_upd)` with `t_upd = BL * t_pulse`, and Fig. 7j
`R_E(kappa_e, r) = kappa_e sum(MN) / ((1 + kappa_e) r sum(M+N))` on the
Q/K/V/O mapping for `r = 16, 32, 64`.

## Parameters

| parameter | default | flag | basis |
|---|---|---|---|
| rank `r` | 32 | `--rank` | nominal operating point |
| pulse-train length `BL` | 31 | `--bl` | `BL * t_pulse = 155 ns` |
| sequence length `S`, batch `B` | 384, 48 | `--seq-len`, `--batch` | SQuAD fine-tuning; `B` enters only the transfer amortization |
| tile size `T`, MVM latency `T_MVM` | 256, 100 ns | `--tile`, `--t-mvm` | ALPINE tile (Klein et al. 2023) |
| analog energy `eps_AIMC` | 1/12.8 = 0.078 pJ/op | `--eps-aimc` | ALPINE 12.8 TOPS/W, 2 ops per multiply-accumulate |
| pulse duration `t_pulse` | 5 ns | `--t-pulse` | Rasch et al. 2024 |
| read window `t_proj`, reduced | 100 ns, 50 ns | `--t-proj`, `--t-proj-adcless` | one MVM window; co-design case (Fig. 7b,i) |
| update-to-MVM energy ratio `kappa_e` | 1 | `--kappa-e` | anchors 0.5, 1, 4 (Gokmen and Vlasov 2016, Soliman et al. 2023, Marinella et al. 2018) |
| digital `P_dig`, `eps_dig` | 1280 ops/ns, 1.0 pJ/op | `--p-dig`, `--eps-dig` | Chen et al. 2025, Horowitz 2014 |
| transfer period | 1 step | `--transfer-every` | one transfer event per batch (Methods) |
| `C_sm`, `N_LN`, `C_LN`, `C_GELU`, `N_res`, `C_res` | 4, 25, 18, 18, 2, 1 | `Params` attributes | softmax ops per element; LayerNorms (two per block and the embedding); ops per LayerNorm and GeLU element (forward and backward); residual additions per block; ops per residual element |

## Outputs

The script prints the nominal auxiliary-path table and the full-system
decomposition of both methods and writes to `output/` next to the script
(`--outdir`):

- `fig7a_d_nominal.csv`: TT and LR-TT with the conventional and the reduced
  read window at the nominal point (operations, latency per stage, energy,
  cells, tiles; Fig. 7a-d, Fig. 2b table);
- `fig7e_h_rank_sweep.csv`: TT and LR-TT operations, energy, cells, tiles and
  latency, and the TT/LR-TT ratios of the first four, versus `r` for each
  target (Fig. 7e-h);
- `fig7i_pulse_length.csv`, `fig7j_kappa_e.csv`: `chi(BL)` for both read
  windows and `R_E(kappa_e, r)` for `kappa_e` from 0.1 to 10 (Fig. 7i, 7j);
- `fig7k_l_seq_len.csv`: full-system latency and energy components and their
  shares (`lat_frac_*`, `e_frac_*`) versus `S` for both methods (Fig. 7k,l,
  Supp. Fig. S12b,c);
- `tableS1_decomposition.csv`: operations, latency and energy per component of
  LR-TT at the nominal point (Supp. Table S1);
- `nominal_summary.json`: parameters and the nominal values of all targets and
  both methods; `fig7_model.png`: preview of all panels (`--plots`).
