#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""Closed-form system-level cost model of Tiki-Taka and LR-TT on BERT-base.

Implements the latency, energy and storage analysis of the paper (Methods,
"System-level latency, energy, and storage analysis"; Supplementary Section
S7): per-token operation counts, auxiliary cell and tile counts,
auxiliary-path latency and energy, the pulse-length and update-energy
sensitivities, and the full-system decomposition into core MVM, auxiliary
path, digital attention and remaining digital operations.  Covers
Fig. 7a-l, the Fig. 2b table, Supplementary Table S1 and Supplementary
Fig. S12b,c.

All quantities are per token.  The batch size enters only through the
amortization of transfer events over B x S token positions.

Usage::

    python lrtt_cost_model.py            # nominal point: printed tables, CSV/JSON in ./output
    python lrtt_cost_model.py --plots    # also draws the panels (requires matplotlib)
"""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

TARGETS = ("qkvo", "ffn", "all")
METHODS = ("tt", "lrtt")
RANKS = (1, 2, 4, 8, 16, 32, 64)
PULSE_LENGTHS = (1, 2, 4, 8, 16, 31, 64, 128, 256)
SEQ_LENS = (16, 32, 64, 128, 256, 384, 512, 768, 1024)
KAPPA_RANKS = (16, 32, 64)
KAPPA_ANCHORS = (
    (0.5, "Gokmen 2016, RPU training simulation"),
    (1.0, "Soliman 2023, FeFET crossbar (nominal)"),
    (4.0, "Marinella 2018, ReRAM cell write/read"),
)
TARGET_MATRICES = {
    "qkvo": ("q", "k", "v", "o"),
    "ffn": ("ffn1", "ffn2"),
    "all": ("q", "k", "v", "o", "ffn1", "ffn2"),
}
# Core MVM stages per encoder block: Q/K/V grouped as one input-projection
# stage, then attention output, FFN1 and FFN2 (n_core = 4).
CORE_STAGES = (("q", "k", "v"), ("o",), ("ffn1",), ("ffn2",))
COMPONENTS = ("core", "aux_proj", "aux_upd", "aux_xfer", "attn", "other")
GROUPS = ("core", "aux", "attn", "other")


@dataclass(frozen=True)
class Matrix:
    """Mapped weight matrix W in R^{M x N} (M outputs, N inputs)."""

    name: str
    m: int
    n: int


@dataclass
class Params:
    """Model parameters; the defaults are the nominal operating point of the paper."""

    rank: int = 32  # LR-TT rank r
    bl: int = 31  # pulse-train length BL
    seq_len: int = 384  # sequence length S
    batch: int = 48  # batch size B (enters only the transfer amortization)
    tile: int = 256  # AIMC tile size (ALPINE 256 x 256 tile)
    t_mvm_ns: float = 100.0  # per-tile MVM latency T_MVM (ALPINE)
    t_pulse_ns: float = 5.0  # programming-pulse duration t_pulse (Rasch 2024)
    t_proj_ns: float = 100.0  # projection-read window t_proj (one MVM window)
    t_proj_adcless_ns: float = 50.0  # reduced projection-read window (co-design case)
    eps_aimc_pj: float = 1.0 / 12.8  # analog energy per op: 12.8 TOPS/W, 2 ops per MAC (0.078 pJ/op)
    kappa_e: float = 1.0  # update-to-MVM energy ratio eps_upd / eps_MVM
    p_dig: float = 1280.0  # digital throughput P_dig [ops/ns]
    eps_dig_pj: float = 1.0  # digital energy per op eps_dig [pJ/op]
    transfer_every: int = 1  # optimizer steps between transfer events
    # Digital operation counts, Supplementary Section S7 (O_attn and O_other).
    c_sm: int = 4  # ops per softmax element, forward or backward
    n_ln: int = 25  # LayerNorm instances: two per block plus the embedding LayerNorm
    c_ln: int = 18  # ops per LayerNorm element, forward plus backward
    c_gelu: int = 18  # ops per GeLU element, forward plus backward
    n_res: int = 2  # residual additions per block
    c_res: int = 1  # ops per residual element
    # BERT-base encoder.
    n_blk: int = 12
    d_model: int = 768
    d_ff: int = 3072
    n_heads: int = 12

    def block_matrices(self):
        d, f = self.d_model, self.d_ff
        return {
            "q": Matrix("q", d, d),
            "k": Matrix("k", d, d),
            "v": Matrix("v", d, d),
            "o": Matrix("o", d, d),
            "ffn1": Matrix("ffn1", f, d),
            "ffn2": Matrix("ffn2", d, f),
        }

    def matrices(self, target):
        """Mapped matrices of one target, repeated over the encoder blocks."""
        blk = self.block_matrices()
        return [blk[k] for _ in range(self.n_blk) for k in TARGET_MATRICES[target]]


def ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Auxiliary state: cells, tiles, operations
# ---------------------------------------------------------------------------


def packed_factor_tiles(n_factors, rank, long_dim, tile):
    """Tiles holding n_factors low-rank factors of shape rank x long_dim (A)
    or long_dim x rank (B).  Factors of the same shape are packed along the
    rank dimension, floor(tile / rank) per tile row (column) group."""
    if rank >= tile:
        groups = n_factors * ceil_div(rank, tile)
    else:
        groups = ceil_div(n_factors, tile // rank)
    return groups * ceil_div(long_dim, tile)


def aux_cells(mats, method, rank):
    """Trainable auxiliary cells: MN (TT) or r(M+N) (LR-TT) per matrix."""
    if method == "tt":
        return sum(w.m * w.n for w in mats)
    return sum(rank * (w.m + w.n) for w in mats)


def aux_tiles(mats, method, rank, tile):
    """Mapped auxiliary tiles: ceil(M/T) ceil(N/T) per TT matrix; LR-TT factors
    of the same parent shape packed along the rank dimension."""
    if method == "tt":
        return sum(ceil_div(w.m, tile) * ceil_div(w.n, tile) for w in mats)
    tiles = 0
    for (m, n), count in Counter((w.m, w.n) for w in mats).items():
        tiles += packed_factor_tiles(count, rank, n, tile)  # A: r x N
        tiles += packed_factor_tiles(count, rank, m, tile)  # B: M x r
    return tiles


def aux_path(p, target, method, rank=None, bl=None, kappa_e=None, t_proj_ns=None, seq_len=None):
    """Auxiliary-path cost per token of one method on one mapping target.

    Operations (one multiplication and one accumulation counted separately):
    TT update 2MN per matrix; LR-TT projection read and pulsed update
    2r(M+N) each.  Latency: one pulsed-update window BL*t_pulse per matrix
    for both methods, plus one projection-read window t_proj per matrix for
    LR-TT.  Energy: MVM-class ops at eps_AIMC, pulsed-update ops at
    kappa_e*eps_AIMC.  Transfer: one event every transfer_every optimizer
    steps, amortized over B*S*transfer_every token positions; per event each
    matrix reads the transferred auxiliary content (LR-TT 2r(M+N) ops, TT 2M
    ops for one column) and spends one pulsed-write window on the core tile.
    """
    rank = p.rank if rank is None else rank
    bl = p.bl if bl is None else bl
    kappa_e = p.kappa_e if kappa_e is None else kappa_e
    t_proj = p.t_proj_ns if t_proj_ns is None else t_proj_ns
    seq_len = p.seq_len if seq_len is None else seq_len
    mats = p.matrices(target)
    n_mat = len(mats)
    lrtt = method == "lrtt"
    if lrtt:
        ops_proj = sum(2 * rank * (w.m + w.n) for w in mats)
        ops_upd = ops_proj
        ops_xfer_event = ops_proj
    else:
        ops_proj = 0
        ops_upd = sum(2 * w.m * w.n for w in mats)
        ops_xfer_event = sum(2 * w.m for w in mats)
    tokens_per_xfer = p.batch * seq_len * p.transfer_every
    t_upd = bl * p.t_pulse_ns
    lat_proj = n_mat * t_proj if lrtt else 0.0
    lat_upd = n_mat * t_upd
    lat_xfer = n_mat * t_upd / tokens_per_xfer
    ops_xfer = ops_xfer_event / tokens_per_xfer
    eps = p.eps_aimc_pj
    e_proj = ops_proj * eps
    e_upd = kappa_e * ops_upd * eps
    e_xfer = kappa_e * ops_xfer * eps
    return {
        "method": method,
        "target": target,
        "rank": rank,
        "bl": bl,
        "t_proj_ns": t_proj if lrtt else 0.0,
        "n_mat": n_mat,
        "ops_proj": ops_proj,
        "ops_upd": ops_upd,
        "ops_xfer": ops_xfer,
        "ops_aux": ops_proj + ops_upd,  # reported auxiliary-path operation count
        "ops_total": ops_proj + ops_upd + ops_xfer,
        "cells": aux_cells(mats, method, rank),
        "tiles": aux_tiles(mats, method, rank, p.tile),
        "lat_proj_ns": lat_proj,
        "lat_upd_ns": lat_upd,
        "lat_xfer_ns": lat_xfer,
        "lat_total_ns": lat_proj + lat_upd + lat_xfer,
        "e_proj_pj": e_proj,
        "e_upd_pj": e_upd,
        "e_xfer_pj": e_xfer,
        "e_total_pj": e_proj + e_upd + e_xfer,
    }


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------


def update_fraction(p, bl, t_proj_ns):
    """chi(BL) = t_upd / (t_proj + t_upd) with t_upd = BL * t_pulse (Methods, pulse-length sensitivity)."""
    t_upd = np.asarray(bl, dtype=float) * p.t_pulse_ns
    return t_upd / (t_proj_ns + t_upd)


def energy_ratio(p, kappa_e, rank, target="qkvo"):
    """R_E(kappa_e, r) = kappa_e sum(MN) / ((1 + kappa_e) r sum(M+N)), Methods Eq. (16)."""
    mats = p.matrices(target)
    sum_mn = sum(w.m * w.n for w in mats)
    sum_m_plus_n = sum(w.m + w.n for w in mats)
    k = np.asarray(kappa_e, dtype=float)
    return k * sum_mn / ((1.0 + k) * rank * sum_m_plus_n)


# ---------------------------------------------------------------------------
# Full system
# ---------------------------------------------------------------------------


def digital_ops(p, seq_len=None):
    """Digital attention and remaining digital operations per token, Eqs. (S12)-(S13)."""
    s = p.seq_len if seq_len is None else seq_len
    attn = p.n_blk * (12 * p.d_model * s + 2 * p.c_sm * p.n_heads * s)
    other = (
        p.n_ln * p.c_ln * p.d_model
        + p.n_blk * p.c_gelu * p.d_ff
        + p.n_blk * p.n_res * p.c_res * p.d_model
    )
    return attn, other


def full_system(p, method, target="all", seq_len=None, rank=None, bl=None, kappa_e=None, t_proj_ns=None):
    """Full-system latency and energy per token, Supplementary Eqs. (S9)-(S15).

    The core path covers every linear operator of the encoder (its core
    tiles exist whatever the auxiliary mapping); the target selects the
    matrices that carry an auxiliary path.
    """
    s = p.seq_len if seq_len is None else seq_len
    core = p.matrices("all")
    ops_core = sum(4 * w.m * w.n for w in core)  # forward and backward MVM
    lat_core = 2 * p.n_blk * len(CORE_STAGES) * p.t_mvm_ns
    aux = aux_path(p, target, method, rank=rank, bl=bl, kappa_e=kappa_e, t_proj_ns=t_proj_ns, seq_len=s)
    ops_attn, ops_other = digital_ops(p, s)
    ops = {
        "core": ops_core,
        "aux_proj": aux["ops_proj"],
        "aux_upd": aux["ops_upd"],
        "aux_xfer": aux["ops_xfer"],
        "attn": ops_attn,
        "other": ops_other,
    }
    lat = {
        "core": lat_core,
        "aux_proj": aux["lat_proj_ns"],
        "aux_upd": aux["lat_upd_ns"],
        "aux_xfer": aux["lat_xfer_ns"],
        "attn": ops_attn / p.p_dig,
        "other": ops_other / p.p_dig,
    }
    energy = {
        "core": ops_core * p.eps_aimc_pj,
        "aux_proj": aux["e_proj_pj"],
        "aux_upd": aux["e_upd_pj"],
        "aux_xfer": aux["e_xfer_pj"],
        "attn": ops_attn * p.eps_dig_pj,
        "other": ops_other * p.eps_dig_pj,
    }
    out = {"method": method, "target": target, "seq_len": s, "rank": aux["rank"], "bl": aux["bl"]}
    for key, table in (("ops", ops), ("lat_ns", lat), ("e_pj", energy)):
        total = sum(table.values())
        for c in COMPONENTS:
            out["%s_%s" % (key, c)] = table[c]
        out["%s_aux" % key] = table["aux_proj"] + table["aux_upd"] + table["aux_xfer"]
        out["%s_total" % key] = total
        for g in GROUPS:
            out["%s_frac_%s" % (key, g)] = out["%s_%s" % (key, g)] / total
    return out


# ---------------------------------------------------------------------------
# Sweeps (one per figure panel)
# ---------------------------------------------------------------------------


def nominal_bars(p, target="qkvo"):
    """Fig. 7a-d: TT, LR-TT (conventional) and LR-TT (ADC-less) at the nominal point."""
    rows = []
    for label, method, t_proj in (
        ("tikitaka", "tt", None),
        ("lrtt_conventional", "lrtt", p.t_proj_ns),
        ("lrtt_adcless", "lrtt", p.t_proj_adcless_ns),
    ):
        a = aux_path(p, target, method, t_proj_ns=t_proj)
        rows.append(
            {
                "case": label,
                "target": target,
                "rank": a["rank"],
                "bl": a["bl"],
                "t_proj_ns": a["t_proj_ns"],
                "ops_proj": a["ops_proj"],
                "ops_upd": a["ops_upd"],
                "ops_aux": a["ops_aux"],
                "lat_proj_us": a["lat_proj_ns"] / 1e3,
                "lat_upd_us": a["lat_upd_ns"] / 1e3,
                "lat_xfer_us": a["lat_xfer_ns"] / 1e3,
                "lat_total_us": a["lat_total_ns"] / 1e3,
                "e_proj_uj": a["e_proj_pj"] / 1e6,
                "e_upd_uj": a["e_upd_pj"] / 1e6,
                "e_xfer_uj": a["e_xfer_pj"] / 1e6,
                "e_total_uj": a["e_total_pj"] / 1e6,
                "cells": a["cells"],
                "tiles": a["tiles"],
            }
        )
    return rows


def rank_sweep(p, ranks=RANKS, targets=TARGETS):
    """Fig. 7e-h: TT / LR-TT ratios of ops, energy, cells and tiles versus rank."""
    rows = []
    for target in targets:
        for r in ranks:
            tt = aux_path(p, target, "tt", rank=r)
            lr = aux_path(p, target, "lrtt", rank=r)
            rows.append(
                {
                    "target": target,
                    "rank": r,
                    "ops_tt": tt["ops_aux"],
                    "ops_lrtt": lr["ops_aux"],
                    "ops_ratio": tt["ops_aux"] / lr["ops_aux"],
                    "e_tt_uj": tt["e_total_pj"] / 1e6,
                    "e_lrtt_uj": lr["e_total_pj"] / 1e6,
                    "energy_ratio": tt["e_total_pj"] / lr["e_total_pj"],
                    "cells_tt": tt["cells"],
                    "cells_lrtt": lr["cells"],
                    "cell_ratio": tt["cells"] / lr["cells"],
                    "tiles_tt": tt["tiles"],
                    "tiles_lrtt": lr["tiles"],
                    "tile_ratio": tt["tiles"] / lr["tiles"],
                    "lat_tt_us": tt["lat_total_ns"] / 1e3,
                    "lat_lrtt_us": lr["lat_total_ns"] / 1e3,
                }
            )
    return rows


def bl_sweep(p, bls=PULSE_LENGTHS):
    """Fig. 7i: update fraction chi(BL) for the conventional and ADC-less read windows."""
    rows = []
    for bl in bls:
        t_upd = bl * p.t_pulse_ns
        rows.append(
            {
                "bl": bl,
                "t_upd_ns": t_upd,
                "chi_conventional": float(update_fraction(p, bl, p.t_proj_ns)),
                "chi_adcless": float(update_fraction(p, bl, p.t_proj_adcless_ns)),
                "t_proj_over_t_upd_conventional": p.t_proj_ns / t_upd,
                "t_proj_over_t_upd_adcless": p.t_proj_adcless_ns / t_upd,
            }
        )
    return rows


def kappa_sweep(p, kappas=None, ranks=KAPPA_RANKS, target="qkvo"):
    """Fig. 7j: R_E(kappa_e, r) on the Q/K/V/O mapping."""
    if kappas is None:
        kappas = np.round(np.logspace(-1, 1, 41), 6)
    rows = []
    for k in kappas:
        row = {"kappa_e": float(k)}
        for r in ranks:
            row["energy_ratio_r%d" % r] = float(energy_ratio(p, k, r, target))
        rows.append(row)
    return rows


def seq_len_sweep(p, seq_lens=SEQ_LENS, target="all"):
    """Fig. 7k,l and Supplementary Fig. S12b,c: full-system latency and energy versus S."""
    rows = []
    for s in seq_lens:
        for method in METHODS:
            f = full_system(p, method, target=target, seq_len=s)
            row = {"seq_len": s, "method": method, "target": target}
            for c in COMPONENTS + ("aux", "total"):
                row["lat_%s_us" % c] = f["lat_ns_%s" % c] / 1e3
                row["e_%s_uj" % c] = f["e_pj_%s" % c] / 1e6
            for g in GROUPS:
                row["lat_frac_%s" % g] = f["lat_ns_frac_%s" % g]
                row["e_frac_%s" % g] = f["e_pj_frac_%s" % g]
            rows.append(row)
    return rows


def decomposition_table(p, method="lrtt", target="all"):
    """Supplementary Table S1: component-level ops, latency and energy per token."""
    f = full_system(p, method, target=target)
    names = {
        "core": "Core-array MVM",
        "aux_proj": "Auxiliary projection read",
        "aux_upd": "Auxiliary pulse update",
        "aux_xfer": "Auxiliary-to-core transfer",
        "attn": "Digital attention",
        "other": "Digital other operations",
    }
    rows = []
    for c in COMPONENTS + ("total",):
        rows.append(
            {
                "component": names.get(c, "Total"),
                "key": c,
                "ops_per_token_M": f["ops_%s" % c] / 1e6,
                "latency_per_token_us": f["lat_ns_%s" % c] / 1e3,
                "energy_per_token_uJ": f["e_pj_%s" % c] / 1e6,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def print_table(header, rows, widths=None):
    widths = widths or [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def print_nominal(p, target):
    bars = nominal_bars(p, target)
    print("Auxiliary path, target=%s, r=%d, BL=%d, kappa_e=%g (per token)" % (target.upper(), p.rank, p.bl, p.kappa_e))
    header = ("case", "ops [M]", "cells [M]", "tiles", "lat proj [us]", "lat upd [us]", "lat total [us]", "E [uJ]")
    rows = []
    for b in bars:
        rows.append(
            (
                b["case"],
                "%.3f" % (b["ops_aux"] / 1e6),
                "%.3f" % (b["cells"] / 1e6),
                "%d" % b["tiles"],
                "%.3f" % b["lat_proj_us"],
                "%.3f" % b["lat_upd_us"],
                "%.3f" % b["lat_total_us"],
                "%.4f" % b["e_total_uj"],
            )
        )
    print_table(header, rows)
    print()


def print_decomposition(p, target):
    for method in ("lrtt", "tt"):
        rows = decomposition_table(p, method, target)
        print("Full system, %s, auxiliary target=%s, S=%d, r=%d, BL=%d (per token)" % (method.upper(), target.upper(), p.seq_len, p.rank, p.bl))
        header = ("component", "ops [x1e6]", "latency [us]", "energy [uJ]")
        out = [(r["component"], "%.3f" % r["ops_per_token_M"], "%.3f" % r["latency_per_token_us"], "%.3f" % r["energy_per_token_uJ"]) for r in rows]
        print_table(header, out)
        print()


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

COLOR_TT = "#eb6834"
COLOR_LRTT = "#2a78d6"
COLOR_TARGET = {"qkvo": "#2a78d6", "ffn": "#eb6834", "all": "#1baf7a"}
COLOR_GROUP = {"core": "#2a78d6", "aux": "#eda100", "attn": "#52514e", "other": "#b5b4ae"}


def draw_panels(p, outdir, bars, ranks, bls, kappas, seqs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    ax = axes.ravel()
    labels = ["TT", "LR-TT\n(conv.)", "LR-TT\n(ADC-less)"]
    x = np.arange(3)

    # (a) ops, (b) latency, (c) energy, (d) cells and tiles
    ax[0].bar(x[:2], [b["ops_upd"] / 1e6 for b in bars[:2]], color=[COLOR_TT, COLOR_LRTT], width=0.6)
    ax[0].bar(x[1], bars[1]["ops_proj"] / 1e6, bottom=bars[1]["ops_upd"] / 1e6, color=COLOR_LRTT, alpha=0.5, width=0.6)
    ax[0].set_xticks(x[:2])
    ax[0].set_xticklabels(labels[:2])
    ax[0].set_ylabel("Auxiliary ops / token [M]")
    ax[0].set_title("(a) ops, %s" % bars[0]["target"].upper())
    ax[1].bar(x, [b["lat_upd_us"] for b in bars], color=[COLOR_TT, COLOR_LRTT, COLOR_LRTT], width=0.6)
    ax[1].bar(x, [b["lat_proj_us"] for b in bars], bottom=[b["lat_upd_us"] for b in bars], color=COLOR_LRTT, alpha=0.5, width=0.6)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("Auxiliary latency / token [us]")
    ax[1].set_title("(b) latency (light: projection read)")
    ax[2].bar(x[:2], [b["e_total_uj"] for b in bars[:2]], color=[COLOR_TT, COLOR_LRTT], width=0.6)
    ax[2].set_xticks(x[:2])
    ax[2].set_xticklabels(labels[:2])
    ax[2].set_ylabel("Auxiliary energy / token [uJ]")
    ax[2].set_title("(c) energy")
    ax[3].bar([0, 1], [bars[0]["cells"], bars[1]["cells"]], color=[COLOR_TT, COLOR_LRTT], width=0.6)
    ax[3].bar([3, 4], [bars[0]["tiles"], bars[1]["tiles"]], color=[COLOR_TT, COLOR_LRTT], width=0.6)
    ax[3].set_yscale("log")
    ax[3].set_xticks([0, 1, 3, 4])
    ax[3].set_xticklabels(["TT\ncells", "LR-TT\ncells", "TT\ntiles", "LR-TT\ntiles"])
    ax[3].set_title("(d) auxiliary cells and tiles")

    # (e)-(h) ratios versus rank
    keys = (("ops_ratio", "(e) ops ratio"), ("energy_ratio", "(f) energy ratio"), ("cell_ratio", "(g) cell ratio"), ("tile_ratio", "(h) tile ratio"))
    for i, (key, title) in enumerate(keys):
        a = ax[4 + i]
        for target in TARGETS:
            rows = [r for r in ranks if r["target"] == target]
            a.plot([r["rank"] for r in rows], [r[key] for r in rows], marker="o", ms=4, lw=1.5, color=COLOR_TARGET[target], label=target.upper())
        a.set_xscale("log", base=2)
        a.set_xticks(RANKS)
        a.set_xticklabels([str(r) for r in RANKS])
        a.set_xlabel("rank r")
        a.set_ylabel("TT / LR-TT")
        a.set_title(title)
        a.legend(frameon=False)

    # (i) chi(BL)
    a = ax[8]
    a.plot([r["bl"] for r in bls], [r["chi_conventional"] for r in bls], marker="o", ms=4, lw=1.5, color=COLOR_LRTT, label="t_proj = %g ns" % p.t_proj_ns)
    a.plot([r["bl"] for r in bls], [r["chi_adcless"] for r in bls], marker="^", ms=4, lw=1.5, color=COLOR_TT, label="t_proj = %g ns" % p.t_proj_adcless_ns)
    a.set_xscale("log", base=2)
    a.set_xticks(PULSE_LENGTHS)
    a.set_xticklabels([str(b) for b in PULSE_LENGTHS])
    a.set_xlabel("pulse-train length BL")
    a.set_ylabel("t_upd / (t_proj + t_upd)")
    a.set_title("(i) update fraction")
    a.legend(frameon=False)

    # (j) R_E(kappa_e, r)
    a = ax[9]
    styles = {16: "--", 32: "-", 64: ":"}
    for r in KAPPA_RANKS:
        a.plot([k["kappa_e"] for k in kappas], [k["energy_ratio_r%d" % r] for k in kappas], styles[r], lw=1.5, color=COLOR_LRTT, label="r = %d" % r)
    for k, name in KAPPA_ANCHORS:
        a.plot([k], [float(energy_ratio(p, k, 32))], marker="s", ms=6, color=COLOR_TT, ls="none")
        a.annotate(name.split(",")[0], (k, float(energy_ratio(p, k, 32))), textcoords="offset points", xytext=(4, -10), fontsize=7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("update-to-MVM energy ratio kappa_e")
    a.set_ylabel("TT / LR-TT energy")
    a.set_title("(j) energy ratio, QKVO")
    a.legend(frameon=False)

    # (k), (l) full system versus S
    for i, (key, unit, title) in enumerate((("lat", "us", "(k) full-system latency"), ("e", "uJ", "(l) full-system energy"))):
        a = ax[10 + i]
        for method, color, name in (("tt", COLOR_TT, "TT"), ("lrtt", COLOR_LRTT, "LR-TT")):
            rows = [r for r in seqs if r["method"] == method]
            a.plot([r["seq_len"] for r in rows], [r["%s_total_%s" % (key, unit.lower())] for r in rows], marker="o", ms=4, lw=1.5, color=color, label=name)
            a.plot([r["seq_len"] for r in rows], [r["%s_aux_%s" % (key, unit.lower())] for r in rows], ls="--", lw=1.0, color=color, label="%s auxiliary" % name)
        a.set_xscale("log", base=2)
        ticks = (16, 32, 64, 128, 384, 1024)
        a.set_xticks(ticks)
        a.set_xticklabels([str(s) for s in ticks])
        a.set_xlabel("sequence length S")
        a.set_ylabel("per token [%s]" % unit)
        a.set_title(title)
        a.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path = os.path.join(outdir, "fig7_model.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--rank", type=int, default=32, help="LR-TT rank r")
    ap.add_argument("--bl", type=int, default=31, help="pulse-train length BL")
    ap.add_argument("--seq-len", type=int, default=384, help="sequence length S")
    ap.add_argument("--batch", type=int, default=48, help="batch size B (transfer amortization)")
    ap.add_argument("--tile", type=int, default=256, help="AIMC tile size")
    ap.add_argument("--t-mvm", type=float, default=100.0, help="per-tile MVM latency T_MVM [ns]")
    ap.add_argument("--t-pulse", type=float, default=5.0, help="programming-pulse duration t_pulse [ns]")
    ap.add_argument("--t-proj", type=float, default=100.0, help="projection-read window t_proj [ns]")
    ap.add_argument("--t-proj-adcless", type=float, default=50.0, help="reduced projection-read window [ns]")
    ap.add_argument("--eps-aimc", type=float, default=1.0 / 12.8, help="analog energy per op eps_AIMC [pJ/op]")
    ap.add_argument("--kappa-e", type=float, default=1.0, help="update-to-MVM energy ratio kappa_e")
    ap.add_argument("--p-dig", type=float, default=1280.0, help="digital throughput P_dig [ops/ns]")
    ap.add_argument("--eps-dig", type=float, default=1.0, help="digital energy per op eps_dig [pJ/op]")
    ap.add_argument("--transfer-every", type=int, default=1, help="optimizer steps between transfer events")
    ap.add_argument("--target", choices=TARGETS, default="qkvo", help="auxiliary mapping target of the nominal auxiliary-path table")
    ap.add_argument("--system-target", choices=TARGETS, default="all", help="auxiliary mapping target of the full-system decomposition")
    ap.add_argument("--outdir", default=os.path.join(here, "output"), help="directory for CSV/JSON (and plot) outputs")
    ap.add_argument("--plots", action="store_true", help="draw the panels with matplotlib")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    p = Params(
        rank=args.rank,
        bl=args.bl,
        seq_len=args.seq_len,
        batch=args.batch,
        tile=args.tile,
        t_mvm_ns=args.t_mvm,
        t_pulse_ns=args.t_pulse,
        t_proj_ns=args.t_proj,
        t_proj_adcless_ns=args.t_proj_adcless,
        eps_aimc_pj=args.eps_aimc,
        kappa_e=args.kappa_e,
        p_dig=args.p_dig,
        eps_dig_pj=args.eps_dig,
        transfer_every=args.transfer_every,
    )
    os.makedirs(args.outdir, exist_ok=True)

    print_nominal(p, args.target)
    print_decomposition(p, args.system_target)

    bars = nominal_bars(p, args.target)
    ranks = rank_sweep(p)
    bls = bl_sweep(p)
    kappas = kappa_sweep(p)
    seqs = seq_len_sweep(p, target=args.system_target)
    table = decomposition_table(p, "lrtt", args.system_target)
    write_csv(os.path.join(args.outdir, "fig7a_d_nominal.csv"), bars)
    write_csv(os.path.join(args.outdir, "fig7e_h_rank_sweep.csv"), ranks)
    write_csv(os.path.join(args.outdir, "fig7i_pulse_length.csv"), bls)
    write_csv(os.path.join(args.outdir, "fig7j_kappa_e.csv"), kappas)
    write_csv(os.path.join(args.outdir, "fig7k_l_seq_len.csv"), seqs)
    write_csv(os.path.join(args.outdir, "tableS1_decomposition.csv"), table)
    summary = {
        "params": asdict(p),
        "aux_path_nominal": {
            t: {
                "tt": aux_path(p, t, "tt"),
                "lrtt": aux_path(p, t, "lrtt"),
                "lrtt_adcless": aux_path(p, t, "lrtt", t_proj_ns=p.t_proj_adcless_ns),
            }
            for t in TARGETS
        },
        "full_system_nominal": {m: full_system(p, m, target=args.system_target) for m in METHODS},
        "kappa_anchors": [{"kappa_e": k, "source": name, "energy_ratio_r%d" % p.rank: float(energy_ratio(p, k, p.rank))} for k, name in KAPPA_ANCHORS],
    }
    write_json(os.path.join(args.outdir, "nominal_summary.json"), summary)

    if args.plots:
        print("plot written to %s" % draw_panels(p, args.outdir, bars, ranks, bls, kappas, seqs))
    print("outputs written to %s" % args.outdir)


if __name__ == "__main__":
    main()
