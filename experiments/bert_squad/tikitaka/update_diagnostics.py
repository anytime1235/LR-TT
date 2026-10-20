#!/usr/bin/env python
# coding=utf-8
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Exact tile-level update diagnostics using AnalogContext internals.

Supports two modes:
  A) grad_accum == 1:
     1. loss.backward()
     2. diag.snapshot_before_step()  -> capture x, d, w_before
     3. optimizer.step()
     4. diag.snapshot_after_step()   -> compare w_after vs w_before

  B) grad_accum > 1 (microbatch accumulation):
     For each microbatch m:
       1. loss.backward()
       2. diag.accumulate_microbatch()  -> G_l += d_m^T @ x_m (CPU)
       3. p.reset() clears analog_ctx
     At accumulation boundary:
       4. diag.snapshot_before_step()   -> uses accumulated G_l + capture w_before
       5. optimizer.step()
       6. diag.snapshot_after_step()    -> compare w_after vs w_before

For BERT AnalogLinear (non-indexed case):
  x_input: cat(ctx.analog_input, axis=-1 if in_trans else 0)
  d_input: cat(ctx.analog_grad_output, axis=-1 if out_trans else 0)
  Target: -lr * d_input.T @ x_input -> shape [out_features, in_features]

Update management (UM) - matches aihwkit defaults:
  update_bl_management=True:
    BL = min(ceil(lr * x_max * d_max / dw_min), desired_bl)
    A_base = B_base = sqrt(lr / (dw_min * BL))
  update_management=True:
    gamma = x_max / (alpha * d_max)      where alpha = um_grad_scale (default 1.0)
    A = A_base * gamma
    B = B_base / gamma
  Per-element pulse probabilities:
    p_i = clip(A * |d_i|, 0, 1)    (d-side, column pulse)
    q_j = clip(B * |x_j|, 0, 1)    (x-side, row pulse)
  Expected pulses per (i,j):
    mu_ij = BL * p_i * q_j = lr * |d_i| * |x_j| / dw_min   (invariant to BL/UM)
    mu < 1:  sub-pulse (fundamental dw_min limitation)
    p_i > 1 or q_j > 1: probability clipping (saturation)
"""

import os
import csv
import json
import math
from collections import defaultdict

import torch
import numpy as np

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext


class UpdateDiagnostics:
    """Exact per-tile update diagnostics with UM-aware pulse probability tracking.

    Tracks target (FP32) vs actual (pulsed) weight updates per AnalogLinear module.
    Uses wrapper-level tracking (not sub-tile level) since AnalogContext is shared.
    """

    def __init__(self, model, dw_min, layer_set=None, method=None, lr=0.016,
                 device_w_max=1.0, desired_bl=31, um_grad_scale=1.0):
        self.dw_min = dw_min
        self.method = method
        self.lr = lr  # for per-microbatch mu calculation
        self.device_w_max = device_w_max
        self.desired_bl = desired_bl
        self.um_grad_scale = um_grad_scale
        self.tile_registry = self._build_registry(model, layer_set)
        self.cumulative_target = {}
        self.cumulative_actual = {}
        self.cumulative_slow = {}  # TTv1: slow tile cumulative
        self.records = []
        self._before_cache = {}
        # Microbatch G_l accumulators: {key: Tensor on CPU}
        self._grad_accum = {}
        # Per-microbatch outer-product mu stats: {key: [list of mu_stats dicts]}
        self._mu_per_mb = {}
        # Per-microbatch UM scaling stats: {key: [list of scaling dicts]}
        self._mb_scaling = {}
        # Mu distribution histograms for ECDF plotting
        self.mu_histograms = []
        self._mu_bin_edges = np.concatenate([[0], np.logspace(-6, 4, 1000)])
        # Per-element p_i, q_j histograms for clipping analysis
        self.pi_histograms = []
        self.qj_histograms = []
        self._pq_bin_edges = np.concatenate([[0], np.logspace(-6, 4, 1000)])
        # Per-microbatch p_i/q_j accumulators: {key: [list of (pi_counts, qj_counts)]}
        self._pq_per_mb = {}

        # Initialize cumulative trackers
        for entry in self.tile_registry:
            key = entry["key"]
            self.cumulative_target[key] = None
            self.cumulative_actual[key] = None
            self.cumulative_slow[key] = None

    @staticmethod
    def _build_registry(model, layer_set=None):
        """Build registry of AnalogLinear modules with their contexts.

        Args:
            layer_set: If provided, only include modules from these encoder layer indices.
        """
        registry = []
        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue

            # Filter by layer index if layer_set is specified
            if layer_set is not None:
                import re
                m = re.search(r'layer\.(\d+)', name)
                if m and int(m.group(1)) not in layer_set:
                    continue

            # Get the AnalogContext parameter for this module
            analog_ctx = None
            for pname, param in module.named_parameters():
                if isinstance(param, AnalogContext):
                    analog_ctx = param
                    break

            if analog_ctx is None:
                continue

            # Determine subtype
            subtype = "unknown"
            if "query" in name:
                subtype = "query"
            elif "key" in name:
                subtype = "key"
            elif "value" in name:
                subtype = "value"
            elif "dense" in name:
                subtype = "dense"

            # Get the tile from the analog module
            tile = module.analog_module

            registry.append({
                "key": name,
                "module_name": name,
                "module": module,
                "tile": tile,
                "analog_ctx": analog_ctx,
                "subtype": subtype,
            })
        return registry

    def _compute_um_scaling(self, x_abs_max, d_abs_max):
        """Compute UM-aware scaling factors matching aihwkit behavior.

        Returns dict with BL_eff, gamma, A, B, p_max, q_max.
        """
        if self.dw_min <= 0 or x_abs_max <= 0 or d_abs_max <= 0:
            return {
                "x_abs_max": x_abs_max, "d_abs_max": d_abs_max,
                "bl_call": 0, "bl_eff": self.desired_bl,
                "gamma": 1.0, "A": 0.0, "B": 0.0,
                "p_max": 0.0, "q_max": 0.0,
                "p_clipped": 0, "q_clipped": 0,
            }
        # update_bl_management: dynamic BL
        bl_call = math.ceil(self.lr * x_abs_max * d_abs_max / self.dw_min)
        bl_eff = min(bl_call, self.desired_bl) if bl_call > 0 else self.desired_bl

        # Base scaling
        A_base = math.sqrt(self.lr / (self.dw_min * bl_eff))

        # update_management: gamma balancing
        gamma = x_abs_max / (self.um_grad_scale * d_abs_max)
        A = A_base * gamma
        B = A_base / gamma  # B_base == A_base, then /gamma

        # Max pulse probabilities (before clipping)
        p_max = A * d_abs_max
        q_max = B * x_abs_max

        return {
            "x_abs_max": x_abs_max,
            "d_abs_max": d_abs_max,
            "bl_call": bl_call,
            "bl_eff": bl_eff,
            "gamma": gamma,
            "A": A,
            "B": B,
            "p_max": p_max,
            "q_max": q_max,
            "p_clipped": int(p_max > 1.0),
            "q_clipped": int(q_max > 1.0),
        }

    @torch.no_grad()
    def snapshot_weights_before(self):
        """Capture w_before at the start of a grad_accum group.

        Must be called before the first microbatch's tile.update().
        For grad_accum > 1, tile updates happen per microbatch, so
        w_before must be captured before any of them fire.
        For TTv1: also captures slow tile weights separately.
        """
        self._w_before_cache = {}
        self._slow_before_cache = {}
        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]
            try:
                self._w_before_cache[key] = tile.get_weights()[0].clone().detach().cpu()
            except Exception:
                continue
            # TTv1: capture slow tile (hidden_weights_1)
            if self.method == "ttv1":
                try:
                    # tile may be TileModuleArray; iterate sub-tiles
                    sub_tiles = list(entry["module"].analog_tiles())
                    for st in sub_tiles:
                        if hasattr(st, 'get_hidden_parameters'):
                            hidden = st.get_hidden_parameters()
                            for hname, htensor in hidden.items():
                                if "hidden_weights_1" in hname:
                                    self._slow_before_cache[key] = htensor.clone().detach().cpu()
                                    break
                            if key in self._slow_before_cache:
                                break
                except Exception:
                    pass

    @torch.no_grad()
    def accumulate_microbatch(self, model):
        """Accumulate G_l += d_l^T @ x_l for current microbatch.

        Must be called after loss.backward() and before p.reset() clears
        analog_input/analog_grad_output. Raw x, d are not kept after
        computing d^T @ x, so memory cost is one [out, in] tensor per tile.

        Also captures UM-aware scaling stats (x_max, d_max -> BL, gamma, A, B)
        per microbatch using cheap scalar operations only.
        """
        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]
            ctx = entry["analog_ctx"]

            if not (ctx.analog_input and ctx.analog_grad_output):
                continue

            try:
                in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False
                x = torch.cat(
                    ctx.analog_input, dim=-1 if in_trans else 0
                ).detach().float()
                d = torch.cat(
                    ctx.analog_grad_output, dim=-1 if out_trans else 0
                ).detach().float()
            except Exception:
                continue

            # Flatten to 2D
            if x.ndim > 2:
                x = x.reshape(-1, x.shape[-1])
            if d.ndim > 2:
                d = d.reshape(-1, d.shape[-1])

            # d^T @ x -> [out_features, in_features], compute on GPU then move to CPU
            # No /N: aihwkit tile.update() uses raw outer product, not averaged
            G_m = (d.mT @ x).cpu()

            # --- Per-microbatch mu from the outer product G_m (cheap) ---
            mu_m = (G_m.abs() * self.lr / self.dw_min)
            mb_stats = {
                "frac_mu_lt_1": (mu_m < 1.0).float().mean().item(),
                "frac_mu_lt_0p25": (mu_m < 0.25).float().mean().item(),
                "mu_p50": mu_m.median().item(),
                "mu_p90": mu_m.quantile(0.9).item(),
                "mu_mean": mu_m.mean().item(),
                "mu_max": mu_m.max().item(),
            }
            if key not in self._mu_per_mb:
                self._mu_per_mb[key] = []
            self._mu_per_mb[key].append(mb_stats)

            # --- UM-aware scaling (cheap: 2 scalar reductions + arithmetic) ---
            x_abs_max = x.abs().max().item()
            d_abs_max = d.abs().max().item()
            scaling = self._compute_um_scaling(x_abs_max, d_abs_max)

            if key not in self._mb_scaling:
                self._mb_scaling[key] = []
            self._mb_scaling[key].append(scaling)

            # --- Per-element p_i, q_j histograms (cheap: abs + scalar mul) ---
            A = scaling["A"]
            B = scaling["B"]
            if A > 0 and B > 0:
                # p_i = A * |d_i| for each row sample, q_j = B * |x_j| for each col sample
                # Flatten across batch: d is [batch, out_features], x is [batch, in_features]
                pi_vals = (A * d.abs()).cpu().flatten().numpy()
                qj_vals = (B * x.abs()).cpu().flatten().numpy()
                pi_counts, _ = np.histogram(pi_vals, bins=self._pq_bin_edges)
                qj_counts, _ = np.histogram(qj_vals, bins=self._pq_bin_edges)
                if key not in self._pq_per_mb:
                    self._pq_per_mb[key] = []
                self._pq_per_mb[key].append((pi_counts, qj_counts))

            # G_l accumulation
            if key in self._grad_accum:
                self._grad_accum[key] += G_m
            else:
                self._grad_accum[key] = G_m

    def snapshot_before_step(self, model, optimizer):
        """Prepare target update info before optimizer.step().

        For grad_accum == 1: extracts x, d from AnalogContext + captures w_before.
        For grad_accum > 1: uses pre-accumulated G_l + w_before from snapshot_weights_before().
        """
        lr = _get_current_analog_lr(optimizer)
        use_accum = bool(self._grad_accum)
        self._before_cache = {}

        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]
            ctx = entry["analog_ctx"]

            # w_before: from early snapshot (grad_accum>1) or capture now (grad_accum==1)
            if use_accum and hasattr(self, '_w_before_cache') and key in self._w_before_cache:
                w_before = self._w_before_cache[key]
            else:
                try:
                    w_before = tile.get_weights()[0].clone().detach().cpu()
                except Exception:
                    continue

            x_input = None
            d_input = None
            G_accumulated = None

            if use_accum and key in self._grad_accum:
                G_accumulated = self._grad_accum[key]
            elif ctx.analog_input and ctx.analog_grad_output:
                try:
                    in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                    out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False
                    x_input = torch.cat(
                        ctx.analog_input, dim=-1 if in_trans else 0
                    ).detach().cpu().float()
                    d_input = torch.cat(
                        ctx.analog_grad_output, dim=-1 if out_trans else 0
                    ).detach().cpu().float()
                except Exception:
                    pass

            self._before_cache[key] = {
                "w_before": w_before,
                "x_input": x_input,
                "d_input": d_input,
                "G_accumulated": G_accumulated,
                "lr": lr,
            }

        # Clear accumulators and caches
        self._grad_accum = {}
        self._w_before_cache = {}

    def snapshot_after_step(self, model, step):
        """Capture weights after optimizer.step() and compute metrics.

        Must be called after optimizer.step().
        """
        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]

            if key not in self._before_cache:
                continue
            cache = self._before_cache[key]

            try:
                w_after = tile.get_weights()[0].clone().detach().cpu()
            except Exception:
                continue

            w_before = cache["w_before"]
            x_input = cache["x_input"]
            d_input = cache["d_input"]
            G_accumulated = cache["G_accumulated"]
            lr = cache["lr"]

            delta_actual = (w_after - w_before).float()

            # Compute FP32 target update
            delta_target = None
            if G_accumulated is not None:
                # grad_accum > 1: target = -lr * G_l (already accumulated)
                delta_target = -lr * G_accumulated

                # Truncate to matching shape
                min_d = min(delta_target.shape[0], delta_actual.shape[0])
                min_x = min(delta_target.shape[1], delta_actual.shape[1])
                delta_target = delta_target[:min_d, :min_x]
                delta_actual_matched = delta_actual[:min_d, :min_x]
            elif x_input is not None and d_input is not None:
                # grad_accum == 1: compute directly
                if x_input.ndim > 2:
                    x_input = x_input.reshape(-1, x_input.shape[-1])
                if d_input.ndim > 2:
                    d_input = d_input.reshape(-1, d_input.shape[-1])

                n_x = x_input.shape[0]
                n_d = d_input.shape[0]
                if n_x != n_d:
                    x_mean = x_input.mean(dim=0)
                    d_mean = d_input.mean(dim=0)
                    delta_target = -lr * torch.outer(d_mean, x_mean)
                else:
                    delta_target = -lr * (d_input.mT @ x_input) / n_x

                min_d = min(delta_target.shape[0], delta_actual.shape[0])
                min_x = min(delta_target.shape[1], delta_actual.shape[1])
                delta_target = delta_target[:min_d, :min_x]
                delta_actual_matched = delta_actual[:min_d, :min_x]
            else:
                delta_actual_matched = delta_actual

            # Compute metrics
            actual_norm = torch.norm(delta_actual).item()

            # Weight distribution stats
            w_flat = w_after.float()
            w_abs_max = w_flat.abs().max().item()
            w_abs_p99 = w_flat.abs().quantile(0.99).item()
            w_std = w_flat.std().item()
            w_mean = w_flat.mean().item()
            w_utilization = w_abs_max / self.device_w_max if self.device_w_max > 0 else 0.0
            w_clipped_frac = (w_flat.abs() >= self.device_w_max * 0.999).float().mean().item()

            record = {
                "step": step,
                "tile_name": key,
                "subtype": entry["subtype"],
                "actual_norm": actual_norm,
                "lr": lr,
                "w_abs_max": w_abs_max,
                "w_abs_p99": w_abs_p99,
                "w_std": w_std,
                "w_mean": w_mean,
                "w_utilization": w_utilization,
                "w_clipped_frac": w_clipped_frac,
            }

            if delta_target is not None:
                residual = delta_actual_matched - delta_target

                target_norm = torch.norm(delta_target).item()
                residual_norm = torch.norm(residual).item()

                # Cosine similarity
                flat_actual = delta_actual_matched.flatten()
                flat_target = delta_target.flatten()
                cos_denom = (torch.norm(flat_actual) * torch.norm(flat_target))
                if cos_denom > 0:
                    cosine_sim = (flat_actual @ flat_target / cos_denom).item()
                else:
                    cosine_sim = 0.0

                # Signed bias: mean((actual - target) * sign(target))
                sign_target = torch.sign(delta_target)
                signed_bias = (residual * sign_target).mean().item()

                # Zero fraction: elements where |actual| < 0.5 * dw_min
                zero_frac = (delta_actual_matched.abs() < 0.5 * self.dw_min).float().mean().item()

                # Cumulative recovery
                if self.cumulative_target[key] is None:
                    self.cumulative_target[key] = delta_target.clone()
                    self.cumulative_actual[key] = delta_actual_matched.clone()
                else:
                    self.cumulative_target[key] += delta_target
                    self.cumulative_actual[key] += delta_actual_matched

                cum_target_norm = torch.norm(self.cumulative_target[key]).item()
                cum_actual_norm = torch.norm(self.cumulative_actual[key]).item()
                recovery_ratio = cum_actual_norm / max(cum_target_norm, 1e-12)

                # ============================================================
                # Mu distribution - accumulated target (total expected pulses)
                # ============================================================
                mu_eff = delta_target.abs() / self.dw_min
                eff_frac_mu_lt_1 = (mu_eff < 1.0).float().mean().item()
                eff_frac_mu_lt_0p25 = (mu_eff < 0.25).float().mean().item()
                eff_mu_p50 = mu_eff.median().item()
                eff_mu_p90 = mu_eff.quantile(0.9).item()
                eff_mu_mean = mu_eff.mean().item()
                eff_mu_max = mu_eff.max().item()

                # Per-microbatch mu (from the outer-product stats)
                if key in self._mu_per_mb and self._mu_per_mb[key]:
                    mb_list = self._mu_per_mb[key]
                    mb_frac_mu_lt_1 = float(np.mean([s["frac_mu_lt_1"] for s in mb_list]))
                    mb_frac_mu_lt_0p25 = float(np.mean([s["frac_mu_lt_0p25"] for s in mb_list]))
                    mb_mu_p50 = float(np.mean([s["mu_p50"] for s in mb_list]))
                    mb_mu_p90 = float(np.mean([s["mu_p90"] for s in mb_list]))
                    mb_mu_mean = float(np.mean([s["mu_mean"] for s in mb_list]))
                    mb_mu_max = float(np.max([s["mu_max"] for s in mb_list]))
                else:
                    # grad_accum == 1: same as eff
                    mb_frac_mu_lt_1 = eff_frac_mu_lt_1
                    mb_frac_mu_lt_0p25 = eff_frac_mu_lt_0p25
                    mb_mu_p50 = eff_mu_p50
                    mb_mu_p90 = eff_mu_p90
                    mb_mu_mean = eff_mu_mean
                    mb_mu_max = eff_mu_max

                # Store mu histogram for ECDF plotting (per-microbatch representative)
                n_mb = max(len(self._mu_per_mb.get(key, [])), 1)
                mu_for_hist = (delta_target.abs() / n_mb) / self.dw_min
                mu_counts, _ = np.histogram(
                    mu_for_hist.flatten().numpy(), bins=self._mu_bin_edges)
                self.mu_histograms.append({
                    "step": step,
                    "tile_name": key,
                    "subtype": entry["subtype"],
                    "n_elements": int(mu_for_hist.numel()),
                    "counts": mu_counts.tolist(),
                })

                # Store per-element p_i, q_j histograms (aggregated across microbatches)
                if key in self._pq_per_mb and self._pq_per_mb[key]:
                    pi_total = sum(pc for pc, _ in self._pq_per_mb[key])
                    qj_total = sum(qc for _, qc in self._pq_per_mb[key])
                    self.pi_histograms.append({
                        "step": step, "tile_name": key, "subtype": entry["subtype"],
                        "counts": pi_total.tolist(),
                    })
                    self.qj_histograms.append({
                        "step": step, "tile_name": key, "subtype": entry["subtype"],
                        "counts": qj_total.tolist(),
                    })

                # NSR and effective bits (cumulative)
                cum_residual = self.cumulative_actual[key] - self.cumulative_target[key]
                cum_residual_norm = torch.norm(cum_residual).item()
                cum_nsr = cum_residual_norm / max(cum_target_norm, 1e-12)
                cum_eff_bits = max(0, np.log2(max(cum_target_norm, 1e-30) / max(cum_residual_norm, 1e-30)))

                record.update({
                    "target_norm": target_norm,
                    "residual_norm": residual_norm,
                    "cosine_sim": cosine_sim,
                    "signed_bias": signed_bias,
                    "zero_frac": zero_frac,
                    "recovery_ratio": recovery_ratio,
                    "cum_nsr": cum_nsr,
                    "cum_eff_bits": cum_eff_bits,
                    # Effective mu of the accumulated update (full batch)
                    "eff_frac_mu_lt_1": eff_frac_mu_lt_1,
                    "eff_frac_mu_lt_0p25": eff_frac_mu_lt_0p25,
                    "eff_mu_p50": eff_mu_p50,
                    "eff_mu_p90": eff_mu_p90,
                    "eff_mu_mean": eff_mu_mean,
                    "eff_mu_max": eff_mu_max,
                    # Per-microbatch mu (one tile.update() per micro-batch)
                    "mb_frac_mu_lt_1": mb_frac_mu_lt_1,
                    "mb_frac_mu_lt_0p25": mb_frac_mu_lt_0p25,
                    "mb_mu_p50": mb_mu_p50,
                    "mb_mu_p90": mb_mu_p90,
                    "mb_mu_mean": mb_mu_mean,
                    "mb_mu_max": mb_mu_max,
                })

                # ============================================================
                # UM-aware pulse probability (matches aihwkit tile.update())
                # ============================================================
                if key in self._mb_scaling and self._mb_scaling[key]:
                    sc_list = self._mb_scaling[key]
                    n_sc = len(sc_list)

                    # Aggregate per-microbatch UM stats
                    avg_bl_eff = float(np.mean([s["bl_eff"] for s in sc_list]))
                    max_bl_call = max(s["bl_call"] for s in sc_list)
                    any_bl_saturated = int(max_bl_call > self.desired_bl)
                    avg_gamma = float(np.mean([s["gamma"] for s in sc_list]))
                    avg_A = float(np.mean([s["A"] for s in sc_list]))
                    avg_B = float(np.mean([s["B"] for s in sc_list]))
                    avg_p_max = float(np.mean([s["p_max"] for s in sc_list]))
                    avg_q_max = float(np.mean([s["q_max"] for s in sc_list]))
                    frac_mb_p_clip = float(np.mean([s["p_clipped"] for s in sc_list]))
                    frac_mb_q_clip = float(np.mean([s["q_clipped"] for s in sc_list]))
                    max_x_abs = float(np.max([s["x_abs_max"] for s in sc_list]))
                    max_d_abs = float(np.max([s["d_abs_max"] for s in sc_list]))

                    # Per-call mu: average expected pulses per single tile.update()
                    mu_per_call = mu_eff / n_sc

                    # Per-slot coincidence probability: p_i * q_j = mu_ij / BL_eff
                    # This is what the hardware actually sees per pulse slot
                    slot_prob = mu_per_call / max(avg_bl_eff, 1.0)

                    record.update({
                        # UM scaling factors
                        "um_bl_eff": avg_bl_eff,
                        "um_bl_call_max": max_bl_call,
                        "um_bl_saturated": any_bl_saturated,
                        "um_gamma": avg_gamma,
                        "um_A": avg_A,
                        "um_B": avg_B,
                        # Max pulse probabilities (before clipping)
                        "um_p_max": avg_p_max,
                        "um_q_max": avg_q_max,
                        # Clipping analysis (saturation)
                        "um_frac_mb_p_clip": frac_mb_p_clip,
                        "um_frac_mb_q_clip": frac_mb_q_clip,
                        # Input magnitudes
                        "um_x_abs_max": max_x_abs,
                        "um_d_abs_max": max_d_abs,
                        # Per-slot coincidence probability distribution
                        "slot_prob_mean": slot_prob.mean().item(),
                        "slot_prob_p50": slot_prob.median().item(),
                        "slot_prob_p90": slot_prob.quantile(0.9).item(),
                        "slot_prob_p99": slot_prob.quantile(0.99).item(),
                        "slot_prob_max": slot_prob.max().item(),
                        # Sub-pulse: elements with < 1 expected pulse per call
                        "um_frac_mu_per_call_lt_1": (mu_per_call < 1.0).float().mean().item(),
                    })
                else:
                    # grad_accum==1 or no scaling data: compute from x_input/d_input
                    if x_input is not None and d_input is not None:
                        x_max = x_input.abs().max().item()
                        d_max = d_input.abs().max().item()
                        sc = self._compute_um_scaling(x_max, d_max)
                        bl_eff = sc["bl_eff"]
                        slot_prob = mu_eff / max(bl_eff, 1.0)
                        record.update({
                            "um_bl_eff": float(bl_eff),
                            "um_bl_call_max": sc["bl_call"],
                            "um_bl_saturated": int(sc["bl_call"] > self.desired_bl),
                            "um_gamma": sc["gamma"],
                            "um_A": sc["A"],
                            "um_B": sc["B"],
                            "um_p_max": sc["p_max"],
                            "um_q_max": sc["q_max"],
                            "um_frac_mb_p_clip": float(sc["p_clipped"]),
                            "um_frac_mb_q_clip": float(sc["q_clipped"]),
                            "um_x_abs_max": x_max,
                            "um_d_abs_max": d_max,
                            "slot_prob_mean": slot_prob.mean().item(),
                            "slot_prob_p50": slot_prob.median().item(),
                            "slot_prob_p90": slot_prob.quantile(0.9).item(),
                            "slot_prob_p99": slot_prob.quantile(0.99).item(),
                            "slot_prob_max": slot_prob.max().item(),
                            "um_frac_mu_per_call_lt_1": eff_frac_mu_lt_1,
                        })
                    else:
                        record.update(_um_defaults())

            else:
                record.update({
                    "target_norm": 0.0,
                    "residual_norm": 0.0,
                    "cosine_sim": 0.0,
                    "signed_bias": 0.0,
                    "zero_frac": 0.0,
                    "recovery_ratio": 0.0,
                    "cum_nsr": 0.0,
                    "cum_eff_bits": 0.0,
                    "eff_frac_mu_lt_1": 0.0, "eff_frac_mu_lt_0p25": 0.0,
                    "eff_mu_p50": 0.0, "eff_mu_p90": 0.0,
                    "eff_mu_mean": 0.0, "eff_mu_max": 0.0,
                    "mb_frac_mu_lt_1": 0.0, "mb_frac_mu_lt_0p25": 0.0,
                    "mb_mu_p50": 0.0, "mb_mu_p90": 0.0,
                    "mb_mu_mean": 0.0, "mb_mu_max": 0.0,
                })
                record.update(_um_defaults())

            # TTv1: fast tile (A) weight distribution
            if self.method == "ttv1":
                try:
                    sub_tiles = list(entry["module"].analog_tiles())
                    for st in sub_tiles:
                        if hasattr(st, 'get_hidden_parameters'):
                            hidden = st.get_hidden_parameters()
                            for hname, htensor in hidden.items():
                                if "hidden_weights_0" in hname:
                                    fast_w = htensor.clone().detach().cpu().float()
                                    record.update({
                                        "fast_w_abs_max": fast_w.abs().max().item(),
                                        "fast_w_abs_p99": fast_w.abs().quantile(0.99).item(),
                                        "fast_w_abs_p50": fast_w.abs().median().item(),
                                        "fast_w_std": fast_w.std().item(),
                                        "fast_w_mean": fast_w.mean().item(),
                                    })
                                    break
                            if "fast_w_abs_max" in record:
                                break
                except Exception:
                    pass
            if "fast_w_abs_max" not in record:
                record.update({"fast_w_abs_max": 0.0, "fast_w_abs_p99": 0.0,
                               "fast_w_abs_p50": 0.0, "fast_w_std": 0.0, "fast_w_mean": 0.0})

            # TTv1: slow tile metrics
            if self.method == "ttv1" and key in getattr(self, '_slow_before_cache', {}):
                try:
                    slow_after = None
                    sub_tiles = list(entry["module"].analog_tiles())
                    for st in sub_tiles:
                        if hasattr(st, 'get_hidden_parameters'):
                            hidden = st.get_hidden_parameters()
                            for hname, htensor in hidden.items():
                                if "hidden_weights_1" in hname:
                                    slow_after = htensor.clone().detach().cpu()
                                    break
                            if slow_after is not None:
                                break
                    if slow_after is not None:
                        slow_before = self._slow_before_cache[key]
                        delta_slow = (slow_after - slow_before).float()
                        slow_norm = torch.norm(delta_slow).item()

                        # Cumulative slow
                        if self.cumulative_slow[key] is None:
                            self.cumulative_slow[key] = delta_slow.clone()
                        else:
                            self.cumulative_slow[key] += delta_slow

                        # Slow vs target comparison
                        slow_cos = 0.0
                        slow_recovery = 0.0
                        if delta_target is not None:
                            min_d = min(delta_slow.shape[0], delta_target.shape[0])
                            min_x = min(delta_slow.shape[1], delta_target.shape[1])
                            ds = delta_slow[:min_d, :min_x].flatten()
                            dt = delta_target[:min_d, :min_x].flatten()
                            denom = torch.norm(ds) * torch.norm(dt)
                            if denom > 0:
                                slow_cos = (ds @ dt / denom).item()

                            cum_slow_norm = torch.norm(self.cumulative_slow[key]).item()
                            cum_target_norm_s = torch.norm(self.cumulative_target[key]).item() if self.cumulative_target[key] is not None else 1e-12
                            slow_recovery = cum_slow_norm / max(cum_target_norm_s, 1e-12)

                        # Slow NSR and effective bits (cumulative)
                        slow_nsr = 0.0
                        slow_eff_bits = 0.0
                        if delta_target is not None and self.cumulative_target[key] is not None:
                            min_d = min(self.cumulative_slow[key].shape[0], self.cumulative_target[key].shape[0])
                            min_x = min(self.cumulative_slow[key].shape[1], self.cumulative_target[key].shape[1])
                            cum_slow_matched = self.cumulative_slow[key][:min_d, :min_x]
                            cum_target_matched = self.cumulative_target[key][:min_d, :min_x]
                            cum_slow_residual = cum_slow_matched - cum_target_matched
                            cum_slow_res_norm = torch.norm(cum_slow_residual).item()
                            cum_tgt_norm = torch.norm(cum_target_matched).item()
                            slow_nsr = cum_slow_res_norm / max(cum_tgt_norm, 1e-12)
                            slow_eff_bits = max(0, np.log2(max(cum_tgt_norm, 1e-30) / max(cum_slow_res_norm, 1e-30)))

                        record.update({
                            "slow_norm": slow_norm,
                            "slow_cosine_sim": slow_cos,
                            "slow_recovery": slow_recovery,
                            "slow_nsr": slow_nsr,
                            "slow_eff_bits": slow_eff_bits,
                        })
                except Exception:
                    pass

            # Fill slow fields with 0 if not TTv1
            if "slow_norm" not in record:
                record.update({"slow_norm": 0.0, "slow_cosine_sim": 0.0, "slow_recovery": 0.0, "slow_nsr": 0.0, "slow_eff_bits": 0.0})

            self.records.append(record)

        self._before_cache = {}
        self._slow_before_cache = {}
        self._mu_per_mb = {}
        self._mb_scaling = {}
        self._pq_per_mb = {}

    def save(self, output_dir):
        """Save diagnostics to CSV and JSON summary."""
        os.makedirs(output_dir, exist_ok=True)

        # CSV: per-step, per-tile metrics
        csv_path = os.path.join(output_dir, "update_diagnostics.csv")
        if self.records:
            fieldnames = list(self.records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.records:
                    writer.writerow(record)

        # JSON: summary statistics per tile
        summary = {}
        tile_records = defaultdict(list)
        for r in self.records:
            tile_records[r["tile_name"]].append(r)

        for tile_name, recs in tile_records.items():
            if not recs:
                continue
            summary[tile_name] = {
                "n_steps": len(recs),
                "subtype": recs[0]["subtype"],
                "mean_target_norm": float(np.mean([r["target_norm"] for r in recs])),
                "mean_actual_norm": float(np.mean([r["actual_norm"] for r in recs])),
                "mean_residual_norm": float(np.mean([r["residual_norm"] for r in recs])),
                "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in recs])),
                "mean_signed_bias": float(np.mean([r["signed_bias"] for r in recs])),
                "mean_zero_frac": float(np.mean([r["zero_frac"] for r in recs])),
                "final_recovery_ratio": recs[-1]["recovery_ratio"],
                "mean_eff_frac_mu_lt_1": float(np.mean([r["eff_frac_mu_lt_1"] for r in recs])),
                "mean_eff_mu_p50": float(np.mean([r["eff_mu_p50"] for r in recs])),
                "mean_eff_mu_mean": float(np.mean([r["eff_mu_mean"] for r in recs])),
                "max_eff_mu_max": float(np.max([r["eff_mu_max"] for r in recs])),
                "mean_mb_frac_mu_lt_1": float(np.mean([r["mb_frac_mu_lt_1"] for r in recs])),
                "mean_mb_mu_p50": float(np.mean([r["mb_mu_p50"] for r in recs])),
                "mean_mb_mu_mean": float(np.mean([r["mb_mu_mean"] for r in recs])),
                "max_mb_mu_max": float(np.max([r["mb_mu_max"] for r in recs])),
                "final_cum_nsr": recs[-1]["cum_nsr"],
                "final_cum_eff_bits": recs[-1]["cum_eff_bits"],
                "mean_slow_norm": float(np.mean([r["slow_norm"] for r in recs])),
                "mean_slow_cosine_sim": float(np.mean([r["slow_cosine_sim"] for r in recs])),
                "final_slow_recovery": recs[-1]["slow_recovery"],
                "final_slow_nsr": recs[-1]["slow_nsr"],
                "final_slow_eff_bits": recs[-1]["slow_eff_bits"],
                "mean_w_abs_max": float(np.mean([r["w_abs_max"] for r in recs])),
                "max_w_abs_max": float(np.max([r["w_abs_max"] for r in recs])),
                "final_w_abs_max": recs[-1]["w_abs_max"],
                "mean_w_utilization": float(np.mean([r["w_utilization"] for r in recs])),
                "max_w_clipped_frac": float(np.max([r["w_clipped_frac"] for r in recs])),
                "mean_w_std": float(np.mean([r["w_std"] for r in recs])),
                # TTv1 fast tile weight distribution
                "max_fast_w_abs_max": float(np.max([r.get("fast_w_abs_max", 0) for r in recs])),
                "mean_fast_w_abs_p99": float(np.mean([r.get("fast_w_abs_p99", 0) for r in recs])),
                "mean_fast_w_abs_p50": float(np.mean([r.get("fast_w_abs_p50", 0) for r in recs])),
                "mean_fast_w_std": float(np.mean([r.get("fast_w_std", 0) for r in recs])),
                # UM-aware pulse probability
                "mean_um_bl_eff": float(np.mean([r.get("um_bl_eff", 0) for r in recs])),
                "mean_um_gamma": float(np.mean([r.get("um_gamma", 0) for r in recs])),
                "mean_um_p_max": float(np.mean([r.get("um_p_max", 0) for r in recs])),
                "mean_um_q_max": float(np.mean([r.get("um_q_max", 0) for r in recs])),
                "frac_steps_p_clipped": float(np.mean([r.get("um_frac_mb_p_clip", 0) for r in recs])),
                "frac_steps_q_clipped": float(np.mean([r.get("um_frac_mb_q_clip", 0) for r in recs])),
                "mean_slot_prob_p50": float(np.mean([r.get("slot_prob_p50", 0) for r in recs])),
                "mean_slot_prob_p90": float(np.mean([r.get("slot_prob_p90", 0) for r in recs])),
                "mean_um_frac_mu_per_call_lt_1": float(np.mean([r.get("um_frac_mu_per_call_lt_1", 0) for r in recs])),
            }

        summary_path = os.path.join(output_dir, "update_diagnostics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Diagnostics saved: {csv_path} ({len(self.records)} records)")

        # Save mu distribution histograms for ECDF plotting
        if self.mu_histograms:
            mu_path = os.path.join(output_dir, "mu_distribution.json")
            mu_data = {
                "bin_edges": self._mu_bin_edges.tolist(),
                "histograms": self.mu_histograms,
            }
            with open(mu_path, "w") as f:
                json.dump(mu_data, f)
            print(f"Mu distribution saved: {mu_path} "
                  f"({len(self.mu_histograms)} tile x step entries)")

        # Save per-element p_i, q_j distribution histograms
        if self.pi_histograms:
            pq_path = os.path.join(output_dir, "pq_distribution.json")
            pq_data = {
                "bin_edges": self._pq_bin_edges.tolist(),
                "pi_histograms": self.pi_histograms,
                "qj_histograms": self.qj_histograms,
            }
            with open(pq_path, "w") as f:
                json.dump(pq_data, f)
            print(f"p_i/q_j distribution saved: {pq_path} "
                  f"({len(self.pi_histograms)} tile x step entries)")

        return csv_path, summary_path


def _um_defaults():
    """Default values for UM-aware fields when no data is available."""
    return {
        "um_bl_eff": 0.0, "um_bl_call_max": 0, "um_bl_saturated": 0,
        "um_gamma": 0.0, "um_A": 0.0, "um_B": 0.0,
        "um_p_max": 0.0, "um_q_max": 0.0,
        "um_frac_mb_p_clip": 0.0, "um_frac_mb_q_clip": 0.0,
        "um_x_abs_max": 0.0, "um_d_abs_max": 0.0,
        "slot_prob_mean": 0.0, "slot_prob_p50": 0.0,
        "slot_prob_p90": 0.0, "slot_prob_p99": 0.0,
        "slot_prob_max": 0.0,
        "um_frac_mu_per_call_lt_1": 0.0,
    }


def _get_current_analog_lr(optimizer):
    """Get the current learning rate for analog (AnalogContext) param groups."""
    for pg in optimizer.param_groups:
        if any(isinstance(p, AnalogContext) for p in pg["params"]):
            return float(pg["lr"])
    return 0.0
