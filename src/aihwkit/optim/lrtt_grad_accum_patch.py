# -*- coding: utf-8 -*-

# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Gradient accumulation for LR-TT analog tiles.

aihwkit's AnalogOptimizer.step() concatenates the inputs and errors stored
for a tile since the last step and applies a single tile.update(), so the
update processes tensors of the full accumulated batch. Importing this module
sets three methods so that the analog update of an LR-TT layer is applied per
gradient-accumulation group:

- AnalogOptimizerMixin._pad_and_cat (added): entries with different sequence
  lengths are zero-padded before concatenation.
- LRTTController._ab_weight_update_lora (replaced): when a snapshot of A and B
  is set (_snapshot_ab), the projections B@x and A^T@d use the snapshot A0/B0.
- AnalogOptimizerMixin.step (replaced): a single stored entry is applied as in
  aihwkit. With several entries, the stored entries of an LR-TT tile are grouped
  by gradient-accumulation step; entries of the same forward pass (a
  weight-shared layer applied at several depths) are concatenated within a
  group. Without forward injection, each group is passed to
  controller.ab_weight_update() (bypassing the _update_handled guard of
  LRTTSimulatorTile.update()) with projections from the A0/B0 snapshot taken
  at the start of the step; the auto-scale EMA, the dynamic transfer period
  and the transfer counter advance once per optimizer step, and the transfer
  condition is checked once after all groups. With forward injection, the
  A and B updates of each group go through the forward-injection hooks, with
  the EMA computed once over all groups. Other analog tiles are updated once
  per stored entry.

Usage:
    from aihwkit.optim import AnalogSGD, AnalogAdam
    import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401  (applies patches on import)

    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer._grad_accum_steps = GRAD_ACCUM  # forward passes per optimizer step
"""

import math as _math
import torch as _torch
from torch import cat as _cat
from torch.nn.functional import pad as _F_pad
from torch.autograd import no_grad as _no_grad

from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin
from aihwkit.optim.context import AnalogContext as _AnalogContext
from aihwkit.simulator.tiles.lrtt_controller import LRTTController


# Patch 0: _pad_and_cat for dynamic-padding grad accumulation
def _pad_and_cat(tensors, axis):
    """Cat tensors, zero-padding non-cat dimensions if shapes differ.

    When using dynamic padding with gradient accumulation, accumulated
    inputs may have different sequence lengths. Zero-padding before cat
    is safe because padded positions contribute zero to the update.
    """
    if len(tensors) <= 1:
        return _cat(tensors, axis=axis)

    ndim = tensors[0].dim()
    # Find max size for each dimension
    max_sizes = list(tensors[0].shape)
    needs_pad = False
    for t in tensors[1:]:
        for d in range(ndim):
            if d != (axis % ndim) and t.shape[d] != max_sizes[d]:
                needs_pad = True
            if t.shape[d] > max_sizes[d]:
                max_sizes[d] = t.shape[d]

    if not needs_pad:
        return _cat(tensors, axis=axis)

    cat_dim = axis % ndim
    padded = []
    for t in tensors:
        # _F_pad format: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        pad_sizes = []
        for d in range(ndim - 1, -1, -1):
            if d == cat_dim:
                pad_sizes.extend([0, 0])
            else:
                pad_sizes.extend([0, max_sizes[d] - t.shape[d]])
        padded.append(_F_pad(t, pad_sizes) if any(pad_sizes) else t)
    return _cat(padded, axis=axis)


AnalogOptimizerMixin._pad_and_cat = staticmethod(_pad_and_cat)


# Patch 1: _ab_weight_update_lora with A/B snapshot
_orig_ab_update = LRTTController._ab_weight_update_lora


def _ab_update_with_snapshot(self, x, d, lr, in_trans=False, out_trans=False):
    """LoRA update that uses snapshot A0/B0 for projections when available."""
    if not hasattr(self, '_snapshot_ab'):
        return _orig_ab_update(self, x, d, lr, in_trans, out_trans)

    # Snapshot mode
    if in_trans:
        x = x.t()
    if out_trans:
        d = d.t()

    A0, B0 = self._snapshot_ab
    with _torch.no_grad():
        # Use snapshot weights for projection consistency across micro-batches.
        # For non-perfect tiles, swap weights to include IO noise/quantization.
        _fwd_perfect = getattr(self.tile_b.rpu_config.forward, 'is_perfect', True)
        _bwd_perfect = getattr(self.tile_a.rpu_config.backward, 'is_perfect', True)
        if _fwd_perfect and _bwd_perfect:
            XB = x @ B0.t()
            DA = d @ A0
        else:
            # Save current weights, set snapshot, forward/backward, restore
            cur_B = self.tile_b.get_weights()[0].clone()
            cur_A = self.tile_a.get_weights()[0].clone()
            self.tile_b.set_weights(B0)
            self.tile_a.set_weights(A0)
            XB = self.tile_b.forward(x)
            DA = self.tile_a.backward(d)
            self.tile_b.set_weights(cur_B)
            self.tile_a.set_weights(cur_A)

    # lr_eff_a, lr_eff_b: use pre-computed values from step() if available,
    # otherwise compute them here
    m_batch = x.shape[0]
    if not hasattr(self, '_snapshot_lr_eff'):
        # No learning rates pre-computed by step(): compute them here
        self._last_lr_sgd = lr
        if self._is_hardware_mode():
            self.last_lr_eff_a = self.last_lr_eff_b = 1.0
        elif self.auto_scale_mode == "none":
            self.last_lr_eff_a = self.last_lr_eff_b = self.fast_lr
        else:
            self._update_autoscale_ema(x, d, XB, DA, m_batch)

    lr_eff_a = self.last_lr_eff_a
    lr_eff_b = self.last_lr_eff_b

    # Optional logging of the update inputs (log_ab_scaling)
    self._ab_update_step += 1
    if self.log_ab_scaling and (self._ab_update_step % self.log_ab_scaling_every == 1):
        print(f"  [AB-Scaling Step {self._ab_update_step}] "
              f"A: x_max={XB.abs().max().item():.4f}, d_max={d.abs().max().item():.4f} | "
              f"B: x_max={x.abs().max().item():.4f}, d_max={DA.abs().max().item():.4f} | "
              f"lr_eff_a={lr_eff_a:.6f}, lr_eff_b={lr_eff_b:.6f}")

    # dA = -lr_eff_a * D^T @ XB
    lr_a_old = self.tile_a.get_learning_rate()
    self.tile_a.set_learning_rate(lr_eff_a)
    if hasattr(self.tile_a, '_orig_update'):
        self.tile_a._orig_update(XB, d)
    else:
        self.tile_a.update(XB, d)
    self.tile_a.set_learning_rate(lr_a_old)
    self.num_a_updates += 1

    # dB = -lr_eff_b * DA^T @ X
    lr_b_old = self.tile_b.get_learning_rate()
    self.tile_b.set_learning_rate(lr_eff_b)
    if hasattr(self.tile_b, '_orig_update'):
        self.tile_b._orig_update(x, DA)
    else:
        self.tile_b.update(x, DA)
    self.tile_b.set_learning_rate(lr_b_old)
    self.num_b_updates += 1

    self._update_dynamic_te(lr)
    self._last_m_batch = m_batch
    self.transfer_counter += m_batch


LRTTController._ab_weight_update_lora = _ab_update_with_snapshot


# Patch 2: step() with per-group tile update
@_no_grad()
def _step_mem_opt(self, closure=None, **kwargs):
    # Digital parameter update; skip AnalogOptimizerMixin in MRO
    # so the real optimizer (SGD, Adam, AdamW, etc.) step() is called.
    ret = super(AnalogOptimizerMixin, self).step(closure, **kwargs)

    # How many grad_accum forward passes were done before this step().
    # Must be set by the training script: optimizer._grad_accum_steps = K
    # Default 1 = no grad_accum (all entries from weight sharing).
    grad_accum_steps = getattr(self, '_grad_accum_steps', 1)

    # Analog parameter update
    for group in self.param_groups:
        learning_rate = group.get("lr")
        for param in group["params"]:
            if not isinstance(param, _AnalogContext):
                continue
            analog_ctx = param
            analog_tile = analog_ctx.analog_tile
            if analog_ctx.use_torch_update or not analog_ctx.has_gradient():
                continue
            if learning_rate == 0.0:
                analog_ctx.reset()
                continue
            if learning_rate is not None:
                analog_tile.set_learning_rate(learning_rate)

            runtime = analog_tile.get_runtime()
            n_micro = len(analog_ctx.analog_input)

            if n_micro <= 1:
                # Single entry: same as aihwkit's step()
                if analog_ctx.use_indexed:
                    for x_i, d_i in zip(
                        analog_ctx.analog_input, analog_ctx.analog_grad_output
                    ):
                        analog_tile.update_indexed(
                            x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                            d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                        )
                else:
                    x_input = self._pad_and_cat(
                        analog_ctx.analog_input,
                        axis=-1 if analog_tile.in_trans else 0,
                    )
                    d_input = self._pad_and_cat(
                        analog_ctx.analog_grad_output,
                        axis=-1 if analog_tile.out_trans else 0,
                    )
                    analog_tile.update(
                        x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                        d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                    )
            else:
                # Multiple entries: weight-sharing and/or grad_accum
                # Detect LRTT sub-tiles via back-reference set in
                # LRTTSimulatorTile._hook_tile_updates()
                controller = getattr(analog_tile, '_lrtt_controller', None)
                is_lrtt = controller is not None
                tile_name = getattr(analog_tile, '_lrtt_tile_name', None)

                if is_lrtt and not controller.forward_inject_enabled:
                    # LRTT NFI mode
                    # In NFI mode, only tile_c's update triggers
                    # ab_weight_update; tile_a/tile_b hooks are no-ops.
                    if tile_name != 'tile_c':
                        # tile_a/tile_b: skip (no-op in NFI mode)
                        analog_ctx.reset()
                        continue

                    # tile_c: group the entries and call the controller directly
                    # (bypassing the _update_handled guard of the tile)
                    #
                    # n_micro = entries_per_fwd * grad_accum_steps
                    #   entries_per_fwd: uses of the layer per forward pass
                    #                    (1 without weight sharing)
                    #   grad_accum_steps: number of forward passes before step()
                    #
                    # Within each grad_accum group, concatenate weight-sharing
                    # entries (same forward pass). Process groups separately
                    # for memory savings.
                    entries_per_fwd = n_micro // grad_accum_steps

                    # Snapshot A0, B0 for consistent projections
                    _dev = analog_tile.device
                    A0 = controller.tile_a.get_weights()[0].clone().to(_dev)
                    B0 = controller.tile_b.get_weights()[0].clone().to(_dev)
                    controller._snapshot_ab = (A0, B0)

                    # Save state that _update_dynamic_te / transfer_counter
                    # modify inside ab_weight_update. Restore and
                    # recompute once after the loop so that a single optimizer
                    # step counts as exactly 1 logical update.
                    te_step_before = controller.te_step_counter
                    te_before = controller.transfer_every
                    counter_before = controller.transfer_counter

                    lr = analog_tile.get_learning_rate()

                    # Pre-compute lr_eff using full-batch stats
                    # Gather amax across all micro-batches so EMA matches
                    # the grad_accum=1 path (single call with full batch).
                    controller._last_lr_sgd = lr
                    if controller._is_hardware_mode():
                        controller.last_lr_eff_a = controller.last_lr_eff_b = 1.0
                    elif controller.auto_scale_mode == "none":
                        controller.last_lr_eff_a = controller.last_lr_eff_b = controller.fast_lr
                    elif controller.auto_scale_mode in ("shared", "separate"):
                        # Per-sample amax sequences (max over the feature dimension),
                        # as aihwkit's C++ chopped-transfer device calls updateAutoScale
                        # once per outer product (rpu_chopped_transfer_device.cpp).
                        x_amax_seqs = []
                        d_amax_seqs = []
                        xb_amax_seqs = []
                        da_amax_seqs = []
                        for g in range(grad_accum_steps):
                            start = g * entries_per_fwd
                            end = start + entries_per_fwd
                            x_g = self._pad_and_cat(
                                analog_ctx.analog_input[start:end],
                                axis=-1 if analog_tile.in_trans else 0,
                            )
                            d_g = self._pad_and_cat(
                                analog_ctx.analog_grad_output[start:end],
                                axis=-1 if analog_tile.out_trans else 0,
                            )
                            if runtime.offload_input:
                                x_g = x_g.to(_dev)
                            if runtime.offload_gradient:
                                d_g = d_g.to(_dev)
                            # Per-sample amax: max over feature dim only, flatten others
                            x_amax_seqs.append(x_g.abs().amax(dim=-1).reshape(-1))
                            d_amax_seqs.append(d_g.abs().amax(dim=-1).reshape(-1))
                            with _torch.no_grad():
                                # EMA stats: use tile forward/backward for IO fidelity.
                                # Weights haven't been updated yet (before the loop),
                                # so tile weights == snapshot A0/B0.
                                xb_g = controller.tile_b.forward(x_g)
                                da_g = controller.tile_a.backward(d_g)
                            xb_amax_seqs.append(xb_g.abs().amax(dim=-1).reshape(-1))
                            da_amax_seqs.append(da_g.abs().amax(dim=-1).reshape(-1))
                            del x_g, d_g, xb_g, da_g
                        # Concatenate across grad_accum into single per-sample sequence
                        x_amax_seq = _torch.cat(x_amax_seqs)   # shape [total_m_batch]
                        d_amax_seq = _torch.cat(d_amax_seqs)
                        xb_amax_seq = _torch.cat(xb_amax_seqs)
                        da_amax_seq = _torch.cat(da_amax_seqs)
                        # EMA over the concatenated sequence in closed form
                        # (LRTTController._update_ema_exact), with cold start.
                        controller._lazy_init_ema(A0.device)
                        _ema_fn = controller._update_ema_exact
                        if controller.auto_scale_mode == "shared":
                            controller.m_x = _ema_fn(controller.m_x, x_amax_seq, controller.auto_momentum)
                            controller.m_d = _ema_fn(controller.m_d, d_amax_seq, controller.auto_momentum)
                            denom = controller.m_x * controller.m_d
                            fast_lr_t = x_amax_seq.new_tensor(controller.fast_lr)
                            lr_eff = _torch.where(denom > 0, fast_lr_t / denom, fast_lr_t).item()
                            controller.last_lr_eff_a = lr_eff
                            controller.last_lr_eff_b = lr_eff
                        elif controller.auto_scale_mode == "separate":
                            controller.m_x = _ema_fn(controller.m_x, x_amax_seq, controller.auto_momentum)
                            controller.m_d = _ema_fn(controller.m_d, d_amax_seq, controller.auto_momentum)
                            controller.m_xb = _ema_fn(controller.m_xb, xb_amax_seq, controller.auto_momentum)
                            controller.m_da = _ema_fn(controller.m_da, da_amax_seq, controller.auto_momentum)
                            denom_a = controller.m_xb * controller.m_d
                            fa = x_amax_seq.new_tensor(controller.fast_lr * controller.gran_a)
                            controller.last_lr_eff_a = _torch.where(denom_a > 0, fa / denom_a, fa).item()
                            denom_b = controller.m_x * controller.m_da
                            fb = x_amax_seq.new_tensor(controller.fast_lr * controller.gran_b)
                            controller.last_lr_eff_b = _torch.where(denom_b > 0, fb / denom_b, fb).item()

                    # Signal snapshot path to skip per-call EMA update
                    controller._snapshot_lr_eff = True

                    for g in range(grad_accum_steps):
                        start = g * entries_per_fwd
                        end = start + entries_per_fwd

                        # Concatenate weight-sharing entries within this group
                        x_group = self._pad_and_cat(
                            analog_ctx.analog_input[start:end],
                            axis=-1 if analog_tile.in_trans else 0,
                        )
                        d_group = self._pad_and_cat(
                            analog_ctx.analog_grad_output[start:end],
                            axis=-1 if analog_tile.out_trans else 0,
                        )

                        if runtime.offload_input:
                            x_group = x_group.to(analog_tile.device)
                        if runtime.offload_gradient:
                            d_group = d_group.to(analog_tile.device)

                        # Direct controller call; bypasses _update_handled
                        # and should_transfer() inside LRTTSimulatorTile.update()
                        controller.ab_weight_update(
                            x=x_group, d=d_group, lr=lr,
                            in_trans=False, out_trans=False,
                        )

                    del controller._snapshot_lr_eff

                    # Counters: ab_weight_update called _update_dynamic_te once per
                    # group (advancing te_step_counter and possibly transfer_every);
                    # restore both and advance once for this optimizer step.
                    controller.te_step_counter = te_step_before
                    controller.transfer_every = te_before
                    controller._update_dynamic_te(lr)  # advances by +1

                    # transfer_counter: each call added the rows of its group
                    # (entries_per_fwd uses of the layer per forward pass); count
                    # each forward pass once. _last_m_batch is set to this increment
                    # for the modulo boundary-crossing check in should_transfer().
                    inflated = controller.transfer_counter - counter_before
                    corrected_increment = inflated // entries_per_fwd
                    controller.transfer_counter = counter_before + corrected_increment
                    controller._last_m_batch = corrected_increment

                    # Check transfer once after all groups processed
                    if controller.should_transfer():
                        controller.ab_weight_transfer()

                    del controller._snapshot_ab

                elif is_lrtt and controller.forward_inject_enabled:
                    # LRTT FI mode
                    # FI hooks pair tile_a + tile_b updates: tile_a.update()
                    # caches data, tile_b.update() triggers both _orig_update.
                    # Must process tile_a and tile_b together per group.
                    entries_per_fwd = n_micro // grad_accum_steps

                    if tile_name == 'tile_c':
                        # tile_c: no gradient update (C changes only by transfers);
                        # advance the transfer counter once per optimizer step.
                        lr = analog_tile.get_learning_rate()
                        controller._last_lr_sgd = lr  # sync for effective_alpha
                        m_batch = analog_ctx.analog_input[0].shape[0]
                        controller._update_dynamic_te(lr)
                        corrected_increment = m_batch * grad_accum_steps
                        controller._last_m_batch = corrected_increment
                        controller.transfer_counter += corrected_increment
                        if controller.should_transfer():
                            controller.ab_weight_transfer()
                    elif tile_name == 'tile_a':
                        # Defer: save for paired processing with tile_b
                        controller._fi_ga_a_tile = analog_tile
                        controller._fi_ga_a_ctx = analog_ctx
                        controller._fi_ga_a_runtime = runtime
                        continue  # skip analog_ctx.reset(); tile_b will handle it
                    elif tile_name == 'tile_b':
                        # Process tile_a and tile_b together, group by group.
                        # This ensures the FI hook pairs matching groups.
                        a_tile = controller._fi_ga_a_tile
                        a_ctx = controller._fi_ga_a_ctx
                        a_runtime = controller._fi_ga_a_runtime

                        # First pass: collect the per-sample amax sequences of all
                        # groups and update the EMA once over all samples (N = total
                        # number of samples), as for a single batch.
                        #
                        # The FI hook uses d_rescaled = d_input / alpha for the EMA; the
                        # same is done here. Set _last_lr_sgd first so effective_alpha is
                        # current (matches what the FI hook would see).
                        controller._last_lr_sgd = learning_rate
                        _fi_alpha = controller.effective_alpha
                        groups_data = []  # cache materialized x_a, d_a, x_b, d_b per group
                        if controller.auto_scale_mode in ("shared", "separate"):
                            x_a_amax_seqs = []
                            d_a_amax_seqs = []   # = d_a / alpha (matches FI hook rescaling)
                            x_b_amax_seqs = []
                            d_b_amax_seqs = []   # = d_b / alpha (matches FI hook rescaling)

                        for g in range(grad_accum_steps):
                            start = g * entries_per_fwd
                            end = start + entries_per_fwd

                            # tile_a group (x_a = XB, d_a = d/alpha after FI rescaling)
                            x_a = self._pad_and_cat(
                                a_ctx.analog_input[start:end],
                                axis=-1 if a_tile.in_trans else 0,
                            )
                            d_a = self._pad_and_cat(
                                a_ctx.analog_grad_output[start:end],
                                axis=-1 if a_tile.out_trans else 0,
                            )
                            if a_runtime.offload_input:
                                x_a = x_a.to(a_tile.device)
                            if a_runtime.offload_gradient:
                                d_a = d_a.to(a_tile.device)

                            # tile_b group (x_b = x, d_b = DA after FI rescaling)
                            x_b = self._pad_and_cat(
                                analog_ctx.analog_input[start:end],
                                axis=-1 if analog_tile.in_trans else 0,
                            )
                            d_b = self._pad_and_cat(
                                analog_ctx.analog_grad_output[start:end],
                                axis=-1 if analog_tile.out_trans else 0,
                            )
                            if runtime.offload_input:
                                x_b = x_b.to(analog_tile.device)
                            if runtime.offload_gradient:
                                d_b = d_b.to(analog_tile.device)

                            groups_data.append((x_a, d_a, x_b, d_b))

                            if controller.auto_scale_mode in ("shared", "separate"):
                                # Per-sample amax sequences (FI hook expects:
                                # x=fi_b_x=x_b, d=fi_a_d=d_a/alpha (rescaled), XB=fi_a_x=x_a, DA=fi_b_d=d_b/alpha)
                                x_b_amax_seqs.append(x_b.abs().amax(dim=-1).reshape(-1))
                                # d_a and d_b come from analog_grad_output (includes alpha from forward).
                                # FI hook divides by alpha before EMA; mirror that here.
                                d_a_amax_seqs.append((d_a / _fi_alpha).abs().amax(dim=-1).reshape(-1))
                                x_a_amax_seqs.append(x_a.abs().amax(dim=-1).reshape(-1))
                                d_b_amax_seqs.append((d_b / _fi_alpha).abs().amax(dim=-1).reshape(-1))

                        # Compute EMA once externally with concatenated sequences
                        if controller.auto_scale_mode in ("shared", "separate"):
                            x_seq  = _torch.cat(x_b_amax_seqs)   # x in EMA signature
                            d_seq  = _torch.cat(d_a_amax_seqs)   # d
                            xb_seq = _torch.cat(x_a_amax_seqs)   # XB
                            da_seq = _torch.cat(d_b_amax_seqs)   # DA
                            controller._lazy_init_ema(x_seq.device)
                            _ema_fn = controller._update_ema_exact
                            if controller.auto_scale_mode == "shared":
                                controller.m_x = _ema_fn(controller.m_x, x_seq, controller.auto_momentum)
                                controller.m_d = _ema_fn(controller.m_d, d_seq, controller.auto_momentum)
                                denom = controller.m_x * controller.m_d
                                fast_lr_t = x_seq.new_tensor(controller.fast_lr)
                                lr_eff = _torch.where(denom > 0, fast_lr_t / denom, fast_lr_t).item()
                                controller.last_lr_eff_a = lr_eff
                                controller.last_lr_eff_b = lr_eff
                            elif controller.auto_scale_mode == "separate":
                                controller.m_x  = _ema_fn(controller.m_x,  x_seq,  controller.auto_momentum)
                                controller.m_d  = _ema_fn(controller.m_d,  d_seq,  controller.auto_momentum)
                                controller.m_xb = _ema_fn(controller.m_xb, xb_seq, controller.auto_momentum)
                                controller.m_da = _ema_fn(controller.m_da, da_seq, controller.auto_momentum)
                                denom_a = controller.m_xb * controller.m_d
                                fa = x_seq.new_tensor(controller.fast_lr * controller.gran_a)
                                controller.last_lr_eff_a = _torch.where(denom_a > 0, fa / denom_a, fa).item()
                                denom_b = controller.m_x * controller.m_da
                                fb = x_seq.new_tensor(controller.fast_lr * controller.gran_b)
                                controller.last_lr_eff_b = _torch.where(denom_b > 0, fb / denom_b, fb).item()
                            # Signal FI hook to use pre-computed lr_eff (skip per-chunk EMA)
                            controller._fi_skip_ema = True

                        # Second pass: actual tile updates (FI hook reads pre-computed lr_eff)
                        for x_a, d_a, x_b, d_b in groups_data:
                            # Call in order: tile_a then tile_b
                            # FI hook caches _fi_a_*, then _fi_b_* triggers _orig_update
                            a_tile.update(x_a, d_a)
                            analog_tile.update(x_b, d_b)

                        # Clean up
                        if hasattr(controller, '_fi_skip_ema'):
                            del controller._fi_skip_ema
                        a_ctx.reset()
                        del controller._fi_ga_a_tile, controller._fi_ga_a_ctx, controller._fi_ga_a_runtime

                else:
                    # Other tiles: one update per stored entry
                    if analog_ctx.use_indexed:
                        for x_i, d_i in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                            )
                    else:
                        for x_i, d_i in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update(
                                x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                            )

            analog_ctx.reset()

    # Post-update step (diffuse, decay, etc.)
    for group in self.param_groups:
        for param in group["params"]:
            if isinstance(param, _AnalogContext):
                param.analog_tile.post_update_step()
    return ret


AnalogOptimizerMixin.step = _step_mem_opt
