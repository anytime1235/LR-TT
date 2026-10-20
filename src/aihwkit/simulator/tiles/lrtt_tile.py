# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""LR-TT simulator tile.

Integrates LRTTController with the aihwkit tile system: three aihwkit tiles
A, B and C, whose updates and transfers are orchestrated in Python.
"""

from typing import Optional, Tuple, Any, Dict
import math
import torch
from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.tiles.floating_point import FloatingPointTile
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.configs.devices import FloatingPointDevice
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig

from aihwkit.exceptions import ConfigError, TileError


class LRTTSimulatorTile(SimulatorTile, Module):
    """LR-TT simulator tile.

    Uses three aihwkit tiles orchestrated by LRTTController:

    - tile_a: A, [d_size, rank]
    - tile_b: B, [rank, x_size]
    - tile_c: core weights C, [d_size, x_size]

    The forward pass uses C (y = C@x), or y = C@x + alpha*A@(B@x) with forward
    injection. A and B are updated by pulsed updates from the projections B@x
    and A^T@d, and every transfer_every steps the product A @ B is transferred
    into C by pulsed updates (one-hot, direct) or exactly ('set').
    """

    supports_indexed: bool = False

    def __init__(
        self,
        d_size: int,  # out_features from AnalogLinear
        x_size: int,  # in_features from AnalogLinear
        rpu_config: UnitCellRPUConfig,
        bias: bool = False,  # digital bias of the C tile
        dtype: Optional[
            RPUDataType
        ] = None,  # float32 if None
        **kwargs,  # further keyword arguments are ignored
    ):
        """Initialize LRTT simulator tile.

        Args:
            d_size: Output size (out_features from AnalogLinear)
            x_size: Input size (in_features from AnalogLinear)
            rpu_config: Must contain a PythonLRTTDevice
            dtype: Data type for tiles
            bias: Whether the C tile has a digital bias
        """
        Module.__init__(self)

        self.x_size = x_size
        self.d_size = d_size
        # Default data type: float32
        if dtype is None:
            from aihwkit.simulator.parameters.enums import RPUDataType

            dtype = RPUDataType.FLOAT  # Default to float32
        self.dtype = dtype
        self.bias = bias  # Passed to tile_c for digital_bias support

        # Validate configuration - check for PythonLRTTDevice
        from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

        if not isinstance(getattr(rpu_config, "device", None), PythonLRTTDevice):
            raise ConfigError(
                "LRTTSimulatorTile requires a PythonLRTTDevice configuration"
            )

        self.lrtt_config = rpu_config.device
        self.rank = self.lrtt_config.rank

        if self.rank <= 0 or self.rank > min(d_size, x_size):
            raise ConfigError(
                f"Invalid rank {self.rank} for dimensions {d_size}x{x_size}"
            )

        # Extract LRTT parameters
        self.transfer_lr = getattr(self.lrtt_config, "transfer_lr", 1.0)
        self.transfer_every = getattr(self.lrtt_config, "transfer_every", 32)
        self.units_in_mbatch = getattr(self.lrtt_config, "units_in_mbatch", False)
        self.lora_alpha = getattr(self.lrtt_config, "lora_alpha", 1.0)
        self.reinit_gain = getattr(self.lrtt_config, "reinit_gain", 1.0)
        self.correct_gradient_magnitudes = getattr(
            self.lrtt_config, "correct_gradient_magnitudes", False
        )
        # Note: forward_inject flag is managed by the controller only
        self.rank_chunk = getattr(self.lrtt_config, "rank_chunk", None)

        # Create individual tiles using unit cell devices
        unit_devices = self.lrtt_config.unit_cell_devices
        if len(unit_devices) < 3:
            # Replicate first device if not enough specified
            while len(unit_devices) < 3:
                unit_devices.append(unit_devices[0])

        # Helper function to select tile class based on device type
        def get_tile_class(device):
            if isinstance(device, FloatingPointDevice):
                return FloatingPointTile
            else:
                return AnalogTile

        # Helper function to create UpdateParameters with separate A/B scaling
        def create_update_params(base_update, tile_type):
            """Create UpdateParameters with optional tile-specific scaling.

            Args:
                base_update: Base UpdateParameters from rpu_config
                tile_type: 'a', 'b', or 'c'

            Returns:
                UpdateParameters with tile-specific scaling if configured
            """
            from copy import deepcopy
            from aihwkit.simulator.configs import UpdateParameters

            from aihwkit.simulator.parameters.enums import PulseType

            # Get tile-specific scaling overrides from lrtt_config
            a_x = getattr(self.lrtt_config, "a_x_scaling", None)
            a_d = getattr(self.lrtt_config, "a_d_scaling", None)
            b_x = getattr(self.lrtt_config, "b_x_scaling", None)
            b_d = getattr(self.lrtt_config, "b_d_scaling", None)
            a_bl = getattr(self.lrtt_config, "a_desired_bl", None)
            b_bl = getattr(self.lrtt_config, "b_desired_bl", None)
            c_bl = getattr(self.lrtt_config, "c_desired_bl", None)

            # AB pulse type override
            ab_pulse_type_str = getattr(self.lrtt_config, 'ab_pulse_type', 'default')
            _AB_PULSE_MAP = {
                'none': PulseType.NONE,
                'none_with_device': PulseType.NONE_WITH_DEVICE,
                'stochastic_compressed': PulseType.STOCHASTIC_COMPRESSED,
                'stochastic': PulseType.STOCHASTIC,
                'mean_count': PulseType.MEAN_COUNT,
                'deterministic_implicit': PulseType.DETERMINISTIC_IMPLICIT,
            }
            ab_needs_pulse_override = (
                tile_type in ("a", "b") and ab_pulse_type_str != "default"
            )

            # Check if any tile-specific config is set
            transfer_method = getattr(self.lrtt_config, 'transfer_method', 'direct')
            c_needs_nwd = (tile_type == "c" and transfer_method == "set")
            has_tile_specific = any(x is not None for x in [a_x, a_d, b_x, b_d, a_bl, b_bl, c_bl])

            if not has_tile_specific and not c_needs_nwd and not ab_needs_pulse_override:
                # No tile-specific config, use base update params
                return base_update

            # Create a copy and apply tile-specific overrides
            # Copy all fields explicitly
            update_copy = UpdateParameters(
                desired_bl=base_update.desired_bl,
                pulse_type=base_update.pulse_type,
                use_manual_scaling=base_update.use_manual_scaling,
                manual_x_scaling=base_update.manual_x_scaling,
                manual_d_scaling=base_update.manual_d_scaling,
                update_bl_management=base_update.update_bl_management,
                update_management=base_update.update_management,
            )

            # Apply tile-specific overrides
            if tile_type == "a":
                if a_x is not None:
                    update_copy.manual_x_scaling = a_x
                if a_d is not None:
                    update_copy.manual_d_scaling = a_d
                if a_bl is not None:
                    update_copy.desired_bl = a_bl
                if ab_needs_pulse_override:
                    update_copy.pulse_type = _AB_PULSE_MAP[ab_pulse_type_str]
            elif tile_type == "b":
                if b_x is not None:
                    update_copy.manual_x_scaling = b_x
                if b_d is not None:
                    update_copy.manual_d_scaling = b_d
                if b_bl is not None:
                    update_copy.desired_bl = b_bl
                if ab_needs_pulse_override:
                    update_copy.pulse_type = _AB_PULSE_MAP[ab_pulse_type_str]
            elif tile_type == "c":
                # C tile: apply separate BL for transfer
                if c_bl is not None:
                    update_copy.desired_bl = c_bl
                # The transfer writes C with dynamic scaling (no manual scales);
                # the manual scale fields still need valid values
                update_copy.use_manual_scaling = False
                if update_copy.manual_x_scaling is None:
                    update_copy.manual_x_scaling = 1.0
                if update_copy.manual_d_scaling is None:
                    update_copy.manual_d_scaling = 1.0

                # For "set" transfer: use NoneWithDevice for deterministic FP
                # update with weight clipping (must be set at construction time)
                if c_needs_nwd:
                    from aihwkit.simulator.parameters.enums import PulseType
                    update_copy.pulse_type = PulseType.NONE_WITH_DEVICE

            return update_copy

        # Tile A/B creation: use mapping_ab from device config
        from aihwkit.simulator.parameters.mapping import MappingParameter
        tile_class_a = get_tile_class(unit_devices[0])
        update_a = create_update_params(rpu_config.update, "a")
        tile_class_b = get_tile_class(unit_devices[1])
        update_b = create_update_params(rpu_config.update, "b")

        mapping_ab = getattr(self.lrtt_config, 'mapping_ab', MappingParameter())
        # Per-tile overrides for A/B mapping (None -> fall back to mapping_ab)
        mapping_a_override = getattr(self.lrtt_config, 'mapping_a', None)
        mapping_b_override = getattr(self.lrtt_config, 'mapping_b', None)
        mapping_a = mapping_a_override if mapping_a_override is not None else mapping_ab
        mapping_b = mapping_b_override if mapping_b_override is not None else mapping_ab

        # A/B tile IO: optionally make the low-rank A and B reads fully ideal
        # (digital-adapter model: no out_noise / ADC / DAC / bound on A & B
        # forward and backward). The visible C tile stays non-ideal.
        from copy import deepcopy
        ab_io_perfect = getattr(self.lrtt_config, 'ab_io_perfect', False)

        forward_a = rpu_config.forward
        backward_a = rpu_config.backward
        forward_b = rpu_config.forward
        backward_b = rpu_config.backward
        if ab_io_perfect:
            forward_a = deepcopy(rpu_config.forward);   forward_a.is_perfect = True
            backward_a = deepcopy(rpu_config.backward); backward_a.is_perfect = True
            forward_b = deepcopy(rpu_config.forward);   forward_b.is_perfect = True
            backward_b = deepcopy(rpu_config.backward); backward_b.is_perfect = True

        rpu_config_a = SingleRPUConfig(
            device=unit_devices[0],
            forward=forward_a,
            backward=backward_a,
            update=update_a,
            tile_class=tile_class_a,
            mapping=mapping_a,
        )
        self.tile_a = rpu_config_a.tile_class(d_size, self.rank, rpu_config_a)

        rpu_config_b = SingleRPUConfig(
            device=unit_devices[1],
            forward=forward_b,
            backward=backward_b,
            update=update_b,
            tile_class=tile_class_b,
            mapping=mapping_b,
        )
        self.tile_b = rpu_config_b.tile_class(self.rank, x_size, rpu_config_b)

        # Tile C: visible [d_size, x_size] - use mapping_c from device config
        tile_class_c = get_tile_class(unit_devices[2])
        update_c = create_update_params(rpu_config.update, "c")
        mapping_c = getattr(self.lrtt_config, 'mapping_c', MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        ))
        rpu_config_c = SingleRPUConfig(
            device=unit_devices[2],
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=update_c,
            tile_class=tile_class_c,
            mapping=mapping_c,
        )
        # Pass bias to tile_c for digital_bias support
        # When bias=True, tile_c will have digital_bias=True and create self.bias Parameter
        self.tile_c = rpu_config_c.tile_class(
            d_size, x_size, rpu_config_c, bias=self.bias
        )

        # Freeze/unfreeze bias in tile_c based on config
        # (C tile weights are already untrainable via update hooks)
        # Note: out_scaling trainability is controlled by mapping_c.learn_out_scaling
        _train_bias = getattr(self.lrtt_config, 'train_c_bias', False)
        for name, param in self.tile_c.named_parameters():
            if 'bias' in name:
                param.requires_grad = _train_bias

        # Create LRTT controller with all parameters
        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            d_size=d_size,
            x_size=x_size,
            rank=self.rank,
            transfer_lr=self.transfer_lr,
            transfer_every=self.transfer_every,
            units_in_mbatch=self.units_in_mbatch,
            lora_alpha=self.lora_alpha,
            reinit_gain=self.reinit_gain,
            reinit_mode=getattr(self.lrtt_config, "reinit_mode", "standard"),
            a_init_mode=getattr(
                self.lrtt_config, "a_init_mode", "zero"
            ),  # A matrix initialization mode
            b_init_mode=getattr(
                self.lrtt_config, "b_init_mode", "kaiming"
            ),  # B matrix initialization mode
            correct_gradient_magnitudes=self.correct_gradient_magnitudes,
            fast_lr=getattr(self.lrtt_config, "fast_lr", 1.0),
            scale_transfer_lr=getattr(self.lrtt_config, "scale_transfer_lr", True),
            transfer_fast_lr_ref=getattr(self.lrtt_config, "transfer_fast_lr_ref", "geomean"),
            rank_chunk=self.rank_chunk,
            forward_inject=getattr(self.lrtt_config, "forward_inject", False),
            dynamic_te=getattr(self.lrtt_config, "dynamic_te", False),
            dynamic_te_power=getattr(self.lrtt_config, "dynamic_te_power", 1.0),
            dynamic_te_min=getattr(self.lrtt_config, "dynamic_te_min", None),
            dynamic_te_max=getattr(self.lrtt_config, "dynamic_te_max", None),
            te_warmup_schedule=getattr(self.lrtt_config, "te_warmup_schedule", None),
            te_warmup_steps=getattr(self.lrtt_config, "te_warmup_steps", 0),
            num_reads=getattr(self.lrtt_config, "num_reads", 1),
            multi_read_mode=getattr(self.lrtt_config, "multi_read_mode", "average"),
            update_mode=getattr(self.lrtt_config, "update_mode", "lora"),
            transfer_method=getattr(self.lrtt_config, "transfer_method", "onehot"),
            transfer_rank_schedule=getattr(self.lrtt_config, "transfer_rank_schedule", "all"),
            transfer_ranks_per_step=getattr(self.lrtt_config, "transfer_ranks_per_step", 1),
            fi_continuous_alpha=getattr(self.lrtt_config, "fi_continuous_alpha", False),
        )

        # Apply post-init settings from config._post_init
        kwargs = self.lrtt_config.to_controller_kwargs()
        post_init = kwargs.get("_post_init", {})

        # Transfer mode & calibration
        self.controller.transfer_mode = post_init.get("transfer_mode", "pilot")
        self.controller.transfer_micro_steps = post_init.get("transfer_micro_steps", 1)
        self.controller.transfer_pilot_frac = post_init.get(
            "transfer_pilot_frac", 1.0 / 16.0
        )
        self.controller.sd_quantum = post_init.get("sd_quantum", None)

        # Read noise reduction
        self.controller.read_n_avg = post_init.get("read_n_avg", 1)
        self.controller.differential_read = post_init.get("differential_read", False)

        # AGC settings
        self.controller.agc_enabled = post_init.get("agc_enabled", False)
        self.controller.agc_margin = post_init.get("agc_margin", 0.85)
        self.controller.agc_max_iters = post_init.get("agc_max_iters", 6)

        # Two-amplitude settings
        self.controller.two_amp_enabled = post_init.get("two_amp_enabled", False)
        self.controller.two_amp_ratio = post_init.get("two_amp_ratio", 0.5)

        # Reconstruction parameters (for update_mode='reconstruction')
        self.controller.recon_lambda_a = post_init.get("recon_lambda_a", 1e-3)
        self.controller.recon_lambda_b = post_init.get("recon_lambda_b", 1e-3)
        self.controller.recon_use_scalar_stabilizer = post_init.get(
            "recon_use_scalar_stabilizer", False
        )
        self.controller.recon_use_exact_gram = post_init.get(
            "recon_use_exact_gram", False
        )
        self.controller.recon_exact_gram_every = post_init.get(
            "recon_exact_gram_every", 0
        )
        self.controller.recon_ema_beta = post_init.get("recon_ema_beta", 0.9)
        self.controller.recon_lr_scale = post_init.get("recon_lr_scale", 1.0)
        self.controller.recon_clip_norm = post_init.get("recon_clip_norm", 10.0)
        self.controller.recon_use_clip_norm = post_init.get(
            "recon_use_clip_norm", False
        )

        # Logging of the A/B update inputs
        self.controller.log_ab_scaling = post_init.get("log_ab_scaling", False)
        self.controller.log_ab_scaling_every = post_init.get("log_ab_scaling_every", 10)

        # Auto-scale settings (from post_init)
        self.controller.auto_scale_mode = post_init.get("auto_scale_mode", "none")
        self.controller.auto_momentum = post_init.get("auto_momentum", 0.99)
        # EMA is lazily initialized by controller._lazy_init_ema(x.device) on first update

        # Granularity for separate auto-scale mode
        if self.controller.auto_scale_mode == "separate":
            dw_min_a = getattr(self.lrtt_config.unit_cell_devices[0], "dw_min", 1.0)
            dw_min_b = getattr(self.lrtt_config.unit_cell_devices[1], "dw_min", 1.0)
            desired_bl = rpu_config.update.desired_bl
            self.controller.gran_a = desired_bl * dw_min_a
            self.controller.gran_b = desired_bl * dw_min_b

        # Initialize LRTT weights
        self.controller.reinit()

        # Hook individual tile updates to route through controller
        self._hook_tile_updates()

    def _hook_tile_updates(self) -> None:
        """Hook individual tile update methods to route through controller.

        When the analog optimizer calls update on the individual tiles, the
        hooks route the call through the controller.

        Two modes depending on forward_inject_enabled:

        forward_inject=False:
            All 3 tiles hooked. tile_a/tile_b -> no-op, tile_c triggers
            ab_weight_update(), which computes XB=B@x and DA=A^T@d with tile
            reads and calls tile_a/tile_b._orig_update with the A/B learning
            rates lr_eff_a/lr_eff_b (fast_lr, auto-scaled, or 1 with manual
            scaling).

        forward_inject=True (autograd-driven):
            Autograd populates the tile_a/tile_b AnalogContexts with the
            gradient components of y = y_c + alpha*A@(B@x):
              tile_a: x_input=g=B@x, d_input=alpha*d
              tile_b: x_input=x,     d_input=alpha*A^T@d
            The hooks divide d_input by alpha and update the tiles with the
            same learning rates lr_eff_a/lr_eff_b:
              tile_a: orig_update(g, d)
              tile_b: orig_update(x, A^T@d)
            tile_c -> no-op (C is changed only by transfers).
        """
        # Store original update methods
        self.tile_a._orig_update = self.tile_a.update
        self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update

        # Back-references so the optimizer patch can find the controller
        # from any sub-tile's AnalogContext.analog_tile
        self.tile_a._lrtt_controller = self.controller
        self.tile_b._lrtt_controller = self.controller
        self.tile_c._lrtt_controller = self.controller
        self.tile_a._lrtt_tile_name = 'tile_a'
        self.tile_b._lrtt_tile_name = 'tile_b'
        self.tile_c._lrtt_tile_name = 'tile_c'

        # Whether this batch has already been handled
        self._update_handled = False

        if self.controller.forward_inject_enabled:
            self._hook_tile_updates_fi()
        else:
            self._hook_tile_updates_nfi()

    def _hook_tile_updates_fi(self) -> None:
        """Hook setup for forward_inject=True.

        tile_a/tile_b: rescale d_input (remove alpha), cache values, and defer
                       tile updates until both tiles' hooks have fired. This
                       allows EMA statistics to be updated from autograd-cached
                       values before computing lr_eff_a/b.
        tile_c: no gradient update of C; advances the transfer counter and
                runs the transfer when it is due.
        """
        ctrl = self.controller

        def hooked_ab(tile, tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                alpha = ctrl.effective_alpha
                if alpha == 0.0:
                    # AB contributes nothing to forward (y = y_c + 0*y_ab),
                    # so gradients are zero and no tile update is needed.
                    return None
                d_rescaled = d_input / alpha
                m_batch = x_input.shape[0]

                # Cache this tile's inputs (order-independent)
                if tile_name == "tile_a":
                    ctrl._fi_a_tile = tile
                    ctrl._fi_a_x = x_input       # XB = B@x
                    ctrl._fi_a_d = d_rescaled     # d (alpha removed)
                    ctrl._fi_a_m_batch = m_batch
                else:  # tile_b
                    ctrl._fi_b_tile = tile
                    ctrl._fi_b_x = x_input       # x
                    ctrl._fi_b_d = d_rescaled     # DA = A^T@d (alpha removed)

                # Once both tiles have cached, do EMA -> lr_eff -> update both
                if hasattr(ctrl, '_fi_a_tile') and hasattr(ctrl, '_fi_b_tile'):
                    # Compute lr_eff_a/b (mirroring _ab_weight_update_lora logic)
                    if ctrl._is_hardware_mode():
                        lr_eff_a = lr_eff_b = 1.0
                    elif ctrl.auto_scale_mode == "none":
                        lr_eff_a = lr_eff_b = ctrl.fast_lr
                    elif hasattr(ctrl, '_fi_skip_ema'):
                        # lrtt_grad_accum_patch has updated the EMA once over the
                        # micro-batches of this optimizer step; use its lr_eff.
                        lr_eff_a = ctrl.last_lr_eff_a
                        lr_eff_b = ctrl.last_lr_eff_b
                    else:
                        ctrl._update_autoscale_ema(
                            ctrl._fi_b_x, ctrl._fi_a_d,     # x, d
                            ctrl._fi_a_x, ctrl._fi_b_d,     # XB, DA
                            ctrl._fi_a_m_batch)
                        lr_eff_a = ctrl.last_lr_eff_a
                        lr_eff_b = ctrl.last_lr_eff_b
                    # Store for correct_gradient_magnitudes / transfer LR
                    ctrl.last_lr_eff_a = lr_eff_a
                    ctrl.last_lr_eff_b = lr_eff_b

                    # tile_a update
                    lr_a_old = ctrl._fi_a_tile.get_learning_rate()
                    ctrl._fi_a_tile.set_learning_rate(lr_eff_a)
                    ctrl._fi_a_tile._orig_update(ctrl._fi_a_x, ctrl._fi_a_d)
                    ctrl._fi_a_tile.set_learning_rate(lr_a_old)
                    ctrl.num_a_updates += 1

                    # tile_b update
                    lr_b_old = ctrl._fi_b_tile.get_learning_rate()
                    ctrl._fi_b_tile.set_learning_rate(lr_eff_b)
                    ctrl._fi_b_tile._orig_update(ctrl._fi_b_x, ctrl._fi_b_d)
                    ctrl._fi_b_tile.set_learning_rate(lr_b_old)
                    ctrl.num_b_updates += 1

                    # Clean up cached tensors
                    del ctrl._fi_a_tile, ctrl._fi_a_x, ctrl._fi_a_d, ctrl._fi_a_m_batch
                    del ctrl._fi_b_tile, ctrl._fi_b_x, ctrl._fi_b_d

                return None
            return update_wrapper

        def hooked_c_noop(x_input, d_input, *args, **kwargs):
            # No gradient update of C (C changes only by transfers)
            # Track lr_sgd for transfer LR computation
            ctrl._last_lr_sgd = self.tile_c.get_learning_rate()
            # Handle transfer counter and dynamic TE
            lr = self.tile_c.get_learning_rate()
            ctrl._update_dynamic_te(lr)
            m_batch = x_input.shape[0]
            ctrl._last_m_batch = m_batch
            ctrl.transfer_counter += m_batch
            if ctrl.should_transfer():
                ctrl.ab_weight_transfer()
            return None

        self.tile_a.update = hooked_ab(self.tile_a, "tile_a")
        self.tile_b.update = hooked_ab(self.tile_b, "tile_b")
        self.tile_c.update = hooked_c_noop

    def _hook_tile_updates_nfi(self) -> None:
        """Hook setup for forward_inject=False.

        tile_a/tile_b -> no-op, tile_c triggers ab_weight_update.
        """
        def hooked_update(tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                # Prevent double updates - only handle once per batch
                if self._update_handled:
                    return None

                # Tile C gets the full inputs, use those for LRTT update
                if tile_name == "tile_c":
                    self._update_handled = True  # Mark as handled

                    # Get learning rate
                    lr = self.tile_c.get_learning_rate()

                    # Route through controller for proper LRTT update
                    self.controller.ab_weight_update(
                        x=x_input,
                        d=d_input,
                        lr=lr,
                        in_trans=False,
                        out_trans=False,
                    )

                    # Check for transfer
                    if self.controller.should_transfer():
                        self.controller.ab_weight_transfer()

                # Don't call original update on any tile - LRTT handles all updates
                return None

            return update_wrapper

        # Replace update methods
        self.tile_a.update = hooked_update("tile_a")
        self.tile_b.update = hooked_update("tile_b")
        self.tile_c.update = hooked_update("tile_c")

    def _reset_update_flag(self) -> None:
        """Reset the update handled flag for next batch."""
        self._update_handled = False

    def get_tensor_view(self, ndim: int, dim: Optional[int] = None) -> tuple:
        """Return the tensor view for ndim vector at dim.

        Args:
            ndim: number of dimensions
            dim: the dimension to set to -1

        Returns:
            Tuple of ones with the `dim` index sets to -1
        """
        if dim is None:
            dim = 0 if getattr(self, "out_trans", False) else ndim - 1
        tensor_view = [1] * ndim
        tensor_view[dim] = -1
        return tuple(tensor_view)

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
        tensor_view: Optional[Tuple] = None,  # For array compatibility
    ) -> Tensor:
        """Forward pass with LRTT forward injection.

        Args:
            x_input: Input tensor
            bias: Bias flag (not supported)
            in_trans: Input transposed
            out_trans: Output transposed
            is_test: Not used
            non_blocking: Non-blocking flag

        Returns:
            Output tensor
        """
        # Reset update flag for this forward pass
        self._reset_update_flag()
        self._sync_controller_device(x_input)

        # The bias argument is not used here: the digital bias of tile_c
        # (set in __init__) is added by tile_c.forward()

        # The controller's forward_inject_enabled flag selects the forward path
        if self.controller.forward_inject_enabled:
            return self.controller.forward_inject(
                x_input, out_trans=out_trans, in_trans=in_trans
            )
        else:
            # Without forward injection: C only
            # (transposes are applied here; AnalogTile.forward takes no transpose flags)
            x = x_input.t() if in_trans else x_input
            y = self.tile_c.forward(x)
            return y.t() if out_trans else y

    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """LRTT backward pass using only analog tile operations.

        Computes:
        - If forward_inject_enabled: x_grad = C^T @ d + alpha * B^T @ (A^T @ d)
        - If forward_inject_disabled: x_grad = C^T @ d; the A/B updates are formed by
          LRTTController.ab_weight_update() from the input and error of the update call
        All operations use tile.backward(), so the backward IO model of each tile applies.

        The bias argument is not used; bias gradients are not computed here
        (the digital bias of tile_c is handled outside the analog backward pass).
        """
        # 1) Input to batch-first
        d_bf = d_input.t() if in_trans else d_input  # [batch, d_size]

        # 2) Always compute C gradient for upstream propagation
        xg_c = self.tile_c.backward(d_bf)  # [batch, x_size]

        if self.controller.forward_inject_enabled:
            # Full LRTT backward: C^T@d + alpha * B^T@(A^T@d)
            da = self.tile_a.backward(d_bf)  # [batch, rank]
            xg_ab = self.tile_b.backward(da)  # [batch, x_size]
            x_grad = xg_c + self.controller.effective_alpha * xg_ab
        else:
            # forward_inject=False: Upstream gets C-only gradient.
            # A,B updates are handled entirely by ab_weight_update() which does
            # its own tile_a.backward() and tile_b.forward() projections.
            # update() receives x_input/d_input directly; no need to store here.
            x_grad = xg_c

        # 3) Output transpose
        return x_grad.t() if out_trans else x_grad

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """LRTT update: A/B LoRA updates + periodic transfer.

        Args:
            x_input: Input tensor
            d_input: Error tensor
            bias: Bias flag (not used)
            in_trans: Input transposed
            out_trans: Output transposed
            non_blocking: Non-blocking flag

        The digital bias of tile_c is a torch parameter and is updated by the
        digital part of the optimizer (if train_c_bias is set).
        """
        # Prevent double updates
        if self._update_handled:
            return None
        self._update_handled = True
        self._sync_controller_device(x_input)

        # Get current learning rate (assuming all tiles have same LR)
        lr = self.get_learning_rate()

        # Perform A/B LoRA-style updates with projections
        # ab_weight_update does its own tile_b.forward() and tile_a.backward()
        # internally, so x_input and d_input are used directly.
        self.controller.ab_weight_update(
            x=x_input, d=d_input, lr=lr, in_trans=in_trans, out_trans=out_trans
        )

        # Check for transfer
        if self.controller.should_transfer():
            self.controller.ab_weight_transfer()

    def get_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weights of the C tile.

        The auxiliary product A @ B is not included (see get_effective_weights).

        Returns:
            Tuple of (C weights, C bias or None), as returned by tile_c.get_weights()
        """
        return self.tile_c.get_weights()

    def get_effective_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get effective LRTT weights: W_eff = W_visible + alpha * A @ B.

        This is a separate method for when effective weights are explicitly needed.

        Returns:
            Tuple of (effective_weights, None)
        """
        from aihwkit.linalg.lrtt_kernels import compose_lrtt_weights

        # Get individual component weights
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]

        # Compose effective weights
        W_eff = compose_lrtt_weights(
            visible_weights, A_weights, B_weights, self.controller.effective_alpha, self.rank
        )

        return W_eff, None

    def set_weights(
        self, weight: Tensor, bias: Optional[Tensor] = None, **kwargs
    ) -> None:
        """Set visible weights (and bias) on tile_c.

        When tile_c is created with bias=True, it supports digital_bias:
        - bias is stored as a PyTorch Parameter in tile_c.bias
        - forward() automatically adds bias to output
        - update() and transfer only affect weights, not bias

        Args:
            weight: Weight tensor [d_size, x_size]
            bias: Bias tensor [d_size] (stored in tile_c.bias if digital_bias=True)
            **kwargs: Additional arguments (passed to tile_c.set_weights)
        """
        # Pass bias to tile_c - it will handle digital_bias internally
        # If tile_c.digital_bias=True, bias is stored in tile_c.bias Parameter
        # If tile_c.digital_bias=False, bias is ignored
        self.tile_c.set_weights(weight, bias, **kwargs)

    def get_lrtt_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get individual LRTT component weights.

        Returns:
            Tuple of (visible_weights, A_weights, B_lr_weights)
        """
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]  # [rank, x_size]

        return visible_weights, A_weights, B_lr

    def set_lrtt_component_weights(
        self, visible: Tensor, A: Tensor, B_lr: Tensor
    ) -> None:
        """Set individual LRTT component weights.

        Args:
            visible: Visible weights [d_size, x_size]
            A: A weights [d_size, rank]
            B_lr: B weights [rank, x_size]
        """
        # Set C weights
        self.tile_c.set_weights(visible, None)

        # Set A weights
        self.tile_a.set_weights(A, None)

        # Set B weights
        self.tile_b.set_weights(B_lr, None)

    def get_x_size(self) -> int:
        """Get input size."""
        return self.x_size

    def get_d_size(self) -> int:
        """Get output size."""
        return self.d_size

    def get_learning_rate(self) -> float:
        """Get learning rate from visible tile."""
        return self.tile_c.get_learning_rate()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate for all tiles."""
        self.tile_a.set_learning_rate(learning_rate)
        self.tile_b.set_learning_rate(learning_rate)
        self.tile_c.set_learning_rate(learning_rate)

    def get_hidden_parameters(self) -> Tensor:
        """Get concatenated hidden parameters from all tiles."""
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters()
        params_c = self.tile_c.get_hidden_parameters()

        return torch.cat([params_a, params_b, params_c])

    def set_hidden_parameters(self, data: Tensor) -> None:
        """Set hidden parameters for all tiles."""
        # Split data based on tile parameter counts
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters()
        params_c = self.tile_c.get_hidden_parameters()

        size_a = params_a.numel()
        size_b = params_b.numel()
        size_c = params_c.numel()

        if data.numel() != size_a + size_b + size_c:
            raise TileError(
                f"Hidden parameter size mismatch: expected {size_a + size_b + size_c}, got {data.numel()}"
            )

        self.tile_a.set_hidden_parameters(data[:size_a])
        self.tile_b.set_hidden_parameters(data[size_a : size_a + size_b])
        self.tile_c.set_hidden_parameters(data[size_a + size_b :])

    def decay_weights(self, alpha: float = 0.0) -> None:
        """Apply weight decay to all tiles."""
        self.tile_a.decay_weights(alpha)
        self.tile_b.decay_weights(alpha)
        self.tile_c.decay_weights(alpha)

    def diffuse_weights(self, alpha: float = 0.0) -> None:
        """Apply weight diffusion to all tiles."""
        self.tile_a.diffuse_weights(alpha)
        self.tile_b.diffuse_weights(alpha)
        self.tile_c.diffuse_weights(alpha)

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Called by the analog optimizer after the weight updates. Applies decay
        and diffusion to each of the three tiles (A, B, C) whose device
        requires it, e.g. retention decay of a device with a finite lifetime
        such as the capacitor-based 6T1C model of the A/B tiles.
        """
        # Apply decay to each tile based on its device config (lifetime decay)
        # tile_a
        if (
            hasattr(self.tile_a, "rpu_config")
            and self.tile_a.rpu_config.device.requires_decay()
        ):
            self.tile_a.decay_weights()
        # tile_b
        if (
            hasattr(self.tile_b, "rpu_config")
            and self.tile_b.rpu_config.device.requires_decay()
        ):
            self.tile_b.decay_weights()
        # tile_c
        if (
            hasattr(self.tile_c, "rpu_config")
            and self.tile_c.rpu_config.device.requires_decay()
        ):
            self.tile_c.decay_weights()

        # Apply diffusion if needed
        if (
            hasattr(self.tile_a, "rpu_config")
            and self.tile_a.rpu_config.device.requires_diffusion()
        ):
            self.tile_a.diffuse_weights()
        if (
            hasattr(self.tile_b, "rpu_config")
            and self.tile_b.rpu_config.device.requires_diffusion()
        ):
            self.tile_b.diffuse_weights()
        if (
            hasattr(self.tile_c, "rpu_config")
            and self.tile_c.rpu_config.device.requires_diffusion()
        ):
            self.tile_c.diffuse_weights()

    def clip_weights(self, clip_type: str = "", sigma: float = 0.0) -> None:
        """Apply weight clipping to all tiles."""
        self.tile_a.clip_weights(clip_type, sigma)
        self.tile_b.clip_weights(clip_type, sigma)
        self.tile_c.clip_weights(clip_type, sigma)

    def reset_columns(
        self, start_column_idx: int = 0, num_columns: int = 1, sigma: float = 1.0
    ) -> None:
        """Reset columns in visible tile."""
        # Only reset visible tile columns (A/B managed by controller)
        self.tile_c.reset_columns(start_column_idx, num_columns, sigma)

    def get_brief_info(self) -> str:
        """Get brief tile information."""
        return f"LRTTSimulatorTile({self.d_size}, {self.x_size}, rank={self.rank})"

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_size={self.d_size}, x_size={self.x_size}, rank={self.rank}, "
            f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"
        )

    def get_controller_state(self) -> Dict[str, Any]:
        """Get the counters and settings of the LR-TT controller."""
        return self.controller.get_state_dict()

    def manual_transfer(self, method: Optional[str] = None) -> None:
        """Run an A@B -> C transfer now, outside the transfer schedule.

        Args:
            method: Transfer method override.
                   "onehot" - One-hot reading (pulsed update)
                   "direct" - Direct weight access (pulsed update)
                   "set" - Exact floating-point update of C (no pulses)
                   If None, use controller's transfer_method setting.
        """
        self.controller.ab_weight_transfer(method=method)

    def _sync_controller_device(self, x_input: Tensor) -> None:
        """Let the controller follow the input when the device types differ.

        A tile that stays in CPU memory never passes through ``cpu()``, so on a
        machine with a GPU the controller would keep its CUDA default.
        """
        if x_input.device.type != self.controller.device.type:
            self.controller.set_device(x_input.device)

    def _infer_device_from_self(self) -> torch.device:
        """Infer device from submodule parameters/buffers."""
        # Check parameters
        for p in self.parameters(recurse=True):
            return p.device
        # Check buffers
        for b in self.buffers(recurse=True):
            return b.device
        # Default to CUDA if available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, *args, **kwargs):
        """Move to device and synchronize controller."""
        super().to(*args, **kwargs)

        # Extract device from arguments
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])

        # If still no device, infer from self
        if device is None:
            device = self._infer_device_from_self()

        # Synchronize controller device
        if hasattr(self, "controller"):
            self.controller.set_device(device)

        return self

    def cuda(self, device=None):
        """Move to CUDA and synchronize controller."""
        super().cuda(device=device)

        # Determine CUDA device
        if device is None:
            cuda_device = torch.device("cuda")
        else:
            cuda_device = torch.device(f"cuda:{device}")

        # Synchronize controller
        if hasattr(self, "controller"):
            self.controller.set_device(cuda_device)

        return self

    def cpu(self):
        """Move to CPU and synchronize controller."""
        super().cpu()

        # Synchronize controller
        if hasattr(self, "controller"):
            self.controller.set_device(torch.device("cpu"))

        return self
