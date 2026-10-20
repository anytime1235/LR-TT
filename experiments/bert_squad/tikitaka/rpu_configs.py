#!/usr/bin/env python
# coding=utf-8
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""RPU config factories for the experiments of paper_experiment.py (3 methods).

Devices default to ConstantStepDevice (noise-free, dw_min-parameterized);
single_rpu and ttv1 also take LinearStep, ExpStep and SoftBounds devices.

IO paths (forward, backward, device.transfer_forward) are configured by
`_apply_io_config`:
  - io_bits=0/None -> all paths perfect (no quantization, no noise)
  - io_bits=N      -> the same IO applied to all three paths, transfer read
                      included (aihwkit models the A->C read as an analog MVM)

Methods:
  1. single_rpu  - SingleRPUConfig with pulsed update
  2. ttv1        - TransferCompound (Tiki-Taka v1, Gokmen & Haensch 2020)
  3. ideal       - IdealDevice (FP32 update, upper bound baseline)

W_eff formula for TTv1 (n=2 devices):
  W_eff = gamma * W_fast + 1.0 * W_slow
  gamma=0.0: W_eff = W_slow only (Gokmen & Haensch 2020 default)
  gamma=0.5: W_eff = 0.5*W_fast + 1.0*W_slow
  gamma=1.0: W_eff = W_fast + W_slow
"""

from aihwkit.simulator.configs.devices import (
    ConstantStepDevice, LinearStepDevice, ExpStepDevice,
    SoftBoundsReferenceDevice, IdealDevice,
)
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.helpers import build_config
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.enums import (
    PulseType, BoundManagementType, NoiseManagementType,
)


# ============================================================================
# Constants
# ============================================================================

DW_MIN_14BIT = 2.0 / (2 ** 14)  # 1.22e-4


PULSE_TYPE_MAP = {
    "stochastic": PulseType.STOCHASTIC_COMPRESSED,
    "deterministic": PulseType.DETERMINISTIC_IMPLICIT,
    "mean_count": PulseType.MEAN_COUNT,
    "none": PulseType.NONE,
    "none_with_device": PulseType.NONE_WITH_DEVICE,
}


# ============================================================================
# Helpers
# ============================================================================

def dw_min_for_bits(n_bits: int, w_max: float = 1.0) -> float:
    """Convert bit-resolution to dw_min step size."""
    return 2.0 * w_max / (2 ** n_bits)


def make_constant_step_device(dw_min=None, w_max=1.0, count_pulses=False):
    """Create a ConstantStepDevice with zero noise.

    Args:
        dw_min: Weight update step size. Defaults to 14-bit.
        w_max: Symmetric weight bound [-w_max, w_max]. Default 1.0.
        count_pulses: Enable hardware pulse counters (GPU only).
    """
    if dw_min is None:
        dw_min = DW_MIN_14BIT
    return ConstantStepDevice(
        dw_min=dw_min,
        w_max=w_max,
        w_min=-w_max,
        up_down=0.0,
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        count_pulses=count_pulses,
    )


def make_linear_step_device(dw_min=None, count_pulses=False,
                            gamma_up=-0.1678, gamma_down=0.1410,
                            gamma_up_ratio=1.0, gamma_down_ratio=1.0,
                            noise_ratio=0.0,
                            abs_gamma_up=None, abs_gamma_down=None,
                            dw_min_std_override=None, dw_min_dtod_override=None):
    """Create a LinearStepDevice with configurable gamma parameters.

    Two modes:
      1. Ratio mode (default): gamma = base * ratio  (6T1C base values)
      2. Absolute mode: if abs_gamma_up/abs_gamma_down are set, use them directly
         (overrides ratio-based scaling). For ECRAM/other device presets.

    Args:
        dw_min: Weight update step size. Defaults to 14-bit.
        count_pulses: Enable hardware pulse counters.
        gamma_up: 6T1C baseline gamma_up (-0.1678).
        gamma_down: 6T1C baseline gamma_down (0.1410).
        gamma_up_ratio: Scale factor for gamma_up (1.0 = 6T1C measured).
        gamma_down_ratio: Scale factor for gamma_down (1.0 = 6T1C measured).
        noise_ratio: Scale factor for all noise params (0.0 = noise-free,
                     1.0 = 6T1C measured).
        abs_gamma_up: Absolute gamma_up value (overrides ratio mode).
        abs_gamma_down: Absolute gamma_down value (overrides ratio mode).
        dw_min_std_override: If not None, override dw_min_std (ignoring noise_ratio scaling).
        dw_min_dtod_override: If not None, override dw_min_dtod only (other dtod params still scale by noise_ratio).
    """
    if dw_min is None:
        dw_min = DW_MIN_14BIT
    r_n = noise_ratio

    # Determine final gamma values
    final_gamma_up = abs_gamma_up if abs_gamma_up is not None else gamma_up * gamma_up_ratio
    final_gamma_down = abs_gamma_down if abs_gamma_down is not None else gamma_down * gamma_down_ratio

    return LinearStepDevice(
        dw_min=dw_min,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        mult_noise=False,
        gamma_up=final_gamma_up,
        gamma_down=final_gamma_down,
        # Noise: scaled by noise_ratio (dw_min_dtod can be overridden independently)
        dw_min_std=dw_min_std_override if dw_min_std_override is not None else 0.3 * r_n,
        dw_min_dtod=dw_min_dtod_override if dw_min_dtod_override is not None else 0.1 * r_n,
        up_down_dtod=0.01 * r_n,
        w_max_dtod=0.05 * r_n,
        w_min_dtod=0.05 * r_n,
        gamma_up_dtod=0.05 * r_n,
        gamma_down_dtod=0.05 * r_n,
        write_noise_std=0.0,
        mean_bound_reference=True,
        count_pulses=count_pulses,
    )


def make_exp_step_device(dw_min=None, count_pulses=False,
                         gamma_up=5.0, gamma_down=5.0,
                         A_up=-1.18445, A_down=-0.081404,
                         a=-0.5, b=-0.5,
                         noise_ratio=0.0):
    """Create an ExpStepDevice with RRAM-like parameters.

    Default values match ReRamESPresetDevice (Gong et al., Nat. Commun., 2018).

    Args:
        dw_min: Weight update step size. Defaults to 14-bit.
        count_pulses: Enable hardware pulse counters.
        gamma_up/gamma_down: Exponential nonlinearity parameters.
        A_up/A_down: Amplitude parameters for up/down pulses.
        a/b: Shape parameters.
        noise_ratio: Scale factor for noise (0=noise-free, 1=RRAM measured).
    """
    if dw_min is None:
        dw_min = DW_MIN_14BIT
    r_n = noise_ratio
    return ExpStepDevice(
        dw_min=dw_min,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        a=a,
        b=b,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        A_up=A_up,
        A_down=A_down,
        # Noise: scaled by noise_ratio
        dw_min_dtod=0.2 * r_n,
        up_down_dtod=0.05 * r_n,
        w_max_dtod=0.3 * r_n,
        w_min_dtod=0.3 * r_n,
        dw_min_std=5.0 * r_n,
        write_noise_std=75.0 * r_n,
        count_pulses=count_pulses,
    )


def make_soft_bounds_device(dw_min=None, count_pulses=False, noise_ratio=0.0):
    """Create a SoftBoundsReferenceDevice (noise-free by default).

    Matches ReRamArrayHfO2PresetDevice structure (Gong & Rasch, IEDM 2022).
    Nonlinearity comes from soft bounds behavior (update size depends on
    proximity to w_max/w_min), not from gamma parameters.

    Args:
        dw_min: Weight update step size. Defaults to 14-bit.
        count_pulses: Enable hardware pulse counters.
        noise_ratio: Scale factor for noise (0=noise-free, 1=HfO2 measured).
    """
    if dw_min is None:
        dw_min = DW_MIN_14BIT
    r_n = noise_ratio
    return SoftBoundsReferenceDevice(
        dw_min=dw_min,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        mult_noise=False,
        # Noise: scaled by noise_ratio (HfO2 baseline values)
        dw_min_dtod=0.7125 * r_n,
        up_down_dtod=0.01 * r_n,
        w_max_dtod=0.4295 * r_n,
        w_min_dtod=0.5990 * r_n,
        dw_min_std=0.2174 * r_n,
        write_noise_std=0.5841 * r_n,
        subtract_symmetry_point=True,
        reference_std=0.05 * r_n,
        count_pulses=count_pulses,
    )


def io_res_from_bits(n_bits: int) -> float:
    """Convert IO bit-width to DAC/ADC resolution parameter.

    Uses aihwkit convention: res = 1 / (2^N - 2).
    This matches the default values:
      - 7-bit DAC: inp_res = 1/126  (aihwkit forward default)
      - 9-bit ADC: out_res = 1/510  (aihwkit forward default)
    """
    return 1.0 / (2 ** n_bits - 2)


NOISE_MGMT_MAP = {
    "none": NoiseManagementType.NONE,
    "abs_max": NoiseManagementType.ABS_MAX,
}


def _apply_io_config(config, io_bits=None, noise_management="abs_max", out_noise=0.0):
    """Configure all analog IO paths: forward, backward and transfer_forward.

    Args:
        config: RPU config to modify.
        io_bits: DAC/ADC bit precision. None or 0 means perfect IO.
                 Positive integer sets finite resolution (same for
                 inp_res/out_res on every path).
                 Uses aihwkit convention: res = 1/(2^N - 2).
        noise_management: "abs_max" or "none".
        out_noise: Analog output (ADC) noise std (default 0.0 = pure
                   quantization). e.g. 0.04.

    When io_bits > 0, each path gets identical settings:
      - inp_res = out_res = 1/(2^io_bits - 2)
      - inp_noise = 0, out_noise = <out_noise>  (stochastic ADC output noise)
      - bound_management = ITERATIVE
      - noise_management = as specified

    The A->C transfer read is an analog MVM (Gokmen & Haensch 2020), and
    aihwkit models it as such: `TransferCompound.transfer_forward` defaults to
    a non-perfect `IOParameters`, and the built-in TikiTaka presets give it the
    same quantization and noise as forward/backward. It is configured here with
    the same settings as forward/backward; at gamma=0 the transfer is the only
    path into the core tile. aihwkit applies bound management in the forward
    pass only, and TransferRPUDevice runs the transfer read without noise
    management and bound management (input/output resolution and output noise
    still apply), so the noise and bound management set here act on the
    forward pass (noise management also on the backward pass).
    """
    if io_bits is None or io_bits == 0:
        config.forward.is_perfect = True
        config.backward.is_perfect = True
        if hasattr(config, "device") and hasattr(config.device, "transfer_forward"):
            config.device.transfer_forward.is_perfect = True
        return config

    res = io_res_from_bits(io_bits)
    nm = NOISE_MGMT_MAP.get(noise_management, NoiseManagementType.ABS_MAX)

    paths = [config.forward, config.backward]
    if hasattr(config, "device") and hasattr(config.device, "transfer_forward"):
        paths.append(config.device.transfer_forward)

    for io in paths:
        io.is_perfect = False
        io.inp_res = res
        io.out_res = res
        io.inp_noise = 0.0
        io.out_noise = out_noise
        io.bound_management = BoundManagementType.ITERATIVE
        io.noise_management = nm

    return config


def _apply_common_mapping(config, omega=1.0):
    """Apply common mapping settings to any RPU config.

    Args:
        omega: weight_scaling_omega. Set to 0.0 to disable weight scaling
               (required when using reduced w_max without pretrained weight mapping).
    """
    config.mapping.digital_bias = True
    config.mapping.weight_scaling_omega = omega
    config.mapping.weight_scaling_columnwise = True if omega > 0 else False
    config.mapping.learn_out_scaling = False
    config.mapping.out_scaling_columnwise = False
    return config


# ============================================================================
# Config builders
# ============================================================================

def build_single_rpu_config(pulse_type=PulseType.STOCHASTIC_COMPRESSED,
                            desired_bl=31, dw_min=None, w_max=1.0,
                            omega=1.0, count_pulses=False,
                            io_bits=None, noise_management="abs_max",
                            out_noise=0.0,
                            device_type="constant_step",
                            ls_gamma_up_ratio=1.0, ls_gamma_down_ratio=1.0,
                            ls_noise_ratio=0.0,
                            ls_gamma_up=None, ls_gamma_down=None):
    """SingleRPU - analog pulsed update.

    Args:
        pulse_type: PulseType enum value.
        desired_bl: Max pulse train length (default 31).
        dw_min: Override dw_min (default None -> 14-bit).
        w_max: Symmetric weight bound. Default 1.0.
        omega: weight_scaling_omega. 0 disables scaling.
        count_pulses: Enable hardware pulse counters.
        io_bits: DAC/ADC bit precision (None or 0 = perfect).
        out_noise: Analog output (ADC) noise std on the forward and backward
            paths. Default 0.0 = quantization only.
        device_type: 'constant_step', 'linear_step', 'exp_step', or 'soft_bounds'.
        ls_gamma_up_ratio: Scale factor for gamma_up (LinearStep).
        ls_gamma_down_ratio: Scale factor for gamma_down (LinearStep).
        ls_noise_ratio: Scale factor for noise (0=noise-free).
        ls_gamma_up: Absolute gamma_up value (overrides ratio mode).
        ls_gamma_down: Absolute gamma_down value (overrides ratio mode).
    """
    if device_type == "linear_step":
        device = make_linear_step_device(
            dw_min=dw_min, count_pulses=count_pulses,
            gamma_up_ratio=ls_gamma_up_ratio,
            gamma_down_ratio=ls_gamma_down_ratio,
            noise_ratio=ls_noise_ratio,
            abs_gamma_up=ls_gamma_up,
            abs_gamma_down=ls_gamma_down,
        )
    elif device_type == "exp_step":
        device = make_exp_step_device(
            dw_min=dw_min, count_pulses=count_pulses,
            gamma_up=ls_gamma_up if ls_gamma_up is not None else 5.0,
            gamma_down=ls_gamma_down if ls_gamma_down is not None else 5.0,
            noise_ratio=ls_noise_ratio,
        )
    elif device_type == "soft_bounds":
        device = make_soft_bounds_device(
            dw_min=dw_min, count_pulses=count_pulses,
            noise_ratio=ls_noise_ratio,
        )
    else:
        device = make_constant_step_device(dw_min=dw_min, w_max=w_max, count_pulses=count_pulses)
    up = UpdateParameters(
        pulse_type=pulse_type,
        desired_bl=desired_bl,
        fixed_bl=True,
    )
    config = build_config("sgd", device, up_parameters=up)
    _apply_io_config(config, io_bits=io_bits, noise_management=noise_management,
                     out_noise=out_noise)
    return _apply_common_mapping(config, omega=omega)


def build_ttv1_config(gamma=0.0, dw_min=None, dw_min_slow=None,
                      w_max_fast=None,
                      transfer_every=None,
                      units_in_mbatch=None, fast_lr=None, transfer_lr=None,
                      scale_transfer_lr=None, n_reads_per_transfer=None,
                      with_reset_prob=None, desired_bl=31, transfer_bl=31,
                      count_pulses=False,
                      fast_pulse_type=None, transfer_pulse_type=None,
                      io_bits=None, noise_management="abs_max", out_noise=0.0,
                      device_type="constant_step",
                      ls_gamma_up_ratio=1.0, ls_gamma_down_ratio=1.0,
                      ls_noise_ratio=0.0,
                      ls_gamma_up=None, ls_gamma_down=None,
                      device_type_slow="constant_step",
                      ls_gamma_up_slow=None, ls_gamma_down_slow=None,
                      ls_noise_ratio_slow=0.0,
                      ls_dw_min_std=None, ls_dw_min_dtod=None):
    """TTv1 - TransferCompound (Gokmen & Haensch 2020).

    W_eff = gamma * W_fast + 1.0 * W_slow

    Args:
        gamma: Forward weighting of fast tile. Default 0.0.
        dw_min: Override dw_min for fast tile (default None -> 14-bit).
        dw_min_slow: Override dw_min for slow tile (default None -> same as dw_min).
        transfer_every: Transfer cycle length. Units depend on units_in_mbatch.
        units_in_mbatch: True=mini-batch units, False=mat-vec units.
        fast_lr: Pulse-count learning rate of the fast (auxiliary) tile; when
            > 0 it replaces the optimizer learning rate for the fast-tile update.
        transfer_lr: Learning rate for the A->C transfer write.
        scale_transfer_lr: If True, scale transfer_lr by current analog LR.
        n_reads_per_transfer: Columns transferred per event.
        with_reset_prob: Prob of resetting fast tile columns after transfer.
        desired_bl: SGD update pulse train length (config.update.desired_bl).
        transfer_bl: Transfer write pulse train length (config.device.transfer_update.desired_bl).
        count_pulses: Enable hardware pulse counters.
        fast_pulse_type: Override PulseType for fast-tile SGD update (config.update.pulse_type).
        transfer_pulse_type: Override PulseType for the A->C transfer write (config.device.transfer_update.pulse_type).
        io_bits: DAC/ADC bit precision (None or 0 = perfect).
    """
    # --- Fast tile device ---
    if device_type == "linear_step":
        fast_device = make_linear_step_device(
            dw_min=dw_min, count_pulses=count_pulses,
            gamma_up_ratio=ls_gamma_up_ratio,
            gamma_down_ratio=ls_gamma_down_ratio,
            noise_ratio=ls_noise_ratio,
            abs_gamma_up=ls_gamma_up,
            abs_gamma_down=ls_gamma_down,
            # --ls-dw-min-std / --ls-dw-min-dtod override the noise-ratio
            # scaling of these two fields.
            dw_min_std_override=ls_dw_min_std,
            dw_min_dtod_override=ls_dw_min_dtod,
        )
    elif device_type == "exp_step":
        fast_device = make_exp_step_device(
            dw_min=dw_min, count_pulses=count_pulses,
            gamma_up=ls_gamma_up if ls_gamma_up is not None else 5.0,
            gamma_down=ls_gamma_down if ls_gamma_down is not None else 5.0,
            noise_ratio=ls_noise_ratio,
        )
    elif device_type == "soft_bounds":
        fast_device = make_soft_bounds_device(
            dw_min=dw_min, count_pulses=count_pulses,
            noise_ratio=ls_noise_ratio,
        )
    else:
        fast_device = make_constant_step_device(dw_min=dw_min, count_pulses=count_pulses)

    # --- Slow tile device ---
    _dw_min_slow = dw_min_slow if dw_min_slow is not None else dw_min
    if device_type_slow == "linear_step":
        slow_device = make_linear_step_device(
            dw_min=_dw_min_slow, count_pulses=count_pulses,
            noise_ratio=ls_noise_ratio_slow,
            abs_gamma_up=ls_gamma_up_slow,
            abs_gamma_down=ls_gamma_down_slow,
        )
    elif device_type_slow == "constant_step":
        if device_type != "constant_step" or dw_min_slow is not None:
            slow_device = make_constant_step_device(
                dw_min=_dw_min_slow, count_pulses=count_pulses,
            )
        else:
            slow_device = None  # use default (same as fast)
    else:
        slow_device = make_constant_step_device(
            dw_min=_dw_min_slow, count_pulses=count_pulses,
        )
    up = UpdateParameters(
        pulse_type=PulseType.STOCHASTIC_COMPRESSED,
        desired_bl=desired_bl,
        fixed_bl=True,
    )
    config = build_config("ttv1", fast_device, up_parameters=up)
    config.device.gamma = gamma

    # Set slow tile device
    if slow_device is not None:
        config.device.unit_cell_devices[1] = slow_device
    elif dw_min_slow is not None:
        config.device.unit_cell_devices[1].dw_min = dw_min_slow

    # Set different w_max for fast tile (A tile)
    # Fast tile starts at 0, no pretrained weights -> can use smaller range
    # This reduces dw_min_fast = 2*w_max_fast/2^n, increasing mu
    if w_max_fast is not None:
        fast_dev = config.device.unit_cell_devices[0]
        fast_dev.w_max = w_max_fast
        fast_dev.w_min = -w_max_fast
        # Recalculate dw_min for the fast tile to match the new range
        if dw_min is not None:
            fast_dev.dw_min = dw_min * w_max_fast  # scale proportionally
        else:
            fast_dev.dw_min = DW_MIN_14BIT * w_max_fast

    # Transfer update pulse train length (independent from SGD update BL)
    config.device.transfer_update.desired_bl = transfer_bl

    if transfer_every is not None:
        config.device.transfer_every = transfer_every
    if units_in_mbatch is not None:
        config.device.units_in_mbatch = units_in_mbatch
    if fast_lr is not None:
        config.device.fast_lr = fast_lr
    if transfer_lr is not None:
        config.device.transfer_lr = transfer_lr
    if scale_transfer_lr is not None:
        config.device.scale_transfer_lr = scale_transfer_lr
    if n_reads_per_transfer is not None:
        config.device.n_reads_per_transfer = n_reads_per_transfer
    if with_reset_prob is not None:
        config.device.with_reset_prob = with_reset_prob

    # Override pulse types if specified
    if fast_pulse_type is not None:
        config.update.pulse_type = fast_pulse_type
    if transfer_pulse_type is not None:
        config.device.transfer_update.pulse_type = transfer_pulse_type

    _apply_io_config(config, io_bits=io_bits, noise_management=noise_management,
                     out_noise=out_noise)
    return _apply_common_mapping(config)


def build_ideal_config(io_bits=None, noise_management="abs_max"):
    """IdealDevice - FP32 update. Upper bound baseline.

    Args:
        io_bits: DAC/ADC bit precision (None or 0 = perfect).
    """
    config = SingleRPUConfig(device=IdealDevice())
    _apply_io_config(config, io_bits=io_bits, noise_management=noise_management)
    return _apply_common_mapping(config)


# ============================================================================
# Dispatcher
# ============================================================================

METHOD_BUILDERS = {
    "single_rpu": build_single_rpu_config,
    "ttv1": build_ttv1_config,
    "ideal": build_ideal_config,
}


def get_config(method: str, **kwargs):
    """Get RPU config by method name.

    Args:
        method: One of 'single_rpu', 'ttv1', 'ideal'.
        **kwargs: Forwarded to the method builder. Keyword arguments that a
                  builder does not declare are dropped, so one argument set
                  can be passed to every builder.

    Returns:
        RPU config instance.

    Raises:
        ValueError: If method is unknown.
    """
    if method not in METHOD_BUILDERS:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(METHOD_BUILDERS.keys())}"
        )

    builder = METHOD_BUILDERS[method]

    # Filter kwargs to only those the builder accepts
    import inspect
    sig = inspect.signature(builder)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return builder(**valid_kwargs)


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    from aihwkit.simulator.configs.compounds import TransferCompound

    for name in METHOD_BUILDERS:
        config = get_config(name)
        print(f"\n{'='*60}")
        print(f"Method: {name}")
        print(f"{'='*60}")
        print(f"Config type: {type(config).__name__}")
        print(f"Forward perfect: {config.forward.is_perfect}")
        print(f"Backward perfect: {config.backward.is_perfect}")
        print(f"learn_out_scaling: {config.mapping.learn_out_scaling}")
        if hasattr(config, "device"):
            print(f"Device type: {type(config.device).__name__}")
            if hasattr(config.device, "transfer_forward"):
                print(f"  transfer_forward.is_perfect: {config.device.transfer_forward.is_perfect}")
            if hasattr(config.device, "gamma"):
                print(f"  gamma: {config.device.gamma}")
            if hasattr(config.device, "fast_lr"):
                print(f"  fast_lr: {config.device.fast_lr}")
            if hasattr(config.device, "units_in_mbatch"):
                print(f"  units_in_mbatch: {config.device.units_in_mbatch}")
            if hasattr(config.device, "transfer_every"):
                print(f"  transfer_every: {config.device.transfer_every}")

    # Assertions
    ttv1_cfg = get_config("ttv1")
    assert isinstance(ttv1_cfg.device, TransferCompound), "TTv1 must use TransferCompound"
    assert ttv1_cfg.device.transfer_forward.is_perfect, \
        "default (io_bits=0) must leave transfer_forward perfect"

    ideal_cfg = get_config("ideal")
    assert isinstance(ideal_cfg.device, IdealDevice), "ideal must use IdealDevice"

    # With io_bits > 0 the transfer read must be configured on equal terms with
    # forward/backward, not forced perfect.
    io_fields = ("is_perfect", "inp_res", "out_res", "inp_noise", "out_noise",
                 "noise_management", "bound_management")

    cfg = get_config("ttv1", io_bits=8, out_noise=0.04, noise_management="abs_max")
    tf = cfg.device.transfer_forward
    assert not tf.is_perfect, "ttv1: transfer_forward must not be perfect when io_bits>0"
    for f in io_fields:
        assert getattr(tf, f) == getattr(cfg.forward, f), (
            f"ttv1 transfer_forward.{f}: got {getattr(tf, f)!r}, "
            f"forward has {getattr(cfg.forward, f)!r}")

    # ttv1 passes out_noise to all three paths.
    ttv1_io = get_config("ttv1", io_bits=8, out_noise=0.04)
    for tag, io in [("forward", ttv1_io.forward), ("backward", ttv1_io.backward),
                    ("transfer_forward", ttv1_io.device.transfer_forward)]:
        assert io.out_noise == 0.04, f"{tag}.out_noise = {io.out_noise}, expected 0.04"
        assert abs(io.inp_res - 1.0 / (2 ** 8 - 2)) < 1e-12, \
            f"{tag}.inp_res = {io.inp_res}, expected 1/254"

    print("\n\nAll assertions passed!")
