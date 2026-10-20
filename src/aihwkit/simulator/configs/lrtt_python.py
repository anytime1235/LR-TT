# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""Device configuration of Low-Rank Tiki-Taka (LR-TT).

``PythonLRTTDevice`` holds the parameters of an LR-TT block: rank, transfer
schedule and method, reinitialization of the auxiliary factors, learning rate
of the auxiliary updates, optional forward injection and the devices of the
three tiles. ``LRTTSimulatorTile`` builds the A, B and C tiles from it as
standard aihwkit tiles, and ``LRTTController`` performs the LR-TT update and
transfer on these tiles in Python. ``PythonLRTTPreset`` collects example
device combinations, and ``get_6t1c_lifetime_for_dt_batch`` converts the 6T1C
retention time constant into the ``lifetime`` parameter of the simulator.

Notation: ``A`` is the ``[d_size, rank]`` factor (tile_a) and ``B`` the
``[rank, x_size]`` factor (tile_b); a transfer adds ``A @ B``, scaled by the
transfer learning rate, to the core tile C. In the notation of the paper these
factors are B (M x r) and A (r x N), and the transferred product is BA.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import warnings

from aihwkit.simulator.configs.devices import PulsedDevice, ConstantStepDevice, LinearStepDevice
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.parameters.helpers import _PrintableMixin
from aihwkit.simulator.parameters.mapping import MappingParameter


@dataclass  
class PythonLRTTDevice(_PrintableMixin):
    """Parameters of an LR-TT block.

    The three entries of ``unit_cell_devices`` are the devices of the A tile
    (``[d_size, rank]``), the B tile (``[rank, x_size]``) and the core tile C
    (``[d_size, x_size]``). ``LRTTSimulatorTile`` passes the fields to
    ``LRTTController`` (see ``to_controller_kwargs()``).

    The defaults below are generic; the experiment scripts set the LR-TT
    parameters explicitly. The BERT-base LR-TT scripts, for example, use
    ``reinit_mode='decay'``, ``scale_transfer_lr=False``,
    ``units_in_mbatch=True``, ``transfer_method='onehot'`` (``'set'`` for the
    ideal-transfer runs) and ``forward_inject=False``.
    """

    # === Core LR-TT parameters ===
    rank: int = 4
    """Rank r of the auxiliary factors. Must be > 0 and <= min(d_size, x_size)."""

    transfer_every: int = 32
    """Transfer period: every N mini-batches (if units_in_mbatch=True)
    or every N mat-vec operations, i.e. samples (if units_in_mbatch=False)."""

    transfer_lr: float = 1.0
    """Transfer learning rate of the A @ B -> C transfer (see also
    scale_transfer_lr and correct_gradient_magnitudes)."""

    scale_transfer_lr: bool = True
    """If True, multiply transfer_lr by the current SGD learning rate (lr_sgd).
    Effective: lr_tr = transfer_lr * lr_sgd (when True) or transfer_lr (when False)."""

    transfer_fast_lr_ref: str = "geomean"
    """Reference for correct_gradient_magnitudes transfer LR division:
    - 'A': divide by lr_eff_a
    - 'B': divide by lr_eff_b
    - 'geomean': divide by sqrt(lr_eff_a * lr_eff_b)
    Only used when correct_gradient_magnitudes=True."""

    lora_alpha: float = 1.0
    """Scale alpha of the auxiliary product in W_eff = C + alpha * A @ B.
    Used by forward injection (y = Cx + alpha*ABx) and by
    LRTTSimulatorTile.get_effective_weights(); it does not enter the A/B
    learning rate (see fast_lr)."""

    fi_continuous_alpha: bool = False
    """When True and forward_inject=True, alpha is set to transfer_lr
    (or transfer_lr * lr_sgd when scale_transfer_lr=True) instead of lora_alpha,
    so that the alpha of y = Cx + alpha*ABx follows the transfer learning rate
    of C += lr*AB at every step."""

    fast_lr: float = 1.0
    """Learning rate of the A/B weight updates.
    In auto_scale_mode='none': lr_eff = fast_lr.
    In auto_scale_mode='shared'/'separate': fast_lr is normalized by the signal EMA.
    With use_manual_scaling=True in rpu_config.update the A/B tiles are updated
    with learning rate 1 and the pulse probabilities follow from the manual scales."""

    auto_scale_mode: str = "none"
    """Auto-scale mode for A/B learning rate normalization:
    - 'none': lr_eff = fast_lr (no signal-based scaling)
    - 'shared': lr_eff = fast_lr / (m_x * m_d), shared across A and B
    - 'separate': lr_eff_a = fast_lr * gran_a / (m_xb * m_d) and
      lr_eff_b = fast_lr * gran_b / (m_x * m_da), with gran = desired_bl * dw_min
      of the A or B device
    m_x, m_d, m_xb and m_da are EMAs of the per-sample absolute maxima of x, d,
    Bx and A^T d."""

    auto_momentum: float = 0.99
    """EMA momentum for auto-scale signal tracking. tau = (1 - auto_momentum) / m_batch.
    Higher values = smoother EMA, slower adaptation. Must be in (0, 1)."""

    reinit_gain: float = 1.0
    """Gain of the Kaiming-uniform draws of A and B (kaiming_uniform_ with
    a=sqrt(5); 1.0 gives the PyTorch nn.Linear default)."""

    reinit_mode: str = "standard"
    """Treatment of A and B after a transfer:
    - 'standard': A and B are reinitialized after every transfer
      (a_init_mode, b_init_mode)
    - 'decay': A and B are initialized once and kept after every transfer;
      between transfers they change by their updates and, for devices with a
      finite lifetime, by retention decay
    - 'hybrid': A is reset after every transfer; B is initialized once and kept
    """

    a_init_mode: str = "zero"
    """Initialization of A:
    - 'zero': reset through the device reset (reset_columns), i.e. the reset
      value of the device plus reset_std noise; exactly zero for floating-point tiles
    - 'kaiming': Kaiming-uniform draw scaled by reinit_gain
    """

    b_init_mode: str = "kaiming"
    """Initialization of B:
    - 'kaiming': Kaiming-uniform draw scaled by reinit_gain
    - 'zero': B = 0
    """

    # === Transfer read settings ===
    num_reads: int = 1
    """Number of one-hot reads per rank in the transfer (used when read_n_avg is 1).
    With more than one read, each read is differential (+e/-e) and the reads
    are combined as set by multi_read_mode."""

    multi_read_mode: str = "average"
    """Use of several reads (num_reads > 1) in transfer_mode='pilot':
    - 'average': average the reads, then transfer once
    - 'per_read': transfer after each read with lr / num_reads
    The other transfer modes average the reads."""

    # === Transfer mode ===
    transfer_mode: str = "off"
    """Variant of the one-hot transfer (transfer_method='onehot'):
    - 'off': each selected rank k is written into C with transfer_micro_steps
      pulsed updates a_k b_k^T of learning rate |lr_tr| / transfer_micro_steps
    - 'pilot': as 'off' with at least two micro-steps. If the controller
      attribute transfer_gamma_mode is also 'pilot', a pilot transfer of
      transfer_pilot_frac * |lr_tr| first measures the transfer gain gamma
      (clamped to [0.05, 20]), and the remaining micro-steps write
      gamma * |lr_tr| minus the pilot learning rate.
    - 'sigma_delta': sigma-delta quantization per rank: the target learning rate
      accumulates in a residual h_k and is written as n_k = round(h_k / g)
      updates of learning rate g; the residual is kept for the next transfer.
    Default is 'off'."""

    transfer_micro_steps: int = 1
    """M: number of pulsed updates per rank and transfer, each with 1/M of the
    transfer learning rate ('off' and 'pilot' modes; 'pilot' uses at least 2).
    For 'sigma_delta' mode: g = |lr_tr| / M if sd_quantum is not set."""

    transfer_pilot_frac: float = 0.0625  # 1/16
    """Fraction of |transfer_lr| used for the pilot transfer in 'pilot' mode.
    Default is 1/16 = 0.0625."""

    sd_quantum: Optional[float] = None
    """Unit quantum g for sigma-delta mode. If None, derived as |lr_tr| / transfer_micro_steps.
    Each rank accumulates the target learning rate and is written as n_k updates of size g."""

    # === Read averaging ===
    read_n_avg: int = 1
    """Number of reads averaged per rank in the one-hot transfer.
    - 1: single read (num_reads applies)
    - N > 1: N differential (+e/-e) reads are averaged
    Default is 1."""

    differential_read: bool = False
    """Differential one-hot read (+e, -e) for single reads.
    - True: read with +e and -e and use 0.5*(f(+e) - f(-e)), which cancels
      offsets and doubles the number of reads.
    - False: read with +e only.
    Default is False."""

    # === AGC (automatic gain control) of the one-hot read ===
    agc_enabled: bool = False
    """Enable AGC of the one-hot read amplitude: a search over the amplitude
    keeps the read output below agc_margin * out_bound."""

    agc_margin: float = 0.85
    """Target output bound fraction for AGC (avoid clipping).
    0.85 = 85% of out_bound. Default is 0.85."""

    agc_max_iters: int = 6
    """Max iterations for AGC binary search. Default is 6."""

    # === Two-amplitude differential read ===
    two_amp_enabled: bool = False
    """Enable the two-amplitude differential read, which reads each rank at two
    amplitudes a1 < a2 and removes an amplitude-independent offset."""

    two_amp_ratio: float = 0.5
    """Ratio a1/a2 for two-amplitude method. Default is 0.5."""

    # === Update mode ===
    update_mode: str = "lora"
    """A/B update mode without forward injection:
    - 'lora': LR-TT update from the projections: A is updated with (Bx, d) and
      B with (x, A^T d) (with forward injection, autograd gives the same update)
    - 'reconstruction': gradient-reconstruction update that drives A @ B toward
      the negative gradient, with L2 and stabilizer terms (recon_* parameters)
    Default is 'lora'."""

    # === Reconstruction update parameters (update_mode='reconstruction') ===
    recon_lambda_a: float = 1e-3
    """L2 regularization coefficient for A in reconstruction loss."""

    recon_lambda_b: float = 1e-3
    """L2 regularization coefficient for B in reconstruction loss."""

    recon_use_scalar_stabilizer: bool = False
    """Use scalar approximation for stabilizer terms (BB^T ~ sB*I, A^TA ~ sA*I).
    Disabled by default."""

    recon_use_exact_gram: bool = False
    """Use the exact Gram matrices (BB^T, A^TA) in the stabilizer terms at every
    step (reads A and B and computes two rank x rank products)."""

    recon_exact_gram_every: int = 0
    """Use the exact Gram matrices every N steps (0 = disabled)."""

    recon_ema_beta: float = 0.9
    """EMA decay of the norm estimates sA, sB of the scalar stabilizer."""

    recon_lr_scale: float = 1.0
    """Learning rate scale of the reconstruction updates (lr_rec = lr * recon_lr_scale)."""

    recon_clip_norm: float = 10.0
    """Maximum Frobenius norm of A and B. Only used if recon_use_clip_norm=True."""

    recon_use_clip_norm: bool = False
    """Enable norm clipping for A,B. Disabled by default."""

    # === Transfer method ===
    transfer_method: str = "onehot"
    """Transfer method:
    - "onehot": One-hot transfer (rank-by-rank one-hot read of A and B,
      pulsed update of C; see transfer_mode)
    - "direct": Direct transfer (A and B from get_weights, pulsed update of C)
    - "set": Exact transfer (A and B from the stored weights, floating-point
      update of C with device clipping, PulseType.NONE_WITH_DEVICE)
    Default is "onehot"."""

    # === Transfer rank scheduling ===
    transfer_rank_schedule: str = "all"
    """Transfer rank scheduling mode:
    - "all": Transfer all ranks at every transfer (default)
    - "round_robin": Cycle through the ranks, transferring transfer_ranks_per_step
      ranks at every transfer; only these ranks are reinitialized afterwards
    Default is "all"."""

    transfer_ranks_per_step: int = 1
    """Number of ranks to transfer per transfer event in round_robin mode.
    Only used when transfer_rank_schedule="round_robin".
    Must be >= 1 and <= rank. Default is 1."""

    # === Further parameters ===
    units_in_mbatch: bool = False
    """If True, transfer_every is in mini-batch units (TE=1 -> every batch).
    If False, transfer_every is in mat-vec units (TE=1 -> every sample).
    Same convention as aihwkit's Tiki-Taka transfer devices."""

    ab_io_perfect: bool = False
    """If True, make the low-rank A and B tile reads fully ideal
    (is_perfect: no out_noise / ADC / DAC / bound on A & B forward and
    backward). Models the A/B low-rank update as a digital adapter while
    keeping the visible C tile non-ideal."""
    
    correct_gradient_magnitudes: bool = False
    """If True, divide the transfer LR by the effective A/B learning rate
    (lr_eff_a, lr_eff_b or their geometric mean, see transfer_fast_lr_ref)."""

    forward_inject: bool = False
    """Forward injection: if True, the forward pass computes
    y = Cx + alpha * A(Bx) and A, B receive their updates through autograd;
    if False, the forward pass uses C only and the A/B updates are formed from
    the projections Bx and A^T d."""

    ab_pulse_type: str = "default"
    """Pulse type for A/B tile weight updates:
    - 'default': pulse type of rpu_config.update (typically StochasticCompressed)
    - 'none': Pure floating point update (no pulses, no device effects)
    - 'none_with_device': Floating point update with device effects (weight clipping)
    - 'stochastic': Stochastic bit lines
    - 'stochastic_compressed': Stochastic bit lines, +/- in same pass
    - 'mean_count': Coincidence based in prob (p_a * p_b)
    - 'deterministic_implicit': Deterministic coincidences with quantized input/error
    """

    # === C tile bias ===
    train_c_bias: bool = False
    """Allow C tile bias to be trainable. Default False (frozen).
    Set True for scratch training where bias needs to be learned."""

    # === Dynamic transfer period ===
    dynamic_te: bool = False
    """Enable dynamic transfer_every that increases as LR decays.
    TE(t) = clip(te_min, te_max, round(TE_0 * (lr_peak / lr_current)^p)).
    Only activates in LR decay phase (lr_current < lr_peak)."""

    dynamic_te_power: float = 1.0
    """Power p for dynamic TE. 1.0=exact LR inverse, 0.5=smoother."""

    dynamic_te_min: Optional[int] = None
    """Minimum TE floor. None=use TE_0 (never decrease below initial value)."""

    dynamic_te_max: Optional[int] = None
    """Maximum TE ceiling. None=10*TE_0."""

    te_warmup_schedule: Optional[List[int]] = None
    """Step-wise TE warmup schedule, e.g. [32, 64, 128, 230].
    During warmup, TE transitions through these values in equal intervals.
    The last value should match transfer_every (TE_0). None=no warmup ramp."""

    te_warmup_steps: int = 0
    """Number of steps for TE warmup ramp. The schedule is divided equally."""

    # === Tile mapping parameters ===
    mapping_ab: MappingParameter = field(default_factory=lambda: MappingParameter(
        weight_scaling_omega=0.0,
        learn_out_scaling=False,
    ))
    """A/B tile mapping (default for both A and B). Default: no weight scaling
    (omega=0), no learnable out_scaling. Overridden per-tile by mapping_a / mapping_b
    when those are not None."""

    mapping_a: Optional[MappingParameter] = None
    """Optional override for A tile mapping. None -> use mapping_ab."""

    mapping_b: Optional[MappingParameter] = None
    """Optional override for B tile mapping. None -> use mapping_ab."""

    mapping_c: MappingParameter = field(default_factory=lambda: MappingParameter(
        weight_scaling_omega=1.0,
        weight_scaling_columnwise=True,
        learn_out_scaling=True,
        out_scaling_columnwise=True,
    ))
    """C tile mapping. Default: pretrained weight scaling + trainable columnwise out_scaling."""

    rank_chunk: Optional[int] = None
    """Number of ranks written per pulsed update in transfer_method='direct'
    (None = all selected ranks at once)."""

    columns_mode: bool = True
    """Not read by LRTTSimulatorTile; the one-hot transfer always reads A with
    forward and B with backward passes."""

    # === Devices ===
    unit_cell_devices: List[PulsedDevice] = field(default_factory=lambda: [
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0),
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0),
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0)
    ])
    """Devices of the [A, B, C] tiles. A FloatingPointDevice gives a
    FloatingPointTile, any other device an AnalogTile."""

    # === Bit-length management dictionaries ===
    ab_bl_mgmt: Optional[Dict[str, Any]] = None
    """Not read by LRTTSimulatorTile. The bit length of the A/B updates is set by
    rpu_config.update.desired_bl, a_desired_bl and b_desired_bl."""

    transfer_bl_mgmt: Optional[Dict[str, Any]] = None
    """Not read by LRTTSimulatorTile. The bit length of the transfer into C is
    set by c_desired_bl (default: rpu_config.update.desired_bl)."""

    # === Separate A/B scaling parameters ===
    # When use_manual_scaling=True in UpdateParameters, these override the global scales
    a_x_scaling: Optional[float] = None
    """X (input) scaling factor for A tile. If None, uses global manual_x_scaling."""

    a_d_scaling: Optional[float] = None
    """D (gradient) scaling factor for A tile. If None, uses global manual_d_scaling."""

    b_x_scaling: Optional[float] = None
    """X (input) scaling factor for B tile. If None, uses global manual_x_scaling."""

    b_d_scaling: Optional[float] = None
    """D (gradient) scaling factor for B tile. If None, uses global manual_d_scaling."""

    # === Separate bit lengths (BL) ===
    a_desired_bl: Optional[int] = None
    """Desired BL for A tile updates. If None, uses global desired_bl from UpdateParameters."""

    b_desired_bl: Optional[int] = None
    """Desired BL for B tile updates. If None, uses global desired_bl from UpdateParameters."""

    c_desired_bl: Optional[int] = None
    """Desired BL of the pulsed updates that write the transfer into the C tile.
    If None, uses global desired_bl from UpdateParameters."""

    # === Logging of the A/B update inputs ===
    log_ab_scaling: bool = False
    """Print the maxima of the inputs (x) and errors (d) of the A/B updates."""

    log_ab_scaling_every: int = 10
    """Log x,d max values every N steps (only when log_ab_scaling=True)."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate fast_lr and auto-scale
        if self.fast_lr <= 0:
            raise ValueError(f"fast_lr must be positive, got {self.fast_lr}")
        if self.auto_scale_mode not in ["none", "shared", "separate"]:
            raise ValueError(f"auto_scale_mode must be 'none', 'shared', or 'separate', got '{self.auto_scale_mode}'")
        if not (0.0 < self.auto_momentum < 1.0):
            raise ValueError(f"auto_momentum must be in (0, 1), got {self.auto_momentum}")
        if self.transfer_fast_lr_ref not in ["A", "B", "geomean"]:
            raise ValueError(f"transfer_fast_lr_ref must be 'A', 'B', or 'geomean', got '{self.transfer_fast_lr_ref}'")

        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
            
        # Validate transfer parameters
        if self.transfer_every <= 0:
            raise ValueError(f"transfer_every must be positive, got {self.transfer_every}")
            
        if self.transfer_lr <= 0:
            raise ValueError(f"transfer_lr must be positive, got {self.transfer_lr}")
            
        # Validate LoRA parameters
        if self.lora_alpha < 0:
            raise ValueError(f"lora_alpha must be non-negative, got {self.lora_alpha}")
            
        if self.reinit_gain < 0:
            raise ValueError(f"reinit_gain must be non-negative, got {self.reinit_gain}")

        # Validate reinit_mode
        valid_modes = ["standard", "decay", "hybrid"]
        if self.reinit_mode not in valid_modes:
            raise ValueError(f"reinit_mode must be one of {valid_modes}, got '{self.reinit_mode}'")

        # Validate num_reads
        if self.num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {self.num_reads}")

        # Validate multi_read_mode
        valid_read_modes = ["average", "per_read"]
        if self.multi_read_mode not in valid_read_modes:
            raise ValueError(f"multi_read_mode must be one of {valid_read_modes}, got '{self.multi_read_mode}'")

        # Validate dynamic TE parameters
        if self.dynamic_te_power <= 0:
            raise ValueError(f"dynamic_te_power must be positive, got {self.dynamic_te_power}")
        if self.dynamic_te_min is not None and self.dynamic_te_min <= 0:
            raise ValueError(f"dynamic_te_min must be positive or None, got {self.dynamic_te_min}")
        if self.dynamic_te_max is not None and self.dynamic_te_max <= 0:
            raise ValueError(f"dynamic_te_max must be positive or None, got {self.dynamic_te_max}")
        if (self.dynamic_te_min is not None and self.dynamic_te_max is not None
                and self.dynamic_te_min > self.dynamic_te_max):
            raise ValueError(f"dynamic_te_min ({self.dynamic_te_min}) must be <= dynamic_te_max ({self.dynamic_te_max})")

        # Validate read_n_avg
        if self.read_n_avg < 1:
            raise ValueError(f"read_n_avg must be >= 1, got {self.read_n_avg}")

        # Validate AGC settings
        if not (0 < self.agc_margin <= 1.0):
            raise ValueError(f"agc_margin must be in (0, 1], got {self.agc_margin}")
        if self.agc_max_iters < 1:
            raise ValueError(f"agc_max_iters must be >= 1, got {self.agc_max_iters}")

        # Validate two_amp_ratio
        if not (0 < self.two_amp_ratio < 1.0):
            raise ValueError(f"two_amp_ratio must be in (0, 1), got {self.two_amp_ratio}")

        # Validate update_mode
        valid_update_modes = ["lora", "reconstruction"]
        if self.update_mode not in valid_update_modes:
            raise ValueError(f"update_mode must be one of {valid_update_modes}, got '{self.update_mode}'")

        # Validate reconstruction parameters
        if self.recon_lambda_a < 0:
            raise ValueError(f"recon_lambda_a must be non-negative, got {self.recon_lambda_a}")
        if self.recon_lambda_b < 0:
            raise ValueError(f"recon_lambda_b must be non-negative, got {self.recon_lambda_b}")
        if not (0 < self.recon_ema_beta < 1):
            raise ValueError(f"recon_ema_beta must be in (0, 1), got {self.recon_ema_beta}")
        if self.recon_lr_scale <= 0:
            raise ValueError(f"recon_lr_scale must be positive, got {self.recon_lr_scale}")
        if self.recon_clip_norm <= 0:
            raise ValueError(f"recon_clip_norm must be positive, got {self.recon_clip_norm}")

        # Validate transfer_mode
        valid_transfer_modes = ["pilot", "sigma_delta", "off"]
        if self.transfer_mode not in valid_transfer_modes:
            raise ValueError(f"transfer_mode must be one of {valid_transfer_modes}, got '{self.transfer_mode}'")

        # Validate transfer_micro_steps
        if self.transfer_micro_steps < 1:
            raise ValueError(f"transfer_micro_steps must be >= 1, got {self.transfer_micro_steps}")

        # Validate transfer_pilot_frac
        if not (0.0 < self.transfer_pilot_frac < 1.0):
            raise ValueError(f"transfer_pilot_frac must be in (0, 1), got {self.transfer_pilot_frac}")

        # Validate sd_quantum
        if self.sd_quantum is not None and self.sd_quantum <= 0:
            raise ValueError(f"sd_quantum must be positive or None, got {self.sd_quantum}")

        # Validate transfer_rank_schedule
        valid_rank_schedules = ["all", "round_robin"]
        if self.transfer_rank_schedule not in valid_rank_schedules:
            raise ValueError(f"transfer_rank_schedule must be one of {valid_rank_schedules}, got '{self.transfer_rank_schedule}'")

        # Validate transfer_ranks_per_step
        if self.transfer_ranks_per_step < 1:
            raise ValueError(f"transfer_ranks_per_step must be >= 1, got {self.transfer_ranks_per_step}")
        if self.transfer_ranks_per_step > self.rank:
            raise ValueError(f"transfer_ranks_per_step ({self.transfer_ranks_per_step}) must be <= rank ({self.rank})")

        # Validate rank_chunk
        if self.rank_chunk is not None and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or None, got {self.rank_chunk}")
            
        # Validate unit cell devices
        if len(self.unit_cell_devices) != 3:
            raise ValueError(f"Must provide exactly 3 unit_cell_devices for [fastA, fastB, visible], got {len(self.unit_cell_devices)}")
            
        # Set default rank_chunk
        if self.rank_chunk is None:
            self.rank_chunk = self.rank
            
        # Initialize BL management if not provided
        if self.ab_bl_mgmt is None:
            self.ab_bl_mgmt = {}
        if self.transfer_bl_mgmt is None:
            self.transfer_bl_mgmt = {}
    
    def get_device_for_tile(self, tile_type: str) -> PulsedDevice:
        """Get device configuration for specific tile type.
        
        Args:
            tile_type: 'fastA', 'fastB', or 'visible'
            
        Returns:
            Device configuration for the specified tile
        """
        tile_map = {'fastA': 0, 'fastB': 1, 'visible': 2}
        if tile_type not in tile_map:
            raise ValueError(f"Unknown tile_type '{tile_type}', must be one of {list(tile_map.keys())}")
            
        return self.unit_cell_devices[tile_map[tile_type]]
    
    def to_controller_kwargs(self) -> Dict[str, Any]:
        """Convert to LRTTController constructor arguments.

        Returns:
            Dictionary of arguments for LRTTController.__init__(). The entry
            '_post_init' holds the settings that LRTTSimulatorTile assigns as
            controller attributes after construction.
        """
        kwargs = {
            'transfer_lr': self.transfer_lr,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'reinit_mode': self.reinit_mode,
            'correct_gradient_magnitudes': self.correct_gradient_magnitudes,
            'fast_lr': self.fast_lr,
            'scale_transfer_lr': self.scale_transfer_lr,
            'transfer_fast_lr_ref': self.transfer_fast_lr_ref,
            'rank_chunk': self.rank_chunk,
            'ab_bl_mgmt': self.ab_bl_mgmt,
            'transfer_bl_mgmt': self.transfer_bl_mgmt,
            'forward_inject': self.forward_inject,
            'dynamic_te': self.dynamic_te,
            'dynamic_te_power': self.dynamic_te_power,
            'dynamic_te_min': self.dynamic_te_min,
            'dynamic_te_max': self.dynamic_te_max,
            'te_warmup_schedule': self.te_warmup_schedule,
            'te_warmup_steps': self.te_warmup_steps,
            'num_reads': self.num_reads,
            'multi_read_mode': self.multi_read_mode,
            'update_mode': self.update_mode,
            'transfer_method': self.transfer_method,
            'transfer_rank_schedule': self.transfer_rank_schedule,
            'transfer_ranks_per_step': self.transfer_ranks_per_step,
            'fi_continuous_alpha': self.fi_continuous_alpha,
        }
        # Post-init settings (set on controller after creation)
        kwargs['_post_init'] = {
            # Auto-scale settings
            'auto_scale_mode': self.auto_scale_mode,
            'auto_momentum': self.auto_momentum,
            # Transfer mode & calibration
            'transfer_mode': self.transfer_mode,
            'transfer_micro_steps': self.transfer_micro_steps,
            'transfer_pilot_frac': self.transfer_pilot_frac,
            'sd_quantum': self.sd_quantum,
            # Read noise reduction
            'read_n_avg': self.read_n_avg,
            'differential_read': self.differential_read,
            # AGC settings
            'agc_enabled': self.agc_enabled,
            'agc_margin': self.agc_margin,
            'agc_max_iters': self.agc_max_iters,
            # Two-amplitude settings
            'two_amp_enabled': self.two_amp_enabled,
            'two_amp_ratio': self.two_amp_ratio,
            # Reconstruction parameters
            'recon_lambda_a': self.recon_lambda_a,
            'recon_lambda_b': self.recon_lambda_b,
            'recon_use_scalar_stabilizer': self.recon_use_scalar_stabilizer,
            'recon_use_exact_gram': self.recon_use_exact_gram,
            'recon_exact_gram_every': self.recon_exact_gram_every,
            'recon_ema_beta': self.recon_ema_beta,
            'recon_lr_scale': self.recon_lr_scale,
            'recon_clip_norm': self.recon_clip_norm,
            'recon_use_clip_norm': self.recon_use_clip_norm,
            # Separate A/B scaling
            'a_x_scaling': self.a_x_scaling,
            'a_d_scaling': self.a_d_scaling,
            'b_x_scaling': self.b_x_scaling,
            'b_d_scaling': self.b_d_scaling,
            # Logging of the A/B update inputs
            'log_ab_scaling': self.log_ab_scaling,
            'log_ab_scaling_every': self.log_ab_scaling_every,
            # Separate BL for C tile
            'c_desired_bl': self.c_desired_bl,
        }
        return kwargs
    
    @classmethod
    def from_attributes(cls, source) -> 'PythonLRTTDevice':
        """Create a PythonLRTTDevice from the LR-TT attributes of another object.

        Args:
            source: any object with rank, transfer_every, transfer_lr,
                lora_alpha, ... attributes (missing ones take the defaults below)

        Returns:
            PythonLRTTDevice with these attributes
        """
        return cls(
            rank=getattr(source, 'rank', 4),
            transfer_every=getattr(source, 'transfer_every', 32),
            transfer_lr=getattr(source, 'transfer_lr', 1.0),
            lora_alpha=getattr(source, 'lora_alpha', 1.0),
            reinit_gain=getattr(source, 'reinit_gain', 1.0),
            units_in_mbatch=getattr(source, 'units_in_mbatch', False),
            correct_gradient_magnitudes=getattr(source, 'correct_gradient_magnitudes', False),
            forward_inject=getattr(source, 'forward_inject', False),
            rank_chunk=getattr(source, 'rank_chunk', None),
            unit_cell_devices=getattr(source, 'unit_cell_devices', [ConstantStepDevice()] * 3)
        )


@dataclass
class PythonLRTTPreset(_PrintableMixin):
    """Example device combinations for PythonLRTTDevice.

    The experiment scripts of this repository build their devices directly
    and do not use these presets.
    """

    @staticmethod
    def idealized(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTDevice:
        """LR-TT with aihwkit's IdealizedPresetDevice (symmetric ConstantStepDevice) for all tiles.

        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            lora_alpha: LoRA scaling factor
            
        Returns:
            Idealized PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        
        ideal_device = IdealizedPresetDevice()
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[ideal_device, ideal_device, ideal_device]
        )
    
    @staticmethod
    def constant_step(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTDevice:
        """LRTT with ConstantStepDevice for all tiles.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency  
            dw_min: Minimum weight update step
            
        Returns:
            ConstantStep PythonLRTTDevice configuration
        """
        device = ConstantStepDevice(
            dw_min=dw_min,
            dw_min_dtod=0.0,
            up_down_dtod=0.0,
            w_min=-1.0,
            w_max=1.0
        )
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[device, device, device]
        )
    
    @staticmethod  
    def lora_style(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTDevice:
        """LR-TT with frequent transfers (defaults: rank 8, lora_alpha 16, transfer_every 1).

        Uses correct_gradient_magnitudes=True and IdealizedPresetDevice for all tiles.

        Args:
            rank: LoRA rank
            lora_alpha: LoRA alpha
            transfer_every: Transfer frequency (1 = every step)

        Returns:
            PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=1.0,
            forward_inject=False,
            correct_gradient_magnitudes=True,  # transfer LR divided by the A/B learning rate
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(), IdealizedPresetDevice()]
        )
    
    @staticmethod
    def mixed_precision(rank: int = 4, transfer_every: int = 16) -> PythonLRTTDevice:
        """Mixed precision: fine-step C tile, coarse-step A/B tiles.

        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency

        Returns:
            Mixed precision PythonLRTTDevice configuration
        """
        # A/B: dw_min 0.05, range [-0.8, 0.8]
        low_precision = ConstantStepDevice(dw_min=0.05, w_min=-0.8, w_max=0.8)

        # C: dw_min 0.001, range [-2, 2]
        high_precision = ConstantStepDevice(dw_min=0.001, w_min=-2.0, w_max=2.0)
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[low_precision, low_precision, high_precision]
        )
    
    @staticmethod
    def inference_optimized(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTDevice:
        """LR-TT with transfer_every 1 and reinit_gain 0 (defaults: rank 2, lora_alpha 0.5).

        Forward injection is off and all tiles use IdealizedPresetDevice.

        Args:
            rank: LoRA rank
            lora_alpha: LoRA alpha

        Returns:
            PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=1,  # transfer after every update
            lora_alpha=lora_alpha,
            reinit_gain=0.0,  # Kaiming draws scaled by 0
            forward_inject=False,  # forward pass through C only
            columns_mode=True,  # not read by the tile
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(), IdealizedPresetDevice()]
        )

    @staticmethod
    def floating_ab_softbound_c(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dw_min: float = 0.001,
        lifetime: float = 0.0,
    ) -> 'PythonLRTTDevice':
        """LRTT with FloatingPoint A/B tiles and SoftBounds C tile.

        - A, B tiles: FloatingPointDevice (exact arithmetic, no noise)
        - C tile: SoftBoundsReferenceDevice (noise=0, w_max=1, w_min=-1)

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dw_min: Minimum weight update step for C tile
            lifetime: Retention lifetime for C tile (0 = no decay)

        Returns:
            PythonLRTTDevice configuration with FloatingPoint A/B and SoftBounds C
        """
        from aihwkit.simulator.configs.devices import (
            FloatingPointDevice,
            SoftBoundsReferenceDevice,
        )

        # SoftBounds C tile with all noise sources disabled
        c_device = SoftBoundsReferenceDevice(
            # Weight bounds
            w_max=1.0,
            w_min=-1.0,
            dw_min=dw_min,

            # All noise terms set to zero
            dw_min_std=0.0,         # cycle-to-cycle noise
            write_noise_std=0.0,    # write noise
            diffusion=0.0,          # diffusion noise

            # Device-to-device variation = 0
            dw_min_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            up_down=0.0,
            up_down_dtod=0.0,

            # Lifetime (configurable)
            lifetime=lifetime,
            lifetime_dtod=0.0,

            # Slope variations = 0
            slope_up_dtod=0.0,
            slope_down_dtod=0.0,
        )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[
                FloatingPointDevice(),  # A: exact arithmetic
                FloatingPointDevice(),  # B: exact arithmetic
                c_device,               # C: SoftBounds (noise=0)
            ]
        )

    @staticmethod
    def sixt1c_ab(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True,
        c_device: Optional[PulsedDevice] = None,
        reinit_mode: str = "decay",
    ) -> 'PythonLRTTDevice':
        """LR-TT with 6T1C devices for the A/B tiles and a configurable C tile.

        The A/B tiles use a LinearStepDevice model of the 6T1C (six transistors,
        one capacitor) cell; the C tile can use any device.

        6T1C model (A/B tiles):
            - weight range [-1, 1] with dw_min = 0.001981 (about 1000 states)
            - gamma_up = -0.1678, gamma_down = +0.1410
            - device-to-device variation: dw_min 0.1, up_down 0.01,
              w_max/w_min 0.05, gamma_up/gamma_down 0.05
            - cycle-to-cycle variation: dw_min_std = 0.3, write_noise_std = 0.0182
            - capacitor retention: exponential decay toward 0 with time constant
              tau = 775.1 min (46505 s)

        The LR-TT scripts of the BERT-base and MNIST experiments do not use this
        preset; they define their 6T1C-based devices directly, with
        write_noise_std = 0 (with retention in the MNIST scripts, without
        retention, tau_sec = 0, in the BERT-base scripts).

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds (for 6T1C retention)
            include_retention: Whether to include retention effects for 6T1C
            c_device: Device for C tile. If None, IdealizedPresetDevice with
                      dw_min=0.001, dw_min_std=0 and dw_min_dtod=0.
                      Can be any PulsedDevice: IdealizedPresetDevice, PCM, RRAM, etc.
            reinit_mode: Reinit strategy after transfer ('standard', 'decay', 'hybrid').
                         Default 'decay' (A and B kept across transfers).

        Returns:
            PythonLRTTDevice configuration with 6T1C A/B and custom C device

        Example:
            >>> from aihwkit.simulator.presets.devices import PCMPresetDevice, ReRamESPresetDevice
            >>> # 6T1C A/B with PCM C tile
            >>> device = PythonLRTTPreset.sixt1c_ab(c_device=PCMPresetDevice())
            >>> # 6T1C A/B with RRAM C tile
            >>> device = PythonLRTTPreset.sixt1c_ab(c_device=ReRamESPresetDevice())
        """
        import math

        # Calculate lifetime from physical tau for 6T1C
        TAU_SEC = 46505.0  # Physical time constant: 775.1 min = 46505 sec
        if include_retention and dt_batch_sec > 0:
            delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
            lifetime = 1.0 / delta
        else:
            lifetime = 0.0  # No retention

        # Create 6T1C device for A/B tiles (LinearStepDevice)
        sixt1c_device = LinearStepDevice(
            # Core update parameters (fitted from 6T1C data)
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            mult_noise=False,

            # Device-to-device variation
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,

            # Cycle-to-cycle variation
            dw_min_std=0.3,
            write_noise_std=0.0182,

            # LinearStepDevice specific
            mean_bound_reference=True,

            # Retention (capacitor leakage)
            lifetime=lifetime,
            lifetime_dtod=0.1 if include_retention else 0.0,
            reset=0.0,  # Decay toward 0V
            reset_dtod=0.0,
        )

        # C tile device: the given device, or an idealized device with dw_min=0.001
        # and no cycle-to-cycle or device-to-device variation of the step size
        if c_device is None:
            from aihwkit.simulator.presets.devices import IdealizedPresetDevice
            c_device = IdealizedPresetDevice(
                dw_min=0.001,
                dw_min_std=0.0,
                dw_min_dtod=0.0,
            )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=1.0,
            reinit_mode=reinit_mode,
            forward_inject=False,
            unit_cell_devices=[sixt1c_device, sixt1c_device, c_device]
        )

    @staticmethod
    def sixt1c_ab_pcm(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and PCM C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and PCM C
        """
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=PCMPresetDevice()
        )

    @staticmethod
    def sixt1c_ab_rram(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and RRAM C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and RRAM C
        """
        from aihwkit.simulator.presets.devices import ReRamESPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=ReRamESPresetDevice()
        )

    @staticmethod
    def sixt1c_ab_ideal(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and Idealized C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and Idealized C
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=IdealizedPresetDevice()
        )

    @staticmethod
    def sixt1c_all(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C devices for ALL tiles (A, B, and C).

        All three tiles use identical 6T1C device characteristics.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds (for retention)
            include_retention: Whether to include retention effects

        Returns:
            PythonLRTTDevice configuration with 6T1C for all tiles
        """
        import math

        # Calculate lifetime from physical tau
        TAU_SEC = 46505.0
        if include_retention and dt_batch_sec > 0:
            delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
            lifetime = 1.0 / delta
        else:
            lifetime = 0.0

        # Create 6T1C device (same for A, B, C)
        sixt1c_device = LinearStepDevice(
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            mult_noise=False,
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,
            dw_min_std=0.3,
            write_noise_std=0.0182,
            mean_bound_reference=True,
            lifetime=lifetime,
            lifetime_dtod=0.1 if include_retention else 0.0,
            reset=0.0,
            reset_dtod=0.0,
        )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[sixt1c_device, sixt1c_device, sixt1c_device]
        )


# =============================================================================
# 6T1C Device Utility Functions
# =============================================================================

def get_6t1c_lifetime_for_dt_batch(dt_batch_sec: float) -> float:
    """Calculate AIHWKit lifetime parameter for 6T1C given dt_batch.

    The 6T1C capacitor has a physical time constant tau = 46505 seconds (775.1 min).
    For one update every dt_batch_sec seconds the per-update lifetime is
    1 / (1 - exp(-dt_batch_sec / tau)).

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds.

    Returns:
        Lifetime parameter for AIHWKit configuration.

    Example:
        >>> # For 1 second per batch
        >>> lifetime = get_6t1c_lifetime_for_dt_batch(1.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 46506

        >>> # For 1 minute per batch
        >>> lifetime = get_6t1c_lifetime_for_dt_batch(60.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 776
    """
    import math
    TAU_SEC = 46505.0  # Physical time constant in seconds
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    return 1.0 / delta