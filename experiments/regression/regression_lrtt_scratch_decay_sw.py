#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LRTT training from scratch on a 9x9 multi-output linear regression task.

Core-state trajectory analysis of Fig. 4d,e and Supplementary Fig. S5.
Default settings: 6T1C A/B tiles, SoftBounds core C, one-hot rank-wise
transfer. run_multi_param_experiments() trains several configurations on the
same problem and projects their C trajectories with PCA; with
save_figures = True the __main__ block runs the asymmetry sweeps of
Supplementary Fig. S5.
"""

import os
import json
from datetime import datetime
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from aihwkit.optim import AnalogSGD, AnalogAdam
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import wandb

# AIHWKit imports
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice, LinearStepDevice, SoftBoundsReferenceDevice
from aihwkit.simulator.configs import MappingParameter, SingleRPUConfig, UpdateParameters
from aihwkit.simulator.parameters import PulseType

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# Configuration
# ============================================================================
class ScratchExperimentConfig:
    """Configuration for the LRTT scratch experiment."""

    # Random seeds
    seeds = [42, 1, 2, 3, 4, 5]
    primary_seed = 42

    # Data dimensions
    input_dim = 9
    output_dim = 9

    # Dataset sizes
    D_prime_train_size = 200
    D_prime_test_size = 200

    # Noise parameters
    noise_std = 0.02

    # Input data type
    # Options: 'continuous', 'ternary' (0, 0.5, 1), 'binary' (0, 1)
    input_type = 'ternary'  # 'continuous' (U[-1, 1]), 'ternary' {0, 0.5, 1} or 'binary' {0, 1}

    custom_input_file = None

    # Complexity levels to test
    # Options: 'simple' (0.5), 'medium' (0.8), 'complex' (1.0)
    complexity_levels = ['medium']

    # Training hyperparameters - LRTT scratch training
    lrtt_epochs = 2000
    lrtt_batch_size = 1
    lrtt_patience = 7  # Early-stopping patience, counted in epochs that contain a transfer
    lrtt_grad_clip = 2.0  # Gradient-norm clipping

    # Scaling mode
    # True: fixed x/d pulse scaling (a/b_x/d_scaling), independent of lrtt_lr
    # False: dynamic scaling with update_management, lrtt_lr sets the update size
    # In both modes lrtt_lr also multiplies the transfer rate (scale_transfer_lr=True)
    use_manual_scaling = False
    lrtt_lr = 0.2852  # Optimizer learning rate

    # LRTT configuration
    lrtt_rank = 4  # Rank r of the auxiliary factors
    lrtt_transfer_every = 116  # Transfer period (updates)
    lora_alpha = 12.89  # Transfer rate (transfer_lr)

    # Reinit configuration
    # Options:
    #   "standard"        - A and B re-initialized (a_init_mode, b_init_mode) after every transfer
    #   "decay"           - A and B initialized once and kept after a transfer (only retention decay acts)
    #   "hybrid"          - A reset to zero after every transfer, B kept
    reinit_mode = "decay"

    # A matrix initialization mode
    # Options: 'zero' (LoRA-style, dW=0 initially), 'kaiming' (random Kaiming initialization)
    a_init_mode = 'zero'

    # B matrix initialization mode
    # Options: 'kaiming' (standard LoRA initialization), 'zero' (dW=0 initially)
    b_init_mode = 'kaiming'

    # C matrix initialization value
    # All elements of C are initialized to this value (must be within [-1, 1] for analog tile)
    c_init_value = 0

    # Device configuration
    # Use 6T1C device for A/B matrices (capacitor-based with retention decay)
    # False: IdealizedPresetDevice (idealized, noise only)
    USE_6T1C_AB = True

    # C matrix device configuration
    # 'idealized': IdealizedPresetDevice (idealized, noise only)
    # 'floating_point': FloatingPointDevice (idealized, no quantization)
    # 'softbounds': SoftBoundsReferenceDevice (realistic analog with bounds)
    # 'linearstep': LinearStepDevice (configurable gamma nonlinearity)
    c_device_type = 'softbounds'

    # Transfer method for C update
    # 'set': Exact weight setting (no pulsed update, precise)
    # 'onehot': One-hot transfer (analog-realistic pulsed update)
    # 'direct': Direct transfer (matrix multiply, pulsed update)
    transfer_method = 'onehot'

    # Transfer rank scheduling
    # 'all': Transfer all ranks at once (default)
    # 'round_robin': Cycle through ranks, transferring a subset each time
    transfer_rank_schedule = 'all'
    transfer_ranks_per_step = 1  # Number of ranks per transfer in round_robin mode

    # C device parameters (for pulsed transfer methods: onehot, direct)
    # Default dw_min by device type:
    #   - idealized (IdealizedPresetDevice): 0.0002
    #   - softbounds (SoftBoundsReferenceDevice): 0.001
    #   - floating_point (FloatingPointDevice): N/A (exact, no pulse)
    ab_up_down = 0.0  # A/B tile up/down asymmetry (0 = symmetric)
    ab_gamma_up = -0.1678  # A/B tile gamma_up (nonlinearity, from 6T1C fitting)
    ab_gamma_down = 0.1410  # A/B tile gamma_down (nonlinearity, from 6T1C fitting)
    ab_dw_min = 0.001981  # A/B tile dw_min (from 6T1C fitting)
    c_w_max = 1.0  # Maximum weight value
    c_w_min = -1.0  # Minimum weight value
    c_dw_min = 0.000804  # C tile dw_min
    c_desired_bl = 10  # Bit length for C transfer (higher for accuracy)

    # SoftBoundsReferenceDevice / LinearStepDevice additional parameters
    # Asymmetry
    c_up_down = 0.0  # up/down asymmetry (0 = symmetric)
    c_up_down_dtod = 0.01  # up/down asymmetry dtod variation
    # Nonlinearity (LinearStepDevice only)
    c_gamma_up = -0.1678  # C tile gamma_up (default: same as AB)
    c_gamma_down = 0.1410  # C tile gamma_down (default: same as AB)

    # Retention configuration (batch units, converted to pulse units internally)
    # create_6t1c_device() converts the batch lifetime to a pulse lifetime;
    # the training loop calls decay_weights() desired_bl times per batch
    # (1 - 1/lifetime_pulse)^desired_bl = (1 - 1/lifetime_batch)
    # e.g. lifetime=24.89 keeps 1 - 1/24.89 = 96% of the A/B weights per batch
    lifetime = 24.89  # A tile lifetime in batch units (0 = no decay)
    b_lifetime = None  # B tile lifetime (None = same as lifetime, 0 = no decay)
    include_retention = True  # Enable/disable retention effects

    # I/O configuration
    is_perfect = True  # True = ideal fp forward/backward (bypass IO noise/quantization)

    # Pulse/Update configuration (Hardware-realistic settings)
    desired_bl = 10  # Bit length for A/B updates (pulse train length)
    pulse_type = PulseType.STOCHASTIC_COMPRESSED  # Pulse generation type
    # Options: NONE (fp update), STOCHASTIC_COMPRESSED (stochastic, single pass),
    #          STOCHASTIC (two pass, CPU only), NONE_WITH_DEVICE (fp + device effects),
    #          MEAN_COUNT (prob-based p_a*p_b), DETERMINISTIC_IMPLICIT (deterministic BL*x_q*d_q)

    # Manual scaling factors (used when use_manual_scaling=True)
    # x_scaling: applied to input x (B factor in aihwkit)
    # d_scaling: applied to gradient d (A factor in aihwkit)
    x_scaling = None  # Input (x) scaling factor (global default)
    d_scaling = None  # Gradient (d) scaling factor (global default)

    # Separate A/B tile scaling factors (override global if set)
    # A tile update: x=XB (B projection of input), d=original gradient
    # B tile update: x=original input, d=DA (A^T projection of gradient)
    a_x_scaling = 0.604  # A tile x scaling
    a_d_scaling = 0.343  # A tile d scaling
    b_x_scaling = 1.0   # B tile x scaling (None = use global x_scaling)
    b_d_scaling = 0.419  # B tile d scaling

    # Debug logging for A/B scaling
    log_ab_scaling = True  # Enable x,d max value logging
    log_ab_scaling_every = 10  # Log every N steps

    # Quantization of the recorded x and d values (pulse vectors in the step-wise
    # history and the Excel export); the tile I/O is set by is_perfect
    quantize_x = False  # Enable input quantization
    quantize_d = False  # Enable gradient quantization
    x_resolution = 0.1  # Input quantization resolution (e.g., 0.1 = one decimal place)
    d_resolution = 0.1  # Gradient quantization resolution (e.g., 0.1 = one decimal place)

    # Output options
    # True: __main__ runs the asymmetry sweeps, and run_scratch_experiment()
    # saves the training figures as PNG
    save_figures = False

    # Results directory
    results_dir = "results/lrtt_scratch_decay"

# ============================================================================
# Device Configuration
# ============================================================================

def create_6t1c_device(lifetime=0, transfer_every=10, include_retention=True, desired_bl=10, dw_min=0.001981, up_down=0.0, gamma_up=-0.1678, gamma_down=0.1410):
    """Create 6T1C device for A/B tiles.

    6T1C Device Characteristics:
        - Step dw_min = 0.001981 with bounds +-1 (the 6T1C fit): about 1,000
          conductance states across the weight range
        - Capacitor-based weight storage with exponential decay

    Args:
        lifetime: Lifetime in BATCH units. Decay per batch = (1 - 1/lifetime).
                  0 = no decay (perfect retention)
        transfer_every: Number of batches between transfers (for display only)
        include_retention: Whether to include retention effects
        desired_bl: Number of pulses per batch. Used to convert batch lifetime
                    to pulse lifetime so decay is applied per-pulse in post_update_step().

    Note:
        Lifetime is specified in BATCH units but internally converted to PULSE units.
        The training loop calls decay_weights() desired_bl times per batch
        (1 from post_update_step + desired_bl-1 extra calls after optimizer.step).
        Per-pulse lifetime satisfies: (1 - 1/lifetime_pulse)^desired_bl = (1 - 1/lifetime_batch)
    """
    import math

    if not include_retention or lifetime <= 0:
        lifetime_pulse = 0.0
        print(f"  6T1C retention: DISABLED (perfect retention)")
    else:
        # Convert batch lifetime to pulse lifetime
        # (1 - 1/lifetime_pulse)^desired_bl = (1 - 1/lifetime_batch)
        decay_per_batch = 1.0 - 1.0 / lifetime
        decay_per_pulse = math.pow(decay_per_batch, 1.0 / desired_bl)
        lifetime_pulse = 1.0 / (1.0 - decay_per_pulse)

        # Calculate retention at transfer for display
        retention_at_transfer = math.pow(decay_per_batch, transfer_every)
        print(f"  6T1C retention: lifetime={lifetime:.1f} batches -> lifetime_pulse={lifetime_pulse:.1f} pulses (desired_bl={desired_bl})")
        print(f"    -> {retention_at_transfer*100:.1f}% after {transfer_every} batches")

    return LinearStepDevice(
        # Core update parameters (fitted from 6T1C data)
        dw_min=dw_min,
        up_down=up_down,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        mult_noise=True,

        # Device-to-device variation
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,

        # Cycle-to-cycle variation
        dw_min_std=0.3,
        write_noise_std=0,  # no apparent write noise

        # LinearStepDevice specific
        mean_bound_reference=True,

        # Retention (capacitor leakage), lifetime in pulse units
        lifetime=lifetime_pulse,
        lifetime_dtod=0.1 if include_retention else 0.0
    )


# ============================================================================
# Data Generation (simplified for scratch training)
# ============================================================================

def generate_target_matrix(complexity_level: str, config: ScratchExperimentConfig, seed: int = 42) -> torch.Tensor:
    """Generate the target matrix T' with values in [-1, 1] range."""
    torch.manual_seed(seed)  # Use consistent seed for T' generation
    complexity_scales = {
        'simple': 0.5,    # Simple target function
        'medium': 0.8,    # Medium complexity target function
        'complex': 1.0    # Complex target function (full range)
    }
    scale = complexity_scales.get(complexity_level, 0.8)

    # Generate uniform random values in [-scale, scale] range
    T_prime = (torch.rand(config.output_dim, config.input_dim) * 2 - 1) * scale
    return T_prime


def generate_target_dataset(complexity_level: str, config: ScratchExperimentConfig,
                           train: bool = True, seed: int = 42) -> TensorDataset:
    """Generate dataset with target function T' for scratch training.

    No baseline needed - T' is the actual target function to learn.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed + 1000 if train else seed + 2000)

    # Check if custom input file is specified
    if config.custom_input_file is not None:
        # Load inputs from custom file
        import numpy as np
        custom_path = config.custom_input_file
        if not os.path.isabs(custom_path):
            # Make relative path relative to this script's directory
            custom_path = os.path.join(os.path.dirname(__file__), custom_path)

        # Load CSV data
        data = np.loadtxt(custom_path, delimiter=',')
        X_all = torch.tensor(data, dtype=torch.float32)

        # Update input_dim based on file (only on first call)
        actual_input_dim = X_all.shape[1]
        if config.input_dim != actual_input_dim:
            print(f"[Custom Input] Adjusting input_dim: {config.input_dim} -> {actual_input_dim}")
            config.input_dim = actual_input_dim
            config.output_dim = actual_input_dim  # Keep square matrix

        # Custom input: use all data for training only (no test split, no shuffle)
        X = X_all  # Keep original order

        # For custom input, return None for test dataset
        if not train:
            return None
    else:
        # Generate inputs based on input_type configuration
        size = config.D_prime_train_size if train else config.D_prime_test_size

        if config.input_type == 'ternary':
            # Generate inputs from {0, 0.5, 1}
            X = torch.randint(0, 3, (size, config.input_dim)).float() * 0.5
        elif config.input_type == 'binary':
            # Generate inputs from {0, 1}
            X = torch.randint(0, 2, (size, config.input_dim)).float()
        else:  # continuous (default)
            # Generate inputs from uniform distribution
            X = torch.rand(size, config.input_dim) * 2 - 1  # U([-1, 1]^input_dim)

    # Generate target matrix T' directly based on complexity level
    T_prime = generate_target_matrix(complexity_level, config, seed)

    # Generate labels with target transition matrix
    Y = X @ T_prime.T
    Y += torch.randn_like(Y) * config.noise_std  # Add output noise

    # Print target statistics for first call only
    if train and seed == 42:
        print(f"[Target T'] Complexity level: {complexity_level}, ||T'||: {T_prime.norm():.3f}")
        if config.custom_input_file:
            print(f"[Input Data] Custom file: {config.custom_input_file}, Shape: {X.shape}")
            print(f"[Input Data] Sample: {X[0].tolist()}")
        else:
            print(f"[Input Data] Type: {config.input_type}, Shape: {X.shape}")
            if config.input_type == 'ternary':
                print(f"[Input Data] Values: {0, 0.5, 1}, Sample: {X[0].tolist()[:4]}")
            elif config.input_type == 'binary':
                print(f"[Input Data] Values: {0, 1}, Sample: {X[0].tolist()[:4]}")
            else:
                print(f"[Input Data] Range: [-1, 1], Sample: {X[0].tolist()[:4]}")

    return TensorDataset(X, Y)


# ============================================================================
# Model Definitions
# ============================================================================
class LRTTModel(nn.Module):
    """LRTT model for scratch training."""

    def __init__(self, config: ScratchExperimentConfig, pretrained_C: torch.Tensor = None, seed: int = 42):
        super().__init__()

        # Create LRTT configuration
        from aihwkit.simulator.configs import IOParameters
        from aihwkit.simulator.parameters import WeightNoiseType, BoundManagementType, NoiseManagementType

        # Select devices for A/B tiles
        b_lifetime = config.b_lifetime if config.b_lifetime is not None else config.lifetime
        if config.USE_6T1C_AB:
            a_device = create_6t1c_device(
                lifetime=config.lifetime,
                transfer_every=config.lrtt_transfer_every,
                include_retention=config.include_retention,
                desired_bl=config.desired_bl,
                dw_min=config.ab_dw_min,
                up_down=config.ab_up_down,
                gamma_up=config.ab_gamma_up,
                gamma_down=config.ab_gamma_down
            )
            b_device = create_6t1c_device(
                lifetime=b_lifetime,
                transfer_every=config.lrtt_transfer_every,
                include_retention=config.include_retention,
                desired_bl=config.desired_bl,
                dw_min=config.ab_dw_min,
                up_down=config.ab_up_down,
                gamma_up=config.ab_gamma_up,
                gamma_down=config.ab_gamma_down
            )
        else:
            a_device = IdealizedPresetDevice()
            b_device = IdealizedPresetDevice()
            print("Using IdealizedPresetDevice for A/B matrices")

        # Create C device based on configuration
        if config.c_device_type == 'softbounds':
            c_device = SoftBoundsReferenceDevice(
                # Weight bounds (configurable)
                w_max=config.c_w_max,
                w_min=config.c_w_min,
                dw_min=config.c_dw_min,
                # Asymmetry (configurable)
                up_down=config.c_up_down,
                up_down_dtod=config.c_up_down_dtod,
                # Device-to-device variations (hardcoded)
                w_max_dtod=0.0,
                w_min_dtod=0.0,
                dw_min_dtod=0.0,
                dw_min_std=0.0,
                # Noise (hardcoded)
                write_noise_std=0.0,
                diffusion=0.0,
                # Lifetime (hardcoded)
                lifetime=0.0,
                lifetime_dtod=0.0,
                # Slope variations (hardcoded)
                slope_up_dtod=0.0,
                slope_down_dtod=0.0,
            )
            print(f"Using SoftBoundsReferenceDevice for C matrix:")
            print(f"  Bounds: w_min={config.c_w_min}, w_max={config.c_w_max}, dw_min={config.c_dw_min}")
            print(f"  Asymmetry: up_down={config.c_up_down}, up_down_dtod={config.c_up_down_dtod}")
        elif config.c_device_type == 'linearstep':
            c_device = LinearStepDevice(
                dw_min=config.c_dw_min,
                up_down=config.c_up_down,
                w_max=config.c_w_max,
                w_min=config.c_w_min,
                gamma_up=config.c_gamma_up,
                gamma_down=config.c_gamma_down,
                mult_noise=True,
                # Device-to-device variation
                dw_min_dtod=0.1,
                up_down_dtod=config.c_up_down_dtod,
                w_max_dtod=0.0,
                w_min_dtod=0.0,
                gamma_up_dtod=0.05,
                gamma_down_dtod=0.05,
                # Cycle-to-cycle variation
                dw_min_std=0.3,
                write_noise_std=0.0,
            )
            print(f"Using LinearStepDevice for C matrix:")
            print(f"  Bounds: w_min={config.c_w_min}, w_max={config.c_w_max}, dw_min={config.c_dw_min}")
            print(f"  Gamma: up={config.c_gamma_up}, down={config.c_gamma_down}")
            print(f"  Asymmetry: up_down={config.c_up_down}")
        elif config.c_device_type == 'floating_point':
            c_device = FloatingPointDevice()
            print("Using FloatingPointDevice for C matrix (idealized, no quantization)")
        else:  # 'idealized' (default)
            c_device = IdealizedPresetDevice(
                dw_min=config.c_dw_min,
                dw_min_dtod=0.0,  # No device-to-device variation
                dw_min_std=0.0,   # No cycle-to-cycle variation
                w_max_dtod=0.0,   # No weight bound variation
                w_min_dtod=0.0,   # No weight bound variation
            )
            print(f"Using IdealizedPresetDevice for C matrix (dw_min={config.c_dw_min}, all dtod=0)")

        device_config = PythonLRTTDevice(
            rank=config.lrtt_rank,
            transfer_every=config.lrtt_transfer_every,
            lora_alpha=config.lora_alpha,
            reinit_mode=config.reinit_mode,
            a_init_mode=config.a_init_mode,  # A initialization mode
            b_init_mode=config.b_init_mode,  # B initialization mode
            forward_inject=False,
            correct_gradient_magnitudes=False,
            transfer_micro_steps=1,  # Microsteps for smoother transfer (1=off)
            differential_read=False,  # Use +e/-e differential read
            unit_cell_devices=[
                a_device,  # A matrix
                b_device,  # B matrix
                c_device,   # C matrix
            ],
            # Separate A/B tile scaling factors
            a_x_scaling=config.a_x_scaling,
            a_d_scaling=config.a_d_scaling,
            b_x_scaling=config.b_x_scaling,
            b_d_scaling=config.b_d_scaling,
            # Debug logging
            log_ab_scaling=config.log_ab_scaling,
            log_ab_scaling_every=config.log_ab_scaling_every,
            # Separate BL for C tile transfer
            c_desired_bl=config.c_desired_bl,
            # Transfer method for C update (set, onehot, or direct)
            transfer_method=config.transfer_method,
            # Transfer rank scheduling
            transfer_rank_schedule=config.transfer_rank_schedule,
            transfer_ranks_per_step=config.transfer_ranks_per_step,
        )

        print(f"A initialization mode: {config.a_init_mode}")
        print(f"B initialization mode: {config.b_init_mode}")

        device_config.transfer_lr = device_config.lora_alpha

        # I/O configuration
        if config.is_perfect:
            forward_io = IOParameters(is_perfect=True)
            backward_io = IOParameters(is_perfect=True)
        else:
            forward_io = IOParameters(
                inp_res=0.007937,
                inp_bound=1.0,
                out_res=0.001961,
                out_bound=12.0,
                out_noise=0.06
            )
            backward_io = IOParameters(
                inp_res=0.007937,
                inp_bound=1.0,
                out_res=0.001961,
                out_bound=12.0,
                out_noise=0.06
            )

        # The LR-TT tiles take their mappings from the device configuration:
        # A and B use device_config.mapping_ab (no weight scaling), C uses
        # device_config.mapping_c (column-wise weight scaling with learnable
        # output scales)
        mapping = MappingParameter(
            learn_out_scaling=False,
            weight_scaling_omega=0.0
        )

        # Update parameters for pulse generation
        # When use_manual_scaling=True, A and B are set directly from manual_d_scaling and manual_x_scaling
        # bypassing the dynamic calculation based on lr, dw_min, and input magnitudes
        # When use_manual_scaling=False, enable BL/update management for dynamic scaling
        update = UpdateParameters(
            desired_bl=config.desired_bl,              # Bit length (pulse train length)
            pulse_type=config.pulse_type,              # Stochastic pulse generation
            use_manual_scaling=config.use_manual_scaling,  # Enable hardware-realistic fixed scaling
            manual_x_scaling=config.x_scaling,         # B factor: scaling for input x
            manual_d_scaling=config.d_scaling,         # A factor: scaling for gradient d
            update_bl_management=not config.use_manual_scaling,  # Enable when not using manual scaling
            update_management=not config.use_manual_scaling,     # Enable when not using manual scaling
        )

        rpu_config = PythonLRTTRPUConfig(
            device=device_config,
            mapping=mapping,
            forward=forward_io,
            backward=backward_io,
            update=update
        )

        # Reset random state right before tile creation for reproducibility
        # This ensures consistent device-to-device variation across different configs
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Create LRTT linear layer
        self.lrtt_layer = AnalogLinear(
            config.input_dim,
            config.output_dim,
            bias=False,
            rpu_config=rpu_config
        )

        # Reset seed again before C initialization to ensure consistency
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Initialize C with pretrained weights if provided, otherwise use c_init_value
        if pretrained_C is not None:
            self.set_C_weights(pretrained_C)
        else:
            # Set C matrix to c_init_value (must be within analog tile bounds [-1,1])
            C_init = torch.ones(config.output_dim, config.input_dim) * config.c_init_value
            self.set_C_weights(C_init)

        # Reset seed again before A/B reinit for consistent initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Explicitly reinit A and B after setting C to ensure proper initialization
        analog_tile = self.lrtt_layer.analog_module
        if hasattr(analog_tile, 'controller'):
            analog_tile.controller.reinit()

    def set_C_weights(self, C: torch.Tensor):
        """Set the C matrix directly via tile_c to avoid quantization."""
        analog_tile = self.lrtt_layer.analog_module

        # Access tile_c directly to bypass any LRTT-level processing
        if hasattr(analog_tile, 'tile_c'):
            # Try to access the underlying weights tensor directly if possible
            if hasattr(analog_tile.tile_c, 'weight'):
                with torch.no_grad():
                    analog_tile.tile_c.weight.copy_(C.to(DEVICE))
            else:
                analog_tile.tile_c.set_weights(C.to(DEVICE), None)
        else:
            analog_tile.set_weights(C.to(DEVICE))

        # Verify what was actually set
        if hasattr(analog_tile, 'tile_c'):
            C_actual = analog_tile.tile_c.get_weights()[0]
        else:
            C_actual = analog_tile.get_weights()[0]

        print(f"C initialized: min={C_actual.min():.4f}, max={C_actual.max():.4f}, mean={C_actual.mean():.4f}")

    def forward(self, x):
        return self.lrtt_layer(x)

    def get_lrtt_components(self):
        """Get the C, A, B components from LRTT layer."""
        analog_tile = self.lrtt_layer.analog_module
        if hasattr(analog_tile, 'get_lrtt_component_weights'):
            C, A, B = analog_tile.get_lrtt_component_weights()
            return C, A, B
        return None, None, None


# ============================================================================
# Training Functions
# ============================================================================
def train_lrtt_scratch(config: ScratchExperimentConfig,
                      train_loader: DataLoader, val_loader: DataLoader,
                      seed: int = 42, use_wandb: bool = True,
                      collect_history: bool = True) -> tuple:
    """Train LRTT from scratch on D'.

    Args:
        collect_history: If False, skip detailed step-wise history collection
                        for faster training.

    Returns:
        Tuple of (model, training_history, epoch_history, C_init, A_init, B_init)
    """

    print("\n" + "="*60)
    print("SCRATCH TRAINING: LRTT directly on D'")
    print("="*60)

    # Set all random states for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create LRTT model without pre-trained C (random initialization)
    model = LRTTModel(config, pretrained_C=None, seed=seed).to(DEVICE)

    # Check initial A,B,C
    C_init, A_init, B_init = model.get_lrtt_components()
    if A_init is not None and B_init is not None:
        print(f"Initial: ||A||={A_init.norm():.4f}, ||B||={B_init.norm():.4f}, ||C||={C_init.norm():.4f}")

    # Use AnalogSGD for LRTT tiles (no momentum, simple vanilla SGD)
    optimizer = AnalogSGD(model.parameters(), lr=config.lrtt_lr, momentum=0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    last_transfer_count = 0  # Track transfers for transfer-based early stopping

    # Training history for cell-wise tracking
    training_history = []
    epoch_history = []  # Epoch-level history (val_loss, train_loss)

    # Record initial state (step=0, before any training)
    C_init_log, A_init_log, B_init_log = model.get_lrtt_components()
    if collect_history and A_init_log is not None and B_init_log is not None:
        # Create NaN vectors for initial state (no update yet)
        # A tile (down-proj, code B): x [input_dim], d [rank]
        # B tile (up-proj, code A): x [rank], d [output_dim]
        nan_input = np.full(config.input_dim, np.nan)
        nan_output = np.full(config.output_dim, np.nan)
        nan_rank = np.full(config.lrtt_rank, np.nan)

        initial_history = {
            'step': 0,  # 0 indicates initial state
            'epoch': 0,
            'batch_idx': -1,  # -1 indicates before any batch
            'batch_loss': float('nan'),
            'is_transfer': False,
            'grad_norm': 0.0,
            'grad_C_matrix': np.zeros_like(C_init_log.cpu().detach().numpy()),
            'A_matrix': A_init_log.cpu().detach().numpy().copy(),
            'B_matrix': B_init_log.cpu().detach().numpy().copy(),
            'C_matrix': C_init_log.cpu().detach().numpy().copy(),
            'A_norm': A_init_log.norm().item(),
            'B_norm': B_init_log.norm().item(),
            'C_norm': C_init_log.norm().item(),
            'delta_W_norm': (A_init_log @ B_init_log).norm().item(),
            # Pulse vectors - NaN for initial state (no update yet)
            'A_x_vec': nan_input.copy(),  # [input_dim]
            'A_d_vec': nan_rank.copy(),   # [rank]
            'A_p_x_vec': nan_input.copy(),
            'A_p_d_vec': nan_rank.copy(),
            'B_x_vec': nan_rank.copy(),   # [rank]
            'B_d_vec': nan_output.copy(), # [output_dim]
            'B_p_x_vec': nan_rank.copy(),
            'B_p_d_vec': nan_output.copy(),
        }
        training_history.append(initial_history)

    # Log initial state to wandb (before any training) as step 0
    if use_wandb and A_init_log is not None and B_init_log is not None:
        log_dict = {
            'scratch/step': 0,
            'scratch/epoch': -1,
            'scratch/batch_idx': -1,
            'scratch/batch_loss': float('nan'),
            'scratch/is_transfer_step': False,
            'scratch/A_norm_step': A_init_log.norm().item(),
            'scratch/B_norm_step': B_init_log.norm().item(),
            'scratch/C_norm_step': C_init_log.norm().item(),
            'scratch/delta_norm_step': (A_init_log @ B_init_log).norm().item()
        }

        # Log individual A cells (A is [input_dim, rank])
        for i in range(A_init_log.shape[0]):
            for r in range(A_init_log.shape[1]):
                log_dict[f'scratch/A[{i},{r}]_step'] = A_init_log[i, r].item()

        # Log individual B cells (B is [rank, input_dim])
        for r in range(B_init_log.shape[0]):
            for j in range(B_init_log.shape[1]):
                log_dict[f'scratch/B[{r},{j}]_step'] = B_init_log[r, j].item()

        # Log individual C cells (C is [input_dim, input_dim])
        for i in range(C_init_log.shape[0]):
            for j in range(C_init_log.shape[1]):
                log_dict[f'scratch/C[{i},{j}]_step'] = C_init_log[i, j].item()

        wandb.log(log_dict)

    # Global step counter for batch-wise logging (starts at 1)
    global_step = 1

    for epoch in range(config.lrtt_epochs):
        # Training
        model.train()
        train_loss = torch.tensor(0.0, device=DEVICE)  # Accumulate on GPU
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = F.mse_loss(Y_pred, Y_batch)

            # Debug: Check gradients on first batch of first epoch (BEFORE backward)
            if epoch == 0 and batch_idx == 0:
                print(f"\n  Debug Gradients (Epoch {epoch}, Batch {batch_idx}):")
                print(f"    Loss value: {loss.item():.6f}")
                print(f"    Learning rate: {config.lrtt_lr}")

                C, A, B = model.get_lrtt_components()
                print(f"    Before step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

                # Show hardware pulse parameters (fixed scaling mode)
                import math

                # Get actual data values for reference
                x_abs_max = X_batch.abs().max().item()
                d_batch = 2 * (Y_pred - Y_batch) / Y_batch.size(0)
                d_abs_max = d_batch.abs().max().item()

                # Fixed-scaling (use_manual_scaling=True) pulse parameters
                dw_min = config.ab_dw_min  # A/B tile step
                BL = config.desired_bl

                # Get tile-specific scaling factors
                # A tile: uses a_x_scaling, a_d_scaling (or global if None)
                # B tile: uses b_x_scaling, b_d_scaling (or global if None)
                a_x_factor = config.a_x_scaling if config.a_x_scaling is not None else config.x_scaling
                a_d_factor = config.a_d_scaling if config.a_d_scaling is not None else config.d_scaling
                b_x_factor = config.b_x_scaling if config.b_x_scaling is not None else config.x_scaling
                b_d_factor = config.b_d_scaling if config.b_d_scaling is not None else config.d_scaling

                # Format global scaling (handle None)
                global_x = f"{config.x_scaling:.4f}" if config.x_scaling is not None else "None"
                global_d = f"{config.d_scaling:.4f}" if config.d_scaling is not None else "None"
                a_x_str = f"{a_x_factor:.4f}" if a_x_factor is not None else "None"
                a_d_str = f"{a_d_factor:.4f}" if a_d_factor is not None else "None"
                b_x_str = f"{b_x_factor:.4f}" if b_x_factor is not None else "None"
                b_d_str = f"{b_d_factor:.4f}" if b_d_factor is not None else "None"

                print(f"\n    === Pulse parameters for fixed scaling (use_manual_scaling=True) ===")
                print(f"    dw_min (A/B tiles): {dw_min:.6f}")
                print(f"    BL: {BL}")
                print(f"    Global: x_scaling={global_x}, d_scaling={global_d}")
                print(f"    --- Tile-Specific Scaling ---")
                print(f"    A tile (code): x_scaling={a_x_str}, d_scaling={a_d_str}")
                print(f"    B tile (code): x_scaling={b_x_str}, d_scaling={b_d_str}")
                print(f"    --- Current Batch Data ---")
                print(f"    x_abs_max: {x_abs_max:.6f}")
                print(f"    d_abs_max: {d_abs_max:.6f}")
                print(f"    ======================================\n")

            # Compute history-related values only if collecting history
            if collect_history:
                # Compute gradient norm before backward (d = dL/dy)
                # MSE gradient: d = (2/N) * (y_pred - y_target)
                N = Y_pred.numel()
                grad_d = (2.0 / N) * (Y_pred - Y_batch)
                grad_norm = grad_d.norm().item()

                # Apply quantization to gradient if enabled
                if config.quantize_d:
                    grad_d = torch.round(grad_d / config.d_resolution) * config.d_resolution

                # Compute gradient matrix for C: dL/dC = grad_d.T @ X_batch / batch_size
                # Shape: [output_dim, input_dim] - same as C
                grad_C_matrix = (grad_d.T @ X_batch) / X_batch.size(0)

                # Compute x, d values for A and B tiles (before update)
                # Get current A, B for computing projected values
                C_pre, A_pre, B_pre = model.get_lrtt_components()
                A_pre = A_pre.to(DEVICE)
                B_pre = B_pre.to(DEVICE)

                # Apply quantization to input if enabled
                X_quantized = X_batch
                if config.quantize_x:
                    X_quantized = torch.round(X_batch / config.x_resolution) * config.x_resolution

                # A tile (code) = B tile (standard, up-projection) [output_dim, rank]
                # x_A = input projected through B: X @ B.T [batch, rank]
                # d_A = gradient at output: grad_d [batch, output_dim]
                x_A = X_quantized @ B_pre.T  # [batch, rank]
                d_A = grad_d  # [batch, output_dim]

                # B tile (code) = A tile (standard, down-projection) [rank, input_dim]
                # x_B = original input: X [batch, input_dim]
                # d_B = gradient projected through A: grad_d @ A [batch, rank]
                x_B = X_quantized  # [batch, input_dim]
                d_B = grad_d @ A_pre  # [batch, rank]

                # Compute max absolute values per position (over batch dimension)
                # For A tile (code, up-proj): x_A [batch, rank], d_A [batch, output_dim]
                # For B tile (code, down-proj): x_B [batch, input_dim], d_B [batch, rank]
                x_A_vec = x_A.abs().max(dim=0).values  # [rank]
                d_A_vec = d_A.abs().max(dim=0).values  # [output_dim]
                x_B_vec = x_B.abs().max(dim=0).values  # [input_dim]
                d_B_vec = d_B.abs().max(dim=0).values  # [rank]

                # Compute scaled probabilities (clipped to [0, 1])
                # Note: In Excel notation, A=down-proj (code B), B=up-proj (code A)
                a_x_scale = config.a_x_scaling or 1.0
                a_d_scale = config.a_d_scaling or 1.0
                b_x_scale = config.b_x_scaling or 1.0
                b_d_scale = config.b_d_scaling or 1.0

                p_x_A_vec = (a_x_scale * x_A_vec).clamp(max=1.0)  # [rank]
                p_d_A_vec = (a_d_scale * d_A_vec).clamp(max=1.0)  # [output_dim]
                p_x_B_vec = (b_x_scale * x_B_vec).clamp(max=1.0)  # [input_dim]
                p_d_B_vec = (b_d_scale * d_B_vec).clamp(max=1.0)  # [rank]

            # Backward pass
            loss.backward()

            # Gradient clipping
            # Track num_transfers before step to detect transfer
            analog_tile = model.lrtt_layer.analog_module
            num_transfers_before = 0
            if hasattr(analog_tile, 'controller'):
                num_transfers_before = analog_tile.controller.num_transfers

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.lrtt_grad_clip)
            optimizer.step()

            # Per-pulse decay: post_update_step() already called decay_weights() once,
            # call desired_bl - 1 more times to simulate per-pulse capacitor decay
            if config.desired_bl > 1:
                for _ in range(config.desired_bl - 1):
                    if hasattr(analog_tile.tile_a, "rpu_config") and analog_tile.tile_a.rpu_config.device.requires_decay():
                        analog_tile.tile_a.decay_weights()
                    if hasattr(analog_tile.tile_b, "rpu_config") and analog_tile.tile_b.rpu_config.device.requires_decay():
                        analog_tile.tile_b.decay_weights()

            # Debug: Check if A changed after optimizer step
            if epoch == 0 and batch_idx == 0:
                C, A, B = model.get_lrtt_components()
                print(f"    After step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

            train_loss += loss.detach() * X_batch.size(0)  # Keep on GPU

            # Log A, B, C cell values after each batch update (skip if not needed)
            if collect_history or use_wandb:
                C, A, B = model.get_lrtt_components()
                if A is not None and B is not None:
                    # Check if transfer occurred by comparing num_transfers
                    is_transfer_step = False
                    if hasattr(analog_tile, 'controller'):
                        num_transfers_after = analog_tile.controller.num_transfers
                        if num_transfers_after > num_transfers_before:
                            is_transfer_step = True

                    # Record step-wise history for Excel export (skip if not collecting)
                    if collect_history:
                        # Note: Excel uses standard LoRA notation
                        # Code A tile (up-proj) -> Excel B tile, Code B tile (down-proj) -> Excel A tile
                        step_history = {
                            'step': global_step,
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'batch_loss': loss.item(),
                            'is_transfer': is_transfer_step,
                            'grad_norm': grad_norm,
                            'grad_C_matrix': grad_C_matrix.cpu().detach().numpy().copy(),
                            'A_matrix': A.cpu().detach().numpy().copy(),
                            'B_matrix': B.cpu().detach().numpy().copy(),
                            'C_matrix': C.cpu().detach().numpy().copy(),
                            'A_norm': A.norm().item(),
                            'B_norm': B.norm().item(),
                            'C_norm': C.norm().item(),
                            'delta_W_norm': (A @ B).norm().item(),
                            # Pulse input vectors - Excel notation (A=down-proj, B=up-proj)
                            # A tile (down-proj) = code B tile: x [input_dim], d [rank]
                            'A_x_vec': x_B_vec.cpu().detach().numpy().copy(),  # [input_dim]
                            'A_d_vec': d_B_vec.cpu().detach().numpy().copy(),  # [rank]
                            'A_p_x_vec': p_x_B_vec.cpu().detach().numpy().copy(),  # [input_dim]
                            'A_p_d_vec': p_d_B_vec.cpu().detach().numpy().copy(),  # [rank]
                            # B tile (up-proj) = code A tile: x [rank], d [output_dim]
                            'B_x_vec': x_A_vec.cpu().detach().numpy().copy(),  # [rank]
                            'B_d_vec': d_A_vec.cpu().detach().numpy().copy(),  # [output_dim]
                            'B_p_x_vec': p_x_A_vec.cpu().detach().numpy().copy(),  # [rank]
                            'B_p_d_vec': p_d_A_vec.cpu().detach().numpy().copy(),  # [output_dim]
                        }
                        training_history.append(step_history)

                    if use_wandb:
                        log_dict = {
                            'scratch/step': global_step,
                            'scratch/epoch': epoch,
                            'scratch/batch_idx': batch_idx,
                            'scratch/batch_loss': loss.item(),
                            'scratch/is_transfer_step': is_transfer_step,
                            'scratch/A_norm_step': A.norm().item(),
                            'scratch/B_norm_step': B.norm().item(),
                            'scratch/C_norm_step': C.norm().item(),
                            'scratch/delta_norm_step': (A @ B).norm().item()
                        }

                        # Log individual A cells (A is [input_dim, rank])
                        for i in range(A.shape[0]):
                            for r in range(A.shape[1]):
                                log_dict[f'scratch/A[{i},{r}]_step'] = A[i, r].item()

                        # Log individual B cells (B is [rank, input_dim])
                        for r in range(B.shape[0]):
                            for j in range(B.shape[1]):
                                log_dict[f'scratch/B[{r},{j}]_step'] = B[r, j].item()

                        # Log individual C cells (C is [input_dim, input_dim])
                        for i in range(C.shape[0]):
                            for j in range(C.shape[1]):
                                log_dict[f'scratch/C[{i},{j}]_step'] = C[i, j].item()

                        wandb.log(log_dict)
                else:
                    if global_step <= 5:
                        print(f"  [WARNING] Step {global_step}: A or B is None, skipping log")

            global_step += 1

        train_loss = (train_loss / len(train_loader.dataset)).item()  # GPU sync once per epoch

        # Check for transfer event (compare with last_transfer_count)
        # NOTE: Use num_transfers (cumulative), not transfer_counter (resets after each transfer)
        transfer_occurred = False
        current_transfer_count = 0
        analog_tile = model.lrtt_layer.analog_module
        if hasattr(analog_tile, 'controller'):
            current_transfer_count = analog_tile.controller.num_transfers
            if current_transfer_count > last_transfer_count:
                # Transfer(s) occurred during this epoch
                transfer_occurred = True
                C, A, B = model.get_lrtt_components()
                transfers_this_epoch = current_transfer_count - last_transfer_count

                # Clamp C to [-1, 1] for FloatingPointDevice (no built-in bounds)
                if config.c_device_type == 'floating_point':
                    C_clamped = torch.clamp(C, config.c_w_min, config.c_w_max)
                    model.set_C_weights(C_clamped)
                    C = C_clamped

                print(f"  [TRANSFER] Epoch {epoch}: {transfers_this_epoch} transfer(s), A norm={A.norm():.4f}, B norm={B.norm():.4f}, C norm={C.norm():.4f}")

        # Validation - Only run when needed for early stopping (epoch 0 or transfer)
        # This saves significant time by skipping validation on most epochs
        need_validation = (epoch == 0 or transfer_occurred) and val_loader is not None
        if need_validation:
            model.eval()
            val_loss = torch.tensor(0.0, device=DEVICE)  # Accumulate on GPU
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                    Y_pred = model(X_batch)
                    loss = F.mse_loss(Y_pred, Y_batch)
                    val_loss += loss * X_batch.size(0)  # Keep on GPU

            val_loss = (val_loss / len(val_loader.dataset)).item()  # GPU sync once
            loss_for_stopping = val_loss
        elif val_loader is None:
            # No validation set - use train_loss for early stopping
            val_loss = None
            loss_for_stopping = train_loss
        else:
            # Skip validation this epoch (not needed for early stopping)
            val_loss = None
            loss_for_stopping = None  # Won't be used since early stopping only checks on transfer

        # Record epoch-level history
        epoch_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        # Early stopping - check only at transfer events (not every epoch)
        # This makes patience count transfers, not epochs
        # Always save best state on first epoch or when transfer occurs
        if epoch == 0 or transfer_occurred:
            if loss_for_stopping < best_val_loss:
                best_val_loss = loss_for_stopping
                best_model_state = model.state_dict()
                patience_counter = 0
            elif transfer_occurred:  # Only increment patience on transfer (not first epoch)
                patience_counter += 1
            # Update transfer count tracker
            last_transfer_count = current_transfer_count

        # Wandb logging (epoch summary - norms and losses only, no cell values)
        # Use the last step of the epoch for epoch-level metrics
        if use_wandb:
            epoch_step = global_step - 1  # Last step of this epoch
            log_dict = {
                'scratch/step': epoch_step,  # Same step as last batch
                'scratch/train_loss_epoch': train_loss,
            }
            if val_loss is not None:
                log_dict['scratch/val_loss_epoch'] = val_loss

            # Log component norms if available
            C, A, B = model.get_lrtt_components()
            if A is not None and B is not None:
                log_dict.update({
                    'scratch/A_norm_epoch': A.norm().item(),
                    'scratch/B_norm_epoch': B.norm().item(),
                    'scratch/C_norm_epoch': C.norm().item(),
                    'scratch/delta_norm_epoch': (A @ B).norm().item()
                })

            wandb.log(log_dict, commit=False)  # commit=False to merge with last batch's log

        if epoch % 10 == 0:
            if val_loss is not None:
                print(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d}: Train={train_loss:.6f}")

        if transfer_occurred and patience_counter >= config.lrtt_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} transfers)")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if val_loader is not None:
        print(f"\nBest val loss: {best_val_loss:.6f}")
    else:
        print(f"\nBest train loss: {best_val_loss:.6f}")

    # Print final norms
    C, A, B = model.get_lrtt_components()
    if A is not None and B is not None:
        print(f"Final: ||A||={A.norm():.4f}, ||B||={B.norm():.4f}, ||A@B||={(A @ B).norm():.4f}")

    return model, training_history, epoch_history, C_init, A_init, B_init


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Evaluate model performance on test set."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            Y_pred = model(X_batch)
            all_preds.append(Y_pred.cpu())
            all_targets.append(Y_batch.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = F.mse_loss(preds, targets).item()
    rmse = np.sqrt(mse)

    # R^2 score
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    results = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
    }

    return results


def compare_matrices(learned_model: nn.Module, target_matrix: torch.Tensor) -> Dict[str, float]:
    """Distance between C + A @ B and the target matrix (Frobenius norm, MSE, relative error)."""
    learned_model.eval()

    # Extract learned weight matrix from the model
    with torch.no_grad():
        # Get the combined matrix (C + A @ B)
        C, A, B = learned_model.get_lrtt_components()
        if A is not None and B is not None:
            learned_matrix = C + A @ B
        else:
            learned_matrix = C

    # Move target to same device
    target_matrix = target_matrix.to(learned_matrix.device)

    # Compute differences
    diff_matrix = learned_matrix - target_matrix
    frobenius_diff = torch.norm(diff_matrix, 'fro').item()
    mse_matrix = F.mse_loss(learned_matrix, target_matrix).item()

    # Relative error
    target_norm = torch.norm(target_matrix, 'fro').item()
    relative_error = frobenius_diff / target_norm if target_norm > 0 else float('inf')

    return {
        'frobenius_diff': frobenius_diff,
        'mse_matrix': mse_matrix,
        'relative_error': relative_error,
        'target_norm': target_norm,
        'learned_norm': torch.norm(learned_matrix, 'fro').item()
    }


def plot_all_training_figures(training_history: list,
                               epoch_history: list,
                               config: ScratchExperimentConfig,
                               complexity_level: str,
                               seed: int,
                               timestamp: str,
                               final_mse: float = None,
                               final_r2: float = None,
                               target_matrix: torch.Tensor = None) -> dict:
    """Plot all training metrics and save them as images.

    Generates multiple figures:
    1. training_cell: Single cell [0,0] tracking - grad_C, C, B@A on one graph
    2. all_norms: ||A||, ||B||, ||C||, ||B@A||
    3. loss: val_loss over epochs (with final MSE and R^2 annotation)
    4. A_cells: individual A matrix cell values (down-projection)
    5. B_cells: individual B matrix cell values (up-projection)
    6. C_cells: individual C matrix cell values

    Note: Uses standard LoRA notation where A=down-projection, B=up-projection.

    Args:
        training_history: List of step-wise training history dictionaries
        epoch_history: List of epoch-wise history (train_loss, val_loss)
        config: Experiment configuration
        complexity_level: Complexity level string
        seed: Random seed
        timestamp: Timestamp string for filename
        final_mse: Final MSE loss on test set
        final_r2: Final R^2 score on test set
        target_matrix: Target matrix T' for displaying target values on C cell plots

    Returns:
        Dictionary of saved figure paths
    """
    if not training_history:
        print("No training history to plot")
        return {}

    os.makedirs(config.results_dir, exist_ok=True)
    saved_paths = {}

    # Extract common data
    steps = [h['step'] for h in training_history]
    is_transfer = [h['is_transfer'] for h in training_history]
    transfer_steps = [s for s, t in zip(steps, is_transfer) if t]

    def add_transfer_lines(ax):
        """Add vertical lines at transfer steps."""
        for ts in transfer_steps:
            ax.axvline(x=ts, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)

    # =========================================================================
    # Figure 1: Single Cell Tracking (grad_C, C, B@A for cell [0,0])
    # Gradient, core weight and LoRA update of one cell on a common axis
    # =========================================================================
    cell_row, cell_col = 0, 0  # Cell to track (can be changed)

    # Extract single cell values from each step
    grad_C_cells = [h['grad_C_matrix'][cell_row, cell_col] if 'grad_C_matrix' in h else 0
                    for h in training_history]
    C_cells = [h['C_matrix'][cell_row, cell_col] for h in training_history]
    # A @ B in code notation = B @ A in standard LoRA notation
    AB_cells = [np.dot(h['A_matrix'], h['B_matrix'])[cell_row, cell_col]
                for h in training_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Gradient with filled area to y=0
    ax.fill_between(steps, grad_C_cells, 0, alpha=0.3, color='blue')
    ax.plot(steps, grad_C_cells, 'b-', linewidth=1, label=f'dL/dC[{cell_row},{cell_col}] (gradient)')
    ax.plot(steps, C_cells, 'g-', linewidth=1, label=f'C[{cell_row},{cell_col}] (core weight)')
    ax.plot(steps, AB_cells, 'r-', linewidth=1, label=f'(B@A)[{cell_row},{cell_col}] (LoRA update)')

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(f'Single Cell Tracking [{cell_row},{cell_col}] (complexity={complexity_level}, seed={seed})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # No transfer lines in this panel (three curves and the gradient fill)

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"training_cell_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['training_cell'] = path

    # =========================================================================
    # Figure 2: All Matrix Norms (A, B, C, delta_W)
    # Note: A=down-proj, B=up-proj (standard LoRA notation)
    # Code's A_norm -> B (up-proj), Code's B_norm -> A (down-proj)
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Code A = up-proj = standard B, Code B = down-proj = standard A
    code_A_norms = [h['A_norm'] for h in training_history]  # up-proj -> B
    code_B_norms = [h['B_norm'] for h in training_history]  # down-proj -> A
    C_norms = [h['C_norm'] for h in training_history]
    delta_W_norms = [h['delta_W_norm'] for h in training_history]

    axes[0].plot(steps, code_B_norms, 'cyan', linewidth=1, label='||A|| (down-proj)')
    axes[0].set_ylabel('||A|| Norm')
    axes[0].set_title(f'All Matrix Norms (complexity={complexity_level}, seed={seed})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    add_transfer_lines(axes[0])

    axes[1].plot(steps, code_A_norms, 'purple', linewidth=1, label='||B|| (up-proj)')
    axes[1].set_ylabel('||B|| Norm')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    add_transfer_lines(axes[1])

    axes[2].plot(steps, C_norms, 'g-', linewidth=1, label='||C|| (Core)')
    axes[2].set_ylabel('||C|| Norm')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    add_transfer_lines(axes[2])

    axes[3].plot(steps, delta_W_norms, 'r-', linewidth=1, label='||B@A|| (LoRA update)')
    axes[3].set_ylabel('||B@A|| Norm')
    axes[3].set_xlabel('Step')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    add_transfer_lines(axes[3])

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"all_norms_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['all_norms'] = path

    # =========================================================================
    # Figure 3: Loss (val_loss per epoch with final MSE and R^2 annotation)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 4))

    if epoch_history:
        epochs = [h['epoch'] for h in epoch_history]
        val_losses = [h['val_loss'] for h in epoch_history]
        train_losses = [h['train_loss'] for h in epoch_history]

        ax.plot(epochs, val_losses, 'b-', linewidth=1, label='Val Loss (MSE)')
        ax.plot(epochs, train_losses, 'r--', linewidth=1, alpha=0.7, label='Train Loss (MSE)')
        ax.set_xlabel('Epoch')
    else:
        # Batch loss when no epoch history is available
        batch_losses = [h['batch_loss'] for h in training_history]
        ax.plot(steps, batch_losses, 'b-', linewidth=1, label='Batch Loss (MSE)')
        ax.set_xlabel('Step')

    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss (complexity={complexity_level}, seed={seed})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add final MSE and R^2 as text annotation
    if final_mse is not None or final_r2 is not None:
        text_lines = []
        if final_mse is not None:
            text_lines.append(f'Final MSE: {final_mse:.6f}')
        if final_r2 is not None:
            text_lines.append(f'Final R^2: {final_r2:.4f}')
        text_str = '\n'.join(text_lines)

        # Position text in upper left (avoiding the legend in upper right)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"loss_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['loss'] = path

    # =========================================================================
    # Figure 4: A matrix cell values (down-projection)
    # Note: Code's B_matrix = standard A (down-projection)
    # =========================================================================
    if training_history and 'B_matrix' in training_history[0]:
        B_shape = training_history[0]['B_matrix'].shape
        n_cells = B_shape[0] * B_shape[1]

        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 3 * n_cells), sharex=True)
        if n_cells == 1:
            axes = [axes]

        cell_idx = 0
        for r in range(B_shape[0]):
            for j in range(B_shape[1]):
                values = [h['B_matrix'][r, j] for h in training_history]
                axes[cell_idx].plot(steps, values, 'cyan', linewidth=1, label=f'A[{r},{j}]')
                axes[cell_idx].set_ylabel(f'A[{r},{j}]')
                axes[cell_idx].legend(loc='upper right')
                axes[cell_idx].grid(True, alpha=0.3)
                add_transfer_lines(axes[cell_idx])
                cell_idx += 1

        axes[0].set_title(f'A Matrix Cells (down-proj) (complexity={complexity_level}, seed={seed})')
        axes[-1].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"A_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['A_cells'] = path

    # =========================================================================
    # Figure 5: B matrix cell values (up-projection)
    # Note: Code's A_matrix = standard B (up-projection)
    # =========================================================================
    if training_history and 'A_matrix' in training_history[0]:
        A_shape = training_history[0]['A_matrix'].shape
        n_cells = A_shape[0] * A_shape[1]

        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 3 * n_cells), sharex=True)
        if n_cells == 1:
            axes = [axes]

        cell_idx = 0
        for i in range(A_shape[0]):
            for r in range(A_shape[1]):
                values = [h['A_matrix'][i, r] for h in training_history]
                axes[cell_idx].plot(steps, values, 'purple', linewidth=1, label=f'B[{i},{r}]')
                axes[cell_idx].set_ylabel(f'B[{i},{r}]')
                axes[cell_idx].legend(loc='upper right')
                axes[cell_idx].grid(True, alpha=0.3)
                add_transfer_lines(axes[cell_idx])
                cell_idx += 1

        axes[0].set_title(f'B Matrix Cells (up-proj) (complexity={complexity_level}, seed={seed})')
        axes[-1].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"B_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['B_cells'] = path

    # =========================================================================
    # Figure 6: C matrix cell values (Core weights)
    # =========================================================================
    if training_history and 'C_matrix' in training_history[0]:
        C_shape = training_history[0]['C_matrix'].shape
        n_cells = C_shape[0] * C_shape[1]

        # For large C matrices, create a grid layout instead
        n_rows = C_shape[0]
        n_cols = C_shape[1]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_rows):
            for j in range(n_cols):
                values = [h['C_matrix'][i, j] for h in training_history]
                axes[i, j].plot(steps, values, 'g-', linewidth=1, label='C (learned)')
                # Add target value as horizontal dashed line
                if target_matrix is not None:
                    target_val = target_matrix[i, j].item() if torch.is_tensor(target_matrix) else target_matrix[i, j]
                    axes[i, j].axhline(y=target_val, color='r', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Target={target_val:.3f}')
                axes[i, j].set_title(f'C[{i},{j}]', fontsize=10)
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].legend(fontsize=7, loc='best')
                add_transfer_lines(axes[i, j])

        fig.suptitle(f'C Matrix Cells (Core) vs Target (complexity={complexity_level}, seed={seed})', fontsize=12)
        axes[-1, n_cols // 2].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"C_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['C_cells'] = path

    # Print summary
    print(f"\nTraining figures saved to {config.results_dir}/:")
    for name, path in saved_paths.items():
        print(f"  - {os.path.basename(path)}")

    return saved_paths


def plot_combined_training_figures(all_results: dict,
                                    config: ScratchExperimentConfig,
                                    complexity_level: str,
                                    seed: int,
                                    timestamp: str) -> dict:
    """Plot combined training metrics for multiple parameter configurations.

    Args:
        all_results: Dict of {label: {'training_history': [...], 'epoch_history': [...],
                                      'final_mse': float, 'final_r2': float}}
        config: Experiment configuration
        complexity_level: Complexity level string
        seed: Random seed
        timestamp: Timestamp string for filename

    Returns:
        Dictionary of saved figure paths
    """
    import matplotlib.cm as cm

    os.makedirs(config.results_dir, exist_ok=True)
    saved_paths = {}

    labels = list(all_results.keys())
    n_configs = len(labels)
    colors = cm.tab10(np.linspace(0, 1, max(n_configs, 10)))[:n_configs]

    # =========================================================================
    # Figure 1: Combined Loss Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (label, data) in enumerate(all_results.items()):
        epoch_history = data.get('epoch_history', [])
        if epoch_history:
            epochs = [h['epoch'] for h in epoch_history]
            val_losses = [h['val_loss'] for h in epoch_history]
            final_r2 = data.get('final_r2', None)
            label_str = f"{label}" + (f" (R^2={final_r2:.3f})" if final_r2 else "")
            ax.plot(epochs, val_losses, color=colors[idx], linewidth=1.5, label=label_str)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.set_title(f'Combined Training Loss (complexity={complexity_level}, seed={seed})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"combined_loss_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['combined_loss'] = path

    # =========================================================================
    # Figure 2: Combined Norm Plots (||A||, ||B||, ||C||, ||B@A||)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (label, data) in enumerate(all_results.items()):
        training_history = data.get('training_history', [])
        if not training_history:
            continue

        steps = [h['step'] for h in training_history]
        # Code A = up-proj = standard B, Code B = down-proj = standard A
        code_A_norms = [h['A_norm'] for h in training_history]
        code_B_norms = [h['B_norm'] for h in training_history]
        C_norms = [h['C_norm'] for h in training_history]
        delta_W_norms = [h['delta_W_norm'] for h in training_history]

        axes[0, 0].plot(steps, code_B_norms, color=colors[idx], linewidth=1, label=label, alpha=0.8)
        axes[0, 1].plot(steps, code_A_norms, color=colors[idx], linewidth=1, label=label, alpha=0.8)
        axes[1, 0].plot(steps, C_norms, color=colors[idx], linewidth=1, label=label, alpha=0.8)
        axes[1, 1].plot(steps, delta_W_norms, color=colors[idx], linewidth=1, label=label, alpha=0.8)

    axes[0, 0].set_ylabel('||A|| (down-proj)')
    axes[0, 0].set_title('A Matrix Norm')
    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel('||B|| (up-proj)')
    axes[0, 1].set_title('B Matrix Norm')
    axes[0, 1].legend(loc='upper right', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_ylabel('||C|| (core)')
    axes[1, 0].set_title('C Matrix Norm')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend(loc='upper right', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_ylabel('||B@A|| (LoRA update)')
    axes[1, 1].set_title('LoRA Update Norm')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend(loc='upper right', fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Combined Matrix Norms (complexity={complexity_level}, seed={seed})', fontsize=12)
    plt.tight_layout()
    path = os.path.join(config.results_dir, f"combined_norms_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['combined_norms'] = path

    # =========================================================================
    # Figure 3: Final R^2 Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    r2_values = [data.get('final_r2', 0) for data in all_results.values()]
    x_pos = np.arange(len(labels))

    bars = ax.bar(x_pos, r2_values, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R^2 Score')
    ax.set_title(f'Final R^2 Comparison (complexity={complexity_level}, seed={seed})')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"combined_r2_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['combined_r2'] = path

    print(f"\nCombined figures saved to {config.results_dir}/:")
    for name, path in saved_paths.items():
        print(f"  - {os.path.basename(path)}")

    return saved_paths


def run_multi_param_experiments(param_configs: list,
                                 complexity_level: str = 'medium',
                                 seed: int = 42,
                                 use_wandb: bool = False) -> dict:
    """Run experiments with multiple parameter configurations and generate combined plots.

    Args:
        param_configs: List of dicts with parameter overrides and 'label' key
        complexity_level: Complexity level for target matrix
        seed: Random seed
        use_wandb: Whether to use wandb logging

    Returns:
        Dictionary of all results
    """
    from torch.utils.data import DataLoader

    all_results = {}
    trajectories = {}
    base_config = ScratchExperimentConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate target matrix (same for all experiments)
    target_matrix = generate_target_matrix(complexity_level, base_config, seed)
    train_dataset = generate_target_dataset(complexity_level, base_config, train=True, seed=seed)

    for param_config in param_configs:
        label = param_config.get('label', str(param_config))

        # Create config with overrides
        config = ScratchExperimentConfig()
        for key, value in param_config.items():
            if key != 'label' and hasattr(config, key):
                setattr(config, key, value)

        # Disable verbose output
        config.log_ab_scaling = False
        config.save_figures = False  # Individual figures disabled

        # Generate dataset for this config
        train_ds = generate_target_dataset(complexity_level, config, train=True, seed=seed)
        val_ds = generate_target_dataset(complexity_level, config, train=False, seed=seed)

        train_loader = DataLoader(train_ds, batch_size=config.lrtt_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.lrtt_batch_size, shuffle=False)

        print(f"\n{'='*60}")
        print(f"Running experiment: {label}")
        print(f"{'='*60}")

        # Initialize wandb run for this experiment
        if use_wandb:
            run_name = f"multi_{label}_r{config.lrtt_rank}_t{config.lrtt_transfer_every}_alpha{config.lora_alpha}"
            wandb.init(
                project="regression-lrtt-scratch",
                name=run_name,
                config={
                    'experiment_type': 'multi-param',
                    'label': label,
                    'complexity_level': complexity_level,
                    'seed': seed,
                    'lrtt_lr': config.lrtt_lr,
                    'lrtt_rank': config.lrtt_rank,
                    'transfer_every': config.lrtt_transfer_every,
                    'lora_alpha': config.lora_alpha,
                    'use_6t1c_ab': config.USE_6T1C_AB,
                    'a_lifetime': config.lifetime if config.USE_6T1C_AB else None,
                    'b_lifetime': (config.b_lifetime if config.b_lifetime is not None else config.lifetime) if config.USE_6T1C_AB else None,
                    'include_retention': config.include_retention if config.USE_6T1C_AB else None,
                },
                reinit=True
            )

        # Reset random state
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Train
        model, training_history, epoch_history, _, _, _ = train_lrtt_scratch(
            config, train_loader, val_loader, seed=seed, use_wandb=use_wandb
        )

        # Evaluate (same seed as training to use same target matrix)
        test_ds = generate_target_dataset(complexity_level, config, train=False, seed=seed)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        X_test, Y_test = next(iter(test_loader))
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test)
            mse = torch.nn.functional.mse_loss(Y_pred, Y_test).item()
            ss_res = ((Y_test - Y_pred) ** 2).sum().item()
            ss_tot = ((Y_test - Y_test.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Strip training_history to keep only fields needed for plotting
        # PCA trajectory: C_matrix, is_transfer
        # Combined norms: step, A_norm, B_norm, C_norm, delta_W_norm
        _keep_keys = {'step', 'epoch', 'batch_idx', 'batch_loss', 'is_transfer',
                       'A_norm', 'B_norm', 'C_norm', 'delta_W_norm', 'C_matrix'}
        training_history_slim = [
            {k: v for k, v in h.items() if k in _keep_keys}
            for h in training_history
        ]
        del training_history

        all_results[label] = {
            'training_history': training_history_slim,
            'epoch_history': epoch_history,
            'final_mse': mse,
            'final_r2': r2,
            'config': {k: getattr(config, k) for k in dir(config) if not k.startswith('_') and not callable(getattr(config, k))}
        }
        trajectories[label] = training_history_slim

        print(f"  Final MSE: {mse:.6f}, R^2: {r2:.4f}")

        # Free model and CUDA memory before next experiment
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Finish wandb run for this experiment
        if use_wandb:
            wandb.log({
                'final/mse': mse,
                'final/r2': r2,
            })
            wandb.summary['final_mse'] = mse
            wandb.summary['final_r2'] = r2
            wandb.finish()

    # Save raw data (training_history, epoch_history, config per experiment)
    import pickle
    raw_data_path = os.path.join(base_config.results_dir, f"raw_data_{complexity_level}_seed{seed}_{timestamp}.pkl")
    with open(raw_data_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nRaw data saved to: {raw_data_path}")

    # Generate combined plots
    print(f"\n{'='*60}")
    print("Generating combined plots...")
    print(f"{'='*60}")

    plot_combined_training_figures(
        all_results=all_results,
        config=base_config,
        complexity_level=complexity_level,
        seed=seed,
        timestamp=timestamp
    )

    # Generate trajectory comparison plots
    plot_learning_trajectories_pca(
        trajectories=trajectories,
        target_matrix=target_matrix,
        config=base_config,
        train_dataset=train_dataset,
        grid_resolution=80
    )

    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-PARAM EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<30} {'MSE':<12} {'R^2':<10}")
    print("-" * 52)
    for label, data in all_results.items():
        print(f"{label:<30} {data['final_mse']:<12.6f} {data['final_r2']:<10.4f}")

    return all_results


def plot_loss_surface_with_trajectory(training_history: list,
                                       target_matrix: torch.Tensor,
                                       train_dataset: TensorDataset,
                                       config: ScratchExperimentConfig,
                                       complexity_level: str,
                                       seed: int,
                                       timestamp: str,
                                       cell1: tuple = (0, 0),
                                       cell2: tuple = (1, 1),
                                       grid_resolution: int = 50,
                                       margin: float = 0.5) -> str:
    """Plot 2D loss surface with learning trajectory overlaid.

    Args:
        training_history: List of step-wise training history
        target_matrix: Target matrix T'
        train_dataset: Training dataset
        config: Experiment configuration
        complexity_level: Complexity level string
        seed: Random seed
        timestamp: Timestamp for filename
        cell1: First C matrix cell to vary (row, col)
        cell2: Second C matrix cell to vary (row, col)
        grid_resolution: Number of points along each axis
        margin: Extra margin around trajectory for grid

    Returns:
        Path to saved figure
    """
    import numpy as np
    from torch.utils.data import DataLoader

    os.makedirs(config.results_dir, exist_ok=True)

    # Extract C matrix trajectory
    C_trajectory = [h['C_matrix'] for h in training_history]
    cell1_values = [C[cell1[0], cell1[1]] for C in C_trajectory]
    cell2_values = [C[cell2[0], cell2[1]] for C in C_trajectory]

    # Target values
    target1 = target_matrix[cell1[0], cell1[1]].item()
    target2 = target_matrix[cell2[0], cell2[1]].item()

    # Grid range
    all_vals1 = cell1_values + [target1]
    all_vals2 = cell2_values + [target2]
    min1, max1 = min(all_vals1), max(all_vals1)
    min2, max2 = min(all_vals2), max(all_vals2)
    range1 = max(max1 - min1, 0.1)
    range2 = max(max2 - min2, 0.1)

    grid_min1, grid_max1 = min1 - margin * range1, max1 + margin * range1
    grid_min2, grid_max2 = min2 - margin * range2, max2 + margin * range2

    x = np.linspace(grid_min1, grid_max1, grid_resolution)
    y = np.linspace(grid_min2, grid_max2, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Get training data
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)

    # Use final C as base (fix other cells)
    base_C = torch.tensor(C_trajectory[-1], dtype=torch.float32)

    # Compute loss surface
    Z = np.zeros_like(X)
    print(f"Computing loss surface ({grid_resolution}x{grid_resolution})...")

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            C_test = base_C.clone()
            C_test[cell1[0], cell1[1]] = X[i, j]
            C_test[cell2[0], cell2[1]] = Y[i, j]
            Y_pred = X_train @ C_test.T.to(DEVICE)
            Z[i, j] = F.mse_loss(Y_pred, Y_train).item()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot (log scale for better visualization)
    Z_safe = np.clip(Z, 1e-8, None)
    levels = np.logspace(np.log10(Z_safe.min()), np.log10(Z_safe.max()), 30)
    contour = ax.contourf(X, Y, Z_safe, levels=levels, cmap='viridis',
                          norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(contour, ax=ax, label='MSE Loss (log scale)')
    ax.contour(X, Y, Z_safe, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # Learning trajectory
    ax.plot(cell1_values, cell2_values, 'r-', linewidth=2, alpha=0.9, label='Learning path')
    ax.scatter(cell1_values[0], cell2_values[0], c='lime', s=150, marker='o',
               edgecolors='white', linewidths=2, zorder=5, label='Start')
    ax.scatter(cell1_values[-1], cell2_values[-1], c='red', s=200, marker='*',
               edgecolors='white', linewidths=2, zorder=5, label='End')

    # Target
    ax.scatter(target1, target2, c='yellow', s=250, marker='X',
               edgecolors='black', linewidths=3, zorder=6,
               label=f'Target ({target1:.2f}, {target2:.2f})')

    # Transfer markers
    transfers = [i for i, h in enumerate(training_history) if h.get('is_transfer', False)]
    if transfers and len(transfers) < 50:
        t_c1 = [cell1_values[i] for i in transfers]
        t_c2 = [cell2_values[i] for i in transfers]
        ax.scatter(t_c1, t_c2, c='orange', s=30, marker='o', alpha=0.6, label='Transfer')

    ax.set_xlabel(f'C[{cell1[0]},{cell1[1]}]', fontsize=12)
    ax.set_ylabel(f'C[{cell2[0]},{cell2[1]}]', fontsize=12)
    ax.set_title(f'Loss Surface with Learning Trajectory\n(complexity={complexity_level}, seed={seed})', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"loss_surface_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Loss surface saved to: {path}")
    return path


def plot_learning_trajectories_pca(trajectories: dict,
                                    target_matrix: torch.Tensor,
                                    config: ScratchExperimentConfig,
                                    train_dataset: TensorDataset,
                                    save_path: str = None,
                                    zoom_target: float = None,
                                    grid_resolution: int = 80,
                                    margin: float = 0.3) -> str:
    """Plot loss surface in PCA space with multiple learning trajectories.

    Args:
        trajectories: Dict of {label: training_history} for each experiment
        target_matrix: Target matrix T' (same for all experiments)
        config: Experiment configuration
        train_dataset: Training dataset for computing loss
        save_path: Path to save figure (auto-generated if None)
        grid_resolution: Resolution of the loss surface grid
        margin: Extra margin around trajectories

    Returns:
        Path to saved figure
    """
    from sklearn.decomposition import PCA
    from torch.utils.data import DataLoader
    import numpy as np

    os.makedirs(config.results_dir, exist_ok=True)

    # Collect all C matrices from all trajectories
    all_C_flat = []
    all_is_transfer = []  # Track transfer points
    trajectory_indices = {}  # {label: (start_idx, end_idx)}

    idx = 0
    for label, history in trajectories.items():
        start_idx = idx
        for h in history:
            C = h['C_matrix'].flatten()
            all_C_flat.append(C)
            all_is_transfer.append(h.get('is_transfer', False))
            idx += 1
        trajectory_indices[label] = (start_idx, idx)

    # Add target to the data
    target_flat = target_matrix.cpu().numpy().flatten()
    all_C_flat.append(target_flat)
    target_idx = idx

    # Stack and fit PCA on trajectory data
    all_C_array = np.array(all_C_flat)
    all_is_transfer = np.array(all_is_transfer)

    # Debug: Check if C_matrix values are within [-1, 1]
    c_min, c_max = all_C_array.min(), all_C_array.max()
    print(f"[DEBUG] C_matrix range: [{c_min:.4f}, {c_max:.4f}]")
    if c_min < -1.0 or c_max > 1.0:
        print(f"[WARNING] C_matrix values outside [-1, 1] range!")

    pca = PCA(n_components=2)
    all_C_2d = pca.fit_transform(all_C_array)

    # Extract target position in 2D
    target_2d = all_C_2d[target_idx]

    # Compute valid region FIRST (zonotope projection of [-1,1]^n hypercube)
    n_dims = pca.components_.shape[1]  # output_dim * input_dim

    # Generators: projection of each axis onto 2D PCA space
    # g_i = how much moving +1 in dimension i changes the 2D projection
    generators = pca.components_.T  # Shape: (n_dims, 2)

    # Center of the zonotope in PCA space (projection of origin)
    hypercube_center = np.zeros((1, n_dims))
    center = pca.transform(hypercube_center)[0]

    # Compute zonotope boundary by finding extreme points in many directions
    # For each direction theta, the extreme point is center + sum_i sign(g_i . direction) * g_i
    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    vertices = []

    for theta in angles:
        direction = np.array([np.cos(theta), np.sin(theta)])
        # For each generator, choose sign to maximize projection onto direction
        # lambda_i = sign(g_i . direction)
        signs = np.sign(generators @ direction)
        # Handle zero case (shouldn't happen often)
        signs[signs == 0] = 1
        # Extreme point: center + sum_i lambda_i * g_i
        extreme_point = center + (signs[:, np.newaxis] * generators).sum(axis=0)
        vertices.append(extreme_point)

    vertices = np.array(vertices)

    # Remove duplicate vertices (convex hull will handle this, but good for efficiency)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
    except:
        pass  # If the convex hull cannot be computed, use all vertices

    # Use trajectory bounds for grid (zoom into relevant region)
    traj_only = all_C_2d[:target_idx]  # exclude target from grid computation
    all_pts = np.vstack([traj_only, target_2d.reshape(1, -1)])
    x_min_t, x_max_t = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min_t, y_max_t = all_pts[:, 1].min(), all_pts[:, 1].max()
    grid_margin_x = (x_max_t - x_min_t) * 0.2
    grid_margin_y = (y_max_t - y_min_t) * 0.2

    # Create 2D grid in PCA space (covering trajectory region with margin)
    x_grid = np.linspace(x_min_t - grid_margin_x, x_max_t + grid_margin_x, grid_resolution)
    y_grid = np.linspace(y_min_t - grid_margin_y, y_max_t + grid_margin_y, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Get training data
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)

    # Compute loss surface: for each 2D point, reconstruct C matrix and compute loss
    # Use target T' as the base for reconstruction so that target = loss minimum.
    # C(a,b) = T' + (a - a_target)*PC1 + (b - b_target)*PC2
    # This ensures the remaining 79 dimensions match T' exactly.
    print(f"Computing loss surface in PCA space ({grid_resolution}x{grid_resolution})...")
    Z = np.zeros_like(X_grid)

    pc1 = pca.components_[0]  # (81,)
    pc2 = pca.components_[1]  # (81,)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            da = X_grid[i, j] - target_2d[0]
            db = Y_grid[i, j] - target_2d[1]
            C_flat = target_flat + da * pc1 + db * pc2
            C_matrix = torch.tensor(C_flat.reshape(config.output_dim, config.input_dim),
                                   dtype=torch.float32)

            # Compute loss
            Y_pred = X_train @ C_matrix.T.to(DEVICE)
            Z[i, j] = F.mse_loss(Y_pred, Y_train).item()

    # Set axis limits based on trajectories (zoom in) with target included
    traj_points = all_C_2d[:target_idx]
    all_points = np.vstack([traj_points, target_2d.reshape(1, -1)])
    if zoom_target is not None:
        # Zoom into target neighborhood: zoom_target = half-width of view
        xlim = (target_2d[0] - zoom_target, target_2d[0] + zoom_target)
        ylim = (target_2d[1] - zoom_target, target_2d[1] + zoom_target)
    else:
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        margin_x = (x_max - x_min) * 0.15
        margin_y = (y_max - y_min) * 0.15
        xlim = (x_min - margin_x, x_max + margin_x)
        ylim = (y_min - margin_y, y_max + margin_y)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14})

    from matplotlib.patches import Polygon
    from matplotlib.ticker import LogLocator

    # Plot loss surface: filled contour with muted colors
    Z_safe = np.clip(Z, 1e-8, None)
    levels = np.logspace(np.log10(Z_safe.min()), np.log10(Z_safe.max()), 30)
    contour = ax.contourf(X_grid, Y_grid, Z_safe, levels=levels, cmap='viridis', alpha=0.35,
                          norm=plt.matplotlib.colors.LogNorm())
    # Subtle contour lines
    contour_lines = ax.contour(X_grid, Y_grid, Z_safe, levels=levels[::3],
                               colors='gray', alpha=0.3, linewidths=0.4)

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('log10(MSE)', fontsize=12)
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    cbar.ax.yaxis.set_major_formatter(lambda x, pos: f'{np.log10(x):.0f}' if x > 0 else '')

    # Draw valid region boundary (may be partially outside view)
    zonotope_patch = Polygon(vertices, fill=False, edgecolor='gray', linewidth=1.5,
                             linestyle='--', alpha=0.5, label='Valid region (C in [-1,1])')
    ax.add_patch(zonotope_patch)

    # Apply axis limits BEFORE drawing trajectories (clips contours to axes)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Generate distinct colors for each trajectory
    n_trajectories = len(trajectory_indices)
    traj_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'][:n_trajectories]

    for i, (label, (start, end)) in enumerate(trajectory_indices.items()):
        color = traj_colors[i % len(traj_colors)]
        traj_2d = all_C_2d[start:end]
        traj_transfers = all_is_transfer[start:end]

        # Plot trajectory line
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-', color=color, linewidth=1.2, alpha=0.85, label=label)

        # Mark transfer points (all transfers, capped at 5000 for SVG size)
        transfer_idx = np.where(traj_transfers.astype(bool))[0]
        if len(transfer_idx) > 0:
            if len(transfer_idx) > 5000:
                transfer_idx = transfer_idx[::len(transfer_idx) // 5000]
            ax.scatter(traj_2d[transfer_idx, 0], traj_2d[transfer_idx, 1],
                       color=color, s=12, marker='o', alpha=0.6, edgecolors='black',
                       linewidths=0.2, zorder=4)

        # Mark start point (circle)
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=color, s=50, marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5)
        # Mark end point (star)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=color, s=100, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=6)

    # Plot target
    ax.scatter(target_2d[0], target_2d[1], c='yellow', s=100, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10, label='Target')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
    ax.set_title('Loss Surface with Learning Trajectories (PCA)', fontsize=14, fontweight='bold')
    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    for line in leg.get_lines():
        line.set_linewidth(3.0)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(config.results_dir, f"loss_surface_pca_{timestamp}.svg")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Loss surface with trajectories saved to: {save_path}")
    return save_path


def plot_learning_trajectories_pca_3d(trajectories: dict,
                                       target_matrix: torch.Tensor,
                                       config: ScratchExperimentConfig,
                                       train_dataset: TensorDataset,
                                       save_path: str = None,
                                       grid_resolution: int = 50) -> str:
    """Plot 3D learning trajectories in PCA space.

    Args:
        trajectories: Dict of {label: training_history} for each experiment
        target_matrix: Target matrix T' (same for all experiments)
        config: Experiment configuration
        train_dataset: Training dataset for computing loss
        save_path: Path to save figure (auto-generated if None)
        grid_resolution: Resolution of the loss surface grid

    Returns:
        Path to saved figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA

    os.makedirs(config.results_dir, exist_ok=True)

    # Collect all C matrices from all trajectories
    all_C_flat = []
    all_is_transfer = []  # Track transfer points
    trajectory_indices = {}  # {label: (start_idx, end_idx)}

    idx = 0
    for label, history in trajectories.items():
        start_idx = idx
        for h in history:
            C = h['C_matrix'].flatten()
            all_C_flat.append(C)
            all_is_transfer.append(h.get('is_transfer', False))
            idx += 1
        trajectory_indices[label] = (start_idx, idx)

    # Add target to the data
    target_flat = target_matrix.cpu().numpy().flatten()
    all_C_flat.append(target_flat)
    target_idx = idx

    # Stack and fit PCA with 3 components
    all_C_array = np.array(all_C_flat)
    all_is_transfer = np.array(all_is_transfer)

    # Debug: Check if C_matrix values are within [-1, 1]
    c_min, c_max = all_C_array.min(), all_C_array.max()
    print(f"[DEBUG 3D] C_matrix range: [{c_min:.4f}, {c_max:.4f}]")
    if c_min < -1.0 or c_max > 1.0:
        print(f"[WARNING 3D] C_matrix values outside [-1, 1] range!")

    pca = PCA(n_components=3)
    all_C_3d = pca.fit_transform(all_C_array)

    # Separate target from trajectories
    target_3d = all_C_3d[target_idx]
    all_C_3d = all_C_3d[:target_idx]

    # Compute valid region (zonotope projection of [-1,1]^n hypercube)
    # Use same algorithm as 2D: find extreme points in many directions on the sphere
    from scipy.spatial import ConvexHull, Delaunay
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n_dims = pca.components_.shape[1]  # output_dim * input_dim

    # Generators: projection of each axis onto 3D PCA space
    generators = pca.components_.T  # Shape: (n_dims, 3)

    # Center of the zonotope in PCA space (projection of origin)
    hypercube_center = np.zeros((1, n_dims))
    center = pca.transform(hypercube_center)[0]

    # Sample directions on unit sphere using Fibonacci lattice for uniform distribution
    n_directions = 1000
    indices = np.arange(n_directions, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1 - (indices / (n_directions - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    directions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), y])

    # For each direction, compute extreme point of zonotope
    vertices = []
    for direction in directions:
        # lambda_i = sign(g_i . direction)
        signs = np.sign(generators @ direction)
        signs[signs == 0] = 1
        # Extreme point: center + sum_i lambda_i * g_i
        extreme_point = center + (signs[:, np.newaxis] * generators).sum(axis=0)
        vertices.append(extreme_point)

    corners_3d = np.array(vertices)

    # Compute convex hull for valid region check
    hull = ConvexHull(corners_3d)
    delaunay = Delaunay(corners_3d[hull.vertices])

    # Get training data
    X_train = train_dataset.tensors[0].to(DEVICE)
    Y_train = train_dataset.tensors[1].to(DEVICE)

    # Compute 3D loss grid inside valid region
    # Use bounds from the convex hull
    pc1_min, pc1_max = corners_3d[:, 0].min(), corners_3d[:, 0].max()
    pc2_min, pc2_max = corners_3d[:, 1].min(), corners_3d[:, 1].max()
    pc3_min, pc3_max = corners_3d[:, 2].min(), corners_3d[:, 2].max()

    # Create 3D grid (lower resolution for 3D)
    res_3d = min(grid_resolution, 25)
    x = np.linspace(pc1_min, pc1_max, res_3d)
    y = np.linspace(pc2_min, pc2_max, res_3d)
    z = np.linspace(pc3_min, pc3_max, res_3d)

    # Collect points inside valid region and compute their loss
    points_inside = []
    loss_values = []

    print(f"Computing 3D loss values ({res_3d}^3 grid)...")
    for xi in x:
        for yi in y:
            for zi in z:
                point = np.array([xi, yi, zi])
                # Check if point is inside convex hull
                if delaunay.find_simplex(point) >= 0:
                    # Compute loss at this point
                    pc_point = point.reshape(1, -1)
                    C_flat = pca.inverse_transform(pc_point)
                    C_matrix = torch.tensor(C_flat.reshape(config.output_dim, config.input_dim),
                                            dtype=torch.float32)
                    Y_pred = X_train @ C_matrix.T.to(DEVICE)
                    loss = F.mse_loss(Y_pred, Y_train).item()
                    points_inside.append(point)
                    loss_values.append(loss)

    points_inside = np.array(points_inside)
    loss_values = np.array(loss_values)

    # Create figure (publication-quality settings)
    fig = plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})
    ax = fig.add_subplot(111, projection='3d')

    # Plot loss as scatter with color (inside valid region only)
    if len(points_inside) > 0:
        # Use log scale for loss colors
        loss_log = np.log10(np.clip(loss_values, 1e-8, None))
        scatter = ax.scatter(points_inside[:, 0], points_inside[:, 1], points_inside[:, 2],
                            c=loss_log, cmap='viridis', alpha=0.3, s=15,
                            vmin=loss_log.min(), vmax=loss_log.max())
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('log10(MSE)', fontsize=11)
        # Format colorbar ticks with fewer decimals
        from matplotlib.ticker import FormatStrFormatter
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Generate distinct colors for each trajectory using colormap
    n_trajectories = len(trajectory_indices)
    cmap = plt.cm.get_cmap('tab20', n_trajectories)
    colors = [cmap(i) for i in range(n_trajectories)]

    for i, (label, (start, end)) in enumerate(trajectory_indices.items()):
        color = colors[i]
        traj_3d = all_C_3d[start:end]
        traj_transfers = all_is_transfer[start:end]

        # Plot trajectory line (thinner)
        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                '-', color=color, linewidth=0.6, alpha=0.7, label=label)

        # Mark transfer points with dots (on top of line)
        transfer_mask = traj_transfers.astype(bool)
        if transfer_mask.any():
            ax.scatter(traj_3d[transfer_mask, 0], traj_3d[transfer_mask, 1], traj_3d[transfer_mask, 2],
                      color=color, s=10, marker='o', edgecolors='white', linewidths=0.15, alpha=1.0, zorder=7)

        # Mark start point (circle)
        ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
                   color=color, s=30, marker='o', edgecolors='black', linewidths=0.4)
        # Mark end point (star)
        ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
                   color=color, s=60, marker='*', edgecolors='black', linewidths=0.4)

    # Plot target (smaller marker)
    ax.scatter(target_3d[0], target_3d[1], target_3d[2],
               c='yellow', s=70, marker='X', edgecolors='black', linewidths=1.5, label='Target')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
    ax.set_title('Learning Trajectories in PCA Space', fontsize=13, fontweight='bold')
    leg = ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    # Make legend lines thicker
    for line in leg.get_lines():
        line.set_linewidth(3.0)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(config.results_dir, f"loss_surface_pca_3d_{timestamp}.svg")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f"3D Loss surface with trajectories saved to: {save_path}")
    return save_path


def run_trajectory_comparison(param_configs: list,
                               complexity_level: str = 'medium',
                               seed: int = 42,
                               grid_resolution: int = 80) -> str:
    """Run multiple experiments with different parameters and compare trajectories.

    Creates a single figure with loss surface in PCA space and all learning
    trajectories overlaid.

    Args:
        param_configs: List of dicts with parameter overrides and 'label' key
                      e.g., [{'label': 't=1', 'lrtt_transfer_every': 1},
                             {'label': 't=10', 'lrtt_transfer_every': 10}]
        complexity_level: Complexity level for target matrix
        seed: Random seed
        grid_resolution: Resolution for loss surface grid

    Returns:
        Path to saved figure
    """
    from torch.utils.data import DataLoader

    trajectories = {}
    base_config = ScratchExperimentConfig()

    # Generate target matrix and dataset (same for all experiments)
    target_matrix = generate_target_matrix(complexity_level, base_config, seed)
    train_dataset = generate_target_dataset(complexity_level, base_config, train=True, seed=seed)

    for param_config in param_configs:
        label = param_config.pop('label', str(param_config))

        # Create config with overrides
        config = ScratchExperimentConfig()
        for key, value in param_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Disable verbose output
        config.log_ab_scaling = False

        # Generate dataset for this config
        train_ds = generate_target_dataset(complexity_level, config, train=True, seed=seed)
        val_ds = generate_target_dataset(complexity_level, config, train=False, seed=seed)

        train_loader = DataLoader(train_ds, batch_size=config.lrtt_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.lrtt_batch_size, shuffle=False) if val_ds else None

        print(f"\n{'='*60}")
        print(f"Running experiment: {label}")
        print(f"{'='*60}")

        # Reset random state before each experiment to ensure consistent initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Train
        model, training_history, epoch_history, _, _, _ = train_lrtt_scratch(
            config, train_loader, val_loader, seed=seed, use_wandb=False
        )

        trajectories[label] = training_history

        # Restore label for next iteration
        param_config['label'] = label

    # Plot 2D figure with loss surface + all trajectories
    save_path_2d = plot_learning_trajectories_pca(
        trajectories=trajectories,
        target_matrix=target_matrix,
        config=base_config,
        train_dataset=train_dataset,
        grid_resolution=grid_resolution
    )

    # Plot 3D figure with trajectories
    save_path_3d = plot_learning_trajectories_pca_3d(
        trajectories=trajectories,
        target_matrix=target_matrix,
        config=base_config,
        train_dataset=train_dataset,
        grid_resolution=min(grid_resolution, 50)  # Lower resolution for 3D (faster)
    )

    return save_path_2d, save_path_3d


def save_experiment_details_to_excel(config: ScratchExperimentConfig,
                                     complexity_level: str,
                                     seed: int,
                                     target_matrix: torch.Tensor,
                                     train_dataset: TensorDataset,
                                     C_init: torch.Tensor,
                                     A_init: torch.Tensor,
                                     B_init: torch.Tensor,
                                     training_history: list,
                                     timestamp: str,
                                     final_r2: float = None) -> None:
    """Save detailed experiment data to a single Excel file with multiple sheets."""
    import math

    # Create results directory if needed
    os.makedirs(config.results_dir, exist_ok=True)

    # Excel file path
    excel_path = os.path.join(
        config.results_dir,
        f"experiment_{complexity_level}_seed{seed}_{timestamp}.xlsx"
    )

    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # 1. Hyperparameters (simplified and reorganized)
        # Note: Scaling parameters use Excel notation (swapped from code)
        # Code A tile -> Excel B tile (up-projection)
        # Code B tile -> Excel A tile (down-projection)
        param_names = [
            'loss_function',
            'rank',
            'transfer_lr',
            'transfer_every',
            'input_type',
            'complexity',
            'seed',
            'c_init_value',
            'a_init_mode',
            'b_init_mode',
            'reinit_mode',
            'use_6t1c',
            'a_lifetime',
            'b_lifetime',
            'include_retention',
            'desired_bl',
            'dw_min',
            'pulse_type',
            'a_x_scaling',
            'a_d_scaling',
            'b_x_scaling',
            'b_d_scaling',
            'use_manual_scaling',
            'lrtt_lr',
            'quantize_x',
            'quantize_d',
            'x_resolution',
            'd_resolution',
            'c_device_type',
            'c_dw_min',
            'c_desired_bl',
            'c_w_min',
            'c_w_max',
            'transfer_method',
            'transfer_rank_schedule',
            'transfer_ranks_per_step',
            'batch_size',
            'epochs',
            'patience',
            'grad_clip',
            'noise_std',
            'train_samples',
            'test_samples',
            'input_dim',
            'output_dim',
            'total_steps',
        ]

        # Calculate total steps from training history
        total_steps = training_history[-1]['step'] if training_history else 0

        param_values = [
            'MSE (Mean Squared Error)',
            config.lrtt_rank,
            config.lora_alpha,  # transfer_lr = lora_alpha
            config.lrtt_transfer_every,
            config.input_type,
            complexity_level,
            seed,
            config.c_init_value,
            config.b_init_mode,  # Excel A = Code B (down-projection)
            config.a_init_mode,  # Excel B = Code A (up-projection)
            config.reinit_mode,
            config.USE_6T1C_AB,
            config.lifetime if config.USE_6T1C_AB else None,
            (config.b_lifetime if config.b_lifetime is not None else config.lifetime) if config.USE_6T1C_AB else None,
            config.include_retention if config.USE_6T1C_AB else None,
            config.desired_bl,
            config.ab_dw_min,
            str(config.pulse_type),
            config.b_x_scaling,  # Excel A = Code B (down-projection)
            config.b_d_scaling,  # Excel A = Code B
            config.a_x_scaling,  # Excel B = Code A (up-projection)
            config.a_d_scaling,  # Excel B = Code A
            config.use_manual_scaling,
            config.lrtt_lr,
            config.quantize_x,
            config.quantize_d,
            config.x_resolution if config.quantize_x else None,
            config.d_resolution if config.quantize_d else None,
            config.c_device_type,
            config.c_dw_min,
            config.c_desired_bl,
            config.c_w_min,
            config.c_w_max,
            config.transfer_method,
            config.transfer_rank_schedule,
            config.transfer_ranks_per_step,
            config.lrtt_batch_size,
            config.lrtt_epochs,
            config.lrtt_patience,
            config.lrtt_grad_clip,
            config.noise_std,
            config.D_prime_train_size,
            config.D_prime_test_size,
            config.input_dim,
            config.output_dim,
            total_steps,
        ]
        param_descs = [
            '(1/N) * sum((y_pred - y_target)^2)',
            'LoRA rank',
            'Transfer rate lora_alpha (multiplied by lrtt_lr, scale_transfer_lr=True)',
            'Transfer frequency (steps)',
            'Input data type (continuous/ternary/binary)',
            'Target complexity level',
            'Random seed',
            'C matrix initial value (all elements)',
            'A (down-proj) init mode (zero/kaiming)',
            'B (up-proj) init mode (zero/kaiming)',
            'Reinit mode (standard/decay/hybrid)',
            'Use 6T1C device for A/B matrices',
            'A tile batch lifetime for retention decay',
            'B tile batch lifetime for retention decay',
            'Include retention effects',
            'Bit length (pulse train length) for A/B',
            'A/B tile minimum weight update step',
            'Pulse generation type',
            'A tile (down-proj) x scaling',
            'A tile (down-proj) d scaling',
            'B tile (up-proj) x scaling',
            'B tile (up-proj) d scaling',
            'Use manual scaling (True) or dynamic scaling (False)',
            'Optimizer learning rate (update size when use_manual_scaling=False; multiplies transfer_lr)',
            'Enable input (x) quantization',
            'Enable gradient (d) quantization',
            'Input quantization resolution',
            'Gradient quantization resolution',
            'C matrix device type (idealized/floating_point/softbounds/linearstep)',
            'C matrix minimum weight update step',
            'C matrix bit length for transfer',
            'C matrix minimum weight value',
            'C matrix maximum weight value',
            'C transfer method (set/onehot/direct)',
            'Transfer rank schedule (all/round_robin)',
            'Ranks transferred per step (round_robin only)',
            'Training batch size',
            'Maximum training epochs',
            'Early stopping patience',
            'Gradient clipping value',
            'Data noise standard deviation',
            'Number of training samples',
            'Number of test samples',
            'Input dimension',
            'Output dimension',
            'Total training steps',
        ]

        hyperparams = {
            'Parameter': param_names,
            'Value': param_values,
            'Description': param_descs
        }
        pd.DataFrame(hyperparams).to_excel(writer, sheet_name='Hyperparameters', index=False)

        # 2. Target Matrix (T')
        target_df = pd.DataFrame(
            target_matrix.cpu().numpy(),
            columns=[f'x{i}' for i in range(target_matrix.shape[1])],
            index=[f'y{i}' for i in range(target_matrix.shape[0])]
        )
        target_df.to_excel(writer, sheet_name='Target_Matrix_T')

        # 3. Input Data (X) - training set
        X_train = train_dataset.tensors[0].cpu().numpy()
        input_df = pd.DataFrame(
            X_train,
            columns=[f'x{i}' for i in range(X_train.shape[1])]
        )
        input_df.to_excel(writer, sheet_name='Input_Data_X', index=False)

        # 4. Target Labels (Y) - training set
        Y_train = train_dataset.tensors[1].cpu().numpy()
        labels_df = pd.DataFrame(
            Y_train,
            columns=[f'y{i}' for i in range(Y_train.shape[1])]
        )
        labels_df.to_excel(writer, sheet_name='Target_Labels_Y', index=False)

        # 5. Initial B matrix (code A, up-projection)
        # Note: Code uses A @ B, but standard LoRA notation is B @ A
        # Code's A = up-projection [output, rank] -> Excel's B
        B_up_df = pd.DataFrame(
            A_init.cpu().numpy(),
            columns=[f'rank{i}' for i in range(A_init.shape[1])],
            index=[f'y{i}' for i in range(A_init.shape[0])]
        )
        B_up_df.to_excel(writer, sheet_name='Initial_B_up')

        # 6. Initial A matrix (code B, down-projection)
        # Code's B = down-projection [rank, input] -> Excel's A
        A_down_df = pd.DataFrame(
            B_init.cpu().numpy(),
            columns=[f'x{i}' for i in range(B_init.shape[1])],
            index=[f'rank{i}' for i in range(B_init.shape[0])]
        )
        A_down_df.to_excel(writer, sheet_name='Initial_A_down')

        # 7. Initial C matrix (Core array)
        C_df = pd.DataFrame(
            C_init.cpu().numpy(),
            columns=[f'x{i}' for i in range(C_init.shape[1])],
            index=[f'y{i}' for i in range(C_init.shape[0])]
        )
        C_df.to_excel(writer, sheet_name='Initial_C_Core')

        # 8. LoRA update dW (code: A @ B, notation: B @ A)
        # Standard notation: dW = B @ A (up @ down)
        # Code implementation: dW = A @ B
        delta_W = (A_init @ B_init).cpu().numpy()
        delta_df = pd.DataFrame(
            delta_W,
            columns=[f'x{i}' for i in range(delta_W.shape[1])],
            index=[f'y{i}' for i in range(delta_W.shape[0])]
        )
        delta_df.to_excel(writer, sheet_name='Initial_dW_LoRA')

        # 9. Notation guide
        notation_guide = pd.DataFrame({
            'Notation': ['Code', 'Code', 'Code', 'Excel', 'Excel', 'Standard LoRA'],
            'Matrix': ['A', 'B', 'A @ B', 'B_up', 'A_down', 'B @ A'],
            'Shape': ['[out, rank]', '[rank, in]', '[out, in]', '[out, rank]', '[rank, in]', '[out, in]'],
            'Description': [
                'Up-projection (output side)',
                'Down-projection (input side)',
                'LoRA update',
                'Up-projection (same as code A)',
                'Down-projection (same as code B)',
                'Standard notation: up @ down'
            ]
        })
        notation_guide.to_excel(writer, sheet_name='Notation_Guide', index=False)

        # 10. Training Summary
        if training_history:
            # Add final_r2 column (only last row has the value)
            n_steps = len(training_history)
            r2_column = [None] * n_steps
            if final_r2 is not None:
                r2_column[-1] = final_r2  # Put R^2 in the last row

            # Note: Excel uses standard LoRA notation
            # Code A_norm (up-proj) -> Excel B_norm
            # Code B_norm (down-proj) -> Excel A_norm
            summary_data = {
                'step': [h['step'] for h in training_history],
                'epoch': [h['epoch'] for h in training_history],
                'batch_idx': [h['batch_idx'] for h in training_history],
                'batch_loss': [h['batch_loss'] for h in training_history],
                'is_transfer': [h['is_transfer'] for h in training_history],
                'grad_norm': [h.get('grad_norm', None) for h in training_history],
                'A_norm': [h['B_norm'] for h in training_history],  # code B = standard A (down-proj)
                'B_norm': [h['A_norm'] for h in training_history],  # code A = standard B (up-proj)
                'C_norm': [h['C_norm'] for h in training_history],
                'delta_W_norm': [h['delta_W_norm'] for h in training_history],
                'final_r2': r2_column
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Training_Summary', index=False)

            # 11. Cell-wise values for all steps (flattened format)
            # A matrix values (each row is one step, columns are flattened A values)
            A_history = []
            B_history = []
            C_history = []

            for h in training_history:
                A_flat = h['A_matrix'].flatten()
                B_flat = h['B_matrix'].flatten()
                C_flat = h['C_matrix'].flatten()

                A_history.append(A_flat)
                B_history.append(B_flat)
                C_history.append(C_flat)

            # B_up matrix history (code A = standard B, up-projection)
            # B tile: x [rank], d [output_dim]
            B_up_cols = [f'B[{i//A_init.shape[1]},{i%A_init.shape[1]}]' for i in range(A_flat.size)]
            B_up_history_df = pd.DataFrame(A_history, columns=B_up_cols)
            B_up_history_df.insert(0, 'step', [h['step'] for h in training_history])
            B_up_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            B_up_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])

            # Add pulse vectors for B tile (up-proj)
            # x values: [rank] - input projected through A_down
            for r in range(config.lrtt_rank):
                B_up_history_df[f'x[{r}]'] = [h['B_x_vec'][r] for h in training_history]
            for r in range(config.lrtt_rank):
                B_up_history_df[f'p_x[{r}]'] = [h['B_p_x_vec'][r] for h in training_history]
            # d values: [output_dim] - gradient at output
            for i in range(config.output_dim):
                B_up_history_df[f'd[{i}]'] = [h['B_d_vec'][i] for h in training_history]
            for i in range(config.output_dim):
                B_up_history_df[f'p_d[{i}]'] = [h['B_p_d_vec'][i] for h in training_history]

            B_up_history_df.to_excel(writer, sheet_name='B_up_History', index=False)

            # A_down matrix history (code B = standard A, down-projection)
            # A tile: x [input_dim], d [rank]
            A_down_cols = [f'A[{i//B_init.shape[1]},{i%B_init.shape[1]}]' for i in range(B_flat.size)]
            A_down_history_df = pd.DataFrame(B_history, columns=A_down_cols)
            A_down_history_df.insert(0, 'step', [h['step'] for h in training_history])
            A_down_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            A_down_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])

            # Add pulse vectors for A tile (down-proj)
            # x values: [input_dim] - original input
            for j in range(config.input_dim):
                A_down_history_df[f'x[{j}]'] = [h['A_x_vec'][j] for h in training_history]
            for j in range(config.input_dim):
                A_down_history_df[f'p_x[{j}]'] = [h['A_p_x_vec'][j] for h in training_history]
            # d values: [rank] - gradient projected through B_up
            for r in range(config.lrtt_rank):
                A_down_history_df[f'd[{r}]'] = [h['A_d_vec'][r] for h in training_history]
            for r in range(config.lrtt_rank):
                A_down_history_df[f'p_d[{r}]'] = [h['A_p_d_vec'][r] for h in training_history]

            A_down_history_df.to_excel(writer, sheet_name='A_down_History', index=False)

            # C matrix history (step-wise)
            C_cols = [f'C[{i//C_init.shape[1]},{i%C_init.shape[1]}]' for i in range(C_flat.size)]
            C_history_df = pd.DataFrame(C_history, columns=C_cols)
            C_history_df.insert(0, 'step', [h['step'] for h in training_history])
            C_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            C_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])
            C_history_df.to_excel(writer, sheet_name='C_Core_History', index=False)

    print(f"\nDetailed experiment data saved to Excel file:")
    print(f"  {excel_path}")
    print(f"  Sheets:")
    print(f"    - Hyperparameters, Target_Matrix_T, Input_Data_X, Target_Labels_Y")
    print(f"    - Initial_B_up, Initial_A_down, Initial_C_Core, Initial_dW_LoRA")
    print(f"    - Notation_Guide")
    print(f"    - Training_Summary (step-wise norms and losses)")
    print(f"    - A_down_History, B_up_History, C_Core_History (cell-wise values)")
    print(f"\n  Note: Excel uses standard LoRA notation (B@A = up@down)")
    print(f"        Code A [out, rank] -> Excel B_up (up-projection)")
    print(f"        Code B [rank, in] -> Excel A_down (down-projection)")
    print(f"        Recorded {len(training_history)} steps (including the initial state)")


def run_scratch_experiment(config: ScratchExperimentConfig, complexity_level: str = 'medium',
                          seed: int = 42, use_wandb: bool = True) -> Dict[str, any]:
    """Run LRTT experiment from scratch on target dataset."""

    print(f"\n{'='*60}")
    print(f"Running SCRATCH experiment with seed={seed}, complexity_level={complexity_level}")
    print(f"REINIT CONFIG: mode={config.reinit_mode}")
    if config.USE_6T1C_AB and config.include_retention:
        _b_life = config.b_lifetime if config.b_lifetime is not None else config.lifetime
        print(f"DEVICE CONFIG: 6T1C_AB=True, a_lifetime={config.lifetime} b_lifetime={_b_life} batches")
    else:
        print(f"DEVICE CONFIG: 6T1C_AB={config.USE_6T1C_AB}, retention={'OFF' if not config.include_retention else 'N/A'}")
    if config.quantize_x or config.quantize_d:
        print(f"QUANTIZATION: x={config.quantize_x} (res={config.x_resolution}), d={config.quantize_d} (res={config.d_resolution})")
    print(f"{'='*60}")

    # Initialize wandb run for this experiment
    if use_wandb:
        run_name = f"lrtt_scratch_{complexity_level}_r{config.lrtt_rank}_t{config.lrtt_transfer_every}_alpha{config.lora_alpha}_LR{config.lrtt_lr}"

        wandb.init(
            project="regression-lrtt-scratch",
            name=run_name,
            config={
                'experiment_type': 'LRTT-scratch',
                'complexity_level': complexity_level,
                'seed': seed,
                'input_type': config.input_type,
                'lrtt_lr': config.lrtt_lr,
                'lrtt_rank': config.lrtt_rank,
                'transfer_every': config.lrtt_transfer_every,
                'lora_alpha': config.lora_alpha,
                'use_6t1c_ab': config.USE_6T1C_AB,
                'a_lifetime': config.lifetime if config.USE_6T1C_AB else None,
                'b_lifetime': (config.b_lifetime if config.b_lifetime is not None else config.lifetime) if config.USE_6T1C_AB else None,
                'include_retention': config.include_retention if config.USE_6T1C_AB else None,
                'quantize_x': config.quantize_x,
                'quantize_d': config.quantize_d,
                'x_resolution': config.x_resolution if config.quantize_x else None,
                'd_resolution': config.d_resolution if config.quantize_d else None,
            },
            reinit=True
        )

    # Generate target matrix and datasets
    target_matrix = generate_target_matrix(complexity_level, config, seed)
    target_train = generate_target_dataset(complexity_level, config, train=True, seed=seed)
    target_test = generate_target_dataset(complexity_level, config, train=False, seed=seed)
    # A custom input file sets input_dim/output_dim from its column count
    if target_matrix.shape != (config.output_dim, config.input_dim):
        target_matrix = generate_target_matrix(complexity_level, config, seed)

    # Custom input: train only 1 epoch (each sample seen once)
    if config.custom_input_file is not None:
        config.lrtt_epochs = 1
        config.lrtt_patience = 1  # No early stopping needed for 1 epoch
        print(f"[Custom Input] Training for 1 epoch ({len(target_train)} samples)")

    if use_wandb:
        wandb.log({
            'data/complexity_level': complexity_level,
        })

    # Create data loaders (no shuffle for custom input to preserve order)
    shuffle_train = config.custom_input_file is None
    target_train_loader = DataLoader(target_train, batch_size=config.lrtt_batch_size, shuffle=shuffle_train)
    target_test_loader = DataLoader(target_test, batch_size=config.lrtt_batch_size) if target_test is not None else None

    # Train LRTT from scratch on target dataset (also returns initial A, B, C)
    lrtt_model, training_history, epoch_history, C_init, A_init, B_init = train_lrtt_scratch(
        config, target_train_loader, target_test_loader, seed, use_wandb)

    # Evaluate LRTT on target dataset (skip if no test set)
    if target_test_loader is not None:
        lrtt_results = evaluate_model(lrtt_model, target_test_loader)
    else:
        # Use train set for evaluation when no test set
        lrtt_results = evaluate_model(lrtt_model, target_train_loader)

    # Compare learned matrix with target matrix
    matrix_comparison = compare_matrices(lrtt_model, target_matrix)

    print(f"\nLRTT (scratch) on target: MSE={lrtt_results['MSE']:.6f}, R^2={lrtt_results['R2']:.4f}")
    print(f"\nMatrix Comparison:")
    print(f"  Target ||T'||: {matrix_comparison['target_norm']:.4f}")
    print(f"  Learned ||W||: {matrix_comparison['learned_norm']:.4f}")
    print(f"  Frobenius diff ||W-T'||: {matrix_comparison['frobenius_diff']:.4f}")
    print(f"  Matrix MSE: {matrix_comparison['mse_matrix']:.6f}")
    print(f"  Relative error: {matrix_comparison['relative_error']:.4f} ({matrix_comparison['relative_error']*100:.1f}%)")

    # Save detailed experiment data to Excel file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_experiment_details_to_excel(
        config=config,
        complexity_level=complexity_level,
        seed=seed,
        target_matrix=target_matrix,
        train_dataset=target_train,
        C_init=C_init,
        A_init=A_init,
        B_init=B_init,
        training_history=training_history,
        timestamp=timestamp,
        final_r2=lrtt_results['R2']
    )

    # Plot and save all training figures
    if config.save_figures:
        plot_all_training_figures(
            training_history=training_history,
            epoch_history=epoch_history,
            config=config,
            complexity_level=complexity_level,
            seed=seed,
            timestamp=timestamp,
            final_mse=lrtt_results['MSE'],
            final_r2=lrtt_results['R2'],
            target_matrix=target_matrix
        )

    if use_wandb:
        wandb.log({
            'lrtt/mse': lrtt_results['MSE'],
            'lrtt/rmse': lrtt_results['RMSE'],
            'lrtt/r2': lrtt_results['R2'],
            'matrix/frobenius_diff': matrix_comparison['frobenius_diff'],
            'matrix/mse': matrix_comparison['mse_matrix'],
            'matrix/relative_error': matrix_comparison['relative_error'],
        })

        wandb.summary['final_lrtt_mse'] = lrtt_results['MSE']
        wandb.summary['final_lrtt_r2'] = lrtt_results['R2']
        wandb.summary['matrix_relative_error'] = matrix_comparison['relative_error']
        wandb.finish()

    # Compile results
    results = {
        'seed': seed,
        'complexity_level': complexity_level,
        'lrtt_results': lrtt_results,
        'matrix_comparison': matrix_comparison,
        'experiment_type': 'scratch'
    }

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================
def main(use_wandb: bool = True):
    """Main experiment runner for scratch training."""

    config = ScratchExperimentConfig()

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments across complexity levels
    all_results = []

    for complexity_level in config.complexity_levels:
        print(f"\n{'#'*60}")
        print(f"# Running SCRATCH experiments for {complexity_level.upper()} complexity level")
        print(f"{'#'*60}")

        for seed in config.seeds[:1]:  # Primary seed only
            results = run_scratch_experiment(config, complexity_level, seed, use_wandb)
            all_results.append(results)

        # Print summary for this complexity level
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        mse_lrtt = [r['lrtt_results']['MSE'] for r in level_results]
        r2_lrtt = [r['lrtt_results']['R2'] for r in level_results]
        relative_errors = [r['matrix_comparison']['relative_error'] for r in level_results]

        print(f"\n{complexity_level.upper()} SCRATCH Summary (n={len(level_results)}):")
        print(f"  LRTT MSE: {np.mean(mse_lrtt):.6f} +/- {np.std(mse_lrtt):.6f}")
        print(f"  LRTT R^2: {np.mean(r2_lrtt):.4f} +/- {np.std(r2_lrtt):.4f}")
        print(f"  Matrix Relative Error: {np.mean(relative_errors):.4f} +/- {np.std(relative_errors):.4f} ({np.mean(relative_errors)*100:.1f}%)")

    # Save results to JSON
    results_file = os.path.join(config.results_dir, f"lrtt_scratch_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print final summary table
    print("\n" + "="*75)
    print("SCRATCH EXPERIMENT SUMMARY")
    print("="*75)
    print(f"{'Complexity':<12} {'LRTT MSE':<15} {'LRTT R^2':<15} {'Matrix Error':<15}")
    print("-" * 75)

    for complexity_level in config.complexity_levels:
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        if level_results:
            lrtt_mse = np.mean([r['lrtt_results']['MSE'] for r in level_results])
            lrtt_r2 = np.mean([r['lrtt_results']['R2'] for r in level_results])
            rel_error = np.mean([r['matrix_comparison']['relative_error'] for r in level_results])

            print(f"{complexity_level.upper():<12} {lrtt_mse:<15.6f} {lrtt_r2:<15.4f} {rel_error:<15.4f}")

    print("="*75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LRTT Scratch Experiment')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override lrtt_epochs (maximum number of training epochs)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Output directory (default: results/lrtt_scratch_decay)')
    args = parser.parse_args()

    if args.epochs is not None:
        ScratchExperimentConfig.lrtt_epochs = args.epochs
    if args.results_dir is not None:
        ScratchExperimentConfig.results_dir = args.results_dir

    config = ScratchExperimentConfig()

    if config.save_figures:
        # ============================================================
        # Asymmetry (gamma) sweeps of the core tile and of the A/B tiles.
        # Common setting: rank 4, transfer every 100 updates, lora_alpha 0.43,
        # lrtt_lr 0.00879, non-ideal I/O, stochastic pulses, one-hot transfer,
        # 6T1C A/B tiles, linear-step core, no retention decay
        # ============================================================
        _hw = {
            'lrtt_transfer_every': 100,
            'lrtt_rank': 4,
            'lora_alpha': 0.43,
            'lrtt_lr': 0.00879,
            'lrtt_epochs': 2000,
            'is_perfect': False,
            'pulse_type': PulseType.STOCHASTIC_COMPRESSED,
            'transfer_method': 'onehot',
            'USE_6T1C_AB': True,
            'c_device_type': 'linearstep',
            'include_retention': True,
            'lifetime': 0, 'b_lifetime': 0,
        }

        # Default gamma (from 6T1C fitting)
        _g0 = {'ab_gamma_up': -0.1678, 'ab_gamma_down': 0.1410,
                'c_gamma_up': -0.1678, 'c_gamma_down': 0.1410}

        # ============================================================
        # Plot A: C gamma (nonlinearity) sweep, AB gamma fixed at default
        # gamma sign: negative=potentiation slope, positive=depression slope
        # ============================================================
        c_gamma_configs = [
            {'label': 'C gamma=default', **_g0, **_hw},
            {'label': 'C gamma=1.0', 'ab_gamma_up': -0.1678, 'ab_gamma_down': 0.1410,
             'c_gamma_up': -1.0, 'c_gamma_down': 1.0, **_hw},
            {'label': 'C gamma=2.0', 'ab_gamma_up': -0.1678, 'ab_gamma_down': 0.1410,
             'c_gamma_up': -2.0, 'c_gamma_down': 2.0, **_hw},
            {'label': 'C gamma=5.0', 'ab_gamma_up': -0.1678, 'ab_gamma_down': 0.1410,
             'c_gamma_up': -5.0, 'c_gamma_down': 5.0, **_hw},
        ]

        print("\n" + "#"*60)
        print("# Plot A: C array gamma (nonlinearity) sweep")
        print("#"*60)
        run_multi_param_experiments(
            param_configs=c_gamma_configs,
            complexity_level='medium',
            seed=config.primary_seed,
            use_wandb=not args.no_wandb
        )

        # ============================================================
        # Plot B: AB gamma (nonlinearity) sweep, C gamma fixed at default
        # ============================================================
        ab_gamma_configs = [
            {'label': 'AB gamma=default', **_g0, **_hw},
            {'label': 'AB gamma=1.0', 'c_gamma_up': -0.1678, 'c_gamma_down': 0.1410,
             'ab_gamma_up': -1.0, 'ab_gamma_down': 1.0, **_hw},
            {'label': 'AB gamma=2.0', 'c_gamma_up': -0.1678, 'c_gamma_down': 0.1410,
             'ab_gamma_up': -2.0, 'ab_gamma_down': 2.0, **_hw},
            {'label': 'AB gamma=5.0', 'c_gamma_up': -0.1678, 'c_gamma_down': 0.1410,
             'ab_gamma_up': -5.0, 'ab_gamma_down': 5.0, **_hw},
        ]

        print("\n" + "#"*60)
        print("# Plot B: AB array gamma (nonlinearity) sweep")
        print("#"*60)
        run_multi_param_experiments(
            param_configs=ab_gamma_configs,
            complexity_level='medium',
            seed=config.primary_seed,
            use_wandb=not args.no_wandb
        )
    else:
        # Single experiment with config defaults
        main(use_wandb=not args.no_wandb)
