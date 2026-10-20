# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""LR-TT RPU configuration based on ``IOManagedRPUConfig`` and configuration helpers.

This ``PythonLRTTRPUConfig`` is used by the MNIST and regression scripts and the
smoke test; ``aihwkit.simulator.configs`` exports the ``SingleRPUConfig``-based
variant of ``lrtt_rpu_config.py``. Both create an ``LRTTSimulatorTile`` from a
``PythonLRTTDevice``. The ``lrtt_*_config()`` helpers wrap the presets of
``PythonLRTTPreset``.
"""

from dataclasses import dataclass, field
from typing import Type, Any, TYPE_CHECKING

from aihwkit.simulator.configs.configs import IOManagedRPUConfig
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.enums import RPUDataType

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


def _get_lrtt_tile_class():
    """Lazy import to avoid circular dependency."""
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
    return LRTTSimulatorTile


@dataclass
class PythonLRTTRPUConfig(IOManagedRPUConfig):
    """RPU configuration of an LR-TT layer.

    The forward, backward and update parameters of this configuration are
    passed to the A, B and C tiles (with the per-tile settings of ``device``);
    the tile mappings come from ``device.mapping_ab``, ``mapping_a``,
    ``mapping_b`` and ``mapping_c``. The tile class is LRTTSimulatorTile.
    """

    tile_class: Type = field(default_factory=_get_lrtt_tile_class)
    """Default tile class: LRTTSimulatorTile (may be overridden by device config)."""
    
    tile_array_class: Type = TileModuleArray
    """Tile array class for multi-tile scenarios."""
    
    device: PythonLRTTDevice = field(default_factory=PythonLRTTDevice)
    """Python LRTT device configuration."""
    
    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Return tile class from device configuration.

        Delegates to device.get_default_tile_module_class() if available,
        so that a device class can provide its own tile class.
        """
        if hasattr(self.device, 'get_default_tile_module_class'):
            return self.device.get_default_tile_module_class()
        return _get_lrtt_tile_class()
    
    def create_tile(self, x_size: int, d_size: int, dtype: RPUDataType = RPUDataType.FLOAT, **kwargs):
        """Create LRTT tile with this configuration.

        Uses get_default_tile_module_class() to determine the tile class.

        Args:
            x_size: Input size
            d_size: Output size
            dtype: Data type for tiles
            **kwargs: Additional arguments

        Returns:
            Configured LRTT tile instance (LRTTSimulatorTile by default)
        """
        tile_class = self.get_default_tile_module_class(out_size=d_size, in_size=x_size)
        return tile_class(
            x_size=x_size,
            d_size=d_size,
            rpu_config=self,
            dtype=dtype,
            **kwargs
        )
    
    def validate_dimensions(self, d_size: int, x_size: int) -> None:
        """Validate that dimensions are compatible with rank.
        
        Args:
            d_size: Output dimension
            x_size: Input dimension
            
        Raises:
            ValueError: If rank is incompatible with dimensions
        """
        if self.device.rank > min(d_size, x_size):
            raise ValueError(
                f"Rank {self.device.rank} too large for dimensions {d_size}x{x_size}. "
                f"Maximum rank is {min(d_size, x_size)}"
            )
    
    def get_brief_info(self) -> str:
        """Get brief configuration info."""
        return (f"PythonLRTTRPUConfig(rank={self.device.rank}, "
                f"transfer_every={self.device.transfer_every}, "
                f"lora_alpha={self.device.lora_alpha})")
    
    def as_bindings(self) -> Any:
        """Return self; the LR-TT tile builds the bindings of its three tiles itself."""
        return self


# Convenience factory functions for common configurations

def lrtt_idealized_config(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTRPUConfig:
    """Create idealized LRTT configuration.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency  
        lora_alpha: LoRA scaling factor
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.idealized(rank=rank, transfer_every=transfer_every, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


def lrtt_constant_step_config(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with ConstantStepDevice.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency
        dw_min: Minimum weight update step
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.constant_step(rank=rank, transfer_every=transfer_every, dw_min=dw_min)
    return PythonLRTTRPUConfig(device=device)


def lrtt_lora_style_config(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTRPUConfig:
    """LR-TT configuration of PythonLRTTPreset.lora_style().

    IdealizedPresetDevice for all tiles, correct_gradient_magnitudes on.

    Args:
        rank: Rank of the auxiliary factors (default 8)
        lora_alpha: Scale alpha of A @ B (default 16)
        transfer_every: Transfer period (default 1, a transfer after every update)

    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.lora_style(rank=rank, lora_alpha=lora_alpha, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_mixed_precision_config(rank: int = 4, transfer_every: int = 16) -> PythonLRTTRPUConfig:
    """LR-TT configuration of PythonLRTTPreset.mixed_precision().

    Constant-step devices: dw_min 0.05 on [-0.8, 0.8] for A and B, dw_min 0.001
    on [-2, 2] for C.

    Args:
        rank: Rank of the auxiliary factors
        transfer_every: Transfer period

    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.mixed_precision(rank=rank, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_inference_config(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTRPUConfig:
    """LR-TT configuration of PythonLRTTPreset.inference_optimized().

    A transfer after every update, reinit_gain 0 (B starts at zero, A at the
    device reset), no forward injection, IdealizedPresetDevice for all tiles.

    Args:
        rank: Rank of the auxiliary factors (default 2)
        lora_alpha: Scale alpha of A @ B (default 0.5)

    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.inference_optimized(rank=rank, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


# =============================================================================
# Floating Point AB Configurations
# =============================================================================

def lrtt_floating_ab_softbound_c_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dw_min: float = 0.001,
    lifetime: float = 0.0,
) -> PythonLRTTRPUConfig:
    """Create LRTT config with FloatingPoint A/B and SoftBounds C.

    - A, B tiles: FloatingPointDevice (exact arithmetic, no noise)
    - C tile: SoftBoundsReferenceDevice (noise=0, w_max=1, w_min=-1)

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dw_min: Minimum weight update step for C tile
        lifetime: Retention lifetime for C tile (0 = no decay)

    Returns:
        Configured PythonLRTTRPUConfig with FloatingPoint A/B and SoftBounds C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.floating_ab_softbound_c(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dw_min=dw_min,
        lifetime=lifetime,
    )
    return PythonLRTTRPUConfig(device=device)


# =============================================================================
# 6T1C Device Configurations
# =============================================================================

def lrtt_sixt1c_ab_pcm_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and PCM C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and PCM C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_pcm(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_ab_rram_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and RRAM C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and RRAM C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_rram(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_ab_ideal_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and Idealized C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and Idealized C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_all_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0,
    include_retention: bool = True
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C devices for ALL tiles (A, B, C).

    All three tiles use identical 6T1C device characteristics.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds (for retention)
        include_retention: Whether to include retention effects

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C for all tiles
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_all(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec,
        include_retention=include_retention
    )
    return PythonLRTTRPUConfig(device=device)


# Conversion from other configuration objects
def lrtt_config_from_attributes(source) -> PythonLRTTRPUConfig:
    """Create a PythonLRTTRPUConfig from the LR-TT attributes of another object.

    Args:
        source: any object with rank, transfer_every, transfer_lr, lora_alpha,
            ... attributes (see PythonLRTTDevice.from_attributes)

    Returns:
        PythonLRTTRPUConfig with these attributes
    """
    python_device = PythonLRTTDevice.from_attributes(source)
    return PythonLRTTRPUConfig(device=python_device)