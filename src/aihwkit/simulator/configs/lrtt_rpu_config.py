# -*- coding: utf-8 -*-

# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT RPU configuration based on ``SingleRPUConfig``.

This ``PythonLRTTRPUConfig`` is exported by ``aihwkit.simulator.configs`` and
used by the BERT-base scripts; ``lrtt_config.py`` has a variant of the same
name based on ``IOManagedRPUConfig``. Both create an ``LRTTSimulatorTile``.
"""

from dataclasses import dataclass
from typing import Type

from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


@dataclass
class PythonLRTTRPUConfig(SingleRPUConfig):
    """RPU configuration of an LR-TT layer.
    
    This configuration uses the LRTTSimulatorTile implementation which
    orchestrates three analog tiles for Low-Rank Tiki-Taka (LR-TT) training.
    Works with both CPU and GPU versions of aihwkit.
    """
    
    tile_class: Type = LRTTSimulatorTile
    """Tile class that corresponds to this RPUConfig."""
    
    device: PythonLRTTDevice = None
    """Device parameters for LRTT."""
    
    def __post_init__(self):
        """Initialize default device if not provided."""
        if self.device is None:
            self.device = PythonLRTTDevice()
    
    def get_brief_info(self) -> str:
        """Get brief configuration info."""
        return (f"PythonLRTTRPUConfig(rank={self.device.rank}, "
                f"transfer_every={self.device.transfer_every}, "
                f"lora_alpha={self.device.lora_alpha})")