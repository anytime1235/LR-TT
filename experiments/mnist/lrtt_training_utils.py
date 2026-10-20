#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Helpers shared by the MNIST LR-TT scripts.

- ``get_lrtt_tiles``: the LR-TT tiles of a model
- ``sync_fast_lr``: sets the A/B update rate of every LR-TT tile to the current
  optimizer learning rate (called at the start of every epoch)
- ``reinit_b_gaussian``: draws the initial B from a Gaussian distribution
  (called once right after model construction)

See README.md, "Common training setting".
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import torch
import torch.nn as nn
from typing import Dict, Any

from aihwkit.nn import AnalogSequential


def get_lrtt_tiles(model: AnalogSequential) -> Dict[str, Any]:
    """Extract LRTT tiles from model.

    Args:
        model: AnalogSequential model

    Returns:
        Dict mapping layer names to their LRTT tile objects
    """
    tiles = {}
    for name, module in model.named_modules():
        # Check for analog_module (LRTT uses this instead of analog_tile)
        if hasattr(module, 'analog_module'):
            tile = module.analog_module
            if hasattr(tile, 'controller') and hasattr(tile, 'get_lrtt_component_weights'):
                tiles[name] = tile
    return tiles


def sync_fast_lr(model: nn.Module, lr: float) -> None:
    """Set the A/B update learning rate of every LRTT tile to ``lr``.

    The controller updates A and B with its ``fast_lr``. The MNIST scripts use
    the scheduled SGD learning rate (StepLR, halved every 10 epochs) as the
    A/B update rate and call this at the start of every epoch.

    Args:
        model: Model containing LRTT tiles
        lr: Current optimizer learning rate
    """
    for tile in get_lrtt_tiles(model).values():
        tile.controller.fast_lr = lr


def reinit_b_gaussian(model: nn.Module, reinit_gain: float) -> None:
    """Redraw the initial B matrices from N(0, (reinit_gain * sqrt(2 / x_size))^2).

    Replaces the Kaiming-uniform B drawn by the controller. Called once right
    after model construction; A is zero, so the initial product A @ B stays zero.

    Args:
        model: Model containing LRTT tiles
        reinit_gain: Gain of the Gaussian initialization
    """
    with torch.no_grad():
        for tile in get_lrtt_tiles(model).values():
            ctrl = tile.controller
            std = reinit_gain * math.sqrt(2.0 / ctrl.x_size)
            b_init = torch.normal(0.0, std, size=(ctrl.rank, ctrl.x_size),
                                  device=ctrl.device, dtype=ctrl.dtype)
            tile.tile_b.set_weights(b_init)
