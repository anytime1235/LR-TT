# -*- coding: utf-8 -*-

# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Smoke tests for the Low-Rank Tiki-Taka (LR-TT) tile."""

from unittest import SkipTest

from torch import device as torch_device, isfinite, manual_seed, randn
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

from .helpers.testcases import AihwkitTestCase, SKIP_CUDA_TESTS

IN_FEATURES = 8
OUT_FEATURES = 6
BATCH_SIZE = 4
RANK = 2
TRANSFER_EVERY = 2
NUM_STEPS = 6


def get_lrtt_rpu_config(forward_inject: bool = False) -> PythonLRTTRPUConfig:
    """Return a small LR-TT configuration with pulsed A, B and C tiles."""
    device = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TRANSFER_EVERY,
        units_in_mbatch=True,
        lora_alpha=1.0,
        forward_inject=forward_inject,
        unit_cell_devices=[ConstantStepDevice(), ConstantStepDevice(), ConstantStepDevice()],
    )
    device.transfer_lr = 1.0
    return PythonLRTTRPUConfig(device=device)


class LRTTSmokeTest(AihwkitTestCase):
    """Build an LR-TT layer, train it for a few steps and check the transfer."""

    def train_layer(self, use_cuda: bool, forward_inject: bool) -> None:
        """Run several SGD steps on an analog linear layer with an LR-TT tile."""
        manual_seed(1)
        layer = AnalogLinear(
            IN_FEATURES, OUT_FEATURES, bias=True, rpu_config=get_lrtt_rpu_config(forward_inject)
        )
        if use_cuda:
            layer = layer.cuda()
        tile = layer.analog_module
        self.assertIsInstance(tile, LRTTSimulatorTile)
        # Place the controller on the device of the layer.
        tile.controller.set_device(torch_device("cuda" if use_cuda else "cpu"))

        visible_before, _, _ = [w.clone() for w in tile.get_lrtt_component_weights()]
        optimizer = AnalogSGD(layer.parameters(), lr=0.1)
        optimizer.regroup_param_groups(layer)

        x_input = randn(BATCH_SIZE, IN_FEATURES)
        target = randn(BATCH_SIZE, OUT_FEATURES)
        if use_cuda:
            x_input, target = x_input.cuda(), target.cuda()

        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            output = layer(x_input)
            self.assertTrue(isfinite(output).all())
            loss = mse_loss(output, target)
            self.assertTrue(isfinite(loss).all())
            loss.backward()
            optimizer.step()

        # transfer_every is in mini-batch units, so NUM_STEPS // TRANSFER_EVERY transfers.
        self.assertEqual(tile.controller.num_transfers, NUM_STEPS // TRANSFER_EVERY)

        visible_after, a_weights, b_weights = tile.get_lrtt_component_weights()
        self.assertEqual(tuple(visible_after.shape), (OUT_FEATURES, IN_FEATURES))
        self.assertEqual(tuple(a_weights.shape), (OUT_FEATURES, RANK))
        self.assertEqual(tuple(b_weights.shape), (RANK, IN_FEATURES))
        self.assertTrue(isfinite(visible_after).all())
        self.assertTrue(isfinite(a_weights).all())
        self.assertTrue(isfinite(b_weights).all())
        self.assertNotAlmostEqualTensor(visible_before, visible_after)

        weights, _ = tile.get_weights()
        self.assertTensorAlmostEqual(weights, visible_after)

    def test_train_cpu(self):
        """Train on CPU without forward injection."""
        self.train_layer(use_cuda=False, forward_inject=False)

    def test_train_cpu_forward_inject(self):
        """Train on CPU with forward injection."""
        self.train_layer(use_cuda=False, forward_inject=True)

    def test_train_cuda(self):
        """Train on CUDA without forward injection."""
        if SKIP_CUDA_TESTS or not cuda_is_available():
            raise SkipTest("not compiled with CUDA support")
        self.train_layer(use_cuda=True, forward_inject=False)

    def test_train_cuda_forward_inject(self):
        """Train on CUDA with forward injection."""
        if SKIP_CUDA_TESTS or not cuda_is_available():
            raise SkipTest("not compiled with CUDA support")
        self.train_layer(use_cuda=True, forward_inject=True)

    def test_configs_exports(self):
        """Check the exports of aihwkit.simulator.configs (LR-TT and aihwkit classes)."""
        from aihwkit.simulator import configs

        for name in ("PythonLRTTDevice", "PythonLRTTRPUConfig", "QuantizedTorchInferenceRPUConfig"):
            self.assertTrue(hasattr(configs, name), name)
        self.assertIs(configs.PythonLRTTDevice, PythonLRTTDevice)
        rpu_config = configs.PythonLRTTRPUConfig(device=get_lrtt_rpu_config().device)
        self.assertEqual(rpu_config.device.rank, RANK)
        self.assertIs(rpu_config.tile_class, LRTTSimulatorTile)
