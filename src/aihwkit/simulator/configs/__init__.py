# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Modifications (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won
# and Sangbum Kim, Seoul National University. Licensed under the MIT license.

"""Configurations for resistive processing units."""

from aihwkit.simulator.parameters import (
    IOParameters,
    IOParametersIRDropT,
    UpdateParameters,
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
    SimpleDriftParameter,
    DriftParameter,
    MappingParameter,
    InputRangeParameter,
    PrePostProcessingParameter,
)
from aihwkit.simulator.parameters.enums import (
    RPUDataType,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    PulseType,
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    VectorUnitCellUpdatePolicy,
    AnalogMVType,
    CountLRFeedbackPolicy,
)
from .devices import (
    FloatingPointDevice,
    IdealDevice,
    ConstantStepDevice,
    LinearStepDevice,
    SoftBoundsDevice,
    SoftBoundsPmaxDevice,
    SoftBoundsReferenceDevice,
    ExpStepDevice,
    PowStepDevice,
    PowStepReferenceDevice,
    PiecewiseStepDevice,
)
from .compounds import (
    VectorUnitCell,
    ReferenceUnitCell,
    OneSidedUnitCell,
    DifferenceUnitCell,
    TransferCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
    MixedPrecisionCompound,
)
from .configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
    TorchInferenceRPUConfig,
    TorchInferenceRPUConfigIRDropT,
    QuantizedTorchInferenceRPUConfig,
)

from .helpers import build_config

# Low-Rank Tiki-Taka (LR-TT) additions.
from .lrtt_python import PythonLRTTDevice
from .lrtt_rpu_config import PythonLRTTRPUConfig
from .lrtt_config import (
    lrtt_idealized_config,
    lrtt_constant_step_config,
    lrtt_lora_style_config,
    lrtt_mixed_precision_config,
    lrtt_inference_config,
    lrtt_config_from_attributes,
)
