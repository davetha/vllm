# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The quant config ``benchmark_moe.py`` builds must match the dtype requested.

``--dtype int8_w8a16`` means int8 weights and 16-bit activations.
``FusedMoEQuantConfig`` distinguishes the two sides itself::

    use_int8_w8a8  = self.quant_dtype == torch.int8
    use_int8_w8a16 = self._a1.dtype is None and self._w1.dtype == torch.int8

so passing ``quant_dtype=torch.int8`` builds a **w8a8** config -- and because
``make()`` falls back to ``weight_dtype = quant_dtype`` when the weight side is
left unset, the mistake is invisible in the weights and shows up only as the
kernel reading bf16 activations as int8.

These assertions are on the config's own accessors rather than on the tuner's
output, so they state the contract in the same vocabulary the class does.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

BENCHMARK_MOE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "kernels" / "benchmark_moe.py"
)


@pytest.fixture(scope="module")
def benchmark_moe():
    pytest.importorskip("ray", reason="benchmark_moe imports ray at module scope")
    if not BENCHMARK_MOE.is_file():
        pytest.skip(f"{BENCHMARK_MOE} not found")
    spec = importlib.util.spec_from_file_location("benchmark_moe", BENCHMARK_MOE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_moe"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop("benchmark_moe", None)


# (use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16) -> (quant_dtype, weight_dtype)
MODES = [
    pytest.param((False, False, False), (None, None), id="unquantized"),
    pytest.param((True, False, False), (torch.float8_e4m3fn, None), id="fp8_w8a8"),
    pytest.param((False, True, False), (None, torch.int8), id="int8_w8a16"),
    pytest.param((False, False, True), (None, "int4"), id="int4_w4a16"),
]


def _config_for(benchmark_moe, flags):
    quant_dtype, weight_dtype = benchmark_moe._moe_quant_dtypes(*flags)
    return FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype, weight_dtype=weight_dtype
    )


@pytest.mark.parametrize("flags, expected", MODES)
def test_quant_dtype_pairs(benchmark_moe, flags, expected):
    assert benchmark_moe._moe_quant_dtypes(*flags) == expected


def test_int8_w8a16_config_is_w8a16_not_w8a8(benchmark_moe):
    """The regression this guards: activations must stay unquantized."""
    config = _config_for(benchmark_moe, (False, True, False))

    assert config.use_int8_w8a16, "int8_w8a16 did not produce a w8a16 config"
    assert not config.use_int8_w8a8, (
        "activations were declared int8; the kernel will read bf16 activations "
        "as int8 and raise in const_data_ptr"
    )
    assert config.quant_dtype is None
    assert config.weight_quant_dtype == torch.int8


def test_int4_w4a16_config_unchanged(benchmark_moe):
    """int4 already declared its weight side correctly; keep it that way."""
    config = _config_for(benchmark_moe, (False, False, True))

    assert config.use_int4_w4a16
    assert config.quant_dtype is None
    assert config.weight_quant_dtype == "int4"


def test_fp8_w8a8_config_unchanged(benchmark_moe):
    """fp8 quantizes BOTH sides, so leaving weight_dtype unset is correct.

    make() falls back to weight_dtype = quant_dtype, which is the behaviour the
    int8 path was accidentally relying on.
    """
    config = _config_for(benchmark_moe, (True, False, False))

    assert config.use_fp8_w8a8
    assert config.quant_dtype == torch.float8_e4m3fn
    assert config.weight_quant_dtype == torch.float8_e4m3fn


def test_unquantized_config_stays_unquantized(benchmark_moe):
    config = _config_for(benchmark_moe, (False, False, False))

    assert not config.is_quantized
    assert not config.use_int8_w8a16
    assert not config.use_int4_w4a16
