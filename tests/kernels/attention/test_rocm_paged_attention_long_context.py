# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context coverage for the ROCm multi-pass paged-attention reduction.

``test_paged_attention`` cannot reach this path. Its sequence lengths are bounded
by::

    MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512

which is 15,872 on an MI210 and yields ``npar_loops == 1``. Its reference also
materialises keys and values with a Python loop over every token, so it is not
usable at these lengths regardless.

So this compares the custom kernel against vLLM's **Triton** paged attention,
driven through the real ``chunked_prefill_paged_decode`` entry point. Triton is
the right reference because it is the path the gate fell back to before the cap
was lifted: "the kernel is sound" means "it agrees with what it replaces". Both
arms consume the same tensors, so they cannot disagree for KV-layout reasons.

The gate is asserted to *accept* before each comparison. Without that a declined
gate sends both arms to Triton, they agree perfectly, and a broken kernel looks
flawless.

Pass structure, from the launcher: ``npar_loops <= 16`` runs the direct
single-pass switch. Above that, ``num_passes = ceil(npar_loops / 8)``, each pass
covering ``8 * WARP_SIZE * PARTITION_SIZE = 131,072`` tokens, and
``paged_attention_merge_reduce_kernel`` combines the passes with log-sum-exp.
So the cases at 17 and beyond exercise both the pass loop and the merge kernel,
while 9..16 stay on the direct path.
"""

import pytest
import torch

from vllm.platforms import current_platform

PARTITION_SIZE_ROCM = 256
CDNA_WARP_SIZE = 64
# Single-pass switch cases go to 16; above that the launcher splits into passes
# of 8 npar_loops each.
MAX_SINGLE_PASS_NPAR = 16
NPAR_LOOPS_PER_PASS = 8

NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_SIZE = 128
BLOCK_SIZE = 16

# Two independent bf16 kernels reducing a multi-hundred-thousand-term softmax
# will not agree to better than a few parts in a hundred. This bound is set to
# catch a dropped partition group -- an order-1 error -- not rounding.
REL_TOL = 2e-2

# (seq_len, expected npar_loops, expected passes)
LONG_CONTEXT_CASES = [
    pytest.param(131_072, 8, 1, id="npar8-control"),
    pytest.param(139_264, 9, 1, id="npar9-single-pass"),
    pytest.param(262_144, 16, 1, id="npar16-last-single-pass"),
    pytest.param(266_240, 17, 3, id="npar17-first-multi-pass"),
    pytest.param(524_288, 32, 4, id="npar32"),
    pytest.param(1_048_576, 64, 8, id="npar64-1M"),
    pytest.param(1_572_864, 96, 12, id="npar96-12-passes"),
]


def _npar_loops(seq_len: int) -> int:
    partitions = (seq_len + PARTITION_SIZE_ROCM - 1) // PARTITION_SIZE_ROCM
    return (partitions + CDNA_WARP_SIZE - 1) // CDNA_WARP_SIZE


def _num_passes(npar_loops: int) -> int:
    if npar_loops <= MAX_SINGLE_PASS_NPAR:
        return 1
    return (npar_loops + NPAR_LOOPS_PER_PASS - 1) // NPAR_LOOPS_PER_PASS


def _gate_accepts(seq_len: int, dtype: torch.dtype, block_size: int = BLOCK_SIZE):
    from vllm.platforms.rocm import use_rocm_custom_paged_attention

    return use_rocm_custom_paged_attention(
        dtype,
        HEAD_SIZE,
        block_size,
        NUM_HEADS // NUM_KV_HEADS,
        seq_len,
        0,
        "auto",
        None,
        None,
    )


def _build_decode_inputs(seq_len: int, dtype: torch.dtype, device: torch.device):
    """One decode token against ``seq_len`` of context.

    K is 5-D ``[blocks, kv_heads, head_size // x, block_size, x]`` and V is 4-D
    ``[blocks, kv_heads, head_size, block_size]``; that is what the Triton decode
    kernel's stride arguments imply, not what ``get_kv_cache_shape`` declares.
    """
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert HEAD_SIZE % x == 0

    torch.manual_seed(0xA1DE)
    return {
        "query": torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=dtype, device=device),
        "key_cache": torch.randn(
            num_blocks + 16,
            NUM_KV_HEADS,
            HEAD_SIZE // x,
            BLOCK_SIZE,
            x,
            dtype=dtype,
            device=device,
        ),
        "value_cache": torch.randn(
            num_blocks + 16,
            NUM_KV_HEADS,
            HEAD_SIZE,
            BLOCK_SIZE,
            dtype=dtype,
            device=device,
        ),
        "block_table": torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0),
        "seq_lens": torch.tensor([seq_len], dtype=torch.int32, device=device),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device=device),
    }


def _run_decode(inputs, seq_len, dtype, device, monkeypatch, force_triton):
    """Drive the production decode entry point, optionally forcing Triton.

    ``chunked_prefill_paged_decode`` imports the gate inside the function body,
    so rebinding the module attribute takes effect at call time and both arms run
    identical surrounding code.
    """
    from vllm.platforms import rocm as rocm_platform
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode,
    )

    output = torch.empty(1, NUM_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    one = torch.tensor(1.0, device=device)

    with monkeypatch.context() as m:
        if force_triton:
            m.setattr(
                rocm_platform, "use_rocm_custom_paged_attention", lambda *a, **k: False
            )
        chunked_prefill_paged_decode(
            query=inputs["query"],
            key=None,
            value=None,
            output=output,
            kv_cache_dtype="auto",
            key_cache=inputs["key_cache"],
            value_cache=inputs["value_cache"],
            block_table=inputs["block_table"],
            query_start_loc=inputs["query_start_loc"],
            seq_lens=inputs["seq_lens"],
            max_seq_len=seq_len,
            max_query_len=1,
            k_scale=one,
            v_scale=one,
            sm_scale=HEAD_SIZE**-0.5,
        )
    return output[0].float()


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only paged attention kernel"
)
@pytest.mark.parametrize("seq_len, expected_npar, expected_passes", LONG_CONTEXT_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_multipass_reduction_matches_triton(
    seq_len: int,
    expected_npar: int,
    expected_passes: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if current_platform.is_navi():
        pytest.skip("multi-pass reduction is GFX9-only")

    assert _npar_loops(seq_len) == expected_npar, (
        f"test is stale: seq_len {seq_len} now yields "
        f"npar_loops {_npar_loops(seq_len)}, not {expected_npar}"
    )
    assert _num_passes(expected_npar) == expected_passes

    assert _gate_accepts(seq_len, dtype), (
        f"gate declined seq_len={seq_len}; the comparison below would be "
        "Triton against Triton and would prove nothing"
    )

    device = torch.device("cuda")
    inputs = _build_decode_inputs(seq_len, dtype, device)
    common = (inputs, seq_len, dtype, device, monkeypatch)

    custom = _run_decode(*common, force_triton=False)
    triton = _run_decode(*common, force_triton=True)

    denom = triton.abs().max().clamp_min(1e-6)
    rel_err = ((custom - triton).abs().max() / denom).item()
    assert rel_err <= REL_TOL, (
        f"custom kernel disagrees with Triton at seq_len={seq_len} "
        f"(npar_loops={expected_npar}, passes={expected_passes}): "
        f"rel_err={rel_err:.2e} > {REL_TOL}"
    )


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only paged attention kernel"
)
@pytest.mark.parametrize("block_size", [128, 544, 1056])
def test_gfx90a_declines_large_block_sizes(block_size: int) -> None:
    """The free kernel returns wrong results on gfx90a above block_size 64.

    Measured on MI210: every free-kernel case with block_size in
    {128, 512, 1024, 2096} mismatches the reference by order 1, while 16/32/64
    agree at every head size the gate admits. Hybrid models select large blocks
    on their own -- attention is aligned to the mamba page size, giving 544 for
    Qwen3-Next -- so the gate has to decline them rather than leaving it to
    whoever set ``--block-size``.
    """
    from vllm.platforms.rocm import on_gfx90a

    if not on_gfx90a():
        pytest.skip("restriction applies to gfx90a only")

    assert not _gate_accepts(65_536, torch.bfloat16, block_size=block_size), (
        f"gate admitted block_size={block_size} on gfx90a; the free kernel "
        "returns incorrect results there instead of falling back to Triton"
    )
    assert _gate_accepts(65_536, torch.bfloat16, block_size=16), (
        "gate should still admit the standard block size"
    )
