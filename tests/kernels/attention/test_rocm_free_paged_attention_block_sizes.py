# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-size coverage for the ROCm *free* paged-attention kernel on gfx90a.

``paged_attention_ll4mi_QKV_mfma16_free_kernel`` takes block_size at runtime, but
until the fix this test guards it derived the slot within a physical KV block
from the **partition-local** token index in two places:

* K: ``kphysical_block_offset = klocal_token_idx % block_size``. The partition
  base is a multiple of ``T_PAR_SIZE`` (256), congruent to 0 mod block_size only
  when block_size divides 256.
* V: ``v_ptr`` folded in ``(rowid * VTOKENS_PER_LANE) % block_size`` and hoisted
  out of the ``vtoken_depth`` loop, dropping both the partition base and the
  ``vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP`` (= 64) term -- correct only
  when block_size divides **both** 64 and 256.

So the bug set was exactly {16, 32, 64} correct and everything above wrong, and
``use_rocm_custom_paged_attention`` had to refuse block_size > 64 on gfx90a.

**256 is the diagnostic case and must stay in the list.** 256 divides 256 but not
64, so a regression that restores only the K bug would still pass at 256 while a
regression restoring the V bug would fail there. 64 passing and 256 failing is
the signature that separates the two.

The reference is vLLM's Triton paged attention driven through the same
``chunked_prefill_paged_decode`` entry point -- the path the gate falls back to
when it refuses -- so "the kernel is sound" means "it agrees with what it
replaces", and both arms consume identical tensors. During development the
Triton arm was itself cross-checked against a float32 gather reference built
from the block table; that is not repeated here because it is quadratic in
seq_len and adds nothing once the two kernels agree.

The gate is bypassed deliberately: this test is about kernel numerics, not gate
policy, and a declined gate would send both arms to Triton, where they agree
perfectly and a broken kernel looks flawless.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

NUM_KV_HEADS = 2
NUM_QUERIES_PER_KV = 8
NUM_HEADS = NUM_KV_HEADS * NUM_QUERIES_PER_KV

# Two bf16 kernels reducing the same softmax will not agree better than a few
# parts in a hundred. This bound catches an order-1 addressing error -- the
# observed failures were 1.1 to 5.0 relative -- not rounding.
REL_TOL = 2e-2

# block_size, and whether the pre-fix kernel got it right.
#   K is correct when block_size divides 256; V additionally needs 64.
BLOCK_SIZE_CASES = [
    pytest.param(16, id="bs16-was-ok"),
    pytest.param(32, id="bs32-was-ok"),
    pytest.param(64, id="bs64-was-ok-largest-correct"),
    pytest.param(128, id="bs128-V-bug-only"),
    pytest.param(256, id="bs256-V-bug-only-diagnostic"),
    pytest.param(512, id="bs512-K-and-V-bugs"),
    pytest.param(784, id="bs784-hybrid-mamba-aligned"),
    pytest.param(1024, id="bs1024-K-and-V-bugs"),
]

# 784 is what a hybrid GDN+attention model actually gets: the attention block
# size is padded up to match the mamba page size. It is not a power of two, so
# it exercises the runtime-block_size path rather than a specialised kernel.
HEAD_SIZES = [128, 256]


def _build_inputs(block_size: int, seq_len: int, head_size: int,
                  dtype: torch.dtype, device: torch.device):
    """One decode token against ``seq_len`` of context, with a scrambled block
    table so a logical block index is never accidentally its physical one."""
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert head_size % x == 0
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = blocks_per_seq + 4

    torch.manual_seed(0xB10C)
    kv = torch.randn(2, num_blocks, block_size, NUM_KV_HEADS, head_size,
                     dtype=dtype, device=device) * 0.5

    from vllm.v1.attention.ops.paged_attn import PagedAttention

    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv, NUM_KV_HEADS, head_size)
    block_table = torch.randperm(
        num_blocks, device=device)[:blocks_per_seq].reshape(
            1, blocks_per_seq).to(torch.int32)
    return {
        "query": torch.randn(1, NUM_HEADS, head_size, dtype=dtype,
                             device=device) * 0.5,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "seq_lens": torch.tensor([seq_len], dtype=torch.int32, device=device),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32,
                                        device=device),
    }


def _run(inputs, seq_len, head_size, device, monkeypatch, force_custom):
    """Drive the production decode entry point, forcing one arm or the other.

    ``chunked_prefill_paged_decode`` imports the gate inside the function body,
    so rebinding the module attribute takes effect at call time and both arms
    run identical surrounding code.
    """
    from vllm.platforms import rocm as rocm_platform
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode,
    )

    monkeypatch.setattr(
        rocm_platform, "use_rocm_custom_paged_attention",
        lambda *args, **kwargs: force_custom,
    )
    out = torch.zeros_like(inputs["query"])
    chunked_prefill_paged_decode(
        query=inputs["query"],
        key=None,
        value=None,
        output=out,
        kv_cache_dtype="auto",
        key_cache=inputs["key_cache"],
        value_cache=inputs["value_cache"],
        block_table=inputs["block_table"],
        query_start_loc=inputs["query_start_loc"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=seq_len,
        max_query_len=1,
        k_scale=torch.tensor(1.0, device=device),
        v_scale=torch.tensor(1.0, device=device),
        sm_scale=1.0 / math.sqrt(head_size),
    )
    return out.float()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("block_size", BLOCK_SIZE_CASES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", [1024, 5000])
def test_free_kernel_matches_triton_across_block_sizes(
    block_size, head_size, seq_len, monkeypatch
):
    if seq_len < block_size:
        pytest.skip("sequence shorter than one block")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = _build_inputs(block_size, seq_len, head_size, dtype, device)
    custom = _run(inputs, seq_len, head_size, device, monkeypatch,
                  force_custom=True)
    triton = _run(inputs, seq_len, head_size, device, monkeypatch,
                  force_custom=False)

    denom = max(triton.abs().max().item(), 1e-6)
    rel = (custom - triton).abs().max().item() / denom
    assert rel < REL_TOL, (
        f"custom free kernel disagrees with Triton at block_size={block_size}, "
        f"head_size={head_size}, seq_len={seq_len}: max_rel={rel:.3e}. "
        "An order-1 value here means a KV slot offset is being taken from the "
        "partition-local token index again; see this module's docstring."
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_gate_opt_in_admits_large_blocks_on_gfx90a(monkeypatch):
    """The gate stays conservative unless explicitly opted in.

    The kernel being correct is necessary but not sufficient to widen the gate
    by default -- this pins the opt-in so the default cannot drift silently.
    """
    from vllm.platforms.rocm import (
        _free_pa_large_block_ok,
        use_rocm_custom_paged_attention,
    )

    def _accepts(block_size):
        return use_rocm_custom_paged_attention(
            torch.bfloat16, 128, block_size, NUM_QUERIES_PER_KV, 4096, 0,
            "auto", None, None,
        )

    monkeypatch.delenv("VLLM_ROCM_FREE_PA_LARGE_BLOCK", raising=False)
    _free_pa_large_block_ok.cache_clear()
    assert _accepts(64), "block_size 64 must always be admitted"

    monkeypatch.setenv("VLLM_ROCM_FREE_PA_LARGE_BLOCK", "1")
    _free_pa_large_block_ok.cache_clear()
    assert _accepts(784), "opt-in must admit the hybrid-aligned block size"
    _free_pa_large_block_ok.cache_clear()
