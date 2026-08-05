# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# Repacking materialises the weights UNPACKED, at int32 -- eight bytes out for
# every packed byte in. Done in one shot the transients dwarf the result:
#
#   lo / hi        E x N x K/2 int32   (2 x 6 GiB for GLM-5.2 at TP=2)
#   stack+reshape  E x N x K   int32   (12 GiB)   <-- the allocation that failed
#   permute+cont   E x N x K   int32   (another 12 GiB)
#   output         E x K x N/8 int32   (1.5 GiB)
#
# ~36 GiB of scratch to produce 1.5 GiB, on a 64 GiB card that already holds
# 50 GiB of weights. Measured: GLM-5.2 (256 experts, N=2048, K=6144) dies here
# with "Tried to allocate 12.00 GiB ... 9.83 GiB is free".
#
# Chunking over the expert dimension fixes it. Experts are independent in this
# transform -- every op is elementwise or acts within a single expert -- so
# slicing E changes nothing about the result, only the peak. Output is
# bit-identical by construction, which the caller's tests then confirm.
DEFAULT_EXPERT_CHUNK = 16


def repack_int4_to_int32(
    w: torch.Tensor, expert_chunk: int = DEFAULT_EXPERT_CHUNK
) -> torch.Tensor:
    """Repack [E, N, K//2] uint8 → [E, K, N//8] int32.

    Input: K-packed uint8 (2 int4 per byte, low nibble first).
    Output: N-packed int32 (8 int4 per int32, GPTQ sequential shifts
            [0,4,...,28]).

    Processed in chunks of `expert_chunk` experts so peak scratch is
    proportional to the chunk rather than to the whole layer. The unchunked
    form needs ~24x the output size in transients, which is enough to fail on
    a large-expert-count MoE.
    """
    E, N, K_half = w.shape
    assert N % 8 == 0, f"N must be divisible by 8 for int4 packing, got N={N}"
    K = K_half * 2
    N8 = N // 8

    packed = torch.empty((E, K, N8), dtype=torch.int32, device=w.device)
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4

    for start in range(0, E, expert_chunk):
        stop = min(start + expert_chunk, E)
        chunk = w[start:stop]
        e = stop - start

        lo = (chunk & 0xF).to(torch.int32)
        hi = ((chunk >> 4) & 0xF).to(torch.int32)
        unpacked = torch.stack([lo, hi], dim=-1).reshape(e, N, K)
        del lo, hi
        transposed = unpacked.permute(0, 2, 1).contiguous()
        del unpacked
        packed[start:stop] = (transposed.view(e, K, N8, 8) << shifts).sum(
            dim=-1, dtype=torch.int32
        )
        del transposed

    return packed


def unpack_zp_int4_to_fp16(zp: torch.Tensor) -> torch.Tensor:
    """Unpack [E, N//2, K_groups] uint8 → [E, K_groups, N] fp16."""
    E, N_half, K_groups = zp.shape
    lo = (zp & 0xF).to(torch.int32)
    hi = ((zp >> 4) & 0xF).to(torch.int32)
    unpacked = torch.stack([lo, hi], dim=2).reshape(E, N_half * 2, K_groups)
    return unpacked.permute(0, 2, 1).contiguous().to(torch.float16)
