# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W8A16 (fp8 weight, bf16/fp16 activation) GEMM for ROCm gfx90a.

Dequantizes float8_e4m3fn weights inside the GEMM's inner loop and feeds the
result straight to MFMA, so the fp8 bytes are the only weight traffic and the
activation is never quantized. This is the W8A16 case: the point is that `x`
arrives unquantized and stays that way.

Weight layout expected by this kernel (post-process_weights_after_loading):
  weight:       [K, N]  float8_e4m3fn, viewed as uint8 by the kernel.
                Arrives as `layer.weight.t()` from the scheme, i.e. a view of a
                C-contiguous [N, K] buffer, so K is the contiguous dim
                (stride_bk == 1, stride_bn == K). Strides are passed
                explicitly; no layout is assumed. See _WEIGHT_LAYOUT below.
  weight_scale: [N]  float32, one scale per output channel.

Checkpoint layout from CompressedTensorsW8A16Fp8.create_weights:
  weight:       [N, K]  float8_e4m3fn
  weight_scale: [N, 1]  float32   (per-tensor is expanded to channel by the
                scheme's process_weights_after_loading before we see it)


DEQUANT: e4m3fn -> fp16 by bit manipulation
------------------------------------------
An e4m3fn byte is [s eeee mmm]: sign, 4-bit exponent (bias 7), 3-bit mantissa.
An fp16 is [s eeeee mmmmmmmmmm]: sign, 5-bit exponent (bias 15), 10-bit
mantissa. The fields are compatible up to a shift and a bias, so the convert
is three integer ops instead of a conversion instruction:

    t     = b_u8.to(uint16) << 7   # e -> fp16 exp[13:10], m -> fp16 mant[9:7],
                                   # sign lands at bit 14
    t     = t + (t & 0x4000)       # carry: sign moves 14 -> 15, and the add
                                   # clears bit 14 (the top exponent bit) in
                                   # the process, which is exactly what we want
    b_f16 = bitcast(t, fp16)

The resulting fp16 exponent field is [0, eeee], so the value is
2^(eeee-15) * 1.mmm against the true 2^(eeee-7) * 1.mmm: every decoded weight
is the true weight times 2^-8. The factor 2^8 is a constant, so it comes back
out at the end -- see FOLD FACTOR below. Verified bit-exact on all 254 finite
codes by prior work and re-verified here (see the smoke tests).

The top exponent bit cannot be set instead of cleared to avoid the 2^-8 in the
first place: eeee goes up to 15, so [1, eeee] would carry into the sign bit.
2^-8 is the largest fold this construction admits.

Verified on GPU against a float64 oracle built from the raw bytes: all 254
finite codes decode bit-exactly (0 mismatches over a 254x254 one-hot probe,
with and without per-channel scales).


SUBNORMAL FLUSH -- why there is a `* 256.0` in the loop
-------------------------------------------------------
The e4m3 denormal codes 0x01..0x07 are 1..7 * 2^-9. Biased by 2^-8 they become
2^-17..7*2^-17, which are fp16 SUBNORMALS (fp16 min normal is 2^-14).

Probed on this hardware (MI210, gfx90a, torch 2.11.0+rocm7.14.0, triton 3.7.1):
v_mfma_f32_*f16 FLUSHES SUBNORMAL INPUTS TO ZERO. Feeding the biased values
straight to tl.dot returned exactly 0.0 for every code in 0x01..0x07 and the
correct value from 0x08 (= 2^-14, the first normal) upward. The result was
identical for all five tile shapes in the ladder below, so it is a property of
the MFMA path and not of one instruction selection.

The VALU does NOT flush: the same bitcast multiplied by 256.0 in fp16 before
the dot returns the exact e4m3 value for those same codes. So the fix is one
fp16 multiply inside the loop, which cancels the 2^-8 at the source and leaves
no subnormal anywhere:

    b_f16 = bitcast(t, fp16) * 256.0     # exact: power-of-two rescale,
                                         # 2^-17 -> 2^-9, max 1.75 -> 448

That multiply is the whole cost of correctness here. Without it the low
seventh of the weight range silently becomes zero, which does not crash -- it
emits fluent wrong text.

FOLD FACTOR: because the `* 256.0` already undoes the bias, the per-channel
scale is used as-is (fp32, upcast once at load). Do NOT also fold 2^8 into the
scale; that would double-apply it.


WHY fp16 MFMA AND NOT bf16
--------------------------
bf16 has 8 exponent bits and 7 mantissa bits, so an e4m3 byte does not shift
into it cleanly: the decode costs 5 ops against fp16's 3 (4 with the rescale).
The bf16 fold would have to be 2^120, which puts a hard ceiling of s > 256 on
the usable per-channel scale, and the resulting products land in fp32-subnormal
territory. bf16 is also an MFMA-operand-only type on CDNA2 -- gfx90a has no
bf16 VALU arithmetic -- so the in-loop rescale above would not even be
expressible without a round trip through fp32.

bf16 -> fp16 on the activation side is exact for |x| in [2^-14, 65504], which
covers post-RMSNorm activations by orders of magnitude. ACTIVATION OVERFLOW
BOUND: |x| >= 65536 would overflow fp16 and produce inf. This is verified
end-to-end rather than clamped -- a clamp in the inner loop costs ~8%.

The MFMA subnormal flush also applies to the ACTIVATION side, below 2^-14
(6.1e-5): such elements contribute 0 instead of their value. This is an
absolute floor, not a relative one, and it is inherited by every fp16-MFMA
path here including the shipped W4A16 magic-bias one. Measured against the
float64 oracle by scaling a random activation vector (K=4096):

    |a| ~ 1e+00    0.0% of A subnormal    err 9.2e-07
    |a| ~ 1e-02    0.5%                   err 3.1e-04
    |a| ~ 1e-04   45.8%                   err 2.5e-01
    |a| ~ 1e-05  100.0%                   err 1.0e+00

i.e. it only bites once a large fraction of the whole activation vector is
below 2^-14, which is a regime post-RMSNorm hidden states do not reach. It
could be bought off by pre-scaling A by 2^k in the loop -- O(BLOCK_M*BLOCK_K),
cheap at decode widths -- but that buys floor headroom by giving up the same
factor of overflow ceiling, and the ceiling is the bound that has actually
been verified end-to-end. Left alone deliberately; do not "fix" one side of
this without re-verifying the other.


NaN POLICY
----------
0x7F and 0xFF are the only NaN codes in e4m3fn. The bit trick does not
propagate them: 0x7F decodes to a finite 480.0 (and 0xFF to -480.0). Rather
than pay for a check in the inner loop, process_weights_after_loading rejects
any checkpoint containing those codes -- loudly, at load, at zero runtime cost.
A checkpoint with NaN weights is broken anyway.
"""

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


@triton.jit
def triton_fp8_w8a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  bf16/fp16 activations
    b_ptr,  # [K, N]  uint8 (raw float8_e4m3fn bytes)
    s_ptr,  # [N]     fp32 per-channel weight scales
    bias_ptr,  # [N]  bf16/fp16, unused when HAS_BIAS is False
    c_ptr,  # [M, N]  bf16/fp16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    # Block sizes (tuned for gfx90a, wavefront=64 -- see the ladder in the
    # wrapper below)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W8A16 GEMM: C[M,N] = A[M,K] @ (dequant(B)[K,N] * scale[N])

    B holds raw float8_e4m3fn bytes. Dequant is the bit trick documented in the
    module header: shift into the fp16 field layout, carry the sign, rescale by
    2^8 to undo the exponent-bias difference.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load raw fp8 weight bytes B: [BLOCK_K, BLOCK_N] ----
        # other=0 is code 0x00, which decodes to +0.0, so masked lanes
        # contribute nothing.
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = mask_k[:, None] & mask_n[None, :]
        b_u8 = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Dequantize e4m3fn -> fp16 (module header for the derivation) ----
        t = b_u8.to(tl.uint16) << 7
        t = t + (t & 0x4000)
        # The `* 256.0` is not a scale fold, it is the subnormal fix: MFMA on
        # gfx90a flushes fp16 subnormal operands to zero and the biased forms
        # of codes 0x01..0x07 are subnormal. Exact (power of two, no overflow:
        # max 1.75 -> 448).
        b_f16 = t.to(tl.float16, bitcast=True) * 256.0

        # bf16 -> fp16 is exact over the activation range; see the module
        # header for the overflow bound.
        a_f16 = a.to(tl.float16)

        accumulator += tl.dot(a_f16, b_f16, out_dtype=tl.float32)

    # ---- Epilogue: per-channel scale (+ bias), in fp32 ----
    # Hoisted out of the loop: the scale is per output channel, so it is
    # O(BLOCK_M*BLOCK_N) once here instead of O(BLOCK_K*BLOCK_N) per K-tile.
    scales = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    accumulator = accumulator * scales[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator = accumulator + bias[None, :].to(tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def triton_fp8_w8a16_gemm(
    a: torch.Tensor,  # [M, K] bf16/fp16
    b_fp8: torch.Tensor,  # [K, N] float8_e4m3fn (or uint8 view)
    scales: torch.Tensor,  # [N] fp32
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Fused W8A16 GEMM with float8_e4m3fn weights.

    Args:
        a:        Activations [M, K], bfloat16 or float16. Must be contiguous.
        b_fp8:    Weights [K, N], float8_e4m3fn. Any strides; the kernel reads
                  it through explicit strides.
        scales:   Per-output-channel scales [N], float32, contiguous.
        bias:     Optional bias [N].
        out_dtype: Output dtype; defaults to a.dtype.

    Returns:
        Output [M, N] in out_dtype.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    assert b_fp8.shape[0] == K, f"b shape {tuple(b_fp8.shape)} does not match K={K}"
    N = b_fp8.shape[1]
    assert scales.numel() == N, f"expected {N} scales, got {scales.numel()}"

    if out_dtype is None:
        out_dtype = a.dtype
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    # Read the fp8 bytes as uint8; Triton has no float8_e4m3fn load that
    # preserves the raw bits, and the whole decode is integer work anyway.
    b_u8 = b_fp8.view(torch.uint8)

    num_warps = None

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        if on_gfx90a():
            # Cloned from the gfx90a ladder in mixed_precision/triton_w4a16.py,
            # which was searched over 69 configurations at M=1 on this exact
            # card. That search was run against int4 decode; fp8 decode is
            # cheaper (4 VALU ops per weight against ~19), so the balance
            # between decode cost and occupancy is not identical and a retune
            # may pay. Shipping the known-good ladder first: it is an
            # occupancy fix at heart -- BLOCK_N=64 leaves a narrow-N layer with
            # fewer workgroups than the card has CUs (104) -- and that argument
            # does not depend on what the decode costs.
            #
            # Known soft spot for whoever retunes: at M=1 this ladder gets
            # 605 GB/s on gate_up (34816x5120) and 345 on down_proj
            # (5120x17408) but only 162 on o_proj (4096x4096). o_proj lands on
            # the same 16x16x128 tile as down_proj with a quarter of down_proj's
            # K, so it has the same 256-320 workgroups over far less work each
            # and never amortises. That shape wants its own entry.
            if M <= 8:
                if N >= 16384:
                    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
                else:
                    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 128
                num_warps = 2
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        else:
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    launch_opts = {} if num_warps is None else {"num_warps": num_warps}

    triton_fp8_w8a16_gemm_kernel[grid](
        a,
        b_u8,
        scales,
        bias if bias is not None else scales,  # dummy ptr; Triton needs one
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_u8.stride(0),
        b_u8.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **launch_opts,
    )
    return c


class TritonW8A16Fp8LinearKernel(FP8ScaledMMLinearKernel):
    """
    Triton W8A16 fp8 GEMM for ROCm gfx90a (MI210).

    Consumes bf16/fp16 activations directly -- no activation quantization --
    and dequantizes float8_e4m3fn weights in the GEMM inner loop.
    """

    _SUPPORTED_WEIGHT_QUANT_KEYS = {
        # TENSOR is promoted to CHANNEL by CompressedTensorsW8A16Fp8's
        # process_weights_after_loading before apply time, but the config is
        # built with the pre-promotion key, so both have to be accepted here.
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Gated on gfx90a, on evidence rather than on validity. The dequant is
        # portable IEEE bit manipulation, but the subnormal-flush workaround
        # and the tile ladder are both measured facts about CDNA2 MFMA and
        # about a 104-CU card. Nobody has measured gfx942 or RDNA here, so
        # they keep their existing kernels. Widen once someone does.
        if not current_platform.is_rocm():
            return False, "TritonW8A16Fp8Linear requires ROCm"
        from vllm.platforms.rocm import on_gfx90a

        if not on_gfx90a():
            return False, "TritonW8A16Fp8Linear is only tuned/verified on gfx90a"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in cls._SUPPORTED_WEIGHT_QUANT_KEYS:
            # BLOCK in particular is deliberately rejected: it does not reach
            # this kernel list at all (it routes through
            # _POSSIBLE_FP8_BLOCK_KERNELS), and its per-block scale would need
            # the scale reload moved back inside the K loop.
            return (
                False,
                "TritonW8A16Fp8Linear only supports per-channel and per-tensor "
                "weight quantization",
            )
        if c.weight_quant_key.dtype != torch.float8_e4m3fn:
            # Not dead code, despite the keys above already pinning a dtype:
            # kFp8StaticChannelSym is built from current_platform.fp8_dtype(),
            # which is float8_e4m3fnUZ on the ROCm targets where is_fp8_fnuz()
            # holds. fnuz has exponent bias 8, no signed zero, and NaN only at
            # 0x80 -- the shift/carry derivation in the module header is for
            # e4m3fn and would silently mis-decode every fnuz weight by a
            # factor of two. e5m2 likewise needs its own derivation. Reject.
            return (
                False,
                "TritonW8A16Fp8Linear only supports float8_e4m3fn weights, got "
                f"{c.weight_quant_key.dtype}",
            )
        if c.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "TritonW8A16Fp8Linear only supports bf16/fp16 activations"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        # Deliberately not FP8ScaledMMLinearKernel.__init__: it constructs a
        # QuantFP8 for the activation, and this kernel never quantizes the
        # activation. Same reimplementation of the grandparent body as
        # XPUW8A16FP8LinearKernel.
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Validate the weight codes and materialize an fp32 [N] scale.

        The scheme hands us `weight` already canonicalized to [K, N] -- it is a
        `.t()` view of the C-contiguous [N, K] checkpoint buffer, so K is the
        contiguous dimension.

        _WEIGHT_LAYOUT: that view is left alone deliberately, not made
        contiguous. With K contiguous, one column of a [BLOCK_K, BLOCK_N] tile
        is BLOCK_K consecutive bytes (128 at the decode tile), which coalesces;
        calling .contiguous() would flip that to BLOCK_N consecutive bytes per
        row -- 16 at the same tile -- and would also double the layer's peak
        memory during load. Both layouts were measured, median of 30, M=1, fp8
        bytes moved per second:

                                [N,K]-backed view    [K,N] contiguous copy
            gate_up  34816x5120      605.6 GB/s             444.4 GB/s
            down_proj 5120x17408     344.6 GB/s             217.2 GB/s
            o_proj    4096x4096      162.4 GB/s             143.5 GB/s

        The view wins everywhere, so the copy costs memory and time for
        nothing. If a future tile ladder pushes BLOCK_N above BLOCK_K this
        conclusion can invert -- re-measure before assuming it holds.
        """
        w = layer.weight

        # NaN codes do not survive the bit trick -- 0x7F would decode to a
        # finite 480.0 -- so reject them at load rather than check per weight
        # in the inner loop. Loud, once, free.
        if ((w.data.view(torch.uint8) & 0x7F) == 0x7F).any():
            raise ValueError(
                "TritonW8A16Fp8Linear: weight contains float8_e4m3fn NaN codes "
                "(0x7F/0xFF). This kernel's bit-trick dequant decodes them to a "
                "finite 480.0 instead of propagating NaN, so the checkpoint is "
                "rejected rather than silently mis-evaluated."
            )

        # Per-channel scale as contiguous fp32 [N]. Checkpoints store this as
        # [N, 1], and depending on the loader path it can arrive bf16, whose
        # 8 mantissa bits would visibly quantize the output; upcast once here
        # where it costs nothing.
        s = layer.weight_scale.data.to(torch.float32).reshape(-1).contiguous()
        expected_n = w.shape[1]
        if s.numel() != expected_n:
            raise ValueError(
                f"TritonW8A16Fp8Linear: expected {expected_n} per-channel weight "
                f"scales for a [K={w.shape[0]}, N={expected_n}] weight, got "
                f"{s.numel()}"
            )
        replace_parameter(layer, "weight_scale", s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x is consumed directly, unquantized -- that is the entire point of
        # W8A16. No _get_layer_params/quant_fp8/super().apply_weights here.
        weight = layer.weight
        scales = layer.weight_scale

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out_shape = x.shape[:-1] + (weight.shape[1],)

        out_dtype = self.config.out_dtype or x.dtype
        output = triton_fp8_w8a16_gemm(
            a=x_2d,
            b_fp8=weight,
            scales=scales,
            bias=bias,
            out_dtype=out_dtype,
        )
        return output.reshape(out_shape)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Dead: required by the FP8ScaledMMLinearKernel ABC but never reached,
        # because apply_weights above is overridden and never calls it. Same
        # stub as MarlinFP8ScaledMMLinearKernel and XPUW8A16FP8LinearKernel.
        pass
