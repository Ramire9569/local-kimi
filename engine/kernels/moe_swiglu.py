"""Fused grouped W4A16 gate, up, and SwiGLU for MoE decode."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None

GROUP_SIZE = 32

# This launch produces 9 * ceil(1024 / 64) = 144 programs at the Kimi K3
# decode shape. That is close to one complete wave on the 142-SM L40S.
BLOCK_N = 64
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 3


if triton is not None:

    @triton.jit
    def _fused_swiglu_w4a16_kernel(
        activation_ptr,
        expert_index_ptr,
        w1_packed_ptr,
        w1_scale_ptr,
        w3_packed_ptr,
        w3_scale_ptr,
        output_ptr,
        ROUTES,
        N,
        K,
        stride_at,
        stride_ak,
        stride_it,
        stride_ir,
        stride_w1e,
        stride_w1n,
        stride_w1k,
        stride_s1e,
        stride_s1n,
        stride_s1k,
        stride_w3e,
        stride_w3n,
        stride_w3k,
        stride_s3e,
        stride_s3n,
        stride_s3k,
        stride_ot,
        stride_or,
        stride_on,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP: tl.constexpr,
        HP_ACTIVATION: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        expert = tl.load(
            expert_index_ptr + token * stride_it + route * stride_ir
        ).to(tl.int64)

        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        n_mask = offsets_n < N
        gate_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            current_k = k_start + offsets_k
            k_mask = current_k < K
            weight_mask = k_mask[:, None] & n_mask[None, :]

            activation = tl.load(
                activation_ptr + token * stride_at + current_k * stride_ak,
                mask=k_mask,
                other=0.0,
            )
            activation_fp32 = activation.to(tl.float32)

            w1_packed_byte = tl.load(
                w1_packed_ptr
                + expert * stride_w1e
                + offsets_n[None, :] * stride_w1n
                + (current_k[:, None] // 2) * stride_w1k,
                mask=weight_mask,
                other=0,
            )
            w1_low_nibble = w1_packed_byte & 0xF
            w1_high_nibble = w1_packed_byte >> 4
            w1_nibble = tl.where(
                (current_k[:, None] & 1) == 0,
                w1_low_nibble,
                w1_high_nibble,
            )
            w1_signed = w1_nibble.to(tl.int32)
            w1_signed = tl.where(w1_signed >= 8, w1_signed - 16, w1_signed)
            w1_scale = tl.load(
                w1_scale_ptr
                + expert * stride_s1e
                + offsets_n[None, :] * stride_s1n
                + (current_k[:, None] // GROUP) * stride_s1k,
                mask=weight_mask,
                other=1.0,
            )
            w1_decoded = (
                w1_signed.to(tl.float32) * w1_scale.to(tl.float32)
            ).to(tl.bfloat16)
            gate_accumulator += tl.sum(
                activation_fp32[:, None] * w1_decoded.to(tl.float32),
                axis=0,
            )

            w3_packed_byte = tl.load(
                w3_packed_ptr
                + expert * stride_w3e
                + offsets_n[None, :] * stride_w3n
                + (current_k[:, None] // 2) * stride_w3k,
                mask=weight_mask,
                other=0,
            )
            w3_low_nibble = w3_packed_byte & 0xF
            w3_high_nibble = w3_packed_byte >> 4
            w3_nibble = tl.where(
                (current_k[:, None] & 1) == 0,
                w3_low_nibble,
                w3_high_nibble,
            )
            w3_signed = w3_nibble.to(tl.int32)
            w3_signed = tl.where(w3_signed >= 8, w3_signed - 16, w3_signed)
            w3_scale = tl.load(
                w3_scale_ptr
                + expert * stride_s3e
                + offsets_n[None, :] * stride_s3n
                + (current_k[:, None] // GROUP) * stride_s3k,
                mask=weight_mask,
                other=1.0,
            )
            w3_decoded = (
                w3_signed.to(tl.float32) * w3_scale.to(tl.float32)
            ).to(tl.bfloat16)
            up_accumulator += tl.sum(
                activation_fp32[:, None] * w3_decoded.to(tl.float32),
                axis=0,
            )

        if HP_ACTIVATION:
            gate_for_silu = gate_accumulator
            up_for_multiply = up_accumulator
            silu_gate = gate_for_silu / (1.0 + tl.exp(-gate_for_silu))
        else:
            # The current path stores both projections as BF16, computes a BF16
            # SiLU result, then performs the BF16 multiply. Preserve those three
            # rounding points even though the fused kernel holds FP32 accumulators.
            gate_for_silu = gate_accumulator.to(tl.bfloat16).to(tl.float32)
            up_for_multiply = up_accumulator.to(tl.bfloat16).to(tl.float32)
            silu_gate = (
                gate_for_silu / (1.0 + tl.exp(-gate_for_silu))
            ).to(tl.bfloat16).to(tl.float32)

        activated = silu_gate * up_for_multiply
        tl.store(
            output_ptr
            + token * stride_ot
            + route * stride_or
            + offsets_n * stride_on,
            activated,
            mask=n_mask,
        )

else:
    _fused_swiglu_w4a16_kernel = None


def fused_swiglu_w4a16(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
    *,
    hp_activation: bool = False,
) -> torch.Tensor:
    """Compute grouped ``silu(w1 @ x) * (w3 @ x)`` in one Triton launch."""
    if triton is None or _fused_swiglu_w4a16_kernel is None:
        raise RuntimeError("Triton is required for fused grouped W4A16 SwiGLU")
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if expert_indices.ndim != 2 or expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must be an int64 [tokens, routes] matrix")
    if w1_packed.ndim != 3 or w3_packed.ndim != 3:
        raise ValueError("grouped W1 and W3 packed weights must be rank three")
    if w1_scales.ndim != 3 or w3_scales.ndim != 3:
        raise ValueError("grouped W1 and W3 scales must be rank three")
    if w1_packed.dtype != torch.uint8 or w3_packed.dtype != torch.uint8:
        raise TypeError("grouped W1 and W3 packed weights must be uint8")
    if w1_scales.dtype != torch.bfloat16 or w3_scales.dtype != torch.bfloat16:
        raise TypeError("grouped W1 and W3 scales must be BF16")
    if not isinstance(hp_activation, bool):
        raise TypeError("hp_activation must be a bool")
    if activations.device.type != "cuda":
        raise ValueError("fused grouped W4A16 SwiGLU requires CUDA")
    if not (
        activations.device
        == expert_indices.device
        == w1_packed.device
        == w1_scales.device
        == w3_packed.device
        == w3_scales.device
    ):
        raise ValueError("fused grouped W4A16 SwiGLU inputs must share one device")

    tokens, reduction = activations.shape
    route_tokens, routes = expert_indices.shape
    experts, output_size, packed_reduction = w1_packed.shape
    if route_tokens != tokens:
        raise ValueError("expert routing must have one row per token")
    if tokens == 0 or routes == 0 or experts == 0 or output_size == 0:
        raise ValueError("fused grouped W4A16 SwiGLU requires non-empty dimensions")
    if packed_reduction * 2 != reduction:
        raise ValueError("packed grouped W1 weights have the wrong reduction width")
    if reduction % GROUP_SIZE != 0:
        raise ValueError("the W4A16 reduction width must be divisible by 32")
    if w3_packed.shape != w1_packed.shape:
        raise ValueError("grouped W1 and W3 packed weights must have the same shape")
    expected_scale_shape = (experts, output_size, reduction // GROUP_SIZE)
    if w1_scales.shape != expected_scale_shape:
        raise ValueError("grouped W1 scales have the wrong shape")
    if w3_scales.shape != expected_scale_shape:
        raise ValueError("grouped W3 scales have the wrong shape")

    output = torch.empty(
        tokens,
        routes,
        output_size,
        dtype=torch.bfloat16,
        device=activations.device,
    )
    grid = (tokens * routes, triton.cdiv(output_size, BLOCK_N))
    with torch.cuda.device(activations.device):
        _fused_swiglu_w4a16_kernel[grid](
            activations,
            expert_indices,
            w1_packed,
            w1_scales,
            w3_packed,
            w3_scales,
            output,
            routes,
            output_size,
            reduction,
            activations.stride(0),
            activations.stride(1),
            expert_indices.stride(0),
            expert_indices.stride(1),
            w1_packed.stride(0),
            w1_packed.stride(1),
            w1_packed.stride(2),
            w1_scales.stride(0),
            w1_scales.stride(1),
            w1_scales.stride(2),
            w3_packed.stride(0),
            w3_packed.stride(1),
            w3_packed.stride(2),
            w3_scales.stride(0),
            w3_scales.stride(1),
            w3_scales.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP=GROUP_SIZE,
            HP_ACTIVATION=hp_activation,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return output
