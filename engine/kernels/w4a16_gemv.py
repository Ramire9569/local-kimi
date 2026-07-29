"""Batch-1 grouped W4A16 GEMV with K-contiguous weight reads."""

from __future__ import annotations

from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None

GROUP_SIZE = 32


class GemvConfig(NamedTuple):
    """One explicit launch candidate for the batch-1 GEMV."""

    name: str
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int


# Keep this list short so BENCH-GEMV.py can measure every candidate quickly.
# BLOCK_N=32 produces 288 programs for the 9-route, N=1024 projection without
# split-K. BLOCK_N=64 produces 324 programs for N=2304. The split candidates
# test whether fewer, wider N tiles plus more K parallelism win on the L40S.
GEMV_CONFIGS = (
    GemvConfig("n32_k64_s1_w4_st3", 32, 64, 1, 4, 3),
    GemvConfig("n64_k64_s1_w4_st3", 64, 64, 1, 4, 3),
    GemvConfig("n64_k64_s2_w4_st4", 64, 64, 2, 4, 4),
    GemvConfig("n128_k32_s4_w8_st3", 128, 32, 4, 8, 3),
    GemvConfig("n32_k128_s1_w8_st2", 32, 128, 1, 8, 2),
)

# The first entry deliberately matches the shipped grouped GEMV tiling. The
# remaining entries are isolated-benchmark hypotheses only. End-to-end decode
# must decide whether any of them should replace the conservative default.
FUSED_SWIGLU_CONFIGS = (
    GemvConfig("fused_n32_k64_s1_w4_st3", 32, 64, 1, 4, 3),
    GemvConfig("fused_n32_k64_s1_w8_st3", 32, 64, 1, 8, 3),
    GemvConfig("fused_n32_k32_s1_w4_st4", 32, 32, 1, 4, 4),
    GemvConfig("fused_n64_k64_s1_w8_st3", 64, 64, 1, 8, 3),
    GemvConfig("fused_n32_k64_s2_w4_st3", 32, 64, 2, 4, 3),
)


if triton is not None:

    @triton.jit
    def _grouped_w4a16_gemv_kernel(
        activation_ptr,
        expert_index_ptr,
        packed_ptr,
        scale_ptr,
        result_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_at,
        stride_ak,
        stride_it,
        stride_ir,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        stride_ot,
        stride_or,
        stride_on,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        split = tl.program_id(2)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        expert = tl.load(
            expert_index_ptr + token * stride_it + route * stride_ir
        ).to(tl.int64)

        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_byte = tl.arange(0, BLOCK_K // 2)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Interleave whole K tiles between split programs. Every K value belongs
        # to exactly one split, and the second pass reduces splits in a fixed
        # tree. The [N, K] tile orientation makes K the lane-contiguous axis for
        # packed weights stored as [expert, N, K / 2].
        for k_group_start in range(0, K, BLOCK_K * SPLIT_K):
            current_even_k = (
                k_group_start + split * BLOCK_K + offsets_byte * 2
            )
            current_odd_k = current_even_k + 1
            activation_even = tl.load(
                activation_ptr
                + token * stride_at
                + current_even_k * stride_ak,
                mask=current_even_k < K,
                other=0.0,
            )
            activation_odd = tl.load(
                activation_ptr
                + token * stride_at
                + current_odd_k * stride_ak,
                mask=current_odd_k < K,
                other=0.0,
            )
            packed_byte = tl.load(
                packed_ptr
                + expert * stride_we
                + offsets_n[:, None] * stride_wn
                + (current_even_k[None, :] // 2) * stride_wk,
                mask=(offsets_n[:, None] < N)
                & (current_even_k[None, :] < K),
                other=0,
            )
            low_nibble = packed_byte & 0xF
            high_nibble = packed_byte >> 4
            low_signed = low_nibble.to(tl.int32)
            low_signed = tl.where(low_signed >= 8, low_signed - 16, low_signed)
            high_signed = high_nibble.to(tl.int32)
            high_signed = tl.where(
                high_signed >= 8, high_signed - 16, high_signed
            )
            scale = tl.load(
                scale_ptr
                + expert * stride_se
                + offsets_n[:, None] * stride_sn
                + (current_even_k[None, :] // GROUP) * stride_sk,
                mask=(offsets_n[:, None] < N)
                & (current_even_k[None, :] < K),
                other=1.0,
            )
            decoded_low = (
                low_signed.to(tl.float32) * scale.to(tl.float32)
            ).to(tl.bfloat16)
            decoded_high = (
                high_signed.to(tl.float32) * scale.to(tl.float32)
            ).to(tl.bfloat16)
            products = (
                activation_even[None, :].to(tl.float32)
                * decoded_low.to(tl.float32)
                + activation_odd[None, :].to(tl.float32)
                * decoded_high.to(tl.float32)
            )
            accumulator += tl.sum(products, axis=1)

        if SPLIT_K == 1:
            output_offsets = (
                result_ptr
                + token * stride_ot
                + route * stride_or
                + offsets_n * stride_on
            )
        else:
            output_offsets = (
                result_ptr
                + assignment * SPLIT_K * N
                + split * N
                + offsets_n
            )
        tl.store(output_offsets, accumulator, mask=offsets_n < N)


    @triton.jit
    def _reduce_grouped_w4a16_partials_kernel(
        partial_ptr,
        output_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        stride_ot,
        stride_or,
        stride_on,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_split = tl.arange(0, SPLIT_K)
        partials = tl.load(
            partial_ptr
            + assignment * SPLIT_K * N
            + offsets_split[:, None] * N
            + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        result = tl.sum(partials, axis=0)
        tl.store(
            output_ptr
            + token * stride_ot
            + route * stride_or
            + offsets_n * stride_on,
            result,
            mask=offsets_n < N,
        )


    @triton.jit
    def _rounded_silu_mul(gate_accumulator, up_accumulator):
        # Match F.silu(gate_bf16) * up_bf16. PyTorch returns BF16 from SiLU
        # for a BF16 input, then rounds the BF16 multiply back to BF16.
        gate = gate_accumulator.to(tl.bfloat16).to(tl.float32)
        up = up_accumulator.to(tl.bfloat16).to(tl.float32)
        silu_gate = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
        return silu_gate.to(tl.float32) * up


    @triton.jit
    def _grouped_w4a16_swiglu_gemv_kernel(
        activation_ptr,
        expert_index_ptr,
        w1_packed_ptr,
        w1_scale_ptr,
        w3_packed_ptr,
        w3_scale_ptr,
        result_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
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
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        split = tl.program_id(2)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        expert = tl.load(
            expert_index_ptr + token * stride_it + route * stride_ir
        ).to(tl.int64)

        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_byte = tl.arange(0, BLOCK_K // 2)
        gate_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_group_start in range(0, K, BLOCK_K * SPLIT_K):
            current_even_k = (
                k_group_start + split * BLOCK_K + offsets_byte * 2
            )
            current_odd_k = current_even_k + 1
            activation_even = tl.load(
                activation_ptr
                + token * stride_at
                + current_even_k * stride_ak,
                mask=current_even_k < K,
                other=0.0,
            ).to(tl.float32)
            activation_odd = tl.load(
                activation_ptr
                + token * stride_at
                + current_odd_k * stride_ak,
                mask=current_odd_k < K,
                other=0.0,
            ).to(tl.float32)
            weight_mask = (
                (offsets_n[:, None] < N)
                & (current_even_k[None, :] < K)
            )

            w1_packed = tl.load(
                w1_packed_ptr
                + expert * stride_w1e
                + offsets_n[:, None] * stride_w1n
                + (current_even_k[None, :] // 2) * stride_w1k,
                mask=weight_mask,
                other=0,
            )
            w1_low = w1_packed & 0xF
            w1_high = w1_packed >> 4
            w1_low_signed = w1_low.to(tl.int32)
            w1_low_signed = tl.where(
                w1_low_signed >= 8, w1_low_signed - 16, w1_low_signed
            )
            w1_high_signed = w1_high.to(tl.int32)
            w1_high_signed = tl.where(
                w1_high_signed >= 8, w1_high_signed - 16, w1_high_signed
            )
            w1_scale = tl.load(
                w1_scale_ptr
                + expert * stride_s1e
                + offsets_n[:, None] * stride_s1n
                + (current_even_k[None, :] // GROUP) * stride_s1k,
                mask=weight_mask,
                other=1.0,
            )
            w1_low_decoded = (
                w1_low_signed.to(tl.float32) * w1_scale.to(tl.float32)
            ).to(tl.bfloat16)
            w1_high_decoded = (
                w1_high_signed.to(tl.float32) * w1_scale.to(tl.float32)
            ).to(tl.bfloat16)
            gate_products = (
                activation_even[None, :] * w1_low_decoded.to(tl.float32)
                + activation_odd[None, :] * w1_high_decoded.to(tl.float32)
            )
            gate_accumulator += tl.sum(gate_products, axis=1)

            w3_packed = tl.load(
                w3_packed_ptr
                + expert * stride_w3e
                + offsets_n[:, None] * stride_w3n
                + (current_even_k[None, :] // 2) * stride_w3k,
                mask=weight_mask,
                other=0,
            )
            w3_low = w3_packed & 0xF
            w3_high = w3_packed >> 4
            w3_low_signed = w3_low.to(tl.int32)
            w3_low_signed = tl.where(
                w3_low_signed >= 8, w3_low_signed - 16, w3_low_signed
            )
            w3_high_signed = w3_high.to(tl.int32)
            w3_high_signed = tl.where(
                w3_high_signed >= 8, w3_high_signed - 16, w3_high_signed
            )
            w3_scale = tl.load(
                w3_scale_ptr
                + expert * stride_s3e
                + offsets_n[:, None] * stride_s3n
                + (current_even_k[None, :] // GROUP) * stride_s3k,
                mask=weight_mask,
                other=1.0,
            )
            w3_low_decoded = (
                w3_low_signed.to(tl.float32) * w3_scale.to(tl.float32)
            ).to(tl.bfloat16)
            w3_high_decoded = (
                w3_high_signed.to(tl.float32) * w3_scale.to(tl.float32)
            ).to(tl.bfloat16)
            up_products = (
                activation_even[None, :] * w3_low_decoded.to(tl.float32)
                + activation_odd[None, :] * w3_high_decoded.to(tl.float32)
            )
            up_accumulator += tl.sum(up_products, axis=1)

        if SPLIT_K == 1:
            output_offsets = (
                result_ptr
                + token * stride_ot
                + route * stride_or
                + offsets_n * stride_on
            )
            activated = _rounded_silu_mul(gate_accumulator, up_accumulator)
            tl.store(output_offsets, activated, mask=offsets_n < N)
        else:
            gate_offsets = (
                result_ptr
                + assignment * 2 * SPLIT_K * N
                + split * N
                + offsets_n
            )
            up_offsets = gate_offsets + SPLIT_K * N
            tl.store(gate_offsets, gate_accumulator, mask=offsets_n < N)
            tl.store(up_offsets, up_accumulator, mask=offsets_n < N)


    @triton.jit
    def _reduce_grouped_w4a16_swiglu_partials_kernel(
        partial_ptr,
        output_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        stride_ot,
        stride_or,
        stride_on,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_split = tl.arange(0, SPLIT_K)
        gate_partials = tl.load(
            partial_ptr
            + assignment * 2 * SPLIT_K * N
            + offsets_split[:, None] * N
            + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        up_partials = tl.load(
            partial_ptr
            + assignment * 2 * SPLIT_K * N
            + SPLIT_K * N
            + offsets_split[:, None] * N
            + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        gate = tl.sum(gate_partials, axis=0)
        up = tl.sum(up_partials, axis=0)
        activated = _rounded_silu_mul(gate, up)
        tl.store(
            output_ptr
            + token * stride_ot
            + route * stride_or
            + offsets_n * stride_on,
            activated,
            mask=offsets_n < N,
        )

else:
    _grouped_w4a16_gemv_kernel = None
    _reduce_grouped_w4a16_partials_kernel = None
    _grouped_w4a16_swiglu_gemv_kernel = None
    _reduce_grouped_w4a16_swiglu_partials_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[int, int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if expert_indices.ndim != 2 or expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must be an int64 [tokens, routes] matrix")
    if packed_weights.ndim != 3 or scales.ndim != 3:
        raise ValueError("grouped packed weights and scales must be rank three")
    if packed_weights.dtype != torch.uint8:
        raise TypeError("packed grouped weights must use torch.uint8 storage")
    if scales.dtype != torch.bfloat16:
        raise TypeError("grouped W4A16 scales must use torch.bfloat16 storage")
    if activations.device.type != "cuda":
        raise ValueError("grouped W4A16 decode requires CUDA")
    if not (
        activations.device
        == expert_indices.device
        == packed_weights.device
        == scales.device
    ):
        raise ValueError("grouped W4A16 inputs must share one device")

    tokens, reduction = activations.shape
    route_tokens, routes = expert_indices.shape
    experts, output_size, packed_reduction = packed_weights.shape
    if route_tokens != tokens:
        raise ValueError("expert routing must have one row per token")
    if routes == 0 or experts == 0 or output_size == 0 or reduction == 0:
        raise ValueError("grouped W4A16 decode requires nonzero dimensions")
    if reduction % GROUP_SIZE:
        raise ValueError("grouped W4A16 reduction width must be divisible by 32")
    if packed_reduction * 2 != reduction:
        raise ValueError("packed grouped weights have the wrong reduction width")
    if scales.shape != (experts, output_size, reduction // GROUP_SIZE):
        raise ValueError("grouped W4A16 scales have the wrong shape")
    return tokens, routes, output_size, reduction


def _select_config(output_size: int, assignments: int) -> GemvConfig:
    # These choices target at least two waves on 142 SMs without split traffic.
    if assignments * triton.cdiv(output_size, 32) <= 384:
        return GEMV_CONFIGS[0]
    return GEMV_CONFIGS[1]


def _launch_grouped_w4a16_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    config: GemvConfig,
) -> torch.Tensor:
    """Launch one explicit candidate. BENCH-GEMV.py uses this for comparison."""
    if triton is None or _grouped_w4a16_gemv_kernel is None:
        raise RuntimeError("Triton is required for grouped W4A16 GEMV")
    tokens, routes, output_size, reduction = _validate_inputs(
        activations, expert_indices, packed_weights, scales
    )
    output = torch.empty(
        tokens,
        routes,
        output_size,
        dtype=torch.bfloat16,
        device=activations.device,
    )
    if tokens == 0:
        return output

    assignments = tokens * routes
    grid = (
        assignments,
        triton.cdiv(output_size, config.block_n),
        config.split_k,
    )
    result = output
    if config.split_k > 1:
        result = torch.empty(
            assignments,
            config.split_k,
            output_size,
            dtype=torch.float32,
            device=activations.device,
        )

    with torch.cuda.device(activations.device):
        _grouped_w4a16_gemv_kernel[grid](
            activations,
            expert_indices,
            packed_weights,
            scales,
            result,
            ROUTES=routes,
            N=output_size,
            K=reduction,
            stride_at=activations.stride(0),
            stride_ak=activations.stride(1),
            stride_it=expert_indices.stride(0),
            stride_ir=expert_indices.stride(1),
            stride_we=packed_weights.stride(0),
            stride_wn=packed_weights.stride(1),
            stride_wk=packed_weights.stride(2),
            stride_se=scales.stride(0),
            stride_sn=scales.stride(1),
            stride_sk=scales.stride(2),
            stride_ot=output.stride(0),
            stride_or=output.stride(1),
            stride_on=output.stride(2),
            GROUP=GROUP_SIZE,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            SPLIT_K=config.split_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if config.split_k > 1:
            reduction_grid = (
                assignments,
                triton.cdiv(output_size, config.block_n),
            )
            _reduce_grouped_w4a16_partials_kernel[reduction_grid](
                result,
                output,
                ROUTES=routes,
                N=output_size,
                stride_ot=output.stride(0),
                stride_or=output.stride(1),
                stride_on=output.stride(2),
                BLOCK_N=config.block_n,
                SPLIT_K=config.split_k,
                num_warps=4,
                num_stages=1,
            )
    return output


def grouped_w4a16_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run one packed batch-1 GEMV for every fixed token-route assignment."""
    if triton is None:
        raise RuntimeError("Triton is required for grouped W4A16 GEMV")
    if (
        activations.ndim != 2
        or expert_indices.ndim != 2
        or packed_weights.ndim != 3
    ):
        # Preserve the detailed public validation messages in the launcher.
        config = GEMV_CONFIGS[0]
    else:
        assignments = activations.shape[0] * expert_indices.shape[1]
        config = _select_config(packed_weights.shape[1], assignments)
    return _launch_grouped_w4a16_gemv(
        activations,
        expert_indices,
        packed_weights,
        scales,
        config,
    )


def _validate_swiglu_inputs(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
) -> tuple[int, int, int, int]:
    dimensions = _validate_inputs(
        activations, expert_indices, w1_packed, w1_scales
    )
    w3_dimensions = _validate_inputs(
        activations, expert_indices, w3_packed, w3_scales
    )
    if w3_dimensions != dimensions:
        raise ValueError("grouped W1 and W3 dimensions must match")
    if w3_packed.shape != w1_packed.shape:
        raise ValueError("grouped W1 and W3 packed shapes must match")
    if w3_scales.shape != w1_scales.shape:
        raise ValueError("grouped W1 and W3 scale shapes must match")
    return dimensions


def _select_fused_swiglu_config() -> GemvConfig:
    # Keep the production-facing default aligned with the shipped grouped tile.
    return FUSED_SWIGLU_CONFIGS[0]


def _launch_grouped_w4a16_swiglu_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
    config: GemvConfig,
) -> torch.Tensor:
    """Launch one fused SwiGLU candidate for isolated and end-to-end A/B tests."""
    if triton is None or _grouped_w4a16_swiglu_gemv_kernel is None:
        raise RuntimeError("Triton is required for grouped W4A16 SwiGLU GEMV")
    tokens, routes, output_size, reduction = _validate_swiglu_inputs(
        activations,
        expert_indices,
        w1_packed,
        w1_scales,
        w3_packed,
        w3_scales,
    )
    output = torch.empty(
        tokens,
        routes,
        output_size,
        dtype=torch.bfloat16,
        device=activations.device,
    )
    if tokens == 0:
        return output

    assignments = tokens * routes
    grid = (
        assignments,
        triton.cdiv(output_size, config.block_n),
        config.split_k,
    )
    result = output
    if config.split_k > 1:
        result = torch.empty(
            assignments,
            2,
            config.split_k,
            output_size,
            dtype=torch.float32,
            device=activations.device,
        )

    with torch.cuda.device(activations.device):
        _grouped_w4a16_swiglu_gemv_kernel[grid](
            activations,
            expert_indices,
            w1_packed,
            w1_scales,
            w3_packed,
            w3_scales,
            result,
            ROUTES=routes,
            N=output_size,
            K=reduction,
            stride_at=activations.stride(0),
            stride_ak=activations.stride(1),
            stride_it=expert_indices.stride(0),
            stride_ir=expert_indices.stride(1),
            stride_w1e=w1_packed.stride(0),
            stride_w1n=w1_packed.stride(1),
            stride_w1k=w1_packed.stride(2),
            stride_s1e=w1_scales.stride(0),
            stride_s1n=w1_scales.stride(1),
            stride_s1k=w1_scales.stride(2),
            stride_w3e=w3_packed.stride(0),
            stride_w3n=w3_packed.stride(1),
            stride_w3k=w3_packed.stride(2),
            stride_s3e=w3_scales.stride(0),
            stride_s3n=w3_scales.stride(1),
            stride_s3k=w3_scales.stride(2),
            stride_ot=output.stride(0),
            stride_or=output.stride(1),
            stride_on=output.stride(2),
            GROUP=GROUP_SIZE,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            SPLIT_K=config.split_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if config.split_k > 1:
            reduction_grid = (
                assignments,
                triton.cdiv(output_size, config.block_n),
            )
            _reduce_grouped_w4a16_swiglu_partials_kernel[reduction_grid](
                result,
                output,
                ROUTES=routes,
                N=output_size,
                stride_ot=output.stride(0),
                stride_or=output.stride(1),
                stride_on=output.stride(2),
                BLOCK_N=config.block_n,
                SPLIT_K=config.split_k,
                num_warps=4,
                num_stages=1,
            )
    return output


def grouped_w4a16_swiglu_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
) -> torch.Tensor:
    """Fuse grouped W1, W3, BF16 SiLU rounding, and the BF16 multiply."""
    config = _select_fused_swiglu_config()
    return _launch_grouped_w4a16_swiglu_gemv(
        activations,
        expert_indices,
        w1_packed,
        w1_scales,
        w3_packed,
        w3_scales,
        config,
    )
