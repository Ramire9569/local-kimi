"""Batch-1 dense W4A16 GEMV with a prefill-safe GEMM fallback."""

from __future__ import annotations

from typing import NamedTuple

import torch

from engine.quant.triton_w4a16 import w4a16_linear
from engine.quant.w4a16 import GROUP_SIZE, W4A16Tensor

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


class DenseGemvConfig(NamedTuple):
    """One explicit launch candidate for the batch-1 dense GEMV."""

    name: str
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int


# Small output dimensions need split-K to create enough work for 142 SMs.
# The 163840-row lm_head already supplies thousands of N tiles and should not
# pay for a partial buffer unless a real measurement proves otherwise.
DENSE_GEMV_CONFIGS = (
    DenseGemvConfig("n32_k64_s2_w4_st3", 32, 64, 2, 4, 3),
    DenseGemvConfig("n32_k64_s4_w4_st3", 32, 64, 4, 4, 3),
    DenseGemvConfig("n64_k64_s4_w4_st4", 64, 64, 4, 4, 4),
    DenseGemvConfig("n32_k128_s2_w8_st2", 32, 128, 2, 8, 2),
    DenseGemvConfig("n64_k64_s1_w4_st3", 64, 64, 1, 4, 3),
    DenseGemvConfig("n128_k32_s1_w8_st3", 128, 32, 1, 8, 3),
    # Added after the first sweep. Every 32-wide candidate above also carried
    # split-K, so a narrow tile WITHOUT a partial-buffer reduction was never
    # measured. It is the config that won the grouped sweep, and at N=4096 it
    # yields 128 programs against 64 for the 64-wide tile, which matters on a
    # 142-SM card.
    DenseGemvConfig("n32_k64_s1_w4_st3", 32, 64, 1, 4, 3),
    DenseGemvConfig("n16_k64_s1_w4_st3", 16, 64, 1, 4, 3),
)


if triton is not None:

    @triton.jit
    def _w4a16_dense_gemv_kernel(
        activation_ptr,
        packed_ptr,
        scale_ptr,
        result_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ak,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sk,
        stride_on,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        block_n = tl.program_id(0)
        split = tl.program_id(1)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_byte = tl.arange(0, BLOCK_K // 2)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Each packed byte supplies the even and odd K values together. K is
        # the lane-contiguous tile axis for packed weights shaped [N, K / 2].
        for k_group_start in range(0, K, BLOCK_K * SPLIT_K):
            current_even_k = (
                k_group_start + split * BLOCK_K + offsets_byte * 2
            )
            current_odd_k = current_even_k + 1
            activation_even = tl.load(
                activation_ptr + current_even_k * stride_ak,
                mask=current_even_k < K,
                other=0.0,
            )
            activation_odd = tl.load(
                activation_ptr + current_odd_k * stride_ak,
                mask=current_odd_k < K,
                other=0.0,
            )
            packed_byte = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + (current_even_k[None, :] // 2) * stride_pk,
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
            output_offsets = result_ptr + offsets_n * stride_on
        else:
            output_offsets = result_ptr + split * N + offsets_n
        tl.store(output_offsets, accumulator, mask=offsets_n < N)


    @triton.jit
    def _reduce_w4a16_dense_partials_kernel(
        partial_ptr,
        output_ptr,
        N: tl.constexpr,
        stride_on,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        block_n = tl.program_id(0)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_split = tl.arange(0, SPLIT_K)
        partials = tl.load(
            partial_ptr
            + offsets_split[:, None] * N
            + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        result = tl.sum(partials, axis=0)
        tl.store(
            output_ptr + offsets_n * stride_on,
            result,
            mask=offsets_n < N,
        )

else:
    _w4a16_dense_gemv_kernel = None
    _reduce_w4a16_dense_partials_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if packed_weights.ndim != 2 or packed_weights.dtype != torch.uint8:
        raise TypeError("packed weights must be a uint8 [output, reduction / 2] matrix")
    if scales.ndim != 2 or scales.dtype != torch.bfloat16:
        raise TypeError("scales must be a BF16 [output, reduction / 32] matrix")
    if activations.device.type != "cuda":
        raise ValueError("W4A16 dense GEMV requires CUDA")
    if not (activations.device == packed_weights.device == scales.device):
        raise ValueError("activations, packed weights, and scales must share a device")

    m, reduction = activations.shape
    output_size, packed_reduction = packed_weights.shape
    if output_size == 0 or reduction == 0:
        raise ValueError("W4A16 dense GEMV requires nonzero dimensions")
    if reduction % GROUP_SIZE:
        raise ValueError("the reduction dimension must be divisible by group size 32")
    if packed_reduction * 2 != reduction:
        raise ValueError("packed weights have the wrong reduction width")
    if scales.shape != (output_size, reduction // GROUP_SIZE):
        raise ValueError("W4A16 scales have the wrong shape")
    return m, output_size, reduction


def _select_dense_config(output_size: int) -> DenseGemvConfig:
    if output_size <= 3072:
        return DENSE_GEMV_CONFIGS[1]
    if output_size <= 8192:
        return DENSE_GEMV_CONFIGS[0]
    return DENSE_GEMV_CONFIGS[4]


def _launch_w4a16_dense_gemv(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    config: DenseGemvConfig,
) -> torch.Tensor:
    """Launch one explicit batch-1 candidate for the benchmark harness."""
    if triton is None or _w4a16_dense_gemv_kernel is None:
        raise RuntimeError("Triton is required for W4A16 dense GEMV")
    m, output_size, reduction = _validate_inputs(
        activations, packed_weights, scales
    )
    if m != 1:
        raise ValueError("the explicit dense GEMV launcher requires exactly one token")

    output = torch.empty(
        (1, output_size), dtype=torch.bfloat16, device=activations.device
    )
    result = output
    if config.split_k > 1:
        result = torch.empty(
            (config.split_k, output_size),
            dtype=torch.float32,
            device=activations.device,
        )

    grid = (triton.cdiv(output_size, config.block_n), config.split_k)
    with torch.cuda.device(activations.device):
        _w4a16_dense_gemv_kernel[grid](
            activations,
            packed_weights,
            scales,
            result,
            N=output_size,
            K=reduction,
            stride_ak=activations.stride(1),
            stride_pn=packed_weights.stride(0),
            stride_pk=packed_weights.stride(1),
            stride_sn=scales.stride(0),
            stride_sk=scales.stride(1),
            stride_on=output.stride(1),
            GROUP=GROUP_SIZE,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            SPLIT_K=config.split_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if config.split_k > 1:
            reduction_grid = (triton.cdiv(output_size, config.block_n),)
            _reduce_w4a16_dense_partials_kernel[reduction_grid](
                result,
                output,
                N=output_size,
                stride_on=output.stride(1),
                BLOCK_N=config.block_n,
                SPLIT_K=config.split_k,
                num_warps=4,
                num_stages=1,
            )
    return output


def _as_encoded(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    reduction: int,
) -> W4A16Tensor:
    return W4A16Tensor(
        packed=packed_weights,
        scales=scales,
        original_shape=(output_size, reduction),
        original_dtype=torch.bfloat16,
        group_size=GROUP_SIZE,
    )


def w4a16_dense_gemv(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Use the GEMV for one token and the existing tensor-core GEMM otherwise."""
    if triton is None:
        raise RuntimeError("Triton is required for W4A16 dense GEMV")
    m, output_size, reduction = _validate_inputs(
        activations, packed_weights, scales
    )
    if m == 0:
        return torch.empty(
            (0, output_size), dtype=torch.bfloat16, device=activations.device
        )
    if m > 1:
        encoded = _as_encoded(
            packed_weights, scales, output_size, reduction
        )
        return w4a16_linear(activations, encoded)

    config = _select_dense_config(output_size)
    return _launch_w4a16_dense_gemv(
        activations, packed_weights, scales, config
    )
