from __future__ import annotations

import pytest
import torch


def _dequantise_dense_w4a16_cpu(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """CPU reference for packed [N, K / 2] dense W4A16 weights."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise TypeError("packed must be a uint8 matrix")
    if scales.dtype != torch.bfloat16 or scales.ndim != 2:
        raise TypeError("scales must be a BF16 matrix")

    low = packed & 0xF
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    group_indices = torch.arange(signed.shape[1], dtype=torch.long) // 32
    return signed.float() * scales.float()[:, group_indices]


def test_dense_w4a16_codec_low_high_sign_and_group_scale() -> None:
    first = torch.arange(64, dtype=torch.uint8) % 16
    second = torch.flip(first, dims=(0,))
    nibbles = torch.stack((first, second))
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scales = torch.tensor([[1.0, 10.0], [0.5, 2.0]], dtype=torch.bfloat16)

    actual = _dequantise_dense_w4a16_cpu(packed, scales)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed).float()
    expected = signed.clone()
    expected[0, 32:] *= 10.0
    expected[1, :32] *= 0.5
    expected[1, 32:] *= 2.0

    assert torch.equal(actual, expected)
    assert actual[0, 7].item() == 7.0
    assert actual[0, 8].item() == -8.0
    assert actual[0, 15].item() == -1.0
    assert actual[0, 33].item() == 10.0
    assert actual[1, 0].item() == -0.5


def _gpu_inputs(m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_size = 256
    reduction = 256
    activations = torch.randn(
        (m, reduction), device="cuda", dtype=torch.bfloat16
    )
    packed = torch.randint(
        0,
        256,
        (output_size, reduction // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = (
        torch.rand(
            (output_size, reduction // 32),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    return activations, packed, scales


@pytest.mark.gpu
def test_dense_w4a16_decode_matches_reference_and_split_k_is_deterministic() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_dense_gemv import (
        DENSE_GEMV_CONFIGS,
        _launch_w4a16_dense_gemv,
        w4a16_dense_gemv,
    )
    from engine.quant.triton_w4a16 import w4a16_linear
    from engine.quant.w4a16 import W4A16Tensor

    torch.manual_seed(20260729)
    activations, packed, scales = _gpu_inputs(m=1)
    encoded = W4A16Tensor(
        packed=packed,
        scales=scales,
        original_shape=(packed.shape[0], activations.shape[1]),
    )

    expected = w4a16_linear(activations, encoded)
    actual = w4a16_dense_gemv(activations, packed, scales)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)
    assert torch.equal(actual.float().argmax(dim=-1), expected.float().argmax(dim=-1))

    split_config = next(
        config for config in DENSE_GEMV_CONFIGS if config.split_k == 2
    )
    first = _launch_w4a16_dense_gemv(
        activations, packed, scales, split_config
    )
    second = _launch_w4a16_dense_gemv(
        activations, packed, scales, split_config
    )
    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected, atol=0.125, rtol=0.05)


@pytest.mark.gpu
def test_dense_w4a16_prefill_uses_the_unchanged_reference_path() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_dense_gemv import w4a16_dense_gemv
    from engine.quant.triton_w4a16 import w4a16_linear
    from engine.quant.w4a16 import W4A16Tensor

    torch.manual_seed(20260730)
    activations, packed, scales = _gpu_inputs(m=3)
    encoded = W4A16Tensor(
        packed=packed,
        scales=scales,
        original_shape=(packed.shape[0], activations.shape[1]),
    )

    expected = w4a16_linear(activations, encoded)
    actual = w4a16_dense_gemv(activations, packed, scales)

    assert torch.equal(actual, expected)
