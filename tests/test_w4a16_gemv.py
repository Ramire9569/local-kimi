from __future__ import annotations

import pytest
import torch


def _dequantise_w4a16_cpu(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Small CPU reference for [N, K / 2] packed weights."""
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


def test_w4a16_codec_nibble_sign_and_group_indexing() -> None:
    nibbles = torch.arange(64, dtype=torch.uint8) % 16
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).reshape(1, 32)
    scales = torch.tensor([[1.0, 10.0]], dtype=torch.bfloat16)

    actual = _dequantise_w4a16_cpu(packed, scales)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed).float()
    expected = signed.clone()
    expected[32:] *= 10.0

    assert torch.equal(actual, expected.reshape(1, 64))
    assert actual[0, 0].item() == 0.0
    assert actual[0, 7].item() == 7.0
    assert actual[0, 8].item() == -8.0
    assert actual[0, 15].item() == -1.0
    assert actual[0, 33].item() == 10.0
    assert actual[0, 40].item() == -80.0


@pytest.mark.gpu
def test_grouped_w4a16_gemv_matches_reference_and_split_k_is_deterministic() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_gemv import (
        GEMV_CONFIGS,
        _launch_grouped_w4a16_gemv,
        grouped_w4a16_gemv,
    )
    from engine.kernels.w4a16_grouped import grouped_w4a16_linear

    torch.manual_seed(20260729)
    tokens = 1
    experts = 3
    output_size = 96
    reduction = 64
    activations = torch.randn(
        (tokens, reduction), device="cuda", dtype=torch.bfloat16
    )
    expert_indices = torch.tensor([[0, 2]], device="cuda", dtype=torch.long)
    packed = torch.randint(
        0,
        256,
        (experts, output_size, reduction // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = (
        torch.rand(
            (experts, output_size, reduction // 32),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    arguments = (activations, expert_indices, packed, scales)

    expected = grouped_w4a16_linear(*arguments)
    actual = grouped_w4a16_gemv(*arguments)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)
    assert torch.equal(actual.float().argmax(dim=-1), expected.float().argmax(dim=-1))

    split_config = next(config for config in GEMV_CONFIGS if config.split_k == 2)
    first = _launch_grouped_w4a16_gemv(*arguments, split_config)
    second = _launch_grouped_w4a16_gemv(*arguments, split_config)
    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected, atol=0.125, rtol=0.05)
