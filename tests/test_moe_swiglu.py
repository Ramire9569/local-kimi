from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from engine.kernels.moe_swiglu import fused_swiglu_w4a16
from engine.kernels.w4a16_grouped import grouped_w4a16_linear


def test_default_api_pins_bf16_rounding_before_silu() -> None:
    gate_accumulator = torch.tensor([1.003], dtype=torch.float32)
    up_accumulator = torch.tensor([1.003], dtype=torch.float32)

    bf16_first = F.silu(gate_accumulator.to(torch.bfloat16)) * up_accumulator.to(
        torch.bfloat16
    )
    float32_through_activation = (
        F.silu(gate_accumulator) * up_accumulator
    ).to(torch.bfloat16)

    assert not torch.equal(bf16_first, float32_through_activation)

    default_hp_activation = inspect.signature(fused_swiglu_w4a16).parameters[
        "hp_activation"
    ].default
    default_result = (
        float32_through_activation if default_hp_activation else bf16_first
    )

    assert default_hp_activation is False
    assert torch.equal(default_result, bf16_first)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_swiglu_matches_current_grouped_path() -> None:
    torch.manual_seed(20260729)
    device = torch.device("cuda")
    tokens = 2
    routes = 3
    experts = 3
    output_size = 65
    reduction = 64

    activations = torch.empty(
        (tokens, reduction),
        device=device,
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.25)
    expert_indices = torch.tensor(
        [[0, 1, 2], [2, 0, 1]],
        device=device,
        dtype=torch.long,
    )
    packed_shape = (experts, output_size, reduction // 2)
    scale_shape = (experts, output_size, reduction // 32)
    w1_packed = torch.randint(
        0,
        256,
        packed_shape,
        device=device,
        dtype=torch.uint8,
    )
    w3_packed = torch.randint(
        0,
        256,
        packed_shape,
        device=device,
        dtype=torch.uint8,
    )
    w1_scales = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(0.01, 0.125)
    w3_scales = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(0.01, 0.125)

    gate = grouped_w4a16_linear(
        activations,
        expert_indices,
        w1_packed,
        w1_scales,
    )
    up = grouped_w4a16_linear(
        activations,
        expert_indices,
        w3_packed,
        w3_scales,
    )
    expected = F.silu(gate) * up

    actual = fused_swiglu_w4a16(
        activations,
        expert_indices,
        w1_packed,
        w1_scales,
        w3_packed,
        w3_scales,
    )

    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
