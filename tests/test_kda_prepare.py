from __future__ import annotations

import math

import pytest
import torch

from engine.kernels.kda_prepare import (
    fused_kda_gates,
    fused_kda_gates_reference,
    fused_short_conv_triple,
    fused_short_conv_triple_reference,
)


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _require_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def test_short_conv_triple_reference_matches_hand_computed_decode() -> None:
    q_in = torch.tensor([[[5.0]]])
    k_in = torch.tensor([[[0.0]]])
    v_in = torch.tensor([[[6.0]]])
    q_state = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    k_state = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    v_state = torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]])
    q_weight = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    k_weight = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    v_weight = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

    q_out, k_out, v_out = fused_short_conv_triple_reference(
        q_in,
        k_in,
        v_in,
        q_state,
        k_state,
        v_state,
        q_weight,
        k_weight,
        v_weight,
    )

    expected_outputs = (
        torch.tensor([[[_silu(2.0)]]]),
        torch.tensor([[[_silu(1.0)]]]),
        torch.tensor([[[_silu(-3.0)]]]),
    )
    for actual, expected in zip((q_out, k_out, v_out), expected_outputs):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(q_state, torch.tensor([[[2.0, 3.0, 4.0, 5.0]]]))
    assert torch.equal(k_state, torch.tensor([[[3.0, 2.0, 1.0, 0.0]]]))
    assert torch.equal(v_state, torch.tensor([[[-2.0, -3.0, -4.0, 6.0]]]))


def test_kda_gates_reference_matches_hand_computed_heads() -> None:
    raw_decay = torch.tensor([[[[0.0, 1.0], [-1.0, 2.0]]]])
    dt_bias = torch.tensor([0.5, -0.5, 1.0, -1.0])
    a_log = torch.log(torch.tensor([2.0, 0.5])).view(1, 1, 2, 1)
    raw_beta = torch.tensor([[[0.0, math.log(3.0)]]])
    q = torch.tensor([[[3.0, 4.0, 0.0, 2.0]]])
    k = torch.tensor([[[5.0, 12.0, 8.0, 15.0]]])
    eps = 1e-6

    decay, beta, q_normalised, k_normalised = fused_kda_gates_reference(
        raw_decay,
        dt_bias,
        a_log,
        raw_beta,
        q,
        k,
        num_heads=2,
        head_dim=2,
        eps=eps,
    )

    expected_decay = torch.tensor(
        [
            [
                [
                    [
                        -2.0 * math.log1p(math.exp(0.5)),
                        -2.0 * math.log1p(math.exp(0.5)),
                    ],
                    [-0.5 * math.log1p(math.exp(0.0)), -0.5 * math.log1p(math.exp(1.0))],
                ]
            ]
        ]
    )
    expected_beta = torch.tensor([[[0.5, 0.75]]])
    # Shape is [batch, sequence, heads, head_dim]. attention.py line 173 views q
    # and k with the sequence axis kept, so the normalised result has four dims
    # even at decode where sequence is 1. An earlier version of this expectation
    # omitted that axis and failed on shape rather than on value.
    expected_q = torch.tensor(
        [
            [
                [
                    [3.0 / math.sqrt(25.0 + eps), 4.0 / math.sqrt(25.0 + eps)],
                    [0.0, 2.0 / math.sqrt(4.0 + eps)],
                ]
            ]
        ]
    )
    expected_k = torch.tensor(
        [
            [
                [
                    [5.0 / math.sqrt(169.0 + eps), 12.0 / math.sqrt(169.0 + eps)],
                    [8.0 / math.sqrt(289.0 + eps), 15.0 / math.sqrt(289.0 + eps)],
                ]
            ]
        ]
    )
    torch.testing.assert_close(decay, expected_decay, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(beta, expected_beta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q_normalised, expected_q, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_normalised, expected_k, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "conv_atol", "conv_rtol"),
    [
        (torch.float16, 0.002, 0.002),
        (torch.bfloat16, 0.02, 0.02),
    ],
)
def test_fused_kda_prepare_matches_references_on_gpu(
    dtype: torch.dtype,
    conv_atol: float,
    conv_rtol: float,
) -> None:
    _require_cuda()
    torch.manual_seed(20260729)
    batch = 2
    channels = 4096
    heads = 32
    head_dim = 128
    values = [
        torch.randn(batch, 1, channels, device="cuda", dtype=dtype) for _ in range(3)
    ]
    initial_states = [
        torch.randn(batch, channels, 4, device="cuda", dtype=dtype) for _ in range(3)
    ]
    reference_states = [state.clone() for state in initial_states]
    fused_states = [state.clone() for state in initial_states]
    weights = [
        torch.randn(channels, 1, 4, device="cuda", dtype=dtype) * 0.02
        for _ in range(3)
    ]

    expected_conv = fused_short_conv_triple_reference(
        *values,
        *reference_states,
        *weights,
    )
    actual_conv = fused_short_conv_triple(
        *values,
        *fused_states,
        *weights,
    )
    for actual, expected in zip(actual_conv, expected_conv):
        torch.testing.assert_close(actual, expected, atol=conv_atol, rtol=conv_rtol)
    for actual, expected in zip(fused_states, reference_states):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    raw_decay = torch.randn(
        batch, 1, heads, head_dim, device="cuda", dtype=dtype
    )
    dt_bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    a_log = torch.log(
        torch.empty(heads, device="cuda", dtype=torch.float32).uniform_(1.0, 16.0)
    ).view(1, 1, heads, 1)
    raw_beta = torch.randn(batch, 1, heads, device="cuda", dtype=dtype)
    expected_gates = fused_kda_gates_reference(
        raw_decay,
        dt_bias,
        a_log,
        raw_beta,
        expected_conv[0],
        expected_conv[1],
        num_heads=heads,
        head_dim=head_dim,
        eps=1e-6,
    )
    actual_gates = fused_kda_gates(
        raw_decay,
        dt_bias,
        a_log,
        raw_beta,
        actual_conv[0],
        actual_conv[1],
        num_heads=heads,
        head_dim=head_dim,
        eps=1e-6,
    )
    for actual, expected in zip(actual_gates, expected_gates):
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)


@pytest.mark.gpu
def test_fused_kda_prepare_is_cuda_graph_capturable() -> None:
    _require_cuda()
    dtype = torch.bfloat16
    channels = 4096
    heads = 32
    head_dim = 128
    values = [torch.randn(1, 1, channels, device="cuda", dtype=dtype) for _ in range(3)]
    states = [torch.zeros(1, channels, 4, device="cuda", dtype=dtype) for _ in range(3)]
    weights = [
        torch.randn(channels, 1, 4, device="cuda", dtype=dtype) * 0.02
        for _ in range(3)
    ]
    raw_decay = torch.randn(1, 1, heads, head_dim, device="cuda", dtype=dtype)
    dt_bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    a_log = torch.zeros(1, 1, heads, 1, device="cuda", dtype=torch.float32)
    raw_beta = torch.randn(1, 1, heads, device="cuda", dtype=dtype)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            q_out, k_out, _ = fused_short_conv_triple(*values, *states, *weights)
            fused_kda_gates(
                raw_decay,
                dt_bias,
                a_log,
                raw_beta,
                q_out,
                k_out,
                num_heads=heads,
                head_dim=head_dim,
                eps=1e-6,
            )
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph):
            captured_conv = fused_short_conv_triple(*values, *states, *weights)
            captured_gates = fused_kda_gates(
                raw_decay,
                dt_bias,
                a_log,
                raw_beta,
                captured_conv[0],
                captured_conv[1],
                num_heads=heads,
                head_dim=head_dim,
                eps=1e-6,
            )
    graph.replay()
    torch.cuda.synchronize()

    for tensor in (*captured_conv, *captured_gates, *states):
        assert torch.isfinite(tensor).all()
