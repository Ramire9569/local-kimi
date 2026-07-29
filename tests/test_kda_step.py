from __future__ import annotations

import pytest
import torch

from engine.kernels.kda_step import (
    _kda_decode_step_with_block_v,
    kda_decode_step,
    kda_decode_step_reference,
)

SINGLE_STEP_ATOL = 3e-5
SINGLE_STEP_RTOL = 3e-5
RECURRENT_ATOL = 5e-4
RECURRENT_RTOL = 5e-4


def test_kda_reference_matches_hand_computed_two_head_case() -> None:
    state = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [2.0, 0.0, 1.0, -1.0],
                    [1.0, 3.0, -2.0, 4.0],
                    [0.0, 5.0, 2.0, 1.0],
                    [4.0, -1.0, 0.0, 2.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    q = torch.tensor(
        [[[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[4.5, 7.0, 9.5, 12.0], [7.0, -1.0, 2.0, 4.0]]],
        dtype=torch.float32,
    )
    decay_factors = torch.tensor(
        [[[1.0, 0.5, 2.0, 0.25], [0.5, 1.0, 0.25, 2.0]]],
        dtype=torch.float32,
    )
    beta = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    expected_state = torch.tensor(
        [
            [
                [
                    [1.5, 3.0, 4.5, 6.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [18.0, 20.0, 22.0, 24.0],
                    [3.25, 3.5, 3.75, 4.0],
                ],
                [
                    [1.0, 0.0, 0.5, -0.5],
                    [0.5, 2.5, -1.0, 3.0],
                    [0.0, 1.25, 0.5, 0.25],
                    [7.5, -2.5, 1.0, 3.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    expected_output = torch.tensor(
        [[[-0.375, -0.25, -0.125, 0.0], [-1.75, 1.25, -0.5, 0.0]]],
        dtype=torch.float32,
    )

    output = kda_decode_step_reference(
        state,
        q,
        k,
        v,
        decay_factors.log(),
        beta,
        scale=0.25,
    )

    torch.testing.assert_close(state, expected_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6)


def _require_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _normalized_random(shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.randn(shape, device="cuda", dtype=torch.float32)
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / norm.clamp_min(1e-6)


def _real_shape_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260729)
    state = 0.05 * torch.randn(
        1,
        32,
        128,
        128,
        device="cuda",
        dtype=torch.float32,
    )
    q = _normalized_random((1, 32, 128))
    k = _normalized_random((1, 32, 128))
    v = 0.2 * torch.randn(1, 32, 128, device="cuda", dtype=torch.float32)
    decay = -(0.005 + 0.075 * torch.rand(
        1,
        32,
        128,
        device="cuda",
        dtype=torch.float32,
    ))
    beta = torch.sigmoid(
        torch.randn(1, 32, device="cuda", dtype=torch.float32)
    )
    return state, q, k, v, decay, beta


@pytest.mark.gpu
@pytest.mark.parametrize("block_v", [16, 32, 64])
def test_fused_kda_matches_reference_at_real_shape(block_v: int) -> None:
    _require_cuda()
    state, q, k, v, decay, beta = _real_shape_inputs()
    expected_state = state.clone()
    actual_state = state.clone()

    expected_output = kda_decode_step_reference(
        expected_state,
        q,
        k,
        v,
        decay,
        beta,
        scale=128**-0.5,
    )
    actual_output = _kda_decode_step_with_block_v(
        actual_state,
        q,
        k,
        v,
        decay,
        beta,
        scale=128**-0.5,
        block_v=block_v,
    )

    torch.testing.assert_close(
        actual_state,
        expected_state,
        atol=SINGLE_STEP_ATOL,
        rtol=SINGLE_STEP_RTOL,
    )
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=SINGLE_STEP_ATOL,
        rtol=SINGLE_STEP_RTOL,
    )


@pytest.mark.gpu
def test_fused_kda_error_stays_bounded_after_256_recurrent_steps() -> None:
    _require_cuda()
    torch.manual_seed(314159)
    steps = 256
    batch = 1
    heads = 4
    dimension = 128
    shape = (steps, batch, heads, dimension)
    q = _normalized_random(shape)
    k = _normalized_random(shape)
    v = 0.2 * torch.randn(shape, device="cuda", dtype=torch.float32)
    decay = -(0.005 + 0.075 * torch.rand(
        shape,
        device="cuda",
        dtype=torch.float32,
    ))
    beta = torch.sigmoid(
        torch.randn(
            steps,
            batch,
            heads,
            device="cuda",
            dtype=torch.float32,
        )
    )
    expected_state = torch.zeros(
        batch,
        heads,
        dimension,
        dimension,
        device="cuda",
        dtype=torch.float32,
    )
    actual_state = expected_state.clone()

    for token_index in range(steps):
        expected_output = kda_decode_step_reference(
            expected_state,
            q[token_index],
            k[token_index],
            v[token_index],
            decay[token_index],
            beta[token_index],
            scale=dimension**-0.5,
        )
        actual_output = kda_decode_step(
            actual_state,
            q[token_index],
            k[token_index],
            v[token_index],
            decay[token_index],
            beta[token_index],
            scale=dimension**-0.5,
        )

    torch.testing.assert_close(
        actual_state,
        expected_state,
        atol=RECURRENT_ATOL,
        rtol=RECURRENT_RTOL,
    )
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=RECURRENT_ATOL,
        rtol=RECURRENT_RTOL,
    )
