"""Benchmark KDA decode preparation at the Kimi K3 production shape."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from engine.kernels.kda_prepare import (
    fused_kda_gates,
    fused_kda_gates_reference,
    fused_short_conv_triple,
    fused_short_conv_triple_reference,
)


BATCH = 1
SEQUENCE = 1
PROJECTION_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
CONV_WIDTH = 4
KDA_LAYERS = 20

# Analytical eager counts for low-precision decode. Views and unsqueezes are
# metadata only and are not counted. The three state copy kernels performed by
# the static-cache path are counted because the fused convolution removes them.
REFERENCE_SHORT_CONV_LAUNCHES = 27
REFERENCE_GATE_LAUNCHES = 20
FUSED_SHORT_CONV_LAUNCHES = 1
FUSED_GATE_LAUNCHES = 1


def _time_cuda(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for _ in range(iterations):
        result = function()
    end.record()
    end.synchronize()
    if result is None:
        raise AssertionError("benchmark produced no result")
    return start.elapsed_time(end) * 1000.0 / iterations


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-7)
    return difference.max().item(), relative.max().item()


def _make_inputs(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    return {
        "q_in": torch.randn(BATCH, SEQUENCE, PROJECTION_SIZE, device=device, dtype=dtype),
        "k_in": torch.randn(BATCH, SEQUENCE, PROJECTION_SIZE, device=device, dtype=dtype),
        "v_in": torch.randn(BATCH, SEQUENCE, PROJECTION_SIZE, device=device, dtype=dtype),
        "q_state": torch.randn(
            BATCH, PROJECTION_SIZE, CONV_WIDTH, device=device, dtype=dtype
        ),
        "k_state": torch.randn(
            BATCH, PROJECTION_SIZE, CONV_WIDTH, device=device, dtype=dtype
        ),
        "v_state": torch.randn(
            BATCH, PROJECTION_SIZE, CONV_WIDTH, device=device, dtype=dtype
        ),
        "q_weight": torch.randn(
            PROJECTION_SIZE, 1, CONV_WIDTH, device=device, dtype=dtype
        )
        * 0.02,
        "k_weight": torch.randn(
            PROJECTION_SIZE, 1, CONV_WIDTH, device=device, dtype=dtype
        )
        * 0.02,
        "v_weight": torch.randn(
            PROJECTION_SIZE, 1, CONV_WIDTH, device=device, dtype=dtype
        )
        * 0.02,
        "raw_decay": torch.randn(
            BATCH, SEQUENCE, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype
        ),
        "dt_bias": torch.randn(PROJECTION_SIZE, device=device, dtype=torch.float32),
        "a_log": torch.log(
            torch.empty(NUM_HEADS, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
        ).view(1, 1, NUM_HEADS, 1),
        "raw_beta": torch.randn(
            BATCH, SEQUENCE, NUM_HEADS, device=device, dtype=dtype
        ),
    }


def _conv_arguments(
    inputs: dict[str, torch.Tensor],
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    return (
        inputs["q_in"],
        inputs["k_in"],
        inputs["v_in"],
        *states,
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["v_weight"],
    )


def _gate_arguments(
    inputs: dict[str, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        inputs["raw_decay"],
        inputs["dt_bias"],
        inputs["a_log"],
        inputs["raw_beta"],
        q,
        k,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    arguments = parser.parse_args()
    if arguments.iterations < 100:
        raise ValueError("iterations must be at least 100")
    if arguments.warmup < 20:
        raise ValueError("warmup must be at least 20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BENCH-KDAPREP.py")

    dtype = torch.float16 if arguments.dtype == "float16" else torch.bfloat16
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    inputs = _make_inputs(dtype)

    correctness_reference_states = tuple(
        inputs[name].clone() for name in ("q_state", "k_state", "v_state")
    )
    correctness_fused_states = tuple(state.clone() for state in correctness_reference_states)
    reference_conv = fused_short_conv_triple_reference(
        *_conv_arguments(inputs, correctness_reference_states)
    )
    fused_conv = fused_short_conv_triple(
        *_conv_arguments(inputs, correctness_fused_states)
    )
    reference_gates = fused_kda_gates_reference(
        *_gate_arguments(inputs, reference_conv[0], reference_conv[1]),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        eps=1e-6,
    )
    fused_gates = fused_kda_gates(
        *_gate_arguments(inputs, fused_conv[0], fused_conv[1]),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        eps=1e-6,
    )
    torch.cuda.synchronize()

    conv_atol = 0.002 if dtype == torch.float16 else 0.02
    conv_rtol = conv_atol
    for actual, expected in zip(fused_conv, reference_conv):
        torch.testing.assert_close(actual, expected, atol=conv_atol, rtol=conv_rtol)
    for actual, expected in zip(fused_gates, reference_gates):
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    for actual, expected in zip(correctness_fused_states, correctness_reference_states):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    reference_states = tuple(
        inputs[name].clone() for name in ("q_state", "k_state", "v_state")
    )
    fused_states = tuple(state.clone() for state in reference_states)

    def reference_chain() -> tuple[torch.Tensor, ...]:
        q_out, k_out, v_out = fused_short_conv_triple_reference(
            *_conv_arguments(inputs, reference_states)
        )
        gates = fused_kda_gates_reference(
            *_gate_arguments(inputs, q_out, k_out),
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            eps=1e-6,
        )
        return q_out, k_out, v_out, *gates

    def fused_chain() -> tuple[torch.Tensor, ...]:
        q_out, k_out, v_out = fused_short_conv_triple(
            *_conv_arguments(inputs, fused_states)
        )
        gates = fused_kda_gates(
            *_gate_arguments(inputs, q_out, k_out),
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            eps=1e-6,
        )
        return q_out, k_out, v_out, *gates

    reference_us = _time_cuda(
        reference_chain,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    fused_us = _time_cuda(
        fused_chain,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )

    reference_launches = REFERENCE_SHORT_CONV_LAUNCHES + REFERENCE_GATE_LAUNCHES
    fused_launches = FUSED_SHORT_CONV_LAUNCHES + FUSED_GATE_LAUNCHES
    saved_launches = reference_launches - fused_launches
    saved_us = reference_us - fused_us

    print("KDA decode preparation launch count")
    print(f"  per layer: {reference_launches} eager -> {fused_launches} fused")
    print(f"  per layer saved: {saved_launches}")
    print(
        f"  across {KDA_LAYERS} layers: "
        f"{reference_launches * KDA_LAYERS} eager -> "
        f"{fused_launches * KDA_LAYERS} fused"
    )
    print(f"  per token saved: {saved_launches * KDA_LAYERS}")
    print("KDA decode preparation time")
    print(f"  eager reference: {reference_us:.3f} us per layer")
    print(f"  fused kernels: {fused_us:.3f} us per layer")
    print(f"  saved: {saved_us:.3f} us per layer")
    print(f"  scaled to {KDA_LAYERS} layers: {saved_us * KDA_LAYERS / 1000.0:.3f} ms/token")
    print("Single-step maximum errors")
    names = ("q_conv", "k_conv", "v_conv", "decay", "beta", "q_norm", "k_norm")
    for name, actual, expected in zip(
        names,
        (*fused_conv, *fused_gates),
        (*reference_conv, *reference_gates),
    ):
        maximum_absolute, maximum_relative = _max_errors(actual, expected)
        print(
            f"  {name}: max_abs={maximum_absolute:.6g}, "
            f"max_rel={maximum_relative:.6g}"
        )


if __name__ == "__main__":
    main()
