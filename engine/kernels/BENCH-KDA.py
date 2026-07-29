"""Benchmark the fused KDA decode recurrence at the production shape."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.kernels.kda_step import (  # noqa: E402
    KDA_BLOCK_V,
    _kda_decode_step_with_block_v,
    kda_decode_step_reference,
)


BATCH = 1
HEADS = 32
D_K = 128
D_V = 128
WARMUP = 20
ITERATIONS = 200
RECURRENT_STEPS = 256
BLOCK_VS = (16, 32, 64)
RELATIVE_DENOMINATOR_FLOOR = 1e-6


def _normalized_random(shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.randn(shape, device="cuda", dtype=torch.float32)
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / norm.clamp_min(1e-6)


def _make_step_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260729)
    state = 0.05 * torch.randn(
        BATCH,
        HEADS,
        D_K,
        D_V,
        device="cuda",
        dtype=torch.float32,
    )
    q = _normalized_random((BATCH, HEADS, D_K))
    k = _normalized_random((BATCH, HEADS, D_K))
    v = 0.2 * torch.randn(
        BATCH,
        HEADS,
        D_V,
        device="cuda",
        dtype=torch.float32,
    )
    decay = -(0.005 + 0.075 * torch.rand(
        BATCH,
        HEADS,
        D_K,
        device="cuda",
        dtype=torch.float32,
    ))
    beta = torch.sigmoid(
        torch.randn(BATCH, HEADS, device="cuda", dtype=torch.float32)
    )
    return state, q, k, v, decay, beta


def _make_recurrent_inputs() -> tuple[torch.Tensor, ...]:
    state = torch.zeros(
        BATCH,
        HEADS,
        D_K,
        D_V,
        device="cuda",
        dtype=torch.float32,
    )
    q = _normalized_random((RECURRENT_STEPS, BATCH, HEADS, D_K))
    k = _normalized_random((RECURRENT_STEPS, BATCH, HEADS, D_K))
    v = 0.2 * torch.randn(
        RECURRENT_STEPS,
        BATCH,
        HEADS,
        D_V,
        device="cuda",
        dtype=torch.float32,
    )
    decay = -(0.005 + 0.075 * torch.rand(
        RECURRENT_STEPS,
        BATCH,
        HEADS,
        D_K,
        device="cuda",
        dtype=torch.float32,
    ))
    beta = torch.sigmoid(
        torch.randn(
            RECURRENT_STEPS,
            BATCH,
            HEADS,
            device="cuda",
            dtype=torch.float32,
        )
    )
    return state, q, k, v, decay, beta


def _max_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(RELATIVE_DENOMINATOR_FLOOR)
    return absolute.max().item(), relative.max().item()


def _time_step(step, initial_state: torch.Tensor) -> float:
    state = initial_state.clone()
    for _ in range(WARMUP):
        step(state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        step(state)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / ITERATIONS


def _fused_minimum_bytes() -> int:
    element_bytes = torch.tensor([], dtype=torch.float32).element_size()
    state_elements = BATCH * HEADS * D_K * D_V
    key_elements = BATCH * HEADS * D_K
    value_elements = BATCH * HEADS * D_V
    beta_elements = BATCH * HEADS
    total_elements = (
        2 * state_elements
        + 3 * key_elements
        + 2 * value_elements
        + beta_elements
    )
    return total_elements * element_bytes


def _reference_estimated_bytes() -> int:
    """Estimate eager traffic, including materialized intermediates and state copy."""
    element_bytes = torch.tensor([], dtype=torch.float32).element_size()
    state_elements = BATCH * HEADS * D_K * D_V
    key_elements = BATCH * HEADS * D_K
    value_elements = BATCH * HEADS * D_V
    beta_elements = BATCH * HEADS

    # The coefficients count eager reads and writes in the exact reference chain.
    total_elements = (
        14 * state_elements
        + 10 * key_elements
        + 6 * value_elements
        + beta_elements
    )
    return total_elements * element_bytes


def _gb_per_second(byte_count: int, milliseconds: float) -> float:
    return byte_count / (milliseconds * 1e-3) / 1e9


def _run_reference_recurrence(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state.clone()
    output = torch.empty(0, device="cuda")
    for token_index in range(RECURRENT_STEPS):
        output = kda_decode_step_reference(
            state,
            q[token_index],
            k[token_index],
            v[token_index],
            decay[token_index],
            beta[token_index],
            scale=scale,
        )
    return state, output


def _run_fused_recurrence(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state.clone()
    output = torch.empty(0, device="cuda")
    for token_index in range(RECURRENT_STEPS):
        output = _kda_decode_step_with_block_v(
            state,
            q[token_index],
            k[token_index],
            v[token_index],
            decay[token_index],
            beta[token_index],
            scale=scale,
            block_v=block_v,
        )
    return state, output


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-KDA.py")

    scale = D_K**-0.5
    state, q, k, v, decay, beta = _make_step_inputs()

    reference_state = state.clone()
    reference_output = kda_decode_step_reference(
        reference_state,
        q,
        k,
        v,
        decay,
        beta,
        scale=scale,
    )

    def reference_step(current_state: torch.Tensor) -> torch.Tensor:
        return kda_decode_step_reference(
            current_state,
            q,
            k,
            v,
            decay,
            beta,
            scale=scale,
        )

    reference_ms = _time_step(reference_step, state)
    reference_bytes = _reference_estimated_bytes()
    fused_bytes = _fused_minimum_bytes()

    recurrent = _make_recurrent_inputs()
    recurrent_reference_state, recurrent_reference_output = (
        _run_reference_recurrence(*recurrent, scale=scale)
    )

    print("KDA decode benchmark")
    print(
        f"shape: batch={BATCH}, heads={HEADS}, d_k={D_K}, d_v={D_V}, "
        "dtype=float32"
    )
    print(f"warmup={WARMUP}, iterations={ITERATIONS}")
    print(
        "relative error denominator floor: "
        f"{RELATIVE_DENOMINATOR_FLOOR:.1e}"
    )
    print(
        f"reference: {reference_ms:.6f} ms, "
        f"estimated traffic={reference_bytes / 2**20:.3f} MiB, "
        f"estimated achieved bandwidth={_gb_per_second(reference_bytes, reference_ms):.2f} GB/s"
    )

    timings: dict[int, float] = {}
    for block_v in BLOCK_VS:
        actual_state = state.clone()
        actual_output = _kda_decode_step_with_block_v(
            actual_state,
            q,
            k,
            v,
            decay,
            beta,
            scale=scale,
            block_v=block_v,
        )
        state_abs, state_rel = _max_errors(actual_state, reference_state)
        output_abs, output_rel = _max_errors(actual_output, reference_output)

        def fused_step(
            current_state: torch.Tensor,
            selected_block_v: int = block_v,
        ) -> torch.Tensor:
            return _kda_decode_step_with_block_v(
                current_state,
                q,
                k,
                v,
                decay,
                beta,
                scale=scale,
                block_v=selected_block_v,
            )

        fused_ms = _time_step(fused_step, state)
        timings[block_v] = fused_ms
        recurrent_state, recurrent_output = _run_fused_recurrence(
            *recurrent,
            scale=scale,
            block_v=block_v,
        )
        recurrent_state_abs, recurrent_state_rel = _max_errors(
            recurrent_state,
            recurrent_reference_state,
        )
        recurrent_output_abs, recurrent_output_rel = _max_errors(
            recurrent_output,
            recurrent_reference_output,
        )

        print()
        print(f"BLOCK_V={block_v}")
        print(
            f"  fused: {fused_ms:.6f} ms, "
            f"minimum traffic={fused_bytes / 2**20:.3f} MiB, "
            f"achieved bandwidth={_gb_per_second(fused_bytes, fused_ms):.2f} GB/s"
        )
        print(f"  speedup: {reference_ms / fused_ms:.3f}x")
        print(
            f"  single-step state error: max_abs={state_abs:.8e}, "
            f"max_rel={state_rel:.8e}"
        )
        print(
            f"  single-step output error: max_abs={output_abs:.8e}, "
            f"max_rel={output_rel:.8e}"
        )
        print(
            f"  {RECURRENT_STEPS}-step state error: "
            f"max_abs={recurrent_state_abs:.8e}, max_rel={recurrent_state_rel:.8e}"
        )
        print(
            f"  {RECURRENT_STEPS}-step output error: "
            f"max_abs={recurrent_output_abs:.8e}, max_rel={recurrent_output_rel:.8e}"
        )
        print(
            f"  20 layers: reference={20 * reference_ms:.6f} ms/token, "
            f"fused={20 * fused_ms:.6f} ms/token, "
            f"saving={20 * (reference_ms - fused_ms):.6f} ms/token"
        )

    winner = min(timings, key=timings.get)
    print()
    print(f"measured winner: BLOCK_V={winner}")
    print(f"production API selection: BLOCK_V={KDA_BLOCK_V}")


if __name__ == "__main__":
    main()
