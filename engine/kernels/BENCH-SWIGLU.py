"""Benchmark fused grouped W4A16 SwiGLU at the Kimi K3 decode shape."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from engine.kernels.moe_swiglu import BLOCK_N, GROUP_SIZE, fused_swiglu_w4a16
from engine.kernels.w4a16_grouped import grouped_w4a16_linear

TOKENS = 1
ROUTES = 9
EXPERTS = 257
N = 1024
K = 2304
MOE_LAYERS = 26
L40S_PEAK_GBPS = 864.0


def _baseline(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
) -> torch.Tensor:
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
    return F.silu(gate) * up


def _measure_ms(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    difference = (candidate.float() - reference.float()).abs()
    relative = difference / reference.float().abs().clamp_min(1.0e-6)
    max_abs, max_rel = torch.stack((difference.max(), relative.max())).cpu().tolist()
    return max_abs, max_rel


def _logical_byte_counts() -> tuple[int, int]:
    """Return a minimum logical byte model for baseline and fused paths."""
    assignments = TOKENS * ROUTES
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    programs = assignments * n_blocks

    packed_weights = 2 * assignments * N * (K // 2)
    scales = 2 * assignments * N * (K // GROUP_SIZE) * 2
    activation_per_projection = programs * K * 2
    expert_indices_per_projection = programs * 8
    output = assignments * N * 2

    fused = (
        packed_weights
        + scales
        + activation_per_projection
        + expert_indices_per_projection
        + output
    )
    baseline = (
        packed_weights
        + scales
        + 2 * activation_per_projection
        + 2 * expert_indices_per_projection
        + 7 * output
    )
    return baseline, fused


def _gbps(byte_count: int, milliseconds: float) -> float:
    return byte_count / (milliseconds * 1.0e6)


def _make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260729)
    activations = torch.empty(
        (TOKENS, K),
        device=device,
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.5)
    expert_indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, EXPERTS - 1]],
        device=device,
        dtype=torch.long,
    )
    packed_shape = (EXPERTS, N, K // 2)
    scale_shape = (EXPERTS, N, K // GROUP_SIZE)
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
    ).uniform_(0.004, 0.02)
    w3_scales = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(0.004, 0.02)
    return (
        activations,
        expert_indices,
        w1_packed,
        w1_scales,
        w3_packed,
        w3_scales,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda")
    inputs = _make_inputs(device)

    def baseline() -> torch.Tensor:
        return _baseline(*inputs)

    def fused_default() -> torch.Tensor:
        return fused_swiglu_w4a16(*inputs, hp_activation=False)

    def fused_high_precision() -> torch.Tensor:
        return fused_swiglu_w4a16(*inputs, hp_activation=True)

    baseline_output = baseline()
    default_output = fused_default()
    high_precision_output = fused_high_precision()
    torch.cuda.synchronize()

    default_abs, default_rel = _error_stats(baseline_output, default_output)
    hp_abs, hp_rel = _error_stats(baseline_output, high_precision_output)

    baseline_ms = _measure_ms(
        baseline,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    default_ms = _measure_ms(
        fused_default,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    hp_ms = _measure_ms(
        fused_high_precision,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    baseline_bytes, fused_bytes = _logical_byte_counts()
    baseline_gbps = _gbps(baseline_bytes, baseline_ms)
    default_gbps = _gbps(fused_bytes, default_ms)
    hp_gbps = _gbps(fused_bytes, hp_ms)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Shape: tokens={TOKENS}, routes={ROUTES}, experts={EXPERTS}, "
        f"N={N}, K={K}"
    )
    print(f"Warmup: {args.warmup}, iterations: {args.iterations}")
    print("Relative error denominator is clamped to 1e-6.")
    print()
    print("Accuracy against the current BF16-first path")
    print(f"  fused default max abs: {default_abs:.8g}")
    print(f"  fused default max rel: {default_rel:.8g}")
    print(f"  fused hp max abs:      {hp_abs:.8g}")
    print(f"  fused hp max rel:      {hp_rel:.8g}")
    print()
    print("Mean GPU time")
    print(f"  baseline:      {baseline_ms * 1000.0:.3f} us")
    print(f"  fused default: {default_ms * 1000.0:.3f} us")
    print(f"  fused hp:      {hp_ms * 1000.0:.3f} us")
    print(f"  default speedup: {baseline_ms / default_ms:.3f}x")
    print(f"  hp speedup:      {baseline_ms / hp_ms:.3f}x")
    print()
    print("Minimum logical bandwidth")
    print(
        f"  baseline:      {baseline_gbps:.1f} GB/s, "
        f"{100.0 * baseline_gbps / L40S_PEAK_GBPS:.1f}% of L40S peak"
    )
    print(
        f"  fused default: {default_gbps:.1f} GB/s, "
        f"{100.0 * default_gbps / L40S_PEAK_GBPS:.1f}% of L40S peak"
    )
    print(
        f"  fused hp:      {hp_gbps:.1f} GB/s, "
        f"{100.0 * hp_gbps / L40S_PEAK_GBPS:.1f}% of L40S peak"
    )
    print()
    default_layer_saving_us = (baseline_ms - default_ms) * 1000.0
    hp_layer_saving_us = (baseline_ms - hp_ms) * 1000.0
    print(f"Projected over {MOE_LAYERS} MoE layers for one decoded token")
    print(
        f"  default saving: {default_layer_saving_us * MOE_LAYERS:.3f} us "
        f"({default_layer_saving_us:.3f} us per layer)"
    )
    print(
        f"  hp saving:      {hp_layer_saving_us * MOE_LAYERS:.3f} us "
        f"({hp_layer_saving_us:.3f} us per layer)"
    )


if __name__ == "__main__":
    main()
