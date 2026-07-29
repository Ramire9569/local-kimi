"""Benchmark the grouped W4A16 reference kernel against the GEMV candidates.

Run from the repository root on an L40S or another CUDA GPU:

    python engine/kernels/BENCH-GEMV.py

The bandwidth column reports effective useful traffic. It counts packed
weights, BF16 scales, activation reads per N tile, expert indices, outputs,
and deterministic split-K partial writes and reads. It does not claim to
measure memory-controller transactions or cache-line amplification.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.kernels.w4a16_gemv import (  # noqa: E402
    GEMV_CONFIGS,
    GemvConfig,
    _launch_grouped_w4a16_gemv,
    _select_config,
)
from engine.kernels.w4a16_grouped import grouped_w4a16_linear  # noqa: E402

EXPERTS = 257
TOKENS = 1
ROUTES = 9
L40S_PEAK_GBPS = 864.0
ATOL = 0.125
RTOL = 0.05


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    output_size: int
    reduction: int


REAL_SHAPES = (
    ProjectionShape("w1/w3", output_size=1024, reduction=2304),
    ProjectionShape("w2", output_size=2304, reduction=1024),
)


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    max_rel: float
    within_tolerance: bool
    argmax_unchanged: bool


@dataclass(frozen=True)
class Timing:
    name: str
    milliseconds: float
    tokens_per_second: float
    bandwidth_gbps: float
    peak_fraction: float


def _make_inputs(shape: ProjectionShape) -> tuple[torch.Tensor, ...]:
    activations = torch.randn(
        (TOKENS, shape.reduction), device="cuda", dtype=torch.bfloat16
    )
    expert_indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, EXPERTS - 1]],
        device="cuda",
        dtype=torch.long,
    )
    packed = torch.randint(
        0,
        256,
        (EXPERTS, shape.output_size, shape.reduction // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = (
        torch.rand(
            (EXPERTS, shape.output_size, shape.reduction // 32),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    return activations, expert_indices, packed, scales


def _measure(
    function: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error_stats(reference: torch.Tensor, actual: torch.Tensor) -> ErrorStats:
    reference_float = reference.float()
    actual_float = actual.float()
    absolute = (reference_float - actual_float).abs()
    relative = absolute / reference_float.abs().clamp_min(1e-6)
    tolerance = ATOL + RTOL * reference_float.abs()
    return ErrorStats(
        max_abs=float(absolute.max().item()),
        max_rel=float(relative.max().item()),
        within_tolerance=bool((absolute <= tolerance).all().item()),
        argmax_unchanged=bool(
            torch.equal(reference_float.argmax(dim=-1), actual_float.argmax(dim=-1))
        ),
    )


def _effective_bytes(
    shape: ProjectionShape,
    *,
    block_n: int,
    split_k: int,
) -> int:
    n_tiles = math.ceil(shape.output_size / block_n)
    packed_weights = ROUTES * shape.output_size * (shape.reduction // 2)
    scales = ROUTES * shape.output_size * (shape.reduction // 32) * 2
    activations = ROUTES * n_tiles * shape.reduction * 2
    expert_indices = ROUTES * n_tiles * split_k * 8
    outputs = ROUTES * shape.output_size * 2
    partials = 0
    if split_k > 1:
        partials = ROUTES * shape.output_size * split_k * 4 * 2
    return packed_weights + scales + activations + expert_indices + outputs + partials


def _timing(
    name: str,
    milliseconds: float,
    effective_bytes: int,
) -> Timing:
    seconds = milliseconds / 1000.0
    bandwidth = effective_bytes / seconds / 1e9
    return Timing(
        name=name,
        milliseconds=milliseconds,
        tokens_per_second=1.0 / seconds,
        bandwidth_gbps=bandwidth,
        peak_fraction=bandwidth / L40S_PEAK_GBPS,
    )


def _print_timing_table(rows: list[Timing]) -> None:
    print(
        f"{'kernel':<30} {'ms':>10} {'tok/s equiv':>14} "
        f"{'GB/s':>12} {'L40S peak':>12}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row.name:<30} {row.milliseconds:>10.4f} "
            f"{row.tokens_per_second:>14.2f} {row.bandwidth_gbps:>12.2f} "
            f"{row.peak_fraction * 100.0:>11.2f}%"
        )


def _benchmark_shape(shape: ProjectionShape, warmup: int, iterations: int) -> None:
    print()
    print(
        f"{shape.name}: E={EXPERTS}, tokens={TOKENS}, routes={ROUTES}, "
        f"N={shape.output_size}, K={shape.reduction}"
    )
    activations, expert_indices, packed, scales = _make_inputs(shape)
    arguments = (activations, expert_indices, packed, scales)

    reference = grouped_w4a16_linear(*arguments)
    reference_ms = _measure(
        lambda: grouped_w4a16_linear(*arguments), warmup, iterations
    )
    reference_bytes = _effective_bytes(shape, block_n=64, split_k=1)
    reference_timing = _timing("grouped_w4a16_linear", reference_ms, reference_bytes)

    candidate_rows: list[tuple[GemvConfig, Timing, ErrorStats]] = []
    for config in GEMV_CONFIGS:
        actual = _launch_grouped_w4a16_gemv(*arguments, config)
        repeated = _launch_grouped_w4a16_gemv(*arguments, config)
        if not torch.equal(actual, repeated):
            raise AssertionError(f"{config.name} is not deterministic run to run")
        errors = _error_stats(reference, actual)
        if not errors.within_tolerance:
            raise AssertionError(
                f"{config.name} exceeds atol={ATOL} and rtol={RTOL}: "
                f"max_abs={errors.max_abs}, max_rel={errors.max_rel}"
            )
        if not errors.argmax_unchanged:
            raise AssertionError(f"{config.name} changed an output-vector argmax")

        milliseconds = _measure(
            lambda config=config: _launch_grouped_w4a16_gemv(*arguments, config),
            warmup,
            iterations,
        )
        traffic = _effective_bytes(
            shape, block_n=config.block_n, split_k=config.split_k
        )
        candidate_rows.append(
            (config, _timing(config.name, milliseconds, traffic), errors)
        )

    best_config, best_timing, best_errors = min(
        candidate_rows, key=lambda row: row[1].milliseconds
    )
    default_config = _select_config(shape.output_size, TOKENS * ROUTES)

    print(
        f"tolerance: atol={ATOL}, rtol={RTOL}; "
        f"best max abs={best_errors.max_abs:.8f}, "
        f"best max rel={best_errors.max_rel:.8f}"
    )
    print(f"random output-vector argmax proxy unchanged: {best_errors.argmax_unchanged}")
    print(f"default config: {default_config.name}")
    print(f"measured best config: {best_config.name}")
    print()
    print("Candidate timings")
    _print_timing_table([row[1] for row in candidate_rows])
    print()
    print("Reference versus measured best")
    _print_timing_table([reference_timing, best_timing])

    del activations, expert_indices, packed, scales, reference
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-GEMV.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"warmup: {args.warmup}, iterations: {args.iterations}")
    print("bandwidth metric: effective useful traffic, decimal GB/s")
    for shape in REAL_SHAPES:
        _benchmark_shape(shape, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
