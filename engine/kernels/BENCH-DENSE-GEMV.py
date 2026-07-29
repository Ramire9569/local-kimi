"""Benchmark dense W4A16 decode GEMV candidates at Kimi model shapes.

Run from the repository root on an L40S or another CUDA GPU:

    python engine/kernels/BENCH-DENSE-GEMV.py

The harness rotates through more than three L40S L2 caches of weights. This
keeps the small projections from benchmarking as repeated L2-resident data.
Bandwidth is useful packed weights, scales, activation, and output bytes per
call. Split-K partial traffic is overhead and is intentionally not credited.
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

from engine.kernels.w4a16_dense_gemv import (  # noqa: E402
    DENSE_GEMV_CONFIGS,
    DenseGemvConfig,
    _launch_w4a16_dense_gemv,
    _select_dense_config,
)
from engine.quant.triton_w4a16 import w4a16_linear  # noqa: E402
from engine.quant.w4a16 import W4A16Tensor  # noqa: E402

M = 1
L40S_PEAK_GBPS = 864.0
L40S_L2_BYTES = 96 * 1024 * 1024
CACHE_WORKING_SET_BYTES = 3 * L40S_L2_BYTES
ATOL = 0.125
RTOL = 0.05


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    output_size: int
    reduction: int


REAL_SHAPES = (
    ProjectionShape("KDA q/k/v", output_size=4096, reduction=2304),
    ProjectionShape("o_proj", output_size=2304, reduction=4096),
    ProjectionShape("MLA q_proj", output_size=6144, reduction=2304),
    ProjectionShape("lm_head", output_size=163840, reduction=2304),
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
    calls_per_second: float
    bandwidth_gbps: float
    peak_fraction: float


def _matrix_storage_bytes(shape: ProjectionShape) -> int:
    packed = shape.output_size * (shape.reduction // 2)
    scales = shape.output_size * (shape.reduction // 32) * 2
    return packed + scales


def _useful_bytes(shape: ProjectionShape) -> int:
    activations = M * shape.reduction * 2
    outputs = M * shape.output_size * 2
    return _matrix_storage_bytes(shape) + activations + outputs


def _bank_count(shape: ProjectionShape) -> int:
    return max(1, math.ceil(CACHE_WORKING_SET_BYTES / _matrix_storage_bytes(shape)))


def _make_inputs(
    shape: ProjectionShape,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[W4A16Tensor, ...],
]:
    count = _bank_count(shape)
    activations = torch.randn(
        (M, shape.reduction), device="cuda", dtype=torch.bfloat16
    )
    packed = torch.randint(
        0,
        256,
        (count, shape.output_size, shape.reduction // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = (
        torch.rand(
            (count, shape.output_size, shape.reduction // 32),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    encoded = tuple(
        W4A16Tensor(
            packed=packed[index],
            scales=scales[index],
            original_shape=(shape.output_size, shape.reduction),
        )
        for index in range(count)
    )
    return activations, packed, scales, encoded


def _measure(
    function: Callable[[int], torch.Tensor],
    bank_count: int,
    warmup: int,
    iterations: int,
) -> float:
    warmup_launches = max(warmup, bank_count)
    for index in range(warmup_launches):
        function(index % bank_count)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        function(index % bank_count)
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


def _timing(name: str, milliseconds: float, useful_bytes: int) -> Timing:
    seconds = milliseconds / 1000.0
    bandwidth = useful_bytes / seconds / 1e9
    return Timing(
        name=name,
        milliseconds=milliseconds,
        calls_per_second=1.0 / seconds,
        bandwidth_gbps=bandwidth,
        peak_fraction=bandwidth / L40S_PEAK_GBPS,
    )


def _print_timing_table(rows: list[Timing]) -> None:
    print(
        f"{'kernel':<30} {'ms':>10} {'calls/s':>14} "
        f"{'GB/s':>12} {'L40S peak':>12}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row.name:<30} {row.milliseconds:>10.4f} "
            f"{row.calls_per_second:>14.2f} {row.bandwidth_gbps:>12.2f} "
            f"{row.peak_fraction * 100.0:>11.2f}%"
        )


def _benchmark_shape(shape: ProjectionShape, warmup: int, iterations: int) -> None:
    print()
    count = _bank_count(shape)
    matrix_megabytes = _matrix_storage_bytes(shape) / 1e6
    working_set_megabytes = count * matrix_megabytes
    print(
        f"{shape.name}: M={M}, N={shape.output_size}, K={shape.reduction}, "
        f"matrix={matrix_megabytes:.2f} MB"
    )
    print(
        f"cold-weight bank: {count} matrices, "
        f"{working_set_megabytes:.2f} MB total"
    )

    activations, packed, scales, encoded = _make_inputs(shape)
    useful_bytes = _useful_bytes(shape)
    reference = w4a16_linear(activations, encoded[0])
    reference_ms = _measure(
        lambda index: w4a16_linear(activations, encoded[index]),
        count,
        warmup,
        iterations,
    )
    reference_timing = _timing(
        "w4a16_linear", reference_ms, useful_bytes
    )

    candidate_rows: list[tuple[DenseGemvConfig, Timing, ErrorStats]] = []
    for config in DENSE_GEMV_CONFIGS:
        actual = _launch_w4a16_dense_gemv(
            activations, packed[0], scales[0], config
        )
        repeated = _launch_w4a16_dense_gemv(
            activations, packed[0], scales[0], config
        )
        if not torch.equal(actual, repeated):
            raise AssertionError(f"{config.name} is not deterministic run to run")
        errors = _error_stats(reference, actual)
        if not errors.within_tolerance:
            raise AssertionError(
                f"{config.name} exceeds atol={ATOL} and rtol={RTOL}: "
                f"max_abs={errors.max_abs}, max_rel={errors.max_rel}"
            )
        if not errors.argmax_unchanged:
            raise AssertionError(f"{config.name} changed the output argmax")

        milliseconds = _measure(
            lambda index, config=config: _launch_w4a16_dense_gemv(
                activations, packed[index], scales[index], config
            ),
            count,
            warmup,
            iterations,
        )
        candidate_rows.append(
            (config, _timing(config.name, milliseconds, useful_bytes), errors)
        )

    best_config, best_timing, best_errors = min(
        candidate_rows, key=lambda row: row[1].milliseconds
    )
    default_config = _select_dense_config(shape.output_size)

    print(
        f"tolerance: atol={ATOL}, rtol={RTOL}; "
        f"best max abs={best_errors.max_abs:.8f}, "
        f"best max rel={best_errors.max_rel:.8f}"
    )
    if shape.name == "lm_head":
        print(f"synthetic logit argmax unchanged: {best_errors.argmax_unchanged}")
    else:
        print(f"output-vector argmax unchanged: {best_errors.argmax_unchanged}")
    print(f"default config: {default_config.name}")
    print(f"measured best config: {best_config.name}")
    print()
    print("Candidate timings")
    _print_timing_table([row[1] for row in candidate_rows])
    print()
    print("Reference versus measured best")
    _print_timing_table([reference_timing, best_timing])

    del activations, packed, scales, encoded, reference
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
        raise SystemExit("CUDA is required for BENCH-DENSE-GEMV.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"warmup: {args.warmup}, iterations: {args.iterations}")
    print("bandwidth metric: cold useful traffic, decimal GB/s")
    for shape in REAL_SHAPES:
        _benchmark_shape(shape, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
