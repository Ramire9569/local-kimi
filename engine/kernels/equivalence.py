"""Numerical equivalence checks for registered kernel variants."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import Signature, signature
from typing import cast

import torch

from engine.kernels import registry


@dataclass(frozen=True)
class EquivalenceReport:
    """The comparison result for one kernel variant."""

    op: str
    variant: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    exact_match: bool
    skipped: bool
    skip_reason: str


def _factory_signature(factory: Callable[..., object]) -> Signature | None:
    try:
        return signature(factory)
    except (TypeError, ValueError):
        return None


def _make_inputs(
    factory: Callable[..., object], device: torch.device
) -> tuple[object, ...]:
    factory_signature = _factory_signature(factory)
    if factory_signature is None:
        produced = factory(device)
    else:
        try:
            factory_signature.bind(device)
        except TypeError:
            try:
                factory_signature.bind(device=device)
            except TypeError:
                produced = factory()
            else:
                produced = factory(device=device)
        else:
            produced = factory(device)

    inputs = produced if isinstance(produced, tuple) else (produced,)
    return tuple(_move_to_device(value, device) for value in inputs)


def _move_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _clone_inputs(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item) for key, item in value.items()}
    return value


def _tensor_outputs(output: object) -> tuple[torch.Tensor, ...]:
    outputs = output if isinstance(output, tuple) else (output,)
    if not all(isinstance(item, torch.Tensor) for item in outputs):
        raise TypeError("kernel output must be a tensor or a tuple of tensors")
    return outputs


def _comparison_dtype(actual: torch.Tensor, expected: torch.Tensor) -> torch.dtype:
    if actual.is_complex() or expected.is_complex():
        return torch.complex128
    return torch.float64


def _compare_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, float, float]:
    if actual.shape != expected.shape:
        return False, False, math.inf, math.inf

    exact_match = (
        actual.dtype == expected.dtype
        and actual.device == expected.device
        and torch.equal(actual, expected)
    )
    comparison_dtype = _comparison_dtype(actual, expected)
    actual_value = actual.detach().to(device="cpu", dtype=comparison_dtype)
    expected_value = expected.detach().to(device="cpu", dtype=comparison_dtype)
    difference = (actual_value - expected_value).abs()
    if difference.numel() == 0:
        return True, exact_match, 0.0, 0.0

    tolerance = atol + rtol * expected_value.abs()
    passed = bool(torch.all(difference <= tolerance).item())
    max_abs_err = float(difference.max().item())
    relative_error = torch.where(
        expected_value.abs() == 0,
        torch.where(
            difference == 0,
            torch.zeros_like(difference),
            torch.full_like(difference, math.inf),
        ),
        difference / expected_value.abs(),
    )
    max_rel_err = float(relative_error.max().item())
    return passed, exact_match, max_abs_err, max_rel_err


def _compare_outputs(
    actual: object,
    expected: object,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, float, float]:
    actual_outputs = _tensor_outputs(actual)
    expected_outputs = _tensor_outputs(expected)
    if len(actual_outputs) != len(expected_outputs):
        return False, False, math.inf, math.inf

    passed = True
    exact_match = True
    max_abs_err = 0.0
    max_rel_err = 0.0
    for actual_tensor, expected_tensor in zip(actual_outputs, expected_outputs):
        tensor_passed, tensor_exact, tensor_abs, tensor_rel = _compare_tensor(
            actual_tensor,
            expected_tensor,
            rtol=rtol,
            atol=atol,
        )
        passed = passed and tensor_passed
        exact_match = exact_match and tensor_exact
        max_abs_err = max(max_abs_err, tensor_abs)
        max_rel_err = max(max_rel_err, tensor_rel)
    return passed, exact_match, max_abs_err, max_rel_err


def check_equivalence(
    op: str,
    make_inputs: Callable[..., object],
    *,
    rtol: float,
    atol: float,
    variants: Sequence[str] | None = None,
    device: str | torch.device | None = None,
) -> list[EquivalenceReport]:
    """Compare registered variants with one reference execution."""
    target_device = torch.device(
        device
        if device is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    inputs = _make_inputs(make_inputs, target_device)
    reference = registry.reference_of(op)
    reference_inputs = cast(tuple[object, ...], _clone_inputs(inputs))
    expected = reference.fn(*reference_inputs)

    selected_names = (
        tuple(variants)
        if variants is not None
        else tuple(name for name in registry.variants(op) if name != reference.name)
    )
    reports: list[EquivalenceReport] = []
    for variant_name in selected_names:
        candidate = registry._variant_of(op, variant_name)
        if candidate.requires_cuda and not torch.cuda.is_available():
            reports.append(
                EquivalenceReport(
                    op=op,
                    variant=variant_name,
                    passed=False,
                    max_abs_err=math.nan,
                    max_rel_err=math.nan,
                    exact_match=False,
                    skipped=True,
                    skip_reason="CUDA is not available",
                )
            )
            continue
        if candidate.requires_cuda and target_device.type != "cuda":
            reports.append(
                EquivalenceReport(
                    op=op,
                    variant=variant_name,
                    passed=False,
                    max_abs_err=math.nan,
                    max_rel_err=math.nan,
                    exact_match=False,
                    skipped=True,
                    skip_reason=f"variant requires CUDA but device is {target_device.type}",
                )
            )
            continue

        candidate_inputs = cast(tuple[object, ...], _clone_inputs(inputs))
        actual = candidate.fn(*candidate_inputs)
        passed, exact_match, max_abs_err, max_rel_err = _compare_outputs(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
        )
        reports.append(
            EquivalenceReport(
                op=op,
                variant=variant_name,
                passed=passed,
                max_abs_err=max_abs_err,
                max_rel_err=max_rel_err,
                exact_match=exact_match,
                skipped=False,
                skip_reason="",
            )
        )
    return reports


def _format_error(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def assert_equivalent(
    op: str,
    make_inputs: Callable[..., object],
    *,
    rtol: float,
    atol: float,
    variants: Sequence[str] | None = None,
    device: str | torch.device | None = None,
) -> list[EquivalenceReport]:
    """Run equivalence checks and raise with a table if any variant fails."""
    reports = check_equivalence(
        op,
        make_inputs,
        rtol=rtol,
        atol=atol,
        variants=variants,
        device=device,
    )
    failures = [report for report in reports if not report.skipped and not report.passed]
    if failures:
        lines = [
            f"kernel equivalence failed for operation {op!r}",
            "variant | status | max_abs_err | max_rel_err | exact_match | reason",
            "------- | ------ | ----------- | ----------- | ----------- | ------",
        ]
        for report in reports:
            status = "SKIP" if report.skipped else "PASS" if report.passed else "FAIL"
            lines.append(
                f"{report.variant} | {status} | {_format_error(report.max_abs_err)} | "
                f"{_format_error(report.max_rel_err)} | {report.exact_match} | "
                f"{report.skip_reason}"
            )
        raise AssertionError("\n".join(lines))
    return reports
