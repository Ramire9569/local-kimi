from __future__ import annotations

import pytest
import torch

from engine.kernels.equivalence import assert_equivalent, check_equivalence
from engine.kernels.registry import active, register, resolve, use, using


def test_duplicate_reference_raises_and_names_both_variants() -> None:
    op = "test_duplicate_reference"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(ValueError) as error:
        @register(op, "replacement", reference=True)
        def replacement(value: torch.Tensor) -> torch.Tensor:
            return value

    message = str(error.value)
    assert "baseline" in message
    assert "replacement" in message


def test_unknown_environment_variant_names_available_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = "test_unknown_environment_variant"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "fast")
    def fast(value: torch.Tensor) -> torch.Tensor:
        return value

    monkeypatch.setenv("KIMI_KERNELS", f"{op}=typo")

    with pytest.raises(ValueError) as error:
        resolve(op)

    message = str(error.value)
    assert op in message
    assert "typo" in message
    assert "baseline" in message
    assert "fast" in message


def test_unknown_environment_operation_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = "test_unknown_environment_operation"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    monkeypatch.setenv("KIMI_KERNELS", "future_operation=future_variant")
    value = torch.tensor([2.0])

    assert active(op) == "baseline"
    assert torch.equal(resolve(op)(value), torch.tensor([3.0]))


def test_using_restores_state_when_body_raises() -> None:
    op = "test_using_restores_state"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "experimental")
    def experimental(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    with pytest.raises(RuntimeError, match="body failed"):
        with using(op, "experimental"):
            assert active(op) == "experimental"
            with using(op, "baseline"):
                assert active(op) == "baseline"
            assert active(op) == "experimental"
            raise RuntimeError("body failed")

    assert active(op) == "baseline"


def test_active_reports_environment_then_programmatic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = "test_active_reports_resolved_name"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "environment_choice")
    def environment_choice(value: torch.Tensor) -> torch.Tensor:
        return value

    monkeypatch.setenv("KIMI_KERNELS", f" {op} = environment_choice ")
    assert active(op) == "environment_choice"

    use(op, "baseline")
    assert active(op) == "baseline"


def test_requires_cuda_variant_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = "test_requires_cuda_variant"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "cuda_fast", requires_cuda=True)
    def cuda_fast(value: torch.Tensor) -> torch.Tensor:
        return value

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    use(op, "cuda_fast")

    with pytest.raises(RuntimeError) as error:
        resolve(op)

    assert "cuda_fast" in str(error.value)
    assert "CUDA" in str(error.value)


def test_equivalence_reports_known_bad_variant_as_failed() -> None:
    op = "test_equivalence_known_bad_variant"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "wrong")
    def wrong(value: torch.Tensor) -> torch.Tensor:
        return value * 1.01

    def make_inputs(device: torch.device) -> tuple[torch.Tensor]:
        return (torch.tensor([1.0, -4.0, 100.0], device=device),)

    reports = check_equivalence(
        op,
        make_inputs,
        rtol=1e-6,
        atol=1e-6,
        device="cpu",
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.variant == "wrong"
    assert not report.passed
    assert not report.exact_match
    assert not report.skipped
    assert report.max_abs_err > 0
    assert report.max_rel_err > 0

    with pytest.raises(AssertionError) as error:
        assert_equivalent(
            op,
            make_inputs,
            rtol=1e-6,
            atol=1e-6,
            device="cpu",
        )
    assert "wrong | FAIL" in str(error.value)


def test_equivalence_supports_tuple_outputs() -> None:
    op = "test_equivalence_tuple_outputs"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value, value.square()

    @register(op, "same")
    def same(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.clone(), value.square()

    reports = check_equivalence(
        op,
        lambda: (torch.tensor([2.0, -3.0]),),
        rtol=0.0,
        atol=0.0,
        device="cpu",
    )

    assert len(reports) == 1
    assert reports[0].passed
    assert reports[0].exact_match


def test_equivalence_skips_cuda_variant_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = "test_equivalence_skips_cuda"

    @register(op, "baseline", reference=True)
    def baseline(value: torch.Tensor) -> torch.Tensor:
        return value

    @register(op, "cuda_fast", requires_cuda=True)
    def cuda_fast(value: torch.Tensor) -> torch.Tensor:
        raise AssertionError("a skipped CUDA variant must not execute")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reports = check_equivalence(
        op,
        lambda: (torch.ones(2),),
        rtol=0.0,
        atol=0.0,
        device="cpu",
    )

    assert len(reports) == 1
    assert reports[0].variant == "cuda_fast"
    assert reports[0].skipped
    assert not reports[0].passed
    assert "CUDA" in reports[0].skip_reason
