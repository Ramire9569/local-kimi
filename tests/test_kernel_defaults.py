"""Pin the kernels an ordinary run actually gets.

This file exists because of a real defect. The fast kernels were registered as
variants but nothing selected them, so `registry.active` returned "reference"
unless the caller set KIMI_KERNELS. Every benchmark selected variants explicitly
and reported 113 tokens per second, while anyone who simply ran the engine got
the reference path at roughly a third of that. The published number described a
path no ordinary run took.

These tests assert the DEFAULT, not the reachable-under-configuration behaviour.
"""

from __future__ import annotations

import pytest

from engine.kernels import W4A16_DENSE, W4A16_GROUPED, W4A16_SWIGLU, registry


def test_grouped_expert_kernel_defaults_to_the_fast_gemv() -> None:
    assert registry.default_of(W4A16_GROUPED) == "triton_gemv"


def test_dense_projection_kernel_defaults_to_the_fast_gemv() -> None:
    assert registry.default_of(W4A16_DENSE) == "triton_gemv"


def test_swiglu_defaults_to_reference_because_the_fusion_measured_slower() -> None:
    # The fused gate and up kernel measured 107.64 tok/s against 109.33 for the
    # two-call path in the same container. It stays registered and switched off.
    assert registry.default_of(W4A16_SWIGLU) == "reference"
    assert "fused" in registry.variants(W4A16_SWIGLU)


def test_no_environment_variable_still_selects_the_fast_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KIMI_KERNELS", raising=False)
    assert registry.active(W4A16_GROUPED) == "triton_gemv"
    assert registry.active(W4A16_DENSE) == "triton_gemv"


def test_environment_variable_still_overrides_the_shipped_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default tier must sit BELOW KIMI_KERNELS. If it sat above, the switch
    # the whole registry exists to provide would stop working.
    monkeypatch.setenv("KIMI_KERNELS", f"{W4A16_GROUPED}=reference")
    assert registry.active(W4A16_GROUPED) == "reference"


def test_programmatic_override_still_wins_over_the_shipped_default() -> None:
    with registry.using(W4A16_GROUPED, "reference"):
        assert registry.active(W4A16_GROUPED) == "reference"
    assert registry.active(W4A16_GROUPED) == "triton_gemv"


def test_default_names_a_variant_that_exists() -> None:
    for op in (W4A16_GROUPED, W4A16_DENSE, W4A16_SWIGLU):
        assert registry.default_of(op) in registry.variants(op)
