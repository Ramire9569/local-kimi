"""Registration and selection for interchangeable kernel implementations."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, cast

import torch

KernelFunction = Callable[..., object]
_Function = TypeVar("_Function", bound=KernelFunction)


@dataclass(frozen=True)
class KernelVariant:
    """One named implementation of a kernel operation."""

    name: str
    fn: KernelFunction
    is_reference: bool
    requires_cuda: bool
    description: str


_REGISTRY: dict[str, dict[str, KernelVariant]] = {}
_OVERRIDES: dict[str, str] = {}
_DEFAULTS: dict[str, str] = {}
_MISSING = object()


def register(
    op: str,
    variant: str,
    *,
    reference: bool = False,
    requires_cuda: bool = False,
    description: str = "",
) -> Callable[[_Function], _Function]:
    """Register a function as a named implementation of an operation."""

    def decorator(fn: _Function) -> _Function:
        registered = _REGISTRY.setdefault(op, {})
        if reference:
            existing_reference = next(
                (item for item in registered.values() if item.is_reference), None
            )
            if existing_reference is not None:
                raise ValueError(
                    f"operation {op!r} already has reference variant "
                    f"{existing_reference.name!r}; cannot register second reference "
                    f"variant {variant!r}"
                )
        if variant in registered:
            raise ValueError(
                f"variant {variant!r} is already registered for operation {op!r}"
            )
        registered[variant] = KernelVariant(
            name=variant,
            fn=fn,
            is_reference=reference,
            requires_cuda=requires_cuda,
            description=description,
        )
        return fn

    return decorator


def reference_of(op: str) -> KernelVariant:
    """Return the reference implementation registered for an operation."""
    for variant in _REGISTRY.get(op, {}).values():
        if variant.is_reference:
            return variant
    raise KeyError(f"operation {op!r} has no reference kernel variant")


def variants(op: str) -> tuple[str, ...]:
    """Return registered names with the reference first, then sorted variants."""
    registered = _REGISTRY.get(op, {})
    reference_names = sorted(
        item.name for item in registered.values() if item.is_reference
    )
    other_names = sorted(
        item.name for item in registered.values() if not item.is_reference
    )
    return tuple(reference_names + other_names)


def _variant_of(op: str, variant: str) -> KernelVariant:
    registered = _REGISTRY.get(op)
    if registered is None:
        raise KeyError(f"operation {op!r} has no registered kernel variants")
    try:
        return registered[variant]
    except KeyError as error:
        available = ", ".join(variants(op)) or "none"
        raise ValueError(
            f"unknown variant {variant!r} for operation {op!r}; "
            f"available variants: {available}"
        ) from error


def _environment_overrides() -> dict[str, str]:
    selected: dict[str, str] = {}
    raw_value = os.getenv("KIMI_KERNELS", "")
    for raw_entry in raw_value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        raw_op, separator, raw_variant = entry.partition("=")
        op = raw_op.strip()
        variant = raw_variant.strip()
        if not separator or not op:
            raise ValueError(
                f"invalid KIMI_KERNELS entry {entry!r}; expected op=variant"
            )
        if op not in _REGISTRY:
            continue
        if not variant:
            available = ", ".join(variants(op)) or "none"
            raise ValueError(
                f"unknown variant {variant!r} for operation {op!r}; "
                f"available variants: {available}"
            )
        _variant_of(op, variant)
        selected[op] = variant
    return selected


def set_default(op: str, variant: str) -> None:
    """Choose the variant used when nothing else selects one.

    This is the SHIPPED choice, and it sits below KIMI_KERNELS so a user can
    still override it. Without this tier the only fallback was the reference
    implementation, which meant the fast kernels were off unless the caller set
    an environment variable. The benchmarks set variants explicitly, so they
    measured a path an ordinary run never took.
    """
    _variant_of(op, variant)
    _DEFAULTS[op] = variant


def default_of(op: str) -> str:
    """Return the shipped default variant name for an operation."""
    return _DEFAULTS.get(op, reference_of(op).name)


def _selected_variant(op: str) -> KernelVariant:
    if op not in _REGISTRY:
        raise KeyError(f"operation {op!r} has no registered kernel variants")
    override = _OVERRIDES.get(op)
    if override is not None:
        return _variant_of(op, override)
    environment_variant = _environment_overrides().get(op)
    if environment_variant is not None:
        return _variant_of(op, environment_variant)
    shipped = _DEFAULTS.get(op)
    if shipped is not None:
        return _variant_of(op, shipped)
    return reference_of(op)


def resolve(op: str) -> KernelFunction:
    """Return the selected function and reject unavailable CUDA variants."""
    selected = _selected_variant(op)
    if selected.requires_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"kernel variant {selected.name!r} for operation {op!r} requires CUDA, "
            "but torch.cuda.is_available() is False"
        )
    return selected.fn


def use(op: str, variant: str) -> None:
    """Set a programmatic variant override for an operation."""
    _variant_of(op, variant)
    _OVERRIDES[op] = variant


def clear_override(op: str) -> None:
    """Drop any programmatic override so selection falls back to the default."""
    _OVERRIDES.pop(op, None)


@contextmanager
def using(op: str, variant: str) -> Iterator[None]:
    """Temporarily override an operation and restore its previous selection."""
    previous = _OVERRIDES.get(op, _MISSING)
    use(op, variant)
    try:
        yield
    finally:
        if previous is _MISSING:
            _OVERRIDES.pop(op, None)
        else:
            _OVERRIDES[op] = cast(str, previous)


def active(op: str) -> str:
    """Return the name selected by override, environment, or reference order."""
    return _selected_variant(op).name
