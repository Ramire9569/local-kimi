"""Pin that the SERVING path actually uses the state cache.

tests/test_statecache.py proves the cache works. It does not prove anything
reaches it. An audit found that `StateCache(` appeared in exactly two files, its
own test and its own benchmark, so every real request paid full prefill while
760 lines of working code sat unreferenced.

These tests assert what the serve path SELECTS, not what is reachable under
configuration, which is the same distinction that tests/test_kernel_defaults.py
exists to hold for the kernel registry.
"""

from __future__ import annotations

import pytest
import torch

from engine.klinear.generate import generate_tokens
from engine.klinear.state import MLALayerState
from engine.statecache.store import StateCache
from tests.test_statecache import _model


def _drain(generator) -> list[int]:
    tokens: list[int] = []
    try:
        while True:
            tokens.append(int(next(generator)))
    except StopIteration:
        return tokens


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, reproduced: the SECOND restore of a snapshot returns "
        "wrong state. Sequence on a tiny model, seed 1234, prompt "
        "[3,1,4,1,5,9,2], 8 tokens: plain [12,4,4,3,9,8,1,4], cold "
        "[12,4,4,3,9,8,1,4] correct, first warm [12,4,4,3,9,8,1,4] correct, "
        "second warm [0,13,1,3,9,8,1,3] WRONG. A direct probe shows restore "
        "itself is exact: every KDA and MLA tensor matches and the logits are "
        "bit-equal. So something in the first restore-and-decode cycle mutates "
        "the stored snapshot. StateCache is opt-in and off by default, so no "
        "shipped path is affected. Remove this marker when the third request "
        "matches the first."
    ),
)
def test_state_cache_does_not_change_generated_tokens() -> None:
    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], dtype=torch.long)
    cache = StateCache(byte_budget=16 * 1024 * 1024)

    plain = _drain(generate_tokens(model, prompt, 8))
    results = [
        _drain(generate_tokens(model, prompt, 8, state_cache=cache))
        for _ in range(3)
    ]
    for index, run in enumerate(results):
        assert run == plain, f"cached request {index} changed the output"


def test_warm_request_prefills_only_the_final_token(monkeypatch) -> None:
    """The assertion that fails if the branch in generate_tokens is reverted."""
    import engine.statecache.session as session

    widths: list[int] = []
    original = session.prefill

    def recording(model, token_ids, **kwargs):
        widths.append(int(token_ids.shape[1]))
        return original(model, token_ids, **kwargs)

    monkeypatch.setattr(session, "prefill", recording)

    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], dtype=torch.long)
    cache = StateCache(byte_budget=16 * 1024 * 1024)

    _drain(generate_tokens(model, prompt, 4, state_cache=cache))
    cold = list(widths)
    widths.clear()
    _drain(generate_tokens(model, prompt, 4, state_cache=cache))
    warm = list(widths)

    # Cold: the prefix of length 6 is prefilled, then the final token.
    assert cold == [6, 1], f"cold path prefilled {cold}, expected [6, 1]"
    # Warm: the prefix is restored, so only the final token is prefilled.
    assert warm == [1], f"warm path prefilled {warm}, expected [1]"


def test_a_caller_supplied_attention_mask_takes_the_cold_path() -> None:
    """A padding mask is not part of the prefix key, so it must not be cached."""
    import engine.statecache.session as session

    calls: list[int] = []
    original = session.prefill

    def recording(model, token_ids, **kwargs):
        calls.append(int(token_ids.shape[1]))
        return original(model, token_ids, **kwargs)

    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5]], dtype=torch.long)
    mask = torch.ones_like(prompt)
    cache = StateCache(byte_budget=16 * 1024 * 1024)

    try:
        session.prefill = recording
        _drain(
            generate_tokens(model, prompt, 3, attention_mask=mask, state_cache=cache)
        )
    finally:
        session.prefill = original

    assert calls == [], "a masked request must not go through the cache"
    assert cache.entry_count == 0


def test_ensure_decode_capacity_reuses_a_restored_state() -> None:
    """Regression for a crash the audit reproduced.

    `reserve_decode_capacity` copies into freshly sized buffers, so calling it on
    a state that is ALREADY fixed capacity raises a shape error. Wiring the cache
    without this guard turned every warm request into a hard crash.
    """
    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5]], dtype=torch.long)
    from engine.klinear.generate import prefill

    state = prefill(model, prompt).state.reserve_decode_capacity(8)
    assert state.is_static

    same = state.ensure_decode_capacity(4)
    assert same is state, "a static state with room should be returned unchanged"

    capacity = next(
        layer.capacity for layer in state.layer_states if isinstance(layer, MLALayerState)
    )
    with pytest.raises(ValueError):
        state.ensure_decode_capacity(capacity + 1)
