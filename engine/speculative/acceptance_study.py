"""How often prompt lookup would predict correctly, measured on real text.

Speculative decoding is only worth its complexity if the draft is accepted often
enough. Acceptance depends entirely on how repetitive the text is, so the honest
way to size the win is to measure it on the kind of text the model will actually
produce rather than to quote a number from a paper.

This runs the draft source over real token sequences and counts how many of its
proposals match what actually came next. That is an UPPER BOUND on acceptance
against a live model, because it assumes the model would emit exactly the text
being replayed. Treat it as sizing, not as a result.

    uv run python engine/speculative/acceptance_study.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from engine.speculative.draft import propose

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Corpus:
    name: str
    text: str
    note: str


def _read(relative: str, limit: int = 24_000) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="replace")[:limit]


def corpora() -> list[Corpus]:
    return [
        Corpus(
            "python source",
            _read("engine/kernels/w4a16_gemv.py"),
            "the workload a coding agent actually produces",
        ),
        Corpus(
            "markdown prose",
            _read("engine/kernels/RESULTS.md"),
            "technical writing, moderately repetitive",
        ),
        Corpus(
            "json-like config",
            _read("pyproject.toml") + _read("CHANGELOG.md", 8000),
            "highly structured, the best case",
        ),
    ]


def _tokenize(text: str) -> list[int]:
    """A byte-level stand-in for the real tokenizer.

    The real tokenizer is a 163,840-entry TikToken vocabulary that is not
    available without the checkpoint. Byte tokens fragment text MORE than BPE
    does, which makes any given n-gram match rarer and shorter, so this
    understates acceptance rather than flattering it.
    """
    return list(text.encode("utf-8"))


def study(k: int, ngram: int) -> dict[str, float]:
    rows: dict[str, float] = {}
    for corpus in corpora():
        tokens = _tokenize(corpus.text)
        accepted_lengths: list[int] = []
        # Walk the sequence, proposing at each position and counting how many
        # proposed tokens match the real continuation before the first miss.
        for position in range(ngram + 64, len(tokens) - k, 7):
            context = tokens[:position]
            actual = tokens[position : position + k]
            proposal = propose(context, k, ngram)
            matched = 0
            for drafted, truth in zip(proposal, actual):
                if drafted != truth:
                    break
                matched += 1
            accepted_lengths.append(matched)
        rows[corpus.name] = statistics.fmean(accepted_lengths) if accepted_lengths else 0.0
    return rows


def report() -> None:
    print("Prompt lookup acceptance, byte tokens, upper bound")
    print()
    header = f"{'corpus':<22}"
    settings = [(4, 3), (4, 4), (8, 3), (8, 4)]
    for k, ngram in settings:
        header += f"{'k=' + str(k) + ' n=' + str(ngram):>12}"
    print(header)
    print("-" * (22 + 12 * len(settings)))

    tables = {setting: study(*setting) for setting in settings}
    names = [corpus.name for corpus in corpora()]
    for name in names:
        line = f"{name:<22}"
        for setting in settings:
            line += f"{tables[setting][name]:>12.2f}"
        print(line)
    print()
    print("Mean tokens accepted per round. A round always emits at least one")
    print("token, so the wall-clock speedup is roughly (1 + accepted) divided by")
    print("the cost of one verification pass relative to one decode step.")
    print()
    best = max(
        (value, setting, name)
        for setting, table in tables.items()
        for name, value in table.items()
    )
    value, setting, name = best
    print(f"Best case here: {value:.2f} tokens accepted on {name} at k={setting[0]}, n={setting[1]}.")
    print(f"That is about {1 + value:.2f} tokens per verification pass.")
    print()
    print("Caveats that matter more than the numbers:")
    print("  - Byte tokens fragment more than the real BPE vocabulary, so real")
    print("    acceptance on the same text should be higher than this.")
    print("  - This assumes the model reproduces the replayed text exactly. A")
    print("    live model diverges, so real acceptance is lower than this bound.")
    print("  - Those two errors push in opposite directions and do not cancel in")
    print("    any principled way. This sizes the idea; it does not measure it.")


if __name__ == "__main__":
    report()
