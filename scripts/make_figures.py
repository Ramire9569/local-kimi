"""Generate the figures in docs/figures from measured numbers.

Every value here is copied from a results document in this repository and the
source is named in the caption of each figure. Nothing is estimated.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b1b1b"
MUTED = "#8a8a8a"
ACCENT = "#2f6f4e"
BASE = "#c9c9c9"
WARN = "#b3541e"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _finish(ax, title: str, subtitle: str, *, width: int = 96) -> None:
    """Title plus a wrapped subtitle.

    The subtitle is wrapped before it is drawn. Left as one long line it widens
    the saved bounding box, and with bbox_inches tight that squashes the axes
    into a fraction of the canvas and collides the tick labels.
    """
    wrapped = "\n".join(textwrap.wrap(subtitle, width=width))
    lines = wrapped.count("\n") + 1
    ax.set_title(title, loc="left", pad=14 + 9 * lines)
    ax.text(
        0.0,
        1.015,
        wrapped,
        transform=ax.transAxes,
        fontsize=8,
        color=MUTED,
        va="bottom",
        linespacing=1.35,
    )


def decode_stages() -> None:
    """Throughput after each optimisation, all with identical output."""
    labels = [
        "Reference\ngrowing state",
        "Preallocated\nstate",
        "Streaming\nend to end",
        "CUDA graph\nreplay",
    ]
    values = [9.02, 24.07, 32.33, 35.71]
    colors = [BASE, ACCENT, ACCENT, ACCENT]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("tokens per second")
    ax.set_ylim(0, 42)
    ax.grid(axis="y", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.annotate(
        "3.96x, byte identical token ids",
        xy=(3, 35.71),
        xytext=(1.35, 39.4),
        fontsize=8.5,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )
    _finish(
        ax,
        "Decode throughput after each optimisation",
        "NVIDIA L40S, INT4 weights, single stream, greedy, 64 tokens, inside a hard 32 GiB cap. "
        "Source: engine/klinear/DECODE-BENCHMARK.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "decode-throughput.png", bbox_inches="tight")
    plt.close(fig)


def weight_bytes() -> None:
    """What the quantisation actually removed."""
    labels = ["BF16 as shipped", "Selective INT4"]
    values = [98.245528576, 28.803304448]

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars = ax.barh(labels, values, color=[BASE, ACCENT], height=0.5)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 1.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} GB",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(0, 118)
    ax.set_xlabel("weight bytes, GB")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(34.36, color=WARN, linestyle="--", linewidth=1.0)
    ax.text(35.6, -0.42, "32 GiB card", color=WARN, fontsize=8)
    _finish(
        ax,
        "Weight bytes, 3.41x smaller",
        "Built and verified on an H100. Planned and actual byte totals agree exactly. "
        "Source: engine/quant/QUANTIZATION-RESULTS.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "weight-bytes.png", bbox_inches="tight")
    plt.close(fig)


def quantisation_quality() -> None:
    """What the quantisation cost, both profiles, measured against BF16."""
    metrics = ["Next token\ntop 1 agreement", "Greedy output\nidentity", "Router set\nagreement"]
    default = [85.16, 37.69, 34.30]
    retained = [91.41, 40.77, 36.74]

    x = range(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    first = ax.bar([i - width / 2 for i in x], default, width, label="default INT4", color=BASE)
    second = ax.bar(
        [i + width / 2 for i in x], retained, width, label="shared experts kept BF16", color=ACCENT
    )
    for group in (first, second):
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=8.5,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("agreement with BF16, percent")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "What INT4 costs, measured against BF16",
        "Both sides served by the same vLLM on one H200, identical prompts, only the weights differ. "
        "Perplexity rises 0.81 percent on the default profile. "
        "Source: engine/accuracy/RESULTS.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "quantisation-quality.png", bbox_inches="tight")
    plt.close(fig)


def memory_headroom() -> None:
    """The speedup was paid for in memory."""
    labels = ["Before optimisation", "After optimisation"]
    peak = [30.515658752, 34.349252608]
    budget = 34.359738368

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars = ax.barh(labels, peak, color=[BASE, WARN], height=0.5)
    for bar, value in zip(bars, peak, strict=True):
        slack = budget - value
        note = f"{value:.2f} GB used, {slack:.2f} GB free"
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, note, va="center", fontsize=8.5)
    ax.axvline(budget, color=INK, linestyle="--", linewidth=1.0)
    ax.text(budget + 0.35, -0.45, "32 GiB cap", fontsize=8)
    ax.set_xlim(0, 42)
    ax.set_xlabel("peak reserved device memory, GB")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "The speedup was paid for in memory",
        "Preallocated buffers, contiguous expert banks and graph capture leave 10.5 MB of slack "
        "where there was 3.58 GiB. Source: engine/klinear/DECODE-BENCHMARK.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "memory-headroom.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    decode_stages()
    weight_bytes()
    quantisation_quality()
    memory_headroom()
    for item in sorted(OUT.glob("*.png")):
        print(f"wrote {item.relative_to(OUT.parent.parent)}  {item.stat().st_size} bytes")
