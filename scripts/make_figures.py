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
    """Throughput after each fused kernel landed."""
    labels = [
        "Before\nfused kernels",
        "KDA\nfusions",
        "+ grouped\nW4A16 GEMV",
        "+ dense\nW4A16 GEMV",
    ]
    values = [35.76, 38.04, 61.88, 109.51]
    colors = [BASE, ACCENT, ACCENT, ACCENT]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("tokens per second")
    ax.set_ylim(0, 132)
    ax.grid(axis="y", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.annotate(
        "3.06x",
        xy=(3, 109.51),
        xytext=(1.5, 122.0),
        fontsize=9,
        color=ACCENT,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )
    _finish(
        ax,
        "Decode throughput after each fused kernel",
        "NVIDIA L40S, INT4 weights, single stream, greedy, 17-token prompt, 64 generated "
        "tokens, inside a hard 32 GiB cap. The kernels are not bit-identical to the "
        "reference: teacher forced they agree on 96.9 percent of next-token choices at "
        "0.0036 nats mean KL, about a tenth of what INT4 quantisation itself costs. "
        "Source: engine/kernels/RESULTS.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "decode-throughput.png", bbox_inches="tight")
    plt.close(fig)


def kernel_bandwidth() -> None:
    """The reason the speedup existed: both hot kernels were shaped wrong."""
    labels = [
        "grouped W4A16\nw1 and w3",
        "grouped W4A16\nw2",
        "dense W4A16\nq/k/v",
        "dense W4A16\no_proj",
    ]
    before = [8.26, 10.19, 9.38, 4.06]
    after = [51.73, 51.18, 11.13, 9.81]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    first = ax.bar([i - width / 2 for i in x], before, width, label="before", color=BASE)
    second = ax.bar([i + width / 2 for i in x], after, width, label="after", color=ACCENT)
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
    ax.set_xticklabels(labels)
    ax.set_ylabel("percent of L40S peak bandwidth")
    ax.set_ylim(0, 68)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.grid(axis="y", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "Both hot kernels were running far below the card",
        "Each was written as a GEMM and used at decode as a GEMV: tl.dot with a 16-row "
        "accumulator for a single token, and packed weights indexed with N on the fastest "
        "axis so every lane pulled its own cache line. The dense kernel remains starved for "
        "parallelism, which is why it gains less. Peak is 864 GB/s. "
        "Source: engine/kernels/RESULTS.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "kernel-bandwidth.png", bbox_inches="tight")
    plt.close(fig)


def decode_time_split() -> None:
    """Where a decode token went before the kernel work."""
    labels = [
        "grouped W4A16",
        "dense W4A16",
        "elementwise",
        "other",
        "reduction",
        "index, copy, softmax",
    ]
    values = [13.184, 9.132, 3.385, 1.838, 0.766, 0.545]
    launches = [78, 104, 2236, 360, 300, 186]
    colors = [WARN, WARN, ACCENT, BASE, BASE, BASE]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, value, count in zip(bars, values, launches, strict=True):
        ax.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ms   {count} launches",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, 20)
    ax.set_xlabel("milliseconds per decoded token")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "Two kernels held 77 percent of decode time",
        "Measured on an L40S over 28.85 ms of GPU work per token. CUDA graph capture had "
        "already removed launch overhead, so the 2,236 elementwise launches cost only 3.4 ms "
        "between them and kernel count was not the bottleneck. "
        "Source: engine/klinear/DECODE-PROFILE.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "decode-time-split.png", bbox_inches="tight")
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
    """The fused kernels gave memory back rather than spending it."""
    labels = ["Reference kernels", "Fused kernels"]
    peak = [29.56, 27.63]
    budget = 32.0

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars = ax.barh(labels, peak, color=[BASE, ACCENT], height=0.5)
    for bar, value in zip(bars, peak, strict=True):
        slack = budget - value
        note = f"{value:.2f} GiB used, {slack:.2f} GiB free"
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2, note, va="center", fontsize=8.5)
    ax.axvline(budget, color=INK, linestyle="--", linewidth=1.0)
    ax.text(budget + 0.3, -0.45, "32 GiB cap", fontsize=8)
    ax.set_xlim(0, 39)
    ax.set_xlabel("peak reserved device memory, GiB")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "The speedup gave memory back",
        "An earlier version of this figure said the opposite, because it was drawn before the "
        "fused kernels existed. The GEMV path allocates no per-call partial buffers, so peak "
        "reserved memory fell by 1.93 GiB while throughput went up 3.06x. "
        "Source: engine/kernels/RESULTS.md",
    )
    fig.tight_layout()
    fig.savefig(OUT / "memory-headroom.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    decode_stages()
    kernel_bandwidth()
    decode_time_split()
    weight_bytes()
    quantisation_quality()
    memory_headroom()
    for item in sorted(OUT.glob("*.png")):
        print(f"wrote {item.relative_to(OUT.parent.parent)}  {item.stat().st_size} bytes")
