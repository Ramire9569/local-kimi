"""Generate the figures in docs/figures from measured numbers.

Every value is copied from a results document in this repository and each figure
names its source. Nothing is estimated or interpolated.

Typography is Computer Modern through matplotlib's mathtext, which matches a
LaTeX document without requiring a LaTeX installation. If a real toolchain is
present, set USE_TEX=1 to render through it instead.

    uv run python scripts/make_figures.py

Each figure is written twice, as PNG for the README and as PDF for inclusion in
a paper, since the PDF keeps the text as vectors.
"""

from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# A real LaTeX run needs latex, dvipng and ghostscript on PATH. Asking for it
# without them raises at draw time, well after the code looks correct, so the
# check is explicit rather than a try/except wrapped around savefig.
USE_TEX = bool(os.environ.get("USE_TEX")) and all(
    shutil.which(binary) for binary in ("latex", "dvipng", "gs")
)

INK = "#111111"
MUTED = "#6b6b6b"
RULE = "#d8d8d8"
ACCENT = "#1f4e79"
ACCENT_LIGHT = "#7fa6c9"
BASE = "#bfbfbf"
WARN = "#a33b21"

plt.rcParams.update(
    {
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        # cmr10 has no U+2212, so a real minus renders as a missing-glyph box.
        "axes.unicode_minus": False,
        # Required alongside cmr10: without it matplotlib warns and falls back
        # to a sans-serif tick formatter, which mixes two typefaces on one axis.
        "axes.formatter.use_mathtext": True,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

PERCENT_LABEL = r"\%" if USE_TEX else "%"


def _caption(ax, label: str, title: str, body: str, *, width: int = 104) -> None:
    """Journal-style caption: a figure label, a title, then the note.

    The note is wrapped before it is drawn. Left as one long line it widens the
    saved bounding box, and with a tight bbox that squashes the axes into a
    fraction of the canvas and collides the tick labels.
    """
    lines = textwrap.wrap(body, width=width)
    # The caption grows upward from the top of the axes, so the title has to
    # clear its full height. Pad is in points, and a line occupies
    # fontsize * linespacing points. Under-estimating this draws the title
    # straight through the first line of the caption.
    body_size, spacing = 7.6, 1.45
    ax.set_title(
        f"{label}  {title}",
        loc="left",
        fontsize=10,
        pad=14 + body_size * spacing * len(lines),
    )
    ax.text(
        0.0,
        1.012,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=body_size,
        color=MUTED,
        va="bottom",
        linespacing=spacing,
    )


def _save(fig, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}")
    plt.close(fig)


def _grid_y(ax, top: float) -> None:
    ax.set_ylim(0, top)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="minor", bottom=False)


def _grid_x(ax) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="x", color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False)


def client_path() -> None:
    """How a coding agent reaches the local engine.

    Every label here is checked against the source. The client presets come from
    k3/presets.py, the three protocol dialects from k3/dialects/, and the tool
    call parsers from k3/toolcalls.py. Nothing on this diagram is aspirational.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill, edge, title_size=8.6):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.6,rounding_size=1.4",
                linewidth=0.8, edgecolor=edge, facecolor=fill, zorder=3,
            )
        )
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top",
                fontsize=title_size, zorder=4)
        for index, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 8.6 - index * 4.3, line, ha="center",
                    va="top", fontsize=7.0, color=MUTED, zorder=4)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=8,
                linewidth=0.8, color=INK, zorder=2,
                shrinkA=1, shrinkB=1,
            )
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.3, label, ha="center",
                    fontsize=6.4, color=MUTED, zorder=4)

    # Clients, each with the wire protocol it actually speaks.
    clients = [
        ("Claude Code", "Anthropic Messages", 44),
        ("Codex", "OpenAI Responses", 26),
        ("Aider, Cline,\nopencode", "OpenAI Chat", 6),
    ]
    for name, protocol, y in clients:
        box(1, y, 22, 14, name, [protocol], "#f4f6f8", MUTED)
        arrow(23.4, y + 7, 33.6, 31)

    box(34, 15, 30, 32, "k3", [
        "detects the client per request",
        "translates all three dialects",
        "tool calls: hermes, json, kimi,",
        "kimi-k3, pythonic, passthrough",
        "restores reasoning across turns",
    ], "#eaf0f6", ACCENT, title_size=10)

    arrow(64.4, 31, 74.6, 31)

    box(75, 15, 24, 32, "local engine", [
        "Kimi-Linear-48B",
        "selective INT4, 26.8 GiB",
        "fused decode kernels",
        "113.83 tok/s on an L40S",
        "OpenAI Chat upstream",
    ], "#eef4ef", "#2f6f4e", title_size=10)

    _caption(
        ax,
        "Figure 0.",
        "One local model, every coding agent, no client-side changes.",
        "Agents disagree about the wire protocol. Claude Code speaks Anthropic Messages, Codex "
        "speaks OpenAI Responses, and most others speak OpenAI Chat Completions. k3 detects "
        "which one is calling from the route, headers, user agent and body shape, then "
        "translates the request and the streamed response, including tool calls in six "
        "formats and reasoning content that would otherwise be dropped between turns. The "
        "engine behind it serves an ordinary OpenAI Chat endpoint, so any backend can take "
        "its place. Verified against k3/presets.py, k3/dialects/ and k3/toolcalls.py.",
    )
    _save(fig, "client-path")


def decode_throughput() -> None:
    """Throughput after each fused kernel landed."""
    labels = [
        "before fused\nkernels",
        "KDA\nfusions",
        "$+$ grouped\nW4A16 GEMV",
        "$+$ dense\nW4A16 GEMV",
    ]
    values = [35.76, 37.98, 63.10, 113.83]
    colors = [BASE, ACCENT_LIGHT, ACCENT, ACCENT]

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    bars = ax.bar(labels, values, color=colors, width=0.58, zorder=3)
    # Only the final configuration was measured in more than one container, so
    # it is the only bar with an interval. Drawing zero-length bars on the rest
    # renders as stray dashes that read as a measurement rather than as nothing.
    ax.errorbar(
        [3],
        [113.83],
        yerr=[[113.83 - 113.77], [114.03 - 113.83]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.7,
        capsize=2.5,
        capthick=0.7,
        zorder=4,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3.2,
            f"{value:.2f}",
            ha="center",
            fontsize=8.5,
            zorder=5,
        )
    ax.set_ylabel(r"decode throughput  /  tokens $\mathrm{s}^{-1}$")
    _grid_y(ax, 132)
    # Aim at the flank of the bar, not its top. Pointing at the top collides
    # with the value label sitting directly above it.
    ax.annotate(
        r"$3.18\times$",
        xy=(2.72, 100.0),
        xytext=(1.55, 120.0),
        fontsize=9.5,
        color=ACCENT,
        ha="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ACCENT,
            "lw": 0.8,
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )
    _caption(
        ax,
        "Figure 1.",
        "Decode throughput after each fused kernel.",
        "NVIDIA L40S, selective INT4 weights, single stream, greedy, 17-token prompt, 64 "
        "generated tokens, inside a hard 32 GiB process cap. Bars are the median of five "
        "repeats in one container; the interval on the final bar is the range across five "
        "independent containers, 113.77 to 114.03. The kernels are not bit-identical to the "
        "reference path: teacher forced they agree on 96.9 percent of next-token choices at "
        "0.0036 nats mean KL, roughly one tenth of the divergence INT4 quantisation itself "
        "introduces. Source: engine/kernels/RESULTS.md",
    )
    _save(fig, "decode-throughput")


def kernel_bandwidth() -> None:
    """Achieved bandwidth against the roofline, before and after."""
    labels = [
        "grouped\n$w_1, w_3$",
        "grouped\n$w_2$",
        "dense\n$q, k, v$",
        "dense\n$o$",
    ]
    before = [8.26, 10.19, 9.38, 4.06]
    after = [51.73, 51.18, 11.13, 9.81]

    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    first = ax.bar(
        [i - width / 2 for i in x], before, width,
        label="before", color=BASE, zorder=3,
    )
    second = ax.bar(
        [i + width / 2 for i in x], after, width,
        label="after", color=ACCENT, zorder=3,
    )
    for group in (first, second):
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.6,
                f"{bar.get_height():.1f}",
                ha="center",
                fontsize=7.8,
                zorder=5,
            )
    ax.axhline(100, color=WARN, linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.text(
        len(labels) - 0.55,
        92,
        r"roofline, $864\ \mathrm{GB\,s^{-1}}$",
        fontsize=7.6,
        color=WARN,
        ha="right",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"achieved bandwidth  /  {PERCENT_LABEL} of peak")
    _grid_y(ax, 112)
    ax.legend(loc="upper left", bbox_to_anchor=(0.006, 0.88))
    _caption(
        ax,
        "Figure 2.",
        "Both hot kernels ran far below the memory roofline.",
        "Each had been written as a matrix-matrix product and was being used at decode as a "
        "matrix-vector product: tl.dot with a 16-row accumulator to multiply a single token, "
        "and packed weights indexed with the output dimension on the fastest-varying axis, so "
        "neighbouring lanes in a warp addressed separate cache lines. The dense kernel gains "
        "less because a single 5.31 MB matrix yields only 64 thread blocks against 142 "
        "streaming multiprocessors, leaving it starved for parallelism rather than limited by "
        "memory. Source: engine/kernels/RESULTS.md",
    )
    _save(fig, "kernel-bandwidth")


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

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    bars = ax.barh(labels, values, color=colors, height=0.6, zorder=3)
    for bar, value, count in zip(bars, values, launches, strict=True):
        ax.text(
            value + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ms" + r"$\quad$" + f"{count} launches",
            va="center",
            fontsize=7.8,
            zorder=5,
        )
    ax.set_xlim(0, 20.5)
    ax.set_xlabel("time per decoded token  /  ms")
    ax.invert_yaxis()
    _grid_x(ax)
    _caption(
        ax,
        "Figure 3.",
        "Two kernels held 77 percent of decode time.",
        "Measured over 28.85 ms of GPU work per decoded token on an L40S. CUDA graph capture "
        "had already removed launch overhead, so the 2,236 elementwise launches cost only "
        "3.4 ms between them and kernel count was not the bottleneck. This measurement is "
        "what redirected the work from reducing launches to reshaping two kernels. "
        "Source: engine/klinear/DECODE-PROFILE.md",
    )
    _save(fig, "decode-time-split")


def weight_bytes() -> None:
    """What the quantisation removed."""
    labels = ["BF16 as shipped", "selective INT4"]
    values = [98.245528576, 28.803304448]

    fig, ax = plt.subplots(figsize=(6.6, 2.1))
    bars = ax.barh(labels, values, color=[BASE, ACCENT], height=0.46, zorder=3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 1.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} GB",
            va="center",
            fontsize=8.5,
            zorder=5,
        )
    ax.set_xlim(0, 118)
    ax.set_xlabel("weight storage  /  GB")
    ax.invert_yaxis()
    ax.axvline(34.36, color=WARN, linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.text(35.8, -0.42, "32 GiB card", color=WARN, fontsize=7.6)
    _grid_x(ax)
    _caption(
        ax,
        "Figure 4.",
        r"Weight storage, $3.41\times$ smaller.",
        "Built and verified from the real checkpoint on one NVIDIA H100. Planned and actual "
        "byte totals agree exactly. Symmetric signed INT4 with group size 32 on the reduction "
        "axis and BF16 per-group scales, which is 4.5 bits per parameter once the scales are "
        "counted. Source: engine/quant/QUANTIZATION-RESULTS.md",
    )
    _save(fig, "weight-bytes")


def quantisation_quality() -> None:
    """What the quantisation cost, measured against BF16."""
    metrics = [
        "next-token\ntop-1 agreement",
        "greedy output\nidentity",
        "router set\nagreement",
    ]
    default = [85.16, 37.69, 34.30]
    retained = [91.41, 40.77, 36.74]

    x = range(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    first = ax.bar(
        [i - width / 2 for i in x], default, width,
        label="default INT4", color=BASE, zorder=3,
    )
    second = ax.bar(
        [i + width / 2 for i in x], retained, width,
        label="shared experts kept BF16", color=ACCENT, zorder=3,
    )
    for group in (first, second):
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.6,
                f"{bar.get_height():.1f}",
                ha="center",
                fontsize=7.8,
                zorder=5,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel(f"agreement with BF16  /  {PERCENT_LABEL}")
    _grid_y(ax, 116)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    _caption(
        ax,
        "Figure 5.",
        "What INT4 costs, measured against the BF16 checkpoint.",
        "Both sides served by the same vLLM build on one H200 with identical prompts, so only "
        "the weights differ. Keeping the shared experts in BF16 improves every agreement "
        "metric shown yet makes perplexity worse by 1.95 percent, which is why it was not "
        "adopted. These quantisation costs are an order of magnitude larger than the kernel "
        "differences quoted in Figure 1. Source: engine/accuracy/RESULTS.md",
    )
    _save(fig, "quantisation-quality")


def memory_headroom() -> None:
    """The speedup gave memory back rather than spending it."""
    labels = ["reference kernels", "fused kernels"]
    peak = [29.56, 27.63]
    budget = 32.0

    fig, ax = plt.subplots(figsize=(6.6, 2.1))
    bars = ax.barh(labels, peak, color=[BASE, ACCENT], height=0.46, zorder=3)
    for bar, value in zip(bars, peak, strict=True):
        ax.text(
            value + 0.32,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} GiB used, {budget - value:.2f} GiB free",
            va="center",
            fontsize=8,
            zorder=5,
        )
    ax.axvline(budget, color=WARN, linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.text(budget + 0.32, -0.42, "32 GiB cap", fontsize=7.6, color=WARN)
    ax.set_xlim(0, 39)
    ax.set_xlabel("peak reserved device memory  /  GiB")
    ax.invert_yaxis()
    _grid_x(ax)
    _caption(
        ax,
        "Figure 6.",
        "The speedup gave memory back.",
        "The GEMV path allocates no per-call partial reduction buffers, so peak reserved "
        "memory fell by 1.93 GiB while throughput rose 3.18 times. An earlier version of this "
        "figure claimed the opposite, because it was drawn from the pre-fusion engine where "
        "preallocation and graph capture had bought speed with memory. "
        "Source: engine/kernels/RESULTS.md",
    )
    _save(fig, "memory-headroom")


if __name__ == "__main__":
    print(
        "LaTeX rendering: "
        + ("on" if USE_TEX else "off, using Computer Modern mathtext")
    )
    client_path()
    decode_throughput()
    kernel_bandwidth()
    decode_time_split()
    weight_bytes()
    quantisation_quality()
    memory_headroom()
    for item in sorted(OUT.glob("*")):
        if item.suffix in {".png", ".pdf"}:
            print(f"wrote {item.relative_to(OUT.parent.parent)}  {item.stat().st_size} bytes")
