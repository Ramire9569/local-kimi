"""Per-kernel decode profile for the INT4 Kimi-Linear engine.

Answers one question: where do the milliseconds of a decode token actually go.

CUDA graph replay appears to the profiler as a single opaque launch, so the
per-kernel breakdown is taken from EAGER decode steps and the headline
throughput is taken from graph replay. Both are reported, along with the gap
between them, which is the launch overhead the graph removes.

    modal run engine/modal_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-decode-profile")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"

BUDGET_BYTES = 34_359_738_368  # 32 GiB, the consumer card budget

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch>=2.5",
        "triton>=3.1",
        "safetensors>=0.4.5",
        "transformers>=4.48",
        "numpy>=2.0",
        "tiktoken>=0.9",
        "blobfile>=3.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


# Kernel name fragments mapped to the part of the model they belong to. Order
# matters: the first match wins, so put the specific fragments first.
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("w4a16_expert_gemv", ("grouped_w4a16",)),
    ("elementwise", ("elementwise", "vectorized_elementwise", "unrolled_elementwise")),
    ("reduction", ("reduce", "Reduce", "sum_functor", "norm")),
    ("gemm", ("gemm", "GemmSm", "cutlass", "sgemm", "ampere", "nvjet", "s16816")),
    ("copy", ("copy", "Copy", "direct_copy", "index_copy", "cat_")),
    ("softmax", ("softmax", "Softmax")),
    ("index", ("index", "Index", "gather", "scatter")),
    ("fill", ("fill", "Fill", "zero")),
)


def classify(name: str) -> str:
    for label, fragments in CATEGORIES:
        for fragment in fragments:
            if fragment in name:
                return label
    return "other"


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=16.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def profile_decode(
    prompt_tokens: int = 8,
    profile_steps: int = 24,
    timed_steps: int = 64,
    cap_memory: bool = True,
) -> dict:
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from engine.klinear.generate import CUDAGraphDecodeRunner, decode, prefill
    from engine.klinear.model import KLinearModel

    if not torch.cuda.is_available():
        raise RuntimeError("the decode profile requires CUDA")

    device = torch.device("cuda:0")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if cap_memory:
        if BUDGET_BYTES > total_bytes:
            raise RuntimeError(
                f"card has {total_bytes} bytes, below the {BUDGET_BYTES} byte budget"
            )
        torch.cuda.set_per_process_memory_fraction(BUDGET_BYTES / total_bytes, device)

    load_started = time.perf_counter()
    model = KLinearModel.from_directory(MODEL_DIR, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    # A fixed synthetic prompt keeps the profile comparable between runs. Token
    # 1000 is an ordinary vocabulary entry, not a control token.
    input_ids = torch.full(
        (1, prompt_tokens), 1000, dtype=torch.long, device=device
    )

    output = prefill(model, input_ids)
    torch.cuda.synchronize(device)

    # --- eager decode, per kernel -------------------------------------------
    # Capacity must cover EVERY eager step below. The warmup, the timed loop and
    # the profiled loop all advance the same static MLA cache position, so
    # reserving one loop's worth walks the index past the end. That fails as a
    # device-side assert surfaced by a later synchronize, which points the
    # traceback at the profiler instead of at the real cause.
    eager_steps = 4 + profile_steps + profile_steps
    eager_state = output.state.reserve_decode_capacity(eager_steps + 16)
    token = output.logits[:, -1].argmax(dim=-1).unsqueeze(1)

    for _ in range(4):  # warm up Triton JIT and cuBLAS handles
        decode(model, token, eager_state)
    torch.cuda.synchronize(device)

    eager_started = time.perf_counter()
    for _ in range(profile_steps):
        decode(model, token, eager_state)
    torch.cuda.synchronize(device)
    eager_seconds = time.perf_counter() - eager_started

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(profile_steps):
            decode(model, token, eager_state)
        torch.cuda.synchronize(device)

    rows = []
    for event in prof.key_averages():
        cuda_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if cuda_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "category": classify(event.key),
                "self_cuda_us_total": cuda_us,
                "self_cuda_us_per_token": cuda_us / profile_steps,
                "count": int(event.count),
                "calls_per_token": event.count / profile_steps,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_us_total"], reverse=True)
    total_cuda_us = sum(row["self_cuda_us_total"] for row in rows)

    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(
            row["category"],
            {"self_cuda_us_total": 0.0, "count": 0, "distinct_kernels": 0},
        )
        bucket["self_cuda_us_total"] += row["self_cuda_us_total"]
        bucket["count"] += row["count"]
        bucket["distinct_kernels"] += 1
    for name, bucket in by_category.items():
        bucket["us_per_token"] = bucket["self_cuda_us_total"] / profile_steps
        bucket["percent"] = (
            100.0 * bucket["self_cuda_us_total"] / total_cuda_us if total_cuda_us else 0.0
        )
        bucket["calls_per_token"] = bucket["count"] / profile_steps

    # --- CUDA graph replay, the headline number -----------------------------
    graph_tokens_per_second = None
    graph_ms_per_token = None
    graph_error = None
    try:
        # Re-prefill so the graph starts from a state the eager loop never
        # touched. reserve_decode_capacity may share storage with the state it
        # was derived from, and a graph captured over a mutated cache would
        # measure something other than a clean decode.
        graph_prefill = prefill(model, input_ids)
        torch.cuda.synchronize(device)
        runner = CUDAGraphDecodeRunner(model, input_ids, graph_prefill, timed_steps)
        runner.reset()
        for _ in range(8):  # warm the replay path
            runner.graph.replay()
        torch.cuda.synchronize(device)

        runner.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_steps):
            runner.graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        graph_ms = start.elapsed_time(end)
        graph_ms_per_token = graph_ms / timed_steps
        graph_tokens_per_second = 1000.0 * timed_steps / graph_ms
    except RuntimeError as error:  # capture can fail on some driver versions
        graph_error = str(error)

    eager_ms_per_token = 1000.0 * eager_seconds / profile_steps

    result = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": total_bytes,
        "memory_capped_to_bytes": BUDGET_BYTES if cap_memory else None,
        "load_seconds": load_seconds,
        "resident_weight_bytes": model.resident_weight_bytes,
        "prompt_tokens": prompt_tokens,
        "profile_steps": profile_steps,
        "timed_steps": timed_steps,
        "eager_ms_per_token": eager_ms_per_token,
        "eager_tokens_per_second": 1000.0 / eager_ms_per_token,
        "graph_ms_per_token": graph_ms_per_token,
        "graph_tokens_per_second": graph_tokens_per_second,
        "graph_error": graph_error,
        "kernel_cuda_us_per_token": total_cuda_us / profile_steps,
        "launch_overhead_ms_per_token": (
            eager_ms_per_token - (total_cuda_us / profile_steps) / 1000.0
        ),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "by_category": by_category,
        "top_kernels": rows[:40],
        "distinct_kernels": len(rows),
        "total_launches_per_token": sum(row["calls_per_token"] for row in rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    prompt_tokens: int = 8,
    profile_steps: int = 24,
    timed_steps: int = 64,
) -> None:
    result = profile_decode.remote(
        prompt_tokens=prompt_tokens,
        profile_steps=profile_steps,
        timed_steps=timed_steps,
    )
    print("\n" + "=" * 78)
    print(f"GPU {result['gpu']}")
    print(
        f"eager {result['eager_ms_per_token']:.2f} ms/token "
        f"({result['eager_tokens_per_second']:.2f} tok/s)"
    )
    if result["graph_tokens_per_second"]:
        print(
            f"graph {result['graph_ms_per_token']:.2f} ms/token "
            f"({result['graph_tokens_per_second']:.2f} tok/s)"
        )
    else:
        print(f"graph FAILED: {result['graph_error']}")
    print(
        f"kernel time {result['kernel_cuda_us_per_token'] / 1000.0:.2f} ms/token, "
        f"{result['total_launches_per_token']:.0f} launches/token"
    )
    print("=" * 78)
    print(f"{'category':<22} {'ms/token':>10} {'percent':>9} {'launches':>10}")
    for name, bucket in sorted(
        result["by_category"].items(),
        key=lambda item: item[1]["self_cuda_us_total"],
        reverse=True,
    ):
        print(
            f"{name:<22} {bucket['us_per_token'] / 1000.0:>10.3f} "
            f"{bucket['percent']:>8.1f}% {bucket['calls_per_token']:>10.1f}"
        )
    print("=" * 78)
    print(f"{'kernel':<52} {'ms/tok':>9} {'calls':>7}")
    for row in result["top_kernels"][:20]:
        print(
            f"{row['name'][:52]:<52} {row['self_cuda_us_per_token'] / 1000.0:>9.3f} "
            f"{row['calls_per_token']:>7.1f}"
        )
