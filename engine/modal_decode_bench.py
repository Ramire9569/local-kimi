"""Decode throughput per kernel variant, with output equivalence as the gate.

Loads the INT4 checkpoint once and measures every registered variant of the
grouped expert kernel against the same prompt, comparing generated token ids
with torch.equal. A faster engine that answers differently is a different
product, so a variant that changes the output is reported as FAILED no matter
what it does to throughput.

    modal run engine/modal_decode_bench.py
    modal run engine/modal_decode_bench.py --variants reference,triton_gemv
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-decode-bench")

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


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=16.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def bench_variants(
    variants: list[str],
    prompt: str = "Explain in two sentences why quantizing a large language model to four bits reduces memory.",
    new_tokens: int = 64,
    repeats: int = 3,
    cap_memory: bool = True,
) -> dict:
    import time

    import torch

    from engine.kernels import W4A16_DENSE, W4A16_GROUPED, W4A16_SWIGLU, registry
    from engine.klinear.generate import CUDAGraphDecodeRunner, prefill
    from engine.klinear.model import KLinearModel
    from engine.klinear.moe import KLinearMoE
    from engine.klinear.quantized import W4A16Linear

    if not torch.cuda.is_available():
        raise RuntimeError("the decode benchmark requires CUDA")

    device = torch.device("cuda:0")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if cap_memory:
        torch.cuda.set_per_process_memory_fraction(BUDGET_BYTES / total_bytes, device)

    load_started = time.perf_counter()
    model = KLinearModel.from_directory(MODEL_DIR, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    moe_modules = [m for m in model.modules() if isinstance(m, KLinearMoE)]
    if not moe_modules:
        raise RuntimeError("no MoE modules found; the checkpoint did not load as expected")
    dense_modules = [
        m
        for m in model.modules()
        if isinstance(m, W4A16Linear) and not m._retained_bf16
    ]

    # A real prompt, not filler. An earlier version used a repeated dummy token
    # and every variant generated the same token 64 times, so "token ids
    # identical" was true but proved almost nothing. Equivalence is only
    # evidence when the reference output actually varies, which is asserted
    # below.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=device)

    results: list[dict] = []
    reference_ids: list[int] | None = None

    for spec in variants:
        # A spec is "grouped" or "grouped/dense". Swapping a variant requires
        # clearing the callable cached on every module, because it is resolved
        # once at prepare time precisely so the decode loop never pays for
        # resolution.
        parts = spec.split("/")
        grouped_name = parts[0]
        dense_name = parts[1] if len(parts) > 1 and parts[1] else "reference"
        swiglu_name = parts[2] if len(parts) > 2 and parts[2] else "reference"

        registry.use(W4A16_GROUPED, grouped_name)
        registry.use(W4A16_DENSE, dense_name)
        registry.use(W4A16_SWIGLU, swiglu_name)
        for module in moe_modules:
            module._grouped_kernel = registry.resolve(W4A16_GROUPED)
            module._swiglu_kernel = registry.resolve(W4A16_SWIGLU)
        for module in dense_modules:
            module._dense_kernel = registry.resolve(W4A16_DENSE)
        variant = spec

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        output = prefill(model, input_ids)
        torch.cuda.synchronize(device)
        runner = CUDAGraphDecodeRunner(model, input_ids, output, new_tokens)

        runner.reset()
        for _ in range(8):
            runner.graph.replay()
        torch.cuda.synchronize(device)

        timings = []
        for _ in range(repeats):
            runner.reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(new_tokens):
                runner.graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))

        timings.sort()
        median_ms = timings[len(timings) // 2]
        generated = runner.result().generated_ids[0].detach().cpu().tolist()

        identical = None
        if reference_ids is None:
            reference_ids = generated
        else:
            identical = generated == reference_ids

        results.append(
            {
                "variant": variant,
                "active_grouped": registry.active(W4A16_GROUPED),
                "active_dense": registry.active(W4A16_DENSE),
                "active_swiglu": registry.active(W4A16_SWIGLU),
                "median_ms_for_all_tokens": median_ms,
                "ms_per_token": median_ms / new_tokens,
                "tokens_per_second": 1000.0 * new_tokens / median_ms,
                "all_timings_ms": timings,
                "generated_token_ids": generated,
                "token_ids_identical_to_reference": identical,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
        print(
            f"{variant:<26} {1000.0 * new_tokens / median_ms:8.2f} tok/s   "
            f"identical={identical}"
        )

    # Identical output is only evidence when the output varies. A degenerate
    # prompt that makes the model repeat one token would make every variant
    # "identical" while proving nothing about the kernels.
    distinct = len(set(reference_ids or []))
    if distinct < 3:
        raise RuntimeError(
            f"the reference generated only {distinct} distinct token ids, so an "
            "identical-output comparison proves nothing. Use a prompt that "
            "produces varied output before trusting this gate."
        )

    baseline = results[0]["tokens_per_second"]
    for row in results:
        row["speedup_over_first"] = row["tokens_per_second"] / baseline

    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": total_bytes,
        "memory_capped_to_bytes": BUDGET_BYTES if cap_memory else None,
        "load_seconds": load_seconds,
        "resident_weight_bytes": model.resident_weight_bytes,
        "prompt": prompt,
        "prompt_tokens": int(input_ids.shape[1]),
        "distinct_generated_ids": len(set(reference_ids or [])),
        "new_tokens": new_tokens,
        "repeats": repeats,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


@app.local_entrypoint()
def main(
    prompt: str = "Explain in two sentences why quantizing a large language model to four bits reduces memory.",
    variants: str = "reference/reference,triton_gemv/reference,triton_gemv/triton_gemv",
    new_tokens: int = 64,
    repeats: int = 3,
) -> None:
    payload = bench_variants.remote(
        [name.strip() for name in variants.split(",") if name.strip()],
        prompt=prompt,
        new_tokens=new_tokens,
        repeats=repeats,
    )
    print("\n" + "=" * 74)
    print(f"GPU {payload['gpu']}")
    print(f"{'variant':<26}{'tok/s':>10}{'ms/tok':>10}{'speedup':>10}{'identical':>12}")
    for row in payload["results"]:
        identical = row["token_ids_identical_to_reference"]
        label = "reference" if identical is None else str(identical)
        print(
            f"{row['variant']:<26}{row['tokens_per_second']:>10.2f}"
            f"{row['ms_per_token']:>10.2f}{row['speedup_over_first']:>10.2f}x"
            f"{label:>12}"
        )
    print("=" * 74)
