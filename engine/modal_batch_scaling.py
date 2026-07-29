"""Aggregate throughput against batch size.

Decode at batch one is weight streaming: 2,338 MB read to emit a single token.
The same read serves every sequence in a batch, so aggregate throughput should
rise close to linearly until something other than bandwidth binds. This is the
only route to a large multiple of today's tokens per second, and it buys
aggregate throughput, NOT single-stream latency. A user waiting on one reply
sees the per-sequence number, which barely moves.

    modal run engine/modal_batch_scaling.py
    modal run engine/modal_batch_scaling.py --batches 1,2,4,8,16
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-batch-scaling")

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
def scale(
    batches: list[int],
    prompt: str = "Explain in two sentences why quantizing a large language model to four bits reduces memory.",
    new_tokens: int = 32,
    repeats: int = 3,
    cap_memory: bool = True,
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import CUDAGraphDecodeRunner, prefill
    from engine.klinear.model import KLinearModel

    device = torch.device("cuda:0")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if cap_memory:
        torch.cuda.set_per_process_memory_fraction(BUDGET_BYTES / total_bytes, device)

    model = KLinearModel.from_directory(MODEL_DIR, device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    single = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=device)

    rows: list[dict] = []
    for batch in batches:
        # The same prompt repeated. That keeps the shape fixed and the routing
        # identical across rows, so the only variable is batch size. Real
        # serving would have different prompts and different expert routing,
        # which is a harder case, so treat this as an upper bound.
        input_ids = single.repeat(batch, 1)
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = prefill(model, input_ids)
            torch.cuda.synchronize(device)
            runner = CUDAGraphDecodeRunner(model, input_ids, output, new_tokens)
            runner.reset()
            for _ in range(4):
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
            per_sequence = 1000.0 * new_tokens / median_ms
            rows.append(
                {
                    "batch": batch,
                    "ok": True,
                    "ms_per_step": median_ms / new_tokens,
                    "tokens_per_second_per_sequence": per_sequence,
                    "tokens_per_second_aggregate": per_sequence * batch,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            )
            print(
                f"batch {batch:>3}  {per_sequence:>7.2f} tok/s per sequence  "
                f"{per_sequence * batch:>8.2f} tok/s aggregate  "
                f"peak {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB"
            )
            del runner, output
            torch.cuda.empty_cache()
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as error:
            rows.append({"batch": batch, "ok": False, "error": str(error)[:300]})
            print(f"batch {batch:>3}  FAILED: {str(error)[:160]}")
            torch.cuda.empty_cache()

    good = [row for row in rows if row["ok"]]
    if good:
        base = good[0]["tokens_per_second_aggregate"]
        for row in good:
            row["aggregate_speedup_over_smallest_batch"] = (
                row["tokens_per_second_aggregate"] / base
            )

    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "prompt_tokens": int(single.shape[1]),
        "new_tokens": new_tokens,
        "repeats": repeats,
        "results": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


@app.local_entrypoint()
def main(batches: str = "1,2,4,8,16,32", new_tokens: int = 32, repeats: int = 3) -> None:
    payload = scale.remote(
        [int(item) for item in batches.split(",") if item.strip()],
        new_tokens=new_tokens,
        repeats=repeats,
    )
    print("\n" + "=" * 78)
    print(f"{'batch':>6}{'ms/step':>11}{'tok/s each':>13}{'tok/s total':>14}{'vs batch 1':>12}")
    for row in payload["results"]:
        if not row["ok"]:
            print(f"{row['batch']:>6}   failed")
            continue
        print(
            f"{row['batch']:>6}{row['ms_per_step']:>11.2f}"
            f"{row['tokens_per_second_per_sequence']:>13.2f}"
            f"{row['tokens_per_second_aggregate']:>14.2f}"
            f"{row.get('aggregate_speedup_over_smallest_batch', 1.0):>11.2f}x"
        )
    print("=" * 78)
