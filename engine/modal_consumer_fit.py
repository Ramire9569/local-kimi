"""Load the INT3 artifact on a real 24 GB card and generate tokens.

Fitting was proven by arithmetic and by the artifact's byte count. This proves
the other half: that the engine can load it, allocate its state, capture a decode
graph and produce tokens inside 24 GB.

The card is an A10G, which has 24 GB like an RTX 3090 and RTX 4090. It is not
those cards: its memory bandwidth is roughly 600 GB/s against 936 and 1008, so
the throughput here is a floor for what a 3090 or 4090 would do, not a
prediction of it.

    modal run engine/modal_consumer_fit.py
    modal run engine/modal_consumer_fit.py --artifact Kimi-Linear-48B-A3B-Instruct-W4A16
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-consumer-fit")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"

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
    gpu="A10G",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def fit(
    artifact: str = "Kimi-Linear-48B-A3B-Instruct-W3A16",
    prompt: str = "Explain in two sentences why quantizing a large language model to four bits reduces memory.",
    new_tokens: int = 32,
    repeats: int = 3,
) -> dict:
    import time
    import traceback

    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import CUDAGraphDecodeRunner, prefill
    from engine.klinear.model import KLinearModel

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    model_dir = f"{MOUNT}/{artifact}"

    result: dict = {
        "gpu": properties.name,
        "gpu_total_bytes": properties.total_memory,
        "artifact": artifact,
        "loaded": False,
    }

    started = time.perf_counter()
    try:
        model = KLinearModel.from_directory(model_dir, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
    except Exception as error:  # noqa: BLE001
        result["error"] = f"{type(error).__name__}: {error}"[:600]
        result["traceback"] = traceback.format_exc()[-4000:]
        result["load_seconds"] = time.perf_counter() - started
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    result["loaded"] = True
    result["load_seconds"] = time.perf_counter() - started
    result["resident_weight_bytes"] = model.resident_weight_bytes
    result["checkpoint_kind"] = str(model.checkpoint_kind)
    result["reserved_after_load_bytes"] = torch.cuda.memory_reserved(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=device)

    try:
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
        median = timings[len(timings) // 2]
        generated = runner.result().generated_ids[0].detach().cpu().tolist()

        result["generated"] = True
        result["tokens_per_second"] = 1000.0 * new_tokens / median
        result["ms_per_token"] = median / new_tokens
        result["all_timings_ms"] = timings
        result["distinct_generated_ids"] = len(set(generated))
        result["continuation"] = tokenizer.decode(generated, skip_special_tokens=True)[:400]
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["headroom_bytes"] = properties.total_memory - torch.cuda.max_memory_reserved(device)
    except Exception as error:  # noqa: BLE001
        result["generated"] = False
        result["error"] = f"{type(error).__name__}: {error}"[:600]
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    artifact: str = "Kimi-Linear-48B-A3B-Instruct-W3A16",
    new_tokens: int = 32,
    repeats: int = 3,
) -> None:
    result = fit.remote(artifact=artifact, new_tokens=new_tokens, repeats=repeats)
    print("\n" + "=" * 72)
    print(f"card       {result['gpu']}  {result['gpu_total_bytes'] / 2**30:.1f} GiB")
    print(f"artifact   {result['artifact']}")
    print(f"loaded     {result['loaded']}")
    if not result["loaded"]:
        print(f"error      {result.get('error')}")
        print("=" * 72)
        return
    print(f"weights    {result['resident_weight_bytes'] / 2**30:.2f} GiB")
    if result.get("generated"):
        print(f"throughput {result['tokens_per_second']:.2f} tok/s")
        print(f"peak       {result['peak_reserved_bytes'] / 2**30:.2f} GiB")
        print(f"headroom   {result['headroom_bytes'] / 2**30:.2f} GiB")
        print(f"distinct generated ids {result['distinct_generated_ids']}")
        print(f"text       {result['continuation'][:180]}")
    else:
        print(f"generation FAILED: {result.get('error')}")
    print("=" * 72)
