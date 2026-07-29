"""Run llama.cpp on the SAME card this engine was measured on.

Every comparison against llama.cpp so far has been a quote from a pull request
run on different hardware with a different harness. That is not a comparison and
this repository has said so. This runs llama.cpp on the same NVIDIA L40S, on the
same model, at a comparable quantisation, and reports both numbers.

What this controls for: card, driver, model, prompt length, generated token
count, single stream, greedy.

What it does not control for: the quantisation formats differ. Ours is symmetric
signed INT4 with group 32 and BF16 scales, roughly 4.5 bits per parameter.
Q4_K_M is a k-quant with a different block layout and a different mix of
per-tensor precisions. They are the same order of precision, not the same codec,
and the comparison is engine against engine at similar bit budgets rather than a
controlled study of one variable.

    modal run engine/modal_llamacpp_headtohead.py
    modal run engine/modal_llamacpp_headtohead.py --quant Q4_K_M
"""

from __future__ import annotations

import json

import modal

app = modal.App("kimi-linear-llamacpp-headtohead")

CACHE = modal.Volume.from_name("llamacpp-gguf", create_if_missing=True)
CACHE_MOUNT = "/gguf"
REPOSITORY = "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF"

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "ccache")
    .pip_install("huggingface_hub[hf_transfer]>=0.26")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build "
        "-DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CUDA_ARCHITECTURES=89",
        "cmake --build /opt/llama.cpp/build --config Release -j 16 --target llama-bench",
        # The link step needs libcuda.so.1, which only exists when a driver is
        # present. Modal image builds have no GPU by default, so the build fails
        # with undefined references to cuMemCreate and friends. Attaching a GPU
        # to the build puts the real driver on the image.
        gpu="L40S",
    )
)


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=16.0,
    memory=65536,
    timeout=60 * 90,
    volumes={CACHE_MOUNT: CACHE},
)
def head_to_head(quant: str = "Q4_K_M", generated: int = 64, prompt: int = 17) -> dict:
    import os
    import subprocess

    from huggingface_hub import hf_hub_download

    filename = f"moonshotai_Kimi-Linear-48B-A3B-Instruct-{quant}.gguf"
    target = os.path.join(CACHE_MOUNT, filename)
    if not os.path.exists(target):
        print(f"downloading {filename}")
        path = hf_hub_download(
            repo_id=REPOSITORY, filename=filename, local_dir=CACHE_MOUNT
        )
        target = path
        CACHE.commit()
    size_bytes = os.path.getsize(target)
    print(f"gguf {filename}: {size_bytes / 2**30:.2f} GiB")

    binary = "/opt/llama.cpp/build/bin/llama-bench"
    command = [
        binary,
        "-m", target,
        "-p", str(prompt),
        "-n", str(generated),
        "-ngl", "999",
        "-t", "16",
        "-r", "3",
        "-o", "json",
    ]
    print(" ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60 * 60)

    result: dict = {
        "quant": quant,
        "gguf_bytes": size_bytes,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-1500:],
    }

    rows = []
    try:
        parsed = json.loads(completed.stdout)
        for row in parsed:
            rows.append(
                {
                    "test": row.get("n_prompt") and f"pp{row['n_prompt']}" or f"tg{row.get('n_gen')}",
                    "n_prompt": row.get("n_prompt"),
                    "n_gen": row.get("n_gen"),
                    "avg_ts": row.get("avg_ts"),
                    "stddev_ts": row.get("stddev_ts"),
                    "gpu": row.get("gpu_info") or row.get("dev_description"),
                    "model_size": row.get("model_size"),
                }
            )
    except json.JSONDecodeError:
        result["stdout_tail"] = completed.stdout[-3000:]

    result["rows"] = rows
    generation = [row for row in rows if row.get("n_gen")]
    if generation:
        result["llamacpp_tokens_per_second"] = generation[0]["avg_ts"]
        result["llamacpp_stddev"] = generation[0].get("stddev_ts")

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(quant: str = "Q4_K_M", generated: int = 64, prompt: int = 17) -> None:
    result = head_to_head.remote(quant=quant, generated=generated, prompt=prompt)
    ours = 113.83
    print("\n" + "=" * 74)
    print("SAME CARD: NVIDIA L40S, single stream, greedy")
    print(f"{'engine':<34}{'quant':<12}{'GiB':>7}{'tok/s':>10}")
    print("-" * 74)
    print(f"{'local-kimi fused kernels':<34}{'INT4 g32':<12}{26.83:>7.2f}{ours:>10.2f}")
    llama = result.get("llamacpp_tokens_per_second")
    if llama:
        print(
            f"{'llama.cpp':<34}{result['quant']:<12}"
            f"{result['gguf_bytes'] / 2**30:>7.2f}{llama:>10.2f}"
        )
        print("-" * 74)
        print(f"ratio: {ours / llama:.2f}x")
    else:
        print(f"llama.cpp FAILED, returncode {result.get('returncode')}")
        print(result.get("stderr_tail", "")[-800:])
    print("=" * 74)
