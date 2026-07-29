"""Run a kernel benchmark script on a GPU.

These benchmarks build synthetic tensors at the real model shapes, so they need
a GPU but not the checkpoint. That makes them cheap and quick compared with the
full decode profile, which spends minutes loading weights.

    modal run engine/modal_kbench.py --script BENCH-GEMV.py
    modal run engine/modal_kbench.py --script BENCH-KDA.py --gpu H100
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("kimi-linear-kernel-bench")

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch>=2.5",
        "triton>=3.1",
        "numpy>=2.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


@app.function(image=IMAGE, gpu="L40S", cpu=8.0, memory=32768, timeout=60 * 45)
def run_bench(script: str, arguments: list[str]) -> str:
    import subprocess
    import sys

    path = Path("/root/engine/kernels") / script
    if not path.exists():
        available = sorted(item.name for item in path.parent.glob("BENCH-*.py"))
        raise FileNotFoundError(f"{script} not found; available: {available}")

    completed = subprocess.run(
        [sys.executable, str(path), *arguments],
        capture_output=True,
        text=True,
        cwd="/root",
        env={"PYTHONPATH": "/root", "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    output = completed.stdout
    if completed.stderr.strip():
        output += "\n--- stderr ---\n" + completed.stderr
    output += f"\n--- exit code {completed.returncode} ---"
    print(output)
    return output


@app.function(image=IMAGE, gpu="H100", cpu=8.0, memory=32768, timeout=60 * 45)
def run_bench_h100(script: str, arguments: list[str]) -> str:
    return run_bench.local(script, arguments)


@app.local_entrypoint()
def main(script: str = "BENCH-GEMV.py", gpu: str = "L40S", args: str = "") -> None:
    arguments = args.split() if args else []
    runner = run_bench_h100 if gpu.upper() == "H100" else run_bench
    print(runner.remote(script, arguments))
