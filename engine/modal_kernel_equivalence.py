"""Teacher-forced kernel equivalence, isolated from greedy compounding.

Free-running greedy decode is a bad equivalence test for a kernel change. Two
kernels that compute the same function in a different reduction order differ by
a tiny amount, and greedy decoding turns the first flipped argmax into a
completely different continuation. Measuring that tells you decoding is chaotic,
not whether the kernel is wrong.

This runner feeds the SAME fixed token sequence through every variant and
compares the logits produced at each step. Divergence cannot compound, so what
is measured is the kernel's own numerical error.

    modal run engine/modal_kernel_equivalence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-kernel-equivalence")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"

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
def compare(
    variants: list[str],
    prompt: str = "Explain in two sentences why quantizing a large language model to four bits reduces memory.",
    steps: int = 48,
) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from engine.kernels import W4A16_DENSE, W4A16_GROUPED, registry
    from engine.klinear.generate import decode, prefill
    from engine.klinear.model import KLinearModel
    from engine.klinear.moe import KLinearMoE
    from engine.klinear.quantized import W4A16Linear

    device = torch.device("cuda:0")
    model = KLinearModel.from_directory(MODEL_DIR, device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=device)

    moe_modules = [m for m in model.modules() if isinstance(m, KLinearMoE)]
    dense_modules = [
        m for m in model.modules() if isinstance(m, W4A16Linear) and not m._retained_bf16
    ]

    def select(spec: str) -> None:
        grouped_name, _, dense_name = spec.partition("/")
        registry.use(W4A16_GROUPED, grouped_name)
        registry.use(W4A16_DENSE, dense_name or "reference")
        for module in moe_modules:
            module._grouped_kernel = registry.resolve(W4A16_GROUPED)
        for module in dense_modules:
            module._dense_kernel = registry.resolve(W4A16_DENSE)

    def run(spec: str, forced: list[int] | None) -> tuple[torch.Tensor, list[int]]:
        """Return [steps, vocab] logits and the greedy ids taken at each step."""
        select(spec)
        output = prefill(model, input_ids)
        state = output.state.reserve_decode_capacity(steps + 4)
        logits = output.logits[:, -1]
        collected = []
        taken = []
        for index in range(steps):
            collected.append(logits[0].detach().float().cpu())
            greedy = int(logits[0].argmax().item())
            taken.append(greedy)
            # Teacher forcing: feed the REFERENCE token, not this variant's own
            # choice, so a single flipped argmax cannot change what comes next.
            feed = greedy if forced is None else forced[index]
            token = torch.tensor([[feed]], dtype=torch.long, device=device)
            output = decode(model, token, state)
            state = output.state
            logits = output.logits[:, -1]
        return torch.stack(collected), taken

    reference_logits, reference_taken = run(variants[0], None)

    rows = []
    for spec in variants:
        variant_logits, variant_taken = run(spec, reference_taken)
        difference = (variant_logits - reference_logits).abs()
        reference_probs = F.softmax(reference_logits, dim=-1)
        variant_probs = F.softmax(variant_logits, dim=-1)
        kl = (
            reference_probs
            * (reference_probs.clamp_min(1e-12).log() - variant_probs.clamp_min(1e-12).log())
        ).sum(dim=-1)
        top1 = sum(1 for a, b in zip(variant_taken, reference_taken) if a == b)
        rows.append(
            {
                "variant": spec,
                "steps": steps,
                "top1_agreement": top1 / steps,
                "top1_matches": top1,
                "max_abs_logit_diff": float(difference.max()),
                "mean_abs_logit_diff": float(difference.mean()),
                "mean_kl_nats": float(kl.mean()),
                "max_kl_nats": float(kl.max()),
            }
        )
        print(
            f"{spec:<26} top1 {top1}/{steps}  maxdiff {float(difference.max()):.5f}  "
            f"meanKL {float(kl.mean()):.3e}"
        )

    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "prompt": prompt,
        "prompt_tokens": int(input_ids.shape[1]),
        "steps": steps,
        "note": "Teacher forced on the first variant's greedy tokens, so logit "
        "differences are the kernel's own error and cannot compound.",
        "results": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


@app.local_entrypoint()
def main(
    variants: str = "reference/reference,triton_gemv/reference,triton_gemv/triton_gemv",
    steps: int = 48,
) -> None:
    payload = compare.remote(
        [name.strip() for name in variants.split(",") if name.strip()], steps=steps
    )
    print("\n" + "=" * 86)
    print(f"{'variant':<26}{'top-1':>10}{'max logit diff':>17}{'mean KL nats':>16}")
    for row in payload["results"]:
        print(
            f"{row['variant']:<26}{row['top1_agreement'] * 100:>9.1f}%"
            f"{row['max_abs_logit_diff']:>17.5f}{row['mean_kl_nats']:>16.3e}"
        )
    print("=" * 86)
