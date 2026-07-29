"""Build a W3A16 artifact from the original BF16 checkpoint.

The INT4 artifact is 26.83 GiB and an RTX 3090 or 4090 has 24 GB, so the two
cards enthusiasts actually own cannot load it. Three-bit expert weights bring it
to about 21.28 GiB.

This quantises from the ORIGINAL BF16 weights, not from the existing INT4
artifact. Quantising an already-quantised tensor compounds both errors for no
benefit, and the source checkpoint is still available.

Which tensors get quantised is decided by the same plan the INT4 build used, so
routers, the vocabulary head, the embedding, the KDA gates and the MLA latent
projection stay in BF16 exactly as before. Only the codec changes.

    modal run engine/modal_quantize_int3.py
    modal run engine/modal_quantize_int3.py --overwrite
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import modal

app = modal.App("kimi-linear-quantize-int3")

SOURCE_VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
OUTPUT_VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=True)
SOURCE_MOUNT = "/weights"
OUTPUT_MOUNT = "/output"
MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
SOURCE_DIR = f"{SOURCE_MOUNT}/{MODEL_NAME}"
OUTPUT_DIR = f"{OUTPUT_MOUNT}/{MODEL_NAME}-W3A16"

GROUP_SIZE = 32

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
        "numpy>=2.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


@app.function(
    image=IMAGE,
    gpu="H100",
    cpu=16.0,
    memory=131072,
    timeout=60 * 120,
    volumes={SOURCE_MOUNT: SOURCE_VOLUME, OUTPUT_MOUNT: OUTPUT_VOLUME},
)
def quantize_int3(overwrite: bool = False, group_size: int = GROUP_SIZE) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from engine.quant.klinear_plan import (
        TensorMetadata,
        build_klinear_quantization_plan,
    )
    from engine.quant.w3a16 import dequantise as dequantise3
    from engine.quant.w3a16 import quantise as quantise3

    index_path = os.path.join(SOURCE_DIR, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    # Collect tensor metadata without loading payloads, so the plan is built
    # before anything touches the GPU. TensorMetadata validates dtype names, so
    # a safetensors dtype string the plan does not know fails here rather than
    # after an hour of quantisation.
    shards: dict[str, None] = {}
    specs: list[TensorMetadata] = []
    for shard in weight_map.values():
        shards.setdefault(shard, None)
    for shard in sorted(shards):
        with safe_open(os.path.join(SOURCE_DIR, shard), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                slice_ = handle.get_slice(name)
                specs.append(
                    TensorMetadata(
                        name=name,
                        shape=tuple(slice_.get_shape()),
                        dtype=slice_.get_dtype(),
                        source_file=shard,
                    )
                )

    plan = build_klinear_quantization_plan(tuple(specs))
    decisions = {decision.name: decision for decision in plan.tensors}

    staging = f"{OUTPUT_DIR}.staging-{uuid.uuid4().hex}"
    if os.path.exists(OUTPUT_DIR) and not overwrite:
        raise FileExistsError(f"{OUTPUT_DIR} exists; pass --overwrite to replace it")
    os.makedirs(staging, exist_ok=False)

    # Everything that is not a weight shard: config, tokenizer, modeling code.
    for root, directories, filenames in os.walk(SOURCE_DIR):
        directories[:] = [item for item in directories if item != ".cache"]
        relative = os.path.relpath(root, SOURCE_DIR)
        target_root = staging if relative == "." else os.path.join(staging, relative)
        os.makedirs(target_root, exist_ok=True)
        for filename in filenames:
            if filename.endswith(".safetensors") or filename == "model.safetensors.index.json":
                continue
            shutil.copy2(os.path.join(root, filename), os.path.join(target_root, filename))

    output_weight_map: dict[str, str] = {}
    quantized_count = 0
    retained_count = 0
    source_bytes = 0
    output_bytes = 0
    worst: list[dict] = []

    for shard in sorted(shards):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(SOURCE_DIR, shard), framework="pt") as handle:
            for name in sorted(handle.keys()):  # noqa: SIM118
                source = handle.get_tensor(name)
                source_bytes += source.numel() * source.element_size()
                decision = decisions.get(name)
                if decision is not None and decision.quantize:
                    reduction = source.shape[-1]
                    flat = source.reshape(-1, reduction).to("cuda", torch.bfloat16)
                    encoded = quantise3(flat, group_size)
                    # Round-trip error on the real tensor, not on random data.
                    restored = dequantise3(encoded)
                    denominator = flat.float().norm().item() or 1.0
                    relative = (restored.float() - flat.float()).norm().item() / denominator
                    worst.append(
                        {
                            "name": name,
                            "tensor_class": decision.tensor_class,
                            "relative_frobenius": relative,
                        }
                    )
                    packed = encoded.packed.cpu().contiguous()
                    scales = encoded.scales.cpu().contiguous()
                    packed_name = f"{name}.w3a16_packed"
                    scales_name = f"{name}.w3a16_scales"
                    tensors[packed_name] = packed
                    tensors[scales_name] = scales
                    output_weight_map[packed_name] = shard
                    output_weight_map[scales_name] = shard
                    output_bytes += packed.numel() + scales.numel() * scales.element_size()
                    quantized_count += 1
                    del encoded, flat, restored, packed, scales
                    torch.cuda.empty_cache()
                else:
                    retained = source.contiguous()
                    tensors[name] = retained
                    output_weight_map[name] = shard
                    output_bytes += retained.numel() * retained.element_size()
                    retained_count += 1
        save_file(tensors, os.path.join(staging, shard), metadata={"format": "pt"})
        del tensors
        print(f"wrote {shard}: running output {output_bytes / 2**30:.2f} GiB")

    with open(os.path.join(staging, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": {"total_size": output_bytes}, "weight_map": output_weight_map},
            handle,
            indent=2,
            sort_keys=True,
        )

    worst.sort(key=lambda row: -row["relative_frobenius"])
    manifest = {
        "codec": "W3A16",
        "group_size": group_size,
        "bits_per_parameter_including_scales": 3 + 16 / group_size,
        "source_tensor_storage_bytes": source_bytes,
        "output_tensor_storage_bytes": output_bytes,
        "compression_ratio": source_bytes / output_bytes,
        "quantized_tensors": quantized_count,
        "retained_tensors": retained_count,
        "worst_relative_frobenius": worst[:10],
        "fits_24gb_card": output_bytes / 2**30 < 21.5,
    }
    with open(os.path.join(staging, "quantization-manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.rename(staging, OUTPUT_DIR)
    OUTPUT_VOLUME.commit()

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@app.local_entrypoint()
def main(overwrite: bool = False, group_size: int = GROUP_SIZE) -> None:
    manifest = quantize_int3.remote(overwrite=overwrite, group_size=group_size)
    print("\n" + "=" * 72)
    print(f"source   {manifest['source_tensor_storage_bytes'] / 2**30:>8.2f} GiB")
    print(f"output   {manifest['output_tensor_storage_bytes'] / 2**30:>8.2f} GiB")
    print(f"ratio    {manifest['compression_ratio']:>8.2f}x")
    print(f"quantized {manifest['quantized_tensors']}, retained {manifest['retained_tensors']}")
    print(f"fits a 24 GB card: {manifest['fits_24gb_card']}")
    print("worst round-trip error:")
    for row in manifest["worst_relative_frobenius"][:5]:
        print(f"  {row['relative_frobenius']:.4f}  {row['tensor_class']:<28}{row['name'][:44]}")
    print("=" * 72)
