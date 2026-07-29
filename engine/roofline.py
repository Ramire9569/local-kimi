"""Bytes read per decoded token, and the throughput ceiling that implies.

Decode at batch one is weight streaming. Every generated token reads the
activated parameters once, so the fastest any kernel can possibly go is
bytes-per-token divided by the card's memory bandwidth. This module computes
that number from the architecture rather than guessing at it, because it decides
whether a throughput target is an engineering problem or an arithmetic
impossibility.

    uv run python engine/roofline.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Kimi-Linear-48B-A3B-Instruct, from config.json.
LAYERS = 27
DENSE_PREFIX_LAYERS = 1  # first_k_dense_replace
HIDDEN = 2304
MOE_INTERMEDIATE = 1024
ROUTED_EXPERTS = 256
EXPERTS_PER_TOKEN = 8
SHARED_EXPERTS = 1
VOCAB = 163840

KDA_HEADS = 32
KDA_HEAD_DIM = 128
KDA_PROJECTION = KDA_HEADS * KDA_HEAD_DIM  # 4096

MLA_LAYERS = 7  # full_attn_layers, one-based [4,8,12,16,20,24,27]
KDA_LAYERS = LAYERS - MLA_LAYERS
MLA_HEADS = 32
KV_LORA_RANK = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD_DIM = 128

BYTES_INT4 = 4.5 / 8  # symmetric signed INT4 plus BF16 per-group scales, group 32
BYTES_BF16 = 2.0

# NVIDIA L40S. Peak is the figure NVIDIA publishes for the memory subsystem.
PEAK_BYTES_PER_SECOND = 864e9


@dataclass
class Component:
    name: str
    parameters: int
    bytes_per_parameter: float
    note: str = ""
    quantized: bool = field(init=False)

    def __post_init__(self) -> None:
        self.quantized = self.bytes_per_parameter < BYTES_BF16

    @property
    def total_bytes(self) -> float:
        return self.parameters * self.bytes_per_parameter


def components() -> list[Component]:
    moe_layers = LAYERS - DENSE_PREFIX_LAYERS
    routes = EXPERTS_PER_TOKEN + SHARED_EXPERTS

    # Each expert is w1, w3 of [intermediate, hidden] and w2 of [hidden, intermediate].
    per_expert = 3 * MOE_INTERMEDIATE * HIDDEN
    expert_params = moe_layers * routes * per_expert

    # q, k, v and o. The gates, the short convolutions and b_proj stay in BF16
    # because router and recurrent controls are quantisation sensitive.
    kda_quantized = KDA_LAYERS * (3 * HIDDEN * KDA_PROJECTION + KDA_PROJECTION * HIDDEN)
    kda_retained = KDA_LAYERS * (
        HIDDEN * KDA_HEAD_DIM  # f_a
        + KDA_HEAD_DIM * KDA_PROJECTION  # f_b
        + HIDDEN * KDA_HEADS  # b_proj
        + HIDDEN * KDA_HEAD_DIM  # g_a
        + KDA_HEAD_DIM * KDA_PROJECTION  # g_b
    )

    q_head_dim = QK_NOPE + QK_ROPE
    mla_quantized = MLA_LAYERS * (
        HIDDEN * MLA_HEADS * q_head_dim  # q_proj
        + KV_LORA_RANK * MLA_HEADS * (QK_NOPE + V_HEAD_DIM)  # kv_b
        + MLA_HEADS * V_HEAD_DIM * HIDDEN  # o_proj
    )
    # kv_a writes the cached latent, so its error persists for a whole sequence.
    mla_retained = MLA_LAYERS * HIDDEN * (KV_LORA_RANK + QK_ROPE)

    return [
        Component("routed and shared experts", expert_params, BYTES_INT4,
                  f"{moe_layers} MoE layers, {routes} of {ROUTED_EXPERTS + SHARED_EXPERTS} experts per token"),
        Component("KDA q, k, v, o", kda_quantized, BYTES_INT4, f"{KDA_LAYERS} KDA layers"),
        Component("KDA gates and convolutions", kda_retained, BYTES_BF16, "quantisation sensitive"),
        Component("MLA q, kv_b, o", mla_quantized, BYTES_INT4, f"{MLA_LAYERS} MLA layers"),
        Component("MLA kv_a", mla_retained, BYTES_BF16, "writes the cached latent"),
        Component("router gates", moe_layers * HIDDEN * ROUTED_EXPERTS, BYTES_BF16,
                  "error changes discrete expert selection"),
        Component("language modelling head", VOCAB * HIDDEN, BYTES_BF16,
                  "retained: output logits are quantisation sensitive"),
    ]


def report() -> None:
    parts = components()
    total = sum(part.total_bytes for part in parts)
    floor_seconds = total / PEAK_BYTES_PER_SECOND

    print(f"{'component':<30}{'params':>16}{'B/param':>9}{'MB':>10}{'share':>8}")
    print("-" * 73)
    for part in sorted(parts, key=lambda item: -item.total_bytes):
        print(
            f"{part.name:<30}{part.parameters:>16,}{part.bytes_per_parameter:>9.4f}"
            f"{part.total_bytes / 1e6:>10.1f}{100 * part.total_bytes / total:>7.1f}%"
        )
    print("-" * 73)
    print(f"{'total read per token':<30}{'':>16}{'':>9}{total / 1e6:>10.1f}{100.0:>7.1f}%")
    print()
    print(f"L40S peak bandwidth          {PEAK_BYTES_PER_SECOND / 1e9:.0f} GB/s")
    print(f"floor at 100 percent of peak {floor_seconds * 1e3:.2f} ms/token"
          f"  =  {1 / floor_seconds:.0f} tok/s")
    print()

    measured_ms = 8.78
    print(f"measured today               {measured_ms:.2f} ms/token"
          f"  =  {1000 / measured_ms:.2f} tok/s")
    print(f"fraction of roofline reached {100 * floor_seconds * 1e3 / measured_ms:.1f} percent")
    print(f"headroom left in kernels     {measured_ms / (floor_seconds * 1e3):.2f}x")
    print()

    head = next(part for part in parts if "language modelling" in part.name)
    without_head = total - head.total_bytes + head.parameters * BYTES_INT4
    print("If the language modelling head were quantised to INT4:")
    print(f"  bytes per token            {without_head / 1e6:.1f} MB, down from {total / 1e6:.1f}")
    print(f"  new floor                  {1e9 * PEAK_BYTES_PER_SECOND / without_head / 1e9 / 1e3:.0f} tok/s"
          if False else
          f"  new floor                  {PEAK_BYTES_PER_SECOND / without_head:.0f} tok/s")
    print()
    print("What a multiple of today's throughput would require:")
    for factor in (2, 3, 5, 10, 20, 40):
        target_ms = measured_ms / factor
        needed = total / (target_ms / 1e3) / 1e9
        verdict = "reachable" if needed <= PEAK_BYTES_PER_SECOND / 1e9 else "IMPOSSIBLE at batch 1"
        print(
            f"  {factor:>2}x  ->  {1000 / target_ms:>7.0f} tok/s  needs "
            f"{needed:>8.0f} GB/s  ({needed / (PEAK_BYTES_PER_SECOND / 1e9):.1f}x the bus)  {verdict}"
        )


if __name__ == "__main__":
    report()
