"""Fused Kimi Delta Attention decode recurrence."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


KDA_BLOCK_V = 32
_BENCHMARK_BLOCK_VS = (16, 32, 64)


if triton is not None:

    @triton.jit
    def _kda_decode_step_kernel(
        state_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        decay_ptr,
        beta_ptr,
        output_ptr,
        HEADS,
        D_K,
        D_V,
        scale,
        stride_sb,
        stride_sh,
        stride_sk,
        stride_sv,
        stride_qb,
        stride_qh,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vv,
        stride_db,
        stride_dh,
        stride_dk,
        stride_bb,
        stride_bh,
        stride_ob,
        stride_oh,
        stride_ov,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        value_block = tl.program_id(1)
        batch_index = batch_head // HEADS
        head_index = batch_head - batch_index * HEADS

        row_offsets = tl.arange(0, BLOCK_K)
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        row_mask = row_offsets < D_K
        value_mask = value_offsets < D_V
        state_mask = row_mask[:, None] & value_mask[None, :]

        state_offsets = (
            batch_index * stride_sb
            + head_index * stride_sh
            + row_offsets[:, None] * stride_sk
            + value_offsets[None, :] * stride_sv
        )
        state_values = tl.load(
            state_ptr + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        decay_values = tl.load(
            decay_ptr
            + batch_index * stride_db
            + head_index * stride_dh
            + row_offsets * stride_dk,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        key_values = tl.load(
            k_ptr
            + batch_index * stride_kb
            + head_index * stride_kh
            + row_offsets * stride_kk,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        query_values = tl.load(
            q_ptr
            + batch_index * stride_qb
            + head_index * stride_qh
            + row_offsets * stride_qk,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        value_values = tl.load(
            v_ptr
            + batch_index * stride_vb
            + head_index * stride_vh
            + value_offsets * stride_vv,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        beta_value = tl.load(
            beta_ptr + batch_index * stride_bb + head_index * stride_bh
        ).to(tl.float32)

        decayed_state = state_values * tl.exp(decay_values)[:, None]
        prediction = tl.sum(decayed_state * key_values[:, None], axis=0)
        delta = value_values - prediction
        updated_state = (
            decayed_state
            + beta_value * key_values[:, None] * delta[None, :]
        )
        scaled_query = query_values * scale
        output_values = tl.sum(updated_state * scaled_query[:, None], axis=0)

        tl.store(state_ptr + state_offsets, updated_state, mask=state_mask)
        tl.store(
            output_ptr
            + batch_index * stride_ob
            + head_index * stride_oh
            + value_offsets * stride_ov,
            output_values,
            mask=value_mask,
        )

else:
    _kda_decode_step_kernel = None


def _validate_inputs(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int]:
    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, d_k, d_v]")
    if any(tensor.dtype != torch.float32 for tensor in (state, q, k, v, decay, beta)):
        raise TypeError("KDA decode inputs must all use torch.float32")
    if not all(
        tensor.device == state.device for tensor in (q, k, v, decay, beta)
    ):
        raise ValueError("KDA decode inputs must share one device")

    batch, heads, d_k, d_v = state.shape
    if batch == 0 or heads == 0 or d_k == 0 or d_v == 0:
        raise ValueError("KDA decode dimensions must be nonzero")
    if q.shape != (batch, heads, d_k):
        raise ValueError("q must have shape [batch, heads, d_k]")
    if k.shape != (batch, heads, d_k):
        raise ValueError("k must have shape [batch, heads, d_k]")
    if v.shape != (batch, heads, d_v):
        raise ValueError("v must have shape [batch, heads, d_v]")
    if decay.shape != (batch, heads, d_k):
        raise ValueError("decay must have shape [batch, heads, d_k]")
    if beta.shape != (batch, heads):
        raise ValueError("beta must have shape [batch, heads]")
    return batch, heads, d_k, d_v


def kda_decode_step_reference(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Run the source PyTorch recurrence and mutate ``state`` in place."""
    _validate_inputs(state, q, k, v, decay, beta)

    candidate = state * torch.exp(decay).unsqueeze(-1)
    prediction = (candidate * k.unsqueeze(-1)).sum(dim=-2)
    delta = v - prediction
    candidate = candidate + (
        beta.unsqueeze(-1).unsqueeze(-1)
        * k.unsqueeze(-1)
        * delta.unsqueeze(-2)
    )
    output = (candidate * (q * scale).unsqueeze(-1)).sum(dim=-2)
    state.copy_(candidate)
    return output


def _num_warps_for_block_v(block_v: int) -> int:
    if block_v == 16:
        return 4
    if block_v in (32, 64):
        return 8
    raise ValueError(f"BLOCK_V must be one of {_BENCHMARK_BLOCK_VS}")


def _kda_decode_step_with_block_v(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    block_v: int,
) -> torch.Tensor:
    if triton is None or _kda_decode_step_kernel is None:
        raise RuntimeError("Triton is required for fused KDA decode")
    batch, heads, d_k, d_v = _validate_inputs(state, q, k, v, decay, beta)
    if state.device.type != "cuda":
        raise ValueError("fused KDA decode requires CUDA inputs")

    num_warps = _num_warps_for_block_v(block_v)
    block_k = triton.next_power_of_2(d_k)
    output = torch.empty(
        (batch, heads, d_v),
        dtype=torch.float32,
        device=state.device,
    )
    grid = (batch * heads, triton.cdiv(d_v, block_v))
    with torch.cuda.device(state.device):
        _kda_decode_step_kernel[grid](
            state,
            q,
            k,
            v,
            decay,
            beta,
            output,
            heads,
            d_k,
            d_v,
            scale,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            decay.stride(0),
            decay.stride(1),
            decay.stride(2),
            beta.stride(0),
            beta.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )
    return output


def kda_decode_step(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Fuse one KDA decode update while mutating the recurrent state in place."""
    return _kda_decode_step_with_block_v(
        state,
        q,
        k,
        v,
        decay,
        beta,
        scale=scale,
        block_v=KDA_BLOCK_V,
    )
