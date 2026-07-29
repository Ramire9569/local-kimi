"""Launch-fused decode preparation for Kimi Delta Attention.

The public fused functions are intended for sequence length one decode. They
allocate only fixed-shape outputs and mutate the three convolution states in
place, so a warmed call can be captured by a CUDA graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only test environments do not install Triton.
    triton = None
    tl = None


CONV_WIDTH = 4
SOFTPLUS_THRESHOLD = 20.0


if triton is not None:

    @triton.jit
    def _short_conv_triple_kernel(
        q_input_ptr,
        k_input_ptr,
        v_input_ptr,
        q_state_ptr,
        k_state_ptr,
        v_state_ptr,
        q_weight_ptr,
        k_weight_ptr,
        v_weight_ptr,
        q_output_ptr,
        k_output_ptr,
        v_output_ptr,
        channels,
        input_stride_batch,
        input_stride_channel,
        state_stride_batch,
        state_stride_channel,
        state_stride_window,
        weight_stride_channel,
        weight_stride_window,
        output_stride_batch,
        output_stride_channel,
        BLOCK_SIZE: tl.constexpr,
    ):
        convolution = tl.program_id(0)
        batch_index = tl.program_id(1)
        block = tl.program_id(2)
        channel_offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        channel_mask = channel_offsets < channels
        is_q = convolution == 0
        is_k = convolution == 1
        is_v = convolution == 2

        input_offsets = (
            batch_index * input_stride_batch
            + channel_offsets * input_stride_channel
        )
        state_offsets = (
            batch_index * state_stride_batch
            + channel_offsets * state_stride_channel
        )
        weight_offsets = channel_offsets * weight_stride_channel

        q_token = tl.load(
            q_input_ptr + input_offsets,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        k_token = tl.load(
            k_input_ptr + input_offsets,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        v_token = tl.load(
            v_input_ptr + input_offsets,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        token = q_token + k_token + v_token

        q_state_1 = tl.load(
            q_state_ptr + state_offsets + state_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_state_2 = tl.load(
            q_state_ptr + state_offsets + 2 * state_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_state_3 = tl.load(
            q_state_ptr + state_offsets + 3 * state_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        k_state_1 = tl.load(
            k_state_ptr + state_offsets + state_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        k_state_2 = tl.load(
            k_state_ptr + state_offsets + 2 * state_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        k_state_3 = tl.load(
            k_state_ptr + state_offsets + 3 * state_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        v_state_1 = tl.load(
            v_state_ptr + state_offsets + state_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        v_state_2 = tl.load(
            v_state_ptr + state_offsets + 2 * state_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        v_state_3 = tl.load(
            v_state_ptr + state_offsets + 3 * state_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        state_1 = q_state_1 + k_state_1 + v_state_1
        state_2 = q_state_2 + k_state_2 + v_state_2
        state_3 = q_state_3 + k_state_3 + v_state_3

        q_weight_0 = tl.load(
            q_weight_ptr + weight_offsets,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_weight_1 = tl.load(
            q_weight_ptr + weight_offsets + weight_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_weight_2 = tl.load(
            q_weight_ptr + weight_offsets + 2 * weight_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_weight_3 = tl.load(
            q_weight_ptr + weight_offsets + 3 * weight_stride_window,
            mask=channel_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        k_weight_0 = tl.load(
            k_weight_ptr + weight_offsets,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        k_weight_1 = tl.load(
            k_weight_ptr + weight_offsets + weight_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        k_weight_2 = tl.load(
            k_weight_ptr + weight_offsets + 2 * weight_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        k_weight_3 = tl.load(
            k_weight_ptr + weight_offsets + 3 * weight_stride_window,
            mask=channel_mask & is_k,
            other=0.0,
        ).to(tl.float32)
        v_weight_0 = tl.load(
            v_weight_ptr + weight_offsets,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        v_weight_1 = tl.load(
            v_weight_ptr + weight_offsets + weight_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        v_weight_2 = tl.load(
            v_weight_ptr + weight_offsets + 2 * weight_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        v_weight_3 = tl.load(
            v_weight_ptr + weight_offsets + 3 * weight_stride_window,
            mask=channel_mask & is_v,
            other=0.0,
        ).to(tl.float32)
        weight_0 = q_weight_0 + k_weight_0 + v_weight_0
        weight_1 = q_weight_1 + k_weight_1 + v_weight_1
        weight_2 = q_weight_2 + k_weight_2 + v_weight_2
        weight_3 = q_weight_3 + k_weight_3 + v_weight_3

        convolved = (
            state_1 * weight_0
            + state_2 * weight_1
            + state_3 * weight_2
            + token * weight_3
        )
        convolved = convolved / (1.0 + tl.exp(-convolved))

        tl.store(
            q_state_ptr + state_offsets,
            state_1,
            mask=channel_mask & is_q,
        )
        tl.store(
            q_state_ptr + state_offsets + state_stride_window,
            state_2,
            mask=channel_mask & is_q,
        )
        tl.store(
            q_state_ptr + state_offsets + 2 * state_stride_window,
            state_3,
            mask=channel_mask & is_q,
        )
        tl.store(
            q_state_ptr + state_offsets + 3 * state_stride_window,
            token,
            mask=channel_mask & is_q,
        )
        tl.store(
            k_state_ptr + state_offsets,
            state_1,
            mask=channel_mask & is_k,
        )
        tl.store(
            k_state_ptr + state_offsets + state_stride_window,
            state_2,
            mask=channel_mask & is_k,
        )
        tl.store(
            k_state_ptr + state_offsets + 2 * state_stride_window,
            state_3,
            mask=channel_mask & is_k,
        )
        tl.store(
            k_state_ptr + state_offsets + 3 * state_stride_window,
            token,
            mask=channel_mask & is_k,
        )
        tl.store(
            v_state_ptr + state_offsets,
            state_1,
            mask=channel_mask & is_v,
        )
        tl.store(
            v_state_ptr + state_offsets + state_stride_window,
            state_2,
            mask=channel_mask & is_v,
        )
        tl.store(
            v_state_ptr + state_offsets + 2 * state_stride_window,
            state_3,
            mask=channel_mask & is_v,
        )
        tl.store(
            v_state_ptr + state_offsets + 3 * state_stride_window,
            token,
            mask=channel_mask & is_v,
        )

        output_offsets = (
            batch_index * output_stride_batch
            + channel_offsets * output_stride_channel
        )
        tl.store(
            q_output_ptr + output_offsets,
            convolved,
            mask=channel_mask & is_q,
        )
        tl.store(
            k_output_ptr + output_offsets,
            convolved,
            mask=channel_mask & is_k,
        )
        tl.store(
            v_output_ptr + output_offsets,
            convolved,
            mask=channel_mask & is_v,
        )

    @triton.jit
    def _kda_gates_kernel(
        raw_decay_ptr,
        dt_bias_ptr,
        a_log_ptr,
        raw_beta_ptr,
        q_ptr,
        k_ptr,
        decay_output_ptr,
        beta_output_ptr,
        q_output_ptr,
        k_output_ptr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EPS: tl.constexpr,
        SOFTPLUS_LIMIT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        head_program = tl.program_id(0)
        head_index = head_program % NUM_HEADS
        dimension_offsets = tl.arange(0, BLOCK_SIZE)
        dimension_mask = dimension_offsets < HEAD_DIM
        vector_offsets = head_program * HEAD_DIM + dimension_offsets
        bias_offsets = head_index * HEAD_DIM + dimension_offsets

        raw_decay = tl.load(
            raw_decay_ptr + vector_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        dt_bias = tl.load(
            dt_bias_ptr + bias_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        biased = raw_decay + dt_bias
        stable_softplus = tl.where(biased > 0.0, biased, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(biased))
        )
        softplus = tl.where(biased > SOFTPLUS_LIMIT, biased, stable_softplus)
        rate = tl.exp(tl.load(a_log_ptr + head_index).to(tl.float32))
        decay = -rate * softplus
        tl.store(
            decay_output_ptr + vector_offsets,
            decay,
            mask=dimension_mask,
        )

        raw_beta = tl.load(raw_beta_ptr + head_program).to(tl.float32)
        beta = 1.0 / (1.0 + tl.exp(-raw_beta))
        tl.store(beta_output_ptr + head_program, beta)

        q = tl.load(
            q_ptr + vector_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + vector_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        q_scale = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) + EPS)
        k_scale = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) + EPS)
        tl.store(
            q_output_ptr + vector_offsets,
            q * q_scale,
            mask=dimension_mask,
        )
        tl.store(
            k_output_ptr + vector_offsets,
            k * k_scale,
            mask=dimension_mask,
        )

else:
    _short_conv_triple_kernel = None
    _kda_gates_kernel = None


def _conv_weight_view(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 3:
        return weight[:, 0, :]
    return weight


def _validate_short_conv_inputs(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    require_cuda: bool,
) -> tuple[int, int]:
    values = (q_in, k_in, v_in)
    states = (q_state, k_state, v_state)
    weights = (q_weight, k_weight, v_weight)
    if any(value.ndim != 3 for value in values):
        raise ValueError("q_in, k_in, and v_in must have shape [batch, 1, channels]")
    if not (q_in.shape == k_in.shape == v_in.shape):
        raise ValueError("q_in, k_in, and v_in must have identical shapes")
    batch, sequence, channels = q_in.shape
    if batch == 0 or channels == 0 or sequence != 1:
        raise ValueError("fused short convolution requires nonempty sequence length one input")
    expected_state = (batch, channels, CONV_WIDTH)
    if any(tuple(state.shape) != expected_state for state in states):
        raise ValueError("each short convolution state must have shape [batch, channels, 4]")
    valid_weight_shapes = {(channels, CONV_WIDTH), (channels, 1, CONV_WIDTH)}
    if any(tuple(weight.shape) not in valid_weight_shapes for weight in weights):
        raise ValueError("each short convolution weight must have shape [channels, 1, 4]")

    tensors = values + states + weights
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("short convolution inputs, states, and weights must be floating point")
    if any(value.dtype != state.dtype for value, state in zip(values, states)):
        raise TypeError("each short convolution input and its state must share a dtype")
    if any(tensor.device != q_in.device for tensor in tensors[1:]):
        raise ValueError("all short convolution tensors must share one device")
    if require_cuda and q_in.device.type != "cuda":
        raise ValueError("fused short convolution requires CUDA tensors")

    input_strides = (q_in.stride(0), q_in.stride(2))
    if any((value.stride(0), value.stride(2)) != input_strides for value in values[1:]):
        raise ValueError("q_in, k_in, and v_in must use the same layout")
    state_strides = q_state.stride()
    if any(state.stride() != state_strides for state in states[1:]):
        raise ValueError("q_state, k_state, and v_state must use the same layout")
    weight_strides = (_conv_weight_view(q_weight).stride(0), _conv_weight_view(q_weight).stride(1))
    if any(
        (_conv_weight_view(weight).stride(0), _conv_weight_view(weight).stride(1))
        != weight_strides
        for weight in weights[1:]
    ):
        raise ValueError("q_weight, k_weight, and v_weight must use the same layout")
    return batch, channels


def _short_conv_reference(
    values: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    token = values[:, 0]
    shifted = torch.cat((state[:, :, 1:], token.unsqueeze(-1)), dim=-1)
    weights = _conv_weight_view(weight)
    convolved = (shifted.float() * weights.float().unsqueeze(0)).sum(dim=-1)
    convolved = F.silu(convolved).to(values.dtype)
    state.copy_(shifted)
    return torch.stack([convolved], dim=1)


def fused_short_conv_triple_reference(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faithful eager decode reference with in-place state updates."""
    _validate_short_conv_inputs(
        q_in,
        k_in,
        v_in,
        q_state,
        k_state,
        v_state,
        q_weight,
        k_weight,
        v_weight,
        require_cuda=False,
    )
    return (
        _short_conv_reference(q_in, q_state, q_weight),
        _short_conv_reference(k_in, k_state, k_weight),
        _short_conv_reference(v_in, v_state, v_weight),
    )


def fused_short_conv_triple(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run all three width-four decode convolutions in one Triton launch."""
    if triton is None or _short_conv_triple_kernel is None:
        raise RuntimeError("Triton is required for fused KDA short convolution")
    batch, channels = _validate_short_conv_inputs(
        q_in,
        k_in,
        v_in,
        q_state,
        k_state,
        v_state,
        q_weight,
        k_weight,
        v_weight,
        require_cuda=True,
    )
    q_output = torch.empty_like(q_in)
    k_output = torch.empty_like(k_in)
    v_output = torch.empty_like(v_in)
    q_weight_view = _conv_weight_view(q_weight)
    block_size = 128
    grid = (3, batch, triton.cdiv(channels, block_size))
    with torch.cuda.device(q_in.device):
        _short_conv_triple_kernel[grid](
            q_in,
            k_in,
            v_in,
            q_state,
            k_state,
            v_state,
            q_weight,
            k_weight,
            v_weight,
            q_output,
            k_output,
            v_output,
            channels,
            q_in.stride(0),
            q_in.stride(2),
            q_state.stride(0),
            q_state.stride(1),
            q_state.stride(2),
            q_weight_view.stride(0),
            q_weight_view.stride(1),
            q_output.stride(0),
            q_output.stride(2),
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return q_output, k_output, v_output


def _validate_gate_inputs(
    raw_decay: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    raw_beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
    require_cuda: bool,
) -> tuple[int, int]:
    if num_heads <= 0 or head_dim <= 0 or head_dim > 1024:
        raise ValueError("num_heads and head_dim must be positive, with head_dim at most 1024")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if raw_decay.ndim != 4:
        raise ValueError("raw_decay must have shape [batch, sequence, heads, head_dim]")
    batch, sequence, decay_heads, decay_dim = raw_decay.shape
    if batch == 0 or sequence == 0:
        raise ValueError("KDA gate inputs must have nonempty batch and sequence dimensions")
    if (decay_heads, decay_dim) != (num_heads, head_dim):
        raise ValueError("raw_decay shape does not match num_heads and head_dim")
    if tuple(raw_beta.shape) != (batch, sequence, num_heads):
        raise ValueError("raw_beta must have shape [batch, sequence, num_heads]")
    if tuple(dt_bias.shape) != (num_heads * head_dim,):
        raise ValueError("dt_bias must be flat with num_heads * head_dim entries")
    if tuple(a_log.shape) != (1, 1, num_heads, 1):
        raise ValueError("a_log must have shape [1, 1, num_heads, 1]")

    flat_shape = (batch, sequence, num_heads * head_dim)
    headed_shape = (batch, sequence, num_heads, head_dim)
    if tuple(q.shape) not in (flat_shape, headed_shape):
        raise ValueError("q must be flat by projection or viewed as heads")
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError("q and k must have identical shapes")
    tensors = (raw_decay, dt_bias, a_log, raw_beta, q, k)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("KDA gate inputs must be floating point")
    if any(tensor.device != raw_decay.device for tensor in tensors[1:]):
        raise ValueError("all KDA gate tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("KDA gate tensors must be contiguous without transposition")
    if require_cuda and raw_decay.device.type != "cuda":
        raise ValueError("fused KDA gates require CUDA tensors")
    return batch, sequence


def fused_kda_gates_reference(
    raw_decay: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    raw_beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faithful eager transcription of KDA decay, beta, q, and k preparation."""
    batch, sequence = _validate_gate_inputs(
        raw_decay,
        dt_bias,
        a_log,
        raw_beta,
        q,
        k,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        require_cuda=False,
    )
    biased = raw_decay.float() + dt_bias.float().view(num_heads, head_dim)
    rate = a_log.float().view(1, 1, num_heads, 1).exp()
    decay = -rate * F.softplus(biased)
    beta = torch.sigmoid(raw_beta.float())
    q_normalised = q.view(batch, sequence, num_heads, head_dim).float()
    k_normalised = k.view(batch, sequence, num_heads, head_dim).float()
    q_normalised = q_normalised * torch.rsqrt(
        q_normalised.square().sum(dim=-1, keepdim=True) + eps
    )
    k_normalised = k_normalised * torch.rsqrt(
        k_normalised.square().sum(dim=-1, keepdim=True) + eps
    )
    return decay, beta, q_normalised, k_normalised


def fused_kda_gates(
    raw_decay: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    raw_beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse decay, beta, and per-head q and k normalization into one launch."""
    if triton is None or _kda_gates_kernel is None:
        raise RuntimeError("Triton is required for fused KDA gate preparation")
    batch, sequence = _validate_gate_inputs(
        raw_decay,
        dt_bias,
        a_log,
        raw_beta,
        q,
        k,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        require_cuda=True,
    )
    decay = torch.empty_like(raw_decay, dtype=torch.float32)
    beta = torch.empty_like(raw_beta, dtype=torch.float32)
    q_normalised = torch.empty(
        (batch, sequence, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    k_normalised = torch.empty_like(q_normalised)
    block_size = triton.next_power_of_2(head_dim)
    head_programs = batch * sequence * num_heads
    with torch.cuda.device(raw_decay.device):
        _kda_gates_kernel[(head_programs,)](
            raw_decay,
            dt_bias,
            a_log,
            raw_beta,
            q,
            k,
            decay,
            beta,
            q_normalised,
            k_normalised,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            EPS=eps,
            SOFTPLUS_LIMIT=SOFTPLUS_THRESHOLD,
            BLOCK_SIZE=block_size,
            num_warps=4 if block_size <= 256 else 8,
        )
    return decay, beta, q_normalised, k_normalised
