"""K3 packed-weight kernels and scheduling helpers.

Kernel variants are registered here rather than inside each kernel module, so
that importing one module cannot decide what the engine runs. Importing this
package registers every variant, and `registry.resolve` picks between them from
KIMI_KERNELS or a programmatic override, defaulting to the reference.

Registration tolerates a missing Triton: on a CPU-only box the kernel modules
still import, their entry points raise when called, and `requires_cuda=True`
makes the registry refuse to hand one out rather than letting a benchmark
quietly measure the reference and report it as the fast path.
"""

from engine.kernels import registry

W4A16_GROUPED = "w4a16_grouped"
W4A16_DENSE = "w4a16_dense"


def _register_grouped() -> None:
    from engine.kernels.w4a16_grouped import grouped_w4a16_linear

    registry.register(
        W4A16_GROUPED,
        "reference",
        reference=True,
        requires_cuda=True,
        description="Original grouped W4A16 kernel. Measured at 8.3 percent of "
        "L40S peak bandwidth because it runs a GEMV through tl.dot.",
    )(grouped_w4a16_linear)

    try:
        from engine.kernels.w4a16_gemv import grouped_w4a16_gemv
    except ImportError:
        return
    registry.register(
        W4A16_GROUPED,
        "triton_gemv",
        requires_cuda=True,
        description="Batch-1 grouped GEMV with K-contiguous weight reads. "
        "Measured at 51 percent of L40S peak, 5.95x the reference.",
    )(grouped_w4a16_gemv)


def _register_dense() -> None:
    # Every variant of this op takes (activations, packed, scales). The existing
    # w4a16_linear takes a W4A16Tensor instead, so it is adapted rather than
    # registered directly. One signature per op is what lets the equivalence
    # harness call the reference and a variant with identical arguments.
    import torch

    from engine.quant.triton_w4a16 import w4a16_linear
    from engine.quant.w4a16 import GROUP_SIZE, W4A16Tensor

    def dense_reference(
        activations: "torch.Tensor",
        packed_weights: "torch.Tensor",
        scales: "torch.Tensor",
    ) -> "torch.Tensor":
        encoded = W4A16Tensor(
            packed=packed_weights,
            scales=scales,
            original_shape=(packed_weights.shape[0], packed_weights.shape[1] * 2),
            original_dtype=torch.bfloat16,
            group_size=GROUP_SIZE,
        )
        return w4a16_linear(activations, encoded)

    registry.register(
        W4A16_DENSE,
        "reference",
        reference=True,
        requires_cuda=True,
        description="Original dense W4A16 GEMM. Measured at 10 percent of L40S "
        "peak bandwidth at decode shapes.",
    )(dense_reference)

    try:
        from engine.kernels.w4a16_dense_gemv import w4a16_dense_gemv
    except ImportError:
        return
    registry.register(
        W4A16_DENSE,
        "triton_gemv",
        requires_cuda=True,
        description="Batch-1 dense GEMV with K-contiguous weight reads.",
    )(w4a16_dense_gemv)

    # The same kernel with the launch configuration that won the isolated sweep
    # by a wide margin. Registered as its own variant so the two can be compared
    # inside ONE process against a baseline measured beside them. Comparing them
    # across separate benchmark runs is what produced a misleading answer: the
    # unchanged shipped configuration measured 109.71 and later 115.31 tok/s in
    # different containers, which is a bigger spread than the effect under test.
    try:
        from engine.kernels.w4a16_dense_gemv import (
            DENSE_GEMV_CONFIGS,
            _launch_w4a16_dense_gemv,
            _validate_inputs,
        )
    except ImportError:
        return

    by_name = {config.name: config for config in DENSE_GEMV_CONFIGS}
    narrow = by_name.get("n16_k64_s1_w4_st3")
    wide = by_name.get("n32_k128_s2_w8_st2")
    if narrow is None or wide is None:
        return

    def dense_narrow(activations, packed_weights, scales):
        rows, output_size, _ = _validate_inputs(activations, packed_weights, scales)
        if rows != 1:
            return w4a16_dense_gemv(activations, packed_weights, scales)
        config = wide if output_size >= 32768 else narrow
        return _launch_w4a16_dense_gemv(activations, packed_weights, scales, config)

    registry.register(
        W4A16_DENSE,
        "triton_gemv_narrow",
        requires_cuda=True,
        description="Dense GEMV using the narrow tile that won the isolated "
        "sweep. Kept registered so the choice can be re-tested in one process.",
    )(dense_narrow)


_register_grouped()
_register_dense()
