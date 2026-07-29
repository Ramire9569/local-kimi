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
    from engine.quant.triton_w4a16 import w4a16_linear

    registry.register(
        W4A16_DENSE,
        "reference",
        reference=True,
        requires_cuda=True,
        description="Original dense W4A16 GEMM. Measured at 10 percent of L40S "
        "peak bandwidth at decode shapes.",
    )(w4a16_linear)

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


_register_grouped()
_register_dense()
