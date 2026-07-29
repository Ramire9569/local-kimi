# Fused decode kernels, measured

Every number here was produced on one NVIDIA L40S with the selective INT4
artifact and the process hard-capped at 32 GiB. Runners are
`engine/modal_profile.py`, `engine/modal_kbench.py`,
`engine/modal_decode_bench.py` and `engine/modal_kernel_equivalence.py`.

## Where the time went before any of this

`engine/klinear/DECODE-PROFILE.md` has the full breakdown. The short version:
graph replay cost 28.11 ms per token against 28.85 ms of actual GPU kernel
time, so CUDA graph capture had already removed launch overhead and the GPU was
busy for nearly the whole wall clock. Two W4A16 kernels held 77.4 percent of
that time while running at 8 to 10 percent of the card's memory bandwidth.

Both had been written as GEMMs and were being used at decode as GEMVs:

1. `tl.dot` with a `[BLOCK_M, BLOCK_N]` accumulator. With one token, fifteen
   sixteenths of the tensor-core work was discarded. `_grouped_w4a16_kernel`
   made this explicit, broadcasting the single activation row with
   `offsets_m[:, None] * 0` and collapsing the identical rows afterwards with
   `tl.max(accumulator, axis=0)`.
2. Packed weights stored `[..., N, K/2]` but indexed with N on the
   fastest-varying axis, so lanes in a warp landed `K/2` bytes apart and every
   read pulled its own cache line.

## Kernel level

| kernel | shape | before | after | speedup | peak bandwidth reached |
|---|---|---:|---:|---:|---:|
| grouped W4A16, w1 and w3 | N=1024, K=2304 | 0.1768 ms | 0.0297 ms | **5.95x** | 8.3% to **51.7%** |
| grouped W4A16, w2 | N=2304, K=1024 | 0.1437 ms | 0.0301 ms | **4.77x** | 10.2% to **51.2%** |
| dense W4A16, KDA q/k/v | N=4096, K=2304 | 0.0656 ms | 0.0553 ms | 1.19x | 9.4% to 11.1% |
| dense W4A16, o_proj | N=2304, K=4096 | 0.1516 ms | 0.0628 ms | 2.41x | 4.1% to 9.8% |
| KDA recurrence | 32 heads, 128x128 state | 0.0781 ms | 0.0315 ms | **2.48x** | |
| KDA preparation | 4096 channels | 0.3623 ms | 0.0972 ms | **3.73x** | |

The KDA preparation fusion also collapses **47 launches per layer into 2**,
which is 940 to 40 across the twenty KDA layers.

## End to end

Greedy, one stream, 64 generated tokens, three repeats, median reported.

| kernels | tok/s | ms per token | speedup |
|---|---:|---:|---:|
| reference everywhere | 35.76 | 27.96 | 1.00x |
| grouped GEMV only | 55.96 | 17.87 | 1.57x |
| grouped and dense GEMV | 92.53 | 10.81 | **2.59x** |

Peak reserved memory fell from 29.56 GiB to 27.61 GiB, so the 32 GiB budget
holds with more room than before rather than less.

## Equivalence, stated honestly

**These kernels do not produce bit-identical output.** An earlier version of the
decode benchmark reported that they did, and it was wrong: the prompt was a
repeated filler token, the model answered with the same token 64 times, and
three variants agreeing on a constant proved nothing. The benchmark now refuses
to run if the reference produces fewer than three distinct tokens.

With a real prompt and free-running greedy decode, the fused path matches the
reference for the first 24 to 26 tokens and then diverges. That is expected and
is not evidence of a broken kernel: a different reduction order changes the
result by a tiny amount, and greedy decoding turns the first flipped argmax into
a different continuation, after which everything is conditioned differently.

The measurement that isolates kernel error from that compounding is teacher
forcing: feed every variant the same fixed token sequence and compare the logits
produced at each step.

| kernels | top-1 agreement | max absolute logit difference | mean KL |
|---|---:|---:|---:|
| reference against itself | 32/32, 100% | 0.00000 | 0.0 |
| grouped GEMV | 30/32, 93.8% | 1.56250 | 3.50e-3 nats |
| grouped and dense GEMV | 31/32, 96.9% | 1.52734 | 3.55e-3 nats |

For scale, `engine/accuracy/RESULTS.md` records that the INT4 quantisation
itself costs 85.16 percent top-1 agreement and 0.0555 nats of mean KL against
the BF16 checkpoint. The kernel swap adds roughly **one tenth** of the
divergence that quantising the model already introduced.

That is the claim this work supports: the fused kernels are a much smaller
perturbation than the quantisation already applied, not that they are exact.

## What was built and not used

The fused SwiGLU kernel in `engine/kernels/moe_swiglu.py` measured 1.133x
against the original three-call path. It is not wired in. Its baseline was the
old kernel running at 8 percent of peak, so it fused two slow kernels together,
and the grouped GEMV makes the same calls 5.95x faster instead. It stays in the
repository as a registered variant so the measurement is reproducible, and it
would need rebuilding on the GEMV tiling to be worth enabling.

## Known limits

- The dense GEMV reaches only 11 to 17 percent of peak bandwidth. A single
  5.31 MB matrix at `BLOCK_N=64` produces 64 programs on a 142-SM card, so it is
  starved for parallelism. Narrower tiles without split-K were added to the
  candidate list after the first sweep but the ceiling here is not solved.
- Every number is one card, one stream, one prompt, greedy. None of this is a
  serving throughput claim.
- Kernel attribution in the profile comes from eager decode, because graph
  replay appears to the profiler as a single opaque launch.
