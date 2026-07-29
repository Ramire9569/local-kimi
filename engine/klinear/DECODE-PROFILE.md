# Where a decode token actually goes

Measured by `engine/modal_profile.py` on one NVIDIA L40S with the selective
INT4 artifact, the process hard-capped at 32 GiB, an 8-token prompt, and greedy
decoding. Kernel attribution comes from `torch.profiler` over eager decode
steps because CUDA graph replay appears to the profiler as a single opaque
launch. Throughput comes from graph replay.

## Headline

| | ms per token | tokens per second |
|---|---:|---:|
| Eager decode | 50.29 | 19.88 |
| **CUDA graph replay** | **28.11** | **35.58** |
| Actual GPU kernel time | 28.85 | |

3,264 kernel launches per token across 71 distinct kernels.

The graph number reproduces the 35.71 tok/s recorded in
`DECODE-BENCHMARK.md` on the same hardware, so the two runs agree.

## The finding

Graph replay costs 28.11 ms and the kernels themselves account for 28.85 ms of
GPU time. **Launch overhead is already gone.** The eager path pays 21.4 ms of
it, and CUDA graph capture removes essentially all of that. Anything that only
reduces the number of launches cannot help much from here, because the GPU is
busy for almost the entire wall clock.

| category | ms per token | percent | launches |
|---|---:|---:|---:|
| `_grouped_w4a16_kernel` (MoE experts) | 13.184 | 45.7% | 78 |
| `_w4a16_gemm_kernel` (dense projections) | 9.132 | 31.7% | 104 |
| all elementwise | 3.385 | 11.7% | 2,236 |
| other | 1.838 | 6.4% | 360 |
| reduction | 0.766 | 2.7% | 300 |
| index | 0.284 | 1.0% | 79 |
| copy | 0.247 | 0.9% | 100 |
| softmax | 0.014 | 0.0% | 7 |

**Two W4A16 kernels are 77.4 percent of decode time.** The 2,236 elementwise
launches, which look alarming, cost 11.7 percent between them.

## How far off the hardware these two kernels are

The L40S moves 864 GB/s.

| kernel | weight bytes read per token | roofline | measured | achieved |
|---|---:|---:|---:|---:|
| `_grouped_w4a16_kernel` | ~931 MB | 1.08 ms | 13.184 ms | ~71 GB/s, 8% of peak |
| `_w4a16_gemm_kernel` | ~762 MB | 0.88 ms | 9.132 ms | ~83 GB/s, 10% of peak |

These kernels are not memory bound. They are shaped wrong.

Both were written as GEMMs and used at decode as GEMVs:

1. They call `tl.dot` with a `[BLOCK_M, BLOCK_N]` accumulator. At decode there
   is one token, so `BLOCK_M` is 16 and fifteen sixteenths of the tensor-core
   work is discarded. `_grouped_w4a16_kernel` makes this explicit: it
   broadcasts the single activation row with `offsets_m[:, None] * 0` and then
   collapses the identical rows with `tl.max(accumulator, axis=0)`. The wide
   accumulator also caps occupancy.
2. Packed weights are stored `[..., N, K/2]`, and both kernels index the tile
   with N on the fastest-varying axis. Consecutive lanes in a warp therefore
   land on addresses `K/2` bytes apart and each read pulls a separate cache
   line.

## What this implies for the target

At 8 to 10 percent of achievable bandwidth there is a large amount of headroom
in the two kernels that dominate the profile. Holding everything else fixed:

| if the two W4A16 kernels reach | their 22.32 ms becomes | total | tokens per second |
|---|---:|---:|---:|
| 30% of peak | 6.20 ms | 12.73 ms | 78.6 |
| 50% of peak | 3.72 ms | 10.25 ms | 97.6 |

Those are arithmetic projections from the measured split, not measurements.
They set the direction of the work, not a claim about the result.

## What this profile does not establish

- It is one prompt, one batch, greedy, on one card. It is not a serving
  throughput number.
- Kernel attribution comes from the eager path. Graph replay may schedule the
  same kernels slightly differently.
- The category mapping is a name-fragment match, so `other` and `gemm` group
  several cuBLAS entry points together.
