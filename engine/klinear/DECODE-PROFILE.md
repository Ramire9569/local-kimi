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
| `_grouped_w4a16_kernel` | 931.6 MB | 1.08 ms | 13.184 ms | ~71 GB/s, 8.2% of peak |
| `_w4a16_gemm_kernel` | 534 MB | 0.62 ms | 9.132 ms | ~58 GB/s, 6.8% of peak |

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

## After the fused kernels

The same profile re-run with `--kernels triton_gemv/triton_gemv`. This was
taken BEFORE the grouped kernel's launch configuration was swept end to end, so
it reads 109.12 tok/s where the shipped engine now measures 113.83. The
breakdown is what matters here, not the headline.

| | before | after |
|---|---:|---:|
| Graph replay | 28.11 ms, 35.58 tok/s | **9.16 ms, 109.12 tok/s** |
| GPU kernel time | 28.85 ms | 9.71 ms |
| Launches per token | 3,264 | 2,226 |
| Distinct kernels | 71 | 48 |

| category | ms per token | percent | launches |
|---|---:|---:|---:|
| other | 3.808 | 39.2% | 364 |
| grouped W4A16 GEMV | 3.019 | 31.1% | 78 |
| elementwise | 1.987 | 20.5% | 1,396 |
| reduction | 0.507 | 5.2% | 262 |
| index | 0.283 | 2.9% | 79 |
| copy | 0.096 | 1.0% | 40 |

The two kernels that were 77 percent of decode are now 51 percent of a total
that is three times smaller: 13.184 ms became 3.019, and 9.132 became 1.980.

`other` is now the largest bucket, and most of it is the dense GEMV at 1.980 ms
plus cuBLAS matrix-vector calls at 1.435 ms. The single largest of those is one
call costing 1.045 ms per token, which is the BF16 language modelling head: at
755 MB it is running near 720 GB/s, roughly 84 percent of the card, so there is
nothing to win there without quantising it and paying for that in quality.

The remaining headroom, in order:

1. The dense GEMV moves 534 MB in 1.980 ms, which is 270 GB/s or **31 percent
   of peak in situ**. Its isolated per-shape benchmarks read 11 to 17 percent;
   back-to-back calls inside a decode step overlap where a tight single-shape
   loop does not. A single 5.31 MB matrix at `BLOCK_N=64` produces 64 programs
   on a 142-SM card, so it is starved for parallelism rather than limited by
   memory.
2. 1,396 elementwise launches still cost 1.987 ms, which is a fifth of decode
   now that the total is smaller.
3. The grouped GEMV moves 931.6 MB in 3.019 ms, which is 309 GB/s or
   **36 percent of peak in situ**, so roughly 2.2x remains against an 80 percent
   target. Its isolated single-shape benchmark reads 51.7 percent. Both numbers
   are real and they measure different things; `engine/kernels/RESULTS.md` holds
   the isolated table and this file holds the in-situ one.

## What this profile does not establish

- It is one prompt, one batch, greedy, on one card. It is not a serving
  throughput number.
- Kernel attribution comes from the eager path. Graph replay may schedule the
  same kernels slightly differently.
- The category mapping is a name-fragment match, so `other` and `gemm` group
  several cuBLAS entry points together.
