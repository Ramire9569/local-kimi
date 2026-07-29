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

Greedy, one stream, 17-token prompt, 64 generated tokens, five repeats, median
reported. Repeat timings varied by under 0.15 ms out of 560, and re-running the
same configuration in the same container reproduced to 0.02 percent.

| engine | tok/s | ms per token | against original |
|---|---:|---:|---:|
| original, before any of this work | 35.76 | 27.96 | 1.00x |
| KDA fusions only | 37.98 | 26.33 | 1.06x |
| KDA plus grouped GEMV | 63.10 | 15.85 | 1.76x |
| **KDA plus grouped and dense GEMV** | **113.83** | **8.78** | **3.18x** |

Read the middle rows carefully. The KDA fusions are applied whenever the decode
shape matches and are not switched by the W4A16 variant selector, so they are
present in every row including the one labelled reference. That is why the
baseline of the final sweep reads 37.98 rather than 35.76. The 3.18x figure is
against the original engine measured before any kernel in this directory
existed; the same sweep read against its own baseline gives 3.00x.

The split by contribution, against the original 35.76:

| change | tok/s after | share of the total gain |
|---|---:|---:|
| KDA preparation and recurrence fusion | 37.98 | 3% |
| grouped W4A16 GEMV | 63.10 | 32% |
| dense W4A16 GEMV | 113.83 | 65% |

Peak reserved memory fell from 29.56 GiB to 27.63 GiB, so the 32 GiB budget
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
| grouped GEMV, earlier config | 30/32, 93.8% | 1.56250 | 3.50e-3 nats |
| grouped and dense, earlier config | 31/32, 96.9% | 1.52734 | 3.55e-3 nats |
| **shipped, `n128_k32_s4`** | **31/32, 96.9%** | **1.50000** | **3.62e-3 nats** |
| the config it replaced | 31/32, 96.9% | 1.78125 | 6.73e-3 nats |

For scale, `engine/accuracy/RESULTS.md` records that the INT4 quantisation
itself costs 85.16 percent top-1 agreement and 0.0555 nats of mean KL against
the BF16 checkpoint. The kernel swap adds roughly **one tenth** of the
divergence that quantising the model already introduced.

That is the claim this work supports: the fused kernels are a much smaller
perturbation than the quantisation already applied, not that they are exact.

## Choosing the launch configuration by sweeping the engine

After two tuning decisions won an isolated benchmark and lost in the engine, the
grouped kernel's configuration was chosen differently: every candidate was
registered as its own variant and swept inside one decode process, with the
incumbent measured first and last to expose its own repeatability.

| variant, one container, 5 repeats each | tok/s | output matches incumbent |
|---|---:|:---:|
| incumbent, branch on estimated wave count | 109.38 | reference |
| pinned `n32_k64_s1` | 110.16 | yes |
| pinned `n64_k64_s1` | 103.67 | yes |
| pinned `n64_k64_s2` | 112.68 | no |
| **pinned `n128_k32_s4`** | **113.43** | no |
| pinned `n32_k128_s1` | 112.97 | no |
| incumbent again | 109.37 | yes |

The incumbent reproduces to 0.01 percent, so a 3.7 percent gap is real. The
incumbent branched on an estimate of how many waves the grid would fill, which
was never tested; one configuration for every shape beats it.

The three fastest change the output, because split-K and a different K block
change the order of the reduction. That is not disqualifying on its own, since
none of these kernels is bit-identical to the reference anyway, but it does mean
the winner needed its own equivalence measurement rather than an assumption.
That measurement came out in its favour: the same 31 of 32 top-1 choices, with
mean KL falling from 6.73e-3 to 3.62e-3 and the largest logit difference from
1.78 to 1.50. It is both faster and closer to the reference.

Confirmed after adopting it as the default: 113.83 tok/s in the ladder above,
and 114.01 then 114.03 in a separate two-run check.

## Two fusions that were built, measured, and left switched off

**Fusing the gate and up projections does not pay.** The expert path calls the
grouped kernel twice on the same activation with identical shapes, so folding
them into one launch looks obviously right: one activation load instead of two,
one launch instead of two, and the SwiGLU applied in registers.

It was built twice. The first attempt, `engine/kernels/moe_swiglu.py`, measured
1.133x but against the old kernel running at 8 percent of peak, so it was fusing
two slow kernels and the grouped GEMV beat it outright. The second attempt,
`grouped_w4a16_swiglu_gemv` in `engine/kernels/w4a16_gemv.py`, was rebuilt on the
fast tiling and registered as `w4a16_swiglu=fused`. Measured in one process
against the shipped path, with the baseline run twice for repeatability:

| variant, one container, 5 repeats each | tok/s |
|---|---:|
| reference everywhere | 37.96 |
| **shipped, two grouped calls plus silu** | **109.33** |
| fused gate and up | 107.64 |
| shipped again | 109.34 |

The shipped path reproduces to 0.01 percent and the fusion is **1.5 percent
slower**. The reason is visible in the kernel: two accumulators instead of one
doubles register pressure and cuts occupancy, and the activation tile it avoids
re-reading is only about 4.6 KB. The launch it removes was never the cost.

Both remain in the repository and the second remains registered, so the result
is reproducible with `KIMI_KERNELS=w4a16_swiglu=fused` and nobody has to rebuild
it to rediscover that it loses.

This is the second tuning idea that won in isolation and lost in the engine. The
first was the narrow dense tile below. Neither was a bad idea; both were tested
against the wrong thing before they were tested against the engine.

## One tuning result that did not survive the engine

The dense GEMV reaches only 11 to 17 percent of peak bandwidth at the projection
shapes. The diagnosis was parallelism: a single 5.31 MB matrix at `BLOCK_N=64`
produces 64 programs on a 142-SM card. An isolated sweep confirmed it, and a
narrower tile with no split-K won clearly:

| shape | previous choice | `n16_k64_s1` |
|---|---:|---:|
| N=4096, KDA q/k/v | 15.5% of peak | **24.8%**, 1.60x faster |
| N=6144, MLA q_proj | 24.2% | **32.4%**, 1.34x faster |
| N=2304, o_proj | 15.9% | **17.0%**, 1.07x faster |

Switching the selector to it measured slower end to end, so it was reverted. The
first evidence for that was two numbers taken in separate containers, 105.44
against 109.71, which is not sound: re-running the unchanged shipped
configuration in a fresh container later measured 115.31 tok/s, a spread larger
than the effect being tested.

The sound version registers both configurations as variants and sweeps them in
one process, against a baseline measured beside them, with the shipped
configuration measured twice to expose its own repeatability:

| variant, one container, 5 repeats each | tok/s |
|---|---:|
| reference | 37.96 |
| **shipped `n64_k64_s1` and `n32_k64_s2`** | **109.51** |
| `n16_k64_s1` everywhere | 100.87 |
| shipped configuration again | 109.49 |

Within one container the same configuration reproduces to **0.02 percent**, and
the narrow tile is **7.9 percent slower**. The revert was right, and now it is
evidenced.

The isolated benchmark runs one shape in a tight loop where 256 small programs
fill the card. A decode step issues about 104 such calls back to back, and there
the narrower tile costs more in per-kernel scheduling than it wins in occupancy.

Two lessons, the second learned by getting it wrong here:

1. A kernel benchmark is a hypothesis. The end to end measurement is the gate.
2. An end to end measurement in a different container is noisy too, by about 5
   percent, which is larger than most tuning effects. Comparisons worth acting
   on belong in one process next to their baseline.

This is why `engine/modal_decode_bench.py` loads the checkpoint once and sweeps
every variant inside a single container rather than comparing across runs, and
why `triton_gemv_narrow` stays registered rather than deleted.

## Known limits

- The dense GEMV reaches only 11 to 17 percent of peak bandwidth. A single
  5.31 MB matrix at `BLOCK_N=64` produces 64 programs on a 142-SM card, so it is
  starved for parallelism. Narrower tiles without split-K were added to the
  candidate list after the first sweep but the ceiling here is not solved.
- Every number is one card, one stream, one prompt, greedy. None of this is a
  serving throughput claim.
- Kernel attribution in the profile comes from eager decode, because graph
  replay appears to the profiler as a single opaque launch.
