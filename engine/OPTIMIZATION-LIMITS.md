# How much faster can this get

An attempt to reach ten to forty times the current throughput, and what the
hardware actually permits. Everything here is measured on one NVIDIA L40S with
the selective INT4 artifact under a hard 32 GiB process cap, except the roofline
itself, which is computed from the architecture by `engine/roofline.py`.

Starting point: **113.83 tokens per second**, 8.78 ms per token.

## The ceiling, computed rather than guessed

Decode at batch one is weight streaming. Every generated token reads the
activated parameters once, so no kernel can beat bytes-per-token divided by
memory bandwidth.

| component | parameters | bytes each | MB per token | share |
|---|---:|---:|---:|---:|
| routed and shared experts | 1,656,225,792 | 0.5625 | 931.6 | 39.8% |
| **language modelling head, BF16** | 377,487,360 | 2.0 | **755.0** | **32.3%** |
| KDA q, k, v, o | 754,974,720 | 0.5625 | 424.7 | 18.2% |
| MLA q, kv_b, o | 194,510,848 | 0.5625 | 109.4 | 4.7% |
| KDA gates and convolutions | 34,242,560 | 2.0 | 68.5 | 2.9% |
| router gates | 15,335,424 | 2.0 | 30.7 | 1.3% |
| MLA kv_a | 9,289,728 | 2.0 | 18.6 | 0.8% |
| **total** | | | **2,338.4** | |

At the L40S's 864 GB/s that is **2.71 ms per token, or 369 tok/s**. We measure
8.78 ms, which is 30.8 percent of the roofline.

| target | tokens per second | bandwidth required | verdict |
|---|---:|---:|---|
| 2x | 228 | 533 GB/s | reachable |
| 3x | 342 | 799 GB/s | reachable, at 92 percent of the bus |
| 5x | 569 | 1,332 GB/s | **impossible at batch 1**, 1.5x the bus |
| 10x | 1,139 | 2,663 GB/s | **impossible**, 3.1x the bus |
| 40x | 4,556 | 10,653 GB/s | **impossible**, 12.3x the bus |

Kernel work alone cannot exceed 3.24x, and in practice less, because not every
kernel can reach peak. Anything beyond that must either read fewer bytes or emit
more than one token per read.

## Route one: read fewer bytes. Measured, and it barely helped

The vocabulary head is 755 MB, 32.3 percent of all traffic, and the quantisation
plan retains it in BF16 because output logits are quantisation sensitive.
Quantising it to INT4 removes 543 MB, 23 percent of the total.

| | tok/s |
|---|---:|
| baseline | 113.83 |
| INT4 vocabulary head | **116.88**, repeat 116.94 |

**Removing 23 percent of the memory traffic bought 2.7 percent.** The reason is
in the profile: the BF16 head runs through cuBLAS at 722 GB/s, 84 percent of
peak, which makes it the single most efficient kernel in the model. Quantising
it moves those bytes onto our dense GEMV at roughly 31 percent of peak, and the
saving is spent paying for a worse kernel.

Not adopted. It trades an unmeasured quality cost on the most sensitive tensor
in the model for 2.7 percent.

The general lesson corrects a naive reading of the roofline above:
bytes-per-token bounds throughput only when every kernel reaches the same
fraction of peak. Ours do not, so the cheapest bytes to remove are not the ones
that cost the most time.

In situ efficiency, which is what actually matters:

| kernel | time | bytes | achieved | of peak |
|---|---:|---:|---:|---:|
| grouped W4A16 GEMV | 3.019 ms | 932 MB | 309 GB/s | 36% |
| dense W4A16 GEMV | 1.980 ms | 534 MB | 270 GB/s | 31% |
| vocabulary head, BF16 | 1.045 ms | 755 MB | 722 GB/s | **84%** |

Bringing both Triton kernels to 80 percent of peak would save about 2.9 ms,
taking decode to roughly 170 tok/s. That is the realistic kernel ceiling, not
369.

## Route two: batching. Measured, and the architecture fights it

The same weight read serves every sequence in a batch, so aggregate throughput
should rise nearly linearly. It does not.

| batch | tok/s per sequence | tok/s aggregate | peak memory |
|---:|---:|---:|---:|
| 1 | 113.64 | 113.64 | 29.56 GiB |
| 2 | 55.45 | 110.91 | 27.84 GiB |
| 4 | 47.21 | 188.86 | 28.17 GiB |
| 8 | 37.73 | 301.82 | 29.07 GiB |
| 16 | 24.97 | **399.53** | 31.01 GiB |
| 32 | out of memory at the 32 GiB cap | | |

Aggregate throughput reaches **3.5x** and then the memory budget ends it, while
per-sequence latency degrades 4.5x. Batch 2 is actually worse in aggregate than
batch 1.

This is an architectural property worth stating clearly. For a conventional
transformer, batched decode amortises the weight read almost for free. Kimi
Linear carries a KDA recurrent state of 2 MiB per layer per sequence, which is
**40 MiB per sequence** across the twenty KDA layers, read and written every
step. At batch 16 that is 640 MiB of state traffic per step against 2,338 MB of
weights, so it stops being a rounding error and starts competing with the thing
batching was supposed to amortise. The weights amortise; the recurrent state
does not.

Batching also buys aggregate throughput, not latency. A user waiting on one
reply sees the per-sequence column, which gets worse.

## Route three: emit more than one token per weight read

This is the only route that improves single-stream latency past the roofline.
Draft several tokens cheaply, verify all of them in one forward pass that reads
the weights once, and keep the ones the model would have chosen anyway.

Under greedy decoding this is **exactly equivalent**, not approximately: every
accepted token is one the model itself would have emitted, and the first
rejected position is replaced by the model's own choice.
`tests/test_speculative.py` asserts that a 32-token speculative decode matches
ordinary greedy decode token for token.

`engine/speculative/` implements it: a draft source, the greedy accept rule, and
the piece that makes it hard here.

**Why it is harder for this model.** There is no key-value cache to truncate on
rejection. KDA state is recurrent and mutated in place, so rolling back requires
having saved it. `state_checkpoint.py` snapshots into preallocated buffers,
40 MiB per round, which is about 0.046 ms at full bandwidth against an 8.78 ms
token.

**How often the draft would be accepted.** Prompt lookup needs no draft model:
it searches the text so far for the most recent occurrence of the last few
tokens and proposes what followed. Acceptance is entirely workload dependent,
so `engine/speculative/acceptance_study.py` measures it on real text:

| corpus | k=4 n=3 | k=4 n=4 | k=8 n=3 | k=8 n=4 |
|---|---:|---:|---:|---:|
| **python source** | 1.96 | 1.99 | 3.02 | **3.12** |
| markdown prose | 0.92 | 0.88 | 1.14 | 1.11 |
| structured config | 0.89 | 0.77 | 1.24 | 1.11 |

On code, the workload a coding agent actually produces, the draft is accepted
**3.12 tokens per round**, or about 4.1 tokens per weight read. On prose it is
close to 1, meaning almost no benefit.

Two caveats push in opposite directions and do not cancel cleanly: byte tokens
fragment more than the real vocabulary, which understates acceptance, while
replaying fixed text assumes the model agrees with it, which overstates it. This
sizes the idea. It does not measure it.

**What blocks it from running today.** `KLinearModel` rejects a sequence length
above one when the state has fixed capacity, so verifying k tokens in one pass
needs the static decode path extended to accept k tokens. That is real surgery
on a CUDA-graph-captured path and it is not done.

## Where this lands

| route | measured | verdict |
|---|---|---|
| Better kernels | 36 and 31 percent of peak today | worth about 1.5x more, to roughly 170 tok/s |
| INT4 vocabulary head | +2.7 percent | not worth the quality cost |
| Batching | 3.5x aggregate, 4.5x worse latency | helps servers, not a single user |
| Speculative decoding | 3.12 tokens accepted on code | the only single-stream lever, needs engine work |

Stacking what is real: kernels to 170 tok/s, then speculation on code at perhaps
3x once the verification pass is paid for, gives roughly **500 tok/s on code**,
about 4.5x today and 14x the original engine.

**Ten times is not reachable on this card at batch one, and forty times is off
by an order of magnitude.** Reaching either requires different hardware, a
smaller model, or accepting a different output distribution. None of those is a
kernel optimisation, and saying otherwise would be arithmetic denial.
