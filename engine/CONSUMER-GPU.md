# Which consumer GPUs this actually runs on

The engine decodes at 113.83 tokens per second on an NVIDIA L40S. That is a
datacenter card with 48 GB. This document is about the cards people own.

## The blocker is capacity, not speed

The selective INT4 artifact holds **26.83 GiB** of weights. A card needs roughly
2.5 GiB beyond that for activations, the KDA and MLA state pools, and the CUDA
context.

| card | memory | headroom after weights | runs today |
|---|---:|---:|:---:|
| RTX 4080 | 16 GB | -10.83 GiB | no |
| **RTX 3090** | 24 GB | -2.83 GiB | **no** |
| **RTX 4090** | 24 GB | -2.83 GiB | **no** |
| RTX 5090 | 32 GB | +5.17 GiB | yes |
| L40S | 48 GB | +21.17 GiB | yes, this is what was measured |

**The two cards enthusiasts actually own cannot load it.** Throughput on
hardware almost nobody has is not a useful result, so shrinking the artifact is
worth more right now than another twenty percent of decode speed.

## What it takes to fit 24 GB

There are 47.29 billion expert parameters: 26 mixture-of-experts layers, 257
experts each, three matrices per expert. They are the model. Everything else,
including both 755 MB vocabulary tensors, is under 2 GiB combined.

The current codec is symmetric signed INT4 at group 32 with BF16 scales, which
is 4.5 bits per parameter once the scales are counted.

| scheme | bits per parameter | experts | total | fits a 24 GB card |
|---|---:|---:|---:|:---:|
| INT4, group 32, BF16 scales (today) | 4.500 | 24.78 GiB | 26.79 GiB | no |
| INT4, group 64, FP8 scales | 4.125 | 22.71 GiB | 24.72 GiB | no |
| **INT3, group 32, BF16 scales** | 3.500 | 19.27 GiB | **21.28 GiB** | **yes** |
| INT3, group 64, FP8 scales | 3.125 | 17.21 GiB | 19.22 GiB | yes, comfortably |
| INT3, group 64, FP8, INT4 vocabulary | 3.125 | 17.21 GiB | 18.21 GiB | yes |

Widening the group or shrinking the scales is not enough on its own: it saves
about eight percent where twenty-two percent is needed. **Three-bit expert
weights are the requirement**, not a preference.

Note that even the most aggressive row does not fit a 16 GB card. An RTX 4080
would need the expert weights near two bits, or offloading, which is a different
project.

## Expected effect on speed

Decode at batch one is weight streaming, and INT3 reads 22 percent fewer bytes
than INT4. The measured in-situ efficiency of the grouped expert kernel is
36 percent of peak, so this path is bandwidth limited rather than compute
limited and fewer bytes should mean less time, even though unpacking three-bit
fields costs more shifts than four-bit fields.

That is a prediction, not a measurement. It is exactly the kind of prediction
that has been wrong twice on this project: quantising the vocabulary head
removed 23 percent of memory traffic and returned 2.7 percent of throughput,
because the tensor it removed was already being read at 84 percent of peak. The
expert path is not in that situation, but the number that settles it is an end
to end decode benchmark, not this paragraph.

## Expected effect on quality

INT3 has 8 code points against 16. This is a real loss and the point of
measuring it is to find out how large, not to argue it away.

For scale, `engine/accuracy/RESULTS.md` records what the existing INT4 already
costs against the BF16 checkpoint: 85.16 percent next-token top-1 agreement,
0.0555 nats of mean KL, and a 0.81 percent rise in perplexity. INT3 will be
worse. Whether it is acceptable is a decision to make against measured numbers.

## Status: the INT3 artifact exists

Built on one NVIDIA H100 from the original BF16 checkpoint, not from the INT4
artifact, so the two quantisation errors do not compound.

| | |
|---|---:|
| source BF16 tensor storage | 91.50 GiB |
| **W3A16 output** | **21.20 GiB** |
| compression against BF16 | 4.32x |
| tensors quantised | 20,150 |
| tensors retained in BF16 | 343 |
| fits a 24 GB card | **yes** |

The prediction above was 21.28 GiB and the artifact is 21.20 GiB, so the
arithmetic held.

Worst round-trip error, measured on the real tensors rather than random data,
is 0.2613 relative Frobenius on the shared expert projections. That is the same
tensor class that was worst under INT4, and it is roughly 2.3 times the INT4
error, which matches the isolated codec measurement.

**Not yet done:** the artifact has not been loaded, so throughput and output
quality on a 24 GB card are unmeasured. Fitting is proven; running is not.
