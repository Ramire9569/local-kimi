# Measured against llama.cpp on the same card

This repository spent a long time saying that llama.cpp's reported throughput
and ours "are not a comparison" because the hardware and harnesses differed.
That was correct but evasive. Here is the comparison, run on one card.

## The result

NVIDIA L40S, Kimi-Linear-48B-A3B-Instruct, single stream, greedy, 64 generated
tokens.

| engine | quantisation | weights | tokens per second |
|---|---|---:|---:|
| local-kimi, fused kernels | INT4, group 32, BF16 scales | 26.83 GiB | 114.01 |
| **llama.cpp** | **Q4_K_M** | **28.00 GiB** | **165.69** |

**llama.cpp is 1.46 times faster.** We reach 0.69 of its throughput.

llama.cpp measured 165.69 tok/s with a standard deviation of 6.78 across three
repeats. Our figure reproduces to 0.02 percent within a container and ranges
113.77 to 114.03 across five independent containers.

Reproduce with:

```bash
modal run engine/modal_llamacpp_headtohead.py
modal run engine/modal_decode_bench.py
```

## Why this is a fair comparison, and where it is not

Controlled: the card, the driver, the model, the prompt length, the generated
token count, single stream, greedy decoding.

Not controlled: the quantisation formats differ. Ours is symmetric signed INT4
at group 32 with BF16 scales. Q4_K_M is a k-quant with a different block layout
and a different mix of per-tensor precisions.

That difference does not rescue the result. **Q4_K_M is the larger artifact**, at
28.00 GiB against our 26.83, so llama.cpp reads about 4 percent more weight bytes
per token and is still 46 percent faster. Decode at batch one is weight
streaming, so a bandwidth advantage would have favoured us. It did not.

The honest conclusion is that llama.cpp's CUDA kernels are more mature than
ours. That is unsurprising. They have years of contributors and we have one
day of profiling.

## What this does not undo

The 3.18x figure in `engine/kernels/RESULTS.md` compares this engine against
itself before and after the fused kernels. It is unaffected: the two W4A16
kernels really were running at 8 percent of the card's memory bandwidth, and
they really are at 51 percent now. Both statements can be true at once. We
improved a slow engine a great deal and it is still slower than the best one.

The profiling findings also stand on their own, and several of them are
transferable rather than specific to this codebase:

- CUDA graph capture had already removed launch overhead, so kernel count was
  not the bottleneck even though 3,264 launches per token looked alarming.
- Two kernels written as matrix-matrix products and used at decode as
  matrix-vector products held 77 percent of decode time.
- Three separate fusions lost end to end because these kernels are occupancy
  limited rather than launch or bandwidth limited.

## What this means for the project

This engine is not the fastest way to run Kimi-Linear on one GPU. llama.cpp is,
today, by a wide margin, and anyone can verify that in ten minutes.

What this repository offers that llama.cpp does not is the protocol adapter:
per-request client detection across Anthropic Messages, OpenAI Responses and
OpenAI Chat Completions, tool call translation in six formats, and reasoning
restored across turns. The engine is a research artifact and it is honest about
being one.

Publishing a speed claim against llama.cpp would have been refuted by the first
person who ran `llama-bench`. Publishing this instead costs nothing that was
true and buys the only thing that matters, which is that every other number
here can be believed.
