# Kernel layer for the VLA flow-matching action loop

The per-step bottleneck in flow-matching VLAs (π0 / GR00T) is the action expert,
launched once per integration step. This directory holds the benchmarks, the
kernels, and the evidence for what actually makes that loop cheap.

## Files
- `flow_expert.py`, reference π0-style action expert (cached VLM prefix, self+cross attn, MLP) + eager flow sampler. Device-agnostic.
- `bench.py`, benchmark + roofline harness. Honest baselines: `eager`, `compile_reduce_overhead`, and a manual CUDA graph.
- `cudagraph.py`, graph capture plus the correctness, stale-input, and no-leak checks.
- `triton_gemm.py` / `quant.py` / `prod_int4.py`, the low-bit path and its fidelity table.
- `compiler.py`, the budget-driven deploy compiler that searches the whole config space.
- `RESULTS.md`, every measured number, on a T4 and an L4.

## The finding (measured, not predicted)

At batch=1 the loop really is memory-bound, arithmetic intensity about 48 FLOP/byte
against a T4 roofline ridge of 203, and the per-step GEMMs are skinny because M is the
horizon of about 50, so each one streams a full `[d,d]` weight for very little work.
That reads like a case for cutting weight bytes, and the measurements say otherwise.

- **CUDA graphs are the latency win.** Manual graph capture of the sampler runs
  **5.9x over eager** (4.818 down to 0.819 ms/step at 10 steps on a T4) and it beats
  `torch.compile(reduce-overhead)`, 0.819 against 0.955.
- **Weight-only low-bit quantization buys no batch-1 latency**, and there are four
  experiments behind that. The hand-written int8/int4 Triton kernel came in about 5.5x
  slower than fp16 with a graph, a tensor-core plus autotune rewrite changed nothing, a
  512 to 4096 size sweep widened the gap instead of closing it, and then the production
  path, torchao's Marlin int4 on a supported L4, lost too at 1.2 to 1.6x slower than
  bf16 at every size. So this was never an implementation gap.
- **Why.** At batch=1 a step is about 36 tiny per-linear matmuls plus a lot of
  non-GEMM work (prefix cross-attention, norms, the flow integration), which is not the
  M=1 decode of a huge model that weight-only int4 was built for. In this regime cuBLAS
  plus graphs already win.
- **What low-bit is for here is memory footprint**, int8 1.97x smaller and int4 3.88x
  smaller at 0.0025 and 0.0423 action rMSE, which is a real lever for fitting a bigger
  policy on an edge device. It is not a latency lever, so don't quote the byte ceiling
  as a speed number.

Full tables, including the size sweep and the production int4 run, are in
[`RESULTS.md`](RESULTS.md).

## Run locally (correctness only, not headline numbers)
```bash
python3 kernel/bench.py --steps 10 4 --batch 1 --dtype fp32
```
Verifies the sampler is correct and the harness works. CPU/MPS launch behaviour does
not represent a GPU, so latency here is not the result.

## Run on a GPU (the real numbers)
The measured numbers come from a free Colab T4 and an L4, and the run takes about ten
minutes. Step by step in [`RUN_ON_T4.md`](../RUN_ON_T4.md), or straight from
[`colab.ipynb`](../colab.ipynb):

```python
# Colab cell, Runtime > Change runtime type > T4 GPU
!git clone https://github.com/LaelaZorana/embodied-efficiency.git
%cd embodied-efficiency
!pip -q install triton  # preinstalled on most Colab images
!python kernel/bench.py --steps 10 4 8 --batch 1 --dtype fp16 --compile --graph --peak T4
```

## How the numbers are gated
Nothing gets quoted as latency until the evals pass. Kernel numerics against a torch
reference, exact CUDA-graph replay, the stale-input case, and zero leaked bytes, all
across fp16/int8/int4. Quality is reported too, so the low-bit paths carry an
action rMSE against fp16 and are never called lossless.
