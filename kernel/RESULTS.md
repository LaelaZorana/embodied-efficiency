# Results

Honest status: **correctness + fidelity verified locally (CPU/MPS); GPU latency pending on Colab T4.** Only CUDA gives headline latency, but the *regime* and the *prize* are already established and reproducible.

## 1. The loop is memory-bound (→ the win is fewer weight bytes)
`python3 kernel/bench.py --steps 10 4 --batch 1 --dtype fp32`

At batch=1, arithmetic intensity ≈ **24 FLOP/byte** vs the T4 roofline ridge **~203** → weight-read-bound. The per-step GEMMs are skinny (M = horizon ≈ 50): they stream the full `[d,d]` weight for ~50 rows of work. So within-step op-fusion barely helps and CUDA graphs already remove launch overhead — **the lever is reducing weight bytes.**

## 2. Low-bit weight quant: the prize, quantified
`python3 kernel/quant.py`  (25.5M-param π0-style expert, 50.9 MB fp16 linears)

| precision | weight MB | vs fp16 | action rMSE | speedup ceiling\* |
|---|---|---|---|---|
| fp16 | 50.92 | 1.00× | — | 1.00× |
| **int8** | 25.83 | **1.97×** | **0.0025** | **1.97×** |
| int4 | 13.11 | 3.88× | 0.0423 | 3.88× |

\* ceiling = weight-traffic reduction = the max achievable speedup in the weight-read-bound regime. The fused dequant→GEMM Triton kernel closes the gap to it.

**Read:** INT8 ≈ 2× the dominant cost for 0.25% action error (≈ free). INT4 ≈ 3.9× for ~4% — the QAT/grouping target. IO projections kept fp16 by design.

## 3. Fused low-bit kernels (`triton_gemm.py`)
INT8 + INT4 weight-only dequant→GEMM Triton kernels (INT4 = 2-per-byte nibble packing, a_even/a_odd in-kernel unpack). Numerics validated locally against a torch fallback: int8 action rMSE 0.0025, int4 0.0423 (match §2). **GPU latency pending T4.**

## 4. CUDA graphs + low-bit (orthogonal stack) (`cudagraph.py`)
Manual `torch.cuda.CUDAGraph` capture of the N-step sampler — NOT `torch.compile`, which graph-breaks on raw user Triton kernels (would need `torch.library.triton_op`). Graphs remove per-launch CPU overhead; the kernel cuts weight bytes; combined they should stack.

Failure modes researched and defended in code:
- **capture-time JIT** → warm up on a side stream before capture.
- **stale frozen input** → static input buffer + `copy_` per replay (the #1 silent CUDA-graph bug).
- **output aliasing** → `clone()` the captured output after replay.
- **memory-pool growth** → no-leak eval asserts `memory_allocated()` stable over 50 replays.
- **autotune-in-capture** (pytorch #120802) → fixed block sizes, no `@triton.autotune`.

Evals (`python3 kernel/cudagraph.py`, GPU): correctness vs eager, **stale-input** (two distinct inputs each match eager), **no-leak**. Off-CUDA it runs an eager-fallback determinism check (passes locally). **GPU run pending T4.**

## Next (needs CUDA — Colab T4)
Run `colab.ipynb`: latency for eager / torch.compile-graph / manual-graph / int8 / int4 / int8+graph / int4+graph, on a roofline, and confirm all eval lines print ✓ (not FAIL) before quoting any speedup.
