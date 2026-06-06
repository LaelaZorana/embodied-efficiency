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

## 3. Next (needs CUDA)
Author the **fused INT4/INT8 dequant→GEMM Triton kernel** and benchmark on Colab T4 vs the fp16 CUDA-graph baseline (`bench.py --compile --dtype fp16`). Target: realized latency reduction approaching the ceilings above, on a roofline, with action rMSE reported.
