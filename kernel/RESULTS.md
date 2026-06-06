# Results

Status: **CPU evals green in CI; GPU evals run on a Tesla T4 (Colab), 2026-06-06.**
Env: Tesla T4, torch 2.11.0+cu128, triton 3.6.0. Action expert 25.5M params, 50.9 MB fp16 linears, batch=1.

## TL;DR (measured on T4)
- ✅ **Loop is memory-bound** on real HW (AI ≈ 48 FLOP/byte vs T4 ridge 203).
- ✅ **CUDA graphs are the win: 5.9× over eager** (4.818 → 0.819 ms/step), and the **manual capture beats `torch.compile(reduce-overhead)`** (0.819 vs 0.955). Composes with quant.
- ✅ **All correctness + no-leak evals pass** (kernel numerics, CUDA-graph exact replay, stale-input, zero memory leak) across fp16/int8/int4.
- ❌ **The low-bit Triton GEMM is *slower* than fp16 cuBLAS on T4**, and a tensor-core + autotune rewrite (v2) did **not** fix it. The 1.97×/3.88× byte-ceiling is NOT a latency win at batch=1; corrected diagnosis in §2 (it's a tiny-matmul/op-overhead regime, not weight-byte-bound).

## 1. fp16 baseline — eager vs torch.compile(graphs) vs manual CUDA graph (ms/step)
| steps | eager | compile_reduce_overhead | **manual graph** |
|---|---|---|---|
| 10 | 4.818 | 0.955 | **0.819** |
| 4  | 3.600 | 0.790 | 0.821 |
| 8  | 3.624 | 0.792 | 0.819 |

Manual CUDA graph: **5.9× over eager** at steps=10, and beats torch.compile. This is the real, validated payoff.

## 2. Low-bit fused kernels (+ CUDA graph), ms/step @ steps=10
| variant | +graph (v1, fp32/no-TC) | +graph (v2, tensor-core+autotune) |
|---|---|---|
| **fp16 (cuBLAS)** | **0.819** | **0.817** |
| int8 | 4.456 | 4.532 |
| int4 | 6.228 | 6.293 |

**The honest negative — and the fix that didn't work.** The low-bit Triton kernel is ~5.5× slower than fp16+graph. I hypothesized it was the fp32/no-tensor-core path, rewrote it to fp16 `tl.dot` (tensor cores) + `@triton.autotune` (v2), re-ran on T4 → **no change** (int8 4.46→4.53, int4 6.23→6.29).

**Corrected diagnosis:** the bottleneck is *not* dot compute or weight bytes. At batch=1 each step is ~36 *tiny* per-linear matmuls (M=50); the work is dominated by per-op launch/overhead across many small ops, where cuBLAS+graphs already win and a custom per-linear Triton kernel can't compete — regardless of tensor cores or low-bit. **Weight-only low-bit quant via a custom GEMM is the wrong lever for batch-1 latency on a fast datacenter GPU.**

**Where low-bit actually pays off (not measured here):** memory *capacity* (fit a bigger model / smaller footprint — int8 2×, int4 4× smaller, fidelity intact per §4), bandwidth-starved **edge HW (Jetson Orin)**, or larger `d`/batch where weights dominate. **Do not quote the byte-ceiling as a latency win.** The real latency win on T4 is CUDA graphs (§1).

## 3. Correctness + no-leak (T4) — all ✓
```
Triton kernel correctness ✓        int8/int4 triton vs torch-dequant max abs err = 1.95e-3
fp16 graph: max_err=0.000e+00  leaked=0B  -> OK
int8 graph: max_err=0.000e+00  leaked=0B  -> OK
int4 graph: max_err=0.000e+00  leaked=0B  -> OK
CUDA-graph correctness + stale-input + no-leak ✓
```
The researched failure-mode defenses held: exact replay (max_err 0), **zero leak**, stale-input safe.

## 4. Weight-only quant fidelity (the prize, quantified)
| precision | weight MB | vs fp16 | action rMSE | byte-ceiling |
|---|---|---|---|---|
| fp16 | 50.92 | 1.00× | — | 1.00× |
| int8 | 25.83 | 1.97× | 0.0025 | 1.97× |
| int4 | 13.11 | 3.88× | 0.0423 | 3.88× |

Fidelity is excellent (int8 0.25%, int4 4.2% action error). The *ceiling* is real; the kernel doesn't yet *reach* it (§2).
