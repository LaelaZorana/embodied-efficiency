# Results

Status: **CPU evals green in CI; GPU evals run on a Tesla T4 (Colab), 2026-06-06.**
Env: Tesla T4, torch 2.11.0+cu128, triton 3.6.0. Action expert 25.5M params, 50.9 MB fp16 linears, batch=1.

## TL;DR (measured on T4)
- ✅ **Loop is memory-bound** on real HW (AI ≈ 48 FLOP/byte vs T4 ridge 203).
- ✅ **CUDA graphs are the win: 5.9× over eager** (4.818 → 0.819 ms/step), and the **manual capture beats `torch.compile(reduce-overhead)`** (0.819 vs 0.955). Composes with quant.
- ✅ **All correctness + no-leak evals pass** (kernel numerics, CUDA-graph exact replay, stale-input, zero memory leak) across fp16/int8/int4.
- ❌ **The low-bit Triton GEMM is *slower* than fp16 cuBLAS on T4** — the 1.97×/3.88× weight-byte *ceiling is NOT realized*. Honest negative; root cause + fix below.

## 1. fp16 baseline — eager vs torch.compile(graphs) vs manual CUDA graph (ms/step)
| steps | eager | compile_reduce_overhead | **manual graph** |
|---|---|---|---|
| 10 | 4.818 | 0.955 | **0.819** |
| 4  | 3.600 | 0.790 | 0.821 |
| 8  | 3.624 | 0.792 | 0.819 |

Manual CUDA graph: **5.9× over eager** at steps=10, and beats torch.compile. This is the real, validated payoff.

## 2. Low-bit fused kernels (+ CUDA graph), ms/step @ steps=10
| variant | eager | +graph |
|---|---|---|
| fp16 | 4.818 | **0.819** |
| int8 | 8.183 | 4.456 |
| int4 | 6.483 | 6.228 |

**The honest negative:** the INT8/INT4 Triton kernel is ~5× slower than fp16+graph. Graphs still help it (int8 8.18 → 4.46), but the weight-byte reduction does **not** translate to latency on T4.

**Root cause:** the kernel was written correctness-first — fp32 accumulate, `allow_tf32=False`, fixed un-tuned 64³ blocks → **no tensor cores**. cuBLAS fp16 (tensor cores) is so efficient at these small shapes (M=50, d=512) that the byte savings are swamped by the kernel's ~5–8× lower compute efficiency. The 1.97×/3.88× ceiling is a *memory-traffic* bound; realizing it needs a kernel that's otherwise competitive with cuBLAS.

**Fix / next:** (a) tensor-core path — fp16/bf16 `tl.dot` + tf32 accumulate, `@triton.autotune` over tiles, software pipelining; (b) re-test where weight-bandwidth actually dominates — Jetson Orin (edge, the real target), larger `d`, or batched sampling. Only then should the low-bit win appear. **Do not quote the byte-ceiling as achieved.**

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
