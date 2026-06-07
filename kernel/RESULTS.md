# Results

Status: **CPU evals green in CI; GPU evals run on a Tesla T4 (Colab), 2026-06-06.**
Env: Tesla T4, torch 2.11.0+cu128, triton 3.6.0. Action expert 25.5M params, 50.9 MB fp16 linears, batch=1.

## TL;DR (measured on T4)
- ✅ **Loop is memory-bound** on real HW (AI ≈ 48 FLOP/byte vs T4 ridge 203).
- ✅ **CUDA graphs are the win: 5.9× over eager** (4.818 → 0.819 ms/step), and the **manual capture beats `torch.compile(reduce-overhead)`** (0.819 vs 0.955). Composes with quant.
- ✅ **All correctness + no-leak evals pass** (kernel numerics, CUDA-graph exact replay, stale-input, zero memory leak) across fp16/int8/int4.
- ❌ **Weight-only low-bit gives NO batch-1 latency win** — confirmed across 4 experiments incl. a **production torchao/Marlin int4 kernel on a supported L4** (still 1.2–1.6× *slower* than bf16, §2c). Not a hand-kernel gap; the batch-1 VLA regime just isn't where weight-only int4 pays off. Low-bit's value here is memory *footprint*, not speed.

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

## 2b. Model-size sweep — does low-bit cross over to win at larger d? NO. (`sweep.py`)
Tested the "model too small" hypothesis. Tesla T4, batch=1, steps=10, +CUDA graph:
| d_model | fp16 ms/step | int8 ms/step | int8/fp16 |
|---|---|---|---|
| 512  | 1.091  | 4.535   | 4.16× |
| 1024 | 1.722  | 12.446  | 7.23× |
| 2048 | 4.846  | 46.040  | 9.50× |
| 4096 | 17.438 | 199.178 | 11.42× |

**Refuted.** Scaling *up* makes the custom kernel relatively *worse* (4.16× → 11.42×). At larger `d` the GEMMs are bigger and cuBLAS's optimization advantage (tiling/pipelining/split-K, years of tuning) dominates; a hand-written Triton kernel falls further behind, and the byte-savings can't overcome being far from the memory roofline.

## 2c. Production int4 (torchao / Marlin) on L4 — the decisive test
The hand kernel losing could have been an implementation gap. So I tested the *production* path: `torchao.quantize_(model, Int4WeightOnlyConfig(...))` (Marlin/tinygemm — cuBLAS-competitive, the canonical batch-1 weight-only int4) on a **Colab L4 (Ada sm_89)**, the mature/supported hardware. Eager... err, +CUDA-graph, batch=1, steps=10:

| d_model | bf16 ms/step | int4 (torchao) ms/step | int4/bf16 | rMSE |
|---|---|---|---|---|
| 512  | 0.452  | 0.736  | 1.63× | 0.045 |
| 2048 | 3.663  | 4.486  | 1.22× | 0.045 |
| 4096 | 13.481 | 19.862 | 1.47× | 0.050 |

**Even the production kernel loses at every size** (int4 1.2–1.6× *slower* than bf16). int4 is relatively closest at d=2048 (weight-read matters more there) but never wins.

(ZeroGPU was tried first for free — its Blackwell sm_120 is too new for torchao stable int4: `cutlass cannot initialize`. L4/Ada is the mature path, and it gives the clean answer above.)

## FINAL VERDICT (4 experiments)
1. hand fp32 kernel — lost.  2. hand tensor-core + autotune — no change.  3. size sweep — *worse* with d.  4. **production torchao int4 on supported L4 — also loses (1.2–1.6×).**

**Weight-only low-bit quantization does NOT give a batch-1 latency win for this VLA flow-matching sampler — not with a hand kernel, not with a production kernel, not at any size.** Why: at batch=1 the per-step is small skinny-GEMMs (M≈50) plus substantial non-GEMM work (prefix cross-attn, norms, flow integration); weight-only int4 is built for M=1 LLM *decode* of huge models, where weight reads dominate — a different regime. Here, bf16 tensor cores win.

**What low-bit IS for here: memory *footprint*** (int8 2× / int4 4× smaller, ≤5% action error — §4), e.g. fitting a bigger policy on an edge device. **The real *latency* win is CUDA graphs (§1, 5.9×).** Do not quote the byte-ceiling as a latency win.

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
