# embodied-efficiency

[![ci](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml/badge.svg)](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml)

**Getting embodied and multimodal models to run on the robot, not just in the demo.**

Kernels, quantization, and a runtime safety layer for vision-language-action (VLA) models, so they run on the robot inside a real latency and power budget.

> A VLA can fold laundry in a lab demo today. Then you put it on the robot and it stalls, not because it can't do the task, but because it can't do it fast enough. It thinks 3 to 5 times a second, and the arm needs 50 to 100. That gap is the whole problem, and almost none of it is about whether the model is smart.

📄 **Read the thesis →** [THESIS.md](THESIS.md)

## The problem

The question at the frontier of embodied AI isn't "can the model do it" anymore. It's whether it runs on this robot, inside this latency and power budget, cheaply enough to put a thousand of them on a floor. The candidate levers, quantization and CUDA graphs and action chunking and speculative decoding, are scattered across papers, and which ones actually pay off depends on the workload. So this repo measures them on one concrete batch-of-one VLA flow-matching sampler, and reports what wins (CUDA graphs, almost 6×) and what doesn't (weight-only low-bit, across four experiments), as code you can run rather than claims you have to trust.

## What's here

| Component | What it does | Status |
|---|---|---|
| **Thesis** | Why efficiency, not capability, is the bottleneck | ✅ [THESIS.md](THESIS.md) |
| **CUDA-graph sampler** | manual graph capture of the N-step flow loop | ✅ **5.9× measured on T4** ([RESULTS.md](kernel/RESULTS.md)) |
| **Low-bit weight quant** | hand kernel + production torchao/Marlin int4 | 🔬 investigated → **negative** (slower than bf16 even on a supported L4; see findings) |
| **Autotuning deploy-compiler** | budget in → Pareto frontier + best config out; searches precision × steps × cuda-graph | ✅ **v0** — [kernel/compiler.py](kernel/compiler.py) |
| **Runtime trust layer** | statistically certified non-regression + OOD abstention + intervention logging | 🔜 planned |

## Findings (measured on T4 + L4 — full data in [kernel/RESULTS.md](kernel/RESULTS.md))

- ✅ **CUDA graphs are the win.** Capturing the sampler as one replayable graph takes it from 4.82 to 0.82 ms/step on a T4, almost six times faster, beating `torch.compile`, and the replay matches the original exactly with nothing leaking across fp16/int8/int4.
- 🔬 **Weight-only low-bit doesn't buy latency here, and I checked it four ways.** The hand-written int8/int4 Triton kernel lost to cuBLAS, a tensor-core rewrite with auto-tuning didn't move it, a 512→4096 size sweep made it worse, and even the production kernel (torchao/Marlin int4) on a supported L4 came in 1.2 to 1.6× slower than bf16. It was never the implementation: a batch-of-one VLA sampler is small matrices plus a lot of non-matmul work, which isn't the regime weight-only int4 is built for (one-token-at-a-time decode of a huge LLM). Low-bit here is a footprint lever (int8 2×, int4 4× smaller, under 5% action error), not a speed one.
- Every number is gated by correctness, stale-input, and no-leak checks. The win and the failure are reported the same way.

## Deploy-compiler (v0)

The findings above aren't just written down, they're built into a budget-driven autotuner ([`kernel/compiler.py`](kernel/compiler.py)). You give it a deployment budget, a latency ceiling or a memory ceiling, and it searches precision × steps × cuda-graph, scores every config on latency, weight-footprint, and action-fidelity, then hands back the Pareto frontier plus the best config that fits. Two budgets, two answers:

```
budget = minimize LATENCY,   rMSE ≤ 0.05  →  bf16 + CUDA graph   (latency-optimal)
budget = minimize FOOTPRINT,  rMSE ≤ 0.05  →  int4               (~7× smaller weights, rMSE 0.04)
```

It carries what the experiments proved: CUDA graphs when you want speed, int4 when you want a smaller model. Footprint and fidelity are computed anywhere; the latency numbers are real on CUDA.

## Reproduce
- **CPU evals (free, automatic):** every push runs `kernel/selfcheck.py` via [`ci.yml`](.github/workflows/ci.yml) — quant fidelity, kernel fallback numerics, no-leak determinism. Run locally with `python3 kernel/selfcheck.py`.
- **GPU latency + evals:** [`RUN_ON_T4.md`](RUN_ON_T4.md) — ~10 min on a free Colab T4 via [`colab.ipynb`](colab.ipynb).
- **Deploy-compiler:** `python3 kernel/compiler.py` — Pareto report + budget picks (footprint and fidelity real anywhere; latency real on CUDA).

## Principles

- **Measure first.** Benchmark against the strong baseline (`torch.compile` with CUDA graphs), never a strawman.
- **Honest claims.** Statistically certified non-regression on a declared distribution, never the word "provable."
- **Vendor-neutral and reproducible**, on the hardware robots actually carry.

## Background

This grew out of golfing neural networks, building the smallest ONNX graph that still solves an ARC-AGI task exactly, scored on a hard parameter and memory budget. Fitting a VLA under a latency budget is the same search: find where precision is free, and where it's load-bearing.

---

*Early-stage and public on purpose. If you're fighting the same deploy gap, or working it from the model side, open an issue or reach out.*
