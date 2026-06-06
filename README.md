# embodied-efficiency

[![ci](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml/badge.svg)](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml)

**Making frontier embodied & multimodal models cheap to run and safe to deploy.**

Kernels, quantization, and the runtime trust layer for vision-language-action (VLA) models — so they run *on the robot*, under a latency and power budget, and can be deployed **and** governed.

> A VLA can fold laundry in a lab demo today. Then you put it on the robot and it falls apart — not because it can't do the task, but because it can't do it *fast enough*. End-to-end inference runs at 3–5 Hz; a robot arm needs 50–100 Hz. That's not a capability gap. It's a **deploy gap.**

📄 **Read the thesis →** [THESIS.md](THESIS.md)

## The problem

The frontier of embodied AI is no longer "can the model do it." It's: *can it run on this robot, under this latency budget, inside this power envelope, cheaply enough to deploy at scale?* The levers exist — INT8 holds ~97% of task success, INT4 buys ~2.5× speedup, action chunking and speculative decoding cut latency — but they live scattered across papers, not as reproducible tooling on the hardware robots actually carry (Jetson Orin/Thor, single-digit-watt silicon). This repo closes that gap.

## What's here

| Component | What it does | Status |
|---|---|---|
| **Thesis** | Why efficiency, not capability, is the bottleneck | ✅ [THESIS.md](THESIS.md) |
| **CUDA-graph sampler** | manual graph capture of the N-step flow loop | ✅ **5.9× measured on T4** ([RESULTS.md](kernel/RESULTS.md)) |
| **Low-bit fused kernel** | hand-written INT8/INT4 Triton dequant→GEMM | 🔬 investigated → **negative** (loses to cuBLAS; see findings) |
| **Autotuning deploy-compiler** | budget in (*"100 Hz on an Orin, <15 W"*) → deployable engine out | 📋 in design |
| **Runtime trust layer** | statistically certified non-regression + OOD abstention + intervention logging | 🔜 planned |

## Findings (measured on a Tesla T4 — full data in [kernel/RESULTS.md](kernel/RESULTS.md))

- ✅ **CUDA-graph capture of the sampler: 5.9× over eager** (4.82 → 0.82 ms/step), beats `torch.compile`, with exact replay + **zero memory leak** across fp16/int8/int4.
- 🔬 **Weight-only INT8/INT4 via a hand-written Triton kernel — a rigorously characterized negative.** Loses to cuBLAS fp16 at batch=1; a tensor-core + autotune rewrite didn't help; a 512→4096 size sweep made it *worse* (4× → 11×). Conclusion: a custom low-bit GEMM isn't competitive with cuBLAS here — realizing low-bit needs a production int8 library (Marlin / CUTLASS / TensorRT) or edge silicon. At batch=1 it's a memory-*footprint* lever (int8 2× / int4 4× smaller, ≤4% action error), not a latency one.
- Method: every number gated by correctness + stale-input + no-leak evals; three experiments, reported win and negative alike.

## Reproduce
- **CPU evals (free, automatic):** every push runs `kernel/selfcheck.py` via [`ci.yml`](.github/workflows/ci.yml) — quant fidelity, kernel fallback numerics, no-leak determinism. Run locally with `python3 kernel/selfcheck.py`.
- **GPU latency + evals:** [`RUN_ON_T4.md`](RUN_ON_T4.md) — ~10 min on a free Colab T4 via [`colab.ipynb`](colab.ipynb).

## Principles

- **Measure first.** Benchmark against *strong* baselines (`torch.compile` + CUDA graphs), never strawmen.
- **Honest claims.** Statistically certified non-regression on a declared distribution — never "provable."
- **Vendor-neutral & reproducible**, on hardware robots actually carry.

## Background

This grew out of *golfing neural networks* — building the smallest ONNX graphs that solve ARC-AGI tasks exactly, scored on a hard parameter/memory budget. Fitting a VLA under a latency budget is the same search: knowing where precision is free and where it's load-bearing.

---

*Early-stage and public by design — the roadmap is the positioning. If you're building robots and fighting the deploy gap, or working the same edge from the lab side, open an issue or reach out.*
