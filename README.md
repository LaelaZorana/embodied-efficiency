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
| **Fused flow-matching kernel** | Triton kernel for the iterative action-sampling loop generic serving stacks ignore | 🚧 scoped — [kernel/PLAN.md](kernel/PLAN.md) |
| **Autotuning deploy-compiler** | budget in (*"100 Hz on an Orin, <15 W"*) → deployable engine out, via precision × chunk-size × pruning search | 📋 in design |
| **Runtime trust layer** | statistically certified non-regression + OOD abstention + intervention logging | 🔜 planned |

## Principles

- **Measure first.** Benchmark against *strong* baselines (`torch.compile` + CUDA graphs), never strawmen.
- **Honest claims.** Statistically certified non-regression on a declared distribution — never "provable."
- **Vendor-neutral & reproducible**, on hardware robots actually carry.

## Background

This grew out of *golfing neural networks* — building the smallest ONNX graphs that solve ARC-AGI tasks exactly, scored on a hard parameter/memory budget. Fitting a VLA under a latency budget is the same search: knowing where precision is free and where it's load-bearing.

---

*Early-stage and public by design — the roadmap is the positioning. If you're building robots and fighting the deploy gap, or working the same edge from the lab side, open an issue or reach out.*
