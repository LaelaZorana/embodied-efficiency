# embodied-efficiency

[![ci](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml/badge.svg)](https://github.com/LaelaZorana/embodied-efficiency/actions/workflows/ci.yml)

**Making frontier embodied and multimodal models cheap to run and safe to deploy.**

Kernels, quantization, and a runtime trust layer for vision-language-action (VLA) models, so they run on the robot, inside a latency and power budget, and can be both deployed and governed.

> A VLA folds laundry in a lab demo today. Put it on the actual robot and it stalls, and not because it can't do the task. It can't do it fast enough. End-to-end inference runs at 3 to 5 Hz, and a robot arm needs 50 to 100 Hz to move smoothly. The capability is already there. What stands between the demo and the robot is engineering.

📄 **Read the thesis →** [THESIS.md](THESIS.md)

🤖 **Try it live →** [interactive demo on Hugging Face](https://huggingface.co/spaces/LaelaZ/embodied-efficiency) (set a budget, watch the compiler pick a config; vet an action through the supervisor)

## The problem

The frontier of embodied AI used to be whether the model could do the task. It can. The question now is whether it runs on this robot, inside this latency budget and this power envelope, cheaply enough to put a thousand of them in the field. That's an efficiency problem, and the levers that solve it, quantization, CUDA graphs, action chunking, speculative decoding, sit scattered across a year of papers with no agreement on which one actually pays off, because the answer depends on the workload. So this repo measures them on one concrete batch-1 VLA flow-matching sampler and reports the result, what wins (CUDA graphs, 5.9x) and what doesn't (weight-only low-bit, across four experiments), as reproducible tooling rather than paper claims.

## What's here

| Component | What it does | Status |
|---|---|---|
| **Thesis** | Why efficiency, not capability, is the bottleneck | ✅ [THESIS.md](THESIS.md) |
| **CUDA-graph sampler** | manual graph capture of the N-step flow loop | ✅ **5.9x measured on T4** ([RESULTS.md](kernel/RESULTS.md)) |
| **Low-bit weight quant** | hand kernel plus production torchao/Marlin int4 | 🔬 investigated, came back **negative** (slower than bf16 even on a supported L4; see findings) |
| **Autotuning deploy-compiler** | budget in, Pareto frontier plus best config out; searches precision, steps, cuda-graph, action-chunking, and a speculative draft | ✅ **v1**, measured on a real L4 ([kernel/compiler.py](kernel/compiler.py)) |
| **Runtime safety supervisor** | per-action finite, bounds, drift (OOD), and jerk checks, safe fallback, intervention log as the governance trail | ✅ **v0, measured** on real robot actions, AUC 0.99 ([safety/supervisor.py](safety/supervisor.py)) |

## Findings (measured on T4 and L4, full data in [kernel/RESULTS.md](kernel/RESULTS.md))

- ✅ **CUDA-graph capture of the sampler runs 5.9x over eager** (4.82 to 0.82 ms/step on T4), beats `torch.compile`, replays exactly, and leaks zero memory across fp16/int8/int4.
- 🔬 **Weight-only low-bit buys no batch-1 latency, and I have the four experiments to show it.** The hand-written INT8/INT4 Triton kernel lost to cuBLAS. A tensor-core plus autotune rewrite changed nothing. A 512 to 4096 size sweep widened the gap instead of closing it. Then the production path, torchao/Marlin int4 on a supported L4 (Ada), lost too, running 1.2 to 1.6x slower than bf16 at every size. So this was never an implementation gap. A batch-1 VLA sampler is small skinny GEMMs plus heavy non-GEMM per-step work, which isn't the regime weight-only int4 was built for (M=1 LLM decode of huge models). Here low-bit is a memory-footprint lever (int8 2x smaller, int4 4x smaller, action error under 5%), not a speed one.
- Every number is gated by correctness, stale-input, and no-leak evals. Four experiments, the win and the negative reported the same way.

## Deploy-compiler (v1)

The findings above aren't just a report, they're built into a budget-driven autotuner ([`kernel/compiler.py`](kernel/compiler.py)). Hand it a deployment budget and it searches precision, integration steps, cuda-graph, action-chunking, and a speculative draft, scores every config on latency, weight-footprint, action-fidelity, and staleness, then hands back the Pareto frontier plus the best config under the budget. v1 runs the whole search on a real L4, so the latencies are measured.

The biggest lever turned out to be action-chunking. Full-fidelity bf16 with a graph is 4.47 ms per call, but if you run all 50 actions in the chunk before recomputing, that's 0.089 ms per action, with the catch that the last action is 49 control steps stale. So you trade latency for staleness, and the compiler makes the trade explicit instead of hiding it. Three budgets, three answers:

```
fastest action, full fidelity, no staleness  ->  bf16 + graph, 10 steps   (4.47 ms/action)
smallest model, full fidelity                ->  int4 + graph             (13.7 MB, vs 51 MB for bf16)
fastest action under 16 MB of weights        ->  int4 + graph + chunk-50  (~0.25 ms/action)
```

It carries what the experiments proved: CUDA graphs and action-chunking buy latency, int4 buys footprint, and the speculative draft doesn't pay here, because a 2-step draft is never close enough to the 10-step full to be accepted. Footprint, fidelity, and staleness compute anywhere; the latency is real on CUDA.

## Safety supervisor (measured)

A fast policy isn't the same as one you can leave running, so the supervisor ([`safety/supervisor.py`](safety/supervisor.py)) sits between the policy and the actuator and vets every action: finite, in joint limits, not drifted far from the calibrated posture (OOD), not too large a jump. When a check trips it holds a safe action and logs why, and that log is the governance trail.

The honest question for any such layer is how well its OOD detector actually works, so I measured it instead of asserting it. [`safety/make_eval_set.py`](safety/make_eval_set.py) builds a labelled set from **real teleoperated actions** (DROID, a Franka arm, 7-dim) plus realistic faults injected into held-out real actions (non-finite, out-of-bounds, drift, jerk), and [`safety/evaluate.py`](safety/evaluate.py) calibrates the supervisor on the real data and sweeps the threshold:

- **Drift (OOD) detector AUC 0.99** separating normal from drifted actions.
- The non-finite and joint-limit checks catch **100%** by construction (they're hard checks).
- The shipped default threshold (4.0) is **deliberately conservative**: it catches every NaN and over-limit command but only ~11% of soft drift, so it almost never cries wolf. Tuned from the ROC to a 1% false-positive budget (threshold ~2.2), it catches **91% of all faults at 1% false positives**. The point is that the threshold is a tunable operating point with a measured curve, not a guess, and you set it to your fleet's tolerance.

The eval set is real-in-distribution with clearly-labelled synthetic faults (nothing synthetic is dressed up as real), and the drift-detector AUC is checked in CI so it can't silently regress.

## Reproduce

- **CPU evals (free, automatic):** every push runs `kernel/selfcheck.py` via [`ci.yml`](.github/workflows/ci.yml), checking quant fidelity, kernel fallback numerics, and no-leak determinism. Run it locally with `python3 kernel/selfcheck.py`.
- **GPU latency and evals:** [`RUN_ON_T4.md`](RUN_ON_T4.md), about 10 minutes on a free Colab T4 via [`colab.ipynb`](colab.ipynb).
- **Deploy-compiler:** `python3 kernel/compiler.py`, prints the Pareto report and the budget picks (footprint and fidelity real anywhere, latency real on CUDA).
- **Safety supervisor eval:** `python3 safety/make_eval_set.py` rebuilds the labelled set from real DROID actions (needs `huggingface_hub` and `pyarrow`), then `python3 safety/evaluate.py` prints the ROC, AUC, and the operating point. The built set is committed, so the AUC check runs in CI with numpy alone.
- **The live demo:** its full source is in [`space/`](space/), a FastAPI plus htmx console (one Docker container, no API key) that drives the compiler against the real-L4 frontier and runs the actual supervisor live. `cd space && docker build -t ee . && docker run -p 7860:7860 ee`.

## Principles

- **Measure first.** Benchmark against strong baselines (`torch.compile` plus CUDA graphs), never strawmen.
- **Honest claims.** Every number comes from real hardware, and I never call a result "provable" for an open-world policy, only measured, or statistical with the distribution stated.
- **Vendor-neutral and reproducible**, on the hardware robots actually carry.

## Background

This grew out of golfing neural networks, building the smallest ONNX graphs that solve ARC-AGI tasks exactly, scored on a hard parameter and memory budget. Fitting a VLA under a latency budget turns out to be the same search: knowing where precision comes free, and where it carries the load.

---

*Early-stage and public by design, because the roadmap is the positioning. If you're building robots and fighting the deploy gap, or working the same edge from the lab side, open an issue or reach out.*
