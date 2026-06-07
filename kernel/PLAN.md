# Hero kernel, fused flow-matching action-sampling loop (Triton)

**One-line:** A fused Triton kernel for the iterative denoising loop in VLA flow-matching/diffusion action experts, the small-head, many-step sampling that generic LLM-serving stacks (vLLM / SGLang / TensorRT-LLM) don't optimize.

## Why this kernel (validated 2026-06-06)
- **Real, VLA-specific bottleneck.** π0 / GR00T / most modern VLAs sample each action chunk via *dozens of iterative integration (denoising) steps*; latency scales **linearly** with step count. Each step = one forward pass through a *small* action expert.
- **Small net × many iterations = launch-overhead- and memory-bound**, not FLOP-bound, the exact regime where kernel fusion wins and where torch eager / naive loops bleed time on per-step kernel launches + HBM round-trips.
- **Un-owned.** Algorithmic step-reduction is crowded (A2A, FASTER, One-Step Flow, FLASH). The *kernel* layer is empty: StableTriton = image diffusion (big UNet, wrong shape); Liger = transformer ops. No fused sampler for the small-head, many-step action loop.
- **Composes, doesn't compete.** A2A/FASTER reduce N (step count); this kernel makes each remaining step cheaper. They multiply. Story = "the systems layer under the algorithmic papers."

## What it beats (baselines, be honest, no strawman)
1. Eager PyTorch sampling loop (N separate forward passes).
2. **The strong baseline: `torch.compile` + CUDA graphs**, this already removes a lot of launch overhead. The win must come from genuine fusion + SRAM residency of the integration state across steps, not from beating a naive loop. Benchmark against THIS, or the result isn't credible.

## The win
Fuse the integration loop so per-step intermediate state stays in SRAM; fuse the action-expert MLP/attention + the Euler/RK integration update; eliminate N× launch overhead and N× HBM traffic. Largest when N is large and the head is memory-bound; smaller when the head is big enough to be compute-bound, state this regime explicitly.

## Hardware plan (works on what Laela can get, see [[laela-api-keys-and-anthropic-models]] / Colab notes)
- **Develop + headline benchmark: T4 / consumer GPU in FP16 + INT8.** No FP8 needed for the proof; the action head is small and fits anywhere. Colab **free/safe** account `laela.zorana@gmail.com` (zero paid units) or free T4 only on the Pro account, NEVER premium A100/L4/V100/TPU.
- **FP8 variant: one rented Hopper hour** (Modal / RunPod / Lambda) for a single headline number, optional, not required for v1.
- **Deployment target to cite: Jetson Orin** (not required to own for the benchmark).
- Local machine is macOS (no NVIDIA GPU) → all kernel work happens on Colab T4 or rented GPU.

## Success metric (honest)
- X× latency reduction on the **action-sampling step** vs `torch.compile`+CUDA-graphs baseline, at representative step counts (e.g. N=10 and N=4), reported with a **roofline** so the memory-bound claim is shown, not asserted.
- Plus: end-to-end Hz delta on a representative action expert, so the system-level meaning is clear.

## Deliverable shape (the proven reputation format)
Clean repo: `kernel/` + microbenchmark + roofline plot + short writeup. Model on `bassrehab/triton-kernels` and the "Beating CUDA with Triton" MoE-dispatch post. Links from the thesis.

## First step (measure-first, matches NeuroGolf discipline)
1. Stand up a minimal flow-matching action expert (from openpi/π0, or a clean toy head) + the eager sampling loop.
2. **Profile it.** Confirm it's launch/memory-bound (Nsight / torch profiler). If torch.compile already fuses it away, pivot the target, don't write a kernel for a bottleneck that isn't one.
3. Only then write the fused Triton kernel.

## ✅ REFINED target (after local profiling, 2026-06-06)
Baseline built (`flow_expert.py`) + harness (`bench.py`), correctness-verified on MPS. Analytic roofline at batch=1: **AI ≈ 24 FLOP/byte vs T4 ridge 203 → memory-bound, weight-read-dominated.** Key consequence:
- The per-step GEMMs are **skinny (M=`horizon`≈50)**: they read the full `[d,d]` weight to compute only ~50 rows → reading weights dominates, compute is tiny.
- Therefore **within-step op-fusion is NOT the win** (weights are still read once/step) and **CUDA graphs already remove launch overhead** (hence it's the baseline to beat).
- **The win = fewer weight BYTES → low-bit weight-only quantization + a fused dequant→GEMM Triton kernel.** INT4 ≈ ¼ the weight traffic ≈ ¼ the dominant cost in this regime. Composes with algorithmic step-reduction (A2A/FASTER cut N; this cuts cost/step) and IS the quantization thesis.

**Hero kernel v1 = fused INT4/INT8 weight-only-quant dequant+GEMM Triton kernel for the action-expert linears**, benchmarked fp16 (eager + compile/CUDA-graphs) vs INT8 vs INT4, with: (a) latency reduction ≈ weight-traffic reduction, (b) roofline showing the regime, (c) **honest quality check**, action/velocity MSE vs fp16 reported, not claimed zero.

## Next concrete step
Get the **real T4 baseline** (`bench.py --compile --dtype fp16`) to size the eager-vs-CUDA-graph gap and the fp16 GEMM time, THEN author the low-bit fused kernel against it. See README "Run on Colab T4".
