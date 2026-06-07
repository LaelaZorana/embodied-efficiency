# X thread, results-backed cut (post-investigation)

*(8 posts, under 280 chars each. Repo: github.com/LaelaZorana/embodied-efficiency)*

---

**1/**
Everyone can train a vision-language-action model now. Almost nobody can run one on the robot.

The bottleneck stopped being capability a while ago. It's deploy. End-to-end VLA inference runs at 3 to 5 Hz; a robot arm needs 50 to 100 Hz. That gap is the whole game.

So I built the deploy layer and measured it. 🧵

**2/**
Target: the flow-matching action sampler that fires every control step on a π0/GR00T-style VLA. I benchmarked it on real GPUs (T4 and L4), every number gated by correctness and memory-leak checks.

Two findings, reported straight, the win and the one that hurt.

**3/**
✅ The win: CUDA graphs.
Capturing the N-step sampler in a CUDA graph: 4.82 to 0.82 ms/step, a 5.9x cut, beating torch.compile's own graph mode, with exact replay and zero memory leak.

**4/**
❌ The negative, and I chased it all the way down.
I bet weight-only INT4 would win in this memory-bound regime. My hand-written Triton kernel lost to cuBLAS. Tensor cores plus autotune: no change. Size sweep 512 to 4096: it got worse.

**5/**
So I stopped blaming my kernel and tested the production path, torchao's Marlin INT4 on a supported L4 (Ada).

It lost too. 1.2 to 1.6x slower than bf16 at every size.

Four experiments, one answer: the implementation was never the problem.

**6/**
Why: weight-only int4 is built for M=1 LLM decode of huge models, where weight reads dominate. A batch-1 VLA sampler, small skinny GEMMs plus heavy non-GEMM per-step work, is a different regime, and bf16 tensor cores win.

Low-bit here's memory footprint, not speed.

**7/**
I came to this from competition ML, golfing neural nets down to their minimum size under hard correctness budgets (ARC-AGI). Same instinct: where is precision free, where is it load-bearing.

Knowing where the lever isn't is half of performance engineering.

**8/**
Repo (public, CI-green, reproducible, notebooks plus evals):
github.com/LaelaZorana/embodied-efficiency

Building robots and fighting the deploy gap, or working the same edge from the lab side? Let's compare notes. ↓
