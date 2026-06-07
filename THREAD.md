# X thread — results-backed cut (post-investigation)

*(8 posts, ≤280 chars each. Repo: github.com/LaelaZorana/embodied-efficiency)*

---

**1/**
Everyone can train a vision-language-action model now. Almost nobody can ship one.

The bottleneck isn't capability — it's deploy. End-to-end VLA inference runs at 3–5 Hz; a robot arm needs 50–100 Hz. That gap is the whole game.

So I built the deploy layer and *measured* it. 🧵

**2/**
Target: the flow-matching action sampler that runs every control step on a π0/GR00T-style VLA. I benchmarked it on real GPUs (T4 + L4), every number gated by correctness + memory-leak checks.

Two findings, reported straight — the win and the one that hurt.

**3/**
✅ The win: **CUDA graphs.**
Capturing the N-step sampler in a CUDA graph: 4.82 → 0.82 ms/step — **5.9×**, beating torch.compile's own graph mode, with exact replay and zero memory leak.

**4/**
❌ The negative — and I chased it to the end.
I bet weight-only INT4 would win in this memory-bound regime. My hand-written Triton kernel lost to cuBLAS. Tensor cores + autotune: no change. Size sweep 512→4096: it got *worse*.

**5/**
So I stopped blaming my kernel and tested the **production** path — torchao's Marlin INT4 on a supported L4 (Ada).

It lost too. 1.2–1.6× *slower* than bf16 at every size.

4 experiments, one answer: it was never the implementation.

**6/**
Why: weight-only int4 is built for M=1 LLM *decode* of huge models, where weight reads dominate. A batch-1 VLA sampler — small skinny GEMMs + heavy non-GEMM per-step work — is a different regime, and bf16 tensor cores win.

Low-bit here = memory footprint, not speed.

**7/**
I came to this from competition ML — golfing neural nets to their minimum size under hard correctness budgets (ARC-AGI). Same instinct: where is precision free, where is it load-bearing.

Knowing *where the lever isn't* is half of performance engineering.

**8/**
Repo (public, CI-green, reproducible — notebooks + evals):
github.com/LaelaZorana/embodied-efficiency

Building robots and fighting the deploy gap, or working the same edge from the lab side? Let's compare notes. ↓
