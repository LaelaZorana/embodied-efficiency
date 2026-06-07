# LinkedIn cut, results-backed (post-investigation, ~230 words)

*(Professional register. Repo: github.com/LaelaZorana/embodied-efficiency)*

---

**The bottleneck in robotics stopped being capability. It's deployment.**

A vision-language-action model can follow a spoken instruction and manipulate objects in a lab demo today. Put it on the actual robot and it stalls: end-to-end inference runs at 3 to 5 Hz, while smooth control needs 50 to 100 Hz. Closing that gap is an efficiency problem, kernels, quantization, scheduling, and it's one of the most under-staffed layers in the stack.

So I built the deploy layer for a VLA flow-matching action sampler and measured it on real GPUs, with every number gated by correctness and memory-leak checks. Two findings, reported straight:

✅ The win, CUDA graphs: 4.82 to 0.82 ms/step (5.9x), beating torch.compile's graph mode, leak-free and correct.

❌ The negative I chased all the way down: I expected weight-only INT4 to win. My hand kernel lost to cuBLAS, a tensor-core plus autotune rewrite didn't help, and a size sweep made it worse. So I tested the production path, torchao/Marlin INT4 on a supported L4, and it lost too, 1.2 to 1.6x slower. Four experiments, one conclusion: the implementation was never the problem. Batch-1 VLA isn't the regime weight-only int4 is built for, so low-bit here's a memory-footprint lever, not a latency one.

Full write-up, code, and reproducible evals: github.com/LaelaZorana/embodied-efficiency

If you're building robots and fighting the deploy gap, or working the same edge from the model side, I would like to compare notes.

*#EmbodiedAI #Robotics #GPU #Inference #VLA*
