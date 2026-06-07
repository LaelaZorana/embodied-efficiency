# LinkedIn — Laela's voice (results-backed, zero em-dashes)

*(Repo: github.com/LaelaZorana/embodied-efficiency)*

---

Training a vision-language-action model is the easy part now.

It can fold laundry and follow a spoken instruction in a lab demo. Then you put it on the real robot and it stalls, not because it can't do the task, but because it can't do it fast enough. The model thinks 3 to 5 times a second, and the arm needs 50 to 100. That gap is the whole problem, and almost none of it is about whether the model is smart.

So I built the deploy layer for the part that runs every control step, the flow-matching action sampler, and measured it on real GPUs, with every number checked against the full-precision model and a memory-leak test.

The clean win was CUDA graphs. When you capture the whole loop as one replayable graph, it drops from 4.8 to 0.82 milliseconds per step, almost six times faster, and it matches the original exactly.

The lever I bet on, weight-only int4, didn't work. My hand kernel lost to the standard one, a tensor-core rewrite didn't move it, a size sweep made it worse, and even the production kernel on the hardware it was built for came in 1.2 to 1.6 times slower than bf16. Four tries, one answer: it was never my kernel. Int4 is built for generating one token at a time out of a huge model, not a batch-of-one robot sampler. Low-bit here buys you a smaller model, not a faster one.

The code, the numbers, and the failures are all open: github.com/LaelaZorana/embodied-efficiency

If you're fighting the same deploy gap, I'd like to compare notes.
