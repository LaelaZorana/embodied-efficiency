# X thread — Laela's voice (results-backed)

*(8 posts, ≤280 chars each. Repo: github.com/LaelaZorana/embodied-efficiency)*

---

**1/**
A vision-language-action model can fold laundry in a lab demo today. Put it on the actual robot and it stalls.

Not because it can't do the task, but because it can't do it fast enough: it thinks 3–5 times a second, the arm needs 50–100.

Training was the easy part. 🧵

**2/**
So I built the deploy layer for the piece that runs every control step, the flow-matching action sampler, and measured it on real GPUs.

Every number is checked against the full-precision model and a memory-leak test, because a fast kernel that corrupts an action is worse than a slow one.

**3/**
The clean win was CUDA graphs.

Capture the whole sampling loop as one replayable graph and it drops from 4.8 ms per step to 0.82. Almost six times faster, the captured run matches the original exactly, with nothing leaking between replays.

**4/**
Then the lever I bet on didn't work.

I expected weight-only int4 to win, because reading a quarter of the weight bytes should cost less. My hand kernel lost to the standard one. I rewrote it for the tensor cores and let it tune itself. Nothing moved.

**5/**
I grew the model small to large expecting a crossover, and the gap got worse instead.

So I stopped blaming my own code and ran the production kernel, torchao's int4, on the hardware it's built for. It lost too: 1.2 to 1.6 times slower than plain bf16 at every size.

**6/**
Four tries, one answer. It was never my kernel.

Weight-only int4 is built for generating one token at a time out of a huge language model, where reading the weights is the whole cost. A batch-of-one robot sampler isn't that. So low-bit here buys a smaller model, not a faster one.

**7/**
I came to this from competition ML, golfing networks down to the smallest size that still solves a task exactly.

Same instinct: find where precision is free, and where one wrong bit breaks the whole thing. That instinct is now a tool that picks the config for your budget.

**8/**
The code, the numbers, and the failures are all in the open.

If you're fighting the same deploy gap, or working it from the model side, I'd like to compare notes.

github.com/LaelaZorana/embodied-efficiency ↓
