# X-thread cut — "Everyone can train a VLA. Almost nobody can ship one."

*(8 posts, each ≤280 chars. Post 1 is the hook; post 5 is the bridge; post 8 is the CTA.)*

---

**1/**
Everyone can train a vision-language-action model now. Almost nobody can ship one.

A VLA folds laundry in a lab demo, then falls apart on the actual robot — not because it can't do the task, but because it can't do it *fast enough*.

The bottleneck isn't capability. It's deploy.

**2/**
End-to-end VLA inference: 3–5 Hz on a research GPU.
A robot arm needs 50–100 Hz to move smoothly.

That gap — not "can the model reason" — is what stands between today's demos and robots you can deploy a thousand of.

**3/**
And we know efficiency closes it.
INT8 keeps ~97% of task success. INT4 → ~2.5× speedup + energy savings. Chunking, speculative decode, fused kernels all exist.

The ideas aren't missing. The *engineering* is — reproducible, vendor-neutral, on the hardware robots actually carry.

**4/**
This layer is saturated with attention and starved of people.
Inference will be ⅔ of all AI compute by end of 2026, and there's an open bidding war for engineers who can write a genuinely fast kernel.

Knowing FP8 exists is easy. Getting a flow-matching action head into budget isn't.

**5/**
I got here sideways.
I've spent the last cycle *golfing neural nets* — smallest ONNX graphs that solve ARC-AGI tasks exactly, on a hard param/memory budget.

Sounds like a puzzle. It's the same problem as fitting a VLA under a latency budget: where is precision free, where is it load-bearing.

**6/**
So that's the work:
• a budget-driven autotuning compiler for VLAs — give it "100 Hz on an Orin, <15W," get back the deployable engine
• Triton kernels for the VLA-specific gaps generic serving stacks ignore — fused flow-matching sampling, INT4 multimodal token paths

**7/**
And once these systems are autonomous, the part that matters: a thin runtime trust layer.
Not red-teaming. Statistically certified non-regression + OOD abstention + intervention logging — the telemetry that lets a hospital or factory robot be deployed *and* governed.

**8/**
Embodied AI's headlines are about capability.
The bottleneck is efficiency. The trust layer is what makes efficiency deployable.

Kernels + quantization for physical AI, safety baked in. That's where I'm putting my work.

Building robots and fighting the deploy gap? Let's compare notes. ↓
