# Everyone Can Train a VLA. Almost Nobody Can Ship One.

*Why efficiency — not capability — is the real bottleneck in embodied AI, and what I'm building at that edge.*

---

A vision-language-action model can fold laundry, sort parts, and follow a spoken instruction in a lab demo today. Then you put it on the actual robot and it falls apart — not because it can't do the task, but because it can't do it *fast enough*. End-to-end VLA inference runs at **3–5 Hz** on a research GPU. A robot arm needs **50–100 Hz** to move smoothly. That's not a capability gap. It's a deploy gap.

The frontier of embodied AI is no longer "can the model do it." It's: *can the model do it on this robot, under this latency budget, inside this power envelope, cheaply enough to deploy a thousand of them?* That's an efficiency problem — kernels, quantization, scheduling, compilation — and it's the least glamorous and most under-staffed layer in the entire stack.

## The prize is real, and it's measurable

We already know efficiency buys you the robot. INT8 quantization keeps roughly **97% of task success** on manipulation while slashing memory and latency. INT4 on open VLAs has hit **~2.5× speedup and energy savings** over BF16. Action chunking, real-time chunking, speculative decoding, fused flow-matching kernels — the toolkit to close the 3-Hz-to-100-Hz gap already exists, scattered across a year of papers.

What's missing isn't the ideas. It's the *engineering*: making those techniques reproducible, vendor-neutral, and deployable on the hardware robots actually carry — Jetson Orin and Thor, custom silicon running at single-digit watts — without losing the fine-dexterity that made the model worth deploying in the first place.

## A field saturated with attention, starved of people

Inference is on track to be **two-thirds of all AI compute** by the end of 2026, and the industry is in an open bidding war for engineers who can write a genuinely fast kernel. Here's the thing about this layer: knowing FP8 exists is easy. Getting a flow-matching action head to run in budget — knowing where you can quantize aggressively and where a single wrong bit breaks the whole policy — is hard, and rare. The topic is crowded with words and money. The work is starved of people who can actually do it.

## How I got here (the part that's not a coincidence)

I came to this sideways. For the last cycle I've been **golfing neural networks** — building the *smallest possible* ONNX graphs that still solve ARC-AGI reasoning tasks *exactly*, scored on a hard budget of parameters and memory. It sounds like a puzzle. It's the same problem.

"Fit a correct solution under a hard size budget" and "fit a VLA under a hard latency budget" are the same search, the same instinct for where precision is free and where it's load-bearing. I spent months developing that instinct against an unforgiving scorer. I'm now pointing it at robots.

## What I'm building

So that's the work:

- **A budget-driven autotuning compiler for VLAs.** You hand it a target — *"100 Hz on an Orin, under 15 watts"* — and it searches the precision × chunk-size × pruning space and hands back the deployable engine, not just a benchmark row. The papers *measure* efficiency; almost nobody ships the *search* that picks the right configuration for *your* robot.
- **A small set of Triton kernels** for the VLA-specific gaps the generic LLM-serving stacks (vLLM, SGLang, TensorRT-LLM) don't cover: fused flow-matching/diffusion sampling, INT4 multimodal token paths, chunked-decode KV reuse.

## Where it points

And then the part that matters the moment these systems are actually autonomous: a thin **runtime trust layer**. Not red-teaming, not another eval harness — *statistically certified non-regression* on a declared distribution, OOD abstention, and intervention logging. The telemetry that lets a hospital-logistics robot or a factory humanoid be both **deployed and governed** under frameworks like the FDA's device-lifecycle guidance or the EU AI Act.

Efficiency is what gets the model onto the robot. The trust layer is what lets it stay there.

---

Embodied AI's headlines are about capability. The bottleneck is efficiency — and a trust layer is what makes efficiency *deployable*. That intersection, **kernels and quantization for physical AI with safety baked in**, is where I'm putting my work.

If you're building robots and fighting the deploy gap, or working the same edge from the lab side, I'd like to compare notes.
