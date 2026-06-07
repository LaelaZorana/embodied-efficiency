# Training the VLA Is the Easy Part

*Why efficiency is the real bottleneck in embodied AI, and what I'm building at that edge.*

---

A vision-language-action model can fold laundry, sort parts, and follow a spoken instruction in a lab demo today. Then you put it on the actual robot and it falls apart, and the reason isn't that it can't do the task. It can't do it fast enough. End-to-end VLA inference runs at 3 to 5 Hz on a research GPU, and a robot arm needs 50 to 100 Hz to move smoothly. The capability is sitting right there. What's missing is everything that gets it onto the robot.

The frontier of embodied AI is no longer whether the model can do it. It's whether the model can do it on this robot, under this latency budget, inside this power envelope, cheaply enough to deploy a thousand of them. That's an efficiency problem, kernels and quantization and scheduling and compilation, and it's the least glamorous and most under-staffed layer in the whole stack.

## The prize is real, and you can measure it

Efficiency is what buys you the robot, and the numbers already exist. INT8 quantization keeps roughly 97% of task success on manipulation while cutting memory and latency hard. INT4 on open VLAs has reached about 2.5x speedup and energy savings over BF16. Action chunking, real-time chunking, speculative decoding, fused flow-matching kernels, the toolkit to close the gap from 3 Hz to 100 Hz is already on the table, scattered across a year of papers.

The ideas aren't the missing piece. The engineering is: making those techniques reproducible, vendor-neutral, and deployable on the hardware robots actually carry, Jetson Orin and Thor, custom silicon running at single-digit watts, without losing the fine dexterity that made the model worth deploying in the first place.

## A field crowded with attention, starved of people

Inference is on track to be two-thirds of all AI compute by the end of 2026, and the industry is in an open bidding war for engineers who can write a genuinely fast kernel. Knowing FP8 exists is easy. Getting a flow-matching action head to run in budget, knowing where you can quantize hard and where one wrong bit breaks the whole policy, is the rare part. The topic is loud with words and money. The work itself is starved of people who can actually do it.

## How I got here, which wasn't an accident

I came at this sideways. For the last cycle I've been golfing neural networks, building the smallest ONNX graphs that still solve ARC-AGI reasoning tasks exactly, scored against a hard budget of parameters and memory. It sounds like a puzzle. It's the same problem.

Fitting a correct solution under a hard size budget and fitting a VLA under a hard latency budget are the same search, the same instinct for where precision comes free and where it carries the load. I spent months sharpening that instinct against a scorer that gave nothing away. I'm pointing it at robots now.

## What I'm building

So here's the work.

- **A budget-driven autotuning compiler for VLAs.** You hand it a target, "100 Hz on an Orin, under 15 watts," and it searches the precision by chunk-size by pruning space and hands back the deployable engine, not just a benchmark row. The papers measure efficiency. Almost nobody publishes the search that picks the right configuration for your robot.
- **A small set of Triton kernels** for the VLA-specific gaps the generic LLM-serving stacks (vLLM, SGLang, TensorRT-LLM) leave open: fused flow-matching and diffusion sampling, INT4 multimodal token paths, chunked-decode KV reuse.

## First results (measured, not promised)

I started at the bottom of that stack, the flow-matching sampling loop, and measured everything on a free T4, with correctness and memory-leak checks gating every number.

**The win was CUDA graphs.** Capturing the N-step sampler in a CUDA graph took it from 4.82 to 0.82 ms/step, a 5.9x cut, beating torch.compile's own graph mode, with exact replay and zero memory leak.

**The negative was weight-only low-bit, and I chased it all the way down.** I expected INT8/INT4 weight quantization to win in this memory-bound regime. It didn't. My hand-written kernel was slower than cuBLAS. I rewrote it for tensor cores and autotuning, and nothing moved. I swept model size from 512 to 4096 expecting a crossover, and the gap widened instead. So I stopped blaming my own kernel and tested the production path, torchao's Marlin int4 on a supported L4, and it lost too, 1.2 to 1.6x slower than bf16 at every size. Four experiments, one conclusion: the implementation was never the problem. Weight-only int4 is built for M=1 LLM decode of huge models, where reading weights dominates. A batch-1 VLA sampler is small skinny GEMMs plus heavy non-GEMM per-step work, a different regime, and bf16 tensor cores win it. Low-bit here's a memory-footprint lever, not a latency one.

I report the negative as plainly as the win. Knowing where the lever isn't is half of performance engineering.

## Where it points

And then the part that matters the moment these systems are actually autonomous: a thin runtime trust layer. Not red-teaming, not one more eval harness, but statistically certified non-regression on a declared distribution, OOD abstention, and intervention logging. The telemetry that lets a hospital-logistics robot or a factory humanoid be both deployed and governed under frameworks like the FDA's device-lifecycle guidance or the EU AI Act.

Efficiency is what gets the model onto the robot. The trust layer is what lets it stay there.

---

Embodied AI's headlines are about capability. The bottleneck is efficiency, and a trust layer is what makes that efficiency safe to deploy. That intersection, kernels and quantization for physical AI with safety built in, is where I'm putting my work.

If you're building robots and fighting the deploy gap, or working the same edge from the lab side, I would like to compare notes.
