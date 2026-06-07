# Training the VLA is the easy part

A vision-language-action model can fold laundry and follow a spoken instruction in a lab demo today. Then you put it on the actual robot, and it stalls. Not because it can't do the task, but because it can't do it fast enough. End to end, the model thinks about 3 to 5 times a second on a research GPU, and a robot arm needs 50 to 100. That gap is the whole problem, and almost none of it is about whether the model is smart.

So the question at the frontier of embodied AI isn't "can the model do it" anymore. It's whether it can run on this robot, inside this latency budget and this power envelope, cheaply enough to put a thousand of them on a floor. That's an efficiency problem, it's the least glamorous layer in the stack, and it's the one almost nobody staffs.

## What I measured

I wanted to know which efficiency levers actually pay off here, so I built the deploy layer for the piece that runs every control step, the flow-matching action sampler, and I measured it on real GPUs. Every number is checked against the unquantized model and against a memory-leak test, because a fast kernel that quietly corrupts an action is worse than a slow one.

The clean win was CUDA graphs. When you capture the whole sampling loop as one replayable graph, it drops from 4.8 milliseconds per step to 0.82, almost six times faster, and the captured run matches the original exactly with nothing leaking between replays.

Then the lever I bet on didn't work, and I chased that all the way down. I expected weight-only int4 to win, because reading a quarter of the weight bytes should cost less. My hand-written kernel lost to the standard one. I rewrote it to run on the tensor cores and tune itself, and nothing moved. I grew the model from small to large expecting a crossover, and the gap got worse instead. So I stopped blaming my own code and ran the production kernel, torchao's int4, on the hardware it's built for, and it lost too, somewhere between 1.2 and 1.6 times slower than plain bf16 at every size.

Four tries, one answer. It was never my kernel. Weight-only int4 is built for a different job, generating one token at a time out of a huge language model, where reading the weights is basically the whole cost. A batch-of-one robot sampler is small matrices plus a lot of work that isn't matrix-multiply at all, and there the ordinary bf16 path wins. So low-bit here buys you a smaller model on the device, not a faster one. That's worth knowing before you spend a week building the wrong kernel.

## Where this comes from

I came to this from competition ML, golfing networks down to the smallest size that still solves a task exactly. It's the same instinct, really: find where precision is free, and where one wrong bit breaks the whole thing.

That instinct is now a small tool. You give it a budget, a latency ceiling or a memory ceiling, and it searches the precision, the number of integration steps, and whether to capture the graph, then it hands back the configuration that fits and the ones it traded off against. Ask it for the fastest model and it picks bf16 with graphs; ask it for the smallest and it picks int4, about seven times lighter for a few percent of action error. What it learned from the experiments is built in.

## Where it points

The part I care about next is what happens once these systems run on their own. A thin layer that watches the policy in real time, notices when it's drifting somewhere it was never trained, falls back to something safe, and keeps a record a regulator can actually read. Efficiency is what gets the model onto the robot. That layer is what lets it stay there.

If you're fighting the same deploy gap, or working it from the model side, I'd like to compare notes. The code, the numbers, and the failures are all in the open: [github.com/LaelaZorana/embodied-efficiency](https://github.com/LaelaZorana/embodied-efficiency)
