# LinkedIn cut — professional register (~210 words)

*(For LinkedIn / recruiter + peer audience. More measured than the X thread; same thesis, same bridge, softer CTA.)*

---

**The bottleneck in robotics isn't capability anymore. It's deployment.**

A vision-language-action model can follow a spoken instruction and manipulate objects in a lab demo today. Put it on the actual robot and it stalls — end-to-end inference runs at 3–5 Hz, while smooth control needs 50–100 Hz. That gap isn't about whether the model can reason. It's about whether it can run on the hardware the robot carries, under a real latency and power budget.

That's an efficiency problem — kernels, quantization, scheduling, compilation — and it's one of the most under-staffed layers in the stack, even as inference heads toward two-thirds of all AI compute.

It's also where I've decided to put my work. I spent the last cycle golfing neural networks to their minimum viable size under hard correctness budgets (ARC-AGI). Fitting a VLA under a latency budget is the same search — knowing where precision is free and where it's load-bearing. I'm now building that out for physical AI: Triton/CUDA kernels and quantization for on-robot inference, with a runtime trust layer (statistically certified non-regression + intervention logging) so these systems can be deployed *and* governed.

If you're building robots and fighting the deploy gap — or working the same edge from the model side — I'd genuinely like to compare notes.

*#EmbodiedAI #Robotics #GPU #Inference #VLA*
