# Fused low-bit kernel for the VLA flow-matching action loop

The per-step bottleneck in flow-matching VLAs (π0 / GR00T) is the action expert,
launched once per integration step. This directory builds the evidence and the
kernel to make that loop cheap.

## Files
- `flow_expert.py`, reference π0-style action expert (cached VLM prefix, self+cross attn, MLP) + eager flow sampler. Device-agnostic.
- `bench.py`, benchmark + roofline harness. Honest baselines: `eager` and `compile_reduce_overhead` (= CUDA graphs, the baseline that matters).
- `PLAN.md`, kernel scope + the refined target.

## The finding (why the kernel is what it is)
At batch=1 the loop is **memory-bound** (arithmetic intensity ≈ 24 FLOP/byte vs the
T4 roofline ridge of ~203), dominated by **reading the weights** once per step. The
per-step GEMMs are skinny (M = horizon ≈ 50), so they stream the full `[d,d]` weight
for ~50 rows of work. Consequences:
- within-step op-fusion barely helps (weights are still read once/step),
- CUDA graphs already remove launch overhead,
- **the win is fewer weight bytes → low-bit weight-only quantization + a fused
  dequant→GEMM kernel.** INT4 ≈ ¼ the weight traffic ≈ ¼ the dominant cost here.

So the hero kernel is a **fused INT4/INT8 weight-only-quant dequant+GEMM** for the
expert's linears, not a generic op-fusion.

## Run locally (correctness only, NOT headline numbers)
```bash
python3 kernel/bench.py --steps 10 4 --batch 1 --dtype fp32
```
Verifies the sampler is correct and the harness works. CPU/MPS launch behaviour does
not represent a GPU, so latency here is not the result.

## Run on Colab T4 (the real baseline)
⚠️ Use the **free, unbilled** account or the **free standard T4** runtime only, never a premium A100/L4/V100/TPU.

```python
# Colab cell, Runtime > Change runtime type > T4 GPU
!git clone https://github.com/<you>/embodied-efficiency.git
%cd embodied-efficiency
!pip -q install triton  # preinstalled on most Colab images
!python kernel/bench.py --steps 10 4 8 --batch 1 --dtype fp16 --compile --peak T4
```
Collect: `eager` vs `compile_reduce_overhead` ms/step (= remaining launch overhead),
the fp16 GEMM time, and confirm AI < ridge on real hardware. That sizes the kernel
opportunity. Then we author the low-bit fused kernel against this baseline.

## Success metric (honest)
INT8/INT4 fused kernel vs the fp16 CUDA-graph baseline:
- latency reduction ≈ weight-traffic reduction, shown on a roofline (regime, not assertion);
- action/velocity **MSE vs fp16 reported**, fast *and* faithful, never "lossless".
