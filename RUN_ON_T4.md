# Run the GPU evals on a free Colab T4

The CPU evals run in CI on every push. This is the other half, real kernel **latency**
plus the **GPU-only evals** (CUDA-graph correctness, stale input, no leak, and Triton
against a torch reference). It takes about ten minutes on a free T4, and it reproduces
the numbers in [`kernel/RESULTS.md`](kernel/RESULTS.md).

## 1. Open the notebook on a T4
1. Go to https://colab.research.google.com, then **File → Upload notebook** and upload
   `colab.ipynb` from this repo, or **File → Open notebook → GitHub** and pick the repo.
2. **Runtime → Change runtime type → T4 GPU**, then Save. The standard T4 is free and
   it's everything these evals need, so there's no reason to reach for a bigger
   accelerator.

## 2. Run all
The repo is public, so the clone cell works as written with no token and nothing to
configure. **Runtime → Run all.**

## 3. What the output should say
- **§1 latency**, `eager` against `compile_reduce_overhead` against `graph`, in ms/step
  at 10, 4, and 8 steps. The manual graph column is the win, about 5.9x over eager.
- **§2 low-bit**, `int8/graph` and `int4/graph` ms/step. These come out *slower* than
  fp16 plus a graph, which is the measured negative written up in `kernel/RESULTS.md`
  rather than a broken run. Compare the weight-traffic ceilings (1.97x and 3.88x)
  against the latency and you can see the gap for yourself.
- **§3 evals** must print `Triton kernel correctness ✓` and
  `CUDA-graph correctness + stale-input + no-leak ✓`, with action rMSE 0.0025 (int8)
  and 0.0423 (int4). Any `FAIL` means the latency above isn't trustworthy yet.
- **§4 the deploy compiler** prints the Pareto frontier and the budget picks. The
  latencies are real because you're on a GPU.
- **§5 the safety supervisor** needs no GPU and runs anywhere.

## No-notebook alternative
Drag the `kernel/` folder into the Colab file panel and run the scripts directly,
`!python kernel/triton_gemm.py`, `!python kernel/cudagraph.py`, and the `bench.py`
lines from `colab.ipynb`. A little more manual, same output.
