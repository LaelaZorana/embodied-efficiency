# Run the GPU evals on a free Colab T4 (step by step)

Closes the only open gate: real kernel **latency** + the **GPU-only evals** (CUDA-graph
correctness / stale-input / no-leak, Triton-vs-torch correctness). ~10 minutes, free.

## 1. Make a read-only token (so Colab can clone the private repo)
1. GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.**
2. **Repository access:** *Only select repositories* → `LaelaZorana/embodied-efficiency`.
3. **Repository permissions:** *Contents → Read-only* (nothing else needed to clone).
4. **Expiration:** 7 days (short, you'll revoke it after the run anyway).
5. Generate, copy the token (starts `github_pat_…`).

> Read-only + single-repo + short expiry = minimal blast radius. **Revoke it after step 4.**

## 2. Open Colab on a free T4
1. https://colab.research.google.com → **File → Upload notebook** → upload `colab.ipynb` from this repo
   (or **File → Open notebook → GitHub**, sign in, pick the repo).
2. **Runtime → Change runtime type → T4 GPU** → Save.
   ⚠️ Standard **T4 only**, never A100/L4/V100/TPU (those burn paid units).

## 3. Wire the token + run
In the **clone cell**, set:
```python
REPO = 'https://LaelaZorana:github_pat_XXXX@github.com/LaelaZorana/embodied-efficiency.git'
```
Then **Runtime → Run all.**

> Don't save the notebook back to GitHub/Drive with the token in it, edit that cell only in the live session. (The repo copy keeps the tokenless placeholder.)

## 4. What to copy back to me
Paste the printed output of the three sections. The numbers that matter:

- **§1 latency**, for `eager`, `compile_reduce_overhead`, `graph`: the `ms/step` at steps 10/4/8.
- **§2 low-bit**, `int8/eager`, `int8/graph`, `int4/eager`, `int4/graph`: `ms/step`.
  (Compare against the **1.97× / 3.88×** weight-traffic ceilings.)
- **§3 evals**, must read `Triton kernel correctness ✓` and
  `CUDA-graph correctness + stale-input + no-leak ✓`. Note any `FAIL`.

## 5. After it passes
1. **Revoke the token** (Settings → the fine-grained token → Delete).
2. Send me the output, I'll fill the real numbers into `kernel/RESULTS.md` and flip the
   thesis/README "pending T4" lines to actual figures.
3. Then flip the repo public when you're happy: `gh repo edit LaelaZorana/embodied-efficiency --visibility public`.

---

### No-token alternative
Skip the token: in Colab, drag the `kernel/` folder into the file panel, then run
`!python kernel/triton_gemm.py`, `!python kernel/cudagraph.py`, and the `bench.py`
lines from `colab.ipynb` directly. Slightly more manual; no credentials involved.
