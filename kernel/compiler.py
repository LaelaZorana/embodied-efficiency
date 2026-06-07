"""
Budget-driven autotuning deploy-compiler for VLA flow-matching samplers.

"NeuroGolf for VLAs": given a deployment budget (latency / memory / fidelity),
search the efficiency-config space and emit (a) the Pareto frontier and (b) the
best deployable config under the budget. The search space is exactly the levers
this repo *measured*, so the findings are baked into the tool:

  - cuda_graph : on / off          (latency, measured 5.9x win; ~always on)
  - precision  : bf16 / int8 / int4  (bf16 = latency-optimal; int8/int4 = memory
                 FOOTPRINT, not latency, see RESULTS.md, offered as a footprint lever)
  - steps      : flow-integration steps  (latency <-> action-fidelity tradeoff)

Each config is scored on 3 axes: latency (ms/step), weight footprint (MB), and
action fidelity (rMSE vs the full-precision / full-step reference). Footprint and
fidelity are hardware-independent (real anywhere); latency is real on CUDA and a
PROXY off-CUDA (flagged). Multi-objective => report the Pareto set, then pick
under a budget.

Run:  python3 kernel/compiler.py
"""
import copy
import json
import time

import torch
import torch.nn as nn

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample
from triton_gemm import quantize_model_triton


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def _weight_mb(model):
    tot = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            tot += m.weight.numel() * m.weight.element_size()
            if m.bias is not None:
                tot += m.bias.numel() * m.bias.element_size()
        elif m.__class__.__name__ == "TritonQuantLinear":  # int8 (int8) / int4 (packed uint8) + fp scales
            tot += m.weight.numel() * m.weight.element_size() + m.scales.numel() * m.scales.element_size()
            if m.bias is not None:
                tot += m.bias.numel() * m.bias.element_size()
    return tot / 1e6


def _latency_ms_per_step(model, x, steps, dev, graph, iters):
    if graph and dev == "cuda":
        from cudagraph import GraphedSampler
        gs = GraphedSampler(model, x, steps)
        fn = lambda: gs.run(x)  # noqa: E731
    else:
        pkv = model.encode_prefix(x.shape[0])
        fn = lambda: flow_sample(model, x, steps, pkv)  # noqa: E731
    for _ in range(max(2, iters // 4)):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1e3 / steps


def search(d_model=512, B=1, max_steps=10, dev=None):
    dev = dev or _device()
    iters = 20 if dev == "cuda" else 5
    base_dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    cfg = ActionExpertConfig()
    cfg.d_model = d_model
    torch.manual_seed(0)
    base = ActionExpert(cfg).to(dev).to(base_dtype).eval()
    x = torch.randn(B, cfg.horizon, cfg.action_dim, device=dev, dtype=base_dtype)

    ref = flow_sample(base, x, max_steps, base.encode_prefix(B)).float()
    rscale = ref.abs().mean().item() + 1e-9

    precisions = ["bf16", "int8", "int4"]
    steps_opts = sorted({max_steps, max(2, max_steps // 2), max(2, max_steps // 4)}, reverse=True)
    graph_opts = [True, False] if dev == "cuda" else [False]
    models = {"bf16": base,
              "int8": quantize_model_triton(copy.deepcopy(base), bits=8),
              "int4": quantize_model_triton(copy.deepcopy(base), bits=4)}

    rows = []
    for p in precisions:
        m = models[p]
        mb = _weight_mb(m)
        for s in steps_opts:
            out = flow_sample(m, x, s, m.encode_prefix(B)).float()
            rmse = (out - ref).pow(2).mean().sqrt().item() / rscale
            for g in graph_opts:
                ms = _latency_ms_per_step(m, x, s, dev, g, iters)
                rows.append({"precision": p, "steps": s, "graph": g,
                             "ms_per_step": round(ms, 4), "weight_mb": round(mb, 2),
                             "rmse": round(rmse, 4)})
    return {"device": dev, "base_dtype": str(base_dtype).split(".")[-1],
            "d_model": d_model, "batch": B, "latency_real": dev == "cuda", "rows": rows}


def _dominates(a, b):
    le = (a["ms_per_step"] <= b["ms_per_step"] and a["weight_mb"] <= b["weight_mb"] and a["rmse"] <= b["rmse"])
    lt = (a["ms_per_step"] < b["ms_per_step"] or a["weight_mb"] < b["weight_mb"] or a["rmse"] < b["rmse"])
    return le and lt


def pareto(rows):
    return [r for r in rows if not any(_dominates(o, r) for o in rows if o is not r)]


def pick_config(rows, budget):
    """Return the feasible config minimizing budget['objective'] under the constraints."""
    obj = budget.get("objective", "ms_per_step")
    feas = [r for r in rows
            if r["ms_per_step"] <= budget.get("max_ms_per_step", float("inf"))
            and r["weight_mb"] <= budget.get("max_weight_mb", float("inf"))
            and r["rmse"] <= budget.get("max_rmse", float("inf"))]
    return (min(feas, key=lambda r: r[obj]) if feas else None)


if __name__ == "__main__":
    res = search()
    dev = res["device"]
    print(f"device={dev}  base_dtype={res['base_dtype']}  d_model={res['d_model']}  batch={res['batch']}")
    if not res["latency_real"]:
        print("WARNING: latency is a CPU/MPS PROXY (not headline). weight_MB + rMSE are real. "
              "Run on CUDA for real ms/step.")
    print(f"\n{'precision':>9}{'steps':>6}{'graph':>6}{'ms/step':>10}{'weight_MB':>11}{'rMSE':>8}")
    for r in sorted(res["rows"], key=lambda r: r["ms_per_step"]):
        print(f"{r['precision']:>9}{r['steps']:>6}{str(r['graph']):>6}"
              f"{r['ms_per_step']:>10.4f}{r['weight_mb']:>11.2f}{r['rmse']:>8.4f}")

    pf = pareto(res["rows"])
    print(f"\nPareto frontier ({len(pf)} configs, none beaten on all of latency/footprint/fidelity):")
    for r in sorted(pf, key=lambda r: r["ms_per_step"]):
        print(f"  {r['precision']} steps={r['steps']} graph={r['graph']}  "
              f"{r['ms_per_step']:.4f} ms/step  {r['weight_mb']:.2f} MB  rMSE {r['rmse']:.4f}")

    print("\n--- budget: minimize LATENCY, fidelity rMSE <= 0.05 ---")
    print("  chosen:", pick_config(res["rows"], {"objective": "ms_per_step", "max_rmse": 0.05}))
    print("--- budget: minimize FOOTPRINT (memory-constrained edge), rMSE <= 0.05 ---")
    print("  chosen:", pick_config(res["rows"], {"objective": "weight_mb", "max_rmse": 0.05}))

    json.dump(res, open("compiler_report.json", "w"), indent=2)
    print("\nwrote compiler_report.json")
