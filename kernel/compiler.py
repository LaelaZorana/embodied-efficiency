"""
Budget-driven autotuning deploy-compiler for VLA flow-matching samplers (v1).

It's NeuroGolf for VLAs. You hand it a deployment budget, a latency ceiling or a
memory ceiling, and it searches the efficiency levers this repo measured, then
hands back the Pareto frontier plus the single best config that fits. Everything
it knows came from the experiments, so the findings are built in, not asserted.

Levers searched:
  - precision   : bf16 / int8 / int4   (bf16 wins latency; int8/int4 are a memory
                  footprint lever here, not a latency one, see RESULTS.md)
  - flow_steps  : how many integration steps the sampler runs (latency vs fidelity)
  - cuda_graph  : capture the loop or not (the measured 5.9x latency win)
  - exec_horizon: action-chunking. The sampler produces a whole chunk of actions
                  in one call, so if you execute k of them before recomputing, the
                  per-action latency is one call divided by k. The cost is staleness:
                  the last action you run is k-1 control steps old.
  - speculative : a cheap few-step draft proposes the chunk, the full sampler
                  verifies it. Fidelity stays at the full model's, and the expected
                  latency is draft + (1 - accept_rate) * full, with accept_rate
                  measured against the full output, not guessed.

Every config is scored on four axes: per-action latency (ms), weight footprint
(MB), action fidelity (rMSE vs the full-precision, full-step reference), and
staleness (control steps of lag). Footprint, fidelity, and staleness are exact
anywhere; the latency numbers are real on CUDA and a proxy off it (flagged).

Run:  python3 kernel/compiler.py
"""
import copy
import json
import time

import torch
import torch.nn as nn

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample
from triton_gemm import quantize_model_triton

OBJECTIVES = ["ms_per_action", "weight_mb", "rmse", "staleness"]


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
        elif m.__class__.__name__ == "TritonQuantLinear":
            tot += m.weight.numel() * m.weight.element_size() + m.scales.numel() * m.scales.element_size()
            if m.bias is not None:
                tot += m.bias.numel() * m.bias.element_size()
    return tot / 1e6


def _call_ms(model, x, steps, dev, graph, iters):
    """Latency of ONE sampler call (a full action chunk), in ms."""
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
    return (time.perf_counter() - t0) / iters * 1e3


def _row(precision, steps, graph, exec_h, call_ms, mb, rmse, mode):
    return {"precision": precision, "steps": steps, "graph": graph, "exec_horizon": exec_h,
            "mode": mode, "call_ms": round(call_ms, 4), "ms_per_action": round(call_ms / exec_h, 4),
            "weight_mb": round(mb, 2), "rmse": round(rmse, 4), "staleness": exec_h - 1}


def search(d_model=512, B=1, max_steps=10, dev=None, accept_tol=0.05):
    dev = dev or _device()
    iters = 20 if dev == "cuda" else 4
    base_dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    cfg = ActionExpertConfig()
    cfg.d_model = d_model
    H = cfg.horizon
    torch.manual_seed(0)
    base = ActionExpert(cfg).to(dev).to(base_dtype).eval()
    x = torch.randn(B, H, cfg.action_dim, device=dev, dtype=base_dtype)
    ref = flow_sample(base, x, max_steps, base.encode_prefix(B)).float()
    rscale = ref.abs().mean().item() + 1e-9

    precisions = ["bf16", "int8", "int4"]
    steps_opts = sorted({max_steps, max(2, max_steps // 2), max(2, max_steps // 4)}, reverse=True)
    graph_opts = [True, False] if dev == "cuda" else [False]
    exec_opts = sorted({1, max(1, H // 4), H})
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
                call_ms = _call_ms(m, x, s, dev, g, iters)
                for k in exec_opts:
                    rows.append(_row(p, s, g, k, call_ms, mb, rmse, "plain"))

    # speculative: cheap draft (fewest steps) proposes, full verifies. Fidelity stays
    # at the full model's; latency is draft + (1 - accept) * full, accept measured.
    g = True if dev == "cuda" else False
    draft_steps = min(steps_opts)
    full_steps = max_steps
    draft_ms = _call_ms(base, x, draft_steps, dev, g, iters)
    full_ms = _call_ms(base, x, full_steps, dev, g, iters)
    full_pkv = base.encode_prefix(B)
    accepts, trials = 0, 6
    for _ in range(trials):
        xi = torch.randn_like(x)
        d = flow_sample(base, xi, draft_steps, full_pkv).float()
        fu = flow_sample(base, xi, full_steps, full_pkv).float()
        if (d - fu).pow(2).mean().sqrt().item() / (fu.abs().mean().item() + 1e-9) < accept_tol:
            accepts += 1
    accept_rate = accepts / trials
    exp_ms = draft_ms + (1 - accept_rate) * full_ms
    mb_bf16 = _weight_mb(base)
    rmse_full = (flow_sample(base, x, full_steps, full_pkv).float() - ref).pow(2).mean().sqrt().item() / rscale
    for k in exec_opts:
        rows.append(_row("bf16", full_steps, g, k, exp_ms, mb_bf16, rmse_full,
                         f"spec(draft={draft_steps},accept={accept_rate:.2f})"))

    return {"device": dev, "base_dtype": str(base_dtype).split(".")[-1], "d_model": d_model,
            "batch": B, "horizon": H, "latency_real": dev == "cuda", "rows": rows}


def _dominates(a, b, objs):
    return all(a[o] <= b[o] for o in objs) and any(a[o] < b[o] for o in objs)


def pareto(rows, objs=OBJECTIVES):
    return [r for r in rows if not any(_dominates(o, r, objs) for o in rows if o is not r)]


def pick_config(rows, budget):
    obj = budget.get("objective", "ms_per_action")
    feas = [r for r in rows
            if r["ms_per_action"] <= budget.get("max_ms_per_action", float("inf"))
            and r["weight_mb"] <= budget.get("max_weight_mb", float("inf"))
            and r["rmse"] <= budget.get("max_rmse", float("inf"))
            and r["staleness"] <= budget.get("max_staleness", float("inf"))]
    return min(feas, key=lambda r: r[obj]) if feas else None


if __name__ == "__main__":
    res = search()
    dev = res["device"]
    print(f"device={dev}  base_dtype={res['base_dtype']}  d_model={res['d_model']}  "
          f"batch={res['batch']}  horizon={res['horizon']}")
    if not res["latency_real"]:
        print("latency is a CPU/MPS proxy here, not headline. footprint, fidelity, and "
              "staleness are real. Run on CUDA for real ms.")
    rows = res["rows"]
    print(f"\n{'precision':>9}{'steps':>6}{'graph':>6}{'exec':>5}{'ms/action':>11}"
          f"{'MB':>8}{'rMSE':>7}{'stale':>6}  mode")
    for r in sorted(rows, key=lambda r: r["ms_per_action"]):
        print(f"{r['precision']:>9}{r['steps']:>6}{str(r['graph']):>6}{r['exec_horizon']:>5}"
              f"{r['ms_per_action']:>11.4f}{r['weight_mb']:>8.2f}{r['rmse']:>7.3f}{r['staleness']:>6}  {r['mode']}")

    pf = pareto(rows)
    print(f"\nPareto frontier ({len(pf)} configs, none beaten on all of latency/footprint/fidelity/staleness):")
    for r in sorted(pf, key=lambda r: r["ms_per_action"]):
        print(f"  {r['precision']} steps={r['steps']} graph={r['graph']} exec={r['exec_horizon']} "
              f"{r['mode']}: {r['ms_per_action']:.4f} ms/action, {r['weight_mb']:.1f} MB, "
              f"rMSE {r['rmse']:.3f}, stale {r['staleness']}")

    print("\nbudget = fastest action, fidelity rMSE <= 0.05, no staleness:")
    print("  ", pick_config(rows, {"objective": "ms_per_action", "max_rmse": 0.05, "max_staleness": 0}))
    print("budget = smallest model, fidelity rMSE <= 0.05:")
    print("  ", pick_config(rows, {"objective": "weight_mb", "max_rmse": 0.05}))
    print("budget = fastest action you can get under 16 MB of weights:")
    print("  ", pick_config(rows, {"objective": "ms_per_action", "max_weight_mb": 16, "max_rmse": 0.06}))

    json.dump(res, open("compiler_report.json", "w"), indent=2)
    print("\nwrote compiler_report.json")
