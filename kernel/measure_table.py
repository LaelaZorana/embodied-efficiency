"""Measure the deploy-compiler frontier across model sizes on one GPU.

The compiler's frontier was measured for one model on one GPU (an L4). This runs
the same search across several model sizes and tags every row with the GPU it ran
on, so the result is a measurement table you can compare across hardware. Run it
on each GPU you care about (a free T4, an L4, ...); each run writes one JSON under
kernel/measurements/, and combine.py merges them.

The compute is cheap: every config is a batch-1 sampler benchmark (warmup + ~20
timed iters), so the whole grid is a few minutes per GPU. Footprint, fidelity and
staleness are exact anywhere; latency is real only on CUDA.

Run on a GPU box:  python3 kernel/measure_table.py
                   python3 kernel/measure_table.py --sizes 256,512,1024,2048
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from compiler import search


def _gpu_slug():
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,1024,2048", help="comma-separated d_model values")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "measurements"))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    gpu = _gpu_slug()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {dev_name}  ({gpu})   sizes: {sizes}")

    all_rows = []
    for d in sizes:
        res = search(d_model=d, max_steps=args.max_steps)
        for r in res["rows"]:
            all_rows.append({"gpu": dev_name, "d_model": d, "latency_real": res["latency_real"], **r})
        best = min((r for r in res["rows"] if r["mode"] == "plain"), key=lambda r: r["ms_per_action"])
        print(f"  d_model={d:<5} configs={len(res['rows'])}  fastest plain: "
              f"{best['precision']} {best['steps']}st graph={best['graph']} chunk{best['exec_horizon']} "
              f"-> {best['ms_per_action']} ms/action")

    out = os.path.join(args.out_dir, f"{gpu}.json")
    with open(out, "w") as f:
        json.dump({"gpu": dev_name, "gpu_slug": gpu, "sizes": sizes,
                   "latency_real": torch.cuda.is_available(), "rows": all_rows}, f, indent=2)
    print(f"\nwrote {out}  ({len(all_rows)} rows)")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA — latency numbers are a CPU proxy, not real. Run this on a GPU.")


if __name__ == "__main__":
    main()
