"""Merge per-GPU measurement JSONs into one table + a cross-hardware summary.

After running measure_table.py on each GPU (a free T4, an L4, ...), this reads
every kernel/measurements/*.json, writes a combined CSV, and prints the fastest
full-fidelity config per (GPU, model size) so you can see how the best deploy
config shifts across hardware. Pure stdlib, runs anywhere.

Run:  python3 kernel/combine_measurements.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

MEAS = os.path.join(os.path.dirname(__file__), "measurements")
FIELDS = ["gpu", "d_model", "precision", "steps", "graph", "exec_horizon",
          "mode", "call_ms", "ms_per_action", "weight_mb", "rmse", "staleness", "latency_real"]


def main():
    files = sorted(glob.glob(os.path.join(MEAS, "*.json")))
    if not files:
        print("no measurement files yet. Run kernel/measure_table.py on a GPU first.")
        return
    rows = []
    for f in files:
        rows.extend(json.load(open(f))["rows"])

    out_csv = os.path.join(MEAS, "combined.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    gpus = sorted({r["gpu"] for r in rows})
    sizes = sorted({r["d_model"] for r in rows})
    print(f"{len(rows)} rows from {len(gpus)} GPU(s): {', '.join(gpus)}\n")
    print(f"Fastest full-fidelity-ish config per GPU x model size (full steps, lowest ms/action):")
    for g in gpus:
        print(f"\n  {g}")
        for d in sizes:
            cand = [r for r in rows if r["gpu"] == g and r["d_model"] == d
                    and r["mode"] == "plain" and r["rmse"] <= 0.05]
            if not cand:
                continue
            b = min(cand, key=lambda r: r["ms_per_action"])
            print(f"    d_model={d:<5} {b['precision']:<4} {b['steps']}st "
                  f"graph={str(b['graph']):<5} chunk{b['exec_horizon']:<3} "
                  f"-> {b['ms_per_action']:.4f} ms/action  ({b['weight_mb']} MB, rMSE {b['rmse']})")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
