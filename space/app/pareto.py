"""The deploy-compiler's Pareto frontier, measured on a real L4.

These rows are the output of kernel/compiler.py in the embodied-efficiency repo,
run on an L4 (Ada) at batch=1 with a CUDA graph. Latency is measured on that
hardware; footprint, fidelity, and staleness compute anywhere. Each row is one
deployable config:

    (precision, flow_steps, exec_horizon, ms_per_action, weight_mb, rmse, staleness)

exec_horizon is action-chunking: run that many actions from one sampler call
before recomputing. It divides the per-call latency down to a per-action cost,
at the price of letting the last action in the chunk get stale.

The pick logic here is the same shape as kernel/compiler.pick_config: filter to
the configs that fit the budget, then take the best one on the chosen objective.
The page serialises ROWS to the client so the slider feedback is instant; this
module keeps the canonical copy and the server-side picker for parity.
"""
from __future__ import annotations

ROWS = [
    # precision, steps, exec_horizon, ms_per_action, weight_mb, rmse, staleness
    ("bf16", 2, 50, 0.0179, 51.0, 0.245, 49), ("int8", 2, 50, 0.0241, 26.4, 0.245, 49),
    ("bf16", 5, 50, 0.0446, 51.0, 0.096, 49), ("int4", 2, 50, 0.0496, 13.7, 0.260, 49),
    ("int8", 5, 50, 0.0600, 26.4, 0.096, 49), ("bf16", 2, 12, 0.0747, 51.0, 0.245, 11),
    ("bf16", 10, 50, 0.0894, 51.0, 0.000, 49), ("int8", 2, 12, 0.1005, 26.4, 0.245, 11),
    ("int8", 10, 50, 0.1199, 26.4, 0.006, 49), ("int4", 5, 50, 0.1238, 13.7, 0.112, 49),
    ("bf16", 5, 12, 0.1858, 51.0, 0.096, 11), ("int4", 2, 12, 0.2068, 13.7, 0.260, 11),
    ("int4", 10, 50, 0.2475, 13.7, 0.046, 49), ("int8", 5, 12, 0.2502, 26.4, 0.096, 11),
    ("bf16", 10, 12, 0.3726, 51.0, 0.000, 11), ("int8", 10, 12, 0.4998, 26.4, 0.006, 11),
    ("int4", 5, 12, 0.5158, 13.7, 0.112, 11), ("bf16", 2, 1, 0.8959, 51.0, 0.245, 0),
    ("int4", 10, 12, 1.0312, 13.7, 0.046, 11), ("int8", 2, 1, 1.2054, 26.4, 0.245, 0),
    ("bf16", 5, 1, 2.2301, 51.0, 0.096, 0), ("int4", 2, 1, 2.4811, 13.7, 0.260, 0),
    ("int8", 5, 1, 3.0025, 26.4, 0.096, 0), ("bf16", 10, 1, 4.4709, 51.0, 0.000, 0),
    ("int8", 10, 1, 5.9973, 26.4, 0.006, 0), ("int4", 5, 1, 6.1898, 13.7, 0.112, 0),
    ("int4", 10, 1, 12.3748, 13.7, 0.046, 0),
]

KEYS = ("precision", "steps", "exec_horizon", "ms_per_action", "weight_mb", "rmse", "staleness")

CONFIGS = [dict(zip(KEYS, r)) for r in ROWS]


def pick(objective: str, max_lat: float, max_mb: float, max_rmse: float, max_stale: int):
    """Best config under the budget, or None if nothing fits."""
    feasible = [c for c in CONFIGS
                if c["ms_per_action"] <= max_lat and c["weight_mb"] <= max_mb
                and c["rmse"] <= max_rmse and c["staleness"] <= max_stale]
    if not feasible:
        return None, 0
    key = "weight_mb" if objective == "footprint" else "ms_per_action"
    return min(feasible, key=lambda c: c[key]), len(feasible)
