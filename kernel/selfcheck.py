"""CPU self-check: every eval that does NOT need a GPU. Exits nonzero on any failure.

Used by CI (.github/workflows/ci.yml) and runnable locally:
    python3 kernel/selfcheck.py

GPU-only evals (kernel latency, CUDA-graph capture/stale-input/no-leak) run
separately on a GPU runner, see .github/workflows/gpu-evals.yml and colab.ipynb.
"""
import copy
import json
import os
import sys

import torch

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAILS.append(name)


def main():
    torch.manual_seed(0)
    cfg = ActionExpertConfig()
    fp = ActionExpert(cfg).float().eval()
    B = 1
    x0 = torch.randn(B, cfg.horizon, cfg.action_dim)
    pkv = fp.encode_prefix(B)
    ref = flow_sample(fp, x0, 10, pkv)
    rs = ref.abs().mean().item() + 1e-9

    # 1. quant.py fidelity
    from quant import quantize_model
    for bits, thr in [(8, 0.01), (4, 0.10)]:
        qm = quantize_model(copy.deepcopy(fp), bits)
        out = flow_sample(qm, x0, 10, qm.encode_prefix(B))
        rmse = (out - ref).pow(2).mean().sqrt().item() / rs
        check(f"quant int{bits} fidelity", rmse < thr, f"rMSE={rmse:.4f} (<{thr})")

    # 2. triton_gemm.py: int4 pack/unpack roundtrip + fallback numerics
    from triton_gemm import _pack_int4, _unpack_int4, quantize_model_triton
    w = torch.randint(-8, 8, (64, 256)).to(torch.int8)
    check("int4 pack/unpack roundtrip exact", torch.equal(_unpack_int4(_pack_int4(w)), w))
    for bits, thr in [(8, 0.01), (4, 0.10)]:
        qt = quantize_model_triton(copy.deepcopy(fp), bits=bits)
        out = flow_sample(qt, x0, 10, qt.encode_prefix(B))
        rmse = (out - ref).pow(2).mean().sqrt().item() / rs
        check(f"triton int{bits} fallback fidelity", rmse < thr, f"rMSE={rmse:.4f} (<{thr})")

    # 3. cudagraph.py eager-fallback determinism
    from cudagraph import GraphedSampler
    gs = GraphedSampler(fp, x0, 10)
    check("cudagraph eager-fallback determinism",
          torch.allclose(gs.run(x0), flow_sample(fp, x0, 10, pkv)))

    # 3b. compiler decision logic (Pareto + budget pick), pure logic, GPU-free
    from compiler import pareto, pick_config
    rows = [
        {"precision": "bf16", "ms_per_action": 0.8, "weight_mb": 51, "rmse": 0.000, "staleness": 0},
        {"precision": "int8", "ms_per_action": 4.5, "weight_mb": 26, "rmse": 0.003, "staleness": 0},
        {"precision": "int4", "ms_per_action": 6.2, "weight_mb": 13, "rmse": 0.040, "staleness": 0},
        {"precision": "int8", "ms_per_action": 9.0, "weight_mb": 26, "rmse": 0.250, "staleness": 0},  # dominated
    ]
    pf = pareto(rows)
    check("compiler pareto drops dominated config", len(pf) == 3 and rows[3] not in pf)
    lat = pick_config(rows, {"objective": "ms_per_action", "max_rmse": 0.05})
    check("compiler picks latency-optimal under fidelity budget", lat["precision"] == "bf16")
    foot = pick_config(rows, {"objective": "weight_mb", "max_rmse": 0.05})
    check("compiler picks footprint-optimal under fidelity budget", foot["precision"] == "int4")
    check("compiler returns None when infeasible", pick_config(rows, {"max_weight_mb": 5}) is None)
    stale_rows = [
        {"precision": "bf16", "ms_per_action": 0.5, "weight_mb": 51, "rmse": 0.0, "staleness": 49},
        {"precision": "bf16", "ms_per_action": 6.0, "weight_mb": 51, "rmse": 0.0, "staleness": 0},
    ]
    check("compiler respects staleness budget",
          pick_config(stale_rows, {"objective": "ms_per_action", "max_staleness": 0})["staleness"] == 0)

    # 3c. safety supervisor: in-distribution passes; NaN and out-of-bounds get caught
    import numpy as np
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "safety"))
    from supervisor import Supervisor, SupervisorConfig
    rng = np.random.default_rng(0)
    A = 7
    scfg = SupervisorConfig(action_low=np.full(A, -1.0), action_high=np.full(A, 1.0))
    sup = Supervisor(scfg).calibrate(rng.normal(0, 0.25, (1500, A)).clip(-1, 1))
    clean = sum(sup.step(rng.normal(0, 0.25, A).clip(-1, 1))[1] is None for _ in range(300))
    check("supervisor passes in-distribution actions", clean >= 295)
    _, iv_nan = sup.step(np.full(A, np.nan))
    check("supervisor catches NaN and sends a finite action", iv_nan is not None and "nonfinite" in iv_nan.reasons)
    _, iv_ood = sup.step(np.full(A, 5.0))
    check("supervisor catches out-of-bounds", iv_ood is not None and len(iv_ood.reasons) > 0)
    check("supervisor logs interventions", sup.report()["interventions"] >= 2)
    # robustness (audit fixes): malformed shape, too-small calibration, capped log with exact counts
    _, iv_bad = sup.step(np.zeros(3))
    check("supervisor handles a wrong-shape action without crashing",
          iv_bad is not None and "bad_shape" in iv_bad.reasons)
    try:
        Supervisor(scfg).calibrate(np.zeros((2, A)))
        small_raises = False
    except ValueError:
        small_raises = True
    check("supervisor rejects a too-small calibration set", small_raises)
    cap = Supervisor(SupervisorConfig(action_low=np.full(A, -1.0), action_high=np.full(A, 1.0), max_log=50))
    cap.calibrate(rng.normal(0, 0.15, (1500, A)).clip(-1, 1))
    for _ in range(500):
        cap.step(np.full(A, np.nan))
    crep = cap.report()
    check("supervisor caps its log but keeps exact counts",
          crep["logged"] <= 50 and crep["interventions"] == 500)

    # 3d. supervisor eval set: the drift detector separates normal vs drift
    # (real DROID actions + injected faults; built by safety/make_eval_set.py)
    eval_npz = os.path.join(os.path.dirname(__file__), "..", "safety", "data", "supervisor_eval.npz")
    if os.path.exists(eval_npz):
        from evaluate import roc as _roc
        dd = np.load(eval_npz)
        se = Supervisor(SupervisorConfig(action_low=dd["action_low"].astype(float),
                                         action_high=dd["action_high"].astype(float))
                        ).calibrate(dd["calib_actions"].astype(float))
        sc = np.array([se.drift_score(a) for a in dd["eval_action"].astype(float)])
        ft = dd["eval_ftype"]
        _, _, _, auc = _roc(sc[ft == 0], sc[ft == 3])
        check("supervisor drift detector AUC > 0.9 on the eval set", auc > 0.9, f"AUC={auc:.3f}")
    else:
        print("[skip] supervisor eval set absent (run safety/make_eval_set.py)")

    # 4. notebook is valid JSON
    nb = os.path.join(os.path.dirname(__file__), "..", "colab.ipynb")
    try:
        json.load(open(nb))
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print("   ", e)
    check("colab.ipynb valid JSON", ok)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
