"""CPU self-check: every eval that does NOT need a GPU. Exits nonzero on any failure.

Used by CI (.github/workflows/ci.yml) and runnable locally:
    python3 kernel/selfcheck.py

GPU-only evals (kernel latency, CUDA-graph capture/stale-input/no-leak) run
separately on a GPU runner — see .github/workflows/gpu-evals.yml and colab.ipynb.
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
