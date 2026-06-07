"""
Production weight-only INT4 path (torchao / Marlin), the right tool for batch-1
low-bit *latency*, replacing the hand-written kernel that lost to cuBLAS.

HARDWARE: Marlin / torchao-int4 (tinygemm) require Ampere (sm_80+). The free T4 is
Turing (sm_75) and CANNOT run these kernels ('.m16n8k16 requires sm_80'). This
benchmark needs an A100 / L4 / A10 / RTX30+, not a T4. It self-guards and exits
cleanly on Turing.

Hypothesis under test: where my hand-written kernel got *worse* with d_model
(4×→11× slower, see RESULTS.md §2b), a production int4 GEMM should CROSS OVER and
beat fp16 at larger d, the large-matmul regime Marlin is built for.

Run on an Ampere+ GPU:  pip install torchao && python3 kernel/prod_int4.py
"""
import time

import torch
import torch.nn as nn

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample
from cudagraph import GraphedSampler


def _bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def _time(model, x, steps):
    """ms/step via CUDA graph; fall back to eager if a kernel rejects capture."""
    try:
        gs = GraphedSampler(model, x, steps)
        ms = _bench(lambda: gs.run(x)) * 1e3 / steps
        del gs
    except Exception:  # noqa: BLE001
        torch.cuda.empty_cache()
        pkv = model.encode_prefix(x.shape[0])
        ms = _bench(lambda: flow_sample(model, x, steps, pkv)) * 1e3 / steps
    torch.cuda.empty_cache()
    return ms


def run(d, steps=10, B=1, dtype=torch.bfloat16):
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    cfg = ActionExpertConfig()
    cfg.d_model = d
    torch.manual_seed(0)
    x = torch.randn(B, cfg.horizon, cfg.action_dim, device="cuda", dtype=dtype)

    m = ActionExpert(cfg).cuda().to(dtype).eval()
    ref = flow_sample(m, x, steps, m.encode_prefix(B)).float()
    t_fp = _time(m, x, steps)

    mq = ActionExpert(cfg).cuda().to(dtype).eval()
    mq.load_state_dict(m.state_dict())
    quantize_(mq, Int4WeightOnlyConfig(group_size=128),
              filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear)
              and not fqn.endswith(("in_proj", "out_proj")))
    out = flow_sample(mq, x, steps, mq.encode_prefix(B)).float()
    rmse = (out - ref).pow(2).mean().sqrt().item() / (ref.abs().mean().item() + 1e-9)
    t_q = _time(mq, x, steps)
    del m, mq
    torch.cuda.empty_cache()
    return t_fp, t_q, rmse


if __name__ == "__main__":
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if cc[0] < 8:
        print(f"GPU compute capability {cc}, torchao int4 (Marlin) needs Ampere sm_80+. "
              "Turing/T4 cannot run it. Use an A100 / L4 / A10 / RTX30+.")
        raise SystemExit(0)
    try:
        import torchao  # noqa: F401
    except ImportError:
        print("needs torchao:  pip install torchao")
        raise SystemExit(0)

    print(f"{torch.cuda.get_device_name(0)}  cc={cc}  batch=1 steps=10  (ms/step, +graph)  [torchao int4 weight-only]")
    print(f"{'d_model':>8}{'bf16':>10}{'int4':>10}{'int4/bf16':>11}{'rMSE':>9}  verdict")
    for d in (512, 2048, 4096):
        try:
            tf, tq, e = run(d)
            print(f"{d:>8}{tf:>10.3f}{tq:>10.3f}{tq / tf:>10.2f}x{e:>9.4f}  "
                  f"{'int4 WINS' if tq < tf else 'bf16 wins'}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{d:>8}  OOM")
    print("\nint4/bf16 < 1.0 => production low-bit wins. Crossover expected at larger d.")
