"""Model-size sweep — the decisive test of the corrected diagnosis.

At d_model=512 the low-bit kernel lost because the per-step matmuls are tiny
(op-overhead-bound, not weight-bandwidth-bound). The hypothesis: as d_model
grows, reading each [d,d] weight starts to dominate, so int8 (½ the weight
bytes) should cross over and BEAT fp16. If it never crosses over, low-bit is the
wrong lever regardless of size. Either way it's an honest answer.

Compares fp16+graph vs int8+graph at batch=1 across d_model on the current GPU.
Run on a GPU:  python3 kernel/sweep.py
"""
import copy
import time

import torch

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample  # noqa: F401
from cudagraph import GraphedSampler
from triton_gemm import quantize_model_triton


def _bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def _time_variant(model, x, steps):
    gs = GraphedSampler(model, x, steps)
    ms = _bench(lambda: gs.run(x)) * 1e3 / steps
    del gs
    torch.cuda.empty_cache()
    return ms


def run(d, steps=10, B=1):
    cfg = ActionExpertConfig()
    cfg.d_model = d
    torch.manual_seed(0)
    x = torch.randn(B, cfg.horizon, cfg.action_dim, device="cuda", dtype=torch.float16)

    m = ActionExpert(cfg).cuda().half().eval()
    t_fp = _time_variant(m, x, steps)
    del m
    torch.cuda.empty_cache()

    mq = quantize_model_triton(ActionExpert(cfg).cuda().half().eval(), bits=8)
    t_q = _time_variant(mq, x, steps)
    del mq
    torch.cuda.empty_cache()
    return t_fp, t_q


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a GPU"
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}  batch=1  steps=10  (ms/step, +CUDA graph)")
    print(f"{'d_model':>8}{'fp16':>10}{'int8':>10}{'int8/fp16':>11}  verdict")
    for d in (512, 1024, 2048, 4096):
        try:
            tf, tq = run(d)
            verdict = "int8 WINS" if tq < tf else "fp16 wins"
            print(f"{d:>8}{tf:>10.3f}{tq:>10.3f}{tq / tf:>10.2f}x  {verdict}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{d:>8}  OOM — skipped")
    print("\nint8/fp16 < 1.0 => low-bit wins. Watch for the crossover as d_model grows.")
