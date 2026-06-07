"""
Benchmark + roofline harness for the flow-matching action-sampling loop.

Goal of this script (measure-first): produce the EVIDENCE that the per-step
sampling loop is launch/memory-bound, and a strong baseline number for the
fused Triton kernel to beat. Honest baselines only:

  * eager                      - the naive PyTorch loop (N kernel launches/step)
  * compile_reduce_overhead    - torch.compile(mode="reduce-overhead") == CUDA
                                 graphs; this already removes most launch
                                 overhead, so it is the baseline that MATTERS.
                                 (CUDA only.)

Run locally (CPU/MPS) to validate correctness + that the harness works.
Run on a CUDA GPU (Colab T4) for the headline numbers. See kernel/README.md.
"""
import argparse
import json
import time

import torch

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample

# Approx hardware peaks for the roofline ridge point (FLOP/byte = peak_flops/peak_bw).
DEVICE_PEAKS = {  # (peak TFLOP/s, peak GB/s), fp16
    "T4": (65.0, 320.0),       # Tesla T4 (Colab free) ridge ~= 203 FLOP/byte
    "A100": (312.0, 1555.0),
    "OrinAGX": (170.0, 205.0),  # Jetson Orin AGX (int8/fp16-ish) ridge very high -> memory-bound
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def time_fn(fn, dev, iters, warmup=5):
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters


def analytic_flops_bytes(cfg: ActionExpertConfig, n_steps: int, B: int, dtype_bytes: int):
    """Rough per-call FLOPs and weight-read bytes for the expert, x N steps.

    For small batch the per-step cost is dominated by *reading the weights* once
    per launch (no reuse across the tiny token set) -> low arithmetic intensity
    -> memory/launch bound. That is the kernel-fusion opportunity.
    """
    d, H, P, L = cfg.d_model, cfg.horizon, cfg.prefix_len, cfg.n_layers
    mlp = int(d * cfg.mlp_ratio)
    per_layer_macs = (
        3 * H * d * d      # qkv
        + H * d * d        # self out proj
        + H * d * d        # cross q
        + H * d * d        # cross out proj
        + 2 * H * d * mlp  # mlp up+down
        + H * H * d        # self attention (scores + a·v, approx)
        + H * P * d        # cross attention (scores + a·v, approx)
    )
    io_macs = H * cfg.action_dim * d + H * d * cfg.action_dim  # in/out projections
    flops = 2.0 * B * n_steps * (L * per_layer_macs + io_macs)

    params = (
        L * (3 * d * d + d * d + d * d + 2 * d * d + 2 * d * mlp)
        + cfg.action_dim * d + d * d + d * cfg.action_dim
    )
    weight_bytes = B * n_steps * params * dtype_bytes  # weights re-read each launch
    return flops, weight_bytes, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 4])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true", help="add torch.compile reduce-overhead (CUDA)")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "int4"], help="swap transformer linears to the fused low-bit kernel")
    ap.add_argument("--graph", action="store_true", help="capture the sampler in a CUDA graph (manual, composes with --quant)")
    ap.add_argument("--peak", default="T4", choices=list(DEVICE_PEAKS), help="device for roofline ridge")
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()

    dev = pick_device()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"device={dev}  dtype={args.dtype}  batch={args.batch}")
    if dev != "cuda":
        print(f"WARNING: only CUDA numbers are headline-valid. Launch-overhead behaviour on "
              f"'{dev}' does NOT represent a Jetson/datacenter GPU. Use this run to validate "
              f"correctness + the harness; produce real numbers on a CUDA GPU (Colab T4).")

    cfg = ActionExpertConfig()
    torch.manual_seed(0)
    model = ActionExpert(cfg).to(dev).to(dtype).eval()
    if args.quant in ("int8", "int4"):
        from triton_gemm import quantize_model_triton
        bits = int(args.quant[3:])
        quantize_model_triton(model, bits=bits)
        print(f"quantized transformer linears -> {args.quant.upper()} fused kernel (IO projections kept fp)")
    qtag = "" if args.quant == "none" else f"{args.quant}/"
    B = args.batch
    x0 = torch.randn(B, cfg.horizon, cfg.action_dim, device=dev, dtype=dtype)
    pkv = model.encode_prefix(B)

    # correctness sanity
    out = flow_sample(model, x0, 4, pkv)
    assert out.shape == x0.shape and torch.isfinite(out).all(), "sampler produced bad output"
    print(f"correctness OK: out shape {tuple(out.shape)}, finite ✓\n")

    variants = [("eager", lambda ns: (lambda: flow_sample(model, x0, ns, pkv)))]
    if args.compile and dev == "cuda":
        cmodel = torch.compile(model, mode="reduce-overhead")
        variants.append(("compile_reduce_overhead", lambda ns: (lambda: flow_sample(cmodel, x0, ns, pkv))))
    if args.graph and dev == "cuda":
        from cudagraph import GraphedSampler

        def _mk_graph(ns):
            gs = GraphedSampler(model, x0, ns)
            return lambda: gs.run(x0)

        variants.append(("graph", _mk_graph))

    ridge = DEVICE_PEAKS[args.peak]
    ridge_ai = ridge[0] * 1e12 / (ridge[1] * 1e9)
    print(f"roofline ridge ({args.peak}): {ridge_ai:.1f} FLOP/byte  "
          f"(AI below this = memory-bound -> fusion helps)\n")
    print(f"{'variant':26s}{'steps':>6s}{'ms':>10s}{'ms/step':>10s}{'TFLOP/s':>10s}{'AI':>8s}")

    results = {"device": dev, "dtype": args.dtype, "batch": B, "ridge_ai": ridge_ai,
               "config": cfg.__dict__, "runs": []}
    dbytes = 2 if dtype != torch.float32 else 4
    for ns in args.steps:
        flops, wbytes, _ = analytic_flops_bytes(cfg, ns, B, dbytes)
        for name, mk in variants:
            sec = time_fn(mk(ns), dev, args.iters)
            ai = flops / wbytes
            row = {"variant": qtag + name, "steps": ns, "ms": sec * 1e3, "ms_per_step": sec * 1e3 / ns,
                   "achieved_tflops": flops / sec / 1e12, "arith_intensity": ai,
                   "memory_bound": ai < ridge_ai}
            results["runs"].append(row)
            print(f"{qtag+name:26s}{ns:6d}{sec*1e3:10.3f}{sec*1e3/ns:10.3f}"
                  f"{flops/sec/1e12:10.3f}{ai:8.2f}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
