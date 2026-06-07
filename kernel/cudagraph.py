"""
CUDA-graph capture of the flow-matching sampling loop, composed with the low-bit
fused kernels.

Manual capture (NOT torch.compile reduce-overhead) on purpose: torch.compile
graph-breaks on raw user Triton kernels unless wrapped as torch.library.triton_op,
so manual CUDAGraph is the reliable way to graph a sampler that calls our hand-
written Triton GEMM. CUDA graphs and low-bit quant are *orthogonal* wins: graphs
remove per-launch CPU overhead (N steps x many small kernels); the kernel cuts
weight-byte traffic. Combined, they should stack.

Failure modes designed against (researched, then defended):
  * capture-time JIT      -> warm up on a side stream before capture (Triton compiles).
  * stale frozen input    -> static input buffer + copy_ each replay; the stale-input
                             eval feeds two DIFFERENT inputs and checks each matches eager.
  * output aliasing       -> clone static_out after replay (its storage is reused).
  * memory-pool growth    -> the no-leak eval replays K times and asserts
                             memory_allocated() does not grow.
  * autotune-in-capture   -> kernels use fixed block sizes (no @triton.autotune),
                             sidestepping pytorch issue #120802.

Run:  python3 kernel/cudagraph.py   (GPU evals on CUDA; eager-fallback check otherwise)
"""
import torch

from flow_expert import flow_sample


class GraphedSampler:
    """Captures flow_sample(expert, x, n_steps) into a CUDA graph with a static input."""

    def __init__(self, expert, x_template, n_steps, warmup=5):
        self.expert, self.n = expert, n_steps
        self.cuda = x_template.is_cuda and torch.cuda.is_available()
        self.prefix_kv = expert.encode_prefix(x_template.shape[0])  # constant, captured
        if not self.cuda:
            self.static_in = None
            return
        self.static_in = x_template.clone()
        # warm up on a side stream so Triton/cuBLAS JIT + allocations settle pre-capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                flow_sample(expert, self.static_in, n_steps, self.prefix_kv)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = flow_sample(expert, self.static_in, n_steps, self.prefix_kv)

    def run(self, x):
        if not self.cuda:
            return flow_sample(self.expert, x, self.n, self.prefix_kv)
        self.static_in.copy_(x)          # never trust the new tensor's address, copy in
        self.graph.replay()
        return self.static_out.clone()   # storage is reused on next replay, clone out


def _build(bits=None, device="cuda", seed=0):
    import copy  # noqa: F401
    from flow_expert import ActionExpertConfig, ActionExpert
    torch.manual_seed(seed)
    m = ActionExpert(ActionExpertConfig()).to(device)
    m = m.half() if device == "cuda" else m.float()
    m.eval()
    if bits:
        from triton_gemm import quantize_model_triton
        quantize_model_triton(m, bits=bits)
    return m


def eval_all(n=10, B=1, tol=2e-1):
    """GPU evals: correctness, stale-input, no-leak. Falls back to a determinism check off-CUDA."""
    if not torch.cuda.is_available():
        print("no CUDA -> graph capture skipped; running eager-fallback determinism check")
        m = _build(bits=None, device="cpu")
        x = torch.randn(B, m.cfg.horizon, m.cfg.action_dim)
        gs = GraphedSampler(m, x, n)
        ok = torch.allclose(gs.run(x), flow_sample(m, x, n, m.encode_prefix(B)))
        print(f"  eager-fallback determinism: {'OK' if ok else 'FAIL'}")
        return

    for label, bits in [("fp16", None), ("int8", 8), ("int4", 4)]:
        m = _build(bits=bits)
        xt = torch.randn(B, m.cfg.horizon, m.cfg.action_dim, device="cuda", dtype=torch.float16)
        gs = GraphedSampler(m, xt, n)
        pkv = m.encode_prefix(B)

        # correctness + stale-input: two DISTINCT inputs must each match fresh eager
        max_err = 0.0
        for _ in range(2):
            x = torch.randn_like(xt)
            max_err = max(max_err, (gs.run(x) - flow_sample(m, x, n, pkv)).abs().max().item())

        # no-leak: replay K times (discard results), live memory must not grow
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        m0 = torch.cuda.memory_allocated()
        for _ in range(50):
            gs.run(xt)
        torch.cuda.synchronize()
        leaked = torch.cuda.memory_allocated() - m0

        ok = max_err < tol and leaked <= 4096
        print(f"  {label:5s} graph: max_err={max_err:.3e}  leaked={leaked}B  -> {'OK' if ok else 'FAIL'}")
        assert max_err < tol, f"{label}: graphed output disagrees with eager (stale input?)"
        assert leaked <= 4096, f"{label}: graph leaks {leaked}B per replay"
    print("CUDA-graph correctness + stale-input + no-leak ✓")


if __name__ == "__main__":
    eval_all()
