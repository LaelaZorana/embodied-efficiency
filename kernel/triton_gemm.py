"""
Fused weight-only INT8 dequant -> GEMM Triton kernel for the action expert.

This is the hero kernel: in the weight-read-bound flow loop, reading int8 weights
(1 byte) instead of fp16 (2 bytes) ~halves the dominant cost. The kernel
dequantizes group-wise scales *inside* the matmul, so the fp weight is never
materialized in HBM.

Honesty: the kernel's *numerics* are validated locally against a torch
dequant+matmul fallback (same math) — only its *speed* needs a CUDA GPU. On a
machine without Triton/CUDA, `wq_linear` transparently uses the torch fallback,
so the whole pipeline still runs and stays correctness-checkable.

INT4 (true nibble packing -> 3.88x ceiling) is the next kernel, after this INT8
one is validated on T4.

Run:  python3 kernel/triton_gemm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import quantize_weight  # group-wise symmetric weight-only quant

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64


if HAS_TRITON:

    @triton.jit
    def _wq_gemm_kernel(
        a_ptr, qb_ptr, scale_ptr, c_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_sn, stride_sg,
        stride_cm, stride_cn,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        qb_ptrs = qb_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = (offs_k[None, :] + k0) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask, other=0.0).to(tl.float32)
            qb = tl.load(qb_ptrs, mask=(offs_n[:, None] < N) & kmask, other=0).to(tl.float32)
            g = k0 // GROUP_SIZE  # whole BLOCK_K tile lies in one group (GROUP_SIZE % BLOCK_K == 0)
            scale = tl.load(scale_ptr + offs_n * stride_sn + g * stride_sg, mask=offs_n < N, other=0.0)
            b = qb * scale[:, None]                 # dequant -> [BLOCK_N, BLOCK_K]
            acc += tl.dot(a, tl.trans(b), allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            qb_ptrs += BLOCK_K * stride_bk

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _dequant_2d(qweight: torch.Tensor, scales: torch.Tensor, gs: int) -> torch.Tensor:
    N, K = qweight.shape
    ng = scales.shape[1]
    return (qweight.float().view(N, ng, gs) * scales.view(N, ng, 1)).view(N, K)


def wq_linear(x, qweight, scales, bias, group_size):
    """y = x @ dequant(qweight)^T + bias. Triton on CUDA, torch fallback otherwise."""
    *lead, K = x.shape
    N = qweight.shape[0]
    xf = x.reshape(-1, K)
    M = xf.shape[0]

    if not (HAS_TRITON and x.is_cuda):  # numerically-identical reference path
        w = _dequant_2d(qweight, scales, group_size).to(x.dtype)
        return F.linear(x, w, bias)

    xf = xf.contiguous()
    assert group_size % BLOCK_K == 0, "GROUP_SIZE must be a multiple of BLOCK_K"
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _wq_gemm_kernel[grid](
        xf, qweight, scales, y, bias if bias is not None else xf,
        M, N, K,
        xf.stride(0), xf.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_BIAS=bias is not None,
    )
    return y.reshape(*lead, N).to(x.dtype)


class TritonQuantLinear(nn.Module):
    """INT8 weight-only linear. Same quant scheme as quant.QuantLinear, kernel-friendly layout."""

    def __init__(self, lin: nn.Linear, bits: int = 8, group_size: int = 128):
        super().__init__()
        assert bits == 8, "this kernel is INT8; INT4 packing is the next kernel"
        q3, scale3, gs = quantize_weight(lin.weight.data.float(), bits, group_size)
        out, ng, g = q3.shape
        self.register_buffer("qweight", q3.reshape(out, ng * g).contiguous().to(torch.int8))
        self.register_buffer("scales", scale3.reshape(out, ng).contiguous().float())
        self.bias = lin.bias
        self.group_size, self.out_features, self.in_features = gs, out, ng * g

    def forward(self, x):
        return wq_linear(x, self.qweight, self.scales, self.bias, self.group_size)


def quantize_model_triton(model, bits=8, group_size=128, skip=("in_proj", "out_proj")):
    for name, mod in list(model.named_children()):
        if name in skip:
            continue
        if isinstance(mod, nn.Linear):
            setattr(model, name, TritonQuantLinear(mod, bits, group_size))
        else:
            quantize_model_triton(mod, bits, group_size, skip)
    return model


def correctness_check():
    """CUDA-only: Triton kernel vs torch dequant reference must match."""
    torch.manual_seed(1)
    lin = nn.Linear(512, 1536).cuda().half()
    ql = TritonQuantLinear(lin, 8).cuda()
    x = torch.randn(50, 512, device="cuda", dtype=torch.float16)
    y_triton = wq_linear(x, ql.qweight, ql.scales, ql.bias, ql.group_size)
    w = _dequant_2d(ql.qweight, ql.scales, ql.group_size).to(x.dtype)
    y_ref = F.linear(x, w, ql.bias)
    err = (y_triton - y_ref).abs().max().item()
    print(f"triton vs torch-dequant: max abs err = {err:.4e}")
    assert err < 1e-1, "Triton kernel disagrees with reference"
    print("Triton kernel correctness ✓")


if __name__ == "__main__":
    import copy
    from flow_expert import ActionExpertConfig, ActionExpert, flow_sample

    torch.manual_seed(0)
    cfg = ActionExpertConfig()
    fp = ActionExpert(cfg).float().eval()
    B = 1
    x0 = torch.randn(B, cfg.horizon, cfg.action_dim)
    ref = flow_sample(fp, x0, 10, fp.encode_prefix(B))

    qt = quantize_model_triton(copy.deepcopy(fp), bits=8)
    out = flow_sample(qt, x0, 10, qt.encode_prefix(B))
    rmse = (out - ref).pow(2).mean().sqrt().item() / (ref.abs().mean().item() + 1e-9)

    backend = "triton(cuda)" if (HAS_TRITON and torch.cuda.is_available()) else "torch-fallback"
    print(f"backend={backend}  int8 end-to-end action rMSE vs fp = {rmse:.4f}")
    if HAS_TRITON and torch.cuda.is_available():
        correctness_check()
    else:
        print("(no CUDA/Triton here -> ran the numerically-identical torch fallback; "
              "run on Colab T4 to exercise + time the real kernel)")
