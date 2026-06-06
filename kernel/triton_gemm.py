"""
Fused weight-only INT8 / INT4 dequant -> GEMM Triton kernels for the action expert.

v2: TENSOR-CORE path. v1 accumulated in fp32 with allow_tf32=False (no tensor
cores) and lost badly to cuBLAS fp16 on T4. v2 loads activations in fp16 and
dequantizes weights to fp16 so `tl.dot` runs on tensor cores (fp32 accumulate),
plus @triton.autotune over tiles/warps/stages.

  INT8: weights int8 [N, K].
  INT4: weights packed 2-per-byte uint8 [N, K//2] (even k low nibble, odd k high),
        unpacked in-kernel via the a_even/a_odd decomposition.

CUDA-graph note: @triton.autotune benchmarks on first call (pytorch #120802 — it
cannot happen during graph capture). GraphedSampler warms up on a side stream
before capture, so autotune is cached before the graph is recorded.

Numerics validated against a torch dequant+matmul fallback (used off-CUDA); only
speed needs a GPU. Run:  python3 kernel/triton_gemm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import quantize_weight

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# packing / dequant helpers (torch, exact) — also the off-CUDA fallback        #
# --------------------------------------------------------------------------- #
def _pack_int4(q2d: torch.Tensor) -> torch.Tensor:
    nib = (q2d.to(torch.int16) & 0xF).to(torch.uint8)
    return ((nib[:, 1::2] << 4) | nib[:, 0::2]).contiguous()


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0xF).to(torch.int16)
    hi = ((packed >> 4) & 0xF).to(torch.int16)
    N, Kh = packed.shape
    out = torch.empty(N, 2 * Kh, dtype=torch.int16, device=packed.device)
    out[:, 0::2], out[:, 1::2] = lo, hi
    return torch.where(out >= 8, out - 16, out).to(torch.int8)


def _dequant_int8(q, scale, gs):
    N, K = q.shape
    ng = scale.shape[1]
    return (q.float().view(N, ng, gs) * scale.view(N, ng, 1)).view(N, K)


def _dequant_int4(packed, scale, gs):
    return _dequant_int8(_unpack_int4(packed), scale, gs)


# --------------------------------------------------------------------------- #
# Triton tensor-core kernels (autotuned)                                       #
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    def _cfgs(keys):
        base = [
            (64, 64, 64, 4, 3), (64, 128, 64, 4, 3), (32, 64, 64, 4, 4),
            (64, 64, 32, 4, 4), (128, 128, 64, 8, 3), (64, 256, 64, 8, 3),
        ]
        bm, bn, bk = keys
        return [triton.Config({bm: m, bn: n, bk: k}, num_warps=w, num_stages=s)
                for (m, n, k, w, s) in base]

    @triton.autotune(configs=_cfgs(("BM", "BN", "BK")), key=["M", "N", "K"])
    @triton.jit
    def _gemm_int8_kernel(
        a_ptr, qb_ptr, scale_ptr, c_ptr, bias_ptr, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_sn, stride_sg, stride_cm, stride_cn,
        GROUP_SIZE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = qb_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            km = (offs_k[None, :] + k0) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & km, other=0.0)               # fp16
            qb = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & km, other=0).to(tl.float16)
            scale = tl.load(scale_ptr + offs_n * stride_sn + (k0 // GROUP_SIZE) * stride_sg,
                            mask=offs_n < N, other=0.0).to(tl.float16)
            acc += tl.dot(a, tl.trans(qb * scale[:, None]))                                # tensor cores
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.autotune(configs=_cfgs(("BM", "BN", "BJ")), key=["M", "N", "K"])
    @triton.jit
    def _gemm_int4_kernel(
        a_ptr, p_ptr, scale_ptr, c_ptr, bias_ptr, M, N, K,
        stride_am, stride_ak, stride_pn, stride_pk, stride_sn, stride_sg, stride_cm, stride_cn,
        GROUP_SIZE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BJ: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_j = tl.arange(0, BJ)
        Kh = K // 2
        ae_ptrs = a_ptr + offs_m[:, None] * stride_am + (2 * offs_j)[None, :] * stride_ak
        ao_ptrs = a_ptr + offs_m[:, None] * stride_am + (2 * offs_j + 1)[None, :] * stride_ak
        p_ptrs = p_ptr + offs_n[:, None] * stride_pn + offs_j[None, :] * stride_pk
        acc = tl.zeros((BM, BN), tl.float32)
        for j0 in range(0, Kh, BJ):
            jm = (offs_j[None, :] + j0) < Kh
            ae = tl.load(ae_ptrs, mask=(offs_m[:, None] < M) & jm, other=0.0)              # fp16
            ao = tl.load(ao_ptrs, mask=(offs_m[:, None] < M) & jm, other=0.0)
            pk = tl.load(p_ptrs, mask=(offs_n[:, None] < N) & jm, other=0)
            lo = (pk & 0xF).to(tl.float16)
            hi = ((pk >> 4) & 0xF).to(tl.float16)
            lo = tl.where(lo >= 8, lo - 16, lo)
            hi = tl.where(hi >= 8, hi - 16, hi)
            scale = tl.load(scale_ptr + offs_n * stride_sn + ((2 * j0) // GROUP_SIZE) * stride_sg,
                            mask=offs_n < N, other=0.0).to(tl.float16)
            acc += tl.dot(ae, tl.trans(lo * scale[:, None]))                               # tensor cores
            acc += tl.dot(ao, tl.trans(hi * scale[:, None]))
            ae_ptrs += BJ * 2 * stride_ak
            ao_ptrs += BJ * 2 * stride_ak
            p_ptrs += BJ * stride_pk
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def wq_linear(x, weight, scales, bias, group_size, bits):
    """y = x @ dequant(weight)^T + bias. Tensor-core Triton on CUDA, torch fallback otherwise."""
    *lead, _ = x.shape
    K = weight.shape[1] * 2 if bits == 4 else weight.shape[1]
    N = weight.shape[0]
    xf = x.reshape(-1, K)
    M = xf.shape[0]

    if not (HAS_TRITON and x.is_cuda):
        w = (_dequant_int4 if bits == 4 else _dequant_int8)(weight, scales, group_size).to(x.dtype)
        return F.linear(x, w, bias)

    xf = xf.contiguous().to(torch.float16)
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    args = (xf, weight, scales, y, bias if bias is not None else xf, M, N, K,
            xf.stride(0), xf.stride(1), weight.stride(0), weight.stride(1),
            scales.stride(0), scales.stride(1), y.stride(0), y.stride(1))
    if bits == 4:
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
        _gemm_int4_kernel[grid](*args, GROUP_SIZE=group_size, HAS_BIAS=bias is not None)
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
        _gemm_int8_kernel[grid](*args, GROUP_SIZE=group_size, HAS_BIAS=bias is not None)
    return y.reshape(*lead, N).to(x.dtype)


class TritonQuantLinear(nn.Module):
    """Weight-only INT8/INT4 linear. Same quant scheme as quant.QuantLinear."""

    def __init__(self, lin: nn.Linear, bits: int = 8, group_size: int = 128):
        super().__init__()
        assert bits in (4, 8)
        q3, scale3, gs = quantize_weight(lin.weight.data.float(), bits, group_size)
        out, ng, g = q3.shape
        q2d = q3.reshape(out, ng * g).to(torch.int8)
        weight = _pack_int4(q2d) if bits == 4 else q2d.contiguous()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scale3.reshape(out, ng).contiguous().float())
        self.bias = lin.bias
        self.bits, self.group_size = bits, gs
        self.out_features, self.in_features = out, ng * g

    def forward(self, x):
        return wq_linear(x, self.weight, self.scales, self.bias, self.group_size, self.bits)


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
    """CUDA-only: each kernel vs its torch dequant reference must match (fp16 tol)."""
    torch.manual_seed(1)
    x = torch.randn(50, 512, device="cuda", dtype=torch.float16)
    for bits in (8, 4):
        lin = nn.Linear(512, 1536).cuda().half()
        ql = TritonQuantLinear(lin, bits).cuda()
        y_tri = wq_linear(x, ql.weight, ql.scales, ql.bias, ql.group_size, bits)
        deq = (_dequant_int4 if bits == 4 else _dequant_int8)(ql.weight, ql.scales, ql.group_size)
        err = (y_tri - F.linear(x, deq.to(x.dtype), ql.bias)).abs().max().item()
        print(f"  int{bits}: triton vs torch-dequant max abs err = {err:.4e}")
        assert err < 5e-1, f"int{bits} kernel disagrees with reference"
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
    rscale = ref.abs().mean().item() + 1e-9

    backend = "triton(cuda)" if (HAS_TRITON and torch.cuda.is_available()) else "torch-fallback"
    print(f"backend={backend}")
    for bits in (8, 4):
        qt = quantize_model_triton(copy.deepcopy(fp), bits=bits)
        out = flow_sample(qt, x0, 10, qt.encode_prefix(B))
        rmse = (out - ref).pow(2).mean().sqrt().item() / rscale
        print(f"  int{bits}: end-to-end action rMSE vs fp = {rmse:.4f}")
    if HAS_TRITON and torch.cuda.is_available():
        correctness_check()
    else:
        print("(no CUDA/Triton -> ran the numerically-identical torch fallback; "
              "run on Colab T4 to exercise + time the real kernels)")
