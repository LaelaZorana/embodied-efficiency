"""
Weight-only quantization for the action expert + a pure-torch dequant+matmul
reference.

This quantifies the kernel prize WITHOUT a GPU: how much quality we lose at
INT8/INT4, and how many weight bytes we save (= the memory-bound speedup
ceiling, since the flow loop is weight-read-bound at small batch). The Triton
fused dequant->GEMM kernel implements this same math fast; here we validate the
math is faithful before optimizing it.

Run:  python3 kernel/quant.py
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_expert import ActionExpertConfig, ActionExpert, flow_sample


def quantize_weight(w: torch.Tensor, bits: int, group_size: int = 128):
    """Symmetric per-group weight-only quant along the input dim. w: [out, in]."""
    out, inf = w.shape
    gs = group_size if inf % group_size == 0 else inf  # per-row fallback for tiny dims
    wg = w.view(out, inf // gs, gs)
    qmax = 2 ** (bits - 1) - 1
    scale = (wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    q = torch.clamp((wg / scale).round(), -qmax - 1, qmax)
    return q.to(torch.int8), scale, gs


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, gs: int) -> torch.Tensor:
    out, ng, _ = q.shape
    return (q.float() * scale).view(out, ng * gs)


class QuantLinear(nn.Module):
    """Drop-in for nn.Linear: stores int weights + fp scales, dequantizes on the fly.

    This is the *reference* (dequant materializes the fp weight, so it is not fast
    in torch), the Triton kernel fuses dequant into the GEMM to actually win.
    """

    def __init__(self, lin: nn.Linear, bits: int, group_size: int = 128):
        super().__init__()
        q, scale, gs = quantize_weight(lin.weight.data.float(), bits, group_size)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.gs, self.bits = gs, bits
        self.bias = lin.bias
        self.out_features, self.in_features = lin.weight.shape

    def forward(self, x):
        w = dequantize_weight(self.q, self.scale, self.gs).to(x.dtype)
        return F.linear(x, w, self.bias)

    def weight_bytes(self) -> float:
        return self.q.numel() * self.bits / 8.0 + self.scale.numel() * 2  # int weights + fp16 scales


def quantize_model(model: nn.Module, bits: int, group_size: int = 128,
                   skip=("in_proj", "out_proj")) -> nn.Module:
    """Quantize the big transformer linears; keep the IO projections (to/from the
    tiny action space) in fp, standard, and where low-bit hurts fidelity most."""
    for name, mod in list(model.named_children()):
        if name in skip:
            continue
        if isinstance(mod, nn.Linear):
            setattr(model, name, QuantLinear(mod, bits, group_size))
        else:
            quantize_model(mod, bits, group_size, skip)
    return model


def linear_bytes(model: nn.Module, dtype_bytes: int = 2) -> float:
    return sum(m.weight.numel() * dtype_bytes for m in model.modules() if isinstance(m, nn.Linear))


def quant_linear_bytes(model: nn.Module) -> float:
    return sum(m.weight_bytes() for m in model.modules() if isinstance(m, QuantLinear))


def main():
    torch.manual_seed(0)
    cfg = ActionExpertConfig()
    fp = ActionExpert(cfg).float().eval()
    B, N = 1, 10
    x0 = torch.randn(B, cfg.horizon, cfg.action_dim)
    pkv = fp.encode_prefix(B)
    ref = flow_sample(fp, x0, N, pkv)
    ref_scale = ref.abs().mean().item()

    fp16_bytes = linear_bytes(fp, dtype_bytes=2)
    print(f"action expert: {sum(p.numel() for p in fp.parameters())/1e6:.2f}M params, "
          f"fp16 linear weights = {fp16_bytes/1e6:.2f} MB\n")
    print(f"{'precision':10s}{'weight MB':>11s}{'vs fp16':>9s}{'action rMSE':>13s}{'ceiling*':>10s}")
    print(f"{'fp16':10s}{fp16_bytes/1e6:11.2f}{'1.00x':>9s}{', ':>13s}{'1.00x':>10s}")

    for bits in (8, 4):
        qm = quantize_model(copy.deepcopy(fp), bits)
        qpkv = qm.encode_prefix(B)
        out = flow_sample(qm, x0, N, qpkv)
        rmse = (out - ref).pow(2).mean().sqrt().item()
        rel = rmse / (ref_scale + 1e-9)
        qb = quant_linear_bytes(qm)
        reduction = fp16_bytes / qb
        print(f"int{bits:<7d}{qb/1e6:11.2f}{reduction:8.2f}x{rel:13.4f}{reduction:9.2f}x")

    print("\n* ceiling = weight-traffic reduction = the max speedup achievable in the")
    print("  weight-read-bound regime (batch=1). The Triton fused dequant->GEMM kernel")
    print("  closes the gap to this ceiling; rMSE is the faithfulness cost, reported honestly.")


if __name__ == "__main__":
    main()
