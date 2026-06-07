"""
Reference flow-matching action expert + eager sampler.

Models the per-step cost of a pi0 / GR00T-style VLA action head: a small
transformer over an action *chunk*, conditioned on a (cached) VLM prefix,
integrated over N flow-matching steps.

The crucial structural fact for the kernel thesis: the VLM prefix K/V is
computed ONCE (`encode_prefix`) and reused identically across all N steps.
So the per-step work is just the small action expert over `horizon` tokens,
launched N times -> the "small head, many steps" regime where the sampling
loop is dominated by kernel-launch overhead + weight-read traffic, not FLOPs.
That is exactly the regime kernel fusion is made for and that generic
autoregressive LLM-serving stacks (vLLM / SGLang / TensorRT-LLM) do not target.

Device-agnostic (CPU / MPS / CUDA). Pure PyTorch reference, the Triton
fused-sampler kernel benchmarks *against* this.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionExpertConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    horizon: int = 50        # action chunk length (tokens generated per call)
    action_dim: int = 32     # per-timestep action dimension
    prefix_len: int = 256    # cached VLM prefix tokens (cross-attended each step)
    mlp_ratio: float = 4.0


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0, 1]. t: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Block(nn.Module):
    """Pre-norm block: self-attn over action tokens + cross-attn to cached prefix + MLP."""

    def __init__(self, cfg: ActionExpertConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        self.h, self.hd = h, d // h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.nx = nn.LayerNorm(d)
        self.q_c = nn.Linear(d, d)
        self.kv_c = nn.Linear(d, 2 * d)   # only used to precompute prefix K/V
        self.out_c = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def _split(self, x):  # [B,T,D] -> [B,h,T,hd]
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.hd).transpose(1, 2)

    def _merge(self, x):  # [B,h,T,hd] -> [B,T,D]
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.h * self.hd)

    def precompute_kv(self, prefix: torch.Tensor):
        """prefix [B,P,D] -> (K,V) each [B,h,P,hd], computed once and cached."""
        k, v = self.kv_c(prefix).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def forward(self, x: torch.Tensor, kv_c):
        # self-attention over the action chunk
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        a = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v))
        x = x + self.proj(self._merge(a))
        # cross-attention into the cached VLM prefix
        q = self._split(self.q_c(self.nx(x)))
        kc, vc = kv_c
        a = F.scaled_dot_product_attention(q, kc, vc)
        x = x + self.out_c(self._merge(a))
        # MLP
        x = x + self.mlp(self.n2(x))
        return x


class ActionExpert(nn.Module):
    def __init__(self, cfg: ActionExpertConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.action_dim, d)
        self.t_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cfg.action_dim)
        # Stand-in for the VLM prefix embedding. In a real system this comes from
        # the VLM backbone; it is identical across all N flow steps, hence cached.
        self.register_buffer("_prefix", torch.randn(1, cfg.prefix_len, d) * 0.02, persistent=False)

    def encode_prefix(self, batch: int):
        """Precompute the per-layer prefix K/V once for a batch. Reused every step."""
        prefix = self._prefix.to(self.out_norm.weight.dtype).expand(batch, -1, -1)
        return [blk.precompute_kv(prefix) for blk in self.blocks]

    def forward(self, x_act: torch.Tensor, t: torch.Tensor, prefix_kv):
        """x_act [B,H,A], t [B] -> velocity [B,H,A]."""
        x = self.in_proj(x_act)
        temb = self.t_proj(timestep_embedding(t, self.cfg.d_model).to(x.dtype))
        x = x + temb[:, None, :]
        for blk, kv in zip(self.blocks, prefix_kv):
            x = blk(x, kv)
        return self.out_proj(self.out_norm(x))


@torch.no_grad()
def flow_sample(expert: ActionExpert, x_init: torch.Tensor, n_steps: int, prefix_kv) -> torch.Tensor:
    """Euler integration of the flow field over n_steps. N forward passes through the expert."""
    x = x_init
    dt = 1.0 / n_steps
    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(n_steps):
        v = expert(x, t, prefix_kv)
        x = x + v * dt
        t = t + dt
    return x
