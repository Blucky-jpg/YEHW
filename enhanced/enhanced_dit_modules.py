from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enhanced.global_scheduler import get_global_scheduler


# 
#  Utility functions
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Stable L2 normalisation."""
    return x / (torch.linalg.vector_norm(x, dim=dim, keepdim=True) + eps)


@torch.jit.script
def fast_conv1d_fir(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    return F.conv1d(x, weight, groups=groups)


@torch.jit.script
def safe_log_clamp(x: torch.Tensor, min_val: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp(min=min_val))


#  Δ-rule kernel  (SIGF-PTU)  –  O(N·d)
# Helper that returns triangular masks *once* per (device,dtype,K)
@lru_cache(maxsize=None)
def _get_delta_masks(device: torch.device,
                     dtype: torch.dtype,
                     chunk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri  = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool,  device=device))
    eye  = torch.eye(chunk,              dtype=dtype,             device=device)
    tri_strict = torch.triu(tri, 1)
    return tri, tri_strict, eye


def _delta_inner(q_i, k_i, v_i, kb_i, S, tri_strict, eye):
    """
    Compiled inner routine that works on a *single* chunk.
    Parameters are already (b,h,c,d) or (b,h,d,d) shaped, no loop inside.
    """
    attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict, 0)
    inv = -(kb_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict | eye.bool(), 0) + eye
    # Simplified Woodbury iteration to avoid torch.compile issues
    # Using explicit matrix multiplication instead of complex slicing
    chunk_size = tri_strict.size(-1)
    for i in range(1, chunk_size):
        if i > 0:  # Only process if there are previous elements
            # Get the i-th row up to position i
            inv_row = inv[..., i:i+1, :i]  # (B, H, 1, i)
            # Get the upper-left submatrix
            inv_sub = inv[..., :i, :i]  # (B, H, i, i)
            # Compute update
            update = torch.matmul(inv_row, inv_sub)  # (B, H, 1, i)
            # Apply update
            inv[..., i:i+1, :i] = inv[..., i:i+1, :i] + update
    u_i = (inv @ v_i) - (inv @ kb_i) @ S
    out_chunk = (q_i @ S) + attn_local @ u_i
    S = S + k_i.transpose(-1, -2) @ u_i
    return out_chunk, S


# Temporarily disable torch.compile due to dimension issues
# TODO: Fix the Woodbury iteration for torch.compile compatibility
# _delta_inner = torch.compile(_delta_inner, dynamic=True)


def delta_rule_chunkwise(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         beta: torch.Tensor,
                         *,
                         chunk_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient causal associative Δ-rule – memory-light and Torch-compile friendly.
    Shapes
    ------
    q,k,v : (B, H, L, D)
    beta  : (B, H, L)
    Returns
    -------
    out   : (B, H, L, D)
    S_fin : (B, H, D, D)  (final state, can be fed into the next segment)
    """
    B, H, L, D = q.shape
    device, dtype = q.device, q.dtype

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q    = F.pad(q,    (0, 0, 0, pad_len))
        k    = F.pad(k,    (0, 0, 0, pad_len))
        v    = F.pad(v,    (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    n_chunks = L_pad // chunk_size

    # l2-norm once
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    # reshape into chunks
    reshape_c = lambda t: rearrange(t, 'b h (n c) d -> b h n c d', c=chunk_size)
    q, k, v, k_beta = map(reshape_c, (q, k, v, k_beta))

    # static masks
    tri, tri_strict, eye = _get_delta_masks(device, dtype, chunk_size)

    S = q.new_zeros(B, H, D, D)
    out_chunks = []

    for idx in range(n_chunks):
        out_c, S = _delta_inner(q[:, :, idx],
                                k[:, :, idx],
                                v[:, :, idx],
                                k_beta[:, :, idx],
                                S,
                                tri_strict,
                                eye)
        out_chunks.append(out_c)

    out = torch.cat(out_chunks, dim=2)        # (B,H,L_pad,D)
    out = out[:, :, :L]                       # remove padding
    return out, S


#  Depth-wise FIR convolution
class DepthwiseFIRConv1d(nn.Module):
    """
    Per-head depth-wise causal FIR filtering.
    """
    def __init__(self,
                 num_heads: int,
                 head_dim:  int,
                 kernel_size: int = 7,
                 noise_std:  float = 1e-3):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.groups      = num_heads * head_dim
        # Parameter initialised once; no need to re-view every forward
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0                    # identity tap
        if noise_std > 0:
            filt += noise_std * torch.randn_like(filt)
        self.filters = nn.Parameter(filt)      # (H,D,K)
        # Cached view
        self.register_buffer('_weight_view', None, persistent=False)

    def _weight(self):
        if self._weight_view is None:
            # (H, D, K) → (H*D, 1, K)
            self._weight_view = rearrange(self.filters, 'h d k -> (h d) 1 k')
        return self._weight_view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, H, D)
        """
        B, L, H, D = x.shape
        x_reshaped = rearrange(x, 'b l h d -> b (h d) l')
        x_pad = F.pad(x_reshaped, (self.kernel_size - 1, 0))
        y = fast_conv1d_fir(x_pad, self._weight().to(x.device, x.dtype), self.groups)
        return rearrange(y, 'b (h d) l -> b l h d', h=H, d=D)


#  Cross-head mixing
class CrossHeadMixing(nn.Module):
    """
    Persistent floor cross-head mixing (DeltaNet len_hgate_mixanneal).
    """
    def __init__(self,
                 num_heads: int,
                 mix_init:  float = 0.03,
                 mix_floor: float = 0.01,
                 mix_decay_steps: int = 4_000):
        super().__init__()
        self.num_heads        = num_heads
        self.mix_floor        = float(max(0.0, mix_floor))
        self.mix_decay_steps  = max(1, mix_decay_steps)

        try:
            self.scheduler = get_global_scheduler()
            self.use_global_scheduler = True
        except Exception:
            self.register_buffer('_step', torch.tensor(0, dtype=torch.long), persistent=False)
            self.use_global_scheduler = False

        init = max(self.mix_floor, float(mix_init))
        self.mix_coeff_base = nn.Parameter(torch.full((num_heads,), init))

    # ------------------------------------------------------------------
    def _step_idx(self) -> int:
        return self.scheduler.get_step() if self.use_global_scheduler else int(self._step)

    def _decay_factor(self) -> float:
        return max(0.0, 1.0 - self._step_idx() / self.mix_decay_steps)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, H, D)
        """
        coeff_base = self.mix_coeff_base.clamp(min=self.mix_floor)
        decay      = self._decay_factor()
        coeff_act  = self.mix_floor + decay * (coeff_base - self.mix_floor)

        # If the largest coefficient is basically zero, skip all work
        if torch.all(coeff_act < 1e-8):
            if not self.use_global_scheduler and self.training:
                self._step += 1
            return x

        mean_heads = x.mean(dim=2, keepdim=True)                      # (B,L,1,D)
        x = x + coeff_act.view(1, 1, self.num_heads, 1) * mean_heads

        if not self.use_global_scheduler and self.training:
            self._step += 1
        return x