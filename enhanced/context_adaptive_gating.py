# ----------------------------------------------------------------------
#  context_adaptive_gating.py   (patched)
# ----------------------------------------------------------------------
from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..core.quantization import QuantizedLinear
from .global_scheduler import get_global_scheduler


#  Helper: cached triangular indices  (device / P specific)
@lru_cache(maxsize=None)
def _cached_triu_idx(P: int, device: torch.device) -> torch.Tensor:
    """Return upper-triangular indices (excluding diag) as a 2×N tensor."""
    return torch.triu_indices(P, P, offset=1, device=device)  # (2, N)


#  Context-Adaptive Gating
class ContextAdaptiveGating(nn.Module):
    """
    DeltaNet-CAGF-DPAF-EASH  –  context-conditioned routing.
    Input
        hidden_states : (B, L, hidden_size)
        path_outputs  : list[(B, L, H, D)]  – up to `num_paths` tensors
    Return
        combined      : (B, L, H, D)
        reg_loss      : scalar tensor
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads:   int,
                 head_dim:    int,
                 num_paths:   int = 5,
                 temp_init:   float = 1.0,
                 fusion_hidden_mult: float = 1.0,
                 floor_cfg: dict | None = None,
                 entropy_cfg: dict | None = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.num_paths   = num_paths

        # Learnable temperature per head (stored in log-space)
        self.log_temp = nn.Parameter(torch.full((num_heads, 1),
                                                math.log(max(temp_init, 1e-4))))

        # Feature dimensions
        self.stat_dim  = 4                                   # mean, var, |mean|, l2
        stats_dim      = num_heads * self.stat_dim * num_paths
        norms_dim      = num_heads * num_paths
        pairwise_dim   = num_heads * (num_paths * (num_paths - 1) // 2)

        gate_in_dim    = hidden_size + stats_dim + norms_dim + pairwise_dim
        gate_hid_dim   = max(1, int(hidden_size * fusion_hidden_mult) // 2)

        self.context_gate = nn.Sequential(
            QuantizedLinear(gate_in_dim, gate_hid_dim, bias=True),
            nn.GELU(),
            QuantizedLinear(gate_hid_dim, num_heads * num_paths, bias=True)
        )

        floor_cfg   = floor_cfg   or {}
        entropy_cfg = entropy_cfg or {}

        self.floor  = DualPhasePathFloor(num_heads=num_heads,
                                         num_paths=num_paths,
                                         **floor_cfg)
        self.entropy_reg = EntropyAnnealedRegularization(**entropy_cfg)

        self._init_biases()

    def _init_biases(self):
        """Bias the last linear layer towards value/identity paths."""
        with torch.no_grad():
            bias = self.context_gate[-1].bias              # (H*P,)
            bias.zero_()
            if self.num_paths >= 5:
                bias_matrix = bias.view(self.num_heads, self.num_paths)
                bias_matrix[:, 3] = 1.0    # value path
                bias_matrix[:, 4] = 2.0    # identity path

    #  Feature helpers
    def _pairwise_diffs(self,
                        path_stack: torch.Tensor,
                        P: int) -> torch.Tensor:
        """
        path_stack : (B, L, H, D, P) of *actual* paths (P ≤ self.num_paths)
        Return     : (B, L, H, num_pairs)  distances (no sqrt for speed)
        """
        if P <= 1:
            B, L, H = path_stack.shape[:3]
            return path_stack.new_zeros(B, L, H, 0)

        B, L, H, D, _ = path_stack.shape
        flat = rearrange(path_stack, 'b l h d p -> (b l h) d p')     # (BLH, D, P)

        #  ||a||^2 for each path
        norms2 = (flat * flat).sum(dim=1)                            # (BLH, P)
        dots   = flat.transpose(1, 2) @ flat                         # (BLH, P, P)
        dist2  = norms2.unsqueeze(1) + norms2.unsqueeze(2) - 2 * dots
        dist2  = dist2.clamp(min=0.0)

        idx = _cached_triu_idx(P, flat.device)
        pair_flat = dist2[:, idx[0], idx[1]]                         # (BLH, N)

        return rearrange(pair_flat, '(b l h) n -> b l h n', b=B, l=L, h=H)

    def _path_stats(self,
                    path_outputs: List[torch.Tensor],
                    B: int,
                    L: int,
                    device,
                    dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return
            stats_flat : (B, L, H, P*stat_dim)
            norms_flat : (B, L, H, P)
            pair_diffs : (B, L, H, P*(P-1)//2)
        All tensors are *already padded* to self.num_paths.
        """
        P_act = min(len(path_outputs), self.num_paths)

        # When no paths are active, return all-zero placeholders
        if P_act == 0:
            zeros_stats = torch.zeros(B, L, self.num_heads,
                                      self.num_paths * self.stat_dim,
                                      dtype=dtype, device=device)
            zeros_norm  = torch.zeros(B, L, self.num_heads,
                                      self.num_paths,
                                      dtype=dtype, device=device)
            zeros_pair  = torch.zeros(B, L, self.num_heads,
                                      self.num_paths * (self.num_paths - 1) // 2,
                                      dtype=dtype, device=device)
            return zeros_stats, zeros_norm, zeros_pair

        # Stack paths once
        path_stack = torch.stack(path_outputs[:P_act], dim=-1)       # (B,L,H,D,P_act)

        # Vectorised statistics
        mean     = path_stack.mean(dim=-2, keepdim=True)
        var      = path_stack.var(dim=-2, unbiased=False, keepdim=True)
        abs_mean = path_stack.abs().mean(dim=-2, keepdim=True)
        l2_norm  = path_stack.norm(dim=-2, keepdim=True)

        stats = torch.cat((mean, var, abs_mean, l2_norm), dim=-2)    # (B,L,H,4,D,P_act)
        stats_flat = rearrange(stats, 'b l h s d p -> b l h (p s)')  # (B,L,H,P_act*4)

        # Path norms
        norms_flat = path_stack.norm(dim=-2)                         # (B,L,H,P_act)

        # Pairwise distances
        pair_diffs = self._pairwise_diffs(path_stack, P_act)        # (B,L,H,N_pairs_act)

        # Pad to full dimension if necessary
        if P_act < self.num_paths:
            pad_paths = self.num_paths - P_act
            stats_pad = torch.zeros(B, L, self.num_heads,
                                    pad_paths * self.stat_dim,
                                    dtype=dtype, device=device)
            norms_pad = torch.zeros(B, L, self.num_heads,
                                    pad_paths,
                                    dtype=dtype, device=device)

            stats_flat = torch.cat([stats_flat, stats_pad], dim=-1)
            norms_flat = torch.cat([norms_flat, norms_pad], dim=-1)

            exp_pairs = self.num_paths * (self.num_paths - 1) // 2
            pair_pad  = torch.zeros(B, L, self.num_heads,
                                    exp_pairs - pair_diffs.shape[-1],
                                    dtype=dtype, device=device)
            pair_diffs = torch.cat([pair_diffs, pair_pad], dim=-1)

        return stats_flat, norms_flat, pair_diffs

    def forward(
        self,
        hidden_states: torch.Tensor,
        path_outputs:   List[torch.Tensor],
        prior_probs: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L, _  = hidden_states.shape
        device   = hidden_states.device
        dtype    = hidden_states.dtype

        # 1. statistics exactly as before
        stats_flat, norms_flat, pair_diffs = self._path_stats(
            path_outputs, B, L, device, dtype
        )

        stats_flat = rearrange(stats_flat, 'b l h s -> b l (h s)')
        norms_flat = rearrange(norms_flat, 'b l h p -> b l (h p)')
        pair_flat  = rearrange(pair_diffs, 'b l h p -> b l (h p)')

        gate_in = torch.cat((hidden_states, stats_flat,
                            norms_flat, pair_flat), dim=-1)

        # 2. raw logits → probabilities
        logits = self.context_gate(gate_in).view(B, L,
                                                self.num_heads,
                                                self.num_paths)

        temp   = torch.exp(self.log_temp).unsqueeze(0).unsqueeze(0)  # (1,1,H,1)
        logits = logits / temp

        probs  = torch.softmax(logits, dim=-1)

        # 3. scheduled ε-floor
        probs = self.floor(probs)

        # 3.5. Combine with prior probabilities if provided
        if prior_probs is not None:
            # Multiply probabilities and re-normalize
            # This allows the pruning system to act as a "prior"
            probs = probs * prior_probs
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # 4. entropy-annealed regularisation term
        reg_loss = self.entropy_reg(probs)

        # 5. path aggregation
        P_act      = min(len(path_outputs), self.num_paths)
        if P_act == 0:
             # Handle case where no paths are active
            head_dim = self.head_dim
            return torch.zeros(B, L, self.num_heads, head_dim, device=device, dtype=dtype), reg_loss

        path_stack = torch.stack(path_outputs[:P_act], dim=-1)        # (B,L,H,D,P)
        combined   = (path_stack *
                    probs[..., :P_act].unsqueeze(-2)).sum(dim=-1)   # (B,L,H,D)

        return combined, reg_loss


class DualPhasePathFloor(nn.Module):
    """
    Dual-phase ε-flooring.  Phase-1 keeps ε high,
    phase-2 decays it, both controlled by the global scheduler.
    """

    def __init__(self,
                 num_heads: int,
                 num_paths: int,
                 schedule_name: str = "epsilon_floor",
                 use_global_scheduler: bool = True):
        super().__init__()
        self.num_heads  = num_heads
        self.num_paths  = num_paths
        self.schedule_name = schedule_name
        self.use_global_scheduler = use_global_scheduler

        if not use_global_scheduler:
            # Local buffer + counters (same shape as before)
            self.register_buffer("_local_step", torch.tensor(0, dtype=torch.long),
                                 persistent=False)
            self.eps_start = 0.10
            self.eps_final = 0.025
            self.decay_steps = 4000

    def _current_eps(self) -> float:
        if self.use_global_scheduler:
            return get_global_scheduler().get_value(self.schedule_name)
        # local fallback
        step = float(self._local_step.item())
        if step >= self.decay_steps:
            return self.eps_final
        prog = step / max(1.0, self.decay_steps)
        return self.eps_start + prog * (self.eps_final - self.eps_start)

    def forward(self,
                probs: torch.Tensor,
                local_path_idx: int | None = None) -> torch.Tensor:
        """
        probs: (B, L, H, P) – output of softmax
        """

        eps = self._current_eps()
        eps = torch.as_tensor(eps, dtype=probs.dtype,
                              device=probs.device)

        # Make sure total mass added ≤ 1.
        floor_val = eps / probs.size(-1)
        probs = torch.clamp(probs, min=floor_val)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if not self.use_global_scheduler and self.training:
            self._local_step += 1

        return probs


class EntropyAnnealedRegularization(nn.Module):
    def __init__(self,
                 coeff_schedule: str = "entropy_coeff",
                 use_global_scheduler: bool = True):
        super().__init__()
        self.coeff_schedule = coeff_schedule
        self.use_global_scheduler = use_global_scheduler
        if not use_global_scheduler:
            self.register_buffer("_local_step", torch.tensor(0, dtype=torch.long),
                                 persistent=False)
            self.coeff_start  = 0.02
            self.coeff_final  = 0.0
            self.decay_steps  = 4000

    def _current_coeff(self, device, dtype) -> torch.Tensor:
        if self.use_global_scheduler:
            c = get_global_scheduler().get_value(self.coeff_schedule)
        else:
            step = float(self._local_step.item())
            if step >= self.decay_steps:
                c = self.coeff_final
            else:
                prog = step / max(1.0, self.decay_steps)
                c = self.coeff_start + prog * (self.coeff_final - self.coeff_start)
            if self.training:
                self._local_step += 1
        return torch.as_tensor(c, device=device, dtype=dtype)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Returns a *scalar* reg-loss tensor."""
        if not self.training:
            return torch.tensor(0.0, device=probs.device,
                                dtype=probs.dtype)

        p = probs.clamp(min=1e-8)
        entropy = -(p * torch.log(p)).sum(dim=-1).mean()

        coeff = self._current_coeff(probs.device, probs.dtype)
        return -coeff * entropy