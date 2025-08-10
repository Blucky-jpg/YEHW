import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from core.quantization import QuantizedLinear
from enhanced.global_scheduler import get_global_scheduler, register_default_schedules

class ProgressivePruningSystem(nn.Module):
    """
    Head-wise, multi-path routing / pruning module.

    Inputs
    ------
    hidden_states : Tensor (B, L, D)
    path_outputs  : list[Tensor]  each (B, L, H, D')
                    where D' = hidden_dim / num_heads
    Returns
    -------
    combined      : Tensor (B, L, H, D')
    entropy_loss  : scalar Tensor
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads:   int,
        num_paths:   int = 5,
        min_active_paths: int = 2,
        soft_pruning: bool = True,
        gumbel_temperature: float = 1.0,
        use_global_scheduler: bool = True,
    ):
        super().__init__()

        # ---------------- basic params ---------------------------
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.min_active_paths = max(1, min(min_active_paths, num_paths))
        self.soft_pruning = bool(soft_pruning)
        self.gumbel_temperature = float(gumbel_temperature)
        self.use_global_scheduler = bool(use_global_scheduler)

        self.soft_contribution = 0.30   # weight of soft branch when mixing

        # ---------------- scheduler handle -----------------------
        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()     # singleton
        else:
            self.register_buffer("_step", torch.zeros(1, dtype=torch.long),
                                 persistent=False)

        # ---------------- learnable temperature ------------------
        self.gate_log_temp = nn.Parameter(torch.zeros(num_heads))

        # ---------------- routing MLP ----------------------------
        self.routing_gate = nn.Sequential(
            QuantizedLinear(hidden_size, hidden_size * 2, bias=True),
            nn.GELU(),
            QuantizedLinear(hidden_size * 2, num_heads * num_paths, bias=True),
        )
        self._init_routing_bias()

        # ---------------- adaptive floor -------------------------
        self.adaptive_floor = TokenAdaptiveFloor(
            num_heads=num_heads,
            num_paths=num_paths,
            use_global_scheduler=use_global_scheduler,
        )

        # ---------------- tiny runtime cache ---------------------
        self._cache = {"step": -1, "prune_threshold": None, "entropy_coeff": None}

    # helpers
    @property
    def _current_step(self) -> int:
        if self.use_global_scheduler:
            return self.scheduler.get_step()
        return int(self._step.item())

    def _init_routing_bias(self) -> None:
        """Bias paths 3 and 4 (direct V + identity) to be selected early on."""
        with torch.no_grad():
            bias = self.routing_gate[-1].bias      # shape (H * P)
            bias.zero_()
            b = bias.view(self.num_heads, self.num_paths)
            if self.num_paths > 3:
                b[:, 3] = 1.0     # direct-V path
            if self.num_paths > 4:
                b[:, 4] = 2.0     # identity path

    # scheduler-driven values (cached each step)
    def _update_cache(self) -> None:
        step = self._current_step
        if step == self._cache["step"]:
            return

        self._cache["step"] = step
        if self.use_global_scheduler:
            self._cache["prune_threshold"] = self.scheduler.get_value("prune_threshold")
            self._cache["entropy_coeff"]   = self.scheduler.get_value("entropy_coeff")
            return

        # -------- local fallback schedule (if no global scheduler) -----
        t = float(step)
        # pruning threshold ramps 0  →  1e-3 between steps 2k-4k
        if t <= 2_000:
            self._cache["prune_threshold"] = 0.0
        elif t >= 4_000:
            self._cache["prune_threshold"] = 1e-3
        else:
            self._cache["prune_threshold"] = (t - 2_000) / 2_000 * 1e-3

        # entropy coefficient decays 0.02 → 0 between 0-4k
        if t >= 4_000:
            self._cache["entropy_coeff"] = 0.0
        else:
            self._cache["entropy_coeff"] = 0.02 * (1 - t / 4_000)

    # differentiable helpers
    def _apply_gumbel_noise(self, logits: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.soft_pruning):
            return logits

        u = torch.rand_like(logits)
        eps = torch.finfo(logits.dtype).eps
        u = u.clamp_(min=eps, max=1.0 - eps)
        g = -torch.log(-torch.log(u))
        return logits + g * self.gumbel_temperature

    def _ensure_min_paths(
        self, probs: torch.Tensor, original_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        probs : (B, L, H, P) after pruning
        Ensures each (B,L,H) slice has at least `min_active_paths` non-zero
        entries (uses top-k from original probs).
        """
        active = (probs > 1e-6).sum(dim=-1)                # (B, L, H)
        mask   = active < self.min_active_paths            # bool

        if not mask.any():
            return probs

        # indices of top-k w.r.t original_probs
        _, topk_idx = torch.topk(original_probs, self.min_active_paths, dim=-1)
        topk_mask = torch.zeros_like(probs)
        topk_mask.scatter_(-1, topk_idx, 1.0)

        uniform = topk_mask / self.min_active_paths        # each active = 1/k
        probs   = torch.where(mask.unsqueeze(-1), uniform, probs)
        return probs

    def _combine_paths(
        self,
        paths: List[torch.Tensor],     # each (B, L, H, D')
        weights: torch.Tensor,         # (B, L, H, P_used)
        out_shape: Tuple[int, ...],    # (B, L, H, D')
    ) -> torch.Tensor:
        P = weights.shape[-1]
        if P == 1:
            return paths[0] * weights[..., 0].unsqueeze(-1)

        if P <= 3:
            stack = torch.stack(paths[:P], dim=-1)         # (B,L,H,D',P)
            return (stack * weights.unsqueeze(-2)).sum(dim=-1)

        # generic (no big intermediate tensor)
        out = torch.zeros(out_shape, device=weights.device, dtype=weights.dtype)
        for p in range(P):
            out += paths[p] * weights[..., p].unsqueeze(-1)
        return out

    # forward
    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, L, D)
        path_outputs:  List[torch.Tensor],    # each (B, L, H, D')
        *,
        return_weights: bool = False,         # if True, return routing weights instead of combined output
        profile: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L, D = hidden_states.shape
        head_dim = D // self.num_heads
        zero = hidden_states.new_zeros(())

        # ---------------- guard ---------------------------------
        if not path_outputs:
            empty = hidden_states.new_zeros(B, L, self.num_heads, head_dim)
            return empty, zero

        P_avail = min(len(path_outputs), self.num_paths)

        # ---------------- routing logits ------------------------
        logits = self.routing_gate(hidden_states)              # (B, L, H*P)
        logits = logits.view(B, L, self.num_heads, self.num_paths)

        # temperature (learned per head)
        temp = F.softplus(self.gate_log_temp).clamp_(min=1e-4)  # (H,)
        logits = logits / temp.view(1, 1, -1, 1)

        # ---------------- pruning --------------------------------
        probs = F.softmax(logits, dim=-1)                      # (B,L,H,P)

        self._update_cache()
        thresh = self._cache["prune_threshold"]
        entropy_coeff = self._cache["entropy_coeff"]

        entropy_loss = zero

        if thresh > 0.0:
            # ---- optional soft pruning with Gumbel -------------
            if self.training and self.soft_pruning:
                noisy = self._apply_gumbel_noise(logits)
                soft  = F.softmax(noisy / self.gumbel_temperature, dim=-1)

                hard_mask = (probs > thresh).float()
                probs = (
                    (1 - self.soft_contribution) * probs * hard_mask
                    + self.soft_contribution * soft
                )
            else:
                probs = probs * (probs > thresh).float()

            # renormalise
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # guarantee min_active_paths
        probs = self._ensure_min_paths(probs, probs.detach())
        probs = self.adaptive_floor(probs)            # final tweak

        # ---------------- entropy regularisation -----------------
        if self.training and entropy_coeff > 0.0:
            p_safe = probs.clamp(min=1e-9)
            ent = -(p_safe * p_safe.log()).sum(dim=-1).mean()
            entropy_loss = -entropy_coeff * ent

        # ---------------- combine outputs or return weights ------------------------
        weights = probs[..., :P_avail]                # (B,L,H,P_used)
        if return_weights:
            out_tensor = weights  # Caller wants the routing weights
        else:
            out_tensor = self._combine_paths(
                path_outputs[:P_avail],
                weights,
                (B, L, self.num_heads, head_dim),
            )

        # ---------------- local step increment -------------------
        if not self.use_global_scheduler and self.training:
            self._step += 1

        return out_tensor, entropy_loss


class TokenAdaptiveFloor(nn.Module):
    """
    DeltaNet TAPR:  ε-floor that adapts per token and is optionally tied
    to a global training schedule.

    Args
    ----
    num_heads : int
    num_paths : int
    floor_start / floor_end : float     maximum ε at step 0 and after
                                    `floor_decay_steps`
    floor_decay_steps       : int       linear decay length (only when no global scheduler is used)
    use_global_scheduler    : bool
    """

    def __init__(
        self,
        num_heads: int,
        num_paths: int,
        floor_start: float = 0.05,
        floor_end: float = 0.0,
        floor_decay_steps: int = 3000,
        use_global_scheduler: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_paths = num_paths
        self.floor_start = float(floor_start)
        self.floor_end   = float(floor_end)
        self.floor_decay_steps = float(floor_decay_steps)
        self.use_global_scheduler = bool(use_global_scheduler)

        # external scheduler handle  --------------------------------------
        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()
            # make sure the schedule exists; otherwise register defaults
            try:
                self.scheduler.get_value("token_floor")
            except (ValueError, AttributeError):
                register_default_schedules(self.scheduler)
        else:
            self.register_buffer(
                "_step", torch.zeros(1, dtype=torch.long), persistent=False
            )

        # learnable ε-base logits (per head / per path) -------------------
        eps_logit_init = -12.0
        self.gate_eps_logit = nn.Parameter(
            torch.full((num_heads, num_paths), eps_logit_init)
        )

        # tiny cache to avoid querying the scheduler at every token --------
        self._floor_cache = {"value": None, "step": -1, "dtype": torch.float32}

    # helpers
    @property
    def _step_int(self) -> int:
        if self.use_global_scheduler:
            return self.scheduler.get_step()
        return int(self._step.item())

    def _current_floor_max(self, dtype: torch.dtype) -> float:
        """
        Get the max ε allowed at the current training step
        (pulled from the scheduler or computed locally).
        Uses caching so the expensive call runs only when the step changes.
        """
        step = self._step_int
        cache_hit = (
            step == self._floor_cache["step"]
            and self._floor_cache["dtype"] == dtype
            and self._floor_cache["value"] is not None
        )
        if cache_hit:
            return self._floor_cache["value"]

        if self.use_global_scheduler:
            try:
                val = float(self.scheduler.get_value("token_floor"))
            except (ValueError, AttributeError):
                val = 0.0
        else:
            if step >= self.floor_decay_steps:
                val = self.floor_end
            else:
                ratio = step / self.floor_decay_steps
                val = self.floor_start + ratio * (self.floor_end - self.floor_start)

        # update cache
        self._floor_cache.update({"value": val, "step": step, "dtype": dtype})
        return val

    # forward (Torch-compile wrapped later for speed)
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Apply ε-floor token-wise.

        Parameters
        ----------
        probs : Tensor  (B, L, H, P)   probabilities after routing softmax

        Returns
        -------
        Tensor of same shape with ε-floor applied (sums still equal 1.0).
        """
        dtype   = probs.dtype
        device  = probs.device
        eps_max = self._current_floor_max(dtype)

        # common shortcut: floor already at 0 → no-op
        if eps_max <= 0.0:
            if not self.use_global_scheduler and self.training:
                self._step += 1
            return probs

        # avoid creating large temporaries by re-using buffers
        with torch.no_grad():
            p_max = probs.max(dim=-1, keepdim=True).values        # (B,L,H,1)
        uncertainty = 1.0 - p_max

        eps_base = torch.sigmoid(self.gate_eps_logit).to(device=device, dtype=dtype)
        eps_base = eps_base.view(1, 1, self.num_heads, self.num_paths)

        eps = eps_max * uncertainty * eps_base   # (B,L,H,P)
        eps_sum = eps.sum(dim=-1, keepdim=True)  # (B,L,H,1)

        #  out = probs * (1-eps_sum) + eps   (done in a memory-friendly manner)
        probs = probs * (1.0 - eps_sum) + eps
        probs = probs.clamp(min=1e-9, max=1.0)

        if not self.use_global_scheduler and self.training:
            self._step += 1

        return probs