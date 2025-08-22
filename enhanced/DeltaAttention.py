import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple

from enhanced.quantization import (
    QuantizedLinear,
    flash_attention_with_fp16,
)
from enhanced.progressive_pruning_module import ProgressivePruningSystem
from enhanced.context_adaptive_gating import ContextAdaptiveGating
from enhanced.enhanced_dit_modules import (
    DepthwiseFIRConv1d,
    CrossHeadMixing,
    delta_rule_chunkwise,
)
from enhanced.global_scheduler import get_global_scheduler

class DeltaNetEnhancedAttention(nn.Module):
    """
    Multi-path attention with FIR branches, Delta-rule branch, identity branch,
    progressive pruning and context-adaptive gating.
    Returns
    -------
    attn_out : Tensor (B, N, C)
    aux_loss : Tensor  scalar   (entropy terms from pruning + gating)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 7,
        id_static_init: float = 0.2,
        fusion_hidden_mult: float = 1.0,
        use_flash_attention: bool = True,
        delta_state_rank: Optional[int] = None,  # New: low-rank approximation for delta state
    ):
        super().__init__()
        assert dim % num_heads == 0, "hidden dim must be divisible by heads"

        # Basic geometry
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        self.delta_state_rank = delta_state_rank

        # Global scheduler handle
        self.scheduler = get_global_scheduler()

        # Main projections (FP16-optimized linear layers)
        self.qkv = QuantizedLinear(dim, dim * 3, bias=False)
        self.proj = QuantizedLinear(dim, dim, bias=True)

        # Multi-path processing modules
        self.fir_short = DepthwiseFIRConv1d(
            num_heads, self.head_dim, fir_short_kernel
        )
        self.fir_long = DepthwiseFIRConv1d(
            num_heads, self.head_dim, fir_long_kernel
        )
        self.cross_head_mixing = CrossHeadMixing(num_heads, mix_init=0.02)

        # Routing systems
        self.progressive_pruning = ProgressivePruningSystem(
            dim, num_heads, num_paths=5, use_global_scheduler=True
        )
        self.context_gating = ContextAdaptiveGating(
            dim,
            num_heads,
            self.head_dim,
            num_paths=5,
            fusion_hidden_mult=fusion_hidden_mult,
        )

        # Identity branch
        self.identity_proj = QuantizedLinear(dim, dim, bias=False)

        # Static part of the identity gate (learned, head-wise)
        self.id_static_logit = nn.Parameter(
            torch.full((num_heads,), math.log(id_static_init / (1.0 - id_static_init)))
        )

        # Dynamic part of the identity gate (depends on token content)
        # Use standard linear for optimal compatibility
        self.id_gate_proj = QuantizedLinear(dim, num_heads, bias=True)
        
        with torch.no_grad():
            # Bias initialised to favour "mostly off" at the beginning
            self.id_gate_proj.bias.fill_(-1.5)

        # Optional dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Simple cache for the static gate so we do not call sigmoid every step
        self.register_buffer("_cached_static_gate", None, persistent=False)
        self._cached_step = -1

    # helpers
    def _process_paths_vectorized(
        self,
        v_hnd: torch.Tensor,     # (B, N, H, D)
        identity_out: torch.Tensor, # (B, N, H, D)
        delta_out: torch.Tensor, # (B, N, H, D)
    ) -> List[torch.Tensor]:
        """
        Process the two FIR paths + Delta + original V + identity in one shot.
        Returns list length 5, in the order expected by pruning / gating.
        Input tensors are expected in (B, N, H, D) format.
        """
        # FIR convolutions expect (B, L, H, D) and will be transposed inside.
        fir_short = self.fir_short(v_hnd)
        fir_long  = self.fir_long(v_hnd)

        # Cross-head mixing applied to *each* FIR output
        fir_short = self.cross_head_mixing(fir_short)
        fir_long  = self.cross_head_mixing(fir_long)

        # Return list in fixed order. All tensors are (B, N, H, D)
        return [fir_short, fir_long, delta_out, v_hnd, identity_out]


    def _get_static_gate(self) -> torch.Tensor:
        """
        Returns a  (H,) tensor with the *sigmoid* of the static logits.
        Re-computes only if the global-scheduler step changed.
        """
        step_now = self.scheduler.get_step()
        if self._cached_static_gate is None or step_now != self._cached_step:
            self._cached_static_gate = torch.sigmoid(self.id_static_logit)
            self._cached_step = step_now
        return self._cached_static_gate

    # forward
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, N, C)

        Returns
        -------
        out      : Tensor (B, N, C)
        aux_loss : Tensor (scalar)   – pruning + gating entropy loss
        """
        B, N, C = x.shape

        # ------------ QKV ------------------------------------------
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv_reshaped = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Create contiguous copies instead of views to avoid in-place modification issues
        q = qkv_reshaped[0].contiguous()  # (B, H, N, D)
        k = qkv_reshaped[1].contiguous()  # (B, H, N, D)
        v = qkv_reshaped[2].contiguous()  # (B, H, N, D)

        # ------------ Flash Attention -------------------------------
        #  (returns context in FP16, cast to input dtype afterwards)
        attn_ctx = flash_attention_with_fp16(
            q, k, v, is_causal=True
        )  # (B, H, N, D)
        attn_ctx = attn_ctx.to(dtype=x.dtype)
        attn_ctx = attn_ctx.transpose(1, 2)  # -> (B, N, H, D)

        # ------------ Identity branch ------------------------------
        identity_out = self.identity_proj(x)              # (B, N, C)
        identity_out = rearrange(identity_out, "b n (h d) -> b n h d", h=self.num_heads)

        dyn_gate    = torch.sigmoid(self.id_gate_proj(x))            # (B, N, H)
        static_gate = self._get_static_gate().to(x.device, x.dtype)  # (H,)
        id_gate     = dyn_gate * static_gate.view(1, 1, -1)          # (B, N, H)

        identity_out.mul_(id_gate.unsqueeze(-1))  # gated identity

        # ------------ Delta-rule branch ----------------------------
        # For now, use a simple approximation for beta to avoid memory issues
        # Beta represents the sum of attention weights for each query position
        # Using a simple heuristic: beta decreases along the sequence (causal)
        # This avoids computing the full NxN attention matrix
        
        # Option 1: Use a learned or fixed beta schedule
        # Simple linear decay from 1.0 to 0.1 along the sequence
        beta = torch.linspace(1.0, 0.1, N, device=q.device, dtype=q.dtype)
        beta = beta.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1)  # (B, H, N)
        
        # Option 2: If you really need attention-based beta, use Flash Attention or similar
        # But for now, the approximation above should work for training
        
        delta_out, _ = delta_rule_chunkwise(q, k, v, beta, state_rank=self.delta_state_rank)  # (B, H, N, D)
        # Note: delta_state_rank can be None, which is handled by the function
        delta_out = delta_out.transpose(1, 2)                   # (B, N, H, D)
        

        # The five paths are: [FIR-short, FIR-long, Delta, original V, Identity]
        # This seems to be the intended design.

        # ------------ Vectorised FIR + gather all paths -----------
        # The original code passed `attn_ctx` to `_process_paths_vectorized`, which is incorrect.
        # It should have passed `delta_out`. I have corrected this logic above and
        # will now pass the correct variables to the path processing function.
        # The original `v` tensor is `v_hnd`.
        v_hnd = v.transpose(1, 2) # -> (B, N, H, D)

        path_outputs = self._process_paths_vectorized(
            v_hnd, identity_out, delta_out
        )

        # ------------ Routing (prune + gate) ----------------------
        # 1. Get the initial routing probabilities from the simpler pruning system.
        #    This acts as a prior. We request the weights instead of the combined output.
        pruning_probs, entropy_loss1 = self.progressive_pruning(
            x, path_outputs, return_weights=True
        )

        # 2. The context-adaptive gating module now receives all original paths.
        #    It computes its own sophisticated, statistics-based probabilities and
        #    refines them using the prior probabilities from the pruning system.
        #    It then performs the final combination of paths.
        context_out, entropy_loss2 = self.context_gating(
            x, path_outputs, prior_probs=pruning_probs
        )

        # ------------ Final projection ----------------------------
        out = rearrange(context_out, "b n h d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out, (entropy_loss1 + entropy_loss2)
