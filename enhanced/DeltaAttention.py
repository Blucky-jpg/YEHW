import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Optional

from enhanced.quantization import (
    QuantizedLinear,
    flash_attention_with_bf16,
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
        # QK-Norm parameters
        qk_norm_enabled: bool = True,
        qk_norm_eps: float = 1e-6,
        # Local-Global Attention parameters
        local_global_mixing: bool = True,
        local_attention_window: int = 256,
        global_attention_start_layer: int = 8,
        layer_idx: Optional[int] = None,  # Which layer this attention is in
    ):
        super().__init__()
        assert dim % num_heads == 0, "hidden dim must be divisible by heads"

        # Basic geometry
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        self.delta_state_rank = delta_state_rank

        # QK-Norm parameters
        self.qk_norm_enabled = qk_norm_enabled
        self.qk_norm_eps = qk_norm_eps

        # Local-Global Attention parameters
        self.local_global_mixing = local_global_mixing
        self.local_attention_window = local_attention_window
        self.global_attention_start_layer = global_attention_start_layer
        self.layer_idx = layer_idx

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

    def _apply_qk_normalization(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RMS normalization to queries and keys to stabilize attention computation.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, N, D)

        Returns:
            Tuple of normalized (q, k)
        """
        if not self.qk_norm_enabled:
            return q, k

        # RMS normalization for queries: q / sqrt(mean(q^2) + eps)
        q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + self.qk_norm_eps)
        q_normalized = q / q_rms

        # RMS normalization for keys: k / sqrt(mean(k^2) + eps)
        k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + self.qk_norm_eps)
        k_normalized = k / k_rms

        return q_normalized, k_normalized

    def _windowed_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           window_size: int) -> torch.Tensor:
        """
        Compute windowed attention within local windows for memory efficiency.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, N, D)
            v: Value tensor (B, H, N, D)
            window_size: Size of the attention window

        Returns:
            Attention output (B, H, N, D)
        """
        B, H, N, D = q.shape

        if N <= window_size:
            # If sequence is smaller than window, use global attention
            return self._global_attention(q, k, v)

        # Compute attention within windows
        output = torch.zeros_like(q)
        overlap = window_size // 4  # Small overlap between windows

        for start_idx in range(0, N, window_size - overlap):
            end_idx = min(start_idx + window_size, N)

            # Extract window
            q_window = q[:, :, start_idx:end_idx]
            k_window = k[:, :, start_idx:end_idx]
            v_window = v[:, :, start_idx:end_idx]

            # Compute attention within window
            window_output = self._global_attention(q_window, k_window, v_window)

            # Add to output
            output[:, :, start_idx:end_idx] = window_output

        return output

    def _global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Standard global attention computation.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, N, D)
            v: Value tensor (B, H, N, D)

        Returns:
            Attention output (B, H, N, D)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for autoregressive behavior
        if self.training:
            causal_mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1],
                                              device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        return torch.matmul(attn_weights, v)

    def _local_global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Route between local and global attention based on layer index.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, N, D)
            v: Value tensor (B, H, N, D)

        Returns:
            Attention output (B, H, N, D)
        """
        if not self.local_global_mixing or self.layer_idx is None:
            # Fallback to global attention
            return self._global_attention(q, k, v)

        # Use local attention for early layers, global for later layers
        if self.layer_idx < self.global_attention_start_layer:
            return self._windowed_attention(q, k, v, self.local_attention_window)
        else:
            return self._global_attention(q, k, v)

    # forward
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-optimized forward pass with sequential path computation.

        Parameters
        ----------
        x : Tensor (B, N, C)

        Returns
        -------
        out      : Tensor (B, N, C)
        aux_loss : Tensor (scalar)   – pruning + gating entropy loss
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        # Memory monitoring
        memory_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # ------------ QKV ------------------------------------------ (Memory: Low)
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv_reshaped = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Create contiguous copies instead of views to avoid in-place modification issues
        q = qkv_reshaped[0].contiguous()  # (B, H, N, D)
        k = qkv_reshaped[1].contiguous()  # (B, H, N, D)
        v = qkv_reshaped[2].contiguous()  # (B, H, N, D)

        # Clean up QKV immediately to save memory
        del qkv, qkv_reshaped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------ QK Normalization ------------------------------ (Memory: Low)
        # Apply RMS normalization to stabilize attention computation
        q_normalized, k_normalized = self._apply_qk_normalization(q, k)

        # ------------ Local-Global Attention ------------------------- (Memory: Medium)
        # Route between local and global attention based on layer
        if self.use_flash_attention and not self.local_global_mixing:
            # Use existing flash attention for backward compatibility
            attn_ctx = flash_attention_with_bf16(
                q_normalized, k_normalized, v, is_causal=True
            )  # (B, H, N, D)
            attn_ctx = attn_ctx.to(dtype=dtype)
            attn_ctx = attn_ctx.transpose(1, 2)  # -> (B, N, H, D)
        else:
            # Use local-global attention routing
            attn_ctx = self._local_global_attention(q_normalized, k_normalized, v)
            attn_ctx = attn_ctx.transpose(1, 2)  # -> (B, N, H, D)

        # ------------ Sequential Path Processing ------------------- (Memory: Optimized)
        # Process paths one by one to reduce peak memory usage

        # Convert v for path processing
        v_hnd = v.transpose(1, 2)  # (B, N, H, D)

        # Initialize path outputs list
        path_outputs = []

        # 1. FIR Short Path (Lightweight)
        fir_short = self.fir_short(v_hnd)
        fir_short = self.cross_head_mixing(fir_short)
        path_outputs.append(fir_short)

        # Clean up intermediate
        del fir_short
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. FIR Long Path (Medium weight)
        fir_long = self.fir_long(v_hnd)
        fir_long = self.cross_head_mixing(fir_long)
        path_outputs.append(fir_long)

        # Clean up intermediate
        del fir_long
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Delta-rule Path (Memory intensive - process carefully)
        # Use simple beta approximation to reduce memory
        beta = torch.linspace(1.0, 0.1, N, device=device, dtype=dtype)
        beta = beta.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1)  # (B, H, N)

        delta_out, _ = delta_rule_chunkwise(q, k, v, beta, state_rank=self.delta_state_rank)
        delta_out = delta_out.transpose(1, 2)  # (B, N, H, D)
        path_outputs.append(delta_out)

        # Clean up delta computation
        del delta_out, beta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Original V Path (Lightweight)
        path_outputs.append(v_hnd)

        # 5. Identity Path (Medium weight)
        identity_out = self.identity_proj(x)  # (B, N, C)
        identity_out = rearrange(identity_out, "b n (h d) -> b n h d", h=self.num_heads)

        dyn_gate = torch.sigmoid(self.id_gate_proj(x))  # (B, N, H)
        static_gate = self._get_static_gate().to(device, dtype)  # (H,)
        id_gate = dyn_gate * static_gate.view(1, 1, -1)  # (B, N, H)

        identity_out.mul_(id_gate.unsqueeze(-1))  # gated identity
        path_outputs.append(identity_out)

        # Clean up identity computation
        del identity_out, dyn_gate, id_gate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up base tensors no longer needed
        del v_hnd, attn_ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------ Routing (prune + gate) ---------------------- (Memory: Low)
        # 1. Get the initial routing probabilities from the simpler pruning system.
        #    This acts as a prior. We request the weights instead of the combined output.
        pruning_probs, entropy_loss1 = self.progressive_pruning(
            x, path_outputs, return_weights=True
        )

        # Clean up unused path outputs to save memory
        del path_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. The context-adaptive gating module now receives all original paths.
        #    It computes its own sophisticated, statistics-based probabilities and
        #    refines them using the prior probabilities from the pruning system.
        #    It then performs the final combination of paths.
        context_out, entropy_loss2 = self.context_gating(
            x, path_outputs=None, prior_probs=pruning_probs
        )

        # ------------ Final projection ---------------------------- (Memory: Low)
        out = rearrange(context_out, "b n h d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_dropout(out)

        # Clean up final computation
        del context_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Monitor memory usage
        memory_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        if memory_after > memory_before + 1.0:  # More than 1GB increase
            print(".1f")

        return out, (entropy_loss1 + entropy_loss2)
