import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
from .global_scheduler import get_global_scheduler

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    L2 normalization with optional epsilon for numerical stability.
    
    Args:
        x: Input tensor
        dim: Dimension along which to normalize (default: -1, last dimension)
        eps: Small value to prevent division by zero (default: 1e-12)
    
    Returns:
        L2 normalized tensor
    """
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)

# --- Performance Optimizations ---

@torch.jit.script 
def fast_conv1d_fir(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """JIT-optimized FIR convolution for better performance."""
    return F.conv1d(x, weight, groups=groups)

@torch.jit.script
def safe_log_clamp(x: torch.Tensor, min_val: float = 1e-8) -> torch.Tensor:
    """Safe logarithm with clamping for numerical stability."""
    return torch.log(x.clamp(min=min_val))

# Delta-rule kernel from SIGF-PTU (O(N) complexity)
@torch.compile(dynamic=True)
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient causal associative Δ-rule (O(N·d)) via fixed-size chunks."""
    b, h, L, d_k = q.shape
    device, dtype = q.device, q.dtype
    
    # Padding optimization - compute once
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalize and apply beta scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape to chunks
    chunk_fn = lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size)
    q, k, v, k_beta = map(chunk_fn, (q, k, v, k_beta))

    # Pre-compute masks (cached on device)
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), 0)
    tri_strict = torch.triu(tri, 1)
    eye = torch.eye(chunk_size, dtype=dtype, device=device)

    # Compute inverse operator
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    
    # Iterative inverse computation with in-place operations
    for i in range(1, chunk_size):
        inv_slice = inv[..., i, :]
        inv[..., i, :i] += (inv_slice[..., None] * inv[..., :, :i]).sum(-2)
    inv = inv + eye

    # Precompute u and w
    u = inv @ v
    w = inv @ k_beta

    # Initialize state and output
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    # Process chunks sequentially
    num_chunks = L_pad // chunk_size
    for blk in range(num_chunks):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        
        # Local attention (causal masking)
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        
        # Update computation
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :, blk] = q_i @ S + attn_local @ u_i
        
        # State update
        S = S + k_i.transpose(-1, -2) @ u_i

    # Reshape and unpad
    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S


class DepthwiseFIRConv1d(nn.Module):
    """
    Depth-wise FIR convolution adapted from DeltaNet for DiT attention enhancement.
    Provides per-head temporal filtering with O(N) complexity.
    """
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 7, noise_std: float = 1e-3):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))  # Ensure positive kernel size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.groups = num_heads * head_dim
        
        # Initialize with Dirac delta (identity) + small noise
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # Identity tap at the end
        if noise_std > 0:
            filt += noise_std * torch.randn_like(filt)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for (B, L, H, D) tensors.
        Uses optimized conv1d with pre-computed weight reshaping.
        """
        b, l, h, d = x.shape
        
        # Reshape input: (B, L, H, D) -> (B, H*D, L)
        x_reshaped = rearrange(x, "b l h d -> b (h d) l")
        
        # Reshape filters: (H, D, K) -> (H*D, 1, K)
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        
        # Apply causal padding
        x_padded = F.pad(x_reshaped, (self.kernel_size - 1, 0))
        
        # Efficient grouped convolution
        y = fast_conv1d_fir(x_padded, weight, self.groups)
        
        # Reshape back: (B, H*D, L) -> (B, L, H, D)
        return rearrange(y, "b (h d) l -> b l h d", h=h, d=d)


class CrossHeadMixing(nn.Module):
    """
    Cross-head mixing with persistent floor from DeltaNet len_hgate_mixanneal.
    Maintains inter-head cooperation throughout training.
    Uses global scheduler for consistent step tracking.
    """
    def __init__(self, num_heads: int, mix_init: float = 0.03, mix_floor: float = 0.01, 
                 mix_decay_steps: int = 4000):
        super().__init__()
        self.num_heads = num_heads
        self.mix_floor = max(0.0, float(mix_floor))
        self.mix_decay_steps = max(1, mix_decay_steps)
        
        # Get global scheduler for consistent step tracking
        try:
            self.scheduler = get_global_scheduler()
            self.use_global_scheduler = True
        except (ImportError, RuntimeError):
            # Fallback if global scheduler is unavailable
            self.register_buffer('_step', torch.tensor(0, dtype=torch.long), persistent=False)
            self.use_global_scheduler = False
        
        # Initialize mixing coefficients
        mix_init = max(self.mix_floor, float(mix_init))
        self.mix_coeff_base = nn.Parameter(torch.full((num_heads,), mix_init))

    def _get_current_step(self) -> int:
        """Get current training step from scheduler or internal counter."""
        if self.use_global_scheduler:
            return self.scheduler.get_step()
        else:
            return int(self._step.item())

    def _decay_factor(self) -> float:
        """Compute decay factor for mixing coefficient using global scheduler."""
        t = float(self._get_current_step())
        return max(0.0, 1.0 - t / self.mix_decay_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-head mixing with persistent floor.
        Args:
            x: (B, L, H, D) multi-head features
        Returns:
            Mixed features with same shape as input
        """
        # Compute current mixing coefficient
        mix_factor = self._decay_factor()
        coeff_base = self.mix_coeff_base.clamp(min=self.mix_floor)
        coeff_actual = self.mix_floor + mix_factor * (coeff_base - self.mix_floor)
        
        # Apply mixing only if coefficients are non-zero
        if (coeff_actual > 1e-8).any():  # More robust zero check
            # Compute head average
            mean_heads = x.mean(dim=2, keepdim=True)  # (B, L, 1, D)
            
            # Apply per-head mixing
            mixing_weights = coeff_actual.view(1, 1, self.num_heads, 1)
            x = x + mixing_weights * mean_heads
        
        # Update step counter if not using global scheduler
        if not self.use_global_scheduler and self.training:
            self._step += 1
        
        return x


# Additional utility for combining these components
class DeltaNetBlock(nn.Module):
    """
    Combined DeltaNet block with FIR filtering and cross-head mixing.
    Demonstrates how to combine the transferable components.
    """
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 7, 
                 mix_init: float = 0.03, mix_floor: float = 0.01):
        super().__init__()
        self.fir_conv = DepthwiseFIRConv1d(num_heads, head_dim, kernel_size)
        self.cross_head_mixing = CrossHeadMixing(num_heads, mix_init, mix_floor)
        self.layer_norm = nn.LayerNorm(num_heads * head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through FIR filtering, cross-head mixing, and normalization.
        Args:
            x: (B, L, H, D) input tensor
        Returns:
            Processed tensor with same shape
        """
        # Store residual
        residual = x
        
        # Apply FIR filtering
        x = self.fir_conv(x)
        
        # Apply cross-head mixing
        x = self.cross_head_mixing(x)
        
        # Layer norm (reshape for proper normalization)
        b, l, h, d = x.shape
        x = rearrange(x, 'b l h d -> b l (h d)')
        x = self.layer_norm(x)
        x = rearrange(x, 'b l (h d) -> b l h d', h=h)
        
        # Residual connection
        return x + residual
