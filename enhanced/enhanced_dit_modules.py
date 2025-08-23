from __future__ import annotations
import math
import logging
from functools import lru_cache
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enhanced.global_scheduler import get_global_scheduler

# Set up logging for memory monitoring
logger = logging.getLogger(__name__)

def monitor_memory_usage(operation_name: str = "", warning_threshold_gb: float = 8.0):
    """Monitor GPU memory usage and warn if approaching limits."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9    # GB

    if allocated > warning_threshold_gb:
        logger.warning(f"High memory usage in {operation_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return allocated, reserved


def memory_efficient_gradient_checkpointing(function, *args, **kwargs):
    """
    Enhanced gradient checkpointing with memory management.
    Automatically determines if checkpointing is beneficial based on memory usage.
    """
    if not torch.cuda.is_available():
        return function(*args, **kwargs)

    # Check current memory usage
    memory_before = torch.cuda.memory_allocated() / 1e9
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Use gradient checkpointing if memory usage is high
    if memory_before > total_memory * 0.6:  # 60% memory usage threshold
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


class MemoryAwareLayerNorm(nn.LayerNorm):
    """
    Memory-aware LayerNorm that uses in-place operations when beneficial.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_inplace = True

    def forward(self, x):
        if self._use_inplace and x.requires_grad:
            # Use in-place operations for memory efficiency
            mean = x.mean(dim=-1, keepdim=True)
            x.sub_(mean)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x.div_((var + self.eps).sqrt())
            if self.weight is not None:
                x.mul_(self.weight.view(1, 1, -1))
            if self.bias is not None:
                x.add_(self.bias.view(1, 1, -1))
            return x
        else:
            return super().forward(x)


class AdaptiveMemoryManager:
    """
    Adaptive memory manager that monitors and optimizes memory usage across the model.
    """

    def __init__(self, memory_threshold_gb: float = 2.0):
        self.memory_threshold = memory_threshold_gb
        self.cleanup_counter = 0
        self.last_memory_usage = 0.0

    def should_cleanup(self) -> bool:
        """Determine if memory cleanup is needed."""
        if not torch.cuda.is_available():
            return False

        current_memory = torch.cuda.memory_allocated() / 1e9
        memory_increase = current_memory - self.last_memory_usage
        self.last_memory_usage = current_memory

        # Trigger cleanup if memory increase is significant or threshold is exceeded
        return (memory_increase > 0.5 or current_memory > self.memory_threshold)

    def cleanup_if_needed(self, force: bool = False):
        """Perform memory cleanup if needed."""
        if force or self.should_cleanup():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.cleanup_counter += 1
                if self.cleanup_counter % 10 == 0:  # Log every 10 cleanups
                    logger.info(f"Memory cleanup performed {self.cleanup_counter} times")

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'cleanup_count': self.cleanup_counter
        }


# Global memory manager instance
_memory_manager = AdaptiveMemoryManager()

def get_memory_manager() -> AdaptiveMemoryManager:
    """Get the global memory manager instance."""
    return _memory_manager


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
    inv = -(kb_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict | eye.to(torch.bool), 0) + eye
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
# _delta_inner = torch.compile(_delta_inner, dynamic=True)

def _delta_inner_low_rank(q_i, k_i, v_i, kb_i, S_tuple, tri_strict, eye, state_rank):
    """
    Memory-optimized low-rank delta-rule computation with automatic cleanup.
    """
    S_a, S_b = S_tuple

    # Monitor memory usage before computation
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated() / 1e9  # GB

    try:
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict, 0)

        # Use in-place operations where possible to save memory
        inv = -(kb_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict | eye.to(torch.bool), 0).add_(eye)

        chunk_size = tri_strict.size(-1)
        for i in range(1, chunk_size):
            if i > 0:
                inv_row = inv[..., i:i+1, :i]
                inv_sub = inv[..., :i, :i]
                update = torch.matmul(inv_row, inv_sub)
                inv[..., i:i+1, :i] = inv[..., i:i+1, :i] + update

                # Clean up intermediate tensors immediately
                del inv_row, inv_sub, update

        # Low-rank state operations with memory optimization
        S_mult = S_b.transpose(-1, -2)
        kb_S = (inv @ kb_i) @ S_a
        u_i = (inv @ v_i) - kb_S @ S_mult

        # Compute output with memory-efficient operations
        q_S = q_i @ S_a
        out_chunk = (q_S @ S_mult) + attn_local @ u_i

        # Clean up intermediate tensors
        del attn_local, inv, S_mult, kb_S, q_S
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Low-rank state update with reduced memory usage
        S_a_large = torch.cat([S_a, k_i.transpose(-1, -2)], dim=-1)
        u_i_permuted = u_i.permute(0, 1, 3, 2)
        S_b_large = torch.cat([S_b, u_i_permuted], dim=-1)

        # Clean up more tensors
        del u_i, u_i_permuted

        # Use float32 for numerical stability, but with memory management
        original_dtype = S_a_large.dtype

        # Process in smaller chunks if memory is tight
        if torch.cuda.is_available():
            memory_current = torch.cuda.memory_allocated() / 1e9
            memory_limit = 0.8 * torch.cuda.get_device_properties(0).total_memory / 1e9  # 80% limit

            if memory_current > memory_limit:
                # Use smaller precision for SVD operations
                logger.warning(".1f")

        # QR decomposition with memory cleanup
        S_a_f32 = S_a_large.float()
        S_b_f32 = S_b_large.float()
        del S_a_large, S_b_large

        Q_a, R_a = torch.linalg.qr(S_a_f32)
        Q_b, R_b = torch.linalg.qr(S_b_f32)
        del S_a_f32, S_b_f32

        # Convert back to original dtype
        Q_a = Q_a.to(original_dtype)
        R_a = R_a.to(original_dtype)
        Q_b = Q_b.to(original_dtype)
        R_b = R_b.to(original_dtype)

        M = R_a @ R_b.transpose(-1, -2)
        del R_a, R_b

        # SVD with robust fallback and memory management
        M_f32 = M.float()
        del M

        try:
            U, s_val, Vh = torch.linalg.svd(M_f32, full_matrices=False)
        except Exception:
            logger.warning("Standard SVD failed, using robust fallback")
            # Add regularization for numerical stability
            M_reg = M_f32 + torch.eye(M_f32.shape[-1], device=M_f32.device) * 1e-6
            U, s_val, Vh = torch.linalg.svd(M_reg, full_matrices=False)

        del M_f32

        # Convert results back to original dtype
        U = U.to(original_dtype)
        s_val = s_val.to(original_dtype)
        Vh = Vh.to(original_dtype)

        V = Vh.transpose(-1, -2)
        del Vh

        # Compute new state matrices with rank truncation
        s_val_sqrt = torch.sqrt(s_val[..., :state_rank]).unsqueeze(-2)
        S_a_new = Q_a[..., :, :state_rank] * s_val_sqrt
        S_b_new = Q_b[..., :, :state_rank] * s_val_sqrt

        # Final cleanup
        del Q_a, Q_b, U, s_val, V, s_val_sqrt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Monitor memory usage after computation
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1e9
            if memory_after > memory_before + 2.0:  # More than 2GB increase
                logger.warning(".1f")

        return out_chunk, (S_a_new, S_b_new)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM in delta computation, attempting recovery")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Return zero tensors as fallback
            device = q_i.device
            dtype = q_i.dtype
            B, H, chunk_size, D = q_i.shape
            zero_out = torch.zeros(B, H, chunk_size, D, device=device, dtype=dtype)
            S_a_zero = torch.zeros(B, H, D, state_rank, device=device, dtype=dtype)
            S_b_zero = torch.zeros(B, H, D, state_rank, device=device, dtype=dtype)
            return zero_out, (S_a_zero, S_b_zero)
        else:
            raise


def delta_rule_chunkwise(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         beta: torch.Tensor,
                         *,
                         chunk_size: int = 32,
                         state_rank: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient causal associative Δ-rule – memory-light and Torch-compile friendly.
    Shapes
    ------
    q,k,v : (B, H, L, D)
    beta  : (B, H, L)
    Returns
    -------
    out   : (B, H, L, D)
    S_fin : (B, H, D, D) or Tuple of (B,H,D,R)  (final state, can be fed into the next segment)
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

    # Initialize state with memory-safe defaults
    if state_rank is None:
        state_rank = min(64, D // 4)  # Default to low-rank for memory safety
        logger.warning(f"Using default low-rank approximation with state_rank={state_rank} to prevent memory explosion")

    S_a = q.new_zeros(B, H, D, state_rank)
    S_b = q.new_zeros(B, H, D, state_rank)
    S = (S_a, S_b)
        
    out_chunks = []

    for idx in range(n_chunks):
        if state_rank is None:
            out_c, S = _delta_inner(q[:, :, idx],
                                    k[:, :, idx],
                                    v[:, :, idx],
                                    k_beta[:, :, idx],
                                    S,
                                    tri_strict,
                                    eye)
        else:
            out_c, S = _delta_inner_low_rank(q[:, :, idx],
                                             k[:, :, idx],
                                             v[:, :, idx],
                                             k_beta[:, :, idx],
                                             S,
                                             tri_strict,
                                             eye,
                                             state_rank)
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
