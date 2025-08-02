# YEHW - Yield-Expert-Hierarchical-Weights
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from .quantization import BlackwellOptimizedLinear, sage_attention_with_fp8
from ..enhanced.enhanced_deltanet_dit_block import UltimateDeltaNetDiTBlock
# Sage attention will be imported conditionally when needed


# --- Helper Functions ---
@torch.jit.script
def fused_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fused modulation with numerical stability improvements."""
    # Clamp scale to prevent extreme values in FP4
    scale_clamped = torch.clamp(scale, min=-2.0, max=2.0)
    return x * (1.0 + scale_clamped.unsqueeze(1)) + shift.unsqueeze(1)

@torch.jit.script
def fast_attention_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Pre-compute and cache attention masks for common sequence lengths."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    return mask.bool()

# Optimized memory pool for common tensor shapes
class TensorMemoryPool:
    def __init__(self, max_pool_size: int = 100):
        self._pools = {}
        self.max_pool_size = max_pool_size
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (shape, dtype, device)
        if key not in self._pools:
            self._pools[key] = []
        
        pool = self._pools[key]
        if pool:
            return pool.pop().zero_()
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if key in self._pools and len(self._pools[key]) < self.max_pool_size:
            self._pools[key].append(tensor.detach())
    
    def clear_pool(self):
        """Clear memory pool to prevent memory leaks"""
        self._pools.clear()

# Global memory pool instance
_memory_pool = TensorMemoryPool()

# --- Embedding Classes ---

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            BlackwellOptimizedLinear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            BlackwellOptimizedLinear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        
        # Use paper-recommended initialization
        self.register_buffer('inv_freq', torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        ))
        
        # Initialize with smaller weights for FP4 stability
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        
    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)

class ClassEmbedder(nn.Module):
    """
    Class embedding module similar to TimestepEmbedder for conditional generation.
    Maps class indices to dense embeddings through learned embedding table and MLP.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Learnable embedding table for class indices
        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        # MLP to process class embeddings (similar to TimestepEmbedder)
        self.mlp = nn.Sequential(
            BlackwellOptimizedLinear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            BlackwellOptimizedLinear(hidden_size, hidden_size, bias=True),
        )
        
        # Initialize with smaller weights for FP4 stability
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[3].weight, std=0.02)
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B,) tensor of class indices
        Returns:
            class_emb: (B, hidden_size) class embeddings
        """
        # Handle both class indices and one-hot encodings
        if y.dtype == torch.long:
            # Class indices
            class_emb = self.embedding(y)
        else:
            # One-hot or soft labels - use weighted sum
            class_emb = torch.matmul(y, self.embedding.weight)
        
        return self.mlp(class_emb)

# --- ALiBi Positional Embedding ---

class ALiBi(nn.Module):
    def __init__(self, num_heads: int, max_cache_size: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.max_cache_size = max_cache_size
        slopes = torch.tensor(self._get_alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)
        self._cache: Dict[int, torch.Tensor] = {}

    def _evict_cache(self):
        """LRU-style cache eviction"""
        if len(self._cache) >= self.max_cache_size:
            min_key = min(self._cache.keys())
            del self._cache[min_key]
    
    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len in self._cache:
            cached = self._cache[seq_len]
            # Move accessed item to end (simple LRU)
            del self._cache[seq_len]
            self._cache[seq_len] = cached
            return cached.to(device=device, dtype=dtype)

        self._evict_cache()
        
        pos = torch.arange(seq_len, device=device)
        rel = (pos[None, :] - pos[:, None]).abs() * -1.0
        bias = self.slopes[:, None, None] * rel
        bias = bias.unsqueeze(0)

        # Cache on CPU to save GPU memory
        self._cache[seq_len] = bias.cpu()
        return bias.to(dtype=dtype)



# --- Attention Module ---

class OptimizedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, use_sage_attention: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_sage_attention = use_sage_attention
        
        self.qkv = BlackwellOptimizedLinear(dim, dim * 3, bias=False)
        self.proj = BlackwellOptimizedLinear(dim, dim, bias=True)
        self.alibi = ALiBi(num_heads)
        
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.proj_dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Optimized Sage_Attention with FP8 integration for Blackwell
        x = sage_attention_with_fp8(
            q, k, v, 
            is_causal=True, 
            use_fp8_context=self.use_sage_attention
        )
        
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        if self.proj_dropout: x = self.proj_dropout(x)
        
        return x

# --- Mixture of Experts ---
class Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = intermediate_size or int(hidden_size * 8 // 3)
        
        self.gate_up_proj = BlackwellOptimizedLinear(hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = BlackwellOptimizedLinear(self.intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class StreamlinedMoE(nn.Module):
    """
    Mixture-of-Experts layer with an explicit split:
      • experts 0 … (high_noise_experts-1)   are *high-noise* specialists
      • the remaining experts               are *low-noise* specialists
    Routing decision is made with per-token SNR.
    """
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 high_noise_experts: int,
                 snr_threshold: float,
                 top_k: int = 2,
                 capacity_factor: float = 1.25,
                 jitter_noise: float = 0.01,
                 load_balance_loss_coef: float = 0.01,
                 use_learnable_threshold: bool = False,
                 activation: str = 'silu',
                 expert_bias: bool = False):
        super().__init__()
        assert 0 < high_noise_experts < num_experts, "split must be non-trivial"
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.high_noise_experts = high_noise_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.load_balance_loss_coef = load_balance_loss_coef

        # Learnable or fixed SNR threshold
        if use_learnable_threshold:
            self.snr_threshold = nn.Parameter(torch.tensor(snr_threshold))
        else:
            self.register_buffer('snr_threshold', torch.tensor(snr_threshold))

        # Gating network - assumes cond has same dimension as hidden_size
        self.gate = BlackwellOptimizedLinear(hidden_size * 2, num_experts, bias=False)
        
        # Expert modules
        self.experts = nn.ModuleList([
            Expert(hidden_size, activation=activation, use_bias=expert_bias) 
            for _ in range(num_experts)
        ])
        
        # Statistics tracking
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_forward_calls', torch.tensor(0))

    def _compute_gate_logits(self, 
                           x_flat: torch.Tensor, 
                           cond_flat: torch.Tensor) -> torch.Tensor:
        """Compute gating logits from content and conditioning."""
        combined_input = torch.cat([x_flat, cond_flat], dim=-1)
        return self.gate(combined_input)

    def _apply_snr_mask(self, 
                       logits: torch.Tensor, 
                       snr_flat: torch.Tensor) -> torch.Tensor:
        """Apply SNR-based expert masking."""
        # Determine which tokens have high noise (low SNR)
        high_noise = (snr_flat < self.snr_threshold).unsqueeze(-1)  # (B·N, 1)
        
        # Create mask: -inf for disallowed experts
        mask = torch.zeros_like(logits)
        
        # For high-noise tokens, mask out low-noise experts (high_noise_experts:)
        # For low-noise tokens, allow all experts
        mask[:, self.high_noise_experts:] = torch.where(
            high_noise, 
            torch.full_like(mask[:, self.high_noise_experts:], -float('inf')), 
            0.0
        )
        
        return logits + mask

    def _add_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        """Add jitter noise during training for regularization."""
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(logits).uniform_(-self.jitter_noise, self.jitter_noise)
            return logits + noise
        return logits

    def _compute_load_balance_loss(self, 
                                  probs: torch.Tensor, 
                                  top_indices: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage expert diversity."""
        if not self.training or self.load_balance_loss_coef <= 0:
            return torch.tensor(0.0, device=probs.device)
        
        # Average gate probability per expert
        gate_mean = probs.mean(0)  # (num_experts,)
        
        # Frequency of expert selection
        freq = torch.bincount(
            top_indices.flatten(), 
            minlength=self.num_experts
        ).float()
        freq = freq / (freq.sum() + 1e-8)
        
        # Auxiliary loss (from Switch Transformer paper)
        aux_loss = (gate_mean * freq).sum() * self.num_experts
        
        # Optional: entropy regularization for diversity
        entropy_reg = -torch.sum(gate_mean * torch.log(gate_mean + 1e-8))
        entropy_weight = 0.01
        
        total_loss = (aux_loss + entropy_weight * entropy_reg) * self.load_balance_loss_coef
        return total_loss

    def _efficient_dispatch(self, 
                           x_flat: torch.Tensor,
                           top_probs: torch.Tensor,
                           top_indices: torch.Tensor) -> torch.Tensor:
        """Memory-efficient expert dispatching with capacity management."""
        B_N, C = x_flat.shape
        out = torch.zeros_like(x_flat)
        capacity = max(1, int(self.capacity_factor * B_N / self.num_experts))
        
        # Track expert usage for monitoring
        if self.training:
            unique_experts, counts = torch.unique(top_indices, return_counts=True)
            self.expert_usage.index_add_(0, unique_experts, counts.float())
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_indices == expert_idx)
            if not expert_mask.any():
                continue
                
            token_idx, k_idx = expert_mask.nonzero(as_tuple=True)
            
            # Apply capacity constraint with priority-based selection
            if token_idx.numel() > capacity:
                priorities = top_probs[token_idx, k_idx]
                _, selected = priorities.topk(capacity, largest=True)
                token_idx = token_idx[selected]
                k_idx = k_idx[selected]
            
            if token_idx.numel() == 0:
                continue
            
            # Get weights for selected tokens
            weights = top_probs[token_idx, k_idx].unsqueeze(1)  # (selected_tokens, 1)
            
            # Process tokens through expert (chunked for memory efficiency)
            chunk_size = min(1024, token_idx.numel())
            for i in range(0, token_idx.numel(), chunk_size):
                end_idx = min(i + chunk_size, token_idx.numel())
                
                chunk_token_idx = token_idx[i:end_idx]
                chunk_weights = weights[i:end_idx]
                
                # Forward through expert
                expert_input = x_flat[chunk_token_idx]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_output = expert_output * chunk_weights
                
                # Accumulate results
                out.index_add_(0, chunk_token_idx, weighted_output)
        
        return out

    def forward(self,
                x: torch.Tensor,      # (B, N, C)
                cond: torch.Tensor,   # (B, C) - conditioning vector
                snr: torch.Tensor     # (B,) - per-sample SNR
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor (B, N, C)
            cond: Conditioning tensor (B, C) 
            snr: SNR values per sample (B,)
            
        Returns:
            output: Processed tensor (B, N, C)
            load_balance_loss: Load balancing loss scalar
        """
        B, N, C = x.shape
        
        # Flatten for expert processing
        x_flat = x.reshape(B * N, C)  # (B·N, C)
        cond_flat = cond.repeat_interleave(N, dim=0)  # (B·N, C)
        snr_flat = snr.repeat_interleave(N)  # (B·N,)
        
        # Update call counter
        self.total_forward_calls += 1
        
        # 1. Compute gate logits
        logits = self._compute_gate_logits(x_flat, cond_flat)  # (B·N, num_experts)
        
        # 2. Apply SNR-based masking
        logits = self._apply_snr_mask(logits, snr_flat)
        
        # 3. Add jitter during training
        logits = self._add_jitter(logits)
        
        # 4. Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # (B·N, num_experts)
        
        # 5. Top-k routing
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)  # (B·N, k)
        top_probs = F.normalize(top_probs, p=1, dim=-1)  # Renormalize
        
        # 6. Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(probs, top_indices)
        
        # 7. Dispatch tokens to experts
        output = self._efficient_dispatch(x_flat, top_probs, top_indices)
        
        # 8. Reshape back to original dimensions
        output = output.reshape(B, N, C)
        
        return output, load_balance_loss

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get comprehensive expert utilization statistics."""
        if not hasattr(self, 'expert_usage') or self.total_forward_calls == 0:
            return {}
        
        usage = self.expert_usage.cpu().numpy()
        total_usage = usage.sum()
        
        if total_usage == 0:
            return {'message': 'No expert usage recorded yet'}
        
        usage_dist = usage / total_usage
        
        stats = {
            'total_forward_calls': int(self.total_forward_calls),
            'expert_usage_counts': usage.tolist(),
            'expert_usage_distribution': usage_dist.tolist(),
            'active_experts': int((usage > 0).sum()),
            'total_expert_calls': int(total_usage),
            'most_used_expert': int(usage.argmax()),
            'least_used_expert': int(usage.argmin()),
            'max_usage_ratio': float(usage.max() / total_usage) if total_usage > 0 else 0.0,
            'min_usage_ratio': float(usage.min() / total_usage) if total_usage > 0 else 0.0,
            'usage_entropy': float(-np.sum(usage_dist * np.log(usage_dist + 1e-8))),
            'current_snr_threshold': float(self.snr_threshold.data),
            'high_noise_experts_range': f"0-{self.high_noise_experts-1}",
            'low_noise_experts_range': f"{self.high_noise_experts}-{self.num_experts-1}"
        }
        
        return stats

    def reset_expert_stats(self):
        """Reset expert usage statistics."""
        if hasattr(self, 'expert_usage'):
            self.expert_usage.zero_()
        if hasattr(self, 'total_forward_calls'):
            self.total_forward_calls.zero_()

    def get_routing_info(self, 
                        x: torch.Tensor, 
                        cond: torch.Tensor, 
                        snr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed routing information for analysis (inference only)."""
        if self.training:
            return {'message': 'Routing info only available in eval mode'}
        
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)
        cond_flat = cond.repeat_interleave(N, dim=0)
        snr_flat = snr.repeat_interleave(N)
        
        # Get routing decisions without forward pass
        with torch.no_grad():
            logits = self._compute_gate_logits(x_flat, cond_flat)
            masked_logits = self._apply_snr_mask(logits, snr_flat)
            probs = F.softmax(masked_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
            
            # Determine which tokens go to which expert type
            high_noise_mask = snr_flat < self.snr_threshold
            
            return {
                'gate_logits': logits.reshape(B, N, -1),
                'gate_probs': probs.reshape(B, N, -1),
                'top_expert_indices': top_indices.reshape(B, N, -1),
                'top_expert_probs': top_probs.reshape(B, N, -1),
                'high_noise_tokens': high_noise_mask.reshape(B, N),
                'snr_values': snr_flat.reshape(B, N),
                'snr_threshold': self.snr_threshold.expand(B, N)
            }

    def extra_repr(self) -> str:
        """Enhanced string representation."""
        return (f'hidden_size={self.hidden_size}, num_experts={self.num_experts}, '
                f'high_noise_experts={self.high_noise_experts}, top_k={self.top_k}, '
                f'capacity_factor={self.capacity_factor}, snr_threshold={self.snr_threshold.item():.4f}')


# Helper for SNR (cosine schedule)
def compute_snr(t: torch.Tensor, noise_schedule: str = 'cosine', T: int = 1000) -> torch.Tensor:
    if noise_schedule == 'cosine':
        alpha_cumprod = torch.cos((t / T + 0.008) / 1.008 * torch.pi / 2) ** 2
    else:  # linear
        betas = torch.linspace(1e-4, 0.02, T, device=t.device)
        alpha_cumprod = torch.cumprod(1 - betas, dim=0)[t.long().clamp(0, T-1)]
    snr = alpha_cumprod / (1 - alpha_cumprod).clamp_min(1e-8)
    return snr

# --- U-Net Components ---

class ConvStem(nn.Module):
    """Enhanced convolutional stem with configurable options."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3,
                 activation: str = 'silu',
                 norm_type: str = 'groupnorm',
                 num_groups: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            bias=norm_type == 'none'
        )
        
        # Flexible normalization
        if norm_type == 'groupnorm':
            groups = num_groups or min(32, out_channels)
            # Ensure groups divides channels
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'layernorm':
            self.norm = nn.Identity()  # Will apply later in forward
        else:
            self.norm = nn.Identity()
        
        # Flexible activation
        self.act = {
            'silu': nn.SiLU(),
            'swish': nn.SiLU(),  # Same as SiLU
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'mish': nn.Mish() if hasattr(nn, 'Mish') else nn.SiLU(),
            'none': nn.Identity()
        }.get(activation, nn.SiLU())
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm_type = norm_type
        
        # Enhanced initialization
        self._init_weights()

    def _init_weights(self):
        """Enhanced weight initialization based on activation."""
        if isinstance(self.act, (nn.SiLU, nn.GELU)):
            # Xavier/Glorot for SiLU/GELU
            nn.init.xavier_uniform_(self.conv.weight)
        elif isinstance(self.act, nn.ReLU):
            # Kaiming for ReLU
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        else:
            # Default Xavier
            nn.init.xavier_uniform_(self.conv.weight)
        
        # Initialize bias if present
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        # Handle LayerNorm specially for 2D
        if self.norm_type == 'layernorm':
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
            x = F.layer_norm(x, [C])
            x = x.transpose(1, 2).view(B, C, H, W)
        else:
            x = self.norm(x)
        
        x = self.act(x)
        x = self.dropout(x)
        return x


class DownsampleBlock(nn.Module):
    """Enhanced downsampling with multiple strategies."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 downsample_method: str = 'conv',
                 activation: str = 'silu',
                 norm_type: str = 'groupnorm',
                 num_groups: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        
        self.downsample_method = downsample_method
        
        if downsample_method == 'conv':
            self.downsample = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, stride=2, padding=1, 
                bias=norm_type == 'none'
            )
        elif downsample_method == 'maxpool':
            self.downsample = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=norm_type == 'none')
            )
        elif downsample_method == 'avgpool':
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=norm_type == 'none')
            )
        else:
            raise ValueError(f"Unknown downsample method: {downsample_method}")
        
        # Normalization
        if norm_type == 'groupnorm':
            groups = num_groups or min(32, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        self.act = {
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'none': nn.Identity()
        }.get(activation, nn.SiLU())
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm_type = norm_type
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all conv layers."""
        def init_conv(m):
            if isinstance(m, nn.Conv2d):
                if isinstance(self.act, nn.ReLU):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.downsample.apply(init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        
        # Handle LayerNorm for 2D
        if self.norm_type == 'layernorm':
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            x = F.layer_norm(x, [C])
            x = x.transpose(1, 2).view(B, C, H, W)
        else:
            x = self.norm(x)
        
        x = self.act(x)
        x = self.dropout(x)
        return x


class UpsampleBlock(nn.Module):
    """Enhanced upsampling with multiple strategies."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 upsample_method: str = 'transconv',
                 activation: str = 'silu',
                 norm_type: str = 'groupnorm',
                 num_groups: Optional[int] = None,
                 dropout: float = 0.0,
                 scale_factor: int = 2):
        super().__init__()
        
        self.upsample_method = upsample_method
        self.scale_factor = scale_factor
        
        if upsample_method == 'transconv':
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size=scale_factor, stride=scale_factor, 
                bias=norm_type == 'none'
            )
        elif upsample_method == 'interpolate':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=norm_type == 'none')
            )
        elif upsample_method == 'nearest':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=norm_type == 'none')
            )
        elif upsample_method == 'pixelshuffle':
            # Pixel shuffle requires specific channel relationships
            intermediate_channels = out_channels * (scale_factor ** 2)
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=norm_type == 'none'),
                nn.PixelShuffle(scale_factor)
            )
        else:
            raise ValueError(f"Unknown upsample method: {upsample_method}")
        
        # Normalization
        if norm_type == 'groupnorm':
            groups = num_groups or min(32, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        self.act = {
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'none': nn.Identity()
        }.get(activation, nn.SiLU())
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm_type = norm_type
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all conv layers."""
        def init_conv(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.act, nn.ReLU):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.upsample.apply(init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle LayerNorm for 2D
        if self.norm_type == 'layernorm':
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            x = F.layer_norm(x, [C])
            x = x.transpose(1, 2).view(B, C, H, W)
        else:
            x = self.norm(x)
        
        x = self.act(x)
        x = self.dropout(x)
        return x


class ChannelMatcher(nn.Module):
    """Enhanced channel matching with optional processing."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 use_conv: bool = True,
                 activation: str = 'none',
                 norm_type: str = 'none',
                 dropout: float = 0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if in_channels != out_channels:
            if use_conv:
                self.conv = nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, bias=norm_type == 'none'
                )
                nn.init.xavier_uniform_(self.conv.weight)
                if self.conv.bias is not None:
                    nn.init.zeros_(self.conv.bias)
            else:
                # Alternative: learnable linear transformation
                self.conv = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    BlackwellOptimizedLinear(in_channels, out_channels),
                    nn.Unflatten(1, (out_channels, 1, 1))
                )
        else:
            self.conv = nn.Identity()
        
        # Optional normalization
        if norm_type == 'groupnorm' and in_channels != out_channels:
            groups = min(32, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm_type == 'batchnorm' and in_channels != out_channels:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Optional activation
        self.act = {
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'none': nn.Identity()
        }.get(activation, nn.Identity())
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels != self.out_channels:
            # Handle dimension mismatch with interpolation if needed
            target_spatial = x.shape[2:]  # (H, W)
            x = self.conv(x)
            
            # Ensure spatial dimensions are preserved
            if x.shape[2:] != target_spatial and not isinstance(self.conv, nn.Identity):
                x = F.interpolate(x, size=target_spatial, mode='bilinear', align_corners=False)
            
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x)
        
        return x

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


class UNetDeltaNetBlock(nn.Module):
    """Enhanced 2-D wrapper around UltimateDeltaNetDiTBlock with additional features."""
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 num_experts: int,
                 top_k: int,
                 jitter_noise: float,
                 dropout: float,
                 use_moe: bool,
                 snr_threshold: float,
                 use_sage_attention: bool,
                 spatial_compression: Optional[int] = None,
                 use_flash_attention: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()

        self.channels = channels
        self.original_num_heads = num_heads
        self.spatial_compression = spatial_compression
        self.gradient_checkpointing = gradient_checkpointing
        
        # Auto-adjust heads to divide channels evenly
        if channels % num_heads != 0:
            for h in (16, 12, 8, 6, 4, 2, 1):
                if channels % h == 0:
                    num_heads = h
                    break
            print(f"Warning: Adjusted num_heads from {self.original_num_heads} to {num_heads} "
                  f"to evenly divide channels ({channels})")

        self.num_heads = num_heads
        
        # Optional spatial compression for efficiency
        if spatial_compression and spatial_compression > 1:
            self.spatial_compress = nn.Conv2d(
                channels, channels, 
                kernel_size=spatial_compression, 
                stride=spatial_compression, 
                padding=0
            )
            self.spatial_decompress = nn.ConvTranspose2d(
                channels, channels,
                kernel_size=spatial_compression,
                stride=spatial_compression,
                padding=0
            )
        else:
            self.spatial_compress = nn.Identity()
            self.spatial_decompress = nn.Identity()

        # Core DeltaNet block
        self.block = UltimateDeltaNetDiTBlock(
            channels=channels,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            dropout=dropout,
            use_moe=use_moe,
            snr_threshold=snr_threshold,
            use_sage_attention=use_sage_attention and use_flash_attention
        )
        
        # Optional residual connection enhancement
        self.use_enhanced_residual = channels >= 512  # Only for larger models
        if self.use_enhanced_residual:
            self.residual_gate = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def _spatial_to_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert spatial tensor to sequence format."""
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x_seq, (H, W)

    def _sequence_to_spatial(self, x_seq: torch.Tensor, spatial_shape: Tuple[int, int], channels: int) -> torch.Tensor:
        """Convert sequence back to spatial format."""
        B = x_seq.shape[0]
        H, W = spatial_shape
        x = x_seq.transpose(1, 2).view(B, channels, H, W)
        return x

    def _forward_impl(self,
                     x: torch.Tensor,
                     c: torch.Tensor,
                     t: torch.Tensor,
                     snr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal forward implementation."""
        B, C, H, W = x.shape
        original_shape = (H, W)
        
        # Optional spatial compression
        if not isinstance(self.spatial_compress, nn.Identity):
            x_compressed = self.spatial_compress(x)
            x_seq, compressed_shape = self._spatial_to_sequence(x_compressed)
        else:
            x_seq, compressed_shape = self._spatial_to_sequence(x)
        
        # Process through DeltaNet block
        x_seq_out, loss = self.block(x_seq, c, t, snr)
        x_processed = self._sequence_to_spatial(x_seq_out, compressed_shape, C)
        
        # Optional spatial decompression
        if not isinstance(self.spatial_decompress, nn.Identity):
            x_processed = self.spatial_decompress(x_processed)
            
            # Ensure output matches input spatial dimensions
            if x_processed.shape[2:] != original_shape:
                x_processed = F.interpolate(
                    x_processed, size=original_shape, 
                    mode='bilinear', align_corners=False
                )
        
        # Enhanced residual connection for large models
        if self.use_enhanced_residual:
            residual_weight = torch.sigmoid(self.residual_gate)
            x_out = x + residual_weight * x_processed
        else:
            x_out = x_processed
        
        return x_out, loss

    def forward(self,
                x: torch.Tensor,    # (B, C, H, W)
                c: torch.Tensor,    # (B, D)  conditioning
                t: torch.Tensor,    # (B,)    timestep
                snr: torch.Tensor   # (B,)    pre-computed snr
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional gradient checkpointing."""
        
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, c, t, snr, use_reentrant=False
            )
        else:
            return self._forward_impl(x, c, t, snr)

    def get_attention_maps(self,
                          x: torch.Tensor,
                          c: torch.Tensor, 
                          t: torch.Tensor,
                          snr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization."""
        if self.training:
            return {'message': 'Attention maps only available in eval mode'}
        
        # This would need to be implemented in the underlying DeltaNet block
        # For now, return placeholder
        B, C, H, W = x.shape
        x_seq, _ = self._spatial_to_sequence(x)
        
        return {
            'input_shape': (B, C, H, W),
            'sequence_length': x_seq.shape[1],
            'num_heads': self.num_heads,
            'message': 'Attention map extraction needs implementation in UltimateDeltaNetDiTBlock'
        }

    def extra_repr(self) -> str:
        return (f'channels={self.channels}, num_heads={self.num_heads}, '
                f'spatial_compression={self.spatial_compression}, '
                f'gradient_checkpointing={self.gradient_checkpointing}')



# --- Main U-Net Architecture ---
# --- Main U-Net Architecture ---
class UNetDeltaNet(nn.Module):
    """
    Enhanced UNet with DeltaNet blocks, MoE, and comprehensive optimization features.
    Target: ~2B parameters with efficient memory usage and training optimizations.
    """
    def __init__(self, 
                 in_channels: int = 4, 
                 out_channels: int = 4, 
                 hidden_size: int = 1536,
                 num_levels: int = 4, 
                 blocks_per_level: int = 4, 
                 num_experts: int = 8,
                 high_noise_experts: int = 4,
                 snr_threshold: float = 0.1, 
                 T: int = 1000, 
                 time_dim: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 top_k: int = 2,
                 jitter_noise: float = 0.01,
                 dropout: float = 0.1,
                 use_sage_attention: bool = True,
                 activation: str = 'silu',
                 norm_type: str = 'groupnorm',
                 downsample_method: str = 'conv',
                 upsample_method: str = 'transconv',
                 use_skip_connections: bool = True,
                 skip_connection_type: str = 'concat',  # 'concat', 'add', 'gated'
                 compile_model: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9999):
        super().__init__()
        
        # Core configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.blocks_per_level = blocks_per_level
        self.num_experts = num_experts
        self.high_noise_experts = high_noise_experts
        self.snr_threshold = snr_threshold
        self.T = T
        self.time_dim = time_dim or hidden_size
        self.num_classes = num_classes
        self.use_skip_connections = use_skip_connections
        self.skip_connection_type = skip_connection_type
        self.compile_model = compile_model
        
        # Training configuration
        self.use_amp = False
        self.gradient_checkpointing = False
        self.selective_checkpointing = False
        self.checkpoint_segments = 4
        self._fused_kernels = False
        
        # EMA configuration
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = None  # Will be initialized after model creation
            self.ema_decay = ema_decay
        
        # Channel progression with better scaling
        self.channel_multipliers = [1, 2, 4, 8, 16][:num_levels + 1]
        base_channels = hidden_size // max(self.channel_multipliers)
        channels = [base_channels * mult for mult in self.channel_multipliers]
        self.channels = channels
        
        # Build architecture
        self._build_stem(channels[0])
        self._build_encoder(channels, num_experts, top_k, jitter_noise, dropout, 
                           use_sage_attention, activation, norm_type, downsample_method)
        self._build_bottleneck(channels[-1], num_experts, top_k, jitter_noise, 
                              dropout, use_sage_attention, activation, norm_type)
        self._build_decoder(channels, num_experts, top_k, jitter_noise, dropout,
                           use_sage_attention, activation, norm_type, upsample_method)
        self._build_output_head(channels[0])
        self._build_embedders()
        
        # Initialize and optimize
        self.initialize_weights()
        
        if torch.cuda.is_available():
            self._apply_cuda_optimizations()
        
        # Model statistics
        self.param_count = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"UNetDeltaNet initialized: {self.param_count:,} total params, "
              f"{self.trainable_params:,} trainable params")

    def _build_stem(self, out_channels: int):
        """Build input stem."""
        self.stem = ConvStem(
            self.in_channels, out_channels,
            activation='silu', norm_type='groupnorm'
        )

    def _build_encoder(self, channels: List[int], num_experts: int, top_k: int,
                      jitter_noise: float, dropout: float, use_sage_attention: bool,
                      activation: str, norm_type: str, downsample_method: str):
        """Build encoder blocks."""
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        for level in range(self.num_levels):
            in_ch = channels[level]
            out_ch = channels[level + 1]
            
            # Add processing blocks for this level
            level_blocks = nn.ModuleList()
            for block_idx in range(self.blocks_per_level):
                block = UNetDeltaNetBlock(
                    channels=in_ch,
                    num_heads=max(1, in_ch // 64),  # Dynamic head count
                    num_experts=num_experts,
                    top_k=top_k,
                    jitter_noise=jitter_noise,
                    dropout=dropout,
                    use_moe=True,
                    snr_threshold=self.snr_threshold,
                    use_sage_attention=use_sage_attention,
                    gradient_checkpointing=False  # Will be enabled later if needed
                )
                level_blocks.append(block)
            
            self.encoder_blocks.append(level_blocks)
            
            # Add downsampling
            downsample = DownsampleBlock(
                in_ch, out_ch,
                downsample_method=downsample_method,
                activation=activation,
                norm_type=norm_type
            )
            self.encoder_downsamples.append(downsample)

    def _build_bottleneck(self, channels: int, num_experts: int, top_k: int,
                         jitter_noise: float, dropout: float, use_sage_attention: bool,
                         activation: str, norm_type: str):
        """Build bottleneck processing."""
        bottleneck_channels = channels * 2
        
        self.bottleneck_expand = nn.Conv2d(channels, bottleneck_channels, kernel_size=1)
        
        self.bottleneck_blocks = nn.ModuleList([
            UNetDeltaNetBlock(
                channels=bottleneck_channels,
                num_heads=max(1, bottleneck_channels // 64),
                num_experts=num_experts * 2,  # More experts in bottleneck
                top_k=top_k,
                jitter_noise=jitter_noise,
                dropout=dropout,
                use_moe=True,
                snr_threshold=self.snr_threshold,
                use_sage_attention=use_sage_attention
            ) for _ in range(3)  # More processing in bottleneck
        ])
        
        self.bottleneck_contract = nn.Conv2d(bottleneck_channels, channels, kernel_size=1)

    def _build_decoder(self, channels: List[int], num_experts: int, top_k: int,
                      jitter_noise: float, dropout: float, use_sage_attention: bool,
                      activation: str, norm_type: str, upsample_method: str):
        """Build decoder blocks."""
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_matchers = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            in_ch = channels[level + 1]
            out_ch = channels[level]
            
            # Upsampling
            upsample = UpsampleBlock(
                in_ch, out_ch,
                upsample_method=upsample_method,
                activation=activation,
                norm_type=norm_type
            )
            self.decoder_upsamples.append(upsample)
            
            # Skip connection matching
            if self.use_skip_connections:
                if self.skip_connection_type == 'concat':
                    skip_matcher = ChannelMatcher(out_ch, out_ch)  # No change needed for concat
                    processing_channels = out_ch * 2  # Concatenated channels
                elif self.skip_connection_type == 'add':
                    skip_matcher = ChannelMatcher(out_ch, out_ch)
                    processing_channels = out_ch
                elif self.skip_connection_type == 'gated':
                    skip_matcher = nn.Sequential(
                        ChannelMatcher(out_ch, out_ch),
                        nn.Conv2d(out_ch, out_ch, kernel_size=1),
                        nn.Sigmoid()
                    )
                    processing_channels = out_ch
                else:
                    raise ValueError(f"Unknown skip connection type: {self.skip_connection_type}")
            else:
                skip_matcher = nn.Identity()
                processing_channels = out_ch
            
            self.skip_matchers.append(skip_matcher)
            
            # Processing blocks for this level
            level_blocks = nn.ModuleList()
            for block_idx in range(self.blocks_per_level + 1):  # +1 for post-skip block
                current_channels = processing_channels if block_idx == 0 else out_ch
                
                block = UNetDeltaNetBlock(
                    channels=current_channels,
                    num_heads=max(1, current_channels // 64),
                    num_experts=num_experts,
                    top_k=top_k,
                    jitter_noise=jitter_noise,
                    dropout=dropout,
                    use_moe=True,
                    snr_threshold=self.snr_threshold,
                    use_sage_attention=use_sage_attention
                )
                level_blocks.append(block)
            
            self.decoder_blocks.append(level_blocks)

    def _build_output_head(self, in_channels: int):
        """Build output head with multiple options."""
        self.output_head = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, in_channels // 2), in_channels // 2),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, self.out_channels, kernel_size=3, padding=1)
        )

    def _build_embedders(self):
        """Build time and class embedders."""
        self.time_embed = TimestepEmbedder(self.time_dim)
        
        if self.num_classes is not None:
            self.y_embedder = ClassEmbedder(self.num_classes, self.time_dim)
            # Learnable mixing weights for time and class embeddings
            self.embed_mixer = nn.Parameter(torch.tensor([1.0, 1.0]))
        else:
            self.y_embedder = None
            self.embed_mixer = None

    def initialize_weights(self):
        """Enhanced weight initialization for stable training."""
        def _enhanced_init(module):
            if isinstance(module, BlackwellOptimizedLinear):
                # Special initialization for quantized layers
                if hasattr(module, 'linear'):
                    nn.init.xavier_uniform_(module.linear.weight)
                    module.linear.weight.data *= 0.8  # Scale for stability
                    if module.linear.bias is not None:
                        nn.init.zeros_(module.linear.bias)
                        
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Activation-aware initialization
                if hasattr(module, '_activation_type'):
                    if module._activation_type in ['relu', 'leaky_relu']:
                        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    else:
                        nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_enhanced_init)
        
        # Special initialization for output head (zero initialization for stability)
        if isinstance(self.output_head[-1], nn.Conv2d):
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)
        
        # Initialize embedding mixer weights
        if self.embed_mixer is not None:
            nn.init.ones_(self.embed_mixer)

    def _compute_conditioning(self, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute conditioning vector from timestep and optional class."""
        c = self.time_embed(t)
        
        if self.y_embedder is not None and y is not None:
            class_embed = self.y_embedder(y)
            
            # Mix embeddings with learnable weights
            if self.embed_mixer is not None:
                mixer_weights = F.softmax(self.embed_mixer, dim=0)
                c = mixer_weights[0] * c + mixer_weights[1] * class_embed
            else:
                c = c + class_embed
        
        return c

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                return_dict: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with comprehensive feature tracking.
        
        Args:
            x: Input tensor (B, C, H, W)
            t: Timestep tensor (B,)
            y: Optional class labels (B,)
            return_dict: Whether to return detailed information
            
        Returns:
            If return_dict=False: (output, aux_loss)
            If return_dict=True: Dictionary with detailed information
        """
        
        # Compute conditioning and SNR
        c = self._compute_conditioning(t, y)
        snr = compute_snr(t, T=self.T)
        
        # Initialize loss accumulator
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # For detailed tracking
        if return_dict:
            layer_outputs = []
            attention_maps = []
            expert_stats = []
        
        # Stem
        x = self.stem(x)
        if return_dict:
            layer_outputs.append(('stem', x.clone().detach()))
        
        # Encoder with skip collection
        skip_connections = []
        
        for level, (blocks, downsample) in enumerate(zip(self.encoder_blocks, self.encoder_downsamples)):
            # Process blocks at this level
            for block_idx, block in enumerate(blocks):
                if self.gradient_checkpointing and self.training:
                    x, loss = torch.utils.checkpoint.checkpoint(
                        block, x, c, t, snr, use_reentrant=False
                    )
                else:
                    x, loss = block(x, c, t, snr)
                
                aux_loss = aux_loss + loss
                
                if return_dict:
                    layer_outputs.append((f'encoder_L{level}_B{block_idx}', x.clone().detach()))
                    if hasattr(block, 'get_expert_stats'):
                        expert_stats.append(block.get_expert_stats())
            
            # Store skip connection
            if self.use_skip_connections:
                skip_connections.append(x)
            
            # Downsample
            x = downsample(x)
            if return_dict:
                layer_outputs.append((f'downsample_L{level}', x.clone().detach()))
        
        # Bottleneck processing
        x = self.bottleneck_expand(x)
        
        for block_idx, block in enumerate(self.bottleneck_blocks):
            if self.gradient_checkpointing and self.training:
                x, loss = torch.utils.checkpoint.checkpoint(
                    block, x, c, t, snr, use_reentrant=False
                )
            else:
                x, loss = block(x, c, t, snr)
            
            aux_loss = aux_loss + loss
            
            if return_dict:
                layer_outputs.append((f'bottleneck_B{block_idx}', x.clone().detach()))
                if hasattr(block, 'get_expert_stats'):
                    expert_stats.append(block.get_expert_stats())
        
        x = self.bottleneck_contract(x)
        
        # Decoder with skip connections
        for level, (upsample, blocks, skip_matcher) in enumerate(zip(
            self.decoder_upsamples, self.decoder_blocks, self.skip_matchers)):
            
            # Upsample
            x = upsample(x)
            
            # Handle skip connections
            if self.use_skip_connections and skip_connections:
                skip = skip_connections.pop()
                skip = skip_matcher(skip)
                
                if self.skip_connection_type == 'concat':
                    x = torch.cat([x, skip], dim=1)
                elif self.skip_connection_type == 'add':
                    x = x + skip
                elif self.skip_connection_type == 'gated':
                    gate = skip_matcher(skip)  # Gate is computed by skip_matcher
                    x = x + gate * skip
            
            # Process blocks at this level
            for block_idx, block in enumerate(blocks):
                if self.gradient_checkpointing and self.training:
                    x, loss = torch.utils.checkpoint.checkpoint(
                        block, x, c, t, snr, use_reentrant=False
                    )
                else:
                    x, loss = block(x, c, t, snr)
                
                aux_loss = aux_loss + loss
                
                if return_dict:
                    layer_outputs.append((f'decoder_L{level}_B{block_idx}', x.clone().detach()))
                    if hasattr(block, 'get_expert_stats'):
                        expert_stats.append(block.get_expert_stats())
        
        # Output head
        output = self.output_head(x)
        
        if return_dict:
            return {
                'output': output,
                'aux_loss': aux_loss,
                'layer_outputs': layer_outputs,
                'expert_stats': expert_stats,
                'conditioning': c,
                'snr': snr,
                'skip_connections_used': len([s for s in skip_connections if s is not None])
            }
        else:
            return output, aux_loss

    def enable_gradient_checkpointing(self, selective: bool = False, checkpoint_ratio: float = 0.5):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
        if selective:
            self.selective_checkpointing = True
            # Enable checkpointing for middle layers (most memory intensive)
            total_blocks = len(self.encoder_blocks) + len(self.decoder_blocks)
            num_checkpoint_blocks = max(1, int(total_blocks * checkpoint_ratio))
            start_idx = (total_blocks - num_checkpoint_blocks) // 2
            
            checkpoint_indices = set(range(start_idx, start_idx + num_checkpoint_blocks))
            
            for idx, blocks in enumerate(self.encoder_blocks):
                for block in blocks:
                    block.gradient_checkpointing = idx in checkpoint_indices
            
            for idx, blocks in enumerate(self.decoder_blocks, len(self.encoder_blocks)):
                for block in blocks:
                    block.gradient_checkpointing = idx in checkpoint_indices
            
            print(f"Enabled selective checkpointing for {num_checkpoint_blocks}/{total_blocks} block groups")
        else:
            # Enable for all blocks
            for blocks in self.encoder_blocks:
                for block in blocks:
                    block.gradient_checkpointing = True
            
            for blocks in self.decoder_blocks:
                for block in blocks:
                    block.gradient_checkpointing = True
            
            for block in self.bottleneck_blocks:
                block.gradient_checkpointing = True
            
            print("Enabled gradient checkpointing for all blocks")

    def _apply_cuda_optimizations(self):
        """Apply comprehensive CUDA optimizations."""
        if not torch.cuda.is_available():
            return
        
        # cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Enable optimized attention backends
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Tensor Core optimizations
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Memory optimizations
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        print("Applied CUDA optimizations: cuDNN benchmark, Flash Attention, TF32, memory optimization")

    def enable_compiled_forward(self, mode: str = "reduce-overhead", dynamic: bool = True):
        """Enable torch.compile for forward pass optimization."""
        if not hasattr(torch, 'compile'):
            print("torch.compile not available in this PyTorch version")
            return self
        
        try:
            self.forward = torch.compile(
                self.forward,
                mode=mode,
                dynamic=dynamic,
                fullgraph=False
            )
            print(f"Enabled torch.compile with mode='{mode}', dynamic={dynamic}")
        except Exception as e:
            print(f"Failed to compile model: {e}")
        
        return self

    def setup_ema(self):
        """Initialize Exponential Moving Average model."""
        if not self.use_ema:
            return
        
        from copy import deepcopy
        self.ema_model = deepcopy(self)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        
        print(f"Initialized EMA model with decay={self.ema_decay}")

    def update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema or self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def get_ema_model(self):
        """Get the EMA model for inference."""
        return self.ema_model if self.use_ema else self

    def optimize_for_training(self, 
                            enable_checkpointing: bool = True,
                            enable_compilation: bool = True,
                            setup_ema: bool = True):
        """Apply all training optimizations."""
        optimizations = []
        
        if torch.cuda.is_available():
            self._apply_cuda_optimizations()
            optimizations.append("CUDA optimizations")
        
        if enable_checkpointing:
            self.enable_gradient_checkpointing(selective=True, checkpoint_ratio=0.6)
            optimizations.append("gradient checkpointing")
        
        if enable_compilation and hasattr(torch, 'compile'):
            self.enable_compiled_forward(mode="reduce-overhead")
            optimizations.append("torch.compile")
        
        if setup_ema and self.use_ema:
            self.setup_ema()
            optimizations.append("EMA model")
        
        self.train()
        print(f"Optimized for training: {', '.join(optimizations)}")
        return self

    def optimize_for_inference(self, use_ema: bool = True):
        """Optimize model for inference."""
        # Use EMA model if available
        model = self.get_ema_model() if use_ema else self
        
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Disable training-specific features in MoE blocks
        def disable_training_features(module):
            if hasattr(module, 'block') and hasattr(module.block, 'moe'):
                module.block.moe.capacity_factor = 2.0  # Higher capacity for inference
                module.block.moe.jitter_noise = 0.0     # No jitter
                module.block.moe.training = False
        
        model.apply(disable_training_features)
        
        # Enable inference optimizations
        model.use_amp = True
        model.gradient_checkpointing = False
        
        if torch.cuda.is_available():
            model._apply_cuda_optimizations()
            if hasattr(torch, 'compile'):
                model.enable_compiled_forward(mode="max-autotune")
        
        print("Optimized for inference: disabled training features, enabled inference optimizations")
        return model

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            stats.update({
                "cuda_allocated_gb": allocated,
                "cuda_reserved_gb": reserved,
                "cuda_max_allocated_gb": max_allocated,
                "cuda_total_gb": total_memory,
                "cuda_free_gb": total_memory - reserved,
                "cuda_utilization": reserved / total_memory
            })
        
        # Model size information
        stats.update({
            "model_parameters": self.param_count,
            "trainable_parameters": self.trainable_params,
            "model_size_mb": self.param_count * 4 / 1024**2,  # Assuming float32
            "gradient_checkpointing": self.gradient_checkpointing,
            "compiled": hasattr(self.forward, '_torchdynamo_orig_callable')
        })
        
        return stats

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "architecture": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "hidden_size": self.hidden_size,
                "num_levels": self.num_levels,
                "blocks_per_level": self.blocks_per_level,
                "channels": self.channels,
                "total_encoder_blocks": sum(len(blocks) for blocks in self.encoder_blocks),
                "total_decoder_blocks": sum(len(blocks) for blocks in self.decoder_blocks),
                "bottleneck_blocks": len(self.bottleneck_blocks)
            },
            "parameters": {
                "total": self.param_count,
                "trainable": self.trainable_params,
                "size_mb": self.param_count * 4 / 1024**2
            },
            "configuration": {
                "num_experts": self.num_experts,
                "high_noise_experts": self.high_noise_experts,
                "snr_threshold": self.snr_threshold,
                "use_skip_connections": self.use_skip_connections,
                "skip_connection_type": self.skip_connection_type,
                "num_classes": self.num_classes
            },
            "optimizations": {
                "gradient_checkpointing": self.gradient_checkpointing,
                "selective_checkpointing": self.selective_checkpointing,
                "use_amp": self.use_amp,
                "compiled": hasattr(self.forward, '_torchdynamo_orig_callable'),
                "use_ema": self.use_ema,
                "fused_kernels": self._fused_kernels
            }
        }

    def save_checkpoint(self, path: str, include_ema: bool = True, **metadata):
        """Save model checkpoint with comprehensive information."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "hidden_size": self.hidden_size,
                "num_levels": self.num_levels,
                "blocks_per_level": self.blocks_per_level,
                "num_experts": self.num_experts,
                "high_noise_experts": self.high_noise_experts,
                "snr_threshold": self.snr_threshold,
                "T": self.T,
                "time_dim": self.time_dim,
                "num_classes": self.num_classes,
                "use_skip_connections": self.use_skip_connections,
                "skip_connection_type": self.skip_connection_type
            },
            "model_info": self.get_model_info(),
            "memory_stats": self.get_memory_stats(),
            **metadata
        }
        
        if include_ema and self.use_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
            checkpoint["ema_decay"] = self.ema_decay
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None, use_ema: bool = False):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        # Extract model configuration
        config = checkpoint["model_config"]
        
        # Create model instance
        model = cls(**config)
        
        # Load appropriate state dict
        if use_ema and "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
            print("Loaded EMA model weights")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded main model weights")
        
        # Restore EMA model if available
        if "ema_state_dict" in checkpoint and model.use_ema:
            model.setup_ema()
            model.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            model.ema_decay = checkpoint.get("ema_decay", 0.9999)
        
        if device:
            model = model.to(device)
        
        print(f"Loaded model from {path}")
        if "model_info" in checkpoint:
            info = checkpoint["model_info"]
            print(f"Parameters: {info['parameters']['total']:,}")
            print(f"Model size: {info['parameters']['size_mb']:.1f} MB")
        
        return model

    def get_flops_estimate(self, input_shape: Tuple[int, int, int, int] = (1, 4, 64, 64)) -> Dict[str, float]:
        """Estimate FLOPs for the model."""
        B, C, H, W = input_shape
        flops = 0
        
        # Stem
        flops += B * H * W * C * self.channels[0] * 9  # 3x3 conv
        
        # Encoder
        current_h, current_w = H, W
        for level in range(self.num_levels):
            in_ch = self.channels[level]
            out_ch = self.channels[level + 1]
            
            # Processing blocks
            for _ in range(self.blocks_per_level):
                # Attention (rough estimate)
                seq_len = current_h * current_w
                num_heads = max(1, in_ch // 64)
                head_dim = in_ch // num_heads
                
                # Self-attention: Q, K, V projections + attention computation
                flops += B * seq_len * in_ch * in_ch * 3  # QKV projections
                flops += B * num_heads * seq_len * seq_len * head_dim  # Attention
                flops += B * seq_len * in_ch * in_ch  # Output projection
                
                # MoE FFN (approximate)
                if hasattr(self, 'num_experts'):
                    expert_size = in_ch * 8 // 3  # Typical expansion
                    active_experts = 2  # top_k
                    flops += B * seq_len * in_ch * expert_size * active_experts * 2  # Up + down
            
            # Downsampling
            flops += B * current_h * current_w * in_ch * out_ch * 9  # 3x3 conv stride 2
            current_h //= 2
            current_w //= 2
        
        # Bottleneck (simplified)
        bottleneck_ch = self.channels[-1] * 2
        seq_len = current_h * current_w
        flops += B * seq_len * bottleneck_ch * bottleneck_ch * 6  # 3 blocks, simplified
        
        # Decoder (mirror of encoder)
        for level in reversed(range(self.num_levels)):
            current_h *= 2
            current_w *= 2
            
            in_ch = self.channels[level + 1]
            out_ch = self.channels[level]
            
            # Upsampling
            flops += B * current_h * current_w * in_ch * out_ch * 4  # 2x2 transconv
            
            # Processing blocks
            for _ in range(self.blocks_per_level + 1):
                seq_len = current_h * current_w
                num_heads = max(1, out_ch // 64)
                head_dim = out_ch // num_heads
                
                # Attention
                flops += B * seq_len * out_ch * out_ch * 3
                flops += B * num_heads * seq_len * seq_len * head_dim
                flops += B * seq_len * out_ch * out_ch
                
                # MoE FFN
                expert_size = out_ch * 8 // 3
                active_experts = 2
                flops += B * seq_len * out_ch * expert_size * active_experts * 2
        
        # Output head
        flops += B * H * W * self.channels[0] * self.out_channels * 9  # Final conv
        
        return {
            "total_flops": flops,
            "gflops": flops / 1e9,
            "flops_per_param": flops / self.param_count,
            "input_shape": input_shape
        }

    def profile_forward_pass(self, 
                           input_shape: Tuple[int, int, int, int] = (2, 4, 64, 64),
                           num_classes: Optional[int] = None,
                           device: str = "cuda",
                           warmup_steps: int = 10,
                           profile_steps: int = 100) -> Dict[str, float]:
        """Profile forward pass performance."""
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
        
        self.to(device)
        self.eval()
        
        # Create dummy inputs
        B, C, H, W = input_shape
        x = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, self.T, (B,), device=device)
        y = torch.randint(0, num_classes or 10, (B,), device=device) if self.num_classes else None
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = self.forward(x, t, y)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(profile_steps):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    _ = self.forward(x, t, y)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
                else:
                    import time
                    start_time = time.time()
                    _ = self.forward(x, t, y)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = times[10:]  # Remove first few for stability
        if memory_usage:
            memory_usage = memory_usage[10:]
        
        flops_info = self.get_flops_estimate(input_shape)
        
        stats = {
            "mean_time_ms": sum(times) / len(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "throughput_samples_per_sec": 1000 * B / (sum(times) / len(times)),
            "gflops_per_sec": flops_info["gflops"] * 1000 / (sum(times) / len(times)),
            "input_shape": input_shape,
            "device": device,
            "parameters": self.param_count
        }
        
        if memory_usage:
            stats.update({
                "mean_memory_gb": sum(memory_usage) / len(memory_usage),
                "max_memory_gb": max(memory_usage)
            })
        
        return stats

    def analyze_expert_usage(self, dataloader, max_batches: int = 10) -> Dict[str, Any]:
        """Analyze expert usage patterns across different noise levels."""
        self.eval()
        
        expert_usage = {}
        snr_distributions = []
        routing_patterns = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                x, t = batch[:2]
                y = batch[2] if len(batch) > 2 else None
                
                # Forward pass with detailed info
                result = self.forward(x, t, y, return_dict=True)
                expert_stats = result.get('expert_stats', [])
                snr = result.get('snr', None)
                
                if snr is not None:
                    snr_distributions.append(snr.cpu().numpy())
                
                # Aggregate expert usage
                for stats in expert_stats:
                    for key, value in stats.items():
                        if key not in expert_usage:
                            expert_usage[key] = []
                        if isinstance(value, torch.Tensor):
                            expert_usage[key].append(value.cpu().numpy())
                        else:
                            expert_usage[key].append(value)
        
        # Process collected data
        analysis = {
            "expert_usage_summary": {},
            "snr_analysis": {},
            "routing_efficiency": {},
        }
        
        # Analyze SNR distribution
        if snr_distributions:
            all_snr = np.concatenate(snr_distributions)
            analysis["snr_analysis"] = {
                "mean": float(np.mean(all_snr)),
                "std": float(np.std(all_snr)),
                "min": float(np.min(all_snr)),
                "max": float(np.max(all_snr)),
                "percentiles": {
                    "25": float(np.percentile(all_snr, 25)),
                    "50": float(np.percentile(all_snr, 50)),
                    "75": float(np.percentile(all_snr, 75)),
                    "95": float(np.percentile(all_snr, 95))
                }
            }
        
        # Analyze expert usage
        for key, values in expert_usage.items():
            if values and isinstance(values[0], np.ndarray):
                all_values = np.concatenate(values)
                analysis["expert_usage_summary"][key] = {
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                    "distribution": np.histogram(all_values, bins=20)[0].tolist()
                }
        
        return analysis

    def visualize_architecture(self, save_path: Optional[str] = None) -> str:
        """Generate a text visualization of the model architecture."""
        lines = []
        lines.append("UNetDeltaNet Architecture")
        lines.append("=" * 50)
        lines.append(f"Parameters: {self.param_count:,}")
        lines.append(f"Size: {self.param_count * 4 / 1024**2:.1f} MB")
        lines.append("")
        
        # Input
        lines.append(f"Input: ({self.in_channels} channels)")
        lines.append("│")
        
        # Stem
        lines.append(f"├─ Stem: {self.in_channels} → {self.channels[0]}")
        lines.append("│")
        
        # Encoder
        lines.append("├─ Encoder:")
        for level in range(self.num_levels):
            in_ch = self.channels[level]
            out_ch = self.channels[level + 1]
            lines.append(f"│  ├─ Level {level}: {in_ch} channels")
            lines.append(f"│  │  ├─ {self.blocks_per_level} × DeltaNet blocks")
            lines.append(f"│  │  └─ Downsample: {in_ch} → {out_ch}")
        lines.append("│")
        
        # Bottleneck
        bottleneck_ch = self.channels[-1] * 2
        lines.append(f"├─ Bottleneck: {self.channels[-1]} → {bottleneck_ch} → {self.channels[-1]}")
        lines.append(f"│  └─ {len(self.bottleneck_blocks)} × DeltaNet blocks")
        lines.append("│")
        
        # Decoder
        lines.append("├─ Decoder:")
        for level in reversed(range(self.num_levels)):
            in_ch = self.channels[level + 1]
            out_ch = self.channels[level]
            lines.append(f"│  ├─ Level {self.num_levels - 1 - level}: {in_ch} channels")
            lines.append(f"│  │  ├─ Upsample: {in_ch} → {out_ch}")
            if self.use_skip_connections:
                lines.append(f"│  │  ├─ Skip connection ({self.skip_connection_type})")
            lines.append(f"│  │  └─ {self.blocks_per_level + 1} × DeltaNet blocks")
        lines.append("│")
        
        # Output
        lines.append(f"└─ Output: {self.channels[0]} → {self.out_channels}")
        lines.append("")
        
        # Configuration details
        lines.append("Configuration:")
        lines.append(f"  • Experts: {self.num_experts} (high-noise: {self.high_noise_experts})")
        lines.append(f"  • SNR threshold: {self.snr_threshold}")
        lines.append(f"  • Skip connections: {self.skip_connection_type if self.use_skip_connections else 'disabled'}")
        if self.num_classes:
            lines.append(f"  • Classes: {self.num_classes}")
        
        lines.append("")
        lines.append("Optimizations:")
        lines.append(f"  • Gradient checkpointing: {self.gradient_checkpointing}")
        lines.append(f"  • Mixed precision: {self.use_amp}")
        lines.append(f"  • EMA: {self.use_ema}")
        lines.append(f"  • Compiled: {hasattr(self.forward, '_torchdynamo_orig_callable')}")
        
        visualization = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(visualization)
            print(f"Architecture visualization saved to {save_path}")
        
        return visualization

    def __repr__(self) -> str:
        return (f"UNetDeltaNet(\n"
                f"  parameters={self.param_count:,},\n"
                f"  size={self.param_count * 4 / 1024**2:.1f}MB,\n"
                f"  levels={self.num_levels},\n"
                f"  experts={self.num_experts},\n"
                f"  channels={self.channels}\n"
                f")")

# Aliases for backward compatibility and convenience
DiT = UNetDeltaNet
