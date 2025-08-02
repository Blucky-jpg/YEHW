import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange
from ..core.quantization import BlackwellOptimizedLinear, sage_attention_with_fp8
from .progressive_pruning_module import ProgressivePruningSystem
from .context_adaptive_gating import EnhancedContextGating
from .enhanced_dit_modules import DepthwiseFIRConv1d, CrossHeadMixing, delta_rule_chunkwise
from .global_scheduler import get_global_scheduler

@torch.jit.script
def fused_modulate_deltanet(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Enhanced fused modulation optimized for DeltaNet integration."""
    # Ultra-conservative clamping for FP4 with DeltaNet routing
    scale_clamped = scale.clamp(-1.0, 1.0)
    return x.mul(scale_clamped.unsqueeze(1).add(1.0)).add_(shift.unsqueeze(1))

@torch.compile(dynamic=True)
def vectorized_expert_computation(x_flat: torch.Tensor, top_k_indices: torch.Tensor, 
                                 top_k_probs: torch.Tensor, experts: nn.ModuleList,
                                 capacity: int, num_experts: int) -> torch.Tensor:
    """Optimized vectorized expert computation."""
    output = torch.zeros_like(x_flat)
    
    # Pre-compute expert masks for all experts at once
    expert_masks = (top_k_indices.unsqueeze(-1) == torch.arange(num_experts, device=x_flat.device))
    
    for expert_idx in range(num_experts):
        expert_mask = expert_masks[..., expert_idx]  # (T, top_k)
        
        if not expert_mask.any():
            continue
        
        token_indices = torch.where(expert_mask)
        token_positions, k_positions = token_indices
        
        if len(token_positions) == 0:
            continue
        
        weights = top_k_probs[token_positions, k_positions]
        
        # Apply capacity constraint with more efficient selection
        if len(token_positions) > capacity:
            _, selected = torch.topk(weights, capacity, sorted=False)
            token_positions = token_positions[selected]
            weights = weights[selected]
        
        if len(token_positions) == 0:
            continue
        
        # Process with optimized indexing
        expert_input = x_flat[token_positions]
        expert_output = experts[expert_idx](expert_input)
        weighted_output = expert_output * weights.unsqueeze(-1)
        output.index_add_(0, token_positions, weighted_output)
    
    return output

class DeltaNetEnhancedAttention(nn.Module):
    """
    Enhanced attention module with optimized path processing and caching.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 fir_short_kernel: int = 3, fir_long_kernel: int = 7,
                 id_static_init: float = 0.2, fusion_hidden_mult: float = 1.0,
                 use_sage_attention: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_sage_attention = use_sage_attention
        
        # Get global scheduler
        self.scheduler = get_global_scheduler()
        
        # Standard attention projections with Blackwell optimization
        self.qkv = BlackwellOptimizedLinear(dim, dim * 3, bias=False)
        self.proj = BlackwellOptimizedLinear(dim, dim, bias=True)
        
        # Multi-path processing - cached for efficiency
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_dim, fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_dim, fir_long_kernel)
        self.cross_head_mixing = CrossHeadMixing(num_heads, mix_init=0.02)
        
        # Advanced routing systems
        self.progressive_pruning = ProgressivePruningSystem(
            dim, num_heads, num_paths=5, use_global_scheduler=True
        )
        
        self.context_gating = EnhancedContextGating(
            dim, num_heads, self.head_dim, num_paths=5,
            fusion_hidden_mult=fusion_hidden_mult
        )
        
        # Enhanced identity path with gating
        self.identity_proj = BlackwellOptimizedLinear(dim, dim, bias=False)
        
        # Pre-computed constants
        self.id_static_logit = nn.Parameter(
            torch.full((num_heads,), math.log(id_static_init / (1.0 - id_static_init)))
        )
        
        self.id_gate_proj = BlackwellOptimizedLinear(dim, num_heads, bias=True)
        with torch.no_grad():
            self.id_gate_proj.bias.fill_(-1.5)
        
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        
        # Optimized dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.proj_dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Cache for frequently accessed tensors
        self._cache = {'static_gate': None, 'step': -1}

    def _get_cached_static_gate(self) -> torch.Tensor:
        """Cache static gate computation for efficiency."""
        current_step = self.scheduler.get_step()
        if self._cache['step'] != current_step or self._cache['static_gate'] is None:
            self._cache['static_gate'] = torch.sigmoid(self.id_static_logit)
            self._cache['step'] = current_step
        return self._cache['static_gate']

    @torch.compile(dynamic=True)
    def _process_paths_vectorized(self, v_reshaped: torch.Tensor, 
                                  identity_out: torch.Tensor, 
                                  delta_out: torch.Tensor) -> List[torch.Tensor]:
        """Vectorized path processing for better performance."""
        # Process FIR paths together
        fir_outputs = torch.stack([
            self.fir_short(v_reshaped),
            self.fir_long(v_reshaped)
        ], dim=0)  # (2, B, N, H, D)
        
        # Apply cross-head mixing to both at once
        mixed_outputs = torch.stack([
            self.cross_head_mixing(fir_outputs[0]),
            self.cross_head_mixing(fir_outputs[1])
        ], dim=0)  # (2, B, N, H, D)
        
        return [mixed_outputs[0], mixed_outputs[1], delta_out, v_reshaped, identity_out]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # QKV computation with better memory layout
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        
        # Optimized Sage_Attention
        attn_out = sage_attention_with_fp8(
            q, k, v, 
            is_causal=True, 
            use_fp8_context=self.use_sage_attention
        ).transpose(1, 2)  # (B, N, H, D)
        
        # Prepare tensors for path processing
        v_reshaped = v.transpose(1, 2)  # (B, N, H, D)
        
        # Enhanced identity path with cached static gate
        identity_out = rearrange(
            self.identity_proj(x), "b n (h d) -> b n h d", h=self.num_heads
        )
        
        # Compute gates efficiently
        dyn_gate = torch.sigmoid(self.id_gate_proj(x))
        static_gate = self._get_cached_static_gate()
        id_gate = dyn_gate * static_gate.view(1, 1, -1)
        identity_out.mul_(id_gate.unsqueeze(-1))
        
        # Delta-rule computation with optimized tensor operations
        q_d = q.transpose(1, 2)  # (B, H, N, D)
        k_d = k.transpose(1, 2)  # (B, H, N, D)
        v_d = v.transpose(1, 2)  # (B, H, N, D)
        
        # More efficient beta computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)).mul_(self.scale)
        beta = F.softmax(attn_scores, dim=-1).sum(dim=-1)
        
        delta_out_d, _ = delta_rule_chunkwise(q_d, k_d, v_d, beta)
        delta_out = delta_out_d.transpose(1, 2)  # (B, N, H, D)
        
        # Vectorized path processing
        path_outputs = self._process_paths_vectorized(v_reshaped, identity_out, delta_out)
        
        # Apply routing systems
        pruned_out, entropy_loss1 = self.progressive_pruning(x, path_outputs)
        context_out, entropy_loss2 = self.context_gating(x, [pruned_out], local_path_idx=0)
        
        # Final processing
        output = rearrange(context_out, "b n h d -> b n (h d)")
        output = self.proj(output)
        
        if self.proj_dropout:
            output = self.proj_dropout(output)
        
        return output, entropy_loss1 + entropy_loss2

class UltimateDeltaNetDiTBlock(nn.Module):
    """
    Ultimate DiT block with optimized scheduling and memory management.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_experts: int = 8, 
                 top_k: int = 2, jitter_noise: float = 0.01, dropout: float = 0.0, 
                 use_moe: bool = False, snr_threshold: float = 0.5, 
                 use_sage_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.snr_threshold = snr_threshold
        
        # Get global scheduler
        self.scheduler = get_global_scheduler()
        
        # Layer normalization with optimized parameters
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Enhanced attention
        self.attn = DeltaNetEnhancedAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, 
            use_sage_attention=use_sage_attention
        )
        
        # Enhanced MoE
        self.moe = DeltaNetEnhancedMoE(
            hidden_size, num_experts, top_k, 
            jitter_noise=jitter_noise, use_moe=use_moe
        )
        
        # Optimized modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            BlackwellOptimizedLinear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Better initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Optimized weight initialization."""
        with torch.no_grad():
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    @torch.compile(dynamic=True)
    def _apply_modulation_and_gating(self, x: torch.Tensor, x_norm: torch.Tensor, 
                                     shift: torch.Tensor, scale: torch.Tensor, 
                                     gate: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        """Fused modulation and gating operation."""
        x_mod = fused_modulate_deltanet(x_norm, shift, scale)
        return x.add_(gate.unsqueeze(1) * processed)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass."""
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        temp = self.scheduler.get_value("mod_temperature")
        modulation.mul_(temp)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
        
        # Attention block with fused operations
        x_norm1 = self.norm1(x)
        x_mod1 = fused_modulate_deltanet(x_norm1, shift_msa, scale_msa)
        
        attn_out, attn_loss = self.attn(x_mod1)
        x.add_(gate_msa.unsqueeze(1) * attn_out)
        
        # MoE block with fused operations
        x_norm2 = self.norm2(x)
        x_mod2 = fused_modulate_deltanet(x_norm2, shift_mlp, scale_mlp)
        
        moe_out, moe_loss = self.moe(x_mod2, t, self.snr_threshold)
        x.add_(gate_mlp.unsqueeze(1) * moe_out)
        
        # Update scheduler
        if self.training:
            self.scheduler.step()
        
        return x, attn_loss + moe_loss

class DeltaNetEnhancedMoE(nn.Module):
    """
    Enhanced MoE with optimized routing and expert computation.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.25, jitter_noise: float = 0.01,
                 load_balance_loss_coef: float = 0.01, use_moe: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.load_balance_loss_coef = load_balance_loss_coef
        self.use_moe = use_moe
        
        self.scheduler = get_global_scheduler()
        
        # Enhanced routing with statistical features
        self.gate = nn.Sequential(
            BlackwellOptimizedLinear(hidden_size + 6, hidden_size // 2, bias=True),
            nn.GELU(),
            BlackwellOptimizedLinear(hidden_size // 2, num_experts, bias=False)
        )
        
        # Experts
        self.high_noise_experts = nn.ModuleList([
            self._create_expert(hidden_size) for _ in range(num_experts)
        ])
        if use_moe:
            self.low_noise_experts = nn.ModuleList([
                self._create_expert(hidden_size) for _ in range(num_experts)
            ])

    def _create_expert(self, hidden_size: int) -> nn.Module:
        """Create optimized expert."""
        intermediate_size = int(hidden_size * 8 // 3)
        expert = nn.Sequential(
            BlackwellOptimizedLinear(hidden_size, intermediate_size * 2, bias=False),
            nn.Lambda(lambda x: F.silu(x[:, :intermediate_size]) * x[:, intermediate_size:]),
            BlackwellOptimizedLinear(intermediate_size, hidden_size, bias=False)
        )
        
        # Optimized initialization
        with torch.no_grad():
            for layer in [expert[0], expert[2]]:
                layer.weight.data.mul_(0.4)
        
        return expert

    @torch.compile(dynamic=True)
    def _compute_activation_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized activation statistics computation."""
        # Compute all stats in one pass for better memory efficiency
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        std = centered.std(dim=-1, keepdim=True)
        min_val, max_val = x.min(dim=-1, keepdim=True)[0], x.max(dim=-1, keepdim=True)[0]
        l2_norm = x.norm(dim=-1, keepdim=True)
        sparsity = (x.abs() < 1e-6).float().mean(dim=-1, keepdim=True)
        
        return torch.cat([mean, std, min_val, max_val, l2_norm, sparsity], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, snr_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        T = x_flat.shape[0]
        
        # Enhanced routing
        stats = self._compute_activation_stats(x_flat)
        gate_input = torch.cat([x_flat, stats], dim=-1)
        gate_logits = self.gate(gate_input)
        
        # Apply temperature and jitter
        temperature = self.scheduler.get_value("moe_temperature")
        gate_logits.div_(max(temperature, 0.01))
        
        if self.training and self.jitter_noise > 0:
            gate_logits.add_(
                torch.empty_like(gate_logits).uniform_(-self.jitter_noise, self.jitter_noise)
            )
        
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)
        
        # Load balancing loss
        load_balance_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts).float()
            expert_freq = expert_counts / expert_counts.sum().clamp_min(1e-8)
            gate_mean = gate_probs.mean(dim=0)
            load_balance_loss = (gate_mean * expert_freq).sum() * self.num_experts * self.load_balance_loss_coef
        
        # Adaptive capacity
        base_capacity = int(self.capacity_factor * T / self.num_experts)
        temp_factor = max(0.5, temperature)
        capacity = int(base_capacity * temp_factor)
        
        # Select experts
        if self.use_moe:
            alphas_cumprod = self.scheduler.get_value("alphas_cumprod")
            snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
            experts = self.high_noise_experts if snr.item() < snr_threshold else self.low_noise_experts
        else:
            experts = self.high_noise_experts
        
        # Vectorized expert computation
        output = vectorized_expert_computation(
            x_flat, top_k_indices, top_k_probs, experts, capacity, self.num_experts
        )
        
        return output.view(B, N, C), load_balance_loss
