import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from enhanced.quantization import QuantizedLinear, safe_torch_compile
from enhanced.global_scheduler import get_global_scheduler
from enhanced.DeltaAttention import DeltaNetEnhancedAttention

@torch.jit.script
def fused_modulate_deltanet(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Ultra-conservative clamping for FP4 with DeltaNet routing
    scale_clamped = scale.clamp(-1.0, 1.0)
    return x.mul(scale_clamped.unsqueeze(1).add(1.0)).add_(shift.unsqueeze(1))

def memory_efficient_expert_computation(
    x_flat: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_probs: torch.Tensor,
    experts: nn.ModuleList,
    capacity: int,
    num_experts: int,
    memory_threshold_gb: float = 2.0
) -> torch.Tensor:
    """
    Memory-efficient expert computation with automatic batching and cleanup.
    """
    # Monitor memory usage
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated() / 1e9

    # Determine if we need chunked processing
    use_chunked = False
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if current_memory > total_memory * 0.7:  # Use 70% as threshold
            use_chunked = True

    if use_chunked:
        return _chunked_expert_computation(
            x_flat, top_k_indices, top_k_probs, experts, capacity, num_experts
        )
    else:
        return _standard_expert_computation(
            x_flat, top_k_indices, top_k_probs, experts, capacity, num_experts
        )

def _chunked_expert_computation(
    x_flat: torch.Tensor, top_k_indices: torch.Tensor, top_k_probs: torch.Tensor,
    experts: nn.ModuleList, capacity: int, num_experts: int
) -> torch.Tensor:
    """Process experts in chunks to reduce memory usage."""
    output = torch.zeros_like(x_flat)
    chunk_size = min(4, num_experts)  # Process 4 experts at a time

    for start_idx in range(0, num_experts, chunk_size):
        end_idx = min(start_idx + chunk_size, num_experts)

        # Process this chunk of experts
        chunk_output = _standard_expert_computation(
            x_flat, top_k_indices, top_k_probs,
            experts[start_idx:end_idx], capacity, chunk_size
        )

        # Add to output
        output.add_(chunk_output)

        # Clean up
        del chunk_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output

def _standard_expert_computation(
    x_flat: torch.Tensor, top_k_indices: torch.Tensor, top_k_probs: torch.Tensor,
    experts: nn.ModuleList, capacity: int, num_experts: int
) -> torch.Tensor:
    """Standard vectorized expert computation with memory optimizations."""
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
        # Ensure dtype consistency for index_add_ operation
        weighted_output = weighted_output.to(output.dtype)
        output.index_add_(0, token_positions, weighted_output)

        # Clean up intermediate tensors
        del expert_input, expert_output, weighted_output, token_indices, weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output

# Compile with guard
vectorized_expert_computation = safe_torch_compile(vectorized_expert_computation, dynamic=True)

def vectorized_expert_computation(x_flat: torch.Tensor, top_k_indices: torch.Tensor, 
                                 top_k_probs: torch.Tensor, experts: nn.ModuleList,
                                 capacity: int, num_experts: int) -> torch.Tensor:
    """Optimized vectorized expert computation.
    Guarded by safe_torch_compile at import time.
    """
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
        # Ensure dtype consistency for index_add_ operation
        weighted_output = weighted_output.to(output.dtype)
        output.index_add_(0, token_positions, weighted_output)
    
    return output

# Compile with guard
vectorized_expert_computation = safe_torch_compile(vectorized_expert_computation, dynamic=True)

# Ultimate DeltaNet DiT Block
class UltimateDeltaNetDiTBlock(nn.Module):
    """
    DiT Transformer block composed of:
        • FiLM-style timestep modulation
        • DeltaNet Enhanced Attention (above)
        • Dual-bank MoE feed-forward
    It returns the updated hidden states and a single scalar
    auxiliary loss (attention-entropy + MoE load balance).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        moe_cls,                          # pass your DeltaNetEnhancedMoE class
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
        lb_loss_coef: float = 0.01,
        # Expert bank size overrides
        low_noise_experts_num: Optional[int] = None,
        high_noise_experts_num: Optional[int] = None,
        share_low_with_high: bool = False,
        dropout: float = 0.0,
        snr_threshold: float = 0.50,
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

        self.hidden_size   = hidden_size
        self.snr_threshold = snr_threshold
        self.dropout       = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- FiLM projection ------------------------------------
        self.mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
        )

        # ---- Attention ------------------------------------------
        self.attn = DeltaNetEnhancedAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            qk_norm_enabled=qk_norm_enabled,
            qk_norm_eps=qk_norm_eps,
            local_global_mixing=local_global_mixing,
            local_attention_window=local_attention_window,
            global_attention_start_layer=global_attention_start_layer,
            layer_idx=layer_idx,
        )

        # ---- MoE feed-forward -----------------------------------
        self.moe = moe_cls(
            hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            jitter_noise=jitter_noise,
            load_balance_loss_coef=lb_loss_coef,
            low_noise_experts_num=low_noise_experts_num,
            high_noise_experts_num=high_noise_experts_num,
            share_low_with_high=share_low_with_high,
        )

        # ---- Layer norms ----------------------------------------
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,      # (B, N, C)
        temb: torch.Tensor,   # (B, C)
        t:    torch.Tensor,   # (B,)  diffusion step indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_out    : Tensor (B, N, C)
        aux_loss : Tensor scalar   attn entropy + MoE load balance
        """
        B, N, C = x.shape
        assert temb.shape == (B, self.hidden_size), "temb must be (B, hidden_size)"

        # ----- FiLM parameters -----------------------------------
        shift1, scale1, shift2, scale2 = self.mod_proj(temb).chunk(4, dim=-1)

        # ----- Attention branch ----------------------------------
        h = self.norm1(x)
        h = fused_modulate_deltanet(h, shift1, scale1)  # user-provided fused kernel
        h, attn_entropy = self.attn(h)                 # returns tuple
        x = x + self.dropout(h)

        # ----- MoE / FF branch -----------------------------------
        h = self.norm2(x)
        h = fused_modulate_deltanet(h, shift2, scale2)
        moe_out, lb_loss = self.moe(h, t, self.snr_threshold)
        x = x + self.dropout(moe_out)

        return x, (attn_entropy + lb_loss)


class DeltaNetEnhancedMoE(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 top_k: int = 2,
                 capacity_factor: float = 1.25,
                 jitter_noise: float = 0.01,
                 load_balance_loss_coef: float = 0.01,
                 low_noise_experts_num: Optional[int] = None,
                 high_noise_experts_num: Optional[int] = None,
                 share_low_with_high: bool = False,
                 **unused):
        super().__init__()

        # Determine per-bank expert counts
        self.num_experts_high = high_noise_experts_num if high_noise_experts_num is not None else num_experts
        self.num_experts_low  = low_noise_experts_num  if low_noise_experts_num  is not None else num_experts
        self.top_k = top_k
        self.num_experts = max(self.num_experts_high, self.num_experts_low)  # gating dimension
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.lb_coef = load_balance_loss_coef

        # ------------- gate -------------
        self.gate = nn.Sequential(
            QuantizedLinear(hidden_size + 6, hidden_size // 2, bias=True),
            nn.GELU(),
            QuantizedLinear(hidden_size // 2, num_experts, bias=False)
        )

        # ------------- experts -----------
        self.high_noise_experts = nn.ModuleList(
            [self._create_expert(hidden_size) for _ in range(self.num_experts_high)]
        )

        if share_low_with_high:         # zero extra parameters
            self.low_noise_experts = self.high_noise_experts
        else:                           # full dual-bank
            self.low_noise_experts = nn.ModuleList(
                [self._create_expert(hidden_size) for _ in range(self.num_experts_low)]
            )

        self.register_buffer("_expert_ids", torch.arange(self.num_experts))
        self.scheduler = get_global_scheduler()


    def _create_expert(self, hidden_size: int) -> nn.Module:
        """Create optimized expert."""
        intermediate_size = int(hidden_size * 8 // 3)
        # Custom SwiGLU module to replace nn.Lambda
        class SwiGLU(nn.Module):
            def __init__(self, intermediate_size):
                super().__init__()
                self.intermediate_size = intermediate_size
            
            def forward(self, x):
                return F.silu(x[:, :self.intermediate_size]) * x[:, self.intermediate_size:]
        
        expert = nn.Sequential(
            QuantizedLinear(hidden_size, intermediate_size * 2, bias=False),
            SwiGLU(intermediate_size),
            QuantizedLinear(intermediate_size, hidden_size, bias=False)
        )
        
        # Optimized initialization
        with torch.no_grad():
            for layer in [expert[0], expert[2]]:
                layer.weight.data.mul_(0.4)
        
        return expert

    def _compute_activation_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized activation statistics computation. Guarded via safe_torch_compile."""
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        std = centered.std(dim=-1, keepdim=True)
        min_val, max_val = x.min(dim=-1, keepdim=True)[0], x.max(dim=-1, keepdim=True)[0]
        l2_norm = x.norm(dim=-1, keepdim=True)
        sparsity = (x.abs() < 1e-6).float().mean(dim=-1, keepdim=True)
        
        return torch.cat([mean, std, min_val, max_val, l2_norm, sparsity], dim=-1)

    # Guard-compiled method assignment performed after class definition

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        snr_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        tokens = x.reshape(-1, C)
        T = tokens.size(0)

        # -----------------------------------------------------------
        # 1)  Gate: compute routing probabilities
        # -----------------------------------------------------------
        stats = self._compute_activation_stats(tokens)          # (T, 6)
        gate_logits = self.gate(torch.cat([tokens, stats], dim=-1))  # (T, E)

        # temperature annealing + optional jitter
        temperature = max(self.scheduler.get_value("moe_temperature"), 0.01)
        gate_logits = gate_logits / temperature
        if self.training and self.jitter_noise > 0.0:
            gate_logits.add_(torch.empty_like(gate_logits)
                            .uniform_(-self.jitter_noise, self.jitter_noise))

        gate_probs = F.softmax(gate_logits, dim=-1)              # (T, E)
        top_k_probs, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        # cheap renormalisation
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # -----------------------------------------------------------
        # 2)  Auxiliary load-balancing loss (Shazeer et al.)
        # -----------------------------------------------------------
        if self.training:
            # density_1 : how often each expert is chosen
            density_1 = gate_probs.mean(0)                                  # (E,)
            # density_2 : same but after top-k sparsification
            chosen_counts = torch.bincount(top_k_idx.flatten(),
                                        minlength=self.num_experts).float()
            density_2 = chosen_counts / chosen_counts.sum().clamp_min(1e-8) # (E,)
            load_balance_loss = (density_1 * density_2).sum() \
                                * self.num_experts * self.lb_coef
        else:
            load_balance_loss = tokens.new_zeros(())

        # -----------------------------------------------------------
        # 3)  Capacity – independent of temperature
        # -----------------------------------------------------------
        # NOTE: This must be outside the training block!
        # For OFM, we use time directly instead of alphas_cumprod
        # t is already in [0,1] where 0=noise, 1=clean
        # So we can approximate SNR from t directly
        t_mean = t.mean()
        # Simple heuristic: use low-noise experts when t > threshold
        # (closer to clean data)
        use_low_bank = (t_mean >= snr_threshold)
        experts = self.low_noise_experts if use_low_bank else self.high_noise_experts
        
        capacity = int(self.capacity_factor * T / len(experts))

        # -----------------------------------------------------------
        # 4)  Expert bank selection already done above for capacity
        # -----------------------------------------------------------
        # (Already selected experts based on SNR above)

        # -----------------------------------------------------------
        # 5)  Memory-efficient expert computation
        # -----------------------------------------------------------
        output_flat = memory_efficient_expert_computation(
            tokens,                 # (T, C)
            top_k_idx,              # (T, k)
            top_k_probs,            # (T, k)
            experts,                # ModuleList
            capacity,
            len(experts)
        )                           # (T, C)

        # NOTE: if you later want *per-token* bank selection,
        # pass an additional boolean mask to the vectorised
        # kernel and invoke it twice (once per bank) – the rest
        # of the logic stays as is.

        y = output_flat.view(B, N, C)
        return y, load_balance_loss

# After class definition, wrap the method with safe_torch_compile if available
try:
    # Bind the compiled function back to the class method if compilation succeeds
    DeltaNetEnhancedMoE._compute_activation_stats = safe_torch_compile(
        DeltaNetEnhancedMoE._compute_activation_stats, dynamic=True
    )
except Exception:
    # If something unexpected happens, leave it as-is
    pass
