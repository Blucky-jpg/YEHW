"""
GroveMoE: Enhanced Mixture of Experts with Adjugate Experts
Implementation of the Grove MoE architecture from the paper.

Key innovations:
1. Adjugate experts shared across groups
2. Dynamic computation allocation
3. Heterogeneous expert sizes
4. Improved load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math

from enhanced.quantization import BlackwellOptimizedLinear, safe_torch_compile
from enhanced.global_scheduler import get_global_scheduler


class AdjugateExpert(nn.Module):
    """
    Lightweight adjugate expert shared across a group of main experts.
    Uses smaller intermediate dimension for efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'swiglu',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        
        if activation == 'swiglu':
            # SwiGLU activation requires 2x intermediate for gate
            self.up_proj = BlackwellOptimizedLinear(
                hidden_size, intermediate_size * 2, bias=False
            )
            self.act_fn = None  # Handled in forward
        else:
            self.up_proj = BlackwellOptimizedLinear(
                hidden_size, intermediate_size, bias=False
            )
            self.act_fn = nn.SiLU() if activation == 'silu' else nn.GELU()
        
        self.down_proj = BlackwellOptimizedLinear(
            intermediate_size, hidden_size, bias=False
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize down projection to zero for stability during upcycling
        with torch.no_grad():
            self.down_proj.weight.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adjugate expert."""
        up = self.up_proj(x)
        
        if self.activation == 'swiglu':
            # SwiGLU: silu(first_half) * second_half
            gate, up = up.chunk(2, dim=-1)
            up = F.silu(gate) * up
        else:
            up = self.act_fn(up)
        
        up = self.dropout(up)
        down = self.down_proj(up)
        
        return down


class GroveMoE(nn.Module):
    """
    Grove Mixture of Experts with adjugate experts and dynamic activation.
    
    Key features:
    - Experts organized into disjoint groups
    - Each group has a shared adjugate expert
    - Dynamic computation based on routing patterns
    - Heterogeneous expert sizes supported
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 128,
        num_groups: int = 64,
        expert_intermediate_size: Optional[int] = None,
        adjugate_intermediate_size: int = 128,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
        load_balance_loss_coef: float = 0.01,
        auxiliary_loss_coef: float = 0.001,
        scaling_factor: float = 0.05,
        dropout: float = 0.0,
        activation: str = 'swiglu',
        # Dual-bank configuration (for diffusion timestep routing)
        use_dual_bank: bool = False,
        low_noise_experts_num: Optional[int] = None,
        high_noise_experts_num: Optional[int] = None,
        share_low_with_high: bool = False,
    ):
        super().__init__()
        
        # Validate configuration
        assert num_experts % num_groups == 0, \
            f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})"
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.experts_per_group = num_experts // num_groups
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.load_balance_loss_coef = load_balance_loss_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef
        self.dropout = dropout
        
        # Scaling factor for adjugate experts (λ in the paper)
        # Restricted to be ≤ g/n to maintain routing weight balance
        max_scaling = num_groups / num_experts
        self.scaling_factor = min(scaling_factor, max_scaling)
        
        # Default expert intermediate size
        if expert_intermediate_size is None:
            expert_intermediate_size = int(hidden_size * 8 / 3)
        
        # Router with improved gating
        self.router = nn.Sequential(
            BlackwellOptimizedLinear(hidden_size + 6, hidden_size // 2, bias=True),
            nn.GELU(),
            BlackwellOptimizedLinear(hidden_size // 2, num_experts, bias=False)
        )
        
        # Sigmoid router for auxiliary loss (decoupled approach)
        self.sigmoid_router = BlackwellOptimizedLinear(
            hidden_size, num_experts, bias=False
        )
        
        # Create main experts
        self.experts = nn.ModuleList([
            self._create_expert(hidden_size, expert_intermediate_size, activation)
            for _ in range(num_experts)
        ])
        
        # Create adjugate experts (one per group)
        self.adjugate_experts = nn.ModuleList([
            AdjugateExpert(
                hidden_size,
                adjugate_intermediate_size,
                activation,
                dropout
            )
            for _ in range(num_groups)
        ])
        
        # Dual-bank configuration for diffusion models
        self.use_dual_bank = use_dual_bank
        if use_dual_bank:
            self.num_experts_high = high_noise_experts_num or num_experts
            self.num_experts_low = low_noise_experts_num or num_experts
            
            if not share_low_with_high:
                # Create separate low-noise experts
                self.low_noise_experts = nn.ModuleList([
                    self._create_expert(hidden_size, expert_intermediate_size, activation)
                    for _ in range(self.num_experts_low)
                ])
                # Create separate low-noise adjugate experts
                num_low_groups = min(num_groups, self.num_experts_low)
                self.low_noise_adjugate_experts = nn.ModuleList([
                    AdjugateExpert(
                        hidden_size,
                        adjugate_intermediate_size,
                        activation,
                        dropout
                    )
                    for _ in range(num_low_groups)
                ])
            else:
                # Share experts between banks
                self.low_noise_experts = self.experts
                self.low_noise_adjugate_experts = self.adjugate_experts
        
        # Global scheduler for temperature annealing
        self.scheduler = get_global_scheduler()
        
        # Statistics tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('group_counts', torch.zeros(num_groups))
    
    def _create_expert(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'swiglu'
    ) -> nn.Module:
        """Create a single expert module."""
        
        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                if activation == 'swiglu':
                    self.up_proj = BlackwellOptimizedLinear(
                        hidden_size, intermediate_size * 2, bias=False
                    )
                    self.act_fn = None
                else:
                    self.up_proj = BlackwellOptimizedLinear(
                        hidden_size, intermediate_size, bias=False
                    )
                    self.act_fn = nn.SiLU() if activation == 'silu' else nn.GELU()
                
                self.down_proj = BlackwellOptimizedLinear(
                    intermediate_size, hidden_size, bias=False
                )
                
                # Initialize with scaled weights for stability
                with torch.no_grad():
                    self.up_proj.weight.data.mul_(0.4)
                    self.down_proj.weight.data.mul_(0.4)
            
            def forward(self, x):
                up = self.up_proj(x)
                if activation == 'swiglu':
                    gate, up = up.chunk(2, dim=-1)
                    up = F.silu(gate) * up
                else:
                    up = self.act_fn(up)
                return self.down_proj(up)
        
        return Expert()
    
    def _compute_activation_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute statistical features for improved routing."""
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        std = centered.std(dim=-1, keepdim=True)
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        l2_norm = x.norm(dim=-1, keepdim=True)
        sparsity = (x.abs() < 1e-6).float().mean(dim=-1, keepdim=True)
        
        return torch.cat([mean, std, min_val, max_val, l2_norm, sparsity], dim=-1)
    
    def _get_expert_group(self, expert_idx: int) -> int:
        """Get the group index for a given expert."""
        return expert_idx // self.experts_per_group
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        snr_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Grove MoE.
        
        Args:
            x: Input tensor (B, N, C) or (B*N, C)
            t: Optional timestep tensor for dual-bank routing (B,)
            snr_threshold: SNR threshold for bank selection
        
        Returns:
            output: Processed tensor
            aux_loss: Auxiliary losses for load balancing
        """
        # Reshape if needed
        original_shape = x.shape
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(-1, C)
        else:
            B, N, C = 1, x.shape[0], x.shape[1]
        
        T = x.size(0)  # Total tokens
        
        # Compute routing statistics
        stats = self._compute_activation_stats(x)
        
        # Main routing (Softmax)
        router_input = torch.cat([x, stats], dim=-1)
        gate_logits = self.router(router_input)
        
        # Temperature annealing
        temperature = max(self.scheduler.get_value("moe_temperature"), 0.01)
        gate_logits = gate_logits / temperature
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(gate_logits).uniform_(
                -self.jitter_noise, self.jitter_noise
            )
            gate_logits = gate_logits + noise
        
        # Softmax routing probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Select expert bank based on timestep (if dual-bank)
        if self.use_dual_bank and t is not None:
            t_mean = t.mean()
            use_low_bank = (t_mean >= snr_threshold)
            if use_low_bank and hasattr(self, 'low_noise_experts'):
                experts = self.low_noise_experts
                adjugate_experts = self.low_noise_adjugate_experts
                num_experts_actual = len(experts)
                num_groups_actual = len(adjugate_experts)
            else:
                experts = self.experts
                adjugate_experts = self.adjugate_experts
                num_experts_actual = len(experts)
                num_groups_actual = len(adjugate_experts)
        else:
            experts = self.experts
            adjugate_experts = self.adjugate_experts
            num_experts_actual = self.num_experts
            num_groups_actual = self.num_groups
        
        # Capacity constraint
        capacity = int(self.capacity_factor * T / num_experts_actual)
        
        # Process tokens through experts
        output = torch.zeros_like(x)
        
        # Track which groups are activated
        activated_groups = set()
        group_weights = {}  # Track cumulative weights per group
        
        # Process each expert
        for expert_idx in range(num_experts_actual):
            # Find tokens routed to this expert
            expert_mask = (top_k_idx == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Apply capacity constraint
            if len(token_indices) > capacity:
                # Keep top tokens by routing probability
                token_probs = gate_probs[token_indices, expert_idx]
                _, keep_indices = torch.topk(token_probs, capacity)
                token_indices = token_indices[keep_indices]
            
            if len(token_indices) == 0:
                continue
            
            # Get routing weights for these tokens
            weights = torch.zeros(len(token_indices), device=x.device)
            for i, token_idx in enumerate(token_indices):
                # Find position in top_k
                k_pos = (top_k_idx[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                if len(k_pos) > 0:
                    weights[i] = top_k_probs[token_idx, k_pos[0]]
            
            # Process through main expert
            expert_input = x[token_indices]
            expert_output = experts[expert_idx](expert_input)
            output[token_indices] += expert_output * weights.unsqueeze(-1)
            
            # Track group activation
            if num_groups_actual > 0:
                group_idx = expert_idx % num_groups_actual
                activated_groups.add(group_idx)
                if group_idx not in group_weights:
                    group_weights[group_idx] = []
                group_weights[group_idx].append((token_indices, weights))
        
        # Process adjugate experts (one computation per activated group)
        for group_idx in activated_groups:
            if group_idx >= len(adjugate_experts):
                continue
                
            # Collect all tokens for this group
            all_tokens = []
            all_weights = []
            for token_indices, weights in group_weights[group_idx]:
                all_tokens.extend(token_indices.tolist())
                all_weights.extend(weights.tolist())
            
            if not all_tokens:
                continue
            
            # Get unique tokens (a token might be routed to multiple experts in same group)
            unique_tokens = list(set(all_tokens))
            token_to_weight = {}
            for tok, w in zip(all_tokens, all_weights):
                if tok not in token_to_weight:
                    token_to_weight[tok] = 0
                token_to_weight[tok] += w
            
            # Process through adjugate expert
            token_indices = torch.tensor(unique_tokens, device=x.device)
            cumulative_weights = torch.tensor(
                [token_to_weight[tok] for tok in unique_tokens],
                device=x.device
            )
            
            adjugate_input = x[token_indices]
            adjugate_output = adjugate_experts[group_idx](adjugate_input)
            
            # Apply scaling factor and accumulated weights
            scaled_output = adjugate_output * (cumulative_weights * self.scaling_factor).unsqueeze(-1)
            output[token_indices] += scaled_output
        
        # Compute auxiliary losses
        aux_loss = torch.tensor(0.0, device=x.device)
        
        if self.training:
            # Load balancing loss (Shazeer et al.)
            density_1 = gate_probs.mean(0)  # How often each expert is in top-1
            
            # Density after top-k selection
            chosen_counts = torch.bincount(
                top_k_idx.flatten(),
                minlength=num_experts_actual
            ).float()
            density_2 = chosen_counts / chosen_counts.sum().clamp_min(1e-8)
            
            # Standard load balance loss
            load_balance_loss = (density_1 * density_2).sum() * \
                               num_experts_actual * self.load_balance_loss_coef
            
            # Sigmoid auxiliary loss (from the paper)
            sigmoid_logits = self.sigmoid_router(x)
            sigmoid_probs = torch.sigmoid(sigmoid_logits)
            sigmoid_density = sigmoid_probs.mean(0)
            
            # Decoupled auxiliary loss
            aux_loss_sigmoid = (sigmoid_density * density_2).sum() * \
                              num_experts_actual * self.auxiliary_loss_coef
            
            aux_loss = load_balance_loss + aux_loss_sigmoid
            
            # Update statistics
            with torch.no_grad():
                self.expert_counts += chosen_counts.cpu()
                for group_idx in activated_groups:
                    self.group_counts[group_idx] += 1
        
        # Reshape output back
        if len(original_shape) == 3:
            output = output.view(original_shape)
        
        return output, aux_loss
    
    def get_activation_stats(self) -> Dict[str, float]:
        """Get statistics about expert and group activation."""
        total_counts = self.expert_counts.sum().item()
        if total_counts == 0:
            return {}
        
        expert_freq = self.expert_counts / total_counts
        group_freq = self.group_counts / self.group_counts.sum().clamp_min(1)
        
        # Calculate dynamic activation range
        avg_groups_activated = (self.group_counts > 0).sum().item()
        min_activated_params = avg_groups_activated * self.adjugate_experts[0].intermediate_size
        max_activated_params = self.top_k * len(self.experts[0].up_proj.weight)
        
        return {
            'expert_entropy': -(expert_freq * expert_freq.log()).sum().item(),
            'group_entropy': -(group_freq * group_freq.log()).sum().item(),
            'avg_groups_activated': avg_groups_activated,
            'min_activated_params': min_activated_params,
            'max_activated_params': max_activated_params,
            'expert_utilization': (self.expert_counts > 0).float().mean().item(),
            'group_utilization': (self.group_counts > 0).float().mean().item(),
        }


# Wrapper to make it compatible with existing code
class GroveMoEWrapper(GroveMoE):
    """
    Wrapper class to make GroveMoE compatible with existing DeltaNetEnhancedMoE interface.
    """
    
    def __init__(self, hidden_size: int, **kwargs):
        # Extract relevant parameters
        num_experts = kwargs.pop('num_experts', 128)
        
        # Map parameters to GroveMoE
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_groups=kwargs.pop('num_groups', min(64, num_experts // 2)),
            expert_intermediate_size=None,  # Use default
            adjugate_intermediate_size=kwargs.pop('adjugate_intermediate_size', 128),
            top_k=kwargs.pop('top_k', 2),
            capacity_factor=kwargs.pop('capacity_factor', 1.25),
            jitter_noise=kwargs.pop('jitter_noise', 0.01),
            load_balance_loss_coef=kwargs.pop('load_balance_loss_coef', 0.01),
            auxiliary_loss_coef=kwargs.pop('auxiliary_loss_coef', 0.001),
            scaling_factor=kwargs.pop('scaling_factor', 0.05),
            dropout=kwargs.pop('dropout', 0.0),
            activation=kwargs.pop('activation', 'swiglu'),
            use_dual_bank=True,  # Enable for diffusion models
            low_noise_experts_num=kwargs.pop('low_noise_experts_num', None),
            high_noise_experts_num=kwargs.pop('high_noise_experts_num', None),
            share_low_with_high=kwargs.pop('share_low_with_high', False),
        )


def test_grove_moe():
    """Test Grove MoE implementation."""
    import time
    
    # Configuration
    batch_size = 2
    seq_len = 256
    hidden_size = 1024
    num_experts = 128
    num_groups = 64
    
    # Create model
    model = GroveMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_groups=num_groups,
        adjugate_intermediate_size=128,
        top_k=2,
        scaling_factor=0.05,
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    t = torch.rand(batch_size)  # Timesteps for dual-bank
    
    # Forward pass
    start_time = time.time()
    output, aux_loss = model(x, t)
    forward_time = time.time() - start_time
    
    # Check output
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert aux_loss.numel() == 1, f"Aux loss should be scalar: {aux_loss.shape}"
    
    # Get statistics
    stats = model.get_activation_stats()
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for p in model.experts.parameters())
    adjugate_params = sum(p.numel() for p in model.adjugate_experts.parameters())
    
    print("Grove MoE Test Results:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Forward time: {forward_time:.3f}s")
    print(f"  Auxiliary loss: {aux_loss.item():.6f}")
    print(f"\nParameter counts:")
    print(f"  Total params: {total_params:,}")
    print(f"  Main expert params: {expert_params:,}")
    print(f"  Adjugate expert params: {adjugate_params:,}")
    print(f"  Adjugate/Main ratio: {adjugate_params/expert_params:.2%}")
    print(f"\nActivation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print("\n  Test passed!")
    
    return model


if __name__ == "__main__":
    test_grove_moe()
