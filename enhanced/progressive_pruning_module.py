import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ..core.quantization import QuantizedLinear
from .global_scheduler import get_global_scheduler, register_default_schedules

class ProgressivePruningSystem(nn.Module):
    """
    Enhanced progressive pruning system with:
    - Global scheduling coordination
    - Minimum active paths constraint (top-K guarantee)
    - Soft-optional pruning with Gumbel noise for differentiable pruning
    - Improved numerical stability
    """
    def __init__(self, hidden_size: int, num_heads: int, num_paths: int = 5,
                 min_active_paths: int = 2, soft_pruning: bool = True, 
                 gumbel_temperature: float = 1.0, use_global_scheduler: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.min_active_paths = max(1, min(min_active_paths, num_paths))
        self.soft_pruning = soft_pruning
        self.gumbel_temperature = gumbel_temperature
        self.use_global_scheduler = use_global_scheduler
        
        # Constants for better performance
        self.soft_contribution = 0.3
        
        # Get global scheduler if enabled
        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()
        else:
            # Fallback to local scheduling
            self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        
        # Learnable temperature per head
        self.gate_log_temp = nn.Parameter(torch.zeros(num_heads))
        
        # Routing gate
        self.routing_gate = nn.Sequential(
            QuantizedLinear(hidden_size, hidden_size * 2, bias=True),
            nn.GELU(),
            QuantizedLinear(hidden_size * 2, num_heads * num_paths, bias=True)
        )
        
        # Initialize with bias toward value + identity paths
        self._initialize_routing_bias()
        
        # Token-adaptive floor module
        self.adaptive_floor = TokenAdaptiveFloor(
            num_heads=num_heads, 
            num_paths=num_paths, 
            use_global_scheduler=use_global_scheduler
        )
        
        # Cache for frequently accessed values
        self._cache = {
            'prune_threshold': None,
            'entropy_coeff': None,
            'step': -1
        }

    def _initialize_routing_bias(self) -> None:
        """Initialize routing gate bias - separated for clarity."""
        with torch.no_grad():
            bias = self.routing_gate[-1].bias
            bias.zero_()
            bias_matrix = bias.view(self.num_heads, self.num_paths)
            bias_matrix[:, 3] = 1.0  # direct value path
            bias_matrix[:, 4] = 2.0  # identity path (strongest initial bias)

    def _get_current_step(self) -> int:
        """Get current training step from global or local scheduler."""
        if self.use_global_scheduler:
            return self.scheduler.get_step()
        else:
            return int(self._step.item())

    def _update_cache(self) -> None:
        """Update cached values if step has changed."""
        current_step = self._get_current_step()
        if self._cache['step'] != current_step:
            self._cache['step'] = current_step
            
            if self.use_global_scheduler:
                self._cache['prune_threshold'] = self.scheduler.get_value("prune_threshold")
                self._cache['entropy_coeff'] = self.scheduler.get_value("entropy_coeff")
            else:
                # Fallback calculations
                t = float(current_step)
                
                # Prune threshold calculation
                if t <= 2000:
                    self._cache['prune_threshold'] = 0.0
                elif t >= 4000:
                    self._cache['prune_threshold'] = 1e-3
                else:
                    frac = (t - 2000) / 2000
                    self._cache['prune_threshold'] = frac * 1e-3
                
                # Entropy coefficient calculation
                if t >= 4000:
                    self._cache['entropy_coeff'] = 0.0
                else:
                    ratio = t / 4000
                    self._cache['entropy_coeff'] = 0.02 + ratio * (0.0 - 0.02)

    @torch.compile(dynamic=True)
    def _apply_gumbel_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Gumbel noise for differentiable sampling."""
        if not self.training or not self.soft_pruning:
            return logits
        
        # Optimized Gumbel noise generation
        u = torch.rand_like(logits)
        u = u.clamp_(min=torch.finfo(logits.dtype).eps, max=1.0 - torch.finfo(logits.dtype).eps)
        gumbel_noise = -torch.log(-torch.log(u))
        return logits.add_(gumbel_noise, alpha=self.gumbel_temperature)

    @torch.compile(dynamic=True)
    def _ensure_minimum_active_paths(self, probs: torch.Tensor, 
                                     original_probs: torch.Tensor) -> torch.Tensor:
        """Ensure minimum number of active paths using top-K constraint."""
        # Count active paths more efficiently
        active_counts = (probs > 1e-6).sum(dim=-1)  # (B, L, H)
        
        # Find positions where we have fewer than minimum active paths
        insufficient_mask = active_counts < self.min_active_paths  # (B, L, H)
        
        if not insufficient_mask.any():
            return probs
        
        # For insufficient positions, use top-K from original probabilities
        _, top_k_indices = torch.topk(original_probs, self.min_active_paths, dim=-1)
        
        # Create one-hot encoding for top-K paths more efficiently
        top_k_mask = torch.zeros_like(probs)
        top_k_mask.scatter_(-1, top_k_indices, 1.0)
        
        # For insufficient positions, replace with uniform over top-K
        uniform_top_k = top_k_mask / self.min_active_paths
        
        # Apply correction using broadcasting
        insufficient_expanded = insufficient_mask.unsqueeze(-1)
        probs = torch.where(insufficient_expanded, uniform_top_k, probs)
        
        return probs

    def _apply_enhanced_pruning(self, probs: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply enhanced progressive pruning with soft/hard options."""
        self._update_cache()  # Update cached values
        
        thresh = self._cache['prune_threshold']
        entropy_coeff = self._cache['entropy_coeff']
        
        entropy_loss = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        
        # Early exit if no pruning needed
        if thresh <= 0.0:
            # Still compute entropy loss if needed
            if entropy_coeff > 0.0 and self.training:
                probs_safe = probs.clamp(min=1e-9)
                entropy = -(probs_safe * probs_safe.log()).sum(dim=-1).mean()
                entropy_loss = -entropy_coeff * entropy
            return probs, entropy_loss
        
        original_probs = probs.detach()  # Avoid unnecessary clone
        
        if self.soft_pruning and self.training:
            # Soft pruning with Gumbel noise
            noisy_logits = self._apply_gumbel_noise(logits)
            soft_probs = F.softmax(noisy_logits / self.gumbel_temperature, dim=-1)
            
            # Mix hard and soft pruning
            hard_mask = probs > thresh
            
            probs_mixed = (1 - self.soft_contribution) * (probs * hard_mask.float()) + \
                          self.soft_contribution * soft_probs
            
            # Renormalize
            probs = probs_mixed / probs_mixed.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            # Hard pruning: set probabilities below threshold to zero
            mask = probs > thresh
            probs_pruned = probs * mask.float()
            
            # Renormalize to maintain probability simplex
            probs = probs_pruned / probs_pruned.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        
        # Ensure minimum active paths constraint
        probs = self._ensure_minimum_active_paths(probs, original_probs)
        
        # Final renormalization and clamping
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        probs.clamp_(min=1e-9, max=1.0)
        
        # Entropy regularization
        if entropy_coeff > 0.0 and self.training:
            probs_safe = probs.clamp(min=1e-9)
            entropy = -(probs_safe * probs_safe.log()).sum(dim=-1).mean()
            entropy_loss = -entropy_coeff * entropy  # Maximize entropy
        
        return probs, entropy_loss

    @torch.compile(dynamic=True)
    def _combine_paths_efficiently(self, path_outputs: List[torch.Tensor], 
                                   gate_weights: torch.Tensor, output_shape: Tuple[int, ...]) -> torch.Tensor:
        """Memory-efficient path combination using fused operations."""
        actual_paths = gate_weights.shape[-1]
        
        if actual_paths == 1:
            return path_outputs[0] * gate_weights[..., 0].unsqueeze(-1)
        
        # Use torch.stack when memory allows, accumulation otherwise
        if actual_paths <= 3:  # Stack for small number of paths
            path_stack = torch.stack(path_outputs[:actual_paths], dim=-1)
            gate_expanded = gate_weights.unsqueeze(-2)
            return (path_stack * gate_expanded).sum(dim=-1)
        else:
            # Accumulation for larger number of paths to save memory
            output = torch.zeros(*output_shape, device=gate_weights.device, dtype=gate_weights.dtype)
            for p in range(actual_paths):
                weighted = path_outputs[p] * gate_weights[..., p].unsqueeze(-1)
                output.add_(weighted)
            return output

    def forward(self, hidden_states: torch.Tensor, path_outputs: List[torch.Tensor], 
                profile: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply enhanced progressive pruning to path routing.
        Args:
            hidden_states: (B, L, D) input features
            path_outputs: List of (B, L, H, D) path outputs
            profile: If True, enable profiling for this forward pass
        Returns:
            output: (B, L, H, D) combined output
            entropy_loss: Scalar entropy regularization loss
        """
        B, L, D = hidden_states.shape
        actual_paths = min(len(path_outputs), self.num_paths)
        head_dim = D // self.num_heads
        
        # Early exits for edge cases
        zero_tensor = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        if actual_paths == 0:
            output = torch.zeros(B, L, self.num_heads, head_dim, 
                                 device=hidden_states.device, dtype=hidden_states.dtype)
            return output, zero_tensor
        
        # Compute routing logits
        routing_logits = self.routing_gate(hidden_states)  # (B, L, H*P)
        routing_logits = routing_logits.view(B, L, self.num_heads, self.num_paths)
        
        # Apply temperature scaling with better numerical stability
        temp = F.softplus(self.gate_log_temp).clamp(min=1e-4).view(1, 1, -1, 1)
        routing_logits = routing_logits / temp
        
        # Compute probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Apply enhanced progressive pruning
        routing_probs, entropy_loss = self._apply_enhanced_pruning(routing_probs, routing_logits)
        
        # Apply token-adaptive floor
        routing_probs = self.adaptive_floor(routing_probs)
        
        # Efficient path combination
        gate_weights = routing_probs[..., :actual_paths]  # (B, L, H, actual_paths)
        output = self._combine_paths_efficiently(
            path_outputs, gate_weights, (B, L, self.num_heads, head_dim)
        )
        
        # Increment step counter if using local scheduling
        if not self.use_global_scheduler and self.training:
            self._step += 1
        
        return output, entropy_loss


class TokenAdaptiveFloor(nn.Module):
    """
    Token-adaptive ε-floor used by DeltaNet-TAPR.
    Optimized version with better caching and numerical stability.
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
        self.floor_start = floor_start
        self.floor_end = floor_end
        self.floor_decay_steps = float(floor_decay_steps)  # Convert to float for division
        self.use_global_scheduler = use_global_scheduler

        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()

            # Make sure the schedule we need exists
            try:
                self.scheduler.get_value("token_floor")
            except (ValueError, AttributeError):
                register_default_schedules(self.scheduler)
        else:
            self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

        # Learnable ε base (logits) per head / path
        eps_logit_init = -12.0
        self.gate_eps_logit = nn.Parameter(
            torch.full((num_heads, num_paths), eps_logit_init)
        )

        # Improved caching
        self._floor_cache = {'value': None, 'step': -1}

    def _get_current_step(self) -> int:
        """Get current step efficiently."""
        if self.use_global_scheduler:
            return self.scheduler.get_step()
        return int(self._step.item())

    def _current_floor_max(self) -> float:
        """Get max ε allowed at the current step with improved caching."""
        step = self._get_current_step()
        
        # Check cache first
        if step == self._floor_cache['step'] and self._floor_cache['value'] is not None:
            return self._floor_cache['value']

        if self.use_global_scheduler:
            try:
                val = self.scheduler.get_value("token_floor")
            except (ValueError, AttributeError):
                val = 0.0
        else:
            # Optimized local linear decay
            if step >= self.floor_decay_steps:
                val = self.floor_end
            else:
                ratio = step / max(1.0, self.floor_decay_steps)
                val = self.floor_start + ratio * (self.floor_end - self.floor_start)

        # Update cache
        self._floor_cache['value'] = float(val)
        self._floor_cache['step'] = step
        return val

    @torch.compile(dynamic=True) 
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Apply the adaptive ε-floor with optimized operations.
        Args:
            probs: (B, L, H, P) probability tensor
        Returns:
            Modified probabilities with ε-floor applied
        """
        eps_max = self._current_floor_max()
        if eps_max <= 0.0:
            return probs

        # Compute uncertainty and eps more efficiently
        p_max = probs.max(dim=-1, keepdim=True).values
        uncertainty = 1.0 - p_max

        # Pre-computed eps_base view for better memory access
        eps_base = torch.sigmoid(self.gate_eps_logit).view(1, 1, self.num_heads, self.num_paths)
        eps = eps_max * uncertainty * eps_base

        # In-place operations for memory efficiency
        eps_sum = eps.sum(dim=-1, keepdim=True)
        probs.mul_(1.0 - eps_sum).add_(eps)
        probs.clamp_(min=1e-9, max=1.0)

        # Update step counter for local scheduler
        if not self.use_global_scheduler and self.training:
            self._step += 1

        return probs
