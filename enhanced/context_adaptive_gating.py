import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
from ..core.quantization import QuantizedLinear
from .global_scheduler import get_global_scheduler

class ContextAdaptiveGating(nn.Module):
    """
    Context-conditioned adaptive gating from DeltaNet-CAGF-DPAF-EASH.
    Uses statistical features and content-aware routing for better decisions.
    """
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, num_paths: int = 5,
                 temp_init: float = 1.0, fusion_hidden_mult: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_paths = num_paths
        
        # Per-head learnable temperature
        self.log_temp = nn.Parameter(torch.full((num_heads, 1), torch.log(torch.tensor(temp_init))))
        
        # Statistical feature dimensions
        self.stat_dim = 4  # mean, var, abs_mean, l2_norm
        
        # Gate input: hidden + per-head stats + per-branch norms + pairwise differences
        stats_dim = num_heads * (self.stat_dim * num_paths)
        norms_dim = num_heads * num_paths
        pairwise_dim = num_heads * (num_paths * (num_paths - 1) // 2)
        
        gate_input_dim = hidden_size + stats_dim + norms_dim + pairwise_dim
        gate_hidden_dim = max(1, int(hidden_size * fusion_hidden_mult) // 2)
        
        self.context_gate = nn.Sequential(
            QuantizedLinear(gate_input_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            QuantizedLinear(gate_hidden_dim, num_heads * num_paths, bias=True)
        )
        
        # Initialize biases to favor value + identity paths
        self._initialize_biases()
        
        # Pre-compute constants
        self.register_buffer('triu_indices', self._get_triu_indices(), persistent=False)
        
    def _initialize_biases(self):
        """Initialize biases to favor value + identity paths."""
        with torch.no_grad():
            bias = self.context_gate[-1].bias
            bias.zero_()
            if self.num_paths >= 5:  # Only if we have enough paths
                bias_matrix = bias.view(self.num_heads, self.num_paths)
                bias_matrix[:, 3] = 1.0  # direct value path
                bias_matrix[:, 4] = 2.0  # identity path

    def _get_triu_indices(self) -> Optional[torch.Tensor]:
        """Pre-compute upper triangular indices for pairwise differences."""
        if self.num_paths <= 1:
            return None
        indices = torch.triu_indices(self.num_paths, self.num_paths, offset=1)
        return indices  # (2, num_pairs)

    def _compute_pairwise_differences(self, path_outputs: list) -> torch.Tensor:
        """
        Compute pairwise L2 differences using optimized batch operations.
        Uses ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ to avoid memory-intensive operations.
        """
        actual_paths = min(len(path_outputs), self.num_paths)
        if actual_paths <= 1 or self.triu_indices is None:
            B, L, H = path_outputs[0].shape[:3] if path_outputs else (1, 1, self.num_heads)
            return torch.zeros(B, L, H, 0, device=path_outputs[0].device if path_outputs else torch.device('cpu'))
        
        # Stack and flatten: (B, L, H, D, P) -> (BLH, D, P)
        path_stack = torch.stack(path_outputs[:actual_paths], dim=-1)
        B, L, H, D, P = path_stack.shape
        flat = rearrange(path_stack, 'b l h d p -> (b l h) d p')
        
        # Efficient pairwise distance computation
        # ||a||² for each path: (BLH, P)
        norms_sq = torch.sum(flat * flat, dim=1)
        
        # Dot products: (BLH, P, P)
        dots = torch.bmm(flat.transpose(1, 2), flat)
        
        # ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
        norms_sq_expanded = norms_sq.unsqueeze(1) + norms_sq.unsqueeze(2) - 2 * dots
        distances = torch.sqrt(torch.clamp(norms_sq_expanded, min=0))
        
        # Extract upper triangular part
        if actual_paths <= self.num_paths:
            triu_idx = torch.triu_indices(P, P, offset=1, device=distances.device)
            pairwise_diffs = distances[:, triu_idx[0], triu_idx[1]]
        else:
            triu_idx = self.triu_indices.to(distances.device)
            pairwise_diffs = distances[:, triu_idx[0], triu_idx[1]]
        
        # Reshape: (BLH, num_pairs) -> (B, L, H, num_pairs)
        return rearrange(pairwise_diffs, '(b l h) p -> b l h p', b=B, l=L, h=H)

    def _compute_path_features(self, path_outputs: list, B: int, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute statistics and norms for all paths efficiently."""
        actual_paths = min(len(path_outputs), self.num_paths)
        
        if actual_paths == 0:
            empty_stats = torch.zeros(B, L, self.num_heads, 0, dtype=path_outputs[0].dtype, device=path_outputs[0].device)
            return empty_stats, empty_stats
        
        # Stack and compute all statistics in one go: (B, L, H, D, P)
        path_stack = torch.stack(path_outputs[:actual_paths], dim=-1)
        
        # Vectorized statistics computation: (B, L, H, 4, P)
        mean = path_stack.mean(dim=-2, keepdim=True)
        var = path_stack.var(dim=-2, unbiased=False, keepdim=True)
        abs_mean = path_stack.abs().mean(dim=-2, keepdim=True)
        l2_norm = path_stack.norm(dim=-2, keepdim=True)
        
        stats = torch.cat([mean, var, abs_mean, l2_norm], dim=-2)
        stats_flat = rearrange(stats, 'b l h s p -> b l h (p s)')
        
        # Path norms: (B, L, H, P)
        norms_flat = path_stack.norm(dim=-2)
        
        return stats_flat, norms_flat

    def _pad_features(self, stats_flat: torch.Tensor, norms_flat: torch.Tensor, 
                     pairwise_diffs: torch.Tensor, actual_paths: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad or truncate features to match expected dimensions."""
        B, L, H = stats_flat.shape[:3]
        device, dtype = stats_flat.device, stats_flat.dtype
        
        if actual_paths < self.num_paths:
            # Pad statistics
            pad_size = (self.num_paths - actual_paths) * self.stat_dim
            pad_stats = torch.zeros(B, L, H, pad_size, device=device, dtype=dtype)
            stats_flat = torch.cat([stats_flat, pad_stats], dim=-1)
            
            # Pad norms
            pad_norms = torch.zeros(B, L, H, self.num_paths - actual_paths, device=device, dtype=dtype)
            norms_flat = torch.cat([norms_flat, pad_norms], dim=-1)
            
            # Pad pairwise differences
            expected_pairs = self.num_paths * (self.num_paths - 1) // 2
            actual_pairs = pairwise_diffs.shape[-1]
            if actual_pairs < expected_pairs:
                pad_pairs = torch.zeros(B, L, H, expected_pairs - actual_pairs, device=device, dtype=dtype)
                pairwise_diffs = torch.cat([pairwise_diffs, pad_pairs], dim=-1)
        
        elif actual_paths > self.num_paths:
            # Truncate
            stats_flat = stats_flat[..., :self.num_paths * self.stat_dim]
            norms_flat = norms_flat[..., :self.num_paths]
            expected_pairs = self.num_paths * (self.num_paths - 1) // 2
            pairwise_diffs = pairwise_diffs[..., :expected_pairs]
        
        return stats_flat, norms_flat, pairwise_diffs

    def forward(self, hidden_states: torch.Tensor, path_outputs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply context-adaptive gating with optimized implementation."""
        B, L, D = hidden_states.shape
        actual_paths = min(len(path_outputs), self.num_paths)
        
        # Compute path features
        stats_flat, norms_flat = self._compute_path_features(path_outputs, B, L)
        
        # Compute pairwise differences
        pairwise_diffs = self._compute_pairwise_differences(path_outputs)
        
        # Pad/truncate features to match expected dimensions
        stats_flat, norms_flat, pairwise_diffs = self._pad_features(
            stats_flat, norms_flat, pairwise_diffs, actual_paths
        )
        
        # Flatten for gate input
        stats_flat = rearrange(stats_flat, 'b l h s -> b l (h s)')
        norms_flat = rearrange(norms_flat, 'b l h p -> b l (h p)')
        pairwise_flat = rearrange(pairwise_diffs, 'b l h p -> b l (h p)')
        
        # Construct gate input
        gate_input = torch.cat([hidden_states, stats_flat, norms_flat, pairwise_flat], dim=-1)
        
        # Compute gate logits with temperature scaling
        gate_logits = self.context_gate(gate_input).view(B, L, self.num_heads, self.num_paths)
        temp = torch.exp(self.log_temp).unsqueeze(0).unsqueeze(0)
        gate_logits = gate_logits / temp
        
        # Compute probabilities with epsilon floor
        gate_probs = torch.softmax(gate_logits, dim=-1)
        gate_probs = torch.clamp(gate_probs, min=0.02)
        gate_probs = gate_probs / gate_probs.sum(-1, keepdim=True)
        
        # Compute regularization loss
        reg_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(-1).mean()
            if torch.isfinite(entropy):
                reg_loss = 0.01 * entropy
        
        # Combine paths
        if actual_paths > 0:
            path_stack = torch.stack(path_outputs[:actual_paths], dim=-1)
            gate_weights = gate_probs[..., :actual_paths].unsqueeze(-2)
            output = (path_stack * gate_weights).sum(dim=-1)
        else:
            output = torch.zeros(B, L, self.num_heads, self.head_dim, 
                               device=hidden_states.device, dtype=hidden_states.dtype)
        
        return output, reg_loss


class DualPhasePathFloor(nn.Module):
    """Dual-phase path floor from DeltaNet-CAGF-DPAF-EASH."""
    
    def __init__(self, num_heads: int, num_paths: int, epsilon_init: float = 0.10,
                 epsilon_final: float = 0.025, epsilon_decay_steps: int = 4000,
                 use_global_scheduler: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.use_global_scheduler = use_global_scheduler
        
        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()
        else:
            self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

    def _dual_phase_epsilon(self) -> float:
        """Compute current epsilon with dual-phase schedule."""
        if self.use_global_scheduler:
            return self.scheduler.get_value("epsilon_floor")
        
        step = float(self._step.item())
        if step >= self.epsilon_decay_steps:
            return self.epsilon_final
        
        ratio = step / max(1.0, self.epsilon_decay_steps)
        return self.epsilon_init + (self.epsilon_final - self.epsilon_init) * ratio

    @torch.compile(dynamic=True)
    def forward(self, probs: torch.Tensor, local_path_idx: int = 0) -> torch.Tensor:
        """Apply dual-phase floor to ensure minimum allocation to local path."""
        eps = self._dual_phase_epsilon()
        
        if eps > 0.0:
            # Apply floor to local path and renormalize
            probs[..., local_path_idx].clamp_(min=eps)
            probs.div_(probs.sum(dim=-1, keepdim=True).clamp_(min=1e-8))
        
        if not self.use_global_scheduler and self.training:
            self._step += 1
        
        return probs


class EntropyAnnealedRegularization(nn.Module):
    """Entropy-annealed gate regularization from DeltaNet-CAGF-DPAF-EASH."""
    
    def __init__(self, entropy_reg_init: float = 0.02, entropy_reg_final: float = 0.001,
                 entropy_decay_steps: int = 12000, use_global_scheduler: bool = True):
        super().__init__()
        self.entropy_reg_init = entropy_reg_init
        self.entropy_reg_final = entropy_reg_final
        self.entropy_decay_steps = entropy_decay_steps
        self.use_global_scheduler = use_global_scheduler
        
        if self.use_global_scheduler:
            self.scheduler = get_global_scheduler()
        else:
            self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

    def _entropy_lambda(self) -> float:
        """Compute current entropy regularization coefficient."""
        if self.use_global_scheduler:
            return self.scheduler.get_value("entropy_reg")
        
        step = float(self._step.item())
        if step >= self.entropy_decay_steps:
            return self.entropy_reg_final
        
        ratio = step / max(1.0, self.entropy_decay_steps)
        return self.entropy_reg_init + (self.entropy_reg_final - self.entropy_reg_init) * ratio

    @torch.compile(dynamic=True)
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss."""
        entropy_coeff = self._entropy_lambda()
        
        entropy_loss = torch.tensor(0.0, device=probs.device)
        if entropy_coeff > 0.0 and self.training:
            entropy = -(probs.clamp(min=1e-8) * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()
            entropy_loss = -entropy_coeff * entropy
        
        if not self.use_global_scheduler and self.training:
            self._step += 1
            
        return entropy_loss
