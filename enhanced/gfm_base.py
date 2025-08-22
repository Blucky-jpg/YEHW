"""
Base class for Graph Flow Matching (GFM) correction modules.
Provides common functionality for MPNN and GPS variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import warnings


class GraphCorrectionHead(nn.Module):
    """
    Abstract base class for graph-based velocity correction in Flow Matching.
    
    The correction module takes the base velocity field v_react and adds a
    lightweight graph-based correction v_diff to improve local coherence:
    v_final = v_react + v_diff
    
    This follows the Graph Flow Matching paper's decomposition.
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        in_channels: int,
        gfm_hidden_ratio: float = 0.05,
        num_layers: int = 3,
        dropout: float = 0.1,
        enforce_param_budget: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the main model
            patch_size: Patch size for unpatchify operation
            in_channels: Number of input channels (typically 4 for latent)
            gfm_hidden_ratio: Ratio to compute GFM hidden dimension
            num_layers: Number of graph processing layers
            dropout: Dropout rate
            enforce_param_budget: Whether to enforce 5% parameter budget
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Compute GFM hidden dimension based on ratio
        self.gfm_hidden = max(32, int(hidden_size * gfm_hidden_ratio))
        
        # Input projection: project patches to GFM hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, self.gfm_hidden),
            nn.LayerNorm(self.gfm_hidden),
            nn.SiLU(),
        )
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.gfm_hidden),
            nn.SiLU(),
            nn.Linear(self.gfm_hidden, self.gfm_hidden),
        )
        
        # Output projection: project back to velocity space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.gfm_hidden),
            nn.Linear(self.gfm_hidden, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels),
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Parameter budget tracking
        self.enforce_param_budget = enforce_param_budget
        self._base_model_params = None
        
    def set_base_model_params(self, base_params: int):
        """Set the base model parameter count for budget enforcement."""
        self._base_model_params = base_params
        if self.enforce_param_budget:
            self._check_parameter_budget()
    
    def _check_parameter_budget(self):
        """Check if GFM module exceeds 5% parameter budget."""
        if self._base_model_params is None:
            return
        
        gfm_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ratio = gfm_params / self._base_model_params
        
        if ratio > 0.05:
            warnings.warn(
                f"GFM module uses {ratio*100:.2f}% of base model parameters "
                f"(exceeds 5% budget). Consider reducing gfm_hidden_ratio. "
                f"GFM params: {gfm_params:,}, Base params: {self._base_model_params:,}"
            )
    
    def get_param_report(self) -> Dict[str, int]:
        """Get parameter count report."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        report = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'gfm_hidden_dim': self.gfm_hidden,
        }
        
        if self._base_model_params is not None:
            report['param_ratio'] = trainable_params / self._base_model_params
            report['base_model_params'] = self._base_model_params
        
        return report
    
    def process_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Process node features through the graph network.
        To be implemented by subclasses (MPNN or GPS).
        
        Args:
            x: Node features (B*N, D)
            edge_index: Graph connectivity (2, E)
            edge_weight: Edge weights (E,)
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
        
        Returns:
            Updated node features (B*N, D)
        """
        raise NotImplementedError("Subclasses must implement process_graph")
    
    def forward(
        self,
        x_patches: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        t: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the graph correction module.
        
        Args:
            x_patches: Patch embeddings (B, N, hidden_size) 
            edge_index: Graph connectivity (2, E)
            edge_weight: Edge weights (E,)
            t: Time values (B,)
            batch_size: Batch size (inferred if None)
        
        Returns:
            Velocity correction v_diff (B, N, patch_size^2 * in_channels)
        """
        B, N, D = x_patches.shape
        if batch_size is None:
            batch_size = B
        
        # Project patches to GFM hidden dimension
        x = self.input_proj(x_patches)  # (B, N, gfm_hidden)
        
        # Add time information
        t_emb = self.time_proj(t.unsqueeze(-1))  # (B, gfm_hidden)
        x = x + t_emb.unsqueeze(1)  # Broadcast time to all patches
        
        # Flatten for graph processing
        x_flat = x.view(B * N, -1)  # (B*N, gfm_hidden)
        
        # Process through graph network (implemented by subclasses)
        x_graph = self.process_graph(
            x_flat, edge_index, edge_weight, 
            batch_size=B, num_nodes=N
        )
        
        # Residual connection
        x_flat = x_flat + self.dropout(x_graph)
        
        # Reshape back to patches
        x = x_flat.view(B, N, -1)  # (B, N, gfm_hidden)
        
        # Project to velocity correction
        v_diff = self.output_proj(x)  # (B, N, patch_size^2 * in_channels)
        
        return v_diff
    
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert patch embeddings back to image format.
        
        Args:
            x: Patch embeddings (B, N, P*P*C)
            H: Height in patches
            W: Width in patches
        
        Returns:
            Image tensor (B, C, H*P, W*P)
        """
        B, N, _ = x.shape
        P = self.patch_size
        C = self.in_channels
        
        assert N == H * W, f"Number of patches {N} != H*W {H*W}"
        
        x = x.reshape(B, H, W, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, H, P, W, P)
        x = x.reshape(B, C, H * P, W * P)
        
        return x


class IdentityGraphCorrection(GraphCorrectionHead):
    """
    Identity correction module for ablation studies.
    Returns zero correction to isolate the effect of graph processing.
    """
    
    def process_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return zeros for ablation."""
        return torch.zeros_like(x)
