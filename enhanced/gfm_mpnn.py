"""
MPNN-based Graph Flow Matching correction module.
Implements message passing neural network for velocity field correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from enhanced.gfm_base import GraphCorrectionHead


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter add operation for aggregating messages.
    This is a simplified version - for production, use torch_scatter.scatter_add
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    # Use index_add_ for the scatter operation
    out.index_add_(dim, index, src)
    return out


class MPNNLayer(nn.Module):
    """
    Single MPNN layer with edge-weighted message passing.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, use_edge_features: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features
        
        # Message function: combines node features (and optionally edge weights)
        if use_edge_features:
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for edge weight
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Update function: combines aggregated messages with node features
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of MPNN layer.
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            num_nodes: Number of nodes (inferred if None)
        
        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        if num_nodes is None:
            num_nodes = x.size(0)
        
        # Get source and target node indices
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Get source and target node features
        src_features = x[src_idx]  # (num_edges, hidden_dim)
        dst_features = x[dst_idx]  # (num_edges, hidden_dim)
        
        # Compute messages
        if self.use_edge_features:
            # Concatenate source, destination, and edge weight
            edge_weight_expanded = edge_weight.unsqueeze(-1)  # (num_edges, 1)
            message_input = torch.cat([src_features, dst_features, edge_weight_expanded], dim=-1)
        else:
            # Just concatenate source and destination
            message_input = torch.cat([src_features, dst_features], dim=-1)
        
        messages = self.message_mlp(message_input)  # (num_edges, hidden_dim)
        
        # Weight messages by edge weights
        messages = messages * edge_weight.unsqueeze(-1)
        
        # Aggregate messages (sum aggregation)
        aggregated = scatter_add(messages, dst_idx, dim=0, dim_size=num_nodes)
        
        # Update node features
        update_input = torch.cat([x, aggregated], dim=-1)
        x_new = self.update_mlp(update_input)
        
        # Residual connection
        x_new = x + self.dropout(x_new)
        
        return x_new


class MPNNGraphCorrection(GraphCorrectionHead):
    """
    MPNN-based graph correction module for Graph Flow Matching.
    
    Uses multiple layers of message passing to propagate information
    between neighboring patches based on attention-derived graphs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        in_channels: int,
        gfm_hidden_ratio: float = 0.05,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        enforce_param_budget: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the main model
            patch_size: Patch size for unpatchify operation
            in_channels: Number of input channels
            gfm_hidden_ratio: Ratio to compute GFM hidden dimension
            num_layers: Number of MPNN layers
            dropout: Dropout rate
            use_edge_features: Whether to use edge weights as features
            enforce_param_budget: Whether to enforce 5% parameter budget
        """
        super().__init__(
            hidden_size=hidden_size,
            patch_size=patch_size,
            in_channels=in_channels,
            gfm_hidden_ratio=gfm_hidden_ratio,
            num_layers=num_layers,
            dropout=dropout,
            enforce_param_budget=enforce_param_budget,
        )
        
        self.use_edge_features = use_edge_features
        
        # Create MPNN layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                hidden_dim=self.gfm_hidden,
                dropout=dropout,
                use_edge_features=use_edge_features,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization after each MPNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.gfm_hidden)
            for _ in range(num_layers)
        ])
    
    def process_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Process node features through MPNN layers.
        
        Args:
            x: Node features (B*N, gfm_hidden)
            edge_index: Graph connectivity (2, E)
            edge_weight: Edge weights (E,)
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
        
        Returns:
            Updated node features (B*N, gfm_hidden)
        """
        # Process through MPNN layers
        for i, (mpnn_layer, norm) in enumerate(zip(self.mpnn_layers, self.layer_norms)):
            # Message passing
            x_new = mpnn_layer(x, edge_index, edge_weight, num_nodes=batch_size * num_nodes)
            
            # Layer normalization
            x_new = norm(x_new)
            
            # Skip connection (except for first layer)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        return x


def test_mpnn_correction():
    """Test MPNN correction module with random inputs."""
    import time
    
    # Setup
    batch_size = 2
    num_patches = 256  # 16x16 patches
    hidden_size = 1152
    patch_size = 2
    in_channels = 4
    
    # Create model
    model = MPNNGraphCorrection(
        hidden_size=hidden_size,
        patch_size=patch_size,
        in_channels=in_channels,
        gfm_hidden_ratio=0.05,
        num_layers=3,
    )
    
    # Create random inputs
    x_patches = torch.randn(batch_size, num_patches, hidden_size)
    t = torch.rand(batch_size)
    
    # Create random graph (simplified)
    num_edges_per_node = 8
    edge_list = []
    weight_list = []
    
    for b in range(batch_size):
        for i in range(num_patches):
            # Random neighbors
            neighbors = torch.randperm(num_patches)[:num_edges_per_node]
            for j in neighbors:
                src = i + b * num_patches
                dst = j.item() + b * num_patches
                edge_list.append([src, dst])
                weight_list.append(torch.rand(1).item())
    
    edge_index = torch.tensor(edge_list).t()
    edge_weight = torch.tensor(weight_list)
    
    # Forward pass
    start_time = time.time()
    v_diff = model(x_patches, edge_index, edge_weight, t)
    forward_time = time.time() - start_time
    
    # Check output shape
    expected_shape = (batch_size, num_patches, patch_size * patch_size * in_channels)
    assert v_diff.shape == expected_shape, f"Shape mismatch: {v_diff.shape} != {expected_shape}"
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    base_model_params = 100_000_000  # Assume 100M base model
    model.set_base_model_params(base_model_params)
    param_report = model.get_param_report()
    
    print("MPNN Correction Module Test:")
    print(f"  Output shape: {v_diff.shape}")
    print(f"  Forward time: {forward_time:.3f}s")
    print(f"  Total params: {total_params:,}")
    print(f"  Param ratio: {param_report['param_ratio']*100:.2f}%")
    print(f"  GFM hidden dim: {param_report['gfm_hidden_dim']}")
    print("  Test passed!")
    
    return model


if __name__ == "__main__":
    test_mpnn_correction()
