"""
GPS-based Graph Flow Matching correction module.
Implements General, Powerful, Scalable graph transformer for velocity field correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from enhanced.gfm_base import GraphCorrectionHead


class SimplifiedGINEConv(nn.Module):
    """
    Simplified Graph Isomorphism Network with Edge features (GINE) convolution.
    This is a custom implementation that doesn't require PyTorch Geometric.
    """
    
    def __init__(self, hidden_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = eps
        
        # Edge embedding
        self.edge_encoder = nn.Linear(1, hidden_dim)
        
        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of GINE convolution.
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
        
        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        num_nodes = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Encode edge weights
        edge_attr = self.edge_encoder(edge_weight.unsqueeze(-1))  # (num_edges, hidden_dim)
        
        # Get source features and add edge attributes
        src_features = x[src_idx] + edge_attr  # (num_edges, hidden_dim)
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out.index_add_(0, dst_idx, src_features)
        
        # Add self-loops with eps
        out = out + (1 + self.eps) * x
        
        # Update
        out = self.update_mlp(out)
        
        return out


class PositionalEncoding(nn.Module):
    """
    RoPE-style positional encoding for graph transformer.
    """
    
    def __init__(self, hidden_dim: int, max_positions: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create sinusoidal position encodings
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * 
                           -(math.log(10000.0) / hidden_dim))
        
        pe = torch.zeros(max_positions, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
            positions: Optional position indices
        
        Returns:
            Input with positional encoding added
        """
        if positions is None:
            if x.dim() == 3:
                seq_len = x.size(1)
                return x + self.pe[:seq_len].unsqueeze(0)
            else:
                seq_len = x.size(0)
                return x + self.pe[:seq_len]
        else:
            return x + self.pe[positions]


class GPSLayer(nn.Module):
    """
    Single GPS layer combining local message passing and global attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        # Local message passing (GINE)
        self.local_conv = SimplifiedGINEConv(hidden_dim)
        
        # Global attention
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Positional encoding for attention
        if use_rope:
            self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        num_nodes_per_graph: int,
    ) -> torch.Tensor:
        """
        Forward pass of GPS layer.
        
        Args:
            x: Node features (B*N, hidden_dim)
            edge_index: Edge connectivity (2, E)
            edge_weight: Edge weights (E,)
            batch_size: Number of graphs
            num_nodes_per_graph: Nodes per graph
        
        Returns:
            Updated node features (B*N, hidden_dim)
        """
        # Local message passing with residual
        x_local = self.local_conv(x, edge_index, edge_weight)
        x = x + self.dropout(x_local)
        x = self.norm1(x)
        
        # Reshape for global attention
        x_reshaped = x.view(batch_size, num_nodes_per_graph, -1)  # (B, N, D)
        
        # Add positional encoding if enabled
        if self.use_rope:
            x_attn = self.pos_encoder(x_reshaped)
        else:
            x_attn = x_reshaped
        
        # Global self-attention
        x_global, _ = self.global_attn(x_attn, x_attn, x_attn)
        x_reshaped = x_reshaped + self.dropout(x_global)
        x_reshaped = self.norm2(x_reshaped)
        
        # Feed-forward network
        x_ffn = self.ffn(x_reshaped)
        x_reshaped = x_reshaped + self.dropout(x_ffn)
        x_reshaped = self.norm3(x_reshaped)
        
        # Reshape back
        x = x_reshaped.view(batch_size * num_nodes_per_graph, -1)
        
        return x


class GPSGraphCorrection(GraphCorrectionHead):
    """
    GPS-based graph correction module for Graph Flow Matching.
    
    Combines local message passing (GINE) with global attention
    for more powerful graph processing.
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        in_channels: int,
        gfm_hidden_ratio: float = 0.05,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = True,
        enforce_param_budget: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the main model
            patch_size: Patch size for unpatchify operation
            in_channels: Number of input channels
            gfm_hidden_ratio: Ratio to compute GFM hidden dimension
            num_layers: Number of GPS layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_rope: Whether to use RoPE positional encoding
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
        
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        # Ensure hidden dim is divisible by num_heads
        if self.gfm_hidden % num_heads != 0:
            self.gfm_hidden = ((self.gfm_hidden // num_heads) + 1) * num_heads
        
        # Create GPS layers
        self.gps_layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=self.gfm_hidden,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=use_rope,
            )
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
        Process node features through GPS layers.
        
        Args:
            x: Node features (B*N, gfm_hidden)
            edge_index: Graph connectivity (2, E)
            edge_weight: Edge weights (E,)
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
        
        Returns:
            Updated node features (B*N, gfm_hidden)
        """
        # Process through GPS layers
        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, edge_weight, batch_size, num_nodes)
        
        return x


def test_gps_correction():
    """Test GPS correction module with random inputs."""
    import time
    
    # Setup
    batch_size = 2
    num_patches = 256  # 16x16 patches
    hidden_size = 1152
    patch_size = 2
    in_channels = 4
    
    # Create model
    model = GPSGraphCorrection(
        hidden_size=hidden_size,
        patch_size=patch_size,
        in_channels=in_channels,
        gfm_hidden_ratio=0.05,
        num_layers=2,  # Fewer layers for GPS due to attention
        num_heads=4,
    )
    
    # Create random inputs
    x_patches = torch.randn(batch_size, num_patches, hidden_size)
    t = torch.rand(batch_size)
    
    # Create random graph
    num_edges_per_node = 8
    edge_list = []
    weight_list = []
    
    for b in range(batch_size):
        for i in range(num_patches):
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_model_params = 100_000_000  # Assume 100M base model
    model.set_base_model_params(base_model_params)
    param_report = model.get_param_report()
    
    print("GPS Correction Module Test:")
    print(f"  Output shape: {v_diff.shape}")
    print(f"  Forward time: {forward_time:.3f}s")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Param ratio: {param_report['param_ratio']*100:.2f}%")
    print(f"  GFM hidden dim: {param_report['gfm_hidden_dim']}")
    print("  Test passed!")
    
    return model


if __name__ == "__main__":
    test_gps_correction()
