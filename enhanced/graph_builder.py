"""
Graph construction utilities for Graph Flow Matching (GFM).
Builds dynamic sparse graphs from attention maps or latent features.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


@torch.jit.script
def topk_masking(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Create a mask for top-k values per row."""
    _, indices = torch.topk(scores, k, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def build_dynamic_graph(
    attn_scores: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    top_k: int = 8,
    batch_size: Optional[int] = None,
    symmetric: bool = True,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dynamic graph from attention scores or feature similarities.
    
    Args:
        attn_scores: Attention scores of shape (B, H, N, N) or (B, N, N)
                    where B=batch, H=heads, N=num_nodes
        features: Feature tensor of shape (B, N, D) for k-NN fallback
        top_k: Number of neighbors per node
        batch_size: Batch size (required if using features)
        symmetric: Whether to make the graph undirected
        temperature: Temperature for softmax normalization
    
    Returns:
        edge_index: Graph connectivity in COO format (2, E)
        edge_weight: Edge weights (E,)
    """
    
    if attn_scores is not None:
        # Use attention scores to build graph
        if attn_scores.dim() == 4:  # (B, H, N, N)
            # Average over attention heads
            attn_scores = attn_scores.mean(dim=1)  # (B, N, N)
        
        B, N, _ = attn_scores.shape
        device = attn_scores.device
        
        # Apply temperature scaling
        if temperature != 1.0:
            attn_scores = attn_scores / temperature
        
        # Get top-k neighbors per node
        # Mask self-connections
        mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask, -float('inf'))
        
        # Get top-k values and indices
        topk_vals, topk_indices = torch.topk(attn_scores, min(top_k, N-1), dim=-1)
        
        # Build edge index with batch offsets
        edge_list = []
        weight_list = []
        
        for b in range(B):
            offset = b * N
            for i in range(N):
                # Source nodes
                src = torch.full((min(top_k, N-1),), i + offset, device=device)
                # Target nodes  
                dst = topk_indices[b, i] + offset
                # Edge weights (apply softmax for normalization)
                weights = F.softmax(topk_vals[b, i], dim=0)
                
                edge_list.append(torch.stack([src, dst], dim=0))
                weight_list.append(weights)
        
        edge_index = torch.cat(edge_list, dim=1)
        edge_weight = torch.cat(weight_list, dim=0)
        
    elif features is not None:
        # Fallback: k-NN in feature space
        if batch_size is None:
            raise ValueError("batch_size required when using feature-based graph construction")
        
        B = batch_size
        N = features.shape[0] // B
        D = features.shape[-1]
        device = features.device
        
        # Reshape to (B, N, D)
        features = features.view(B, N, D)
        
        # Compute pairwise distances
        # Using cosine similarity as distance metric
        features_norm = F.normalize(features, p=2, dim=-1)
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))  # (B, N, N)
        
        # Mask self-connections
        mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        similarity = similarity.masked_fill(mask, -float('inf'))
        
        # Get top-k neighbors
        topk_vals, topk_indices = torch.topk(similarity, min(top_k, N-1), dim=-1)
        
        # Build edge index with batch offsets
        edge_list = []
        weight_list = []
        
        for b in range(B):
            offset = b * N
            for i in range(N):
                src = torch.full((min(top_k, N-1),), i + offset, device=device)
                dst = topk_indices[b, i] + offset
                # Convert similarities to weights
                weights = F.softmax(topk_vals[b, i] / temperature, dim=0)
                
                edge_list.append(torch.stack([src, dst], dim=0))
                weight_list.append(weights)
        
        edge_index = torch.cat(edge_list, dim=1)
        edge_weight = torch.cat(weight_list, dim=0)
        
    else:
        raise ValueError("Either attn_scores or features must be provided")
    
    # Make graph symmetric if requested
    if symmetric:
        # Add reverse edges
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        
        # Remove duplicates by creating a unique edge set
        # Convert to tuple for hashing
        edges_set = set()
        unique_edges = []
        unique_weights = []
        
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edges_set:
                edges_set.add(edge)
                unique_edges.append(edge_index[:, i:i+1])
                unique_weights.append(edge_weight[i:i+1])
        
        if unique_edges:
            edge_index = torch.cat(unique_edges, dim=1)
            edge_weight = torch.cat(unique_weights, dim=0)
    
    return edge_index, edge_weight


def batch_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes_per_graph: int
) -> torch.Tensor:
    """
    Create batched edge index by offsetting node indices.
    
    Args:
        edge_index: Single graph edge index (2, E)
        batch_size: Number of graphs to batch
        num_nodes_per_graph: Number of nodes in each graph
    
    Returns:
        Batched edge index (2, E*batch_size)
    """
    edge_list = []
    for b in range(batch_size):
        offset = b * num_nodes_per_graph
        edge_list.append(edge_index + offset)
    
    return torch.cat(edge_list, dim=1)


def graph_to_dense(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Convert sparse graph to dense adjacency matrix.

    Args:
        edge_index: Edge connectivity (2, E)
        edge_weight: Edge weights (E,)
        num_nodes: Total number of nodes

    Returns:
        Dense adjacency matrix (num_nodes, num_nodes)
    """
    # Memory safety check
    max_safe_nodes = 4096  # Don't create matrices larger than 4096x4096
    if num_nodes > max_safe_nodes:
        raise MemoryError(f"Cannot create {num_nodes}x{num_nodes} dense matrix. "
                         f"Maximum safe size is {max_safe_nodes}x{max_safe_nodes}. "
                         f"Use sparse operations instead.")

    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = edge_weight
    return adj


class GraphStatistics:
    """Track and log graph statistics for monitoring."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.num_edges_history = []
        self.avg_degree_history = []
        self.edge_weight_mean_history = []
        self.edge_weight_std_history = []
    
    def update(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
        """Update statistics with new graph."""
        num_edges = edge_index.shape[1]
        avg_degree = num_edges / num_nodes
        
        self.num_edges_history.append(num_edges)
        self.avg_degree_history.append(avg_degree)
        self.edge_weight_mean_history.append(edge_weight.mean().item())
        self.edge_weight_std_history.append(edge_weight.std().item())
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.num_edges_history:
            return {}
        
        return {
            'avg_num_edges': sum(self.num_edges_history) / len(self.num_edges_history),
            'avg_degree': sum(self.avg_degree_history) / len(self.avg_degree_history),
            'edge_weight_mean': sum(self.edge_weight_mean_history) / len(self.edge_weight_mean_history),
            'edge_weight_std': sum(self.edge_weight_std_history) / len(self.edge_weight_std_history),
        }
