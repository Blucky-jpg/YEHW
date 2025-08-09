import math
import torch
import torch.nn as nn
from typing import Dict
from ..core.quantization import BlackwellOptimizedLinear

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