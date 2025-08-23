"""
Memory Optimization Module for Enhanced Models

This module provides comprehensive memory optimization techniques for training
large-scale models with limited GPU memory. It includes:

1. Memory-aware gradient checkpointing
2. Adaptive memory management
3. Memory-efficient attention mechanisms
4. Optimized tensor operations
5. Automatic memory cleanup and monitoring

Author: DeltaNet Optimization Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Callable, Tuple
from functools import partial
import logging
import gc

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Central memory optimization manager for large-scale model training.

    This class provides:
    - Automatic memory monitoring and cleanup
    - Adaptive gradient checkpointing
    - Memory-efficient tensor operations
    - GPU memory defragmentation
    - BF16-aware optimization for RTX 4090/3090
    """

    def __init__(
        self,
        memory_threshold_gb: float = 2.0,
        cleanup_interval: int = 100,
        enable_gc: bool = True,
        enable_defragmentation: bool = True,
        bf16_optimization_mode: bool = False
    ):
        self.memory_threshold = memory_threshold_gb
        self.cleanup_interval = cleanup_interval
        self.enable_gc = enable_gc
        self.enable_defragmentation = enable_defragmentation
        self.bf16_optimization_mode = bf16_optimization_mode

        # Adjust settings for BF16 optimization
        if bf16_optimization_mode:
            # Less aggressive cleanup for BF16 efficiency
            self.cleanup_interval = 200  # Reduce cleanup frequency
            self.memory_threshold = 4.0  # Higher threshold before cleanup
            logger.info("MemoryOptimizer initialized in BF16 optimization mode")
        else:
            logger.info("MemoryOptimizer initialized in standard mode")

        self.step_counter = 0
        self.memory_history = []
        self.peak_memory = 0.0

        # Memory optimization settings
        self.use_gradient_checkpointing = True
        self.use_memory_efficient_attention = True
        self.use_inplace_operations = True

    def should_optimize(self) -> bool:
        """Determine if memory optimization is needed."""
        if not torch.cuda.is_available():
            return False

        current_memory = torch.cuda.memory_allocated() / 1e9
        return current_memory > self.memory_threshold

    def cleanup_memory(self, force: bool = False):
        """Perform comprehensive memory cleanup."""
        if not torch.cuda.is_available():
            return

        self.step_counter += 1

        # Periodic cleanup
        if force or (self.step_counter % self.cleanup_interval == 0):
            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Force garbage collection
            if self.enable_gc:
                gc.collect()

            # Memory defragmentation (experimental)
            if self.enable_defragmentation:
                self._defragment_memory()

            # Log memory usage
            current_memory = torch.cuda.memory_allocated() / 1e9
            reserved_memory = torch.cuda.memory_reserved() / 1e9

            self.memory_history.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)

            if len(self.memory_history) > 1000:
                self.memory_history.pop(0)

            logger.info(".1f"
                       ".1f"
                       ".1f")

    def _defragment_memory(self):
        """Attempt memory defragmentation by reallocating tensors."""
        try:
            # This is a simplified defragmentation approach
            # In practice, you might want to implement more sophisticated strategies
            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                logger.warning("High memory usage detected, attempting defragmentation")
                # Force a full cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Memory defragmentation failed: {e}")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'peak_allocated_gb': self.peak_memory,
            'utilization_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100,
            'cleanup_count': self.step_counter // self.cleanup_interval,
        }

    def adaptive_gradient_checkpointing(
        self,
        function: Callable,
        *args,
        memory_threshold: float = 0.7,
        **kwargs
    ):
        """
        Apply gradient checkpointing only when memory usage is high.

        Args:
            function: Function to checkpoint
            memory_threshold: Memory usage threshold (0-1) to trigger checkpointing
        """
        if not self.use_gradient_checkpointing:
            return function(*args, **kwargs)

        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory

            if current_memory > memory_threshold:
                return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

        return function(*args, **kwargs)


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanism with automatic optimization.

    Features:
    - Automatic sequence length-based optimization
    - Memory-aware attention computation
    - Support for different attention patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        memory_efficient: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.memory_efficient = memory_efficient

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _memory_efficient_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Memory-efficient attention computation.
        """
        B, H, L, D = q.shape

        # For very long sequences, use memory-efficient approach
        if L > 2048:
            return self._chunked_attention(q, k, v, attn_mask, is_causal, chunk_size=1024)
        elif L > 1024:
            return self._chunked_attention(q, k, v, attn_mask, is_causal, chunk_size=512)
        else:
            # Standard attention for shorter sequences
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if is_causal:
                attn_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            return torch.matmul(attn_weights, v)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Chunked attention for very long sequences.
        """
        B, H, L, D = q.shape
        num_chunks = (L + chunk_size - 1) // chunk_size

        output = torch.zeros_like(q)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, L)

            q_chunk = q[:, :, start_idx:end_idx]
            k_chunk = k[:, :, :end_idx]  # Include all previous tokens

            # Compute attention for this chunk
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale

            if is_causal:
                # Create causal mask for this chunk
                chunk_mask = torch.triu(torch.ones(end_idx - start_idx, end_idx, device=q.device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(chunk_mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            v_chunk = v[:, :, :end_idx]
            output[:, :, start_idx:end_idx] = torch.matmul(attn_weights, v_chunk)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with memory optimization.
        """
        B, L, D = hidden_states.shape

        # Linear projections
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.memory_efficient:
            attn_output = self._memory_efficient_attention(q, k, v, attention_mask, is_causal)
        else:
            # Standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if is_causal:
                causal_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


class InPlaceLayerNorm(nn.LayerNorm):
    """
    In-place LayerNorm for memory efficiency.

    This implementation modifies the input tensor in-place to save memory,
    which is particularly useful during training with large batch sizes.
    """

    def __init__(self, *args, inplace: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace and x.requires_grad and x.is_leaf:
            # In-place normalization for memory efficiency
            mean = x.mean(dim=-1, keepdim=True)
            x.sub_(mean)

            var = x.pow(2).mean(dim=-1, keepdim=True)
            x.div_((var + self.eps).sqrt())

            if self.weight is not None:
                x.mul_(self.weight.view(1, 1, -1))

            if self.bias is not None:
                x.add_(self.bias.view(1, 1, -1))

            return x
        else:
            # Standard LayerNorm
            return super().forward(x)


class MemoryEfficientDropout(nn.Module):
    """
    Memory-efficient dropout that avoids creating mask tensors when not needed.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            return F.dropout(x, p=self.p, training=True, inplace=False)
        return x


def optimize_model_for_memory(model: nn.Module, memory_optimizer: MemoryOptimizer):
    """
    Apply comprehensive memory optimizations to a model.

    Args:
        model: PyTorch model to optimize
        memory_optimizer: MemoryOptimizer instance
    """

    # Replace LayerNorm with memory-efficient version
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            # Replace with memory-efficient LayerNorm
            new_layer = InPlaceLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                inplace=True
            )

            if hasattr(module, 'weight') and module.weight is not None:
                new_layer.weight.data.copy_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)

            # Replace in parent module
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, child_name, new_layer)

        # Replace Dropout with memory-efficient version
        elif isinstance(module, nn.Dropout):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            new_dropout = MemoryEfficientDropout(p=module.p)

            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_dropout)
            else:
                setattr(model, child_name, new_dropout)

    logger.info("Model memory optimizations applied")


def create_memory_efficient_model(
    model_class,
    memory_optimizer: MemoryOptimizer,
    *args,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a memory-efficient model.

    Args:
        model_class: Model class to instantiate
        memory_optimizer: MemoryOptimizer instance
        *args: Positional arguments for model constructor
        **kwargs: Keyword arguments for model constructor

    Returns:
        Memory-optimized model instance
    """

    # Create model
    model = model_class(*args, **kwargs)

    # Apply memory optimizations
    optimize_model_for_memory(model, memory_optimizer)

    return model


# Utility functions for memory management
def log_memory_usage(operation_name: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(".1f")


def clear_memory_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def estimate_memory_requirements(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Estimate memory requirements for model training.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (excluding batch dimension)

    Returns:
        Dictionary with memory estimates
    """
    if not torch.cuda.is_available():
        return {}

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape, device='cuda')

    # Forward pass to estimate memory
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()

        try:
            output = model(dummy_input)
            forward_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
        except Exception as e:
            logger.error(f"Memory estimation failed: {e}")
            return {}

    return {
        'estimated_forward_memory_gb': forward_memory,
        'estimated_peak_memory_gb': peak_memory,
        'input_shape': input_shape,
        'output_shape': output.shape if 'output' in locals() else None
    }


# Global memory optimizer instance
_global_memory_optimizer = None

def get_global_memory_optimizer() -> MemoryOptimizer:
    """Get or create global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    return _global_memory_optimizer
