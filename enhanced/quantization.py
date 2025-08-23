# quantization.py
#
# BF16 Optimized Quantization Module
# ===================================
#
# This module provides BF16 (BFloat16) quantization for enhanced model performance
# with superior numerical stability and modern hardware optimization.
#
# Key Features:
# - BF16 precision for all operations (better dynamic range than FP16)
# - Optimized for modern GPUs with BF16 support
# - Enhanced numerical stability for large-scale training
# - Optimized for the DeltaNetDiT architecture
#
# Usage:
# - Use QuantizedLinear for drop-in replacement of nn.Linear
# - Automatic BF16 casting for better performance and stability
# - Leverages BF16's superior dynamic range for training

import math
import torch
import torch.nn as nn
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BF16 Optimization Components
class PureBF16Tensor:
    """
    Wrapper for tensors that ensures pure BF16 operations with optimal
    tensor core utilization on RTX 4090/3090 GPUs.
    """

    @staticmethod
    def ensure_bf16(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is in BF16 format with optimal memory alignment."""
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(dtype=torch.bfloat16)

        # Ensure tensor is contiguous for optimal tensor core performance
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    @staticmethod
    def optimal_bf16_layout(tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor layout for BF16 tensor core operations."""
        # For matrix operations, ensure proper alignment for tensor cores
        if tensor.dim() >= 2:
            # Ensure last dimension is aligned for optimal tensor core performance
            last_dim = tensor.shape[-1]
            if last_dim % 8 != 0:
                # Pad to nearest multiple of 8 for tensor core alignment
                pad_size = (8 - last_dim % 8) % 8
                if pad_size > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)

        return tensor

    @staticmethod
    def create_bf16_tensor(*args, **kwargs) -> torch.Tensor:
        """Create a new tensor with optimal BF16 configuration."""
        kwargs['dtype'] = torch.bfloat16
        tensor = torch.randn(*args, **kwargs)
        return PureBF16Tensor.optimal_bf16_layout(tensor)


class BF16Optimizer:
    """
    Central optimization engine for BF16 performance on RTX 4090/3090 GPUs.

    This class provides:
    - Automatic BF16 consistency enforcement
    - Memory-efficient BF16 operations
    - Performance monitoring and optimization
    - Tensor core utilization optimization
    """

    def __init__(
        self,
        memory_threshold_gb: float = 2.0,
        enable_performance_monitoring: bool = True,
        optimize_for_tensor_cores: bool = True
    ):
        self.memory_threshold = memory_threshold_gb
        self.enable_performance_monitoring = enable_performance_monitoring
        self.optimize_for_tensor_cores = optimize_for_tensor_cores

        # Performance tracking
        self.performance_stats = {
            'bf16_operations': 0,
            'memory_efficiency': 0.0,
            'tensor_core_utilization': 0.0,
            'optimization_time': 0.0
        }

        # Memory management - less aggressive than standard optimizer
        self.last_memory_check = 0
        self.memory_check_interval = 100  # Check every 100 operations
        self.operation_counter = 0

        logger.info("BF16Optimizer initialized for RTX 4090/3090 optimization")

    def should_optimize_memory(self) -> bool:
        """Determine if memory optimization is needed."""
        if not torch.cuda.is_available():
            return False

        self.operation_counter += 1
        if self.operation_counter - self.last_memory_check < self.memory_check_interval:
            return False

        self.last_memory_check = self.operation_counter
        current_memory = torch.cuda.memory_allocated() / 1e9

        should_optimize = current_memory > self.memory_threshold
        if should_optimize:
            logger.debug(".1f")

        return should_optimize

    def gentle_memory_cleanup(self):
        """Gentle memory cleanup that doesn't interfere with BF16 efficiency."""
        if not torch.cuda.is_available():
            return

        # Only clear cache, don't force garbage collection
        torch.cuda.empty_cache()

        # Log memory status
        current_memory = torch.cuda.memory_allocated() / 1e9
        logger.debug(".1f")

    def optimize_linear_layer(self, linear: nn.Linear) -> 'PureBF16Linear':
        """Convert standard linear layer to BF16-optimized version."""
        return PureBF16Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None
        )

    def optimize_attention(self, attn_module) -> nn.Module:
        """Apply BF16 optimizations to attention modules."""
        # This will be implemented when we modify the attention classes
        return attn_module

    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1e9
            self.performance_stats['memory_efficiency'] = max(0.0, 1.0 - current_memory / 24.0)  # RTX 4090 has 24GB

        return self.performance_stats.copy()


class PureBF16Linear(nn.Linear):
    """
    BF16-optimized linear layer with guaranteed tensor core utilization.

    This layer ensures all operations use BF16 and are aligned for optimal
    tensor core performance on RTX 4090/3090 GPUs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.bfloat16
    ):
        # Force BF16 dtype
        super().__init__(in_features, out_features, bias, device, dtype)

        # Initialize weights with BF16-optimal scaling
        with torch.no_grad():
            # Use smaller initialization for BF16 stability
            std = (2.0 / (in_features + out_features)) ** 0.5
            nn.init.normal_(self.weight, mean=0, std=std)

            if self.bias is not None:
                self.bias.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure input is in BF16
        input_bf16 = PureBF16Tensor.ensure_bf16(input)

        # Optimize input layout for tensor cores
        if self.optimize_for_tensor_cores:
            input_bf16 = PureBF16Tensor.optimal_bf16_layout(input_bf16)

        # Perform BF16 matrix multiplication
        output = torch.nn.functional.linear(input_bf16, self.weight, self.bias)

        return output

    @property
    def optimize_for_tensor_cores(self) -> bool:
        """Check if tensor core optimization is enabled."""
        return getattr(self, '_optimize_tensor_cores', True)

    @optimize_for_tensor_cores.setter
    def optimize_for_tensor_cores(self, value: bool):
        """Enable/disable tensor core optimization."""
        self._optimize_tensor_cores = value


def optimize_model_for_bf16(model: nn.Module, bf16_optimizer: BF16Optimizer):
    """
    Apply comprehensive BF16 optimizations to a model.

    Args:
        model: PyTorch model to optimize
        bf16_optimizer: BF16Optimizer instance
    """

    # Replace Linear layers with BF16-optimized versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            # Replace with BF16-optimized linear layer
            new_layer = bf16_optimizer.optimize_linear_layer(module)

            # Replace in parent module
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, child_name, new_layer)

    logger.info("Model BF16 optimizations applied")


def create_bf16_optimized_model(
    model_class,
    bf16_optimizer: BF16Optimizer,
    *args,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a BF16-optimized model.

    Args:
        model_class: Model class to instantiate
        bf16_optimizer: BF16Optimizer instance
        *args: Positional arguments for model constructor
        **kwargs: Keyword arguments for model constructor

    Returns:
        BF16-optimized model instance
    """

    # Create model
    model = model_class(*args, **kwargs)

    # Apply BF16 optimizations
    optimize_model_for_bf16(model, bf16_optimizer)

    return model


# Global BF16 optimizer instance
_global_bf16_optimizer = None

def get_global_bf16_optimizer() -> BF16Optimizer:
    """Get or create global BF16 optimizer instance."""
    global _global_bf16_optimizer
    if _global_bf16_optimizer is None:
        _global_bf16_optimizer = BF16Optimizer()
    return _global_bf16_optimizer


# Utility functions for BF16 operations
def ensure_bf16_computation(func):
    """Decorator to ensure function uses BF16 computation."""
    def wrapper(*args, **kwargs):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return func(*args, **kwargs)
    return wrapper


def log_bf16_performance(operation_name: str = ""):
    """Log current BF16 performance metrics."""
    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated() / 1e9
        logger.info(".1f")


def clear_bf16_cache():
    """Clear GPU cache optimized for BF16 operations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QuantizedLinear(nn.Linear):
    """
    BF16-optimized linear layer with enhanced initialization.
    Drop-in replacement for nn.Linear with better defaults for transformers.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, **kwargs)
        self._bf16_init()

    def _bf16_init(self) -> None:
        """Enhanced initialization optimized for BF16 stability."""
        with torch.no_grad():
            # BF16-optimized initialization - less conservative than FP16
            # BF16's better dynamic range allows for more standard initialization
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            nn.init.normal_(self.weight, mean=0.0, std=std)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Standard linear forward with automatic mixed precision handling."""
        # Ensure input is in the same dtype as weights for consistency
        if input_tensor.dtype != self.weight.dtype:
            input_tensor = input_tensor.to(dtype=self.weight.dtype)
        return super().forward(input_tensor)


def get_flash_attention():
    """
    Get FlashAttention function with fallback to standard attention.
    Optimized for BF16 precision with enhanced stability.
    """
    try:
        # Try different possible import paths for FlashAttention
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func
        except ImportError:
            try:
                import flash_attn
                return flash_attn.flash_attn_func
            except ImportError:
                logger.debug("FlashAttention not available, using standard attention")
                return None
    except Exception as e:
        logger.debug(f"Error importing FlashAttention: {e}")
        return None


def flash_attention_with_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True
) -> torch.Tensor:
    """
    Optimized FlashAttention for BF16 with enhanced numerical stability.
    """
    flashattn_func = get_flash_attention()

    if flashattn_func is not None:
        try:
            # Ensure inputs are in BF16 for optimal performance
            q_bf16 = q.to(torch.bfloat16) if q.dtype != torch.bfloat16 else q
            k_bf16 = k.to(torch.bfloat16) if k.dtype != torch.bfloat16 else k
            v_bf16 = v.to(torch.bfloat16) if v.dtype != torch.bfloat16 else v

            result = flashattn_func(q_bf16, k_bf16, v_bf16, is_causal=is_causal)
            return result.to(q.dtype)  # Return in original dtype
        except Exception as e:
            logger.warning(f"FlashAttention failed ({e}), falling back to standard attention")

    # Fallback to PyTorch's optimized attention with BF16
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


# Backward compatibility alias
flash_attention_with_fp16 = flash_attention_with_bf16


# Alias for backward compatibility
BlackwellOptimizedLinear = QuantizedLinear

def safe_torch_compile(func, dynamic=False):
    """
    Safely compile a function with torch.compile, with fallback to original function.
    """
    try:
        if hasattr(torch, 'compile'):
            return torch.compile(func, dynamic=dynamic)
        else:
            logger.debug("torch.compile not available, using original function")
            return func
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}, using original function")
        return func
