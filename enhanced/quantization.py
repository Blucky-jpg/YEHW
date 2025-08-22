# quantization.py
#
# FP16 Optimized Quantization Module
# ===================================
#
# This module provides FP16 quantization for enhanced model performance
# with universal hardware compatibility and excellent numerical stability.
#
# Key Features:
# - Standard FP16 precision for all operations
# - Universal GPU compatibility (works on any modern GPU)
# - Simplified implementation with proven stability
# - Optimized for the DeltaNetDiT architecture
#
# Usage:
# - Use QuantizedLinear for drop-in replacement of nn.Linear
# - Automatic FP16 casting for better performance
# - Works on any GPU with CUDA support

import math
import torch
import torch.nn as nn
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Linear):
    """
    FP16-optimized linear layer with enhanced initialization.
    Drop-in replacement for nn.Linear with better defaults for transformers.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = torch.float16,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, **kwargs)
        self._enhanced_init()

    def _enhanced_init(self) -> None:
        """Enhanced initialization for FP16 stability."""
        with torch.no_grad():
            # More conservative initialization for FP16
            std = math.sqrt(2.0 / (self.in_features + self.out_features)) * 0.5
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
    Optimized for FP16 precision.
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


def flash_attention_with_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True
) -> torch.Tensor:
    """
    Optimized FlashAttention for FP16 with fallback to PyTorch attention.
    """
    flashattn_func = get_flash_attention()

    if flashattn_func is not None:
        try:
            return flashattn_func(q, k, v, is_causal=is_causal)
        except Exception as e:
            logger.warning(f"FlashAttention failed ({e}), falling back to standard attention")

    # Fallback to PyTorch's optimized attention
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


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
