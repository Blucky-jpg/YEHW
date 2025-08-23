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

# Import BF16 optimization components
try:
    from enhanced.bf16_optimizer import PureBF16Tensor, BF16Optimizer
    BF16_AVAILABLE = True
except ImportError:
    BF16_AVAILABLE = False
    logger.debug("BF16Optimizer not available, using standard quantization")


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
