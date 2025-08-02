# quantization.py
# 
# This module provides optimized linear layers for Blackwell GPUs (e.g., RTX 5090/B200) using NVIDIA Transformer Engine (TE)
# for native FP4/FP8 quantization. Fallbacks to standard PyTorch are provided when TE is unavailable.
# 
# Key Features:
# - Blackwell-specific FP4/FP8 recipes for low-precision matmuls without quality loss.
# - Optimized attention integration with SageAttention.
# - Gradient monitoring for precision switching.
# 
# Usage:
# - Initialize distributed: torch.distributed.init_process_group(backend='nccl')
# - Use BlackwellOptimizedLinear or its parallel variants in models.
# 
# Dependencies: torch, transformer_engine (optional but recommended for Blackwell).

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from functools import lru_cache
from enum import Enum
from typing import Optional, Tuple
import logging

# Configure logging for production readiness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Integration with NVIDIA Transformer Engine for Native FP4 on Blackwell ---
try:
    import transformer_engine.pytorch as Te
    from transformer_engine.common.recipe import Format, DelayedScaling
    
    # Enhanced FP4 recipe optimized for Blackwell (RTX 5090/B200)
    fp4_recipe = DelayedScaling(
        format=Format.NVTE_FWD_NVFP4_BWD_MXFP8_SCALING,
        fp4_dtype=Te.common.recipe.DType.kNVTEFloat4E2M1,  # E2M1 for Blackwell
        max_history_len=32,  # Increased for better scaling tracking
        override_linear_precision=(False, False, True),  # Enable low-precision matmul
        amax_history_len=1024,  # Extended history for stable scaling
        amax_compute_algo="max",  # Use max for better numerical stability
        scaling_type="delayed"  # Delayed scaling for training stability
    )
    
    # Attention-specific FP8 recipe for Sage_Attention integration
    attention_fp8_recipe = DelayedScaling(
        format=Format.NVTE_FWD_FP8_BWD_FP8_SCALING,
        fp8_dtype_forward=Te.common.recipe.DType.kNVTEFloat8E4M3,  # E4M3 for attention
        fp8_dtype_backward=Te.common.recipe.DType.kNVTEFloat8E5M2,  # E5M2 for gradients
        max_history_len=32,
        override_linear_precision=(True, True, True),  # Full FP8 for attention
        amax_history_len=1024,
        amax_compute_algo="max",
        scaling_type="delayed"
    )
    
    HAS_TRANSFORMER_ENGINE = True
    logger.info("Transformer Engine loaded with Blackwell-optimized FP4/FP8 recipes")
except ImportError:
    logger.warning("Transformer Engine not available, falling back to standard PyTorch linear implementation")
    HAS_TRANSFORMER_ENGINE = False
    Te = None
    fp4_recipe = None
    attention_fp8_recipe = None

class RoundingMode(Enum):
    """Enumeration for quantization rounding modes."""
    ROUND_TO_NEAREST = "rtn"
    STOCHASTIC = "sr"

@lru_cache(maxsize=2)
def get_quantization_tables() -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute quantization lookup tables for faster runtime."""
    # E2M1 format values: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    e2m1_values = torch.tensor([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    
    # E4M3 format values (approximated for scale quantization)
    e4m3_exp_range = torch.arange(-6, 9, dtype=torch.float32)  # 2^-6 to 2^8
    e4m3_values = 2.0 ** e4m3_exp_range
    
    return e2m1_values, e4m3_values

class QuantizedLinear(nn.Module if not HAS_TRANSFORMER_ENGINE else Te.Linear):
    """
    Base linear layer using NVIDIA Transformer Engine for FP4 quantization on Blackwell.
    Falls back to standard nn.Linear if TE is unavailable.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: Optional[torch.dtype] = torch.bfloat16, **kwargs):
        if HAS_TRANSFORMER_ENGINE:
            super().__init__(in_features, out_features, bias=bias, params_dtype=dtype, **kwargs)
        else:
            super().__init__()
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform for stability."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if getattr(self, 'bias', None) is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass: Uses TE if available, else standard linear."""
        if HAS_TRANSFORMER_ENGINE:
            return super().forward(input_tensor)
        else:
            return torch.nn.functional.linear(input_tensor, self.weight, self.bias)

def get_optimal_fp8_context(use_attention_recipe: bool = False) -> Optional[Te.fp8_autocast]:
    """Get the optimal FP8 autocast context for Blackwell architecture."""
    if not HAS_TRANSFORMER_ENGINE:
        return None
    
    recipe = attention_fp8_recipe if use_attention_recipe else fp4_recipe
    return Te.fp8_autocast(enabled=True, fp8_recipe=recipe)

def sage_attention_with_fp8(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    is_causal: bool = True, 
    use_fp8_context: bool = True
    ) -> torch.Tensor:
    """
    Optimized Sage_Attention with FP8 integration for Blackwell.
    Falls back to PyTorch scaled_dot_product_attention if needed.
    """
    if not HAS_TRANSFORMER_ENGINE or not use_fp8_context:
        try:
            from sageattn import sageattn_blackwell
            return sageattn_blackwell(q, k, v, is_causal=is_causal)
        except ImportError:
            logger.warning("SageAttention not available, using PyTorch scaled_dot_product_attention")
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        except Exception as e:
            logger.error(f"SageAttention failed ({e}), falling back to standard attention")
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    # Use FP8 context for optimal Blackwell performance
    with get_optimal_fp8_context(use_attention_recipe=True):
        try:
            from sageattn import sageattn_blackwell
            # Convert to bfloat16 if in FP8 dtypes to match SageAttention expectations
            q_fp8 = q.to(dtype=torch.bfloat16) if q.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else q
            k_fp8 = k.to(dtype=torch.bfloat16) if k.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else k
            v_fp8 = v.to(dtype=torch.bfloat16) if v.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else v
            return sageattn_blackwell(q_fp8, k_fp8, v_fp8, is_causal=is_causal)
        except ImportError:
            logger.warning("SageAttention not available, using PyTorch scaled_dot_product_attention with FP8")
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        except Exception as e:
            logger.error(f"SageAttention with FP8 failed ({e}), falling back to standard attention")
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

class BlackwellOptimizedLinear(QuantizedLinear):
    """
    Enhanced linear layer optimized for Blackwell architecture with FP8 context management.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        use_fp8_context: bool = True, 
        dtype: Optional[torch.dtype] = torch.bfloat16, 
        **kwargs
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, **kwargs)
        self.use_fp8_context = use_fp8_context and HAS_TRANSFORMER_ENGINE
        self._blackwell_init()

    def _blackwell_init(self) -> None:
        """Blackwell-specific initialization for FP4/FP8 stability."""
        with torch.no_grad():
            std = math.sqrt(2.0 / (self.in_features + self.out_features)) * 0.7  # Reduced std for low-precision stability
            nn.init.normal_(self.weight, mean=0.0, std=std)
            if getattr(self, 'bias', None) is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional FP8 context for Blackwell optimization."""
        if self.use_fp8_context:
            with get_optimal_fp8_context(use_attention_recipe=False):
                return super().forward(input_tensor)
        return super().forward(input_tensor)

class GradientThresholdMonitor:
    """Monitors gradient-to-noise ratio for precision switching (inspired by research papers)."""
    
    def __init__(self, switch_threshold: float = math.sqrt(3)):
        self.switch_threshold = switch_threshold
        self.quantization_noise_std: Optional[float] = None
        self.should_switch_precision: bool = False
    
    def update(self, gradients: torch.Tensor, quantization_noise_std: float) -> float:
        """Update monitor and check if precision switch is needed."""
        grad_std = gradients.std().item()
        per_coord_grad_magnitude = grad_std / math.sqrt(gradients.numel())
        
        ratio = per_coord_grad_magnitude / (quantization_noise_std * self.switch_threshold)
        
        if ratio < 1.0 and not self.should_switch_precision:
            logger.info(f"Gradient threshold reached. Ratio: {ratio:.4f}. Consider switching to higher precision.")
            self.should_switch_precision = True
        
        return ratio