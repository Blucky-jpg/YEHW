"""
Advanced Image-centric Rotary Position Embedding (RoPE) Module.

This module provides sophisticated positional encoding schemes optimized for vision tasks,
based on innovations from Qwen-Image and extended with additional features.

Key Features:
- Center-outward encoding for natural image understanding
- Multiple scaling strategies (linear, NTK-aware, dynamic-NTK)
- 2D positional encoding for spatial relationship preservation
- Multiple position generation strategies (radial, spiral, Manhattan, Chebyshev)
- Efficient caching mechanism for performance optimization
- Support for variable image resolutions

Usage:
    # Basic usage with default settings
    rope = ImageRotaryEmbedding(dim=768)
    x_with_rope = rope(x, image_size=(14, 14))
    
    # Advanced usage with NTK-aware scaling
    rope = ImageRotaryEmbedding(
        dim=768,
        rope_scaling={'type': 'ntk-aware', 'factor': 2.0},
        position_strategy='spiral'
    )
    
    # 2D positional encoding
    rope = ImageRotaryEmbedding(dim=768, use_2d_encoding=True)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Union


class ImageRotaryEmbedding(nn.Module):
    """
    MSRoPE: A specialized RoPE for images that uses center-outward encoding.
    
    Features:
    - Center-outward position encoding for better image understanding
    - Support for multiple scaling strategies (linear, NTK-aware, dynamic-NTK)
    - 2D positional encoding option for preserving spatial relationships
    - Caching for improved performance
    - Multiple position generation strategies (radial, spiral, manhattan)
    
    Adapted from the Qwen-Image model to be image-only.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        use_2d_encoding: bool = False,
        position_strategy: str = "radial",
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_2d_encoding = use_2d_encoding
        self.position_strategy = position_strategy
        
        # Validate position strategy
        if position_strategy not in ["radial", "spiral", "manhattan", "chebyshev"]:
            raise ValueError(f"Unknown position strategy: {position_strategy}")
        
        # Initialize caching for embeddings
        self._cache = {}
        self._position_cache = {}  # Separate cache for position computations

        # Compute inverse frequencies
        if use_2d_encoding:
            # For 2D encoding, split dimensions between x and y
            dim_per_axis = self.dim // 4  # Each axis gets half, split between sin/cos
            inv_freq_x = 1.0 / (
                self.base ** (torch.arange(0, dim_per_axis, dtype=torch.float32) / dim_per_axis)
            )
            inv_freq_y = 1.0 / (
                self.base ** (torch.arange(0, dim_per_axis, dtype=torch.float32) / dim_per_axis)
            )
            self.register_buffer("inv_freq_x", inv_freq_x, persistent=False)
            self.register_buffer("inv_freq_y", inv_freq_y, persistent=False)
        else:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Configure scaling
        self._configure_scaling(rope_scaling)

    def _configure_scaling(self, rope_scaling: Optional[dict]):
        """Configure the scaling method based on the provided dictionary."""
        if rope_scaling is None:
            self.scaling_type = "linear"
            self.scaling_factor = 1.0
            self.ntk_alpha = 1.0
            return

        self.scaling_type = rope_scaling.get("type", "linear")
        self.scaling_factor = rope_scaling.get("factor", 1.0)
        self.ntk_alpha = rope_scaling.get("ntk_alpha", 1.0)  # NTK scaling coefficient
        
        if self.scaling_type not in ["linear", "ntk-aware", "dynamic-ntk"]:
            raise ValueError(f"Unknown RoPE scaling type: {self.scaling_type}")

    def _get_or_create_embeddings(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from cache or compute, cache, and return the rotary embeddings.
        """
        cache_key = (
            image_size, seq_len, device, dtype, 
            self.scaling_type, self.scaling_factor,
            self.use_2d_encoding, self.position_strategy
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_2d_encoding:
            cos_emb, sin_emb = self._compute_2d_embeddings(
                image_size, seq_len, device, dtype
            )
        else:
            # Generate positions based on strategy
            positions = self._get_positions_by_strategy(image_size, seq_len, device)

            # Get the appropriate inverse frequencies based on scaling type
            inv_freq = self._get_scaled_inv_freq(positions, device)
            
            # Compute sin/cos embeddings
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            cos_emb = emb.cos().to(dtype)
            sin_emb = emb.sin().to(dtype)
        
        # Cache and return
        self._cache[cache_key] = (cos_emb, sin_emb)
        return cos_emb, sin_emb

    def _get_scaled_inv_freq(self, positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Get scaled inverse frequencies based on the scaling type."""
        if self.scaling_type == "linear":
            # Simple linear scaling of positions
            return self.inv_freq.to(device) / self.scaling_factor
        
        elif self.scaling_type == "ntk-aware":
            # NTK-aware scaling adjusts the base frequency
            # This maintains better long-range attention patterns
            scaled_base = self.base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim)
            )
            return inv_freq
        
        elif self.scaling_type == "dynamic-ntk":
            # Dynamic NTK scaling adjusts based on actual sequence length
            seq_len = positions.shape[0]
            if seq_len > self.max_position_embeddings:
                # Dynamically scale when exceeding max positions
                scale = seq_len / self.max_position_embeddings
                scaled_base = self.base * (scale * self.ntk_alpha) ** (self.dim / (self.dim - 2))
            else:
                scaled_base = self.base
            
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim)
            )
            return inv_freq
        
        return self.inv_freq.to(device)

    def forward(
        self,
        x: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Apply center-outward RoPE to an image patch tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            image_size: (H, W) tuple for the patch grid
        
        Returns:
            Tensor with RoPE applied
        """
        seq_len = x.shape[1]
        
        # Get or compute embeddings
        cos_emb, sin_emb = self._get_or_create_embeddings(
            image_size, seq_len, x.device, x.dtype
        )
        
        # Apply rotary embeddings
        return self._apply_rotary_pos_emb(x, cos_emb, sin_emb)
    
    def _get_positions_by_strategy(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate positions based on the selected strategy."""
        position_cache_key = (image_size, seq_len, self.position_strategy)
        if position_cache_key in self._position_cache:
            return self._position_cache[position_cache_key].to(device)
        
        if self.position_strategy == "radial":
            positions = self._get_radial_positions(image_size, seq_len, device)
        elif self.position_strategy == "spiral":
            positions = self._get_spiral_positions(image_size, seq_len, device)
        elif self.position_strategy == "manhattan":
            positions = self._get_manhattan_positions(image_size, seq_len, device)
        elif self.position_strategy == "chebyshev":
            positions = self._get_chebyshev_positions(image_size, seq_len, device)
        else:
            positions = self._get_radial_positions(image_size, seq_len, device)
        
        self._position_cache[position_cache_key] = positions.cpu()
        return positions
    
    def _get_radial_positions(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate center-outward radial position indices (Euclidean distance).
        This is the key innovation from Qwen-Image.
        """
        H, W = image_size
        center_h, center_w = (H - 1) / 2, (W - 1) / 2
        
        # Create grid of positions
        h_pos = torch.arange(H, device=device) - center_h
        w_pos = torch.arange(W, device=device) - center_w
        
        # Compute distances from center
        h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
        distances = torch.sqrt(h_grid.float()**2 + w_grid.float()**2)
        
        # Flatten and get relative positions
        return distances.flatten()[:seq_len]
    
    def _get_spiral_positions(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate positions following a spiral pattern from center."""
        H, W = image_size
        positions = torch.zeros(H * W, device=device)
        
        # Start from center
        cy, cx = H // 2, W // 2
        positions[cy * W + cx] = 0
        
        # Spiral outward
        idx = 1
        for ring in range(1, max(H, W)):
            # Top edge
            for x in range(max(0, cx - ring), min(W, cx + ring + 1)):
                y = cy - ring
                if 0 <= y < H and idx < H * W:
                    positions[y * W + x] = idx
                    idx += 1
            
            # Right edge
            for y in range(max(0, cy - ring + 1), min(H, cy + ring + 1)):
                x = cx + ring
                if 0 <= x < W and idx < H * W:
                    positions[y * W + x] = idx
                    idx += 1
            
            # Bottom edge
            for x in range(min(W - 1, cx + ring - 1), max(-1, cx - ring - 1), -1):
                y = cy + ring
                if 0 <= y < H and idx < H * W:
                    positions[y * W + x] = idx
                    idx += 1
            
            # Left edge
            for y in range(min(H - 1, cy + ring - 1), max(-1, cy - ring), -1):
                x = cx - ring
                if 0 <= x < W and idx < H * W:
                    positions[y * W + x] = idx
                    idx += 1
        
        return positions[:seq_len]
    
    def _get_manhattan_positions(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate positions based on Manhattan distance from center."""
        H, W = image_size
        center_h, center_w = (H - 1) / 2, (W - 1) / 2
        
        h_pos = torch.arange(H, device=device) - center_h
        w_pos = torch.arange(W, device=device) - center_w
        
        h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
        distances = torch.abs(h_grid) + torch.abs(w_grid)
        
        return distances.flatten()[:seq_len]
    
    def _get_chebyshev_positions(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate positions based on Chebyshev distance from center."""
        H, W = image_size
        center_h, center_w = (H - 1) / 2, (W - 1) / 2
        
        h_pos = torch.arange(H, device=device) - center_h
        w_pos = torch.arange(W, device=device) - center_w
        
        h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
        distances = torch.max(torch.abs(h_grid), torch.abs(w_grid))
        
        return distances.flatten()[:seq_len]
    
    def _compute_2d_embeddings(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 2D positional embeddings that preserve spatial relationships.
        """
        H, W = image_size
        
        # Generate 2D grid positions
        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()
        
        # Center the positions
        h_pos = h_pos - (H - 1) / 2
        w_pos = w_pos - (W - 1) / 2
        
        # Apply scaling
        if self.scaling_type != "linear":
            # Get scaled frequencies for 2D
            inv_freq_x = self._get_scaled_inv_freq_2d(h_pos, device, 'x')
            inv_freq_y = self._get_scaled_inv_freq_2d(w_pos, device, 'y')
        else:
            inv_freq_x = self.inv_freq_x.to(device) / self.scaling_factor
            inv_freq_y = self.inv_freq_y.to(device) / self.scaling_factor
        
        # Create 2D positional encodings
        h_emb = torch.outer(h_pos, inv_freq_x)
        w_emb = torch.outer(w_pos, inv_freq_y)
        
        # Combine into full positional encoding
        h_emb = h_emb.view(H, 1, -1).expand(H, W, -1)
        w_emb = w_emb.view(1, W, -1).expand(H, W, -1)
        
        # Concatenate and compute sin/cos
        pos_emb = torch.cat([h_emb, w_emb], dim=-1).flatten(0, 1)[:seq_len]
        
        # Double for sin and cos
        pos_emb = torch.cat([pos_emb, pos_emb], dim=-1)
        
        cos_emb = pos_emb.cos().to(dtype)
        sin_emb = pos_emb.sin().to(dtype)
        
        return cos_emb, sin_emb
    
    def _get_scaled_inv_freq_2d(self, positions: torch.Tensor, device: torch.device, axis: str) -> torch.Tensor:
        """Get scaled inverse frequencies for 2D encoding."""
        dim_per_axis = self.dim // 2  # Fixed: Correct divisor for 2D splitting

        if self.scaling_type == "ntk-aware":
            scaled_base = self.base * (self.scaling_factor ** (dim_per_axis / (dim_per_axis - 1)))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, dim_per_axis, dtype=torch.float32).to(device) / dim_per_axis)
            )
        elif self.scaling_type == "dynamic-ntk":
            seq_len = positions.shape[0]
            if seq_len > self.max_position_embeddings ** 0.5:  # Sqrt for 2D
                scale = seq_len / (self.max_position_embeddings ** 0.5)
                scaled_base = self.base * (scale * self.ntk_alpha) ** (dim_per_axis / (dim_per_axis - 1))
            else:
                scaled_base = self.base

            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, dim_per_axis, dtype=torch.float32).to(device) / dim_per_axis)
            )
        else:
            inv_freq = getattr(self, f"inv_freq_{axis}").to(device)

        return inv_freq
    
    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor."""
        # Ensure cos and sin have the right shape for broadcasting
        # cos, sin: (seq_len, dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)

        # Reshape input for rotary application: (batch, seq_len, dim) -> (batch, seq_len, dim/2, 2)
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # Apply rotation: [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        x_rotated = torch.stack([
            x_reshaped[..., 0] * cos[..., 0] - x_reshaped[..., 1] * sin[..., 0],
            x_reshaped[..., 0] * sin[..., 0] + x_reshaped[..., 1] * cos[..., 0]
        ], dim=-1)

        # Reshape back to original shape
        return x_rotated.flatten(2).to(x.dtype)
    
    def clear_cache(self):
        """Clear all cached embeddings and positions."""
        self._cache.clear()
        self._position_cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the current cache state."""
        return {
            "embedding_cache_size": len(self._cache),
            "position_cache_size": len(self._position_cache),
            "total_cached_tensors": len(self._cache) * 2,  # cos and sin for each
        }
    
    def extra_repr(self) -> str:
        """String representation with key parameters."""
        parts = [
            f"dim={self.dim}",
            f"max_pos={self.max_position_embeddings}",
            f"base={self.base}",
            f"scaling_type={self.scaling_type}",
            f"scaling_factor={self.scaling_factor}",
            f"use_2d={self.use_2d_encoding}",
            f"position_strategy={self.position_strategy}",
        ]
        return ", ".join(parts)


class AdaptiveImageRoPE(ImageRotaryEmbedding):
    """
    Adaptive version of ImageRotaryEmbedding that automatically adjusts
    scaling based on input resolution.
    """
    
    def __init__(
        self,
        dim: int,
        base_resolution: Tuple[int, int] = (14, 14),
        **kwargs
    ):
        super().__init__(dim, **kwargs)
        self.base_resolution = base_resolution
        self.base_res_product = base_resolution[0] * base_resolution[1]
    
    def forward(
        self,
        x: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Apply RoPE with automatic scaling adjustment based on resolution.
        """
        # Dynamically adjust scaling factor based on resolution
        current_res_product = image_size[0] * image_size[1]
        if current_res_product > self.base_res_product:
            # Automatically increase scaling for higher resolutions
            auto_scale = math.sqrt(current_res_product / self.base_res_product)
            original_factor = self.scaling_factor
            self.scaling_factor *= auto_scale
            
            result = super().forward(x, image_size)
            
            # Restore original scaling factor
            self.scaling_factor = original_factor
            return result
        
        return super().forward(x, image_size)


def create_image_rope(
    dim: int,
    resolution: Optional[Tuple[int, int]] = None,
    scaling_type: str = "linear",
    scaling_factor: float = 1.0,
    use_2d: bool = False,
    position_strategy: str = "radial",
    adaptive: bool = False,
    **kwargs
) -> Union[ImageRotaryEmbedding, AdaptiveImageRoPE]:
    """
    Factory function to create an appropriate RoPE instance.
    
    Args:
        dim: Dimension of the embeddings
        resolution: Base resolution for adaptive scaling
        scaling_type: Type of scaling ('linear', 'ntk-aware', 'dynamic-ntk')
        scaling_factor: Scaling factor for positions
        use_2d: Whether to use 2D positional encoding
        position_strategy: Strategy for position generation
        adaptive: Whether to use adaptive scaling
        **kwargs: Additional arguments
    
    Returns:
        Configured RoPE instance
    """
    rope_scaling = {
        "type": scaling_type,
        "factor": scaling_factor,
    }
    
    if adaptive and resolution is not None:
        return AdaptiveImageRoPE(
            dim=dim,
            base_resolution=resolution,
            rope_scaling=rope_scaling,
            use_2d_encoding=use_2d,
            position_strategy=position_strategy,
            **kwargs
        )
    
    return ImageRotaryEmbedding(
        dim=dim,
        rope_scaling=rope_scaling,
        use_2d_encoding=use_2d,
        position_strategy=position_strategy,
        **kwargs
    )
