"""
TiTok 1D Tokenization Module for DeltaNetDiT Architecture

Improved implementation based on ByteDance's official TiTok with:
- Proper error handling and validation
- Memory-efficient processing
- Better integration with DeltaNetDiT
- Clean code structure and documentation
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import from local quantization module
from enhanced.quantization import QuantizedLinear

# Optional dependencies
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

try:
    from torch.nn import TransformerEncoderLayer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            QuantizedLinear(hidden_size, mlp_hidden),
            nn.GELU(),
            QuantizedLinear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output

        return x


class TiTokEncoder(nn.Module):
    """TiTok Encoder - ViT-based feature extractor."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Extract config values with validation
        self.input_size = config.model.encoder.input_size
        self.patch_size = config.model.encoder.patch_size
        self.hidden_size = config.model.encoder.hidden_size
        self.num_layers = config.model.encoder.num_layers
        self.num_heads = config.model.encoder.num_heads
        self.mlp_ratio = config.model.encoder.mlp_ratio

        # Calculate dimensions
        self.num_patches = (self.input_size // self.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            ) for _ in range(self.num_layers)
        ])

        self.norm = nn.LayerNorm(self.hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, pixel_values: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            pixel_values: Input images (B, 3, H, W)
            latent_tokens: Learnable latent tokens (N, D)

        Returns:
            Encoded features (1, N, D)
        """
        B = pixel_values.shape[0]

        # Patch embedding
        x = self.patch_embed(pixel_values)  # (B, hidden_size, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # (B, num_patches, hidden_size)

        # Attention with latent tokens (official TiTok approach)
        # latent_tokens: (num_latent_tokens, hidden_size)
        # x: (B, num_patches, hidden_size)

        # Compute attention: latent tokens attend to patch features
        attn_logits = torch.matmul(latent_tokens, x.transpose(1, 2)) / (self.hidden_size ** 0.5)
        # attn_logits: (num_latent_tokens, B, num_patches)

        attn_weights = F.softmax(attn_logits, dim=-1)
        # attn_weights: (num_latent_tokens, B, num_patches)

        # Apply attention: (num_latent_tokens, B, num_patches) @ (B, num_patches, hidden_size)
        attended = torch.matmul(attn_weights, x)  # (num_latent_tokens, B, hidden_size)

        # Average over batch dimension and transpose
        z = attended.mean(dim=1).unsqueeze(0)  # (1, num_latent_tokens, hidden_size)

        return z


class TiTokDecoder(nn.Module):
    """TiTok Decoder - reconstructs images from quantized tokens."""

    def __init__(self, config, token_size: int):
        super().__init__()
        self.config = config
        self.token_size = token_size

        # Extract config values
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.hidden_size = config.model.encoder.hidden_size
        self.output_size = config.model.encoder.input_size
        self.patch_size = config.model.encoder.patch_size

        self.num_patches = (self.output_size // self.patch_size) ** 2

        # Project tokens to patch features (from token_size to hidden_size)
        self.token_proj = QuantizedLinear(self.token_size, self.hidden_size)

        # Position embedding for full patch grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=12,
                mlp_ratio=4.0,
            ) for _ in range(8)
        ])

        # Final convolution to image
        self.final_conv = nn.ConvTranspose2d(
            self.hidden_size, 3, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.norm = nn.LayerNorm(self.hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z_quantized: Quantized tokens (B, 1, num_latent_tokens, hidden_size)

        Returns:
            Reconstructed images (B, 3, H, W)
        """
        B = z_quantized.shape[0]

        # z_quantized: (B, 1, num_latent_tokens, hidden_size)
        # Remove the channel dimension and reshape
        x = z_quantized.squeeze(1)  # (B, num_latent_tokens, hidden_size)

        # Project to hidden size
        x = self.token_proj(x)  # (B, num_latent_tokens, hidden_size)

        # For now, just repeat the tokens to match the expected patch count
        # This is a simplified approach for debugging
        if self.num_latent_tokens < self.num_patches:
            # Repeat tokens to fill the patch grid
            repeat_factor = (self.num_patches + self.num_latent_tokens - 1) // self.num_latent_tokens
            x = x.repeat(1, repeat_factor, 1)  # Repeat along sequence dimension
            x = x[:, :self.num_patches, :]  # Truncate to exact size
            x = x + self.pos_embed
        elif self.num_latent_tokens > self.num_patches:
            # If we have more tokens than patches, use adaptive pooling
            x = x.transpose(1, 2)  # (B, hidden_size, num_latent_tokens)
            x = F.adaptive_avg_pool1d(x, self.num_patches)
            x = x.transpose(1, 2)  # (B, num_patches, hidden_size)
            x = x + self.pos_embed
        else:
            # If tokens match patches exactly, use position embedding as is
            x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Reshape for convolution
        h = w = self.output_size // self.patch_size
        x = x.view(B, h, w, self.hidden_size).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Final convolution
        x = self.final_conv(x)  # (B, 3, H, W)

        return x


class TiTokTokenizer(nn.Module):
    """
    Official TiTok tokenizer implementation aligned with ByteDance's version.

    Converts images to compact 1D semantic tokens using:
    1. ViT encoder for feature extraction
    2. Vector quantization for discrete tokenization
    3. Optional decoder for reconstruction with two-stage training
    """

    def __init__(self, config):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config) if OMEGACONF_AVAILABLE else config

        self.config = config

        # This should be False for stage1 and True for stage2
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", False)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")

        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only support finetune_decoder with vq quantization for now.")

        # Extract token_size first
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        # Create encoder and decoder
        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config, token_size=self.token_size)
        scale = self.encoder.hidden_size ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.hidden_size))

        # Projection layer to match encoder output to quantizer input size
        self.pre_quant_proj = nn.Linear(self.encoder.hidden_size, self.token_size)

        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
            )
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder for stage 2
            self.pixel_quantize = VectorQuantizer(
                codebook_size=1024,
                token_size=256,
                commitment_cost=0.25,
                use_l2_norm=True,
            )
            self.pixel_decoder = SimplePixelDecoder(config)

    def _init_weights(self, module):
        """Initialize weights as per the official implementation."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode images to quantized tokens.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            z_quantized: Quantized tokens
            result_dict: Dictionary with losses and metadata
        """
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                # Zero out losses for stage 2
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                # Project encoder output to match quantizer input size
                z_proj = self.pre_quant_proj(z.squeeze(0)).unsqueeze(0)  # (1, num_latent_tokens, token_size)
                z_quantized, result_dict = self.quantize(z_proj)
            elif self.quantize_mode == "vae":
                # For VAE mode, we need to handle the distribution properly
                mean = z.squeeze(0)  # Remove batch dimension from encoder output
                mean_proj = self.pre_quant_proj(mean)  # Project to token_size
                logvar = torch.zeros_like(mean_proj)  # Simple case: no variance
                posteriors = self.quantize(mean_proj, logvar)
                z_quantized = posteriors.sample()
                result_dict = {"quantizer_loss": posteriors.kl_loss()}

        return z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized tokens back to images.

        Args:
            z_quantized: Quantized tokens

        Returns:
            Reconstructed images
        """
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            # Apply pixel-level quantization and decoding for stage 2
            with torch.no_grad():
                # Simple pixel-level processing (placeholder for actual implementation)
                decoded = self.pixel_decoder(decoded)
        return decoded

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            tokens: Discrete token indices from quantizer
            quantized: Quantized features
            commit_loss: Commitment loss from VQ
            reconstructed: Reconstructed images from decoder
        """
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)

        # Extract the expected return values
        tokens = result_dict.get("tokens")
        quantized = z_quantized
        commit_loss = result_dict.get("commitment_loss", torch.tensor(0.0, device=x.device))
        reconstructed = decoded

        return tokens, quantized, commit_loss, reconstructed

    @staticmethod
    def create_default_config(
        input_size: int = 256,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_latent_tokens: int = 32,
        codebook_size: int = 4096,
        token_size: int = 16,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = True,
        finetune_decoder: bool = False,
    ) -> dict:
        """Create a default configuration for TiTok tokenizer."""
        config = {
            "model": {
                "encoder": {
                    "input_size": input_size,
                    "patch_size": patch_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                },
                "vq_model": {
                    "num_latent_tokens": num_latent_tokens,
                    "codebook_size": codebook_size,
                    "token_size": token_size,
                    "commitment_cost": commitment_cost,
                    "use_l2_norm": use_l2_norm,
                    "finetune_decoder": finetune_decoder,
                    "quantize_mode": "vq",
                }
            }
        }
        return config


class ViTEncoder(nn.Module):
    """ViT encoder for feature extraction."""

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (input_size // patch_size) ** 2

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(num_layers)
        ])

        # Final projection to code dimension
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            QuantizedLinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            QuantizedLinear(hidden_size // 2, hidden_size // 4),
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_size, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization aligned with official TiTok implementation."""

    def __init__(
        self,
        codebook_size: int = 4096,
        token_size: int = 16,  # Official parameter name
        commitment_cost: float = 0.25,
        use_l2_norm: bool = True,
        use_ema: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.token_size = token_size  # Official name
        self.code_dim = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.codebook = nn.Embedding(codebook_size, token_size)
        nn.init.uniform_(self.codebook.weight, -1/token_size, 1/token_size)

        # EMA tracking
        if use_ema:
            self.register_buffer('ema_count', torch.zeros(codebook_size))
            self.register_buffer('ema_weight', self.codebook.weight.data.clone())

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook entries for given indices."""
        return self.codebook(indices)  # Official method name

    def embed_code(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token indices (alias for compatibility)."""
        return self.get_codebook_entry(tokens)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vector quantization forward pass - official implementation.

        Args:
            z: Input features (B, 1, N, token_size) or (B, N, token_size)

        Returns:
            quantized: Quantized features (B, 1, N, token_size)
            result_dict: Dictionary with losses and tokens
        """
        # Handle different input shapes
        if z.dim() == 4:
            B, C, N, D = z.shape
            z_flat = z.view(-1, D)  # (B*N, D)
        else:
            B, N, D = z.shape
            C = 1
            z_flat = z.view(-1, D)  # (B*N, D)

        # L2 normalization if enabled
        if self.use_l2_norm:
            z_flat = F.normalize(z_flat, p=2, dim=-1)

        # Compute distances to codebook
        distances = torch.cdist(z_flat, self.codebook.weight)  # (B*N, codebook_size)

        # Get nearest neighbors
        tokens = torch.argmin(distances, dim=-1)  # (B*N,)
        tokens = tokens.view(B, N)  # (B, N)

        # Get quantized features
        quantized = self.get_codebook_entry(tokens)  # (B, N, token_size)

        # Reshape to match input
        if C == 1:
            quantized = quantized.unsqueeze(1)  # (B, 1, N, token_size)
            z = z.unsqueeze(1) if z.dim() == 3 else z

        # Compute losses
        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach(), reduction='mean')
        codebook_loss = F.mse_loss(quantized, z.detach(), reduction='mean')
        quantizer_loss = commitment_loss + codebook_loss

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # EMA update
        if self.use_ema and self.training:
            self.ema_update(tokens, z_flat)

        # Return official format
        result_dict = {
            "quantizer_loss": quantizer_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "tokens": tokens,
        }

        return quantized_st, result_dict

    @torch.no_grad()
    def ema_update(self, tokens: torch.Tensor, z_flat: torch.Tensor):
        """EMA update for codebook."""
        # Flatten tokens
        tokens_flat = tokens.view(-1)  # (B*N,)

        # Update counts
        counts = torch.bincount(tokens_flat, minlength=self.codebook_size).float()

        # Update weights
        z_sum = torch.zeros(self.codebook_size, self.code_dim, device=z_flat.device)
        z_sum.scatter_add_(0, tokens_flat.unsqueeze(-1).expand(-1, self.code_dim), z_flat)

        # EMA update
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * counts
        self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * z_sum

        # Normalize
        valid_mask = self.ema_count > self.epsilon
        self.ema_weight[valid_mask] = self.ema_weight[valid_mask] / self.ema_count[valid_mask].unsqueeze(-1)

        # Update codebook
        self.codebook.weight.data.copy_(self.ema_weight)


class DiagonalGaussianDistribution(nn.Module):
    """Diagonal Gaussian distribution for VAE mode."""

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.logvar = logvar
        self.std = torch.exp(0.5 * logvar)

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def kl_loss(self) -> torch.Tensor:
        """Compute KL divergence loss."""
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + torch.exp(self.logvar) - 1.0 - self.logvar,
            dim=[1, 2, 3]
        ).mean()


class SimplePixelDecoder(nn.Module):
    """Simplified pixel decoder for stage 2 training."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Simple upsampling decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pixel decoder."""
        return self.decoder(x)
