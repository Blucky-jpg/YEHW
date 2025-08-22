import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union, List

# --- Import only the components actually used ---
# Core Building Block & MoE
from enhanced.enhanced_deltanet_dit_block import UltimateDeltaNetDiTBlock, fused_modulate_deltanet

# Embedding layers
from enhanced.Embeding import TimestepEmbedder, ClassEmbedder, CodeEmbedder

# TiTok tokenizer
from enhanced.titok_tokenizer import TiTokTokenizer

# Global scheduling system
from enhanced.global_scheduler import get_global_scheduler, register_default_schedules

# Core optimized layers (used within components)
from enhanced.quantization import QuantizedLinear

# Graph Flow Matching components (only imported when used)
# These are imported conditionally in the code where needed


class FlowTimeEmbedder(nn.Module):
    """
    Time embedding for Optimal Flow Matching.
    Maps continuous time t ∈ [0,1] to hidden_dim dimensional embedding.
    """
    def __init__(self, hidden_dim: int, encoding_type: str = 'sinusoidal', max_period: float = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoding_type = encoding_type
        self.max_period = max_period
        
        if encoding_type == 'learned':
            # Learned embedding with MLP
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif encoding_type == 'sinusoidal':
            # Sinusoidal position encoding adapted for continuous time
            half_dim = hidden_dim // 2
            self.register_buffer('freqs', 
                torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim))
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif encoding_type == 'fourier':
            # Random Fourier features for time encoding
            self.register_buffer('fourier_freqs', torch.randn(1, hidden_dim // 2) * 10)
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
    
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Continuous time values in [0, 1] of shape (B,)
        Returns:
            Time embeddings of shape (B, hidden_dim)
        """
        if self.encoding_type == 'learned':
            # Direct MLP encoding
            t_input = t.unsqueeze(-1)  # (B, 1)
            return self.time_mlp(t_input)
        
        elif self.encoding_type == 'sinusoidal':
            # Sinusoidal encoding similar to position encoding
            t_input = t.unsqueeze(-1)  # (B, 1)
            args = t_input * self.freqs.unsqueeze(0)  # (B, hidden_dim//2)
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, hidden_dim)
            return self.time_mlp(embedding)
        
        elif self.encoding_type == 'fourier':
            # Random Fourier features
            t_input = t.unsqueeze(-1)  # (B, 1)
            args = 2 * math.pi * t_input * self.fourier_freqs  # (B, hidden_dim//2)
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, hidden_dim)
            return self.time_mlp(embedding)
        
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


class FinalLayer(nn.Module):
    """
    The final layer of the DeltaNetDiT. It performs a final AdaLN-style
    modulation, followed by a projection to the output patch format.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = QuantizedLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            QuantizedLinear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        x = self.norm_final(x)
        x = fused_modulate_deltanet(x, shift, scale)
        x = self.proj_out(x)
        return x


# FP4/FP8 Optimization utilities
@torch.jit.script
def fused_add_scale(tensor1: torch.Tensor, tensor2: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Fused addition and scaling operation optimized for CFG (Classifier-Free Guidance).

    This function combines unconditional and conditional predictions:
    result = unconditional + scale * (conditional - unconditional)

    Args:
        tensor1: Unconditional predictions (B, C, H, W)
        tensor2: Conditional predictions (B, C, H, W)
        scale: Guidance scale factor

    Returns:
        Fused predictions for CFG
    """
    return tensor1 + scale * (tensor2 - tensor1)

@torch.jit.script
def stable_chunk(x: torch.Tensor, chunks: int, dim: int) -> List[torch.Tensor]:
    """
    Stable tensor chunking that maintains FP4/FP8 precision requirements.

    This function ensures that tensor chunks maintain proper dimensions
    for quantized computation while preserving numerical stability.

    Args:
        x: Input tensor to chunk
        chunks: Number of chunks to create
        dim: Dimension along which to chunk

    Returns:
        List of tensor chunks
    """
    return torch.chunk(x, chunks, dim)

class DeltaNetDiT(nn.Module):
    """
    A Flow Matching Transformer (DiT) model architected with the full suite of
    DeltaNet enhancements. This model integrates multi-path attention,
    dual-bank Mixture-of-Experts, progressive pruning, and context-adaptive
    gating, all coordinated by a global scheduler.
    
    Enhanced with Qwen-Image innovations:
    - Multimodal Scalable RoPE (MS-RoPE) for better positional encoding
    - Double-stream MMDiT architecture for text-image interaction
    - QK-Norm with RMSNorm for improved stability
    - Center-outward image encoding and diagonal text positioning
    
    Uses Optimal Flow Matching (OFM) with advanced improvements:
    - Predicts both velocity fields v(x,t) AND clean image x₁ for better coherence
    - Time t is continuous in [0,1] where 0=noise and 1=data
    - Supports Min-SNR-γ weighting for better training dynamics
    - Provides superior sampling quality and prompt adherence

    Returns the predictions (velocity and optionally x₁) and auxiliary losses.
    """
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        # MoE specific arguments
        moe_num_experts: int = 8,
        moe_low_noise_experts: Optional[int] = None,  # e.g., 3 for early flow (t~0)
        moe_high_noise_experts: Optional[int] = None, # e.g., 7 for late flow (t~1)
        moe_top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_jitter_noise: float = 0.01,
        moe_lb_loss_coef: float = 0.01,
        moe_share_low_high: bool = False,
        # GroveMoE specific arguments
        use_grove_moe: bool = False,  # Use GroveMoE instead of standard MoE
        grove_num_groups: int = 64,  # Number of expert groups
        grove_adjugate_size: int = 128,  # Adjugate expert intermediate size
        grove_scaling_factor: float = 0.05,  # Scaling factor for adjugate experts
        # Attention specific arguments
        attn_dropout: float = 0.0,
        attn_snr_threshold: float = 0.5,
        # OFM specific arguments
        use_flow_matching: bool = True,  # Enable OFM mode
        flow_time_encoding: str = 'sinusoidal',  # Time encoding for flow
        # Advanced OFM improvements
        predict_x1: bool = True,  # Also predict clean image x₁ alongside velocity
        use_min_snr_gamma: bool = True,  # Use Min-SNR-γ loss weighting
        min_snr_gamma: float = 5.0,  # γ parameter for Min-SNR weighting
        x1_loss_weight: float = 0.1,  # Weight for x₁ prediction loss
        # TiTok specific arguments
        use_titok: bool = False,  # Enable TiTok 1D tokenization
        titok_num_tokens: int = 32,  # Number of TiTok tokens (32, 64, 128)
        titok_codebook_size: int = 4096,  # TiTok codebook size
        titok_code_dim: int = 16,  # TiTok code dimension
        titok_model_size: str = 'base',  # 'small', 'base', 'large'
        titok_pretrained: Optional[str] = None,  # Path to pretrained TiTok model
        # Graph Flow Matching (GFM) arguments
        use_gfm: bool = False,  # Enable Graph Flow Matching
        gfm_variant: str = 'mpnn',  # 'mpnn' or 'gps'
        gfm_hidden_ratio: float = 0.05,  # Ratio of hidden size for GFM (max 5% params)
        gfm_num_layers: int = 3,  # Number of GFM layers
        gfm_graph_neighbors: int = 8,  # Number of neighbors in dynamic graph
        gfm_reg_weight: float = 0.01,  # Regularization weight for v_diff
    ):
        super().__init__()
        self.use_flow_matching = use_flow_matching
        self.flow_time_encoding = flow_time_encoding
        self.predict_x1 = predict_x1
        self.use_min_snr_gamma = use_min_snr_gamma
        self.min_snr_gamma = min_snr_gamma
        self.x1_loss_weight = x1_loss_weight

        # TiTok parameters
        self.use_titok = use_titok
        self.titok_num_tokens = titok_num_tokens
        self.titok_codebook_size = titok_codebook_size
        self.titok_code_dim = titok_code_dim
        self.titok_model_size = titok_model_size
        self.titok_pretrained = titok_pretrained
        
        # Graph Flow Matching parameters
        self.use_gfm = use_gfm
        self.gfm_variant = gfm_variant
        self.gfm_hidden_ratio = gfm_hidden_ratio
        self.gfm_num_layers = gfm_num_layers
        self.gfm_graph_neighbors = gfm_graph_neighbors
        self.gfm_reg_weight = gfm_reg_weight
        
        self.in_channels = in_channels
        # Output channels depend on prediction mode
        if use_flow_matching and predict_x1:
            # Predict both velocity (in_channels) and x₁ (in_channels)
            self.out_channels = in_channels * 2
        else:
            # Standard: just velocity or noise
            self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob
        self.num_patches = (input_size // patch_size) ** 2
        self.input_size = input_size

        # --- Initialize Global Scheduler and Register Default Schedules ---
        # This is a critical step, as many sub-modules rely on it.
        scheduler = get_global_scheduler()
        register_default_schedules(scheduler)

        # Initialize graph statistics tracking
        self.graph_stats = None

        # --- Input & Embedding Layers ---
        if self.use_titok:
            # TiTok tokenizer for 1D tokenization
            self.titok_tokenizer = TiTokTokenizer(
                input_size=input_size,
                patch_size=16,  # Standard for TiTok
                num_tokens=self.titok_num_tokens,
                codebook_size=self.titok_codebook_size,
                code_dim=self.titok_code_dim,
                hidden_size=768,  # Standard ViT hidden size
                num_layers=12,  # Standard ViT depth
                num_heads=12,  # Standard ViT heads
                use_fp16=True,
                use_decoder=False,  # Decoder not needed for training
            )

            # Code embedder for discrete tokens
            self.code_embedder = CodeEmbedder(
                codebook_size=self.titok_codebook_size,
                hidden_size=hidden_size,
            )

            # Update num_patches for 1D tokens
            self.num_patches = self.titok_num_tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        else:
            # Standard 2D patch embedding
            self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # For OFM, we use continuous time in [0,1] instead of discrete timesteps
        if self.use_flow_matching:
            # Enhanced time embedding for flow matching
            self.t_embedder = FlowTimeEmbedder(hidden_size, encoding_type=flow_time_encoding)
        else:
            # Original timestep embedder for standard diffusion
            self.t_embedder = TimestepEmbedder(hidden_size)

        # Conditional embeddings (with support for classifier-free guidance)
        self.num_classes = num_classes
        if num_classes > 0:
            self.y_embedder = ClassEmbedder(num_classes, hidden_size, dropout=0.1)
            # Learned embedding for the unconditional case (when y is dropped)
            self.null_class_embed = nn.Parameter(torch.randn(1, hidden_size))

        # Choose MoE class based on configuration
        if use_grove_moe:
            from enhanced.grove_moe import GroveMoE
            # Use GroveMoE with adjugate experts
            moe_cls = GroveMoE
            moe_kwargs = {
                'hidden_size': hidden_size,
                'num_experts': moe_num_experts,
                'num_groups': grove_num_groups,
                'adjugate_intermediate_size': grove_adjugate_size,
                'top_k': moe_top_k,
                'capacity_factor': moe_capacity_factor,
                'jitter_noise': moe_jitter_noise,
                'load_balance_loss_coef': moe_lb_loss_coef,
                'scaling_factor': grove_scaling_factor,
                'dropout': attn_dropout,
                'use_dual_bank': True,
                'low_noise_experts_num': moe_low_noise_experts,
                'high_noise_experts_num': moe_high_noise_experts,
                'share_low_with_high': moe_share_low_high,
            }
        else:
            # Use standard DeltaNetEnhancedMoE
            from enhanced.enhanced_deltanet_dit_block import DeltaNetEnhancedMoE
            moe_cls = DeltaNetEnhancedMoE
            moe_kwargs = {
                'num_experts': moe_num_experts,
                'top_k': moe_top_k,
                'capacity_factor': moe_capacity_factor,
                'jitter_noise': moe_jitter_noise,
                'load_balance_loss_coef': moe_lb_loss_coef,
                'low_noise_experts_num': moe_low_noise_experts,
                'high_noise_experts_num': moe_high_noise_experts,
                'share_low_with_high': moe_share_low_high,
            }
        
        self.blocks = nn.ModuleList([
            UltimateDeltaNetDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                moe_cls=moe_cls,
                dropout=attn_dropout,
                snr_threshold=attn_snr_threshold,
                **moe_kwargs
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # --- Graph Flow Matching Module ---
        if self.use_gfm:
            # Import GFM components only when needed
            from enhanced.graph_builder import build_dynamic_graph, GraphStatistics
            from enhanced.gfm_mpnn import MPNNGraphCorrection
            from enhanced.gfm_gps import GPSGraphCorrection

            if gfm_variant == 'mpnn':
                self.gfm_module = MPNNGraphCorrection(
                    hidden_size=hidden_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    gfm_hidden_ratio=gfm_hidden_ratio,
                    num_layers=gfm_num_layers,
                    dropout=attn_dropout,
                )
            elif gfm_variant == 'gps':
                self.gfm_module = GPSGraphCorrection(
                    hidden_size=hidden_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    gfm_hidden_ratio=gfm_hidden_ratio,
                    num_layers=gfm_num_layers,
                    num_heads=4,  # Fixed for GPS
                    dropout=attn_dropout,
                )
            else:
                raise ValueError(f"Unknown GFM variant: {gfm_variant}")

            # Set base model params for budget tracking
            base_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.gfm_module.set_base_model_params(base_params)

        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding with a truncated normal distribution (if not using MS-RoPE)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        
        # Zero-out the final projection layer as per DiT paper recommendations.
        # This helps stabilize training in the beginning.
        nn.init.constant_(self.final_layer.proj_out.weight, 0)
        nn.init.constant_(self.final_layer.proj_out.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, P * P * C_out)
        return: (B, C_out, H, W)
        """
        B, N, _ = x.shape
        H = W = int(N ** 0.5)
        P = self.patch_size
        C_out = self.out_channels

        # Use contiguous for better memory access
        x = x.reshape(B, H, W, P, P, C_out)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # (B, C_out, H, P, W, P)
        return x.reshape(B, C_out, H * P, W * P)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass of the DeltaNetDiT model with OFM.

        Args:
            x (torch.Tensor): Input latent tensor (B, C, H, W).
            t (torch.Tensor): For OFM: continuous time in [0,1] (B,)
                             For diffusion: timestep indices (B,)
            y (torch.Tensor, optional): Class labels (B,).
            return_dict (bool): If True, return a dictionary with all outputs.

        Returns:
            If return_dict=False:
                Tuple[torch.Tensor, torch.Tensor]:
                    - The model output (velocity or combined v+x₁) (B, C_out, H, W)
                    - The total auxiliary loss (a scalar tensor).
            If return_dict=True:
                Dict with keys: 'output', 'velocity', 'x1_pred', 'aux_loss', 'snr_weight'
        """
        # Validate inputs
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if t.dim() != 1:
            raise ValueError(f"Expected 1D time tensor (B,), got {t.dim()}D")
        if x.shape[0] != t.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]}, t has {t.shape[0]}")
        if y is not None:
            if y.dim() != 1:
                raise ValueError(f"Expected 1D label tensor (B,), got {y.dim()}D")
            if y.shape[0] != x.shape[0]:
                raise ValueError(f"Batch size mismatch: x has {x.shape[0]}, y has {y.shape[0]}")
            if self.num_classes == 0:
                raise ValueError("Model was initialized with num_classes=0 but labels were provided")
        
        # 1. Input Processing and Embedding
        if self.use_titok:
            # TiTok 1D tokenization
            # x is expected to be images (B, 3, H, W)
            tokens, quantized, commit_loss = self.titok_tokenizer.encode(x)  # (B, num_tokens), (B, num_tokens, code_dim), scalar
            x = self.code_embedder(tokens)  # (B, num_tokens, hidden_size)
            x = x + self.pos_embed  # Add positional embedding
            # Add VQ commitment loss to auxiliary losses
            total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            total_aux_loss = total_aux_loss + 0.25 * commit_loss  # Weight the VQ loss
        else:
            # Standard 2D patch embedding
            x = self.x_embedder(x).flatten(2).transpose(1, 2)  # (B, N, C)
            x = x + self.pos_embed

        # 2. Timestep and Class Embeddings
        t_emb = self.t_embedder(t)  # (B, C)
        
        # Classifier-free guidance logic
        if self.num_classes > 0:
            if y is None:
                # Default to unconditional if no labels provided
                y_emb = self.null_class_embed.expand(x.shape[0], -1).to(dtype=x.dtype)
            else:
                # Randomly drop classes for CFG
                y_mask = (torch.rand(y.shape[0], device=y.device) > self.class_dropout_prob)
                y_emb = self.y_embedder(y)
                # Use null embedding for dropped classes (ensure dtype match)
                y_emb[~y_mask] = self.null_class_embed.to(dtype=y_emb.dtype)
            temb = t_emb + y_emb
        else: # Unconditional model
            temb = t_emb
        
        # Initialize aux_loss if not using TiTok (TiTok already initialized it)
        if not self.use_titok:
            total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, aux_loss = block(x, temb, t)
            total_aux_loss += aux_loss

        # 4. Apply Graph Flow Matching correction if enabled
        x_patches = x.clone()  # Save patches for GFM
        
        # Final Layer and initial unpatching
        x = self.final_layer(x, temb)
        
        # Apply GFM correction before unpatchifying
        if self.use_gfm and self.training:
            # Import GFM components only when needed
            from enhanced.graph_builder import build_dynamic_graph

            # Build dynamic graph from features (using k-NN as fallback)
            # Note: For production, you'd want to extract attention maps from the last block
            edge_index, edge_weight = build_dynamic_graph(
                features=x_patches,
                top_k=self.gfm_graph_neighbors,
                batch_size=x_patches.shape[0],
                symmetric=True,
            )

            # Get velocity correction from GFM
            v_diff = self.gfm_module(
                x_patches=x_patches,
                edge_index=edge_index,
                edge_weight=edge_weight,
                t=t,
            )

            # Add correction to base velocity
            x = x + v_diff

            # Track graph statistics
            if self.graph_stats is not None:
                self.graph_stats.update(edge_index, edge_weight, x_patches.shape[0] * x_patches.shape[1])
        
        # Unpatchify to image format
        x = self.unpatchify(x)
        
        # 5. Handle combined prediction if enabled
        if self.use_flow_matching and self.predict_x1:
            # Split output into velocity and x₁ predictions
            velocity, x1_pred = x.chunk(2, dim=1)
            
            if return_dict:
                # Compute SNR weight if needed
                snr_weight = self._compute_snr_weight(t) if self.use_min_snr_gamma else None
                return {
                    'output': x,  # Full output
                    'velocity': velocity,
                    'x1_pred': x1_pred,
                    'aux_loss': total_aux_loss,
                    'snr_weight': snr_weight
                }
            else:
                # For backward compatibility, return full output
                return x, total_aux_loss
        else:
            # Standard mode: just return the output
            if return_dict:
                snr_weight = self._compute_snr_weight(t) if self.use_min_snr_gamma else None
                return {
                    'output': x,
                    'velocity': x,  # In standard mode, output is velocity
                    'x1_pred': None,
                    'aux_loss': total_aux_loss,
                    'snr_weight': snr_weight
                }
            else:
                return x, total_aux_loss

    
    def _compute_snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Min-SNR-γ weight for loss weighting.

        Args:
            t: Time values in [0, 1]

        Returns:
            SNR-based loss weights
        """
        # Convert OFM time to approximate SNR
        # In OFM: t=0 is pure noise (low SNR), t=1 is clean (high SNR)
        # For linear flow, we need a more accurate SNR calculation

        # Clamp t to avoid division issues near boundaries
        t_safe = torch.clamp(t, min=1e-8, max=1.0 - 1e-8)

        # For linear flow: x_t = (1-t)*x_0 + t*x_1
        # The SNR can be approximated as: SNR(t) = (t/(1-t))^2 for linear flow
        # This gives higher SNR for larger t (cleaner samples)
        snr = (t_safe / (1 - t_safe)) ** 2

        # Apply Min-SNR-γ clipping: min(SNR, γ)
        snr_clipped = torch.minimum(snr, torch.ones_like(snr) * self.min_snr_gamma)

        # Weight is proportional to 1/SNR for balancing
        # Higher weight for noisy samples (low SNR), lower weight for clean samples (high SNR)
        weight = 1.0 / (snr_clipped + 1e-8)  # Add small epsilon to avoid division by zero

        return weight
    
    def compute_training_losses(self, x0: torch.Tensor, x1: torch.Tensor, 
                                t: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for OFM with all improvements.
        
        Args:
            x0: Noise samples (B, C, H, W)
            x1: Clean data samples (B, C, H, W)
            t: Time values in [0, 1] (B,)
            y: Optional class labels (B,)
        
        Returns:
            Dictionary with loss components
        """
        # Interpolate between x0 and x1
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # True velocity for OFM
        v_true = x1 - x0
        
        # Forward pass with dict output
        outputs = self.forward(x_t, t, y, return_dict=True)
        
        # Velocity matching loss
        v_pred = outputs['velocity']
        velocity_loss = F.mse_loss(v_pred, v_true, reduction='none')
        velocity_loss = velocity_loss.mean(dim=[1, 2, 3])  # Per-sample loss
        
        # Apply Min-SNR-γ weighting if enabled
        if self.use_min_snr_gamma:
            snr_weight = outputs['snr_weight']
            velocity_loss = velocity_loss * snr_weight
        
        velocity_loss = velocity_loss.mean()
        
        total_loss = velocity_loss
        
        # x₁ prediction loss if enabled
        if self.predict_x1 and outputs['x1_pred'] is not None:
            x1_pred = outputs['x1_pred']
            x1_loss = F.mse_loss(x1_pred, x1, reduction='mean')
            total_loss = total_loss + self.x1_loss_weight * x1_loss
        else:
            x1_loss = torch.tensor(0.0, device=x1.device)
        
        # Add auxiliary losses
        total_loss = total_loss + outputs['aux_loss']
        
        return {
            'total_loss': total_loss,
            'velocity_loss': velocity_loss,
            'x1_loss': x1_loss,
            'aux_loss': outputs['aux_loss'],
            'snr_weight': outputs.get('snr_weight', torch.tensor(1.0))
        }
