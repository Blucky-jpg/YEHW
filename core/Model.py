import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union, List
import numpy as np

# --- Import all of your custom DeltaNet components ---
# Core Building Block & MoE
from enhanced.enhanced_deltanet_dit_block import UltimateDeltaNetDiTBlock, DeltaNetEnhancedMoE, fused_modulate_deltanet

# Embedding layers
from enhanced.Embeding import TimestepEmbedder, ClassEmbedder

# Global scheduling system
from enhanced.global_scheduler import get_global_scheduler, register_default_schedules

# Core optimized layers (used within components)
from core.quantization import BlackwellOptimizedLinear


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
        self.proj_out = BlackwellOptimizedLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            BlackwellOptimizedLinear(hidden_size, 2 * hidden_size, bias=True)
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
    """Fused addition and scaling for CFG."""
    return tensor1 + scale * (tensor2 - tensor1)

@torch.jit.script
def stable_chunk(x: torch.Tensor, chunks: int, dim: int) -> List[torch.Tensor]:
    """Stable chunking that maintains FP4 precision."""
    return torch.chunk(x, chunks, dim)

class DeltaNetDiT(nn.Module):
    """
    A Flow Matching Transformer (DiT) model architected with the full suite of
    DeltaNet enhancements. This model integrates multi-path attention,
    dual-bank Mixture-of-Experts, progressive pruning, and context-adaptive
    gating, all coordinated by a global scheduler.
    
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
        learn_sigma: bool = False,  # Not used in OFM, kept for compatibility
        # MoE specific arguments
        moe_num_experts: int = 8,
        moe_low_noise_experts: Optional[int] = None,  # e.g., 3 for early flow (t~0)
        moe_high_noise_experts: Optional[int] = None, # e.g., 7 for late flow (t~1)
        moe_top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_jitter_noise: float = 0.01,
        moe_lb_loss_coef: float = 0.01,
        moe_share_low_high: bool = False,
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
    ):
        super().__init__()
        self.use_flow_matching = use_flow_matching
        self.flow_time_encoding = flow_time_encoding
        self.predict_x1 = predict_x1
        self.use_min_snr_gamma = use_min_snr_gamma
        self.min_snr_gamma = min_snr_gamma
        self.x1_loss_weight = x1_loss_weight
        self.learn_sigma = learn_sigma  # Kept for compatibility but not used in OFM
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

        # --- Initialize Global Scheduler and Register Default Schedules ---
        # This is a critical step, as many sub-modules rely on it.
        scheduler = get_global_scheduler()
        register_default_schedules(scheduler)

        # --- Input & Embedding Layers ---
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        
        # For OFM, we use continuous time in [0,1] instead of discrete timesteps
        if self.use_flow_matching:
            # Enhanced time embedding for flow matching
            self.t_embedder = FlowTimeEmbedder(hidden_size, encoding_type=flow_time_encoding)
        else:
            # Original timestep embedder for standard diffusion
            self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Conditional embeddings (with support for classifier-free guidance)
        self.num_classes = num_classes
        if num_classes > 0:
            self.y_embedder = ClassEmbedder(num_classes, hidden_size, dropout=0.1)
            # Learned embedding for the unconditional case (when y is dropped)
            self.null_class_embed = nn.Parameter(torch.randn(1, hidden_size))


        # --- Core Transformer Blocks ---
        # This is the main body of the model, a stack of the custom blocks.
        self.blocks = nn.ModuleList([
            UltimateDeltaNetDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                moe_cls=DeltaNetEnhancedMoE, # Pass the MoE class directly
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                capacity_factor=moe_capacity_factor,
                jitter_noise=moe_jitter_noise,
                lb_loss_coef=moe_lb_loss_coef,
                low_noise_experts_num=moe_low_noise_experts,
                high_noise_experts_num=moe_high_noise_experts,
                share_low_with_high=moe_share_low_high,
                dropout=attn_dropout,
                snr_threshold=attn_snr_threshold,
            ) for _ in range(depth)
        ])

        # --- Final Output Layer ---
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding with a truncated normal distribution
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Zero-out the final projection layer as per DiT paper recommendations.
        # This helps stabilize training in the beginning.
        nn.init.constant_(self.final_layer.proj_out.weight, 0)
        nn.init.constant_(self.final_layer.proj_out.bias, 0)

    @torch.jit.script_method
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
        return_dict: bool = False,
        use_amp: bool = True  # Auto Mixed Precision for FP4/FP8
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
        
        # 1. Input Patching and Embedding
        x = self.x_embedder(x).flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_embed

        # 2. Timestep and Class Embeddings
        t_emb = self.t_embedder(t)  # (B, C)
        
        # Classifier-free guidance logic
        if self.num_classes > 0:
            if y is None:
                # Default to unconditional if no labels provided
                y_emb = self.null_class_embed.expand(x.shape[0], -1)
            else:
                # Randomly drop classes for CFG
                y_mask = (torch.rand(y.shape[0], device=y.device) > self.class_dropout_prob)
                y_emb = self.y_embedder(y)
                # Use null embedding for dropped classes
                y_emb[~y_mask] = self.null_class_embed
            temb = t_emb + y_emb
        else: # Unconditional model
            temb = t_emb

        # 3. Main Transformer Blocks with optional AMP
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Use bfloat16 autocast for better FP4/FP8 compatibility
        if use_amp and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for block in self.blocks:
                    x, aux_loss = block(x, temb, t)
                    total_aux_loss += aux_loss
        else:
            for block in self.blocks:
                x, aux_loss = block(x, temb, t)
                total_aux_loss += aux_loss

        # 4. Final Layer and Unpatching
        x = self.final_layer(x, temb)
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

    @torch.no_grad()
    @torch.jit.script_if_tracing
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Perform classifier-free guidance sampling.
        Works for both OFM (velocity prediction) and diffusion (noise prediction).
        
        Args:
            x: Input tensor (B, C, H, W)
            t: Time values (B,)
            y: Class labels (B,) - optional
            cfg_scale: Guidance scale (1.0 = no guidance)
        
        Returns:
            Guided output tensor
        """
        if cfg_scale == 1.0 or self.num_classes == 0:
            # No guidance, just run the model once
            if self.use_flow_matching and self.predict_x1:
                outputs = self.forward(x, t, y, return_dict=True)
                return outputs['output']
            else:
                model_out, _ = self.forward(x, t, y)
                return model_out
        
        # Prepare batch for conditional and unconditional forward passes
        batch_size = x.shape[0]
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        
        # Optimize by batching conditional and unconditional together
        if y is not None:
            # Stack inputs for single forward pass
            x_batch = torch.cat([x, x], dim=0)
            t_batch = torch.cat([t, t], dim=0)
            
            # Create batch with conditional (first half) and unconditional (second half)
            # We'll modify forward to handle this efficiently
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(x.device.type == 'cuda')):
                # Process both in single forward pass
                output_batch, _ = self._forward_cfg_batch(x_batch, t_batch, y)
            
            # Split results
            cond_out, uncond_out = output_batch.chunk(2, dim=0)
            
            # Apply CFG with fused operation
            return uncond_out + cfg_scale * (cond_out - uncond_out)
        else:
            # If no labels provided, just return unconditional
            model_out, _ = self.forward(x, t, None)
            return model_out
    
    def _forward_cfg_batch(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized batch forward for CFG that processes conditional and unconditional together.
        First half of batch is conditional, second half is unconditional.
        """
        batch_size = x.shape[0] // 2
        
        # Embedding stage
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Time embedding for full batch
        t_emb = self.t_embedder(t)
        
        # Split for conditional and unconditional
        if self.num_classes > 0:
            # First half: conditional with labels
            y_emb_cond = self.y_embedder(y)
            # Second half: unconditional
            y_emb_uncond = self.null_class_embed.expand(batch_size, -1)
            # Combine
            y_emb = torch.cat([y_emb_cond, y_emb_uncond], dim=0)
            temb = t_emb + y_emb
        else:
            temb = t_emb
        
        # Process through blocks with bfloat16 for FP4 compatibility
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(x.device.type == 'cuda')):
            for block in self.blocks:
                x, aux_loss = block(x, temb, t)
                total_aux_loss += aux_loss
        
        # Final layer
        x = self.final_layer(x, temb)
        x = self.unpatchify(x)
        
        return x, total_aux_loss
    
    @torch.jit.script_method
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
        # We use a simple mapping, can be refined based on actual flow dynamics
        
        # Approximate SNR(t) for linear flow
        # SNR increases as we go from t=0 to t=1
        # Clamp t to avoid division issues near boundaries
        t_safe = torch.clamp(t, min=1e-8, max=1.0 - 1e-8)
        snr = t_safe / (1 - t_safe)  # Simple SNR approximation
        
        # Apply Min-SNR-γ clipping
        snr_clipped = torch.minimum(snr, torch.ones_like(snr) * self.min_snr_gamma)
        
        # Weight is proportional to 1/SNR for balancing
        # But we clip it to avoid too much weight on noisy regions
        weight = 1.0 / (snr_clipped + 1.0)
        
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
