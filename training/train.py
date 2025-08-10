"""
Production-ready training script for DeltaNetDiT with Optimal Flow Matching.
Includes distributed training, mixed precision, gradient accumulation, and comprehensive logging.
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from core.Model import DeltaNetDiT

# Optional: import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

# Optional: import accelerate for easier distributed training
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("accelerate not available. Install with: pip install accelerate")


@dataclass
class TrainingConfig:
    """Configuration for training DeltaNetDiT with OFM."""
    
    # Model configuration
    model_config: Dict[str, Any] = None
    
    # Data configuration
    data_dir: str = "./data"
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training configuration
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 10000
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    
    # OFM specific
    use_flow_matching: bool = True
    predict_x1: bool = True
    use_min_snr_gamma: bool = True
    min_snr_gamma: float = 5.0
    x1_loss_weight: float = 0.1
    
    # Mixed precision and optimization
    use_amp: bool = True
    use_channels_last: bool = True
    use_compile: bool = False  # torch.compile
    compile_mode: str = "reduce-overhead"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 1000
    save_latest_every: int = 100
    resume_from: Optional[str] = None
    
    # Logging
    log_dir: str = "./logs"
    log_every: int = 10
    sample_every: int = 500
    num_samples: int = 16
    use_wandb: bool = False
    wandb_project: str = "deltanet-dit-ofm"
    wandb_entity: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    
    # Performance optimization
    tf32: bool = True  # Enable TF32 on Ampere GPUs
    cudnn_benchmark: bool = True
    find_unused_parameters: bool = False
    
    def __post_init__(self):
        """Initialize default model config if not provided."""
        if self.model_config is None:
            self.model_config = {
                'input_size': self.image_size,
                'patch_size': 2,
                'in_channels': 4,  # Assuming latent space
                'hidden_size': 1152,
                'depth': 28,
                'num_heads': 16,
                'num_classes': 1000,
                'use_flow_matching': self.use_flow_matching,
                'predict_x1': self.predict_x1,
                'use_min_snr_gamma': self.use_min_snr_gamma,
                'min_snr_gamma': self.min_snr_gamma,
                'x1_loss_weight': self.x1_loss_weight,
                'moe_num_experts': 8,
                'moe_top_k': 2,
            }


class ExponentialMovingAverage:
    """EMA for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class DeltaNetDiTTrainer:
    """Trainer class for DeltaNetDiT with OFM."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.epoch = 0
        
        # Setup distributed training if enabled
        if config.distributed:
            self.setup_distributed()
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
        # Setup data
        self.setup_data()
        
        # Setup mixed precision
        if config.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Setup EMA
        if config.ema_decay > 0:
            self.ema = ExponentialMovingAverage(self.model, decay=config.ema_decay)
        else:
            self.ema = None
        
        # Setup experiment tracking
        if config.use_wandb and WANDB_AVAILABLE:
            self.setup_wandb()
        
        # Performance optimizations
        self.setup_performance_optimizations()
    
    def setup_distributed(self):
        """Setup distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.config.rank = int(os.environ['RANK'])
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            self.config.local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(self.config.local_rank)
        dist.init_process_group(
            backend=self.config.backend,
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        self.device = torch.device(f'cuda:{self.config.local_rank}')
        
    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging and tensorboard."""
        log_file = Path(self.config.log_dir) / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard
        if self.config.rank == 0:
            self.writer = SummaryWriter(self.config.log_dir)
        else:
            self.writer = None
    
    def setup_model(self):
        """Initialize and setup the model."""
        self.logger.info("Initializing DeltaNetDiT model...")
        
        # Create model
        self.model = DeltaNetDiT(**self.config.model_config)
        self.model.to(self.device)
        
        # Convert to channels_last format for better performance
        if self.config.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # Compile model if requested
        if self.config.use_compile and hasattr(torch, 'compile'):
            self.logger.info(f"Compiling model with mode: {self.config.compile_mode}")
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
        
        # Distributed training wrapper
        if self.config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters
            )
        
        # Load checkpoint if resuming
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Import Lion optimizer
        from training.lion_optimizer import create_lion_optimizer, get_recommended_lion_lr
        
        # Count model parameters for LR recommendation
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Choose optimizer based on config
        optimizer_type = getattr(self.config, 'optimizer_type', 'lion')  # Default to Lion for FP4/FP8
        
        if optimizer_type.lower() == 'lion':
            # Get recommended Lion LR based on model size and precision
            if self.config.learning_rate == 1e-4:  # Default Adam LR
                recommended_lr = get_recommended_lion_lr(
                    model_params=total_params,
                    precision='fp4_fp8'
                )
                self.logger.info(f"Using Lion optimizer with recommended LR: {recommended_lr}")
                lr = recommended_lr
            else:
                # Convert from Adam LR if specified
                lr = get_recommended_lion_lr(adam_lr=self.config.learning_rate)
                self.logger.info(f"Converted Adam LR {self.config.learning_rate} to Lion LR {lr}")
            
            self.optimizer = create_lion_optimizer(
                self.model,
                lr=lr,
                betas=(0.95, 0.98),  # Slightly different betas work better for Lion
                weight_decay=self.config.weight_decay,
                use_decoupled=True,  # Use LionW (decoupled weight decay)
                use_triton=False,  # Set to True if you have Triton installed for ~20% speedup
            )
        else:
            # Fallback to AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Learning rate scheduler
        if self.config.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config.num_epochs
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        if self.config.lr_warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.config.lr_warmup_steps
            )
    
    def setup_data(self):
        """Setup data loaders. Override this method for custom datasets."""
        # This is a placeholder - implement your specific dataset
        self.logger.warning("Using placeholder dataset. Implement setup_data() for your dataset.")
        
        # Placeholder dataset
        class PlaceholderDataset(Dataset):
            def __init__(self, size=10000, image_size=256, channels=4):
                self.size = size
                self.image_size = image_size
                self.channels = channels
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Return random data for testing
                image = torch.randn(self.channels, self.image_size, self.image_size)
                label = torch.randint(0, 1000, (1,)).item()
                return {'image': image, 'label': label}
        
        train_dataset = PlaceholderDataset(
            size=10000,
            image_size=self.config.image_size,
            channels=self.config.model_config['in_channels']
        )
        
        # Distributed sampler
        if self.config.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
        else:
            train_sampler = None
        
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def setup_performance_optimizations(self):
        """Setup performance optimizations."""
        if torch.cuda.is_available():
            # Enable TF32 on Ampere GPUs
            if self.config.tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmark for better performance
            if self.config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
    
    def setup_wandb(self):
        """Setup Weights & Biases tracking."""
        if self.config.rank == 0:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=f"deltanet-dit-{datetime.now():%Y%m%d_%H%M%S}"
            )
            wandb.watch(self.model, log_freq=100)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        images = batch['image'].to(self.device, non_blocking=True)
        labels = batch.get('label', None)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)
        
        # Convert to channels_last if enabled
        if self.config.use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=self.device)
        
        # Generate noise
        x0 = torch.randn_like(images)
        
        # OFM interpolation
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * images
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            # Forward pass
            model_module = self.model.module if hasattr(self.model, 'module') else self.model
            
            if self.config.use_flow_matching:
                # Use the model's built-in loss computation
                losses = model_module.compute_training_losses(x0, images, t, labels)
                loss = losses['total_loss']
                loss_dict = {k: v.item() for k, v in losses.items()}
            else:
                # Standard diffusion training (fallback)
                output, aux_loss = self.model(x_t, t, labels)
                target = images - x0  # Velocity for OFM
                loss = F.mse_loss(output, target) + aux_loss
                loss_dict = {'loss': loss.item(), 'aux_loss': aux_loss.item()}
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss_dict
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # Progress tracking
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            loss_dict = self.train_step(batch)
            
            # Accumulate losses for logging
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None and self.global_step % self.config.ema_update_every == 0:
                    self.ema.update()
                
                # Learning rate warmup
                if self.global_step < self.config.lr_warmup_steps:
                    self.warmup_scheduler.step()
                
                self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0 and self.config.rank == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                self.log_metrics(avg_losses, self.global_step)
                
                # Print progress
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{num_batches}] "
                    f"Step {self.global_step} | LR: {lr:.6f} | "
                    f"Loss: {avg_losses.get('total_loss', avg_losses.get('loss', 0)):.4f}"
                )
            
            # Sampling
            if self.global_step % self.config.sample_every == 0 and self.config.rank == 0:
                self.sample_and_log()
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_every == 0:
                self.save_checkpoint('checkpoint')
            
            if self.global_step % self.config.save_latest_every == 0:
                self.save_checkpoint('latest')
        
        # Average epoch losses
        avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_epoch_losses
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.config.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train epoch
            start_time = time.time()
            avg_losses = self.train_epoch()
            epoch_time = time.time() - start_time
            
            # Learning rate scheduling
            if self.scheduler is not None and self.global_step >= self.config.lr_warmup_steps:
                self.scheduler.step()
            
            # Log epoch summary
            if self.config.rank == 0:
                self.logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                    f"Avg Loss: {avg_losses.get('total_loss', avg_losses.get('loss', 0)):.4f}"
                )
                
                # Log to wandb
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        'epoch_time': epoch_time,
                        **{f'epoch/{k}': v for k, v in avg_losses.items()}
                    })
        
        # Final checkpoint
        self.save_checkpoint('final')
        self.logger.info("Training completed!")
        
        # Cleanup
        if self.config.distributed:
            dist.destroy_process_group()
        
        if self.config.use_wandb and WANDB_AVAILABLE and self.config.rank == 0:
            wandb.finish()
    
    def sample_and_log(self):
        """Generate samples and log them."""
        self.model.eval()
        
        with torch.no_grad():
            # Apply EMA weights for sampling
            if self.ema is not None:
                self.ema.apply_shadow()
            
            # Generate samples using ODE solver
            shape = (
                self.config.num_samples,
                self.config.model_config['in_channels'],
                self.config.image_size,
                self.config.image_size
            )
            
            # Simple sampling loop for OFM
            samples = self.sample_ofm(shape)
            
            # Log samples
            if self.writer is not None:
                self.writer.add_images('samples', samples, self.global_step)
            
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'samples': [wandb.Image(s) for s in samples]
                }, step=self.global_step)
            
            # Restore original weights
            if self.ema is not None:
                self.ema.restore()
        
        self.model.train()
    
    def sample_ofm(self, shape: Tuple[int, ...], steps: int = 50) -> torch.Tensor:
        """Sample using OFM ODE solver."""
        device = next(self.model.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Time steps
        dt = 1.0 / steps
        
        model_module = self.model.module if hasattr(self.model, 'module') else self.model
        
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            
            # Get velocity prediction
            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                output, _ = model_module(x, t)
                
                # If using combined prediction, extract velocity
                if self.config.predict_x1:
                    velocity = output[:, :shape[1]]
                else:
                    velocity = output
            
            # Euler step
            x = x + dt * velocity
        
        return x.clamp(-1, 1)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard and wandb."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'train/{key}', value, step)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=step)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.config.rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = Path(self.config.checkpoint_dir) / f"{name}_step_{self.global_step}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train DeltaNetDiT with OFM')
    
    # Basic arguments
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=1152)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=2)
    
    # OFM arguments
    parser.add_argument('--use-ofm', action='store_true', default=True)
    parser.add_argument('--predict-x1', action='store_true', default=True)
    parser.add_argument('--use-min-snr', action='store_true', default=True)
    parser.add_argument('--min-snr-gamma', type=float, default=5.0)
    
    # Performance arguments
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--channels-last', action='store_true', default=True)
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # Logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='deltanet-dit-ofm')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Extract optimizer field if present (not part of TrainingConfig)
        optimizer_type = config_dict.pop('optimizer', 'adamw')
        config = TrainingConfig(**config_dict)
        # Store optimizer type for later use
        config.optimizer_type = optimizer_type
    else:
        # Create config from command line arguments
        config = TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            image_size=args.image_size,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume,
            use_flow_matching=args.use_ofm,
            predict_x1=args.predict_x1,
            use_min_snr_gamma=args.use_min_snr,
            min_snr_gamma=args.min_snr_gamma,
            use_amp=args.amp,
            use_compile=args.compile,
            use_channels_last=args.channels_last,
            distributed=args.distributed,
            local_rank=args.local_rank,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
        
        # Update model config
        config.model_config.update({
            'hidden_size': args.hidden_size,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'patch_size': args.patch_size,
        })
    
    # Create trainer and start training
    trainer = DeltaNetDiTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
