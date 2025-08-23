"""
TiTok Tokenizer Training Script

Implements two-stage training for TiTok tokenizer:
1. Stage 1: Proxy code training using VAE latents as targets
2. Stage 2: End-to-end training with decoder fine-tuning

This script trains the TiTok tokenizer independently before integrating
with the main DeltaNetDiT training pipeline.
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from enhanced.titok_tokenizer import TiTokTokenizer

# Optional: import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional: import diffusers VAE
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class ImageDataset(torch.utils.data.Dataset):
    """Simple image dataset for TiTok training."""

    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class TiTokTrainer:
    """Trainer for TiTok tokenizer."""

    def __init__(
        self,
        tokenizer: TiTokTokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda',
        use_amp: bool = True,
        use_wandb: bool = False,
        checkpoint_dir: str = './checkpoints/titok',
        log_dir: str = './logs/titok',
    ):
        self.tokenizer = tokenizer.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp

        # Optimizer
        self.optimizer = AdamW(
            tokenizer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # AMP scaler
        self.scaler = GradScaler('cuda') if use_amp else None

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.global_step = 0

        # Wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project='titok-training', name=f'titok-{datetime.now():%Y%m%d_%H%M%S}')

    def train_stage1_proxy_codes(
        self,
        num_epochs: int = 50,
        vae_model: Optional[str] = None,
        reconstruction_weight: float = 1.0,
        commit_weight: float = 0.1,
    ):
        """
        Stage 1: Train tokenizer using proxy codes from VAE latents.

        Args:
            num_epochs: Number of training epochs
            vae_model: Pretrained VAE model name (e.g., 'stabilityai/sd-vae-ft-mse')
            reconstruction_weight: Weight for reconstruction loss
            commit_weight: Weight for VQ commitment loss
        """
        self.logger.info("Starting Stage 1: Proxy Code Training")

        # Load VAE for target latents
        vae = None
        if vae_model and DIFFUSERS_AVAILABLE:
            self.logger.info(f"Loading VAE: {vae_model}")
            vae = AutoencoderKL.from_pretrained(vae_model).to(self.device).eval()
            for param in vae.parameters():
                param.requires_grad = False

        for epoch in range(num_epochs):
            self.tokenizer.train()
            epoch_losses = {'total': 0, 'reconstruction': 0, 'commitment': 0}

            for batch_idx, images in enumerate(self.train_loader):
                images = images.to(self.device)
                batch_size = images.shape[0]

                with torch.no_grad():
                    if vae is not None:
                        # Use VAE latents as targets
                        latents = vae.encode(images).latent_dist.sample()
                        # Convert latents to reconstruction targets
                        targets = vae.decode(latents).sample
                    else:
                        # Use images directly as targets
                        targets = images

                # Forward pass - process batch one by one to handle TiTok's batch size 1 limitation
                all_reconstructed = []
                all_commit_loss = []

                for i in range(batch_size):
                    single_image = images[i:i+1]  # (1, 3, H, W)

                    with autocast('cuda', enabled=self.use_amp):
                        tokens, quantized, commit_loss, reconstructed = self.tokenizer(single_image)

                        all_reconstructed.append(reconstructed)
                        all_commit_loss.append(commit_loss)

                # Stack results back to batch
                reconstructed = torch.cat(all_reconstructed, dim=0)  # (batch_size, 3, H, W)
                commit_loss = torch.stack(all_commit_loss).mean()  # Average commitment losses

                # Ensure reconstructed has the same dtype as targets for AMP compatibility
                if targets.dtype != reconstructed.dtype:
                    reconstructed = reconstructed.to(dtype=targets.dtype)

                # Debug: Check for NaN/inf values before loss computation
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    self.logger.warning(f"NaN/inf detected in reconstructed tensor. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    self.logger.warning(f"NaN/inf detected in targets tensor. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                # Clip values to prevent overflow
                reconstructed = torch.clamp(reconstructed, -10.0, 10.0)
                targets = torch.clamp(targets, -10.0, 10.0)

                # Reconstruction loss with numerical stability
                recon_loss = F.mse_loss(reconstructed, targets, reduction='mean')

                # Check if loss is NaN
                if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                    self.logger.warning(f"NaN/inf loss detected (recon_loss: {recon_loss.item()}). Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                # Total loss
                total_loss = reconstruction_weight * recon_loss + commit_weight * commit_loss

                # Backward pass
                if self.scaler is not None:
                    try:
                        self.scaler.scale(total_loss).backward()

                        # Clip gradients to prevent inf/nan values
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except ValueError as e:
                        if "Attempting to unscale FP16 gradients" in str(e):
                            self.logger.warning("Detected inf gradients, skipping this step and disabling AMP for this batch")
                            # Skip this step and reset for next batch
                            self.optimizer.zero_grad()
                            # Optionally disable AMP for a few steps to recover
                            self.use_amp = False
                            self.scaler = None
                        else:
                            raise e
                else:
                    total_loss.backward()

                    # Clip gradients to prevent inf/nan values
                    torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)

                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update losses
                batch_size = images.shape[0]
                epoch_losses['total'] += total_loss.item() * batch_size
                epoch_losses['reconstruction'] += recon_loss.item() * batch_size
                epoch_losses['commitment'] += commit_loss.item() * batch_size

                # Logging
                if batch_idx % 100 == 0:
                    avg_losses = {k: v / ((batch_idx + 1) * batch_size) for k, v in epoch_losses.items()}
                    self.logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] | "
                        f"Total: {avg_losses['total']:.4f} | "
                        f"Recon: {avg_losses['reconstruction']:.4f} | "
                        f"Commit: {avg_losses['commitment']:.4f}"
                    )

                self.global_step += 1

            # Epoch end
            avg_losses = {k: v / len(self.train_loader.dataset) for k, v in epoch_losses.items()}
            self.logger.info(f"Epoch {epoch} completed | " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))

            # Learning rate scheduling
            self.scheduler.step()

            # Validation
            if self.val_loader is not None:
                val_losses = self.validate()
                self.logger.info(f"Validation | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))

            # Checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'stage1_epoch_{epoch}.pt')

            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    **{f'train/{k}': v for k, v in avg_losses.items()}
                })

        # Save final checkpoint
        self.save_checkpoint('stage1_final.pt')
        self.logger.info("Stage 1 training completed!")

    def train_stage2_end_to_end(
        self,
        num_epochs: int = 30,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.01,
    ):
        """
        Stage 2: End-to-end training with decoder fine-tuning.

        Args:
            num_epochs: Number of training epochs
            perceptual_weight: Weight for perceptual loss
            adversarial_weight: Weight for adversarial loss
        """
        self.logger.info("Starting Stage 2: End-to-End Training")

        # Enable decoder for training
        self.tokenizer.use_decoder = True
        self.tokenizer.decoder = self.tokenizer.decoder.to(self.device)

        # Fine-tune decoder parameters
        decoder_params = list(self.tokenizer.decoder.parameters())
        self.optimizer = AdamW(decoder_params, lr=1e-5, weight_decay=0.01)

        for epoch in range(num_epochs):
            self.tokenizer.train()
            epoch_losses = {'total': 0, 'reconstruction': 0, 'perceptual': 0}

            for batch_idx, images in enumerate(self.train_loader):
                images = images.to(self.device)
                batch_size = images.shape[0]

                # Forward pass - process batch one by one to handle TiTok's batch size 1 limitation
                all_reconstructed = []
                all_commit_loss = []

                for i in range(batch_size):
                    single_image = images[i:i+1]  # (1, 3, H, W)

                    with autocast('cuda', enabled=self.use_amp):
                        tokens, quantized, commit_loss, reconstructed = self.tokenizer(single_image)

                        all_reconstructed.append(reconstructed)
                        all_commit_loss.append(commit_loss)

                # Stack results back to batch
                reconstructed = torch.cat(all_reconstructed, dim=0)  # (batch_size, 3, H, W)
                commit_loss = torch.stack(all_commit_loss).mean()  # Average commitment losses

                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, images)

                # Perceptual loss (simple L1 loss as approximation)
                perceptual_loss = F.l1_loss(reconstructed, images)

                # Total loss
                total_loss = recon_loss + perceptual_weight * perceptual_loss

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update losses
                epoch_losses['total'] += total_loss.item() * batch_size
                epoch_losses['reconstruction'] += recon_loss.item() * batch_size
                epoch_losses['perceptual'] += perceptual_loss.item() * batch_size

                # Logging
                if batch_idx % 100 == 0:
                    avg_losses = {k: v / ((batch_idx + 1) * batch_size) for k, v in epoch_losses.items()}
                    self.logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] | "
                        f"Total: {avg_losses['total']:.4f} | "
                        f"Recon: {avg_losses['reconstruction']:.4f} | "
                        f"Perceptual: {avg_losses['perceptual']:.4f}"
                    )

                self.global_step += 1

            # Epoch end
            avg_losses = {k: v / len(self.train_loader.dataset) for k, v in epoch_losses.items()}
            self.logger.info(f"Epoch {epoch} completed | " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))

            # Validation
            if self.val_loader is not None:
                val_losses = self.validate()
                self.logger.info(f"Validation | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))

            # Checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f'stage2_epoch_{epoch}.pt')

            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    **{f'train/{k}': v for k, v in avg_losses.items()}
                })

        # Save final checkpoint
        self.save_checkpoint('stage2_final.pt')
        self.logger.info("Stage 2 training completed!")

    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_loader is None:
            return {}

        self.tokenizer.eval()
        val_losses = {'total': 0, 'reconstruction': 0, 'commitment': 0}

        with torch.no_grad():
            for images in self.val_loader:
                images = images.to(self.device)
                batch_size = images.shape[0]

                # Process batch one by one to handle TiTok's batch size 1 limitation
                all_reconstructed = []
                all_commit_loss = []

                for i in range(batch_size):
                    single_image = images[i:i+1]  # (1, 3, H, W)
                    tokens, quantized, commit_loss, reconstructed = self.tokenizer(single_image)

                    all_reconstructed.append(reconstructed)
                    all_commit_loss.append(commit_loss)

                # Stack results back to batch
                reconstructed = torch.cat(all_reconstructed, dim=0)  # (batch_size, 3, H, W)
                commit_loss = torch.stack(all_commit_loss).mean()  # Average commitment losses

                recon_loss = F.mse_loss(reconstructed, images)
                total_loss = recon_loss + 0.1 * commit_loss

                val_losses['total'] += total_loss.item() * batch_size
                val_losses['reconstruction'] += recon_loss.item() * batch_size
                val_losses['commitment'] += commit_loss.item() * batch_size

        avg_losses = {k: v / len(self.val_loader.dataset) for k, v in val_losses.items()}
        self.tokenizer.train()
        return avg_losses

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'tokenizer_state_dict': self.tokenizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.tokenizer.load_state_dict(checkpoint['tokenizer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']

        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"Loaded checkpoint: {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TiTok Tokenizer')

    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True, help='Training image directory')
    parser.add_argument('--val_dir', type=str, help='Validation image directory')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Model arguments
    parser.add_argument('--num_tokens', type=int, default=32, help='Number of TiTok tokens')
    parser.add_argument('--codebook_size', type=int, default=4096, help='VQ codebook size')
    parser.add_argument('--code_dim', type=int, default=16, help='VQ code dimension')

    # Training arguments
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Training stage (1 or 2)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--vae_model', type=str, help='Pretrained VAE model for stage 1')

    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/titok', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs/titok', help='Log directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Wandb logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Create tokenizer
    tokenizer = TiTokTokenizer(
        input_size=args.image_size,
        num_tokens=args.num_tokens,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        use_decoder=(args.stage == 2),  # Enable decoder for stage 2
    )

    # Create datasets
    train_dataset = ImageDataset(args.train_dir, args.image_size)
    val_dataset = ImageDataset(args.val_dir, args.image_size) if args.val_dir else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None

    # Create trainer
    trainer = TiTokTrainer(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_wandb=args.wandb,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train based on stage
    if args.stage == 1:
        trainer.train_stage1_proxy_codes(
            num_epochs=args.epochs,
            vae_model=args.vae_model
        )
    elif args.stage == 2:
        trainer.train_stage2_end_to_end(num_epochs=args.epochs)

    # Cleanup
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
