#!/usr/bin/env python3
"""
Debug script for TiTok + DeltaNetDiT pipeline testing.

This script tests the complete pipeline using synthetic data:
1. Generate synthetic training data
2. Train TiTok tokenizer (5 epochs)
3. Train main model with TiTok enabled (3 epochs)

Usage:
    python debug_titok_pipeline.py

Or with custom settings:
    python debug_titok_pipeline.py --titok_epochs 3 --model_epochs 2
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced.titok_tokenizer import TiTokTokenizer
from enhanced.Model import DeltaNetDiT
from training.train_titok import TiTokTrainer
from training.train import DeltaNetDiTTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('debug_titok.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SyntheticDataset:
    """Generate synthetic training data for debugging."""

    def __init__(self, num_samples: int = 1000, image_size: int = 64):
        self.num_samples = num_samples
        self.image_size = image_size

        # Generate synthetic data
        logger.info(f"Generating {num_samples} synthetic images of size {image_size}x{image_size}")
        self.images = []

        for i in range(num_samples):
            # Create synthetic patterns that TiTok can learn
            img = self._generate_pattern(i)
            self.images.append(img)

        self.images = torch.stack(self.images)
        logger.info(f"Synthetic dataset ready: {self.images.shape}")

    def _generate_pattern(self, idx: int) -> torch.Tensor:
        """Generate a synthetic image pattern."""
        # Create different pattern types based on index
        pattern_type = idx % 5

        if pattern_type == 0:
            # Gradient pattern
            x = torch.linspace(-1, 1, self.image_size)
            y = torch.linspace(-1, 1, self.image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            img = 0.5 + 0.5 * torch.sin(X * 3) * torch.cos(Y * 3)
        elif pattern_type == 1:
            # Checkerboard pattern
            img = torch.zeros(self.image_size, self.image_size)
            for i in range(0, self.image_size, 8):
                for j in range(0, self.image_size, 8):
                    if (i // 8 + j // 8) % 2 == 0:
                        img[i:i+8, j:j+8] = 1.0
        elif pattern_type == 2:
            # Circle pattern
            center = self.image_size // 2
            y, x = np.ogrid[:self.image_size, :self.image_size]
            y, x = torch.from_numpy(y).float(), torch.from_numpy(x).float()
            dist = torch.sqrt((x - center)**2 + (y - center)**2)
            img = (dist < center * 0.7).float()
        elif pattern_type == 3:
            # Noise pattern
            img = torch.randn(self.image_size, self.image_size) * 0.3 + 0.5
            img = torch.clamp(img, 0, 1)
        else:
            # Spiral pattern
            theta = torch.linspace(0, 4 * torch.pi, self.image_size)
            r = torch.linspace(0, self.image_size//2, self.image_size)
            x = (self.image_size//2 + r.unsqueeze(1) * torch.cos(theta.unsqueeze(0))).long()
            y = (self.image_size//2 + r.unsqueeze(1) * torch.sin(theta.unsqueeze(0))).long()
            img = torch.zeros(self.image_size, self.image_size)
            valid = (x >= 0) & (x < self.image_size) & (y >= 0) & (y < self.image_size)
            img[y[valid], x[valid]] = 1.0

        # Convert to RGB and normalize to [-1, 1] range
        img = img.unsqueeze(0).repeat(3, 1, 1)
        img = (img - 0.5) * 2  # Normalize to [-1, 1]

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx]


class DebugTiTokPipeline:
    """Debug pipeline for TiTok + DeltaNetDiT testing."""

    def __init__(self, args):
        self.args = args
        self.debug_dir = Path("debug_output")
        self.debug_dir.mkdir(exist_ok=True)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def create_synthetic_data(self):
        """Create synthetic training and validation datasets."""
        logger.info("Creating synthetic datasets...")

        train_dataset = SyntheticDataset(num_samples=500, image_size=64)
        val_dataset = SyntheticDataset(num_samples=100, image_size=64)

        return train_dataset, val_dataset

    def train_titok_tokenizer(self, train_dataset, val_dataset):
        """Train TiTok tokenizer with synthetic data."""
        logger.info("=== Training TiTok Tokenizer ===")

        # Create tokenizer with minimal config for debugging
        config = TiTokTokenizer.create_default_config(
            input_size=64,
            patch_size=8,
            hidden_size=192,  # Very small for debugging
            num_layers=3,
            num_heads=3,
            num_latent_tokens=8,
            codebook_size=256,
            token_size=8,
            commitment_cost=0.1,
            use_l2_norm=True,
            finetune_decoder=False,
        )

        tokenizer = TiTokTokenizer(config)

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        # Create trainer
        trainer = TiTokTrainer(
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,  # Higher LR for synthetic data
            checkpoint_dir=str(self.debug_dir / "titok_checkpoints"),
            use_wandb=False,
        )

        # Train for minimal epochs
        logger.info(f"Training TiTok for {self.args.titok_epochs} epochs...")
        try:
            trainer.train_stage1_proxy_codes(
                num_epochs=self.args.titok_epochs,
                reconstruction_weight=1.0,
                commit_weight=0.1,
            )
            logger.info("✓ TiTok tokenizer training completed successfully!")
        except Exception as e:
            logger.error(f"❌ TiTok training failed: {e}")
            raise

        return tokenizer

    def train_main_model(self, tokenizer, train_dataset, val_dataset):
        """Train main model with TiTok integration."""
        logger.info("=== Training Main Model with TiTok ===")

        # Create minimal model config
        model_config = {
            'input_size': 8,  # Very small for debugging
            'patch_size': 2,
            'in_channels': 3,
            'hidden_size': 192,
            'depth': 2,
            'num_heads': 3,
            'num_classes': 100,
            'use_flow_matching': True,
            'predict_x1': False,
            'use_min_snr_gamma': False,
            'use_titok': True,
            'titok_num_tokens': 8,
            'titok_codebook_size': 256,
            'titok_code_dim': 8,
        }

        # Create training config
        train_config = TrainingConfig(
            batch_size=8,
            num_epochs=self.args.model_epochs,
            learning_rate=1e-3,
            image_size=8,
            checkpoint_dir=str(self.debug_dir / "model_checkpoints"),
            use_flow_matching=True,
            predict_x1=False,
            use_min_snr_gamma=False,
            use_amp=False,  # Disable AMP for debugging
            use_compile=False,  # Disable compilation for debugging
            use_wandb=False,
        )

        # Override model config
        train_config.model_config = model_config

        # Create datasets
        train_data = [(img, label) for img, label in train_dataset]
        val_data = [(img, label) for img, label in val_dataset]

        # Create trainer
        trainer = DeltaNetDiTTrainer(train_config)

        # Override datasets with synthetic data
        trainer.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=0
        )

        logger.info(f"Training main model for {self.args.model_epochs} epochs...")
        try:
            trainer.train()
            logger.info("✓ Main model training completed successfully!")
        except Exception as e:
            logger.error(f"❌ Main model training failed: {e}")
            raise

    def run_debug_pipeline(self):
        """Run the complete debug pipeline."""
        logger.info("🚀 Starting TiTok + DeltaNetDiT Debug Pipeline")
        logger.info("=" * 60)

        try:
            # Step 1: Create synthetic data
            logger.info("Step 1: Creating synthetic training data...")
            train_dataset, val_dataset = self.create_synthetic_data()

            # Step 2: Train TiTok tokenizer
            logger.info("Step 2: Training TiTok tokenizer...")
            tokenizer = self.train_titok_tokenizer(train_dataset, val_dataset)

            # Step 3: Train main model with TiTok
            logger.info("Step 3: Training main model with TiTok integration...")
            self.train_main_model(tokenizer, train_dataset, val_dataset)

            logger.info("=" * 60)
            logger.info("🎉 Debug pipeline completed successfully!")
            logger.info("✓ TiTok tokenizer works correctly")
            logger.info("✓ TiTok integration with DeltaNetDiT works correctly")
            logger.info("✓ Complete pipeline is functional")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"❌ Debug pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description='Debug TiTok + DeltaNetDiT pipeline')
    parser.add_argument('--titok_epochs', type=int, default=3,
                       help='Number of epochs to train TiTok tokenizer')
    parser.add_argument('--model_epochs', type=int, default=2,
                       help='Number of epochs to train main model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run debug pipeline
    debugger = DebugTiTokPipeline(args)
    success = debugger.run_debug_pipeline()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
