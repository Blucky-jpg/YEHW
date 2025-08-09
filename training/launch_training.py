"""
Launcher script for DeltaNetDiT training with various configurations.
Provides easy-to-use presets and distributed training support.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch
import json


def launch_single_gpu(args):
    """Launch training on single GPU."""
    cmd = [
        sys.executable,
        "training/train.py",
        "--config", args.config if args.config else "config_example.json",
    ]
    
    if args.wandb:
        cmd.append("--wandb")
    
    if args.compile:
        cmd.append("--compile")
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"Launching training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def launch_multi_gpu(args):
    """Launch distributed training on multiple GPUs."""
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Launching distributed training...")
    
    cmd = [
        sys.executable,
        "-m", "torch.distributed.launch",
        "--nproc_per_node", str(num_gpus),
        "--master_port", str(args.port),
        "training/train.py",
        "--distributed",
        "--config", args.config if args.config else "config_example.json",
    ]
    
    if args.wandb:
        cmd.append("--wandb")
    
    if args.compile:
        cmd.append("--compile")
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"Launching distributed training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def launch_with_accelerate(args):
    """Launch training with Hugging Face Accelerate."""
    # First create accelerate config if it doesn't exist
    accelerate_config = Path("accelerate_config.yaml")
    
    if not accelerate_config.exists():
        print("Creating default accelerate configuration...")
        config_content = """
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
"""
        accelerate_config.write_text(config_content)
    
    cmd = [
        "accelerate", "launch",
        "--config_file", str(accelerate_config),
        "train.py",
        "--config", args.config if args.config else "config_example.json",
    ]
    
    if args.wandb:
        cmd.append("--wandb")
    
    if args.compile:
        cmd.append("--compile")
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"Launching with accelerate: {' '.join(cmd)}")
    subprocess.run(cmd)


def create_preset_config(preset_name):
    """Create configuration for different training presets."""
    
    base_config = {
        "model_config": {
            "input_size": 256,
            "patch_size": 2,
            "in_channels": 4,
            "use_flow_matching": True,
            "predict_x1": True,
            "use_min_snr_gamma": True,
            "flow_time_encoding": "sinusoidal",
        },
        "use_flow_matching": True,
        "use_amp": True,
        "use_channels_last": True,
        "tf32": True,
        "cudnn_benchmark": True,
        "optimizer": "lion",  # Use Lion for FP4/FP8 stability
        "weight_decay": 0.01,  # Good default for Lion
    }
    
    presets = {
        "small": {
            **base_config,
            "model_config": {
                **base_config["model_config"],
                "hidden_size": 768,
                "depth": 12,
                "num_heads": 12,
                "moe_num_experts": 4,
                "moe_top_k": 2,
            },
            "batch_size": 64,
            "learning_rate": 1e-4,  # Will be auto-converted to Lion LR (~3e-5)
        },
        "base": {
            **base_config,
            "model_config": {
                **base_config["model_config"],
                "hidden_size": 1152,
                "depth": 28,
                "num_heads": 16,
                "moe_num_experts": 8,
                "moe_top_k": 2,
            },
            "batch_size": 32,
            "learning_rate": 1e-4,  # Will be auto-converted to Lion LR (~3e-5)
        },
        "large": {
            **base_config,
            "model_config": {
                **base_config["model_config"],
                "hidden_size": 1536,
                "depth": 36,
                "num_heads": 24,
                "moe_num_experts": 16,
                "moe_top_k": 3,
            },
            "batch_size": 16,
            "learning_rate": 1e-4,  # Will be auto-converted to Lion LR (~1e-5)
            "gradient_accumulation_steps": 2,
        },
        "xl": {
            **base_config,
            "model_config": {
                **base_config["model_config"],
                "hidden_size": 2048,
                "depth": 48,
                "num_heads": 32,
                "moe_num_experts": 32,
                "moe_top_k": 4,
            },
            "batch_size": 8,
            "learning_rate": 1e-4,  # Will be auto-converted to Lion LR (~1e-5)
            "gradient_accumulation_steps": 4,
        },
        "debug": {
            **base_config,
            "model_config": {
                **base_config["model_config"],
                "hidden_size": 384,
                "depth": 6,
                "num_heads": 6,
                "moe_num_experts": 2,
                "moe_top_k": 1,
            },
            "batch_size": 4,
            "num_epochs": 10,
            "learning_rate": 1e-4,  # Will be auto-converted to Lion LR (~3e-5)
            "optimizer": "lion",  # Explicitly use Lion for debug
            "checkpoint_every": 10,
            "sample_every": 5,
            "log_every": 1,
        },
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    config = presets[preset_name]
    config_path = f"config_{preset_name}.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created configuration file: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Launch DeltaNetDiT training')
    
    # Launch mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multi', 'accelerate'],
                       help='Training mode: single GPU, multi-GPU, or with accelerate')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--preset', type=str, 
                       choices=['small', 'base', 'large', 'xl', 'debug'],
                       help='Use a preset configuration')
    
    # Training options
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--port', type=int, default=29500, help='Master port for distributed training')
    
    args = parser.parse_args()
    
    # Create preset config if specified
    if args.preset:
        args.config = create_preset_config(args.preset)
    
    # Launch training based on mode
    if args.mode == 'single':
        launch_single_gpu(args)
    elif args.mode == 'multi':
        launch_multi_gpu(args)
    elif args.mode == 'accelerate':
        launch_with_accelerate(args)


if __name__ == "__main__":
    main()
