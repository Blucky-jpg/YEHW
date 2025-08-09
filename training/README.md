# DeltaNetDiT Training with Optimal Flow Matching

This directory contains the production-ready training infrastructure for DeltaNetDiT with Optimal Flow Matching (OFM).

## Features

### 🚀 Model Improvements
- **Optimal Flow Matching (OFM)**: Superior training dynamics compared to standard diffusion
- **Min-SNR-γ Weighting**: Focuses learning on critical timesteps for faster convergence
- **Combined Prediction Target**: Predicts both velocity field and clean image for better coherence
- **All DeltaNet Enhancements**: MoE, DeltaAttention, Progressive Pruning, Context-Adaptive Gating

### ⚡ Training Optimizations
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) with fp16/bf16
- **Distributed Training**: Multi-GPU support with DDP
- **Memory Optimization**: Channels-last format, gradient accumulation
- **Performance**: TF32, cuDNN benchmark, torch.compile support
- **EMA**: Exponential Moving Average for stable sampling
- **Gradient Clipping**: Prevents training instability

### 📊 Logging & Monitoring
- **TensorBoard**: Real-time metrics visualization
- **Weights & Biases**: Optional cloud-based experiment tracking
- **Comprehensive Metrics**: Loss breakdown, SNR weights, velocity norms
- **Regular Sampling**: Visual progress monitoring

## Quick Start

### 1. Basic Training (Single GPU)

```bash
# Using the example configuration
python train.py --config config_example.json

# Or use command-line arguments
python train.py --batch-size 32 --epochs 1000 --lr 1e-4
```

### 2. Using Presets

The launcher script provides optimized presets for different model sizes:

```bash
# Debug mode (small model for testing)
python launch_training.py --preset debug

# Base model (recommended for most use cases)
python launch_training.py --preset base

# Large model (for high-quality generation)
python launch_training.py --preset large

# Available presets: debug, small, base, large, xl
```

### 3. Multi-GPU Training

```bash
# Automatic multi-GPU detection and launch
python launch_training.py --mode multi --preset base

# Or manually with torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py --distributed --config config_example.json
```

### 4. Resume Training

```bash
# Resume from latest checkpoint
python train.py --config config_example.json --resume checkpoints/latest_step_1000.pt

# Or with launcher
python launch_training.py --preset base --resume checkpoints/latest_step_1000.pt
```

## Configuration

### Model Configuration

Edit `config_example.json` to customize model architecture:

```json
{
    "model_config": {
        "hidden_size": 1152,        # Model dimension
        "depth": 28,                 # Number of transformer blocks
        "num_heads": 16,             # Attention heads
        "moe_num_experts": 8,        # Number of MoE experts
        "moe_top_k": 2,              # Active experts per token
        "predict_x1": true,          # Enable dual prediction (v + x₁)
        "use_min_snr_gamma": true,   # Enable Min-SNR-γ weighting
        "min_snr_gamma": 5.0,        # γ value for SNR weighting
        "x1_loss_weight": 0.1        # Weight for x₁ prediction loss
    }
}
```

### Training Configuration

```json
{
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 1000,
    "lr_warmup_steps": 10000,
    "gradient_accumulation_steps": 1,
    "ema_decay": 0.9999,
    "use_amp": true,              # Mixed precision training
    "use_channels_last": true,    # Memory format optimization
    "use_compile": false,         # torch.compile (PyTorch 2.0+)
    "checkpoint_every": 1000,     # Checkpoint frequency
    "sample_every": 500           # Sampling frequency
}
```

## Custom Dataset Integration

To use your own dataset, modify the `setup_data()` method in `train.py`:

```python
def setup_data(self):
    """Setup data loaders for your dataset."""
    from your_dataset import YourDataset
    
    train_dataset = YourDataset(
        root=self.config.data_dir,
        transform=your_transform,
        # ... other args
    )
    
    # Rest of the dataloader setup remains the same
    self.train_loader = DataLoader(train_dataset, ...)
```

## Monitoring Training

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

### Weights & Biases

```bash
# Enable W&B logging
python train.py --config config_example.json --wandb

# Or with launcher
python launch_training.py --preset base --wandb
```

## Advanced Usage

### Custom Learning Rate Schedule

```python
# In train.py, modify setup_optimization()
self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    self.optimizer,
    max_lr=self.config.learning_rate,
    total_steps=total_training_steps,
    pct_start=0.1
)
```

### Adjusting OFM Parameters

```python
# Fine-tune Min-SNR-γ
"min_snr_gamma": 5.0,  # Try values between 1.0-10.0

# Adjust x₁ prediction weight
"x1_loss_weight": 0.1,  # Increase for stronger coherence

# Change time encoding
"flow_time_encoding": "learned",  # Options: "sinusoidal", "learned", "fourier"
```

### Memory Optimization

For large models or limited VRAM:

```json
{
    "batch_size": 8,
    "gradient_accumulation_steps": 4,  # Effective batch = 32
    "use_amp": true,
    "use_channels_last": true,
    "use_compile": true,
    "compile_mode": "reduce-overhead"
}
```

## Performance Tips

1. **Enable TF32** (Ampere GPUs): Already enabled by default
2. **Use Channels-Last**: Enabled by default for better performance
3. **Compile Model** (PyTorch 2.0+): Add `--compile` flag
4. **Adjust Workers**: Set `num_workers` based on CPU cores
5. **Pin Memory**: Enabled by default for faster transfers

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `use_amp` for mixed precision
- Reduce model size (use smaller preset)

### Slow Training

- Enable `use_compile` (PyTorch 2.0+)
- Ensure `cudnn_benchmark=true`
- Check `num_workers` setting
- Use multi-GPU training

### Unstable Training

- Reduce `learning_rate`
- Increase `lr_warmup_steps`
- Adjust `gradient_clip_norm`
- Check `min_snr_gamma` value

## Model Architectures

| Preset | Hidden | Depth | Heads | Experts | Parameters | VRAM (BS=32) |
|--------|--------|-------|-------|---------|------------|--------------|
| debug  | 384    | 6     | 6     | 2       | ~10M       | ~4GB         |
| small  | 768    | 12    | 12    | 4       | ~100M      | ~8GB         |
| base   | 1152   | 28    | 16    | 8       | ~500M      | ~16GB        |
| large  | 1536   | 36    | 24    | 16      | ~1.2B      | ~24GB        |
| xl     | 2048   | 48    | 32    | 32      | ~3B        | ~40GB        |

## Citation

If you use this code, please cite:

```bibtex
@article{deltanet-dit-ofm,
  title={DeltaNetDiT with Optimal Flow Matching},
  author={Your Name},
  year={2024}
}
```

## License

[Your License Here]

## Support

For issues or questions, please open an issue on GitHub or contact [your contact info].
