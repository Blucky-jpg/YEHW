"""
RTX 5090 Single-GPU Optimization Configuration for DeltaNetDiT
Maximizes training efficiency with gradient accumulation and memory optimization
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class RTX5090Config:
    """Optimized configuration for RTX 5090 training"""
    
    # GPU Specifications
    vram_gb: int = 32
    fp8_tflops: int = 2900  # 2.9 PFLOPS
    memory_bandwidth_gb: int = 1536  # 1.5 TB/s
    
    # Model Architecture (your DeltaNetDiT)
    model_params: Dict = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {
                'hidden_size': 1152,
                'depth': 28,
                'num_heads': 16,
                'patch_size': 2,
                'input_size': 256,
                'in_channels': 4,  # Latent space
                'moe_num_experts': 8,
                'moe_top_k': 2,
            }
    
    def calculate_model_memory(self) -> Dict[str, float]:
        """Calculate memory requirements in GB"""
        
        h = self.model_params['hidden_size']
        d = self.model_params['depth']
        heads = self.model_params['num_heads']
        experts = self.model_params['moe_num_experts']
        
        # Model parameters (in millions)
        # Attention: QKV + Proj per layer
        attn_params = d * (4 * h * h) / 1e6
        
        # MoE: Each expert has ~2.67x hidden size
        ffn_mult = 8/3  # From your MoE implementation
        moe_params = d * experts * (2 * h * int(h * ffn_mult)) / 1e6
        
        # Embeddings + Final layer
        embed_params = 10 / 1e6  # Rough estimate
        
        total_params_m = attn_params + moe_params + embed_params
        
        # Memory calculations
        results = {
            'total_params_millions': total_params_m,
            
            # FP4 model weights (4 bits = 0.5 bytes per param)
            'model_weights_fp4_gb': total_params_m * 0.5 / 1024,
            
            # FP8 model weights (8 bits = 1 byte per param)
            'model_weights_fp8_gb': total_params_m * 1.0 / 1024,
            
            # BF16 model weights (for comparison)
            'model_weights_bf16_gb': total_params_m * 2.0 / 1024,
            
            # Lion optimizer state (only momentum, FP32)
            'optimizer_state_gb': total_params_m * 4.0 / 1024,
            
            # Gradients (usually BF16)
            'gradients_gb': total_params_m * 2.0 / 1024,
        }
        
        return results
    
    def calculate_activation_memory(self, batch_size: int, sequence_length: int) -> float:
        """Calculate activation memory for forward pass in GB"""
        
        h = self.model_params['hidden_size']
        d = self.model_params['depth']
        
        # Activations per layer: roughly 2 * batch * seq * hidden
        # Factor of 2 for intermediate activations
        acts_per_layer = 2 * batch_size * sequence_length * h * 2 / 1e9  # BF16
        
        # Total for all layers (some can be recomputed with checkpointing)
        total_acts = acts_per_layer * d
        
        return total_acts
    
    def get_optimal_batch_config(self) -> Dict[str, any]:
        """Calculate optimal batch size and gradient accumulation"""
        
        mem_info = self.calculate_model_memory()
        
        # Available VRAM after model + optimizer
        model_mem = mem_info['model_weights_fp4_gb']
        optimizer_mem = mem_info['optimizer_state_gb']
        gradient_mem = mem_info['gradients_gb']
        
        # Reserve 4GB for PyTorch overhead and workspace
        overhead_gb = 4
        
        available_for_activations = self.vram_gb - (
            model_mem + optimizer_mem + gradient_mem + overhead_gb
        )
        
        # Image size 256x256, latent space 64x64 (with patch_size=2)
        sequence_length = (256 // self.model_params['patch_size']) ** 2  # 1024
        
        # Calculate max batch size that fits
        max_batch_per_forward = 1
        while True:
            next_batch = max_batch_per_forward + 1
            activation_mem = self.calculate_activation_memory(next_batch, sequence_length)
            if activation_mem > available_for_activations:
                break
            max_batch_per_forward = next_batch
        
        # Optimal configurations for different effective batch sizes
        configs = []
        
        # Target effective batch sizes for stable training
        target_batch_sizes = [32, 64, 128, 256, 512]
        
        for target_bs in target_batch_sizes:
            if target_bs <= max_batch_per_forward:
                # No gradient accumulation needed
                configs.append({
                    'effective_batch_size': target_bs,
                    'micro_batch_size': target_bs,
                    'gradient_accumulation_steps': 1,
                    'memory_usage_gb': (
                        model_mem + optimizer_mem + gradient_mem + 
                        self.calculate_activation_memory(target_bs, sequence_length) + 
                        overhead_gb
                    ),
                    'efficiency': 'Optimal - No accumulation needed'
                })
            else:
                # Need gradient accumulation
                grad_accum_steps = math.ceil(target_bs / max_batch_per_forward)
                micro_batch = target_bs // grad_accum_steps
                
                configs.append({
                    'effective_batch_size': target_bs,
                    'micro_batch_size': micro_batch,
                    'gradient_accumulation_steps': grad_accum_steps,
                    'memory_usage_gb': (
                        model_mem + optimizer_mem + gradient_mem + 
                        self.calculate_activation_memory(micro_batch, sequence_length) + 
                        overhead_gb
                    ),
                    'efficiency': f'Good - {grad_accum_steps}x accumulation'
                })
        
        return {
            'max_batch_without_accumulation': max_batch_per_forward,
            'available_vram_for_activations_gb': available_for_activations,
            'model_memory_breakdown': mem_info,
            'recommended_configs': configs,
            'sequence_length': sequence_length,
        }

def get_rtx5090_training_config():
    """Get optimized training configuration for RTX 5090"""
    
    config = RTX5090Config()
    optimal = config.get_optimal_batch_config()
    
    # Recommended configuration balancing speed and stability
    recommended = {
        'optimizer': 'lion',  # Best for FP4/FP8
        'learning_rate': 3e-5,  # Lion optimal LR
        'weight_decay': 0.01,
        'betas': (0.95, 0.98),  # Lion betas
        
        # Batch configuration (choose based on your needs)
        'configs': {
            'speed_optimized': {
                'batch_size': optimal['max_batch_without_accumulation'],
                'gradient_accumulation_steps': 1,
                'effective_batch_size': optimal['max_batch_without_accumulation'],
                'notes': 'Fastest training, may need lower LR'
            },
            'balanced': {
                'batch_size': min(32, optimal['max_batch_without_accumulation']),
                'gradient_accumulation_steps': max(1, 64 // min(32, optimal['max_batch_without_accumulation'])),
                'effective_batch_size': 64,
                'notes': 'Good balance of speed and stability'
            },
            'quality_optimized': {
                'batch_size': min(16, optimal['max_batch_without_accumulation']),
                'gradient_accumulation_steps': 128 // min(16, optimal['max_batch_without_accumulation']),
                'effective_batch_size': 128,
                'notes': 'Best convergence, slower training'
            },
        },
        
        # Performance optimizations
        'use_amp': True,
        'amp_dtype': 'bfloat16',  # Better than fp16 for FP4/FP8
        'use_channels_last': True,
        'tf32': True,
        'cudnn_benchmark': True,
        'use_compile': True,  # torch.compile for extra speed
        'compile_mode': 'max-autotune',
        
        # Gradient checkpointing (trade compute for memory)
        'gradient_checkpointing': optimal['max_batch_without_accumulation'] < 32,
        'checkpoint_every_n_layers': 4,  # Checkpoint every 4th layer
        
        # Memory efficient attention
        'use_sage_attention': True,  # Your FP8 optimized attention
        'sage_attention_fp8': True,
        
        # Training stability
        'gradient_clip_norm': 1.0,
        'lr_warmup_steps': 10000,
        'lr_schedule': 'cosine',
        
        # Logging
        'log_every': 10,
        'checkpoint_every': 1000,
        'sample_every': 500,
    }
    
    return optimal, recommended

def print_optimization_report():
    """Print detailed optimization report for RTX 5090"""
    
    optimal, recommended = get_rtx5090_training_config()
    
    print("=" * 70)
    print("RTX 5090 OPTIMIZATION REPORT FOR DELTANETDIT")
    print("=" * 70)
    
    print("\n📊 MODEL MEMORY BREAKDOWN:")
    print("-" * 40)
    mem_breakdown = optimal['model_memory_breakdown']
    print(f"Total Parameters: {mem_breakdown['total_params_millions']:.1f}M")
    print(f"FP4 Model Weights: {mem_breakdown['model_weights_fp4_gb']:.2f} GB")
    print(f"FP8 Model Weights: {mem_breakdown['model_weights_fp8_gb']:.2f} GB")
    print(f"Lion Optimizer State: {mem_breakdown['optimizer_state_gb']:.2f} GB")
    print(f"Gradients (BF16): {mem_breakdown['gradients_gb']:.2f} GB")
    
    print(f"\n📈 MAXIMUM BATCH SIZES:")
    print("-" * 40)
    print(f"Max batch without gradient accumulation: {optimal['max_batch_without_accumulation']}")
    print(f"Available VRAM for activations: {optimal['available_vram_for_activations_gb']:.1f} GB")
    print(f"Sequence length: {optimal['sequence_length']} tokens")
    
    print(f"\n🎯 RECOMMENDED CONFIGURATIONS:")
    print("-" * 40)
    for config in optimal['recommended_configs']:
        if config['gradient_accumulation_steps'] == 1:
            print(f"\n✅ Batch Size {config['effective_batch_size']}:")
        else:
            print(f"\n⚡ Effective Batch Size {config['effective_batch_size']}:")
        print(f"  - Micro Batch: {config['micro_batch_size']}")
        print(f"  - Gradient Accumulation: {config['gradient_accumulation_steps']}x")
        print(f"  - Memory Usage: {config['memory_usage_gb']:.1f}/{32} GB")
        print(f"  - Status: {config['efficiency']}")
    
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    print("1. SPEED OPTIMIZED:")
    speed = recommended['configs']['speed_optimized']
    print(f"   - Batch: {speed['batch_size']}, Accum: {speed['gradient_accumulation_steps']}x")
    print(f"   - {speed['notes']}")
    
    print("\n2. BALANCED (RECOMMENDED):")
    balanced = recommended['configs']['balanced']
    print(f"   - Batch: {balanced['batch_size']}, Accum: {balanced['gradient_accumulation_steps']}x")
    print(f"   - {balanced['notes']}")
    
    print("\n3. QUALITY OPTIMIZED:")
    quality = recommended['configs']['quality_optimized']
    print(f"   - Batch: {quality['batch_size']}, Accum: {quality['gradient_accumulation_steps']}x")
    print(f"   - {quality['notes']}")
    
    print("\n💡 ADDITIONAL OPTIMIZATIONS:")
    print("-" * 40)
    print(f"- Gradient Checkpointing: {'Yes' if recommended['gradient_checkpointing'] else 'No'}")
    print(f"- Torch Compile: {recommended['compile_mode']}")
    print(f"- AMP Dtype: {recommended['amp_dtype']}")
    print(f"- Channels Last: {recommended['use_channels_last']}")
    print(f"- SageAttention FP8: {recommended['sage_attention_fp8']}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print_optimization_report()
    
    # Generate optimized config file
    optimal, recommended = get_rtx5090_training_config()
    
    import json
    config = {
        'model_config': {
            'input_size': 256,
            'patch_size': 2,
            'in_channels': 4,
            'hidden_size': 1152,
            'depth': 28,
            'num_heads': 16,
            'num_classes': 1000,
            'use_flow_matching': True,
            'predict_x1': True,
            'use_min_snr_gamma': True,
            'min_snr_gamma': 5.0,
            'x1_loss_weight': 0.1,
            'moe_num_experts': 8,
            'moe_top_k': 2,
            'moe_low_noise_experts': 3,
            'moe_high_noise_experts': 7,
        },
        **recommended['configs']['balanced'],  # Use balanced config
        'optimizer': 'lion',
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'use_amp': True,
        'use_channels_last': True,
        'tf32': True,
        'cudnn_benchmark': True,
        'use_compile': True,
        'compile_mode': 'max-autotune',
        'gradient_clip_norm': 1.0,
        'lr_warmup_steps': 10000,
        'lr_schedule': 'cosine',
        'num_epochs': 1000,
        'checkpoint_every': 1000,
        'log_every': 10,
    }
    
    with open('config_rtx5090_optimal.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n✅ Saved optimal configuration to: config_rtx5090_optimal.json")
