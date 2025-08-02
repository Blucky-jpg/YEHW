import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import os
import time
import math
from typing import Optional, Tuple, Dict, Any, Union, List
from contextlib import contextmanager
from ..core.Model import DiT

# --- Enhanced NVIDIA Transformer Engine Integration ---
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling, Fp8ContextManager

# Enhanced FP4 recipe with adaptive scaling
fp4_recipe = DelayedScaling(
    format=Format.NVTE_FWD_NVFP4_BWD_MXFP8_SCALING,
    fp4_dtype=te.common.recipe.DType.kNVTEFloat4E2M1,
    max_history_len=32,  # Increased for better scaling
    override_linear_precision=(False, False, True),
    amax_history_len=16,  # Better amax tracking
    amax_compute_algo="max"  # More stable than "most_recent"
)

# Memory-efficient context manager
@contextmanager
def memory_efficient_context(enable_cache=True):
    """Context manager for memory efficiency."""
    if enable_cache:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OptimizedDataLoader:
    """Enhanced DataLoader with advanced CUDA optimizations."""
    
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True,
                 num_workers: int = 4, pin_memory: bool = True, 
                 persistent_workers: bool = True, prefetch_factor: int = None,
                 drop_last: bool = True, use_distributed: bool = False,
                 multiprocessing_context: str = None):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_distributed = use_distributed
        
        # Adaptive optimization based on system capabilities
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            pin_memory = True
            persistent_workers = True
            
            # Adaptive prefetch factor based on memory and workers
            if prefetch_factor is None:
                memory_gb = device_props.total_memory / (1024**3)
                prefetch_factor = min(8, max(2, int(memory_gb / 10)))  # Scale with memory
            
            # Use spawn context for better stability with CUDA
            if multiprocessing_context is None and num_workers > 0:
                multiprocessing_context = "spawn"
        else:
            prefetch_factor = prefetch_factor or 2
        
        sampler = None
        if use_distributed and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
            shuffle = False
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            pin_memory_device="cuda" if torch.cuda.is_available() else "",
            multiprocessing_context=multiprocessing_context,
            worker_init_fn=self._worker_init_fn if num_workers > 0 else None,
        )
    
    @staticmethod
    def _worker_init_fn(worker_id):
        """Initialize worker with optimal settings."""
        torch.set_num_threads(1)  # Prevent oversubscription
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed training."""
        if hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(epoch)


class AdaptiveMemoryDataset(Dataset):
    """Enhanced dataset with adaptive memory management."""
    
    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None,
                 preload_to_gpu: bool = False, transform=None, 
                 cache_size: int = 1000, use_memory_mapping: bool = True):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.preload_to_gpu = preload_to_gpu
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
        
        # Adaptive preloading based on dataset size and available memory
        if preload_to_gpu and torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            data_memory = data.numel() * data.element_size()
            
            if data_memory < available_memory * 0.3:  # Use max 30% of GPU memory
                self.data = self.data.cuda(non_blocking=True)
                if self.labels is not None:
                    self.labels = self.labels.cuda(non_blocking=True)
                print(f"Preloaded dataset to GPU: {len(data)} samples ({data_memory/1024**3:.2f}GB)")
            else:
                print(f"Dataset too large for GPU preloading: {data_memory/1024**3:.2f}GB")
                self.preload_to_gpu = False
        
        # Memory mapping for large datasets
        if use_memory_mapping and not self.preload_to_gpu and len(data) > 10000:
            self.use_memory_mapping = True
        else:
            self.use_memory_mapping = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Use cache for frequently accessed items
        if idx in self.cache and len(self.cache) < self.cache_size:
            x = self.cache[idx]
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
        else:
            x = self.data[idx]
            
            # Cache management
            if len(self.cache) >= self.cache_size:
                # Remove least frequently accessed item
                min_idx = min(self.access_count, key=self.access_count.get)
                del self.cache[min_idx]
                del self.access_count[min_idx]
            
            self.cache[idx] = x.clone() if hasattr(x, 'clone') else x
            self.access_count[idx] = 1
        
        if self.transform:
            x = self.transform(x)
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


class AdaptiveBatchSampler:
    """Enhanced batch sampler with dynamic batch size adjustment."""
    
    def __init__(self, dataset_size: int, base_batch_size: int, 
                 max_memory_gb: float = None, min_batch_size: int = 1,
                 shuffle: bool = True, drop_last: bool = True,
                 gradient_accumulation_steps: int = 1):
        self.dataset_size = dataset_size
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Dynamic batch size adjustment
        if torch.cuda.is_available() and max_memory_gb is None:
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            max_memory_gb = available_memory * 0.8  # Use 80% of available memory
        
        self.current_batch_size = base_batch_size
        self.memory_usage_history = []
        
    def adjust_batch_size(self, memory_usage: float, target_memory: float = 0.85):
        """Dynamically adjust batch size based on memory usage."""
        if memory_usage > target_memory and self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, 
                                        int(self.current_batch_size * 0.8))
            print(f"Reduced batch size to {self.current_batch_size} due to memory pressure")
        elif memory_usage < target_memory * 0.7 and self.current_batch_size < self.base_batch_size:
            self.current_batch_size = min(self.base_batch_size, 
                                        int(self.current_batch_size * 1.2))
            print(f"Increased batch size to {self.current_batch_size}")
    
    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            torch.manual_seed(torch.initial_seed())
            indices = torch.randperm(self.dataset_size).tolist()
        
        for i in range(0, len(indices), self.current_batch_size):
            batch_indices = indices[i:i + self.current_batch_size]
            if len(batch_indices) == self.current_batch_size or not self.drop_last:
                yield batch_indices
    
    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.current_batch_size
        return (self.dataset_size + self.current_batch_size - 1) // self.current_batch_size


@torch.compile(dynamic=True)
def create_optimizer_advanced(model: DiT, learning_rate: float = 1e-4, 
                             weight_decay: float = 0.01, optimizer_type: str = "adamw",
                             betas: Tuple[float, float] = (0.9, 0.95),
                             eps: float = 1e-8, fused: bool = True) -> optim.Optimizer:
    """Create optimizer with advanced settings and proper parameter grouping."""
    
    # Enhanced parameter grouping
    decay_params = []
    no_decay_params = []
    expert_params = []
    embedding_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Expert parameters (may need different settings)
        if "expert" in name:
            expert_params.append(param)
        # Embeddings and layer norms
        elif any(nd in name for nd in ["bias", "norm", "embedding", "pos_embed"]):
            if "embedding" in name:
                embedding_params.append(param)
            else:
                no_decay_params.append(param)
        # Regular parameters
        else:
            decay_params.append(param)
    
    # Create parameter groups with different settings
    param_groups = []
    
    if decay_params:
        param_groups.append({
            "params": decay_params, 
            "weight_decay": weight_decay,
            "lr": learning_rate
        })
    
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params, 
            "weight_decay": 0.0,
            "lr": learning_rate
        })
    
    if expert_params:
        param_groups.append({
            "params": expert_params, 
            "weight_decay": weight_decay * 0.1,  # Lower weight decay for experts
            "lr": learning_rate * 0.5  # Lower LR for experts (more stable)
        })
    
    if embedding_params:
        param_groups.append({
            "params": embedding_params, 
            "weight_decay": 0.0,
            "lr": learning_rate * 0.1  # Much lower LR for embeddings
        })
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(param_groups, betas=betas, eps=eps, fused=fused and torch.cuda.is_available())
    elif optimizer_type.lower() == "adam":
        return optim.Adam(param_groups, betas=betas, eps=eps, fused=fused and torch.cuda.is_available())
    elif optimizer_type.lower() == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, betas=betas)
        except ImportError:
            print("⚠️ Lion optimizer not available, falling back to AdamW")
            return optim.AdamW(param_groups, betas=betas, eps=eps, fused=fused and torch.cuda.is_available())
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_advanced_scheduler(optimizer: optim.Optimizer, scheduler_type: str = "cosine_with_warmup", 
                             num_training_steps: int = 100000, warmup_steps: int = 1000,
                             num_cycles: float = 0.5, final_lr_ratio: float = 0.0) -> optim.lr_scheduler._LRScheduler:
    """Create advanced learning rate scheduler that works with your GlobalScheduler."""
    
    if scheduler_type == "cosine_with_warmup":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            lr_factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            return final_lr_ratio + (1.0 - final_lr_ratio) * lr_factor
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == "polynomial":
        return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_training_steps, power=1.0)
    
    elif scheduler_type == "exponential":
        gamma = (final_lr_ratio) ** (1.0 / num_training_steps)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
    elif scheduler_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps,
            anneal_strategy='cos'
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class EnhancedDiffusionLoss(torch.nn.Module):
    """
    Enhanced diffusion loss function with multiple improvements.
    Fixed version without unused variables.
    """
    def __init__(self, loss_type: str = "mse", beta_schedule: str = "linear", 
                 timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 use_rectified_flow: bool = False, snr_gamma: float = 5.0,
                 use_v_parameterization: bool = False, use_snr_weighting: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.use_rectified_flow = use_rectified_flow
        self.snr_gamma = snr_gamma
        self.use_v_parameterization = use_v_parameterization
        self.use_snr_weighting = use_snr_weighting
        
        if not use_rectified_flow:
            # Create noise schedule
            if beta_schedule == "linear":
                betas = torch.linspace(beta_start, beta_end, timesteps)
            elif beta_schedule == "cosine":
                s = 0.008
                steps = timesteps + 1
                x = torch.linspace(0, timesteps, steps)
                alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                betas = torch.clamp(betas, 0.0001, 0.9999)
            elif beta_schedule == "scaled_linear":
                # Better schedule for high-resolution images
                betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
            else:
                raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            # Standard diffusion parameters
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
            
            # Additional parameters for v-parameterization (only if needed)
            if use_v_parameterization:
                self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
                self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
            
            # SNR weighting (only if needed)
            if use_snr_weighting:
                snr = alphas_cumprod / (1 - alphas_cumprod)
                snr_weights = torch.clamp(snr, max=snr_gamma)
                self.register_buffer("snr_weights", snr_weights)
    
    def get_v_target(self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute v-parameterization target."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to clean images according to diffusion schedule."""
        if self.use_rectified_flow:
            if noise is None:
                noise = torch.randn_like(x_start)
            t_expanded = t.view(-1, 1, 1, 1)
            return t_expanded * x_start + (1 - t_expanded) * noise
        else:
            if noise is None:
                noise = torch.randn_like(x_start)
            
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            
            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self, model_output: torch.Tensor, target: torch.Tensor, 
                t: torch.Tensor = None, reduction: str = "mean") -> torch.Tensor:
        """Compute enhanced diffusion loss."""
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction='none')
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction='none')
        elif self.loss_type == "focal_mse":
            # Focal loss variant for MSE
            mse_loss = F.mse_loss(model_output, target, reduction='none')
            pt = torch.exp(-mse_loss)  # Confidence
            focal_weight = (1 - pt) ** 2  # Focus on hard examples
            loss = focal_weight * mse_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply SNR weighting if enabled and timesteps provided
        if self.use_snr_weighting and t is not None and not self.use_rectified_flow:
            snr_weights = self.snr_weights[t].view(-1, 1, 1, 1)
            loss = loss * snr_weights
        
        # Reduction
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def count_parameters_detailed(model: DiT) -> Dict[str, Any]:
    """Enhanced parameter counting with detailed breakdown."""
    param_breakdown = {}
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # Categorize parameters
        if 'embedding' in name:
            category = 'embeddings'
        elif 'attention' in name or 'attn' in name:
            category = 'attention'
        elif 'mlp' in name or 'feed_forward' in name:
            category = 'feed_forward'
        elif 'norm' in name:
            category = 'normalization'
        elif 'expert' in name:
            category = 'experts'
        else:
            category = 'other'
        
        if category not in param_breakdown:
            param_breakdown[category] = {'params': 0, 'trainable': 0}
        
        param_breakdown[category]['params'] += param_count
        if param.requires_grad:
            param_breakdown[category]['trainable'] += param_count
    
    # Calculate memory requirements for different precisions
    memory_requirements = {
        'fp32_gb': total_params * 4 / 1024**3,
        'fp16_gb': total_params * 2 / 1024**3,
        'fp8_gb': total_params * 1 / 1024**3,
        'fp4_gb': total_params * 0.5 / 1024**3,
    }
    
    # Training memory estimates (including gradients, optimizer states)
    training_memory = {
        'adamw_fp32_gb': total_params * 12 / 1024**3,  # params + grads + 2 optimizer states
        'adamw_fp16_gb': total_params * 6 / 1024**3,
        'adamw_fp8_gb': total_params * 3 / 1024**3,
        'adamw_fp4_gb': total_params * 1.5 / 1024**3,
    }
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_M': total_params / 1e6,
        'trainable_parameters_M': trainable_params / 1e6,
        'parameter_breakdown': param_breakdown,
        'memory_requirements': memory_requirements,
        'training_memory_estimates': training_memory
    }
    
    # Print detailed summary
    print(f"\n📊 Model Parameter Analysis:")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"\n📋 Parameter Breakdown:")
    for category, counts in param_breakdown.items():
        print(f"   {category.title()}: {counts['params']:,} ({counts['params']/1e6:.1f}M)")
    
    print(f"\n💾 Memory Requirements:")
    print(f"   FP32: {memory_requirements['fp32_gb']:.2f} GB")
    print(f"   FP16: {memory_requirements['fp16_gb']:.2f} GB")
    print(f"   FP8:  {memory_requirements['fp8_gb']:.2f} GB")
    print(f"   FP4:  {memory_requirements['fp4_gb']:.2f} GB")
    
    print(f"\n🏋️ Training Memory (AdamW):")
    print(f"   FP32: {training_memory['adamw_fp32_gb']:.2f} GB")
    print(f"   FP16: {training_memory['adamw_fp16_gb']:.2f} GB") 
    print(f"   FP8:  {training_memory['adamw_fp8_gb']:.2f} GB")
    print(f"   FP4:  {training_memory['adamw_fp4_gb']:.2f} GB")
    
    return summary


def train_with_global_scheduler_integration(model: DiT, dataloader: Union[DataLoader, OptimizedDataLoader], 
                                          optimizer: optim.Optimizer, num_epochs: int = 100, 
                                          loss_fn: Optional[EnhancedDiffusionLoss] = None,
                                          device: str = "cuda", gradient_clip_val: float = 1.0,
                                          log_interval: int = 100, save_interval: int = 1000,
                                          checkpoint_path: Optional[str] = None,
                                          classifier_free_guidance: bool = False,
                                          cfg_dropout_prob: float = 0.1,
                                          use_compile: bool = True,
                                          accumulation_steps: int = 1,
                                          mixed_precision: bool = True,
                                          lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                                          use_rectified_flow: bool = False) -> Dict[str, Any]:
    """
    Training loop integrated with your GlobalScheduler.
    """
    # Import your scheduler
    from ..enhanced.global_scheduler import get_global_scheduler, register_default_schedules
    
    # Get global scheduler and register default schedules
    global_scheduler = get_global_scheduler()
    register_default_schedules(global_scheduler)
    
    # Initialize loss function
    if loss_fn is None:
        loss_fn = EnhancedDiffusionLoss(
            use_rectified_flow=use_rectified_flow,
            use_snr_weighting=True,
            snr_gamma=5.0
        )
    
    model.train()
    model.to(device)
    loss_fn.to(device)
    
    # Compile model for better performance
    if use_compile and hasattr(torch, 'compile'):
        print("🔧 Compiling model for Blackwell optimization...")
        model = torch.compile(model, mode='max-autotune', dynamic=True)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    # Training statistics
    training_stats = {
        'epoch_losses': [],
        'balance_losses': [],
        'gradient_norms': [],
        'qaf_switch_epoch': None,
        'memory_usage': [],
        'throughput_history': [],
        'learning_rates': [],
        'global_scheduler_values': {}  # Track scheduled values
    }
    
    global_step = 0
    best_loss = float('inf')
    
    print(f"🚀 Starting training with GlobalScheduler integration...")
    print(f"   Model: {count_parameters_detailed(model)['total_parameters_M']:.1f}M parameters")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Accumulation steps: {accumulation_steps}")
    print(f"   Mixed precision: {mixed_precision}")
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_balance_loss = 0.0
            num_batches = 0
            samples_processed = 0
            
            # Set epoch for distributed training
            if hasattr(dataloader, 'set_epoch'):
                dataloader.set_epoch(epoch)
            
            for batch_idx, batch_data in enumerate(dataloader):
                batch_start_time = time.time()
                
                # Handle different batch formats
                if len(batch_data) == 2:
                    x, y = batch_data
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                else:
                    x = batch_data[0].to(device, non_blocking=True)
                    y = None
                
                batch_size = x.shape[0]
                samples_processed += batch_size
                
                # Sample timesteps
                if loss_fn.use_rectified_flow:
                    t = torch.rand(batch_size, device=device)
                else:
                    t = torch.randint(0, loss_fn.timesteps, (batch_size,), device=device)
                
                # Sample noise
                noise = torch.randn_like(x)
                
                # Add noise to images
                x_noisy = loss_fn.q_sample(x, t, noise)
                
                # Classifier-free guidance training
                if classifier_free_guidance and y is not None and model.num_classes > 0:
                    # Use global scheduler for CFG dropout probability
                    current_cfg_prob = global_scheduler.get_value("cfg_dropout_prob") if "cfg_dropout_prob" in global_scheduler._schedules else cfg_dropout_prob
                    mask = torch.rand(batch_size, device=device) < current_cfg_prob
                    y_input = y.clone()
                    y_input[mask] = 0  # null class
                else:
                    y_input = y
                
                # Forward pass with mixed precision and FP4
                with memory_efficient_context():
                    autocast_context = torch.cuda.amp.autocast() if mixed_precision else torch.no_grad.__class__()
                    
                    with autocast_context:
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                            if model.num_classes > 0:
                                model_output, balance_loss = model(x_noisy, t, y_input)
                            else:
                                model_output, balance_loss = model(x_noisy, t)
                    
                    # Compute target based on parameterization
                    if loss_fn.use_v_parameterization:
                        target = loss_fn.get_v_target(x, noise, t)
                    else:
                        target = noise  # Standard noise prediction
                    
                    # Handle learned variance
                    if model.learn_sigma:
                        noise_pred, _ = model_output.chunk(2, dim=1)
                        main_loss = loss_fn(noise_pred, target, t)
                    else:
                        main_loss = loss_fn(model_output, target, t)
                    
                    # Get balance loss coefficient from global scheduler
                    balance_coeff = global_scheduler.get_value("entropy_coeff") if "entropy_coeff" in global_scheduler._schedules else 0.01
                    total_loss = main_loss + balance_coeff * balance_loss
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = total_loss / accumulation_steps
                
                # Backward pass
                if mixed_precision:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Gradient step
                if (batch_idx + 1) % accumulation_steps == 0:
                    if mixed_precision:
                        scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    if gradient_clip_val > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip_val
                        )
                        training_stats['gradient_norms'].append(total_norm.item())
                    
                    # Optimizer step
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Step schedulers
                    if lr_scheduler:
                        lr_scheduler.step()
                        training_stats['learning_rates'].append(lr_scheduler.get_last_lr()[0])
                    
                    # Step global scheduler
                    global_scheduler.step()
                    global_step += 1
                    
                    # Track scheduled values
                    if batch_idx % (log_interval * 2) == 0:
                        scheduled_values = {}
                        for schedule_name in global_scheduler._schedules.keys():
                            scheduled_values[schedule_name] = global_scheduler.get_value(schedule_name)
                        training_stats['global_scheduler_values'][global_step] = scheduled_values
                
                # Statistics
                epoch_loss += main_loss.item()
                epoch_balance_loss += balance_loss.item()
                num_batches += 1
                
                # Logging with scheduled values
                if batch_idx % log_interval == 0:
                    # Memory monitoring
                    memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    training_stats['memory_usage'].append(memory_used)
                    
                    # Throughput
                    batch_time = time.time() - batch_start_time
                    throughput = batch_size / batch_time
                    training_stats['throughput_history'].append(throughput)
                    
                    # Current scheduled values
                    current_lr = training_stats['learning_rates'][-1] if training_stats['learning_rates'] else optimizer.param_groups[0]['lr']
                    entropy_coeff = global_scheduler.get_value("entropy_coeff") if "entropy_coeff" in global_scheduler._schedules else 0.01
                    
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: "
                          f"Loss={main_loss.item():.4f}, "
                          f"Balance={balance_loss.item():.4f}, "
                          f"Total={total_loss.item():.4f}, "
                          f"LR={current_lr:.2e}, "
                          f"EntCoeff={entropy_coeff:.4f}, "
                          f"Step={global_step}, "
                          f"Mem={memory_used:.1f}GB, "
                          f"Throughput={throughput:.1f}/s")
                    
                    # Show QAF status if available
                    if hasattr(model, 'qaf_mode') and model.qaf_mode:
                        print("    🔧 Training in QAF mode (FP4→BF16)")
                
                # Checkpointing
                if checkpoint_path and global_step % save_interval == 0 and global_step > 0:
                    checkpoint_data = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_scheduler_state_dict': global_scheduler.state_dict(),
                        'training_stats': training_stats,
                        'loss': epoch_loss / max(num_batches, 1),
                    }
                    
                    if mixed_precision:
                        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                    if lr_scheduler:
                        checkpoint_data['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
                    
                    os.makedirs(checkpoint_path, exist_ok=True)
                    checkpoint_file = f"{checkpoint_path}/checkpoint_step_{global_step}.pt"
                    torch.save(checkpoint_data, checkpoint_file)
                    print(f"💾 Checkpoint saved: {checkpoint_file}")
                    
                    # Save best model
                    current_loss = epoch_loss / max(num_batches, 1)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_file = f"{checkpoint_path}/best_model.pt"
                        torch.save(checkpoint_data, best_file)
                        print(f"⭐ Best model updated: {best_file}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_balance_loss = epoch_balance_loss / max(num_batches, 1)
            epoch_throughput = samples_processed / epoch_time
            
            training_stats['epoch_losses'].append(avg_loss)
            training_stats['balance_losses'].append(avg_balance_loss)
            
            print(f"\n📈 Epoch {epoch} Summary:")
            print(f"   Avg Loss: {avg_loss:.4f}")
            print(f"   Avg Balance: {avg_balance_loss:.4f}")
            print(f"   Time: {epoch_time:.1f}s")
            print(f"   Throughput: {epoch_throughput:.1f} samples/s")
            print(f"   Global Step: {global_step}")
            
            # Show current scheduled values
            if global_scheduler._schedules:
                print(f"   Scheduled Values:")
                for name, value in [(k, global_scheduler.get_value(k)) for k in list(global_scheduler._schedules.keys())[:5]]:
                    print(f"     {name}: {value:.4f}")
            print("-" * 80)
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
        if checkpoint_path:
            interrupt_file = f"{checkpoint_path}/interrupted_step_{global_step}.pt"
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_scheduler_state_dict': global_scheduler.state_dict(),
                'training_stats': training_stats
            }, interrupt_file)
            print(f"💾 Interrupt checkpoint: {interrupt_file}")
    
    print(f"\n🎯 Training Complete!")
    print(f"   Final Loss: {training_stats['epoch_losses'][-1]:.4f}")
    print(f"   Total Steps: {global_step}")
    print(f"   Best Loss: {best_loss:.4f}")
    
    return training_stats



def enhanced_benchmark_model(model: DiT, input_shape: Tuple[int, ...], device: str = "cuda",
                           num_warmup: int = 10, num_runs: int = 100, 
                           batch_sizes: List[int] = None,
                           test_fp4: bool = True, test_compile: bool = True) -> Dict[str, Any]:
    """Enhanced benchmarking with comprehensive performance analysis."""
    
    model.eval()
    model.to(device)
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16] if input_shape[0] == 1 else [input_shape[0]]
    
    results = {}
    
    # Test different configurations
    configurations = [
        {"name": "baseline", "compile": False, "fp4": False},
    ]
    
    if test_compile:
        configurations.append({"name": "compiled", "compile": True, "fp4": False})
    
    if test_fp4:
        configurations.extend([
            {"name": "fp4", "compile": False, "fp4": True},
            {"name": "compiled_fp4", "compile": True, "fp4": True}
        ])
    
    for config in configurations:
        config_name = config["name"]
        print(f"\n🔍 Benchmarking configuration: {config_name}")
        
        # Setup model for this configuration
        test_model = model
        if config["compile"]:
            test_model = torch.compile(model, mode='max-autotune', dynamic=True)
        
        config_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create test inputs
            channels, height, width = input_shape[1:]
            x = torch.randn(batch_size, channels, height, width, device=device)
            t = torch.randint(0, 1000, (batch_size,), device=device)
            y = torch.randint(0, 10, (batch_size,), device=device) if model.num_classes > 0 else None
            
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    if config["fp4"]:
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                            _ = test_model(x, t, y) if y is not None else test_model(x, t)
                    else:
                        _ = test_model(x, t, y) if y is not None else test_model(x, t)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            peak_memory = 0
            
            with torch.no_grad():
                for i in range(num_runs):
                    if config["fp4"]:
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                            _ = test_model(x, t, y) if y is not None else test_model(x, t)
                    else:
                        _ = test_model(x, t, y) if y is not None else test_model(x, t)
                    
                    # Track peak memory
                    current_memory = torch.cuda.memory_allocated()
                    peak_memory = max(peak_memory, current_memory)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            throughput = batch_size / avg_time
            
            config_results[f'batch_{batch_size}'] = {
                'avg_inference_time_ms': avg_time * 1000,
                'throughput_samples_per_sec': throughput,
                'peak_memory_gb': peak_memory / 1024**3,
                'memory_per_sample_mb': (peak_memory / batch_size) / 1024**2
            }
        
        results[config_name] = config_results
    
    # Performance comparison
    print(f"\n📊 Performance Summary:")
    for config_name, config_results in results.items():
        print(f"\n{config_name.upper()}:")
        for batch_key, metrics in config_results.items():
            batch_size = int(batch_key.split('_')[1])
            print(f"  Batch {batch_size}: "
                  f"{metrics['avg_inference_time_ms']:.1f}ms, "
                  f"{metrics['throughput_samples_per_sec']:.1f} samples/s, "
                  f"{metrics['peak_memory_gb']:.2f}GB")
    
    # Calculate speedups
    if 'baseline' in results and len(results) > 1:
        print(f"\n🚀 Speedup Analysis (vs baseline):")
        baseline = results['baseline']
        for config_name, config_results in results.items():
            if config_name == 'baseline':
                continue
            print(f"\n{config_name}:")
            for batch_key in baseline:
                baseline_time = baseline[batch_key]['avg_inference_time_ms']
                config_time = config_results[batch_key]['avg_inference_time_ms']
                speedup = baseline_time / config_time
                memory_ratio = config_results[batch_key]['peak_memory_gb'] / baseline[batch_key]['peak_memory_gb']
                print(f"  Batch {batch_key.split('_')[1]}: {speedup:.2f}x speed, {memory_ratio:.2f}x memory")
    
    return results


# Utility function for efficient checkpointing
def save_checkpoint_efficiently(checkpoint_data: Dict[str, Any], filepath: str, 
                               compress: bool = True, async_save: bool = False):
    """Save checkpoint with compression and async support."""
    import pickle
    import gzip
    import threading
    
    def _save():
        try:
            if compress:
                with gzip.open(f"{filepath}.gz", 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                torch.save(checkpoint_data, filepath)
            print(f"✅ Checkpoint saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
    
    if async_save:
        thread = threading.Thread(target=_save)
        thread.start()
        return thread
    else:
        _save()


# Memory monitoring utilities
class MemoryMonitor:
    """Real-time memory monitoring for training."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.peak_memory = 0
        self.memory_history = []
    
    def update(self):
        if "cuda" in self.device:
            current_memory = torch.cuda.memory_allocated() / 1024**3
            self.peak_memory = max(self.peak_memory, current_memory)
            self.memory_history.append(current_memory)
            return current_memory
        return 0
    
    def get_stats(self):
        if not self.memory_history:
            return {}
        
        return {
            'current_gb': self.memory_history[-1],
            'peak_gb': self.peak_memory,
            'average_gb': sum(self.memory_history) / len(self.memory_history),
            'samples': len(self.memory_history)
        }
    
    def reset(self):
        self.peak_memory = 0
        self.memory_history.clear()


# Example usage and training script
def main_training_with_global_scheduler():
    """Example training with GlobalScheduler integration."""
    
    from ..core.Model import DiT
    from ..enhanced.global_scheduler import get_global_scheduler
    
    # Setup model
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,  
        num_classes=1000
    )
    
    # Your existing GlobalScheduler will handle the scheduling
    global_scheduler = get_global_scheduler()
    
    # Add any additional schedules specific to training
    global_scheduler.register_schedule("cfg_dropout_prob", 0.15, 0.05, 0, 8000)
    global_scheduler.register_schedule("learning_rate_scale", 1.0, 0.1, 10000, 20000)
    
    # Create dataset and dataloader
    dataset = AdaptiveMemoryDataset(
        data=torch.randn(10000, 4, 32, 32),
        labels=torch.randint(0, 1000, (10000,)),
        preload_to_gpu=False
    )
    
    dataloader = OptimizedDataLoader(
        dataset=dataset,
        batch_size=32,  # Will use FP4, so can be larger
        num_workers=4
    )
    
    # Create advanced optimizer
    optimizer = create_optimizer_advanced(
        model=model,
        learning_rate=2e-4,  # Can be higher with FP4
        optimizer_type="adamw",
        weight_decay=0.01,
        fused=True
    )
    
    # Create LR scheduler (works alongside GlobalScheduler)
    lr_scheduler = create_advanced_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine_with_warmup",
        num_training_steps=20000,
        warmup_steps=1000
    )
    
    # Enhanced loss function
    loss_fn = EnhancedDiffusionLoss(
        loss_type="mse",
        beta_schedule="cosine",
        use_snr_weighting=True,
        snr_gamma=5.0
    )
    
    # Train with GlobalScheduler integration
    training_stats = train_with_global_scheduler_integration(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        num_epochs=100,
        classifier_free_guidance=True,
        accumulation_steps=2,
        checkpoint_path="./checkpoints",
        mixed_precision=True
    )
    
    print("🎉 Training completed with GlobalScheduler!")
    return training_stats, global_scheduler


if __name__ == "__main__":
    main_training_with_global_scheduler()
