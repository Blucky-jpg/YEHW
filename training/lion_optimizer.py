"""
Lion Optimizer - Optimized for Low Precision (FP4/FP8) Training
Paper: Symbolic Discovery of Optimization Algorithms (https://arxiv.org/abs/2302.06675)

Lion is particularly good for low precision training because:
1. Uses sign operations which are robust to quantization
2. Requires less memory than Adam (only momentum, no variance)
3. More stable with FP4/FP8 weights
4. Often converges faster than Adam with proper tuning
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional, Callable


class Lion(Optimizer):
    """
    Lion optimizer for FP4/FP8 low precision training.
    
    Lion uses the sign of the gradient update, making it more robust to 
    quantization noise than Adam/AdamW. It only tracks momentum (not variance),
    reducing memory by 33% compared to Adam.
    
    Args:
        params: Model parameters
        lr: Learning rate (typically 3-10x smaller than Adam, e.g., 1e-4 for Adam -> 1e-5 to 3e-5 for Lion)
        betas: Coefficients for computing running averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
        use_triton: Use Triton kernel for faster execution if available (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,  # Set to False by default, enable if Triton is installed
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        
        super().__init__(params, defaults)
        
        # Try to import Triton kernel if requested
        self.use_triton = use_triton and self._setup_triton()
    
    def _setup_triton(self) -> bool:
        """Setup Triton kernel for faster execution."""
        try:
            import triton
            import triton.language as tl
            
            @triton.jit
            def lion_kernel(
                p_ptr, grad_ptr, m_ptr,
                lr, beta1, beta2, weight_decay,
                N,
                BLOCK_SIZE: tl.constexpr,
            ):
                pid = tl.program_id(axis=0)
                idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = idx < N
                
                # Load values
                p = tl.load(p_ptr + idx, mask=mask)
                grad = tl.load(grad_ptr + idx, mask=mask)
                m = tl.load(m_ptr + idx, mask=mask)
                
                # Lion update
                # c = beta1 * m + (1 - beta1) * grad
                c = beta1 * m + (1.0 - beta1) * grad
                
                # Weight decay
                if weight_decay != 0:
                    p = p * (1.0 - lr * weight_decay)
                
                # Update with sign
                p = p - lr * tl.libdevice.sign(c)
                
                # Update momentum
                m = beta2 * m + (1.0 - beta2) * grad
                
                # Store back
                tl.store(p_ptr + idx, p, mask=mask)
                tl.store(m_ptr + idx, m, mask=mask)
            
            self.lion_kernel = lion_kernel
            return True
        except ImportError:
            return False
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Use FP32 for optimizer state even with FP4/FP8 model
                # This is crucial for stability
                grad = p.grad.float() if p.grad.dtype != torch.float32 else p.grad
                
                # Get hyperparameters
                lr = group['lr']
                weight_decay = group['weight_decay']
                beta1, beta2 = group['betas']
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    # Initialize momentum buffer
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                
                exp_avg = state['exp_avg']
                
                if self.use_triton and p.is_cuda and p.is_contiguous():
                    # Use Triton kernel for faster execution
                    try:
                        import triton
                        N = p.numel()
                        BLOCK_SIZE = 1024
                        grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
                        
                        self.lion_kernel[grid](
                            p.data.data_ptr(),
                            grad.data_ptr(),
                            exp_avg.data_ptr(),
                            lr, beta1, beta2, weight_decay,
                            N,
                            BLOCK_SIZE=BLOCK_SIZE,
                        )
                    except (ImportError, AttributeError):
                        # Fallback to PyTorch implementation if Triton fails
                        self.use_triton = False
                        # Compute update: c = β1 * m_t + (1 - β1) * g_t
                        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                        
                        # Apply weight decay directly to weights
                        if weight_decay != 0:
                            p.data.mul_(1 - lr * weight_decay)
                        
                        # Apply update with sign operation (robust to quantization)
                        p.data.add_(update.sign(), alpha=-lr)
                        
                        # Update momentum: m_t+1 = β2 * m_t + (1 - β2) * g_t
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                else:
                    # Standard PyTorch implementation
                    # Compute update: c = β1 * m_t + (1 - β1) * g_t
                    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                    
                    # Apply weight decay directly to weights
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    
                    # Apply update with sign operation (robust to quantization)
                    p.data.add_(update.sign(), alpha=-lr)
                    
                    # Update momentum: m_t+1 = β2 * m_t + (1 - β2) * g_t
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss
    
    def state_dict(self):
        """Return the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        state_dict['use_triton'] = self.use_triton
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load the optimizer state."""
        self.use_triton = state_dict.pop('use_triton', False)
        super().load_state_dict(state_dict)


class LionW(Lion):
    """
    Lion with decoupled weight decay (similar to AdamW).
    
    This version applies weight decay before the gradient update,
    which often works better for transformer models.
    """
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step with decoupled weight decay."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.float() if p.grad.dtype != torch.float32 else p.grad
                
                lr = group['lr']
                weight_decay = group['weight_decay']
                beta1, beta2 = group['betas']
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                
                exp_avg = state['exp_avg']
                
                # Compute update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                
                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Apply update with sign
                p.data.add_(update.sign(), alpha=-lr)
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


def create_lion_optimizer(
    model,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.01,
    use_decoupled: bool = True,
    use_triton: bool = False,  # Set to False by default, enable manually if Triton is installed
) -> Optimizer:
    """
    Create Lion optimizer with recommended settings for FP4/FP8 training.
    
    Args:
        model: The model to optimize
        lr: Learning rate (use 3-10x smaller than Adam)
        betas: Beta coefficients
        weight_decay: Weight decay coefficient
        use_decoupled: Use decoupled weight decay (LionW)
        use_triton: Use Triton kernels if available
    
    Returns:
        Configured Lion optimizer
    
    Example:
        >>> optimizer = create_lion_optimizer(
        ...     model,
        ...     lr=3e-5,  # 3-10x smaller than Adam
        ...     weight_decay=0.01,
        ... )
    """
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Choose optimizer variant
    optimizer_cls = LionW if use_decoupled else Lion
    
    return optimizer_cls(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        use_triton=use_triton,
    )


# Learning rate recommendations for Lion
LION_LR_RECOMMENDATIONS = {
    'adam_to_lion': {
        1e-3: (1e-4, 3e-4),   # Adam 1e-3 -> Lion 1e-4 to 3e-4
        3e-4: (3e-5, 1e-4),   # Adam 3e-4 -> Lion 3e-5 to 1e-4
        1e-4: (1e-5, 3e-5),   # Adam 1e-4 -> Lion 1e-5 to 3e-5
        3e-5: (3e-6, 1e-5),   # Adam 3e-5 -> Lion 3e-6 to 1e-5
    },
    'model_size': {
        'small': 3e-4,    # < 100M params
        'medium': 1e-4,   # 100M - 1B params
        'large': 3e-5,    # 1B - 10B params
        'xlarge': 1e-5,   # > 10B params
    },
    'fp4_fp8': {
        'conservative': 1e-5,  # Safe starting point
        'standard': 3e-5,      # Good balance
        'aggressive': 1e-4,    # Fast convergence, may be unstable
    }
}


def get_recommended_lion_lr(
    adam_lr: Optional[float] = None,
    model_params: Optional[int] = None,
    precision: str = 'fp4_fp8',
) -> float:
    """
    Get recommended Lion learning rate based on Adam LR or model size.
    
    Args:
        adam_lr: Previous Adam learning rate if converting
        model_params: Number of model parameters
        precision: Training precision mode
    
    Returns:
        Recommended Lion learning rate
    """
    if adam_lr is not None:
        # Find closest Adam LR and get recommendation
        for adam_ref, (lion_min, lion_max) in LION_LR_RECOMMENDATIONS['adam_to_lion'].items():
            if adam_lr >= adam_ref * 0.5:
                return (lion_min + lion_max) / 2
        return 1e-5  # Conservative default
    
    elif model_params is not None:
        # Based on model size
        if model_params < 100_000_000:
            base_lr = LION_LR_RECOMMENDATIONS['model_size']['small']
        elif model_params < 1_000_000_000:
            base_lr = LION_LR_RECOMMENDATIONS['model_size']['medium']
        elif model_params < 10_000_000_000:
            base_lr = LION_LR_RECOMMENDATIONS['model_size']['large']
        else:
            base_lr = LION_LR_RECOMMENDATIONS['model_size']['xlarge']
        
        # Adjust for precision
        if 'fp4' in precision.lower() or 'fp8' in precision.lower():
            base_lr *= 0.5  # More conservative for low precision
        
        return base_lr
    
    else:
        # Default based on precision
        return LION_LR_RECOMMENDATIONS['fp4_fp8']['standard']
