import torch
import torch.nn as nn
from typing import Dict, Any
import threading  # For intra-process thread-safety
from collections import OrderedDict

class GlobalScheduler(nn.Module):
    """
    Global step scheduler for coordinating all scheduling across DeltaNet components.
    Ensures consistent training phases and parameter annealing across all modules.
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long), persistent=False)
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, float] = OrderedDict()  # For LRU caching
        self._max_cache_size = 100  # Bound cache memory
        self._lock = threading.Lock()  # Intra-process thread safety
        
    def register_schedule(self, name: str, start_val: float, end_val: float, 
                         start_step: int = 0, end_step: int = 4000, 
                         schedule_type: str = "linear"):
        """Register a parameter schedule (idempotent)."""
        with self._lock:
            if name not in self._schedules:
                self._schedules[name] = {
                    "start_val": start_val,
                    "end_val": end_val,
                    "start_step": start_step,
                    "end_step": end_step,
                    "schedule_type": schedule_type
                }
    
    def get_value(self, name: str) -> float:
        """Get current value for a scheduled parameter (with caching)."""
        with self._lock:
            if name not in self._schedules:
                raise ValueError(f"Schedule '{name}' not registered")
            
            step = float(self._global_step.item())
            cache_key = f"{name}_{step}"
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)  # LRU: Move to recent
                return self._cache[cache_key]
            
            schedule = self._schedules[name]
            if step <= schedule["start_step"]:
                val = schedule["start_val"]
            elif step >= schedule["end_step"]:
                val = schedule["end_val"]
            else:
                # Linear interpolation (fused computation)
                progress = (step - schedule["start_step"]) / max(1.0, schedule["end_step"] - schedule["start_step"])
                val = schedule["start_val"] + progress * (schedule["end_val"] - schedule["start_val"])
            
            self._cache[cache_key] = val
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)  # Evict least recently used
            return val
    
    def step(self):
        """Increment global step counter."""
        with self._lock:
            self._global_step += 1
            self._cache.clear()  # Invalidate cache on step
    
    def get_step(self) -> int:
        """Get current global step."""
        return int(self._global_step.item())
    
    def reset(self):
        """Reset global step counter."""
        self._global_step.zero_()
        self._cache.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Custom state_dict to capture the buffer and schedules."""
        return {
            "_global_step": self._global_step,
            "_schedules": self._schedules  # Serialize schedules as dict (idempotent)
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Custom load_state_dict to restore buffer and schedules."""
        with self._lock:
            self._global_step = state_dict["_global_step"]
            self._schedules = state_dict["_schedules"]
            self._cache.clear()  # Invalidate cache on load

# Global singleton instance
_global_scheduler = None
_scheduler_lock = threading.Lock()

def get_global_scheduler() -> GlobalScheduler:
    """Get the global scheduler instance (thread-safe)."""
    global _global_scheduler
    with _scheduler_lock:
        if _global_scheduler is None:
            _global_scheduler = GlobalScheduler()
    return _global_scheduler

def register_default_schedules(scheduler: GlobalScheduler):
    """Register default schedules used across DeltaNet modules (idempotent)."""
    # Progressive pruning schedules
    scheduler.register_schedule("prune_threshold", 0.0, 1e-3, 2000, 4000)
    scheduler.register_schedule("entropy_coeff", 0.02, 0.0, 0, 4000)
    
    # Context gating schedules
    scheduler.register_schedule("epsilon_floor", 0.10, 0.025, 0, 4000)
    scheduler.register_schedule("entropy_reg", 0.02, 0.001, 0, 12000)
    scheduler.register_schedule("untie_factor", 0.0, 1.0, 1000, 4000)
    
    # Token adaptive floor schedules
    scheduler.register_schedule("token_floor", 0.05, 0.02, 0, 4000)
    
    # MoE temperature schedules
    scheduler.register_schedule("moe_temperature", 1.0, 0.1, 0, 8000)
    
    # Modulation temperature
    scheduler.register_schedule("mod_temperature", 1.0, 0.3, 0, 10000)
