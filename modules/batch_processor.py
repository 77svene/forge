"""
Streaming LoRA Activation with Dynamic Batching for forge.
Implements batch-aware LoRA management with caching, progressive loading, and memory-mapped weight storage.
"""

import os
import gc
import sys
import time
import torch
import numpy as np
import hashlib
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import mmap
import pickle
import json

# Import existing LoRA modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import shared, devices, sd_models
from extensions_builtin.Lora import lora, network, network_lora, extra_networks_lora
from extensions_builtin.Lora.lora_patches import LoraPatches


class CacheStrategy(Enum):
    """Caching strategies for LoRA weights."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class LoRACombination:
    """Represents a combination of LoRA models with their weights."""
    loras: List[Tuple[str, float]]  # List of (lora_name, weight)
    hash_key: str = field(init=False)
    merged_weights: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    last_used: float = field(default_factory=time.time)
    use_count: int = field(default=0)
    memory_size: int = field(default=0)
    
    def __post_init__(self):
        """Generate hash key for the combination."""
        sorted_loras = sorted(self.loras, key=lambda x: x[0])
        hash_input = json.dumps(sorted_loras, sort_keys=True)
        self.hash_key = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.use_count += 1


class MemoryMappedWeights:
    """Manages memory-mapped weight storage for LoRA models."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "lora_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.mmap_files: Dict[str, Tuple[str, int]] = {}  # key -> (file_path, file_size)
        self.mmap_handles: Dict[str, mmap.mmap] = {}
        
    def save_weights(self, key: str, weights: Dict[str, torch.Tensor]) -> str:
        """Save weights to memory-mapped file."""
        file_path = os.path.join(self.cache_dir, f"{key}.pt")
        
        # Convert tensors to numpy for efficient storage
        np_weights = {}
        for name, tensor in weights.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()
            np_weights[name] = tensor.numpy()
        
        # Save to file
        with open(file_path, 'wb') as f:
            pickle.dump(np_weights, f, protocol=4)
        
        file_size = os.path.getsize(file_path)
        self.mmap_files[key] = (file_path, file_size)
        return file_path
    
    def load_weights(self, key: str, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Load weights from memory-mapped file."""
        if key not in self.mmap_files:
            raise KeyError(f"No weights found for key: {key}")
        
        file_path, _ = self.mmap_files[key]
        
        # Use memory mapping for efficient loading
        if key not in self.mmap_handles:
            with open(file_path, 'rb') as f:
                self.mmap_handles[key] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Load from memory-mapped file
        mm = self.mmap_handles[key]
        mm.seek(0)
        np_weights = pickle.loads(mm.read())
        
        # Convert back to tensors
        weights = {}
        for name, np_array in np_weights.items():
            tensor = torch.from_numpy(np_array.copy())
            if device:
                tensor = tensor.to(device)
            weights[name] = tensor
        
        return weights
    
    def cleanup(self, keep_recent: int = 5):
        """Clean up old memory-mapped files."""
        if len(self.mmap_handles) > keep_recent:
            # Sort by last access time (would need to track this)
            keys_to_remove = list(self.mmap_handles.keys())[:-keep_recent]
            for key in keys_to_remove:
                self.remove_weights(key)
    
    def remove_weights(self, key: str):
        """Remove weights from cache."""
        if key in self.mmap_handles:
            self.mmap_handles[key].close()
            del self.mmap_handles[key]
        
        if key in self.mmap_files:
            file_path, _ = self.mmap_files[key]
            try:
                os.remove(file_path)
            except:
                pass
            del self.mmap_files[key]
    
    def clear_all(self):
        """Clear all cached weights."""
        for key in list(self.mmap_handles.keys()):
            self.remove_weights(key)
        
        # Clean up directory
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pt'):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except:
                    pass


class BatchLoRAManager:
    """
    Batch-aware LoRA manager with dynamic batching and streaming activation.
    Handles multiple LoRA combinations efficiently for batch generation scenarios.
    """
    
    def __init__(self, 
                 max_cache_size: int = 10,
                 cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 enable_mmap: bool = True,
                 num_workers: int = 2):
        
        self.max_cache_size = max_cache_size
        self.cache_strategy = cache_strategy
        self.enable_mmap = enable_mmap
        self.num_workers = num_workers
        
        # Cache for LoRA combinations
        self.combination_cache: OrderedDict[str, LoRACombination] = OrderedDict()
        
        # Memory-mapped weight storage
        self.mmap_storage = MemoryMappedWeights() if enable_mmap else None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Lock for thread-safe operations
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'combinations_processed': 0,
            'total_merging_time': 0,
            'memory_saved': 0
        }
        
        # Currently active LoRA patches
        self.active_patches: Optional[LoraPatches] = None
        self.active_combination_key: Optional[str] = None
        
        # Preload common LoRA combinations
        self.preload_thread = None
        self.preload_queue = []
        
    def get_cache_key(self, lora_names: List[str], weights: List[float]) -> str:
        """Generate cache key for LoRA combination."""
        combined = list(zip(lora_names, weights))
        combined.sort(key=lambda x: x[0])
        hash_input = json.dumps(combined, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def load_lora_model(self, lora_name: str) -> Optional[network.Network]:
        """Load a LoRA model by name."""
        try:
            lora_path = lora.available_loras.get(lora_name)
            if not lora_path:
                print(f"LoRA not found: {lora_name}")
                return None
            
            # Use existing LoRA loading mechanism
            lora_model = lora.load_lora(lora_name, lora_path)
            return lora_model
        except Exception as e:
            print(f"Error loading LoRA {lora_name}: {e}")
            return None
    
    def merge_lora_weights(self, 
                          lora_models: List[network.Network], 
                          weights: List[float],
                          base_model: torch.nn.Module = None) -> Dict[str, torch.Tensor]:
        """
        Merge multiple LoRA models into combined weights.
        
        Args:
            lora_models: List of LoRA network models
            weights: Corresponding weights for each LoRA
            base_model: Base model to apply LoRAs to (optional)
        
        Returns:
            Dictionary mapping module names to merged weights
        """
        start_time = time.time()
        merged_weights = {}
        
        # Get the actual model to patch
        if base_model is None:
            if shared.sd_model is None:
                raise ValueError("No model loaded")
            base_model = shared.sd_model
        
        # Process each module in the model
        for name, module in base_model.named_modules():
            if not hasattr(module, 'weight'):
                continue
            
            # Start with original weights
            if hasattr(module, 'weight') and module.weight is not None:
                merged_weight = module.weight.data.clone()
                
                # Apply each LoRA
                for lora_model, weight in zip(lora_models, weights):
                    if lora_model is None:
                        continue
                    
                    # Find LoRA modules that affect this layer
                    for lora_name, lora_module in lora_model.modules.items():
                        if hasattr(lora_module, 'calc_updown'):
                            # Calculate LoRA contribution
                            up, down = lora_module.calc_updown(merged_weight)
                            if up is not None and down is not None:
                                # Apply scaled LoRA
                                merged_weight += (up @ down) * weight * lora_module.multiplier
                
                merged_weights[name] = merged_weight
        
        merge_time = time.time() - start_time
        self.stats['total_merging_time'] += merge_time
        
        return merged_weights
    
    def get_or_create_combination(self, 
                                 lora_names: List[str], 
                                 weights: List[float]) -> LoRACombination:
        """Get or create a LoRA combination from cache."""
        cache_key = self.get_cache_key(lora_names, weights)
        
        with self.lock:
            # Check cache
            if cache_key in self.combination_cache:
                combination = self.combination_cache[cache_key]
                combination.update_usage()
                self.stats['cache_hits'] += 1
                
                # Move to end for LRU
                self.combination_cache.move_to_end(cache_key)
                return combination
            
            # Cache miss
            self.stats['cache_misses'] += 1
            
            # Create new combination
            lora_pairs = list(zip(lora_names, weights))
            combination = LoRACombination(loras=lora_pairs)
            
            # Load LoRA models
            lora_models = []
            for lora_name, weight in lora_pairs:
                lora_model = self.load_lora_model(lora_name)
                if lora_model:
                    lora_models.append(lora_model)
                else:
                    lora_models.append(None)
            
            # Merge weights
            merged_weights = self.merge_lora_weights(lora_models, weights)
            combination.merged_weights = merged_weights
            
            # Calculate memory size
            memory_size = sum(w.numel() * w.element_size() for w in merged_weights.values())
            combination.memory_size = memory_size
            
            # Save to memory-mapped storage if enabled
            if self.enable_mmap and self.mmap_storage:
                self.mmap_storage.save_weights(cache_key, merged_weights)
            
            # Add to cache
            self.combination_cache[cache_key] = combination
            self.stats['combinations_processed'] += 1
            
            # Manage cache size
            self._manage_cache_size()
            
            return combination
    
    def _manage_cache_size(self):
        """Manage cache size based on strategy."""
        if len(self.combination_cache) <= self.max_cache_size:
            return
        
        if self.cache_strategy == CacheStrategy.LRU:
            # Remove least recently used
            while len(self.combination_cache) > self.max_cache_size:
                self.combination_cache.popitem(last=False)
        
        elif self.cache_strategy == CacheStrategy.LFU:
            # Remove least frequently used
            items = list(self.combination_cache.items())
            items.sort(key=lambda x: x[1].use_count)
            while len(self.combination_cache) > self.max_cache_size:
                key, _ = items.pop(0)
                self._remove_combination(key)
        
        elif self.cache_strategy == CacheStrategy.FIFO:
            # Remove oldest
            while len(self.combination_cache) > self.max_cache_size:
                self.combination_cache.popitem(last=False)
        
        elif self.cache_strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on usage patterns and memory size
            self._adaptive_cache_management()
    
    def _adaptive_cache_management(self):
        """Adaptive cache management based on usage patterns."""
        if len(self.combination_cache) <= self.max_cache_size:
            return
        
        # Score each combination
        scored_items = []
        current_time = time.time()
        
        for key, combo in self.combination_cache.items():
            # Calculate score based on recency, frequency, and size
            recency_score = 1.0 / (current_time - combo.last_used + 1)
            frequency_score = combo.use_count
            size_penalty = combo.memory_size / (1024 * 1024)  # MB
            
            # Weighted score
            score = (recency_score * 0.4 + 
                    frequency_score * 0.4 - 
                    size_penalty * 0.2)
            
            scored_items.append((score, key, combo))
        
        # Sort by score (lowest first for removal)
        scored_items.sort(key=lambda x: x[0])
        
        # Remove lowest scored items
        while len(self.combination_cache) > self.max_cache_size:
            _, key, _ = scored_items.pop(0)
            self._remove_combination(key)
    
    def _remove_combination(self, key: str):
        """Remove a combination from cache."""
        if key in self.combination_cache:
            combo = self.combination_cache[key]
            
            # Update memory saved statistics
            self.stats['memory_saved'] += combo.memory_size
            
            # Remove from memory-mapped storage
            if self.enable_mmap and self.mmap_storage:
                self.mmap_storage.remove_weights(key)
            
            # Clear merged weights
            if combo.merged_weights:
                for tensor in combo.merged_weights.values():
                    del tensor
                combo.merged_weights = None
            
            # Remove from cache
            del self.combination_cache[key]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def apply_lora_combination(self, 
                              combination: LoRACombination,
                              model: torch.nn.Module = None):
        """Apply a LoRA combination to the model."""
        if model is None:
            if shared.sd_model is None:
                raise ValueError("No model loaded")
            model = shared.sd_model
        
        # Remove existing patches
        self.remove_active_patches()
        
        # Create new patches
        self.active_patches = LoraPatches()
        
        # Apply merged weights
        if combination.merged_weights:
            for name, param in model.named_parameters():
                if name in combination.merged_weights:
                    # Store original weight if not already stored
                    if not hasattr(param, 'original_weight'):
                        param.original_weight = param.data.clone()
                    
                    # Apply merged weight
                    param.data = combination.merged_weights[name].to(param.device)
        
        # Update active combination
        self.active_combination_key = combination.hash_key
        combination.update_usage()
        
        # Move to end of cache (LRU)
        with self.lock:
            if combination.hash_key in self.combination_cache:
                self.combination_cache.move_to_end(combination.hash_key)
    
    def remove_active_patches(self):
        """Remove currently active LoRA patches."""
        if self.active_patches:
            try:
                self.active_patches.restore()
            except:
                pass
            self.active_patches = None
        
        # Restore original weights if needed
        if shared.sd_model:
            for name, param in shared.sd_model.named_parameters():
                if hasattr(param, 'original_weight'):
                    param.data = param.original_weight
                    delattr(param, 'original_weight')
        
        self.active_combination_key = None
    
    def process_batch(self, 
                     batch_requests: List[Dict[str, Any]],
                     progress_callback=None) -> List[Any]:
        """
        Process a batch of requests with different LoRA combinations.
        
        Args:
            batch_requests: List of dicts with 'loras', 'weights', and 'request_data'
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of results for each request
        """
        results = [None] * len(batch_requests)
        
        # Group requests by LoRA combination
        grouped_requests = {}
        for idx, request in enumerate(batch_requests):
            lora_names = request.get('loras', [])
            weights = request.get('weights', [1.0] * len(lora_names))
            
            # Ensure weights list matches lora_names
            if len(weights) < len(lora_names):
                weights.extend([1.0] * (len(lora_names) - len(weights)))
            elif len(weights) > len(lora_names):
                weights = weights[:len(lora_names)]
            
            cache_key = self.get_cache_key(lora_names, weights)
            
            if cache_key not in grouped_requests:
                grouped_requests[cache_key] = {
                    'lora_names': lora_names,
                    'weights': weights,
                    'indices': [],
                    'requests': []
                }
            
            grouped_requests[cache_key]['indices'].append(idx)
            grouped_requests[cache_key]['requests'].append(request)
        
        # Process each group
        total_groups = len(grouped_requests)
        for group_idx, (cache_key, group) in enumerate(grouped_requests.items()):
            if progress_callback:
                progress_callback(group_idx / total_groups, 
                                f"Processing LoRA group {group_idx + 1}/{total_groups}")
            
            try:
                # Get or create combination
                combination = self.get_or_create_combination(
                    group['lora_names'],
                    group['weights']
                )
                
                # Apply combination
                self.apply_lora_combination(combination)
                
                # Process requests in this group
                for idx, request in zip(group['indices'], group['requests']):
                    # Here you would call the actual generation function
                    # For now, we'll just return the request data
                    results[idx] = {
                        'status': 'success',
                        'request': request,
                        'combination_key': cache_key
                    }
            
            except Exception as e:
                # Mark all requests in this group as failed
                for idx in group['indices']:
                    results[idx] = {
                        'status': 'error',
                        'error': str(e),
                        'request': batch_requests[idx]
                    }
        
        # Clean up
        self.remove_active_patches()
        
        return results
    
    def preload_combinations(self, combinations: List[Tuple[List[str], List[float]]]):
        """Preload LoRA combinations in background."""
        def preload_worker():
            for lora_names, weights in combinations:
                try:
                    self.get_or_create_combination(lora_names, weights)
                except Exception as e:
                    print(f"Error preloading combination: {e}")
        
        if self.preload_thread and self.preload_thread.is_alive():
            # Add to queue
            self.preload_queue.extend(combinations)
        else:
            # Start new preload thread
            self.preload_queue = list(combinations)
            self.preload_thread = threading.Thread(target=preload_worker, daemon=True)
            self.preload_thread.start()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self.lock:
            cache_size = len(self.combination_cache)
            total_memory = sum(c.memory_size for c in self.combination_cache.values())
            
            return {
                **self.stats,
                'cache_size': cache_size,
                'total_memory_mb': total_memory / (1024 * 1024),
                'cache_hit_rate': (self.stats['cache_hits'] / 
                                 (self.stats['cache_hits'] + self.stats['cache_misses'] + 1e-10)),
                'average_merge_time': (self.stats['total_merging_time'] / 
                                     (self.stats['combinations_processed'] + 1e-10))
            }
    
    def clear_cache(self):
        """Clear all cached combinations."""
        with self.lock:
            # Remove all combinations
            for key in list(self.combination_cache.keys()):
                self._remove_combination(key)
            
            # Clear memory-mapped storage
            if self.mmap_storage:
                self.mmap_storage.clear_all()
            
            # Reset statistics
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'combinations_processed': 0,
                'total_merging_time': 0,
                'memory_saved': 0
            }
    
    def optimize_for_batch_size(self, batch_size: int):
        """Optimize settings for a specific batch size."""
        if batch_size <= 5:
            self.max_cache_size = 5
            self.cache_strategy = CacheStrategy.LRU
        elif batch_size <= 20:
            self.max_cache_size = 10
            self.cache_strategy = CacheStrategy.ADAPTIVE
        else:
            self.max_cache_size = 15
            self.cache_strategy = CacheStrategy.LFU
            self.num_workers = min(4, os.cpu_count() or 2)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.executor.shutdown(wait=False)
        if self.mmap_storage:
            self.mmap_storage.clear_all()


# Global instance for the webui
batch_lora_manager = BatchLoRAManager()


def process_prompts_with_lora(prompts: List[str], 
                             lora_settings: List[Dict[str, Any]],
                             **generation_kwargs) -> List[Any]:
    """
    Process multiple prompts with different LoRA settings.
    
    Args:
        prompts: List of prompts to process
        lora_settings: List of dicts with 'loras' and 'weights' for each prompt
        **generation_kwargs: Additional generation parameters
    
    Returns:
        List of generated results
    """
    # Prepare batch requests
    batch_requests = []
    for prompt, settings in zip(prompts, lora_settings):
        batch_requests.append({
            'prompt': prompt,
            'loras': settings.get('loras', []),
            'weights': settings.get('weights', []),
            'generation_kwargs': generation_kwargs
        })
    
    # Process batch
    return batch_lora_manager.process_batch(batch_requests)


def preload_common_lora_combinations():
    """Preload commonly used LoRA combinations."""
    # This could be called during webui startup
    # Common combinations could be loaded from a config file
    common_combinations = [
        # Example: (['lora1', 'lora2'], [0.8, 0.6])
    ]
    
    if common_combinations:
        batch_lora_manager.preload_combinations(common_combinations)


# Integration with existing webui
def on_model_loaded():
    """Called when a model is loaded."""
    # Reset manager state
    batch_lora_manager.remove_active_patches()


# Register callbacks if in webui environment
try:
    from modules import script_callbacks
    script_callbacks.on_model_loaded(on_model_loaded)
except ImportError:
    pass


# Command line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch LoRA Processor")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        batch_lora_manager.clear_cache()
        print("Cache cleared")
    
    if args.stats:
        stats = batch_lora_manager.get_statistics()
        print("Batch LoRA Manager Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    if args.test:
        # Test with dummy data
        test_loras = ["dummy_lora1", "dummy_lora2"]
        test_weights = [0.8, 0.6]
        
        print("Testing batch processing...")
        results = batch_lora_manager.process_batch([
            {'loras': test_loras, 'weights': test_weights, 'prompt': 'test1'},
            {'loras': test_loras, 'weights': [0.5, 0.5], 'prompt': 'test2'},
        ])
        
        print(f"Processed {len(results)} requests")
        print("Statistics:", batch_lora_manager.get_statistics())