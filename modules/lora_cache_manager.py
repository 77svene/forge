"""LoRA Cache Manager - Streaming LoRA Activation with Dynamic Batching

This module implements batch-aware LoRA management with precomputed weight caching,
progressive loading, and memory-mapped weight storage for efficient multi-LoRA combinations.
"""

import gc
import hashlib
import json
import os
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from modules import devices, paths, shared
from modules.lora import lora_apply, lora_load, network, network_lora
from modules.lora.extra_networks_lora import ExtraNetworkLora
from modules.lora.lora_patches import LoraPatches
from modules.lora.network import Network, NetworkModule
from modules.lora.lora_logger import lora_logger

# Constants
CACHE_VERSION = "1.0.0"
MAX_CACHE_SIZE_GB = 4.0  # Maximum cache size in GB
MAX_BATCH_SIZE = 32  # Maximum batch size for dynamic batching
MEMORY_MAPPED_CACHE = True  # Use memory-mapped files for weight storage
PROGRESSIVE_LOADING_THRESHOLD = 8  # Start progressive loading for batches > this size


@dataclass
class LoRACombination:
    """Represents a unique combination of LoRAs with their weights."""
    lora_names: List[str]
    lora_multipliers: List[float]
    hash_key: str
    
    def __init__(self, lora_names: List[str], lora_multipliers: List[float]):
        self.lora_names = lora_names
        self.lora_multipliers = lora_multipliers
        self.hash_key = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute a unique hash for this LoRA combination."""
        combined = []
        for name, mult in zip(self.lora_names, self.lora_multipliers):
            combined.append(f"{name}:{mult:.6f}")
        combined_str = "|".join(sorted(combined))
        return hashlib.sha256(combined_str.encode()).hexdigest()[:16]
    
    def __eq__(self, other):
        if not isinstance(other, LoRACombination):
            return False
        return self.hash_key == other.hash_key
    
    def __hash__(self):
        return hash(self.hash_key)


@dataclass
class CachedWeights:
    """Cached merged weights for a LoRA combination."""
    combination: LoRACombination
    merged_state_dict: Dict[str, torch.Tensor]
    memory_mapped: bool = False
    mmap_path: Optional[Path] = None
    last_used: float = 0.0
    size_bytes: int = 0
    access_count: int = 0
    
    def update_access(self):
        """Update access timestamp and count."""
        self.last_used = time.time()
        self.access_count += 1


class MemoryMappedWeights:
    """Manages memory-mapped weight storage for efficient loading."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mmap_files: Dict[str, np.memmap] = {}
        self.lock = threading.Lock()
    
    def save_weights(self, state_dict: Dict[str, torch.Tensor], hash_key: str) -> Path:
        """Save weights to memory-mapped file."""
        mmap_path = self.cache_dir / f"{hash_key}.npy"
        
        # Convert state dict to numpy arrays
        numpy_dict = {}
        for key, tensor in state_dict.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()
            numpy_dict[key] = tensor.numpy()
        
        # Create memory-mapped file
        with self.lock:
            # Save metadata
            meta_path = self.cache_dir / f"{hash_key}.meta"
            metadata = {
                "keys": list(numpy_dict.keys()),
                "shapes": {k: list(v.shape) for k, v in numpy_dict.items()},
                "dtypes": {k: str(v.dtype) for k, v in numpy_dict.items()},
                "version": CACHE_VERSION
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            # Save weights
            all_weights = np.concatenate([v.flatten() for v in numpy_dict.values()])
            mmap_file = np.memmap(
                mmap_path,
                dtype=all_weights.dtype,
                mode='w+',
                shape=all_weights.shape
            )
            mmap_file[:] = all_weights[:]
            mmap_file.flush()
            
            self.mmap_files[hash_key] = mmap_file
        
        return mmap_path
    
    def load_weights(self, hash_key: str, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Load weights from memory-mapped file."""
        mmap_path = self.cache_dir / f"{hash_key}.npy"
        meta_path = self.cache_dir / f"{hash_key}.meta"
        
        if not mmap_path.exists() or not meta_path.exists():
            return None
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Load memory-mapped file
            with self.lock:
                if hash_key not in self.mmap_files:
                    self.mmap_files[hash_key] = np.memmap(
                        mmap_path,
                        dtype=np.float32,
                        mode='r'
                    )
                
                mmap_data = self.mmap_files[hash_key]
                
                # Reconstruct state dict
                state_dict = {}
                offset = 0
                for key in metadata["keys"]:
                    shape = metadata["shapes"][key]
                    dtype = getattr(torch, metadata["dtypes"][key].split(".")[-1])
                    numel = np.prod(shape)
                    
                    # Extract from memory-mapped array
                    tensor_data = mmap_data[offset:offset + numel]
                    tensor = torch.from_numpy(tensor_data.copy()).reshape(shape).to(dtype)
                    
                    # Move to target device
                    if device.type != 'cpu':
                        tensor = tensor.to(device)
                    
                    state_dict[key] = tensor
                    offset += numel
                
                return state_dict
        
        except Exception as e:
            lora_logger.warning(f"Failed to load memory-mapped weights for {hash_key}: {e}")
            return None
    
    def cleanup(self, max_age_hours: float = 24.0):
        """Clean up old memory-mapped files."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.lock:
            for hash_key in list(self.mmap_files.keys()):
                meta_path = self.cache_dir / f"{hash_key}.meta"
                if meta_path.exists():
                    file_age = current_time - meta_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        # Close and remove mmap file
                        if hash_key in self.mmap_files:
                            del self.mmap_files[hash_key]
                        
                        # Remove files
                        mmap_path = self.cache_dir / f"{hash_key}.npy"
                        mmap_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)


class DynamicBatchProcessor:
    """Handles dynamic batching for LoRA activation."""
    
    def __init__(self, cache_manager: 'LoRACacheManager'):
        self.cache_manager = cache_manager
        self.batch_queue: List[Tuple[LoRACombination, Any]] = []  # (combination, callback)
        self.batch_lock = threading.Lock()
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def add_to_batch(self, combination: LoRACombination, callback: callable):
        """Add a LoRA combination to the batch queue."""
        with self.batch_lock:
            self.batch_queue.append((combination, callback))
            
            # Trigger batch processing if threshold reached
            if len(self.batch_queue) >= MAX_BATCH_SIZE:
                self._process_batch()
    
    def _process_batch(self):
        """Process the current batch of LoRA combinations."""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            with self.batch_lock:
                if not self.batch_queue:
                    return
                
                # Group by similar combinations for efficiency
                batch_to_process = self.batch_queue[:MAX_BATCH_SIZE]
                self.batch_queue = self.batch_queue[MAX_BATCH_SIZE:]
            
            # Process combinations in parallel
            futures = []
            for combination, callback in batch_to_process:
                future = self.executor.submit(
                    self._process_single_combination,
                    combination,
                    callback
                )
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    lora_logger.error(f"Error in batch processing: {e}")
        
        finally:
            self.processing = False
            
            # Process remaining items if any
            with self.batch_lock:
                if self.batch_queue:
                    self._process_batch()
    
    def _process_single_combination(self, combination: LoRACombination, callback: callable):
        """Process a single LoRA combination."""
        try:
            # Get or compute merged weights
            cached = self.cache_manager.get_merged_weights(combination)
            if cached:
                # Execute callback with cached weights
                callback(cached.merged_state_dict)
            else:
                # Compute and cache weights
                merged_weights = self.cache_manager.compute_merged_weights(combination)
                if merged_weights:
                    callback(merged_weights)
        except Exception as e:
            lora_logger.error(f"Error processing combination {combination.hash_key}: {e}")
    
    def flush(self):
        """Process all remaining items in the batch queue."""
        with self.batch_lock:
            if self.batch_queue:
                self._process_batch()


class LoRACacheManager:
    """
    Manages LoRA weight caching with dynamic batching and progressive loading.
    
    Features:
    - Precomputes and caches merged weights for common LoRA combinations
    - Dynamic batching for efficient multi-LoRA processing
    - Progressive loading for large batches
    - Memory-mapped weight storage for faster switching
    - LRU eviction policy for cache management
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cache: OrderedDict[str, CachedWeights] = OrderedDict()
        self.combination_to_hash: Dict[LoRACombination, str] = {}
        self.cache_lock = threading.RLock()
        self.batch_processor = DynamicBatchProcessor(self)
        
        # Setup cache directory
        self.cache_dir = Path(shared.cmd_opts.lora_cache_dir) if hasattr(shared.cmd_opts, 'lora_cache_dir') else Path(paths.models_path) / "Lora" / "cache"
        self.mmap_manager = MemoryMappedWeights(self.cache_dir / "mmap")
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "computations": 0,
            "evictions": 0,
            "batched_requests": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        lora_logger.info(f"LoRA Cache Manager initialized (enabled: {enabled})")
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        meta_path = self.cache_dir / "cache_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify cache version
                if metadata.get("version") != CACHE_VERSION:
                    lora_logger.info("Cache version mismatch, clearing cache")
                    self.clear_cache()
                    return
                
                # Load cached combinations
                for hash_key, data in metadata.get("combinations", {}).items():
                    combination = LoRACombination(
                        data["lora_names"],
                        data["lora_multipliers"]
                    )
                    self.combination_to_hash[combination] = hash_key
                    
                    # Create cache entry (weights will be loaded on demand)
                    cached = CachedWeights(
                        combination=combination,
                        merged_state_dict={},
                        last_used=data.get("last_used", 0),
                        size_bytes=data.get("size_bytes", 0),
                        access_count=data.get("access_count", 0)
                    )
                    self.cache[hash_key] = cached
                
                lora_logger.info(f"Loaded {len(self.cache)} cached LoRA combinations")
            
            except Exception as e:
                lora_logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / "cache_meta.json"
        
        try:
            metadata = {
                "version": CACHE_VERSION,
                "timestamp": time.time(),
                "combinations": {}
            }
            
            with self.cache_lock:
                for hash_key, cached in self.cache.items():
                    metadata["combinations"][hash_key] = {
                        "lora_names": cached.combination.lora_names,
                        "lora_multipliers": cached.combination.lora_multipliers,
                        "last_used": cached.last_used,
                        "size_bytes": cached.size_bytes,
                        "access_count": cached.access_count
                    }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        except Exception as e:
            lora_logger.warning(f"Failed to save cache metadata: {e}")
    
    def _cleanup_loop(self):
        """Periodically clean up old cache entries."""
        while True:
            time.sleep(300)  # Run every 5 minutes
            try:
                self._cleanup_old_entries()
                self.mmap_manager.cleanup()
            except Exception as e:
                lora_logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_old_entries(self):
        """Remove old cache entries based on LRU policy."""
        with self.cache_lock:
            if not self.cache:
                return
            
            # Calculate total cache size
            total_size = sum(cached.size_bytes for cached in self.cache.values())
            max_size_bytes = MAX_CACHE_SIZE_GB * 1024**3
            
            if total_size <= max_size_bytes:
                return
            
            # Sort by last used (oldest first)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_used
            )
            
            # Remove oldest entries until under size limit
            bytes_to_free = total_size - max_size_bytes
            bytes_freed = 0
            
            for hash_key, cached in sorted_items:
                if bytes_freed >= bytes_to_free:
                    break
                
                # Remove from cache
                self.cache.pop(hash_key)
                
                # Remove from combination mapping
                if cached.combination in self.combination_to_hash:
                    del self.combination_to_hash[cached.combination]
                
                bytes_freed += cached.size_bytes
                self.stats["evictions"] += 1
                
                lora_logger.debug(f"Evicted cache entry {hash_key} ({cached.size_bytes / 1024**2:.2f} MB)")
            
            if bytes_freed > 0:
                lora_logger.info(f"Freed {bytes_freed / 1024**2:.2f} MB from LoRA cache")
                self._save_cache_metadata()
    
    def get_cache_key(self, lora_names: List[str], lora_multipliers: List[float]) -> LoRACombination:
        """Create a cache key from LoRA names and multipliers."""
        return LoRACombination(lora_names, lora_multipliers)
    
    def get_merged_weights(self, combination: LoRACombination) -> Optional[CachedWeights]:
        """Get cached merged weights for a LoRA combination."""
        if not self.enabled:
            return None
        
        with self.cache_lock:
            hash_key = self.combination_to_hash.get(combination)
            if not hash_key or hash_key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            cached = self.cache[hash_key]
            cached.update_access()
            
            # Move to end (most recently used)
            self.cache.move_to_end(hash_key)
            
            # Load weights if not already loaded
            if not cached.merged_state_dict:
                if MEMORY_MAPPED_CACHE and cached.mmap_path:
                    # Load from memory-mapped file
                    device = devices.device
                    cached.merged_state_dict = self.mmap_manager.load_weights(hash_key, device)
                else:
                    # Compute weights
                    cached.merged_state_dict = self.compute_merged_weights(combination)
            
            self.stats["hits"] += 1
            return cached
    
    def compute_merged_weights(self, combination: LoRACombination) -> Optional[Dict[str, torch.Tensor]]:
        """Compute merged weights for a LoRA combination."""
        if not combination.lora_names:
            return None
        
        try:
            self.stats["computations"] += 1
            
            # Load individual LoRA networks
            networks = []
            for lora_name, multiplier in zip(combination.lora_names, combination.lora_multipliers):
                network_on_disk = lora_load.find_network(lora_name)
                if network_on_disk is None:
                    lora_logger.warning(f"LoRA not found: {lora_name}")
                    continue
                
                net = lora_load.load_network(network_on_disk)
                net multiplier = multiplier
                networks.append(net)
            
            if not networks:
                return None
            
            # Get base model state dict
            base_state_dict = shared.sd_model.state_dict()
            
            # Compute merged weights
            merged_state_dict = {}
            
            # Process each layer
            for key in base_state_dict.keys():
                base_weight = base_state_dict[key]
                
                # Skip non-applicable layers
                if not self._is_lora_applicable_layer(key, networks):
                    continue
                
                # Start with base weight
                merged_weight = base_weight.clone()
                
                # Apply each LoRA
                for net in networks:
                    for module in net.modules.values():
                        if module.sd_module is not None and self._module_applies_to_key(module, key):
                            # Apply LoRA delta
                            with torch.no_grad():
                                delta = module.calc_updown(base_weight)
                                if delta is not None:
                                    merged_weight += delta * net.multiplier
                
                merged_state_dict[key] = merged_weight
            
            # Cache the merged weights
            self._cache_weights(combination, merged_state_dict)
            
            return merged_state_dict
        
        except Exception as e:
            lora_logger.error(f"Error computing merged weights: {e}")
            return None
    
    def _is_lora_applicable_layer(self, key: str, networks: List[Network]) -> bool:
        """Check if any LoRA in the list applies to this layer."""
        for net in networks:
            for module in net.modules.values():
                if module.sd_module is not None and self._module_applies_to_key(module, key):
                    return True
        return False
    
    def _module_applies_to_key(self, module: NetworkModule, key: str) -> bool:
        """Check if a LoRA module applies to a specific state dict key."""
        if module.sd_module is None:
            return False
        
        # Get the module's target key
        target_key = module.sd_module_name
        
        # Check if the key matches
        if target_key == key:
            return True
        
        # Check for partial matches (e.g., weight vs bias)
        if key.startswith(target_key):
            return True
        
        return False
    
    def _cache_weights(self, combination: LoRACombination, state_dict: Dict[str, torch.Tensor]):
        """Cache merged weights with optional memory mapping."""
        hash_key = combination.hash_key
        
        with self.cache_lock:
            # Calculate size
            size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
            
            # Create cache entry
            cached = CachedWeights(
                combination=combination,
                merged_state_dict=state_dict,
                size_bytes=size_bytes,
                last_used=time.time()
            )
            
            # Save to memory-mapped file if enabled
            if MEMORY_MAPPED_CACHE:
                try:
                    mmap_path = self.mmap_manager.save_weights(state_dict, hash_key)
                    cached.mmap_path = mmap_path
                    cached.memory_mapped = True
                    
                    # Clear from memory to save RAM
                    cached.merged_state_dict = {}
                except Exception as e:
                    lora_logger.warning(f"Failed to create memory-mapped weights: {e}")
            
            # Add to cache
            self.cache[hash_key] = cached
            self.combination_to_hash[combination] = hash_key
            
            # Move to end (most recently used)
            self.cache.move_to_end(hash_key)
            
            # Save metadata
            self._save_cache_metadata()
    
    def request_merged_weights(self, combination: LoRACombination, callback: callable):
        """
        Request merged weights with dynamic batching.
        
        This method batches multiple requests for efficiency and uses
        progressive loading for large batches.
        """
        if not self.enabled:
            # Fall back to immediate computation
            weights = self.compute_merged_weights(combination)
            if weights:
                callback(weights)
            return
        
        # Check cache first
        cached = self.get_merged_weights(combination)
        if cached and cached.merged_state_dict:
            callback(cached.merged_state_dict)
            return
        
        # Add to batch queue
        self.batch_processor.add_to_batch(combination, callback)
        self.stats["batched_requests"] += 1
    
    def prefetch_combinations(self, combinations: List[LoRACombination]):
        """Prefetch multiple LoRA combinations for progressive loading."""
        if not self.enabled:
            return
        
        def prefetch_worker(combination):
            try:
                # Check if already cached
                cached = self.get_merged_weights(combination)
                if not cached:
                    # Compute and cache
                    self.compute_merged_weights(combination)
            except Exception as e:
                lora_logger.error(f"Error prefetching combination: {e}")
        
        # Use thread pool for parallel prefetching
        with ThreadPoolExecutor(max_workers=2) as executor:
            for combination in combinations:
                executor.submit(prefetch_worker, combination)
    
    def clear_cache(self):
        """Clear all cached weights."""
        with self.cache_lock:
            self.cache.clear()
            self.combination_to_hash.clear()
            
            # Clear memory-mapped files
            if self.cache_dir.exists():
                for file in self.cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
            
            # Reset statistics
            self.stats = {
                "hits": 0,
                "misses": 0,
                "computations": 0,
                "evictions": 0,
                "batched_requests": 0
            }
            
            lora_logger.info("LoRA cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_size = sum(cached.size_bytes for cached in self.cache.values())
            
            return {
                **self.stats,
                "cache_entries": len(self.cache),
                "cache_size_mb": total_size / 1024**2,
                "cache_hit_rate": (
                    self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                    if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
                )
            }
    
    def flush(self):
        """Flush all pending batch operations."""
        self.batch_processor.flush()


# Global cache manager instance
cache_manager = LoRACacheManager(enabled=not shared.cmd_opts.lora_no_cache if hasattr(shared.cmd_opts, 'lora_no_cache') else True)


def apply_lora_with_cache(
    model,
    lora_names: List[str],
    lora_multipliers: List[float],
    extra_network_data: Optional[Dict] = None
):
    """
    Apply LoRA to model using cached merged weights when available.
    
    This is the main integration point with the existing LoRA system.
    """
    if not cache_manager.enabled or not lora_names:
        # Fall back to standard LoRA application
        return _apply_lora_standard(model, lora_names, lora_multipliers, extra_network_data)
    
    # Create combination key
    combination = cache_manager.get_cache_key(lora_names, lora_multipliers)
    
    # Try to get cached weights
    cached = cache_manager.get_merged_weights(combination)
    if cached and cached.merged_state_dict:
        # Apply cached merged weights
        lora_logger.debug(f"Applying cached LoRA weights for {len(lora_names)} LoRAs")
        _apply_merged_weights(model, cached.merged_state_dict)
        return True
    
    # Request weights with batching
    result_holder = [None]
    
    def weights_callback(merged_weights):
        result_holder[0] = merged_weights
    
    cache_manager.request_merged_weights(combination, weights_callback)
    
    # Wait for result (with timeout)
    start_time = time.time()
    while result_holder[0] is None and time.time() - start_time < 5.0:
        time.sleep(0.01)
    
    if result_holder[0]:
        _apply_merged_weights(model, result_holder[0])
        return True
    else:
        # Timeout or error, fall back to standard application
        lora_logger.warning("LoRA cache request timed out, falling back to standard application")
        return _apply_lora_standard(model, lora_names, lora_multipliers, extra_network_data)


def _apply_lora_standard(
    model,
    lora_names: List[str],
    lora_multipliers: List[float],
    extra_network_data: Optional[Dict] = None
):
    """Apply LoRA using standard method (fallback)."""
    # This integrates with the existing LoRA application code
    from modules.lora import lora_apply
    
    # Create extra network data if not provided
    if extra_network_data is None:
        extra_network_data = {"lora": {}}
    
    # Add LoRAs to extra network data
    for name, multiplier in zip(lora_names, lora_multipliers):
        extra_network_data["lora"][name] = {
            "multiplier": multiplier,
            "weights": None
        }
    
    # Apply using existing system
    lora_apply.apply_lora_patches(model, extra_network_data)
    return True


def _apply_merged_weights(model, merged_state_dict: Dict[str, torch.Tensor]):
    """Apply precomputed merged weights to model."""
    # Get current model state dict
    model_state_dict = model.state_dict()
    
    # Apply merged weights
    for key, merged_weight in merged_state_dict.items():
        if key in model_state_dict:
            # Ensure device and dtype match
            target = model_state_dict[key]
            if merged_weight.device != target.device:
                merged_weight = merged_weight.to(target.device)
            if merged_weight.dtype != target.dtype:
                merged_weight = merged_weight.to(target.dtype)
            
            # Update model weight
            model_state_dict[key] = merged_weight
    
    # Load updated state dict
    model.load_state_dict(model_state_dict, strict=False)


def process_lora_batch(
    prompts: List[str],
    lora_configs: List[Dict[str, Any]],
    model
) -> List[torch.Tensor]:
    """
    Process a batch of prompts with different LoRA configurations.
    
    This function implements dynamic batching for LoRA activation,
    grouping prompts by LoRA combination for efficient processing.
    """
    if not cache_manager.enabled:
        # Fall back to sequential processing
        results = []
        for prompt, config in zip(prompts, lora_configs):
            lora_names = config.get("lora_names", [])
            lora_multipliers = config.get("lora_multipliers", [])
            apply_lora_with_cache(model, lora_names, lora_multipliers)
            # Process prompt (placeholder - actual processing would be here)
            results.append(None)
        return results
    
    # Group prompts by LoRA combination
    combination_groups = defaultdict(list)
    for idx, (prompt, config) in enumerate(zip(prompts, lora_configs)):
        lora_names = config.get("lora_names", [])
        lora_multipliers = config.get("lora_multipliers", [])
        combination = cache_manager.get_cache_key(lora_names, lora_multipliers)
        combination_groups[combination].append((idx, prompt, config))
    
    # Prefetch all combinations
    combinations = list(combination_groups.keys())
    cache_manager.prefetch_combinations(combinations)
    
    # Process groups
    results = [None] * len(prompts)
    
    for combination, group in combination_groups.items():
        # Get or compute merged weights for this combination
        cached = cache_manager.get_merged_weights(combination)
        if cached and cached.merged_state_dict:
            # Apply weights once for the entire group
            _apply_merged_weights(model, cached.merged_state_dict)
            
            # Process all prompts in this group
            for idx, prompt, config in group:
                # Process prompt (placeholder - actual processing would be here)
                results[idx] = None  # Replace with actual result
        else:
            # Fall back to standard processing for this group
            for idx, prompt, config in group:
                lora_names = config.get("lora_names", [])
                lora_multipliers = config.get("lora_multipliers", [])
                apply_lora_with_cache(model, lora_names, lora_multipliers)
                results[idx] = None  # Replace with actual result
    
    # Flush any pending batch operations
    cache_manager.flush()
    
    return results


def get_lora_cache_stats() -> Dict[str, Any]:
    """Get LoRA cache statistics."""
    return cache_manager.get_stats()


def clear_lora_cache():
    """Clear the LoRA cache."""
    cache_manager.clear_cache()


# Integration with existing LoRA system
def patch_lora_system():
    """Patch the existing LoRA system to use caching."""
    try:
        from modules.lora import lora_apply
        
        # Store original function
        original_apply = lora_apply.apply_lora_patches
        
        # Create patched function
        def patched_apply_lora_patches(model, extra_network_data):
            """Patched version that uses caching when available."""
            lora_data = extra_network_data.get("lora", {})
            if not lora_data:
                return original_apply(model, extra_network_data)
            
            # Extract LoRA names and multipliers
            lora_names = []
            lora_multipliers = []
            
            for name, data in lora_data.items():
                lora_names.append(name)
                lora_multipliers.append(data.get("multiplier", 1.0))
            
            # Use cached application
            if cache_manager.enabled and lora_names:
                success = apply_lora_with_cache(
                    model,
                    lora_names,
                    lora_multipliers,
                    extra_network_data
                )
                if success:
                    return
            
            # Fall back to original
            return original_apply(model, extra_network_data)
        
        # Apply patch
        lora_apply.apply_lora_patches = patched_apply_lora_patches
        
        lora_logger.info("LoRA system patched for caching")
    
    except Exception as e:
        lora_logger.error(f"Failed to patch LoRA system: {e}")


# Initialize patching when module loads
if shared.opts.lora_cache_enabled if hasattr(shared.opts, 'lora_cache_enabled') else True:
    patch_lora_system()