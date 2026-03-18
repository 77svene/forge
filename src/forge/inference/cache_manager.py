"""
Dynamic Inference Optimization System for forge
Real-time batching, caching, and hardware acceleration based on request patterns
"""

import asyncio
import time
import threading
import queue
import hashlib
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import torch.nn as nn
from enum import Enum

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"
    ADAPTIVE = "adaptive"


class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PRIORITY = "priority"


@dataclass
class RequestPattern:
    """Analyzes and stores request patterns for optimization decisions"""
    arrival_rate: float = 0.0  # requests per second
    sequence_lengths: List[int] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    inter_arrival_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    request_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timestamp: float = field(default_factory=time.time)
    
    def update(self, request: Dict[str, Any]):
        """Update pattern with new request"""
        current_time = time.time()
        if self.inter_arrival_times:
            self.inter_arrival_times.append(current_time - self.timestamp)
        
        self.timestamp = current_time
        
        # Update arrival rate (exponential moving average)
        if len(self.inter_arrival_times) > 1:
            avg_interval = np.mean(list(self.inter_arrival_times))
            self.arrival_rate = 1.0 / max(avg_interval, 0.001)
        
        # Track sequence length
        if "input_length" in request:
            self.sequence_lengths.append(request["input_length"])
            if len(self.sequence_lengths) > 1000:
                self.sequence_lengths.pop(0)
        
        # Track request type
        req_type = request.get("type", "default")
        self.request_types[req_type] += 1
    
    def get_percentile_lengths(self, percentiles: List[float] = [50, 90, 99]) -> Dict[float, int]:
        """Get sequence length percentiles"""
        if not self.sequence_lengths:
            return {p: 0 for p in percentiles}
        
        arr = np.array(self.sequence_lengths)
        return {p: int(np.percentile(arr, p)) for p in percentiles}


@dataclass
class CacheEntry:
    """Represents a cached KV tensor"""
    key: str
    value: torch.Tensor
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    priority: float = 1.0
    
    def __post_init__(self):
        if self.value is not None:
            self.size_bytes = self.value.element_size() * self.value.nelement()


class KVCacheManager:
    """Manages KV cache with dynamic optimization"""
    
    def __init__(self, max_cache_size: int = 10 * 1024 * 1024 * 1024,  # 10GB default
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 device: str = "auto"):
        
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0
        self.policy = policy
        self.device = self._get_device(device)
        
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive policy parameters
        self.hit_rate_window = deque(maxlen=1000)
        self.adaptive_threshold = 0.7  # Switch policies when hit rate drops below
        
        logger.info(f"KVCacheManager initialized with {max_cache_size/1e9:.1f}GB cache on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _generate_key(self, prefix: str, layer_idx: int, head_idx: int) -> str:
        """Generate cache key from components"""
        key_str = f"{prefix}_{layer_idx}_{head_idx}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prefix: str, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """Retrieve from cache"""
        key = self._generate_key(prefix, layer_idx, head_idx)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self.hits += 1
                self.hit_rate_window.append(1)
                return entry.value.to(self.device)
            else:
                self.misses += 1
                self.hit_rate_window.append(0)
                return None
    
    def put(self, prefix: str, layer_idx: int, head_idx: int, value: torch.Tensor, priority: float = 1.0):
        """Store in cache with eviction if needed"""
        key = self._generate_key(prefix, layer_idx, head_idx)
        entry = CacheEntry(key=key, value=value.cpu(), priority=priority)
        
        with self.lock:
            # Check if we need to evict
            while (self.current_cache_size + entry.size_bytes > self.max_cache_size 
                   and len(self.cache) > 0):
                self._evict_one()
            
            # Store new entry
            if key in self.cache:
                self.current_cache_size -= self.cache[key].size_bytes
            
            self.cache[key] = entry
            self.current_cache_size += entry.size_bytes
    
    def _evict_one(self):
        """Evict one entry based on current policy"""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Least Recently Used
            key_to_evict = min(self.cache.keys(), 
                              key=lambda k: self.cache[k].last_access)
        
        elif self.policy == CachePolicy.LFU:
            # Least Frequently Used
            key_to_evict = min(self.cache.keys(),
                              key=lambda k: self.cache[k].access_count)
        
        elif self.policy == CachePolicy.PRIORITY:
            # Lowest priority
            key_to_evict = min(self.cache.keys(),
                              key=lambda k: self.cache[k].priority)
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive: use hit rate to choose policy
            hit_rate = self.get_hit_rate()
            if hit_rate < self.adaptive_threshold:
                # Switch to LRU when hit rate is low
                key_to_evict = min(self.cache.keys(),
                                  key=lambda k: self.cache[k].last_access)
            else:
                # Use LFU when hit rate is good
                key_to_evict = min(self.cache.keys(),
                                  key=lambda k: self.cache[k].access_count)
        
        else:
            key_to_evict = next(iter(self.cache))
        
        # Perform eviction
        entry = self.cache.pop(key_to_evict)
        self.current_cache_size -= entry.size_bytes
        self.evictions += 1
        
        logger.debug(f"Evicted cache entry {key_to_evict[:8]}...")
    
    def get_hit_rate(self) -> float:
        """Calculate current hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size_bytes": self.current_cache_size,
                "size_gb": self.current_cache_size / 1e9,
                "max_size_gb": self.max_cache_size / 1e9,
                "entries": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.get_hit_rate(),
                "evictions": self.evictions,
                "policy": self.policy.value
            }
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.current_cache_size = 0
            logger.info("Cache cleared")


class ContinuousBatcher:
    """Implements continuous batching for dynamic request handling"""
    
    def __init__(self, max_batch_size: int = 32, max_sequence_length: int = 4096,
                 timeout_ms: float = 50.0):
        
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.timeout = timeout_ms / 1000.0
        
        self.request_queue = queue.PriorityQueue()
        self.current_batch: List[Dict[str, Any]] = []
        self.batch_lock = threading.Lock()
        self.running = False
        self.worker_thread = None
        
        # Performance tracking
        self.batch_sizes = deque(maxlen=1000)
        self.latencies = deque(maxlen=1000)
        
        logger.info(f"ContinuousBatcher initialized: max_batch={max_batch_size}, timeout={timeout_ms}ms")
    
    def start(self):
        """Start the batching worker"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        logger.info("ContinuousBatcher worker started")
    
    def stop(self):
        """Stop the batching worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("ContinuousBatcher worker stopped")
    
    def add_request(self, request: Dict[str, Any], priority: int = 0):
        """Add request to batching queue"""
        # Add timestamp for latency tracking
        request["enqueue_time"] = time.time()
        self.request_queue.put((priority, request))
    
    def _batch_worker(self):
        """Worker thread that creates batches"""
        while self.running:
            try:
                # Collect requests until timeout or batch full
                batch = []
                start_time = time.time()
                
                while (len(batch) < self.max_batch_size and 
                       time.time() - start_time < self.timeout):
                    
                    try:
                        # Non-blocking get with short timeout
                        priority, request = self.request_queue.get(timeout=0.001)
                        
                        # Validate request
                        if self._validate_request(request):
                            batch.append(request)
                        
                        self.request_queue.task_done()
                    
                    except queue.Empty:
                        continue
                
                if batch:
                    # Sort by priority and sequence length for optimal batching
                    batch.sort(key=lambda x: (-x.get("priority", 0), 
                                            x.get("input_length", 0)))
                    
                    with self.batch_lock:
                        self.current_batch = batch
                    
                    # Track batch size
                    self.batch_sizes.append(len(batch))
                    
                    # Process the batch
                    self._process_batch(batch)
                    
                    # Clear current batch
                    with self.batch_lock:
                        self.current_batch = []
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
            
            except Exception as e:
                logger.error(f"Error in batch worker: {e}")
                time.sleep(0.1)
    
    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate request before adding to batch"""
        if "input_ids" not in request:
            logger.warning("Request missing input_ids")
            return False
        
        input_length = request.get("input_length", len(request["input_ids"]))
        if input_length > self.max_sequence_length:
            logger.warning(f"Request too long: {input_length} > {self.max_sequence_length}")
            return False
        
        return True
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests (to be overridden)"""
        # This is a placeholder - actual processing would be done by the inference engine
        for request in batch:
            # Calculate latency
            if "enqueue_time" in request:
                latency = time.time() - request["enqueue_time"]
                self.latencies.append(latency)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        with self.batch_lock:
            current_batch_size = len(self.current_batch)
        
        return {
            "queue_size": self.request_queue.qsize(),
            "current_batch_size": current_batch_size,
            "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
            "avg_latency_ms": np.mean(self.latencies) * 1000 if self.latencies else 0,
            "p99_latency_ms": np.percentile(list(self.latencies), 99) * 1000 if self.latencies else 0,
            "max_batch_size": self.max_batch_size
        }


class ModelShardingManager:
    """Manages automatic model sharding for multi-GPU inference"""
    
    def __init__(self, model: nn.Module, num_gpus: Optional[int] = None,
                 sharding_strategy: str = "auto"):
        
        self.model = model
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.sharding_strategy = sharding_strategy
        
        self.shards: Dict[int, nn.Module] = {}
        self.gpu_assignments: Dict[int, List[int]] = defaultdict(list)
        
        # Performance monitoring
        self.gpu_utilization: Dict[int, float] = {}
        self.memory_usage: Dict[int, float] = {}
        
        if self.num_gpus > 1:
            self._analyze_model()
            self._apply_sharding()
    
    def _analyze_model(self):
        """Analyze model structure for optimal sharding"""
        logger.info(f"Analyzing model for sharding across {self.num_gpus} GPUs")
        
        # Calculate layer sizes
        self.layer_sizes = {}
        total_params = 0
        
        for name, param in self.model.named_parameters():
            size_mb = param.numel() * param.element_size() / (1024 * 1024)
            self.layer_sizes[name] = size_mb
            total_params += size_mb
        
        logger.info(f"Total model size: {total_params:.1f} MB")
    
    def _apply_sharding(self):
        """Apply model sharding based on strategy"""
        if self.sharding_strategy == "auto":
            # Simple layer-based sharding
            layers = list(self.model.children())
            layers_per_gpu = max(1, len(layers) // self.num_gpus)
            
            for gpu_id in range(self.num_gpus):
                start_idx = gpu_id * layers_per_gpu
                end_idx = min((gpu_id + 1) * layers_per_gpu, len(layers))
                
                if start_idx < len(layers):
                    shard_layers = layers[start_idx:end_idx]
                    self.shards[gpu_id] = nn.Sequential(*shard_layers)
                    self.gpu_assignments[gpu_id] = list(range(start_idx, end_idx))
                    
                    # Move shard to GPU
                    if torch.cuda.is_available():
                        self.shards[gpu_id] = self.shards[gpu_id].to(f"cuda:{gpu_id}")
                    
                    logger.info(f"GPU {gpu_id}: layers {start_idx}-{end_idx-1}")
        
        elif self.sharding_strategy == "tensor_parallel":
            # Placeholder for tensor parallelism
            logger.warning("Tensor parallelism not yet implemented")
    
    def forward(self, *args, **kwargs):
        """Forward pass with sharded model"""
        if self.num_gpus == 1:
            return self.model(*args, **kwargs)
        
        # Simple sequential execution across shards
        # In production, this would use proper pipeline parallelism
        x = args[0] if args else kwargs.get("input_ids")
        
        for gpu_id in sorted(self.shards.keys()):
            # Move input to current GPU
            if hasattr(x, "to"):
                x = x.to(f"cuda:{gpu_id}")
            
            # Forward through shard
            with torch.cuda.device(gpu_id):
                x = self.shards[gpu_id](x)
        
        return x
    
    def update_utilization(self):
        """Update GPU utilization metrics"""
        for gpu_id in range(self.num_gpus):
            if torch.cuda.is_available():
                self.gpu_utilization[gpu_id] = torch.cuda.utilization(gpu_id) / 100.0
                self.memory_usage[gpu_id] = torch.cuda.memory_allocated(gpu_id) / torch.cuda.get_device_properties(gpu_id).total_memory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sharding statistics"""
        self.update_utilization()
        
        return {
            "num_gpus": self.num_gpus,
            "num_shards": len(self.shards),
            "gpu_utilization": self.gpu_utilization,
            "memory_usage": self.memory_usage,
            "sharding_strategy": self.sharding_strategy
        }


class DynamicInferenceOptimizer:
    """Main optimizer that coordinates batching, caching, and sharding"""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        
        self.model = model
        self.config = config or {}
        
        # Initialize components
        self.cache_manager = KVCacheManager(
            max_cache_size=self.config.get("cache_size_gb", 10) * 1024 * 1024 * 1024,
            policy=CachePolicy(self.config.get("cache_policy", "adaptive"))
        )
        
        self.batcher = ContinuousBatcher(
            max_batch_size=self.config.get("max_batch_size", 32),
            max_sequence_length=self.config.get("max_sequence_length", 4096),
            timeout_ms=self.config.get("batch_timeout_ms", 50.0)
        )
        
        self.sharding_manager = ModelShardingManager(
            model=model,
            num_gpus=self.config.get("num_gpus", None),
            sharding_strategy=self.config.get("sharding_strategy", "auto")
        )
        
        # Pattern analysis
        self.pattern = RequestPattern()
        self.pattern_history = deque(maxlen=100)
        
        # Optimization state
        self.current_strategy = BatchingStrategy.ADAPTIVE
        self.optimization_thread = None
        self.running = False
        
        # Performance tracking
        self.throughput_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        
        logger.info("DynamicInferenceOptimizer initialized")
    
    def start(self):
        """Start the optimizer"""
        self.running = True
        self.batcher.start()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("DynamicInferenceOptimizer started")
    
    def stop(self):
        """Stop the optimizer"""
        self.running = False
        self.batcher.stop()
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        logger.info("DynamicInferenceOptimizer stopped")
    
    def process_request(self, request: Dict[str, Any]) -> Future:
        """Process a single inference request"""
        # Update pattern analysis
        self.pattern.update(request)
        
        # Add to batcher
        future = Future()
        request["future"] = future
        
        priority = request.get("priority", 0)
        self.batcher.add_request(request, priority)
        
        return future
    
    def _optimization_loop(self):
        """Continuous optimization loop"""
        while self.running:
            try:
                # Analyze current patterns
                self._analyze_patterns()
                
                # Adjust strategies based on patterns
                self._adjust_strategies()
                
                # Collect performance metrics
                self._collect_metrics()
                
                # Sleep before next optimization cycle
                time.sleep(self.config.get("optimization_interval_s", 5.0))
            
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(1.0)
    
    def _analyze_patterns(self):
        """Analyze request patterns for optimization"""
        current_time = time.time()
        
        # Store pattern snapshot
        pattern_snapshot = {
            "timestamp": current_time,
            "arrival_rate": self.pattern.arrival_rate,
            "avg_sequence_length": np.mean(self.pattern.sequence_lengths) if self.pattern.sequence_lengths else 0,
            "batch_stats": self.batcher.get_stats()
        }
        
        self.pattern_history.append(pattern_snapshot)
    
    def _adjust_strategies(self):
        """Adjust optimization strategies based on patterns"""
        if not self.pattern_history:
            return
        
        latest = self.pattern_history[-1]
        arrival_rate = latest["arrival_rate"]
        avg_seq_len = latest["avg_sequence_length"]
        
        # Adjust batching strategy
        if arrival_rate < 1.0:
            # Low arrival rate: use static batching
            new_strategy = BatchingStrategy.STATIC
        elif arrival_rate < 10.0:
            # Medium arrival rate: use dynamic batching
            new_strategy = BatchingStrategy.DYNAMIC
        else:
            # High arrival rate: use continuous batching
            new_strategy = BatchingStrategy.CONTINUOUS
        
        if new_strategy != self.current_strategy:
            logger.info(f"Switching batching strategy: {self.current_strategy} -> {new_strategy}")
            self.current_strategy = new_strategy
        
        # Adjust cache policy based on hit rate
        cache_stats = self.cache_manager.get_stats()
        hit_rate = cache_stats["hit_rate"]
        
        if hit_rate < 0.3:
            # Low hit rate: switch to LRU
            self.cache_manager.policy = CachePolicy.LRU
        elif hit_rate < 0.6:
            # Medium hit rate: use adaptive
            self.cache_manager.policy = CachePolicy.ADAPTIVE
        else:
            # High hit rate: use LFU
            self.cache_manager.policy = CachePolicy.LFU
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        batch_stats = self.batcher.get_stats()
        cache_stats = self.cache_manager.get_stats()
        shard_stats = self.sharding_manager.get_stats()
        
        # Calculate throughput (requests per second)
        if self.pattern_history and len(self.pattern_history) > 1:
            time_diff = self.pattern_history[-1]["timestamp"] - self.pattern_history[0]["timestamp"]
            if time_diff > 0:
                throughput = len(self.pattern_history) / time_diff
                self.throughput_history.append(throughput)
        
        # Calculate average latency
        if batch_stats["avg_latency_ms"] > 0:
            self.latency_history.append(batch_stats["avg_latency_ms"])
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        return {
            "timestamp": time.time(),
            "current_strategy": self.current_strategy.value,
            "pattern": {
                "arrival_rate": self.pattern.arrival_rate,
                "avg_sequence_length": np.mean(self.pattern.sequence_lengths) if self.pattern.sequence_lengths else 0,
                "total_requests": sum(self.pattern.request_types.values())
            },
            "batching": self.batcher.get_stats(),
            "caching": self.cache_manager.get_stats(),
            "sharding": self.sharding_manager.get_stats(),
            "performance": {
                "avg_throughput": np.mean(self.throughput_history) if self.throughput_history else 0,
                "avg_latency_ms": np.mean(self.latency_history) if self.latency_history else 0,
                "p99_latency_ms": np.percentile(list(self.latency_history), 99) if self.latency_history else 0
            }
        }
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize for specific workload type"""
        logger.info(f"Optimizing for workload: {workload_type}")
        
        if workload_type == "high_throughput":
            self.batcher.max_batch_size = 64
            self.batcher.timeout = 0.1  # 100ms
            self.cache_manager.policy = CachePolicy.LRU
            self.cache_manager.max_cache_size = 5 * 1024 * 1024 * 1024  # 5GB
        
        elif workload_type == "low_latency":
            self.batcher.max_batch_size = 8
            self.batcher.timeout = 0.01  # 10ms
            self.cache_manager.policy = CachePolicy.PRIORITY
            self.cache_manager.max_cache_size = 20 * 1024 * 1024 * 1024  # 20GB
        
        elif workload_type == "mixed":
            self.batcher.max_batch_size = 32
            self.batcher.timeout = 0.05  # 50ms
            self.cache_manager.policy = CachePolicy.ADAPTIVE
            self.cache_manager.max_cache_size = 10 * 1024 * 1024 * 1024  # 10GB
        
        logger.info(f"Workload optimization complete: {workload_type}")


# Integration with existing forge codebase
class forgeInferenceOptimizer:
    """Wrapper for integration with forge"""
    
    def __init__(self, model_path: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize optimizer
        self.optimizer = DynamicInferenceOptimizer(self.model, kwargs)
        
        # Start optimizer
        self.optimizer.start()
        
        logger.info(f"forgeInferenceOptimizer initialized with model: {model_path}")
    
    def generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate text with optimization"""
        futures = []
        
        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Create request
            request = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "input_length": inputs["input_ids"].shape[1],
                "generation_kwargs": generation_kwargs,
                "type": "generation"
            }
            
            # Process request
            future = self.optimizer.process_request(request)
            futures.append(future)
        
        # Wait for results
        results = []
        for future in futures:
            try:
                output_ids = future.result(timeout=30.0)
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append(text)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                results.append("")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.optimizer.get_optimization_report()
    
    def shutdown(self):
        """Shutdown the optimizer"""
        self.optimizer.stop()


# Factory function for easy integration
def create_inference_optimizer(model_path: str, **kwargs) -> forgeInferenceOptimizer:
    """Factory function to create inference optimizer"""
    return forgeInferenceOptimizer(model_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # This would be integrated into the main inference pipeline
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Inference Optimizer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--cache_size_gb", type=float, default=10.0, help="Cache size in GB")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--optimization_interval", type=float, default=5.0, help="Optimization interval in seconds")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = create_inference_optimizer(
        model_path=args.model_path,
        cache_size_gb=args.cache_size_gb,
        max_batch_size=args.max_batch_size,
        optimization_interval_s=args.optimization_interval
    )
    
    try:
        # Example usage
        prompts = ["Hello, how are you?", "What is the meaning of life?"]
        results = optimizer.generate(prompts, max_length=100)
        
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")
            print("-" * 50)
        
        # Print stats
        stats = optimizer.get_stats()
        print("\nOptimization Statistics:")
        print(json.dumps(stats, indent=2))
    
    finally:
        optimizer.shutdown()