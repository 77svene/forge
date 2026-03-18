"""
Real-time Inference Optimization System for forge.

This module implements dynamic inference optimization with automatic strategy selection,
continuous batching, KV-cache optimization, and multi-GPU model sharding.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import nvtx

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Supported batching strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """KV-cache optimization strategies."""
    STANDARD = "standard"
    PAGED = "paged"
    COMPRESSED = "compressed"
    ADAPTIVE = "adaptive"


class ShardingStrategy(Enum):
    """Model sharding strategies for multi-GPU inference."""
    NONE = "none"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    AUTO = "auto"


@dataclass
class RequestPattern:
    """Analysis of request patterns for optimization decisions."""
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    request_rate: float = 0.0  # requests per second
    burstiness: float = 0.0  # variance in request timing
    sequence_length_variance: float = 0.0
    batch_size_preference: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceConfig:
    """Configuration for inference optimization."""
    max_batch_size: int = 32
    max_sequence_length: int = 4096
    kv_cache_size: int = 1000
    enable_cuda_graphs: bool = True
    use_flash_attention: bool = True
    quantization_bits: Optional[int] = None
    dynamic_batch_timeout_ms: float = 10.0
    pattern_analysis_window_sec: float = 60.0
    min_batch_size_for_optimization: int = 4
    gpu_memory_fraction: float = 0.9


@dataclass
class OptimizationMetrics:
    """Metrics for monitoring optimization performance."""
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    strategy_changes: int = 0
    total_requests_processed: int = 0


class RequestAnalyzer:
    """Analyzes request patterns to inform optimization decisions."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.request_history = deque(maxlen=window_size)
        self.length_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        self._lock = threading.RLock()
        
    def add_request(self, sequence_length: int, timestamp: Optional[float] = None):
        """Record a new request for pattern analysis."""
        with self._lock:
            timestamp = timestamp or time.time()
            self.request_history.append(1)
            self.length_history.append(sequence_length)
            self.timestamp_history.append(timestamp)
    
    def analyze_patterns(self) -> RequestPattern:
        """Analyze current request patterns."""
        with self._lock:
            if not self.length_history:
                return RequestPattern()
            
            lengths = list(self.length_history)
            timestamps = list(self.timestamp_history)
            
            # Calculate basic statistics
            avg_length = np.mean(lengths)
            max_length = max(lengths)
            length_variance = np.var(lengths)
            
            # Calculate request rate and burstiness
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                avg_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 1.0
                request_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
                burstiness = np.std(time_diffs) / avg_interval if avg_interval > 0 else 0.0
            else:
                request_rate = 0.0
                burstiness = 0.0
            
            # Determine batch size preference based on patterns
            if request_rate > 10.0 and length_variance < 100:
                batch_pref = min(32, int(request_rate / 2))
            elif burstiness > 2.0:
                batch_pref = 8
            else:
                batch_pref = 4
            
            return RequestPattern(
                avg_sequence_length=avg_length,
                max_sequence_length=max_length,
                request_rate=request_rate,
                burstiness=burstiness,
                sequence_length_variance=length_variance,
                batch_size_preference=batch_pref,
                timestamp=time.time()
            )


class KVCacheManager:
    """Manages KV-cache with optimization strategies."""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.strategy = strategy
        self.cache = {}
        self.access_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        with self._lock:
            if key in self.cache:
                self.cache_hits += 1
                self.access_times[key] = time.time()
                return self.cache[key]
            self.cache_misses += 1
            return None
    
    def put(self, key: str, value: Any, size_hint: int = 1):
        """Store item in cache with eviction if needed."""
        with self._lock:
            # Simple LRU eviction if cache is full
            if len(self.cache) >= 1000:  # Configurable max size
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        with self._lock:
            total = self.cache_hits + self.cache_misses
            return self.cache_hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()


class ContinuousBatcher:
    """Implements continuous batching for dynamic request handling."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.current_batch = []
        self.batch_lock = threading.RLock()
        self.batch_ready = threading.Event()
        self.processing = False
        
    def add_request(self, request: Dict[str, Any]) -> bool:
        """Add a request to the current batch."""
        with self.batch_lock:
            if len(self.current_batch) >= self.max_batch_size:
                return False
            
            self.current_batch.append(request)
            if len(self.current_batch) >= self.max_batch_size:
                self.batch_ready.set()
            return True
    
    def get_batch(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get a batch of requests, waiting if necessary."""
        timeout = timeout or (self.timeout_ms / 1000.0)
        
        # Wait for batch to fill or timeout
        start_time = time.time()
        while True:
            with self.batch_lock:
                if self.current_batch:
                    batch = self.current_batch.copy()
                    self.current_batch.clear()
                    self.batch_ready.clear()
                    return batch
            
            if time.time() - start_time > timeout:
                with self.batch_lock:
                    if self.current_batch:
                        batch = self.current_batch.copy()
                        self.current_batch.clear()
                        return batch
                return []
            
            time.sleep(0.001)  # Small sleep to avoid busy waiting
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending requests."""
        with self.batch_lock:
            return len(self.current_batch) > 0


class ModelShardingManager:
    """Manages model sharding across multiple GPUs."""
    
    def __init__(self, model: nn.Module, strategy: ShardingStrategy = ShardingStrategy.AUTO):
        self.model = model
        self.strategy = strategy
        self.device_map = {}
        self.gpu_count = torch.cuda.device_count()
        
    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze model structure for sharding decisions."""
        total_params = sum(p.numel() for p in self.model.parameters())
        layer_counts = defaultdict(int)
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                layer_type = type(module).__name__
                layer_counts[layer_type] += 1
        
        return {
            'total_parameters': total_params,
            'layer_counts': dict(layer_counts),
            'gpu_count': self.gpu_count,
            'model_size_gb': total_params * 4 / (1024**3)  # Assuming float32
        }
    
    def determine_sharding_strategy(self) -> ShardingStrategy:
        """Automatically determine optimal sharding strategy."""
        if self.gpu_count == 1:
            return ShardingStrategy.NONE
        
        analysis = self.analyze_model_structure()
        model_size_gb = analysis['model_size_gb']
        
        # Get available GPU memory
        gpu_memory = []
        for i in range(self.gpu_count):
            gpu_memory.append(torch.cuda.get_device_properties(i).total_memory / (1024**3))
        
        avg_gpu_memory = np.mean(gpu_memory)
        
        # Decision logic
        if model_size_gb > avg_gpu_memory * 0.8:
            # Model is too large for single GPU, use tensor parallelism
            if self.gpu_count >= 2:
                return ShardingStrategy.TENSOR_PARALLEL
            else:
                return ShardingStrategy.PIPELINE_PARALLEL
        elif model_size_gb > avg_gpu_memory * 0.5:
            # Model fits but with limited headroom, use pipeline parallelism
            return ShardingStrategy.PIPELINE_PARALLEL
        else:
            # Model fits comfortably, no sharding needed
            return ShardingStrategy.NONE
    
    def apply_sharding(self) -> Dict[int, List[str]]:
        """Apply the determined sharding strategy."""
        if self.strategy == ShardingStrategy.AUTO:
            self.strategy = self.determine_sharding_strategy()
        
        if self.strategy == ShardingStrategy.NONE:
            # Place entire model on first GPU
            device = torch.device('cuda:0')
            self.model.to(device)
            self.device_map = {0: ['all']}
            return self.device_map
        
        elif self.strategy == ShardingStrategy.TENSOR_PARALLEL:
            return self._apply_tensor_parallelism()
        
        elif self.strategy == ShardingStrategy.PIPELINE_PARALLEL:
            return self._apply_pipeline_parallelism()
        
        return {}
    
    def _apply_tensor_parallelism(self) -> Dict[int, List[str]]:
        """Apply tensor parallelism across GPUs."""
        # Simplified implementation - in production would use actual TP libraries
        layers_per_gpu = {}
        all_layers = list(self.model.named_modules())
        layers_per_shard = len(all_layers) // self.gpu_count
        
        for gpu_id in range(self.gpu_count):
            start_idx = gpu_id * layers_per_shard
            end_idx = start_idx + layers_per_shard if gpu_id < self.gpu_count - 1 else len(all_layers)
            
            layer_names = [name for name, _ in all_layers[start_idx:end_idx]]
            layers_per_gpu[gpu_id] = layer_names
            
            # Move layers to GPU
            for name, module in all_layers[start_idx:end_idx]:
                module.to(f'cuda:{gpu_id}')
        
        self.device_map = layers_per_gpu
        return layers_per_gpu
    
    def _apply_pipeline_parallelism(self) -> Dict[int, List[str]]:
        """Apply pipeline parallelism across GPUs."""
        # Simplified implementation - in production would use actual PP libraries
        layers_per_gpu = {}
        all_layers = list(self.model.named_modules())
        
        # Group layers by type for better pipeline efficiency
        layer_groups = defaultdict(list)
        for name, module in all_layers:
            layer_type = type(module).__name__
            layer_groups[layer_type].append((name, module))
        
        # Distribute layer groups across GPUs
        gpu_id = 0
        for layer_type, layers in layer_groups.items():
            layers_per_gpu.setdefault(gpu_id, []).extend([name for name, _ in layers])
            
            # Move layers to current GPU
            for name, module in layers:
                module.to(f'cuda:{gpu_id}')
            
            # Move to next GPU if current one is getting full
            gpu_id = (gpu_id + 1) % self.gpu_count
        
        self.device_map = layers_per_gpu
        return layers_per_gpu


class InferenceOptimizer:
    """
    Main inference optimization system that coordinates all optimization strategies.
    
    Dynamically selects optimal batching, caching, and hardware acceleration
    strategies based on real-time request patterns.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[InferenceConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        
        # Initialize components
        self.request_analyzer = RequestAnalyzer()
        self.cache_manager = KVCacheManager()
        self.batcher = ContinuousBatcher(
            max_batch_size=self.config.max_batch_size,
            timeout_ms=self.config.dynamic_batch_timeout_ms
        )
        self.sharding_manager = ModelShardingManager(model)
        
        # State tracking
        self.current_strategy = BatchingStrategy.ADAPTIVE
        self.metrics = OptimizationMetrics()
        self.optimization_thread = None
        self.running = False
        
        # Pattern history for strategy adaptation
        self.pattern_history = deque(maxlen=100)
        self.strategy_performance = defaultdict(list)
        
        logger.info(f"Initialized InferenceOptimizer with {self.config}")
    
    def start(self):
        """Start the optimization system."""
        if self.running:
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        # Apply initial sharding if needed
        if torch.cuda.device_count() > 1:
            self.sharding_manager.apply_sharding()
        
        logger.info("InferenceOptimizer started")
    
    def stop(self):
        """Stop the optimization system."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        logger.info("InferenceOptimizer stopped")
    
    def add_request(self, request: Dict[str, Any]) -> bool:
        """Add a new inference request to the system."""
        # Extract sequence length for pattern analysis
        prompt = request.get('prompt', '')
        inputs = self.tokenizer(prompt, return_tensors='pt')
        sequence_length = inputs['input_ids'].shape[1]
        
        # Record for pattern analysis
        self.request_analyzer.add_request(sequence_length)
        
        # Add to batcher
        request['sequence_length'] = sequence_length
        request['timestamp'] = time.time()
        
        return self.batcher.add_request(request)
    
    def process_batch(self) -> List[Dict[str, Any]]:
        """Process a batch of requests with current optimizations."""
        batch = self.batcher.get_batch()
        if not batch:
            return []
        
        start_time = time.time()
        
        # Apply current batching strategy
        if self.current_strategy == BatchingStrategy.CONTINUOUS:
            results = self._process_continuous_batch(batch)
        elif self.current_strategy == BatchingStrategy.DYNAMIC:
            results = self._process_dynamic_batch(batch)
        elif self.current_strategy == BatchingStrategy.ADAPTIVE:
            results = self._process_adaptive_batch(batch)
        else:
            results = self._process_static_batch(batch)
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000  # ms
        self._update_metrics(batch, processing_time)
        
        return results
    
    def _process_static_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch with static batching strategy."""
        with nvtx.range("static_batch_processing"):
            # Group by similar sequence lengths
            sorted_batch = sorted(batch, key=lambda x: x['sequence_length'])
            
            results = []
            for i in range(0, len(sorted_batch), self.config.max_batch_size):
                mini_batch = sorted_batch[i:i + self.config.max_batch_size]
                batch_results = self._run_inference(mini_batch)
                results.extend(batch_results)
            
            return results
    
    def _process_dynamic_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch with dynamic batching strategy."""
        with nvtx.range("dynamic_batch_processing"):
            # Dynamically adjust batch size based on sequence lengths
            total_length = sum(req['sequence_length'] for req in batch)
            avg_length = total_length / len(batch) if batch else 0
            
            # Adjust batch size based on average length
            if avg_length > 1024:
                effective_batch_size = max(1, self.config.max_batch_size // 2)
            elif avg_length > 512:
                effective_batch_size = max(2, self.config.max_batch_size // 1.5)
            else:
                effective_batch_size = self.config.max_batch_size
            
            results = []
            for i in range(0, len(batch), int(effective_batch_size)):
                mini_batch = batch[i:i + int(effective_batch_size)]
                batch_results = self._run_inference(mini_batch)
                results.extend(batch_results)
            
            return results
    
    def _process_continuous_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch with continuous batching strategy."""
        with nvtx.range("continuous_batch_processing"):
            # Process requests as they arrive, maintaining a rolling batch
            results = []
            
            # Sort by arrival time for fairness
            batch.sort(key=lambda x: x.get('timestamp', 0))
            
            # Process in chunks while allowing new requests to be added
            chunk_size = min(8, len(batch))  # Smaller chunks for responsiveness
            
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i + chunk_size]
                chunk_results = self._run_inference(chunk)
                results.extend(chunk_results)
                
                # Check for new requests to add to processing
                if self.batcher.has_pending_requests():
                    # In a real implementation, we'd merge new requests
                    pass
            
            return results
    
    def _process_adaptive_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch with adaptive strategy selection."""
        # Analyze current batch characteristics
        lengths = [req['sequence_length'] for req in batch]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        
        # Select strategy based on batch characteristics
        if length_variance > 1000:
            # High variance - use dynamic batching
            return self._process_dynamic_batch(batch)
        elif len(batch) > 16:
            # Large batch - use continuous for better throughput
            return self._process_continuous_batch(batch)
        else:
            # Default to static for consistency
            return self._process_static_batch(batch)
    
    def _run_inference(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on a batch of requests."""
        prompts = [req['prompt'] for req in batch]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length
        )
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Check cache for similar requests
        cache_key = self._generate_cache_key(inputs)
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Run inference with optimizations
        with torch.no_grad():
            if self.config.enable_cuda_graphs and hasattr(self, '_cuda_graph'):
                # Use CUDA graphs if available
                outputs = self._run_with_cuda_graph(inputs)
            else:
                # Standard inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True
                )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            results.append({
                'generated_text': generated_text,
                'request_id': batch[i].get('request_id'),
                'latency_ms': 0  # Will be filled by caller
            })
        
        # Cache results
        self.cache_manager.put(cache_key, results)
        
        return results
    
    def _generate_cache_key(self, inputs: Dict[str, torch.Tensor]) -> str:
        """Generate cache key from inputs."""
        # Simple hash of input tensor shapes and first few elements
        key_parts = []
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                key_parts.append(f"{k}:{v.shape}:{v.flatten()[:5].tolist()}")
        return "|".join(key_parts)
    
    def _run_with_cuda_graph(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference using CUDA graphs for reduced overhead."""
        # Placeholder for CUDA graph implementation
        # In production, this would capture and replay CUDA graphs
        return self.model.generate(**inputs, max_new_tokens=256)
    
    def _optimization_loop(self):
        """Main optimization loop that adjusts strategies based on patterns."""
        while self.running:
            try:
                # Analyze current patterns
                pattern = self.request_analyzer.analyze_patterns()
                self.pattern_history.append(pattern)
                
                # Adjust strategies if needed
                self._adjust_batching_strategy(pattern)
                self._adjust_cache_strategy(pattern)
                
                # Update metrics
                self.metrics.gpu_utilization = self._get_gpu_utilization()
                self.metrics.memory_usage_gb = self._get_memory_usage()
                
                # Sleep before next analysis
                time.sleep(self.config.pattern_analysis_window_sec)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(5.0)
    
    def _adjust_batching_strategy(self, pattern: RequestPattern):
        """Adjust batching strategy based on request patterns."""
        old_strategy = self.current_strategy
        
        # Decision logic for strategy selection
        if pattern.request_rate > 20.0 and pattern.burstiness < 1.0:
            # High steady rate - use continuous batching
            self.current_strategy = BatchingStrategy.CONTINUOUS
        elif pattern.sequence_length_variance > 500:
            # High variance in lengths - use dynamic batching
            self.current_strategy = BatchingStrategy.DYNAMIC
        elif pattern.request_rate < 5.0:
            # Low rate - use static batching
            self.current_strategy = BatchingStrategy.STATIC
        else:
            # Default to adaptive
            self.current_strategy = BatchingStrategy.ADAPTIVE
        
        # Update batcher configuration
        self.batcher.max_batch_size = pattern.batch_size_preference
        
        if old_strategy != self.current_strategy:
            self.metrics.strategy_changes += 1
            logger.info(f"Changed batching strategy from {old_strategy} to {self.current_strategy}")
    
    def _adjust_cache_strategy(self, pattern: RequestPattern):
        """Adjust cache strategy based on request patterns."""
        hit_rate = self.cache_manager.get_hit_rate()
        
        if hit_rate < 0.3 and pattern.request_rate > 10.0:
            # Low hit rate with high traffic - consider compressed cache
            self.cache_manager.strategy = CacheStrategy.COMPRESSED
        elif pattern.avg_sequence_length > 1024:
            # Long sequences - use paged cache
            self.cache_manager.strategy = CacheStrategy.PAGED
        else:
            # Default to adaptive
            self.cache_manager.strategy = CacheStrategy.ADAPTIVE
    
    def _update_metrics(self, batch: List[Dict[str, Any]], processing_time_ms: float):
        """Update performance metrics."""
        self.metrics.total_requests_processed += len(batch)
        
        # Update rolling averages
        alpha = 0.1  # Smoothing factor
        self.metrics.avg_batch_size = (
            alpha * len(batch) + 
            (1 - alpha) * self.metrics.avg_batch_size
        )
        self.metrics.avg_latency_ms = (
            alpha * processing_time_ms + 
            (1 - alpha) * self.metrics.avg_latency_ms
        )
        
        # Calculate throughput
        if processing_time_ms > 0:
            tokens_processed = sum(
                req.get('sequence_length', 0) for req in batch
            )
            self.metrics.throughput_tokens_per_sec = (
                tokens_processed / (processing_time_ms / 1000.0)
            )
        
        self.metrics.cache_hit_rate = self.cache_manager.get_hit_rate()
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        if torch.cuda.is_available():
            try:
                # This is a simplified version - in production use nvidia-smi or similar
                return torch.cuda.utilization() / 100.0
            except:
                return 0.0
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024**3)
            except:
                return 0.0
        return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics."""
        return {
            'batching_strategy': self.current_strategy.value,
            'cache_strategy': self.cache_manager.strategy.value,
            'avg_batch_size': self.metrics.avg_batch_size,
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'throughput_tokens_per_sec': self.metrics.throughput_tokens_per_sec,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'gpu_utilization': self.metrics.gpu_utilization,
            'memory_usage_gb': self.metrics.memory_usage_gb,
            'strategy_changes': self.metrics.strategy_changes,
            'total_requests_processed': self.metrics.total_requests_processed,
            'current_pattern': self.request_analyzer.analyze_patterns().__dict__
        }
    
    def optimize_for_workload(self, workload_type: str):
        """Pre-configure optimizer for specific workload types."""
        workload_configs = {
            'chat': {
                'max_batch_size': 8,
                'dynamic_batch_timeout_ms': 5.0,
                'strategy': BatchingStrategy.CONTINUOUS
            },
            'batch_processing': {
                'max_batch_size': 32,
                'dynamic_batch_timeout_ms': 50.0,
                'strategy': BatchingStrategy.STATIC
            },
            'streaming': {
                'max_batch_size': 4,
                'dynamic_batch_timeout_ms': 2.0,
                'strategy': BatchingStrategy.CONTINUOUS
            },
            'high_throughput': {
                'max_batch_size': 64,
                'dynamic_batch_timeout_ms': 20.0,
                'strategy': BatchingStrategy.DYNAMIC
            }
        }
        
        if workload_type in workload_configs:
            config = workload_configs[workload_type]
            self.config.max_batch_size = config['max_batch_size']
            self.config.dynamic_batch_timeout_ms = config['dynamic_batch_timeout_ms']
            self.current_strategy = config['strategy']
            
            logger.info(f"Optimized for {workload_type} workload")
        else:
            logger.warning(f"Unknown workload type: {workload_type}")


# Integration with existing forge modules
def create_inference_optimizer(
    model_path: str,
    device_map: Optional[Dict] = None,
    config: Optional[InferenceConfig] = None
) -> InferenceOptimizer:
    """
    Factory function to create InferenceOptimizer with forge integration.
    
    Args:
        model_path: Path to the model or model identifier
        device_map: Device mapping for model placement
        config: Optional inference configuration
        
    Returns:
        Configured InferenceOptimizer instance
    """
    # Import here to avoid circular imports
    from forge.model import load_model
    from forge.chat import ChatModel
    
    # Load model using forge's loader
    model, tokenizer = load_model(
        model_name_or_path=model_path,
        device_map=device_map or "auto"
    )
    
    # Create optimizer
    optimizer = InferenceOptimizer(model, tokenizer, config)
    
    return optimizer


# Async wrapper for use in async contexts
class AsyncInferenceOptimizer:
    """Async wrapper for InferenceOptimizer."""
    
    def __init__(self, optimizer: InferenceOptimizer):
        self.optimizer = optimizer
        self.loop = asyncio.get_event_loop()
    
    async def add_request_async(self, request: Dict[str, Any]) -> bool:
        """Add request asynchronously."""
        return await self.loop.run_in_executor(
            None, self.optimizer.add_request, request
        )
    
    async def process_batch_async(self) -> List[Dict[str, Any]]:
        """Process batch asynchronously."""
        return await self.loop.run_in_executor(
            None, self.optimizer.process_batch
        )
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get metrics asynchronously."""
        return await self.loop.run_in_executor(
            None, self.optimizer.get_metrics
        )


# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with the actual forge inference server
    logging.basicConfig(level=logging.INFO)
    
    print("InferenceOptimizer module loaded successfully")
    print("Available strategies:")
    print(f"  Batching: {[s.value for s in BatchingStrategy]}")
    print(f"  Caching: {[s.value for s in CacheStrategy]}")
    print(f"  Sharding: {[s.value for s in ShardingStrategy]}")