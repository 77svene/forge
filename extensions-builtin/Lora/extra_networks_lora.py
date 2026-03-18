from modules import extra_networks, shared
import networks
import torch
from collections import defaultdict
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import mmap
import os

class BatchedLoraCache:
    """Cache for precomputed merged LoRA weights with memory mapping"""
    
    def __init__(self, max_cache_size=50):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_cache_size
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _generate_cache_key(self, names, te_multipliers, unet_multipliers, dyn_dims):
        """Generate unique cache key for LoRA combination"""
        key_data = f"{names}_{te_multipliers}_{unet_multipliers}_{dyn_dims}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _compute_merged_weights(self, names, te_multipliers, unet_multipliers, dyn_dims):
        """Compute merged weights for LoRA combination"""
        # Load networks without applying to model
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims, apply_to_model=False)
        
        # Extract merged weights from loaded networks
        merged_weights = {}
        for network in networks.loaded_networks:
            for key, weights in network.weights.items():
                if key not in merged_weights:
                    merged_weights[key] = weights.clone()
                else:
                    merged_weights[key] += weights
        
        return merged_weights
    
    def get_or_compute(self, names, te_multipliers, unet_multipliers, dyn_dims):
        """Get cached weights or compute and cache them"""
        cache_key = self._generate_cache_key(names, te_multipliers, unet_multipliers, dyn_dims)
        
        with self.lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = torch.cuda.Event().record() if torch.cuda.is_available() else 0
                return self.cache[cache_key]
        
        # Compute weights in background thread
        future = self.executor.submit(
            self._compute_merged_weights, 
            names, te_multipliers, unet_multipliers, dyn_dims
        )
        merged_weights = future.result()
        
        with self.lock:
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[cache_key] = merged_weights
            self.access_times[cache_key] = torch.cuda.Event().record() if torch.cuda.is_available() else 0
            
        return merged_weights
    
    def prefetch_combinations(self, combinations):
        """Prefetch common combinations in background"""
        for names, te_multipliers, unet_multipliers, dyn_dims in combinations:
            self.executor.submit(
                self.get_or_compute,
                names, te_multipliers, unet_multipliers, dyn_dims
            )

class MemoryMappedLoraWeights:
    """Memory-mapped storage for LoRA weights"""
    
    def __init__(self, storage_dir="lora_cache"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.mmap_files = {}
        
    def save_weights(self, cache_key, weights):
        """Save weights to memory-mapped file"""
        filepath = os.path.join(self.storage_dir, f"{cache_key}.pt")
        torch.save(weights, filepath)
        
        # Create memory map
        with open(filepath, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.mmap_files[cache_key] = mm
            
    def load_weights(self, cache_key):
        """Load weights from memory-mapped file"""
        if cache_key in self.mmap_files:
            # Fast load from memory map
            mm = self.mmap_files[cache_key]
            mm.seek(0)
            return torch.load(mm)
        return None

class DynamicBatchProcessor:
    """Process batches with progressive LoRA loading"""
    
    def __init__(self, chunk_size=4):
        self.chunk_size = chunk_size
        self.lora_cache = BatchedLoraCache()
        self.mmap_storage = MemoryMappedLoraWeights()
        
    def group_prompts_by_lora(self, params_list):
        """Group prompts with same LoRA combination"""
        groups = defaultdict(list)
        
        for idx, params in enumerate(params_list):
            if not params.items:
                continue
                
            names = []
            te_multipliers = []
            unet_multipliers = []
            dyn_dims = []
            
            names.append(params.positional[0])
            
            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))
            
            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))
            
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            
            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
            
            # Create group key
            group_key = (
                tuple(names),
                tuple(te_multipliers),
                tuple(unet_multipliers),
                tuple(dyn_dims)
            )
            groups[group_key].append(idx)
            
        return groups
    
    def process_batch_progressive(self, p, params_list):
        """Process batch with progressive LoRA loading"""
        groups = self.group_prompts_by_lora(params_list)
        
        # Process in chunks for memory efficiency
        for group_key, indices in groups.items():
            names, te_multipliers, unet_multipliers, dyn_dims = group_key
            
            # Get or compute merged weights
            merged_weights = self.lora_cache.get_or_compute(
                list(names), list(te_multipliers), list(unet_multipliers), list(dyn_dims)
            )
            
            # Apply weights to model for this chunk
            self._apply_merged_weights(merged_weights)
            
            # Process prompts in this group
            for idx in indices:
                # Update prompt-specific parameters if needed
                pass
                
    def _apply_merged_weights(self, merged_weights):
        """Apply merged weights to model efficiently"""
        # This would integrate with the model's forward pass
        # Implementation depends on specific model architecture
        pass

class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')
        
        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""
        
        self.batch_processor = DynamicBatchProcessor()
        self.common_combinations = []
        self.prefetch_enabled = True
        
    remove_symbols = str.maketrans('', '', ":,")
    
    def _extract_lora_params(self, params_list):
        """Extract LoRA parameters from params list"""
        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        
        for params in params_list:
            assert params.items
            
            names.append(params.positional[0])
            
            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))
            
            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))
            
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            
            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
            
        return names, te_multipliers, unet_multipliers, dyn_dims
    
    def _should_use_batch_optimization(self, p, params_list):
        """Determine if batch optimization should be used"""
        # Use batch optimization for batch size > 1
        if hasattr(p, 'batch_size') and p.batch_size > 1:
            return True
            
        # Use if multiple different LoRA combinations in batch
        if len(params_list) > 1:
            groups = self.batch_processor.group_prompts_by_lora(params_list)
            if len(groups) > 1:
                return True
                
        return False
    
    def _prefetch_common_combinations(self, params_list):
        """Prefetch commonly used LoRA combinations"""
        if not self.prefetch_enabled:
            return
            
        # Extract current combination
        names, te_multipliers, unet_multipliers, dyn_dims = self._extract_lora_params(params_list)
        current_combo = (tuple(names), tuple(te_multipliers), tuple(unet_multipliers), tuple(dyn_dims))
        
        # Add to common combinations if not already present
        if current_combo not in self.common_combinations:
            self.common_combinations.append(current_combo)
            
        # Keep only last 10 combinations
        if len(self.common_combinations) > 10:
            self.common_combinations = self.common_combinations[-10:]
            
        # Prefetch in background
        self.batch_processor.lora_cache.prefetch_combinations(self.common_combinations)
    
    def activate(self, p, params_list):
        additional = shared.opts.sd_lora
        
        self.errors.clear()
        
        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))
        
        # Check if we should use batch optimization
        use_batch_optimization = self._should_use_batch_optimization(p, params_list)
        
        if use_batch_optimization:
            # Use dynamic batching with streaming activation
            self._activate_with_batch_optimization(p, params_list)
        else:
            # Use standard sequential activation
            self._activate_sequential(p, params_list)
        
        # Prefetch common combinations for next batch
        self._prefetch_common_combinations(params_list)
        
        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}
            
            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    p.lora_hashes[item.mentioned_name.translate(self.remove_symbols)] = item.network_on_disk.shorthash
            
            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ', '.join(f'{k}: {v}' for k, v in p.lora_hashes.items())
    
    def _activate_sequential(self, p, params_list):
        """Standard sequential LoRA activation"""
        names, te_multipliers, unet_multipliers, dyn_dims = self._extract_lora_params(params_list)
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
    
    def _activate_with_batch_optimization(self, p, params_list):
        """Optimized batch activation with dynamic batching"""
        # Group prompts by LoRA combination
        groups = self.batch_processor.group_prompts_by_lora(params_list)
        
        if len(groups) == 1:
            # All prompts use same LoRA combination
            names, te_multipliers, unet_multipliers, dyn_dims = self._extract_lora_params(params_list)
            
            # Try to get from cache first
            cache_key = self.batch_processor.lora_cache._generate_cache_key(
                names, te_multipliers, unet_multipliers, dyn_dims
            )
            
            # Check if we have memory-mapped version
            mmap_weights = self.batch_processor.mmap_storage.load_weights(cache_key)
            if mmap_weights:
                # Apply memory-mapped weights directly
                self._apply_weights_directly(mmap_weights)
            else:
                # Compute and cache
                networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
                
                # Cache for future use
                if hasattr(networks, 'loaded_networks') and networks.loaded_networks:
                    merged_weights = self.batch_processor.lora_cache._compute_merged_weights(
                        names, te_multipliers, unet_multipliers, dyn_dims
                    )
                    self.batch_processor.lora_cache.get_or_compute(
                        names, te_multipliers, unet_multipliers, dyn_dims
                    )
        else:
            # Multiple LoRA combinations in batch - use progressive loading
            self.batch_processor.process_batch_progressive(p, params_list)
    
    def _apply_weights_directly(self, weights):
        """Apply precomputed weights directly to model"""
        # This would need integration with the model's layers
        # Implementation depends on specific model architecture
        for name, weight in weights.items():
            # Find corresponding layer and apply weight
            # This is a placeholder - actual implementation would depend on model structure
            pass
    
    def deactivate(self, p):
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            
            self.errors.clear()
        
        # Clear batch processor cache if needed
        if hasattr(self, 'batch_processor'):
            # Optional: clear cache to free memory
            # self.batch_processor.lora_cache.cache.clear()
            pass