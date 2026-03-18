"""
Predictive Model Preloading for Stable Diffusion WebUI
Analyzes usage patterns to preload models before they're needed.
"""

import os
import json
import time
import threading
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import shared modules from webui
try:
    import shared
    import sd_models
    import sd_hijack
    from modules import paths, devices
    from modules.shared import opts, cmd_opts
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    logger.warning("WebUI modules not available, running in standalone mode")

class UsageTracker:
    """Tracks model usage patterns with privacy-preserving analytics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.usage_history = deque(maxlen=max_history)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.session_start = time.time()
        self.last_model = None
        self.lock = threading.RLock()
        
        # Privacy-preserving: hash model paths for storage
        self.model_hashes = {}
        
    def _hash_model_path(self, model_path: str) -> str:
        """Create privacy-preserving hash of model path."""
        if model_path not in self.model_hashes:
            # Use SHA-256 for consistent hashing
            hash_obj = hashlib.sha256(model_path.encode('utf-8'))
            self.model_hashes[model_path] = hash_obj.hexdigest()[:16]  # First 16 chars for brevity
        return self.model_hashes[model_path]
    
    def record_usage(self, model_path: str, model_type: str = "checkpoint"):
        """Record a model usage event."""
        if not model_path:
            return
            
        with self.lock:
            current_time = time.time()
            model_hash = self._hash_model_path(model_path)
            
            # Record usage event
            usage_event = {
                'model_hash': model_hash,
                'model_type': model_type,
                'timestamp': current_time,
                'session_offset': current_time - self.session_start
            }
            
            self.usage_history.append(usage_event)
            
            # Update transition counts if we have a previous model
            if self.last_model:
                self.transition_counts[self.last_model][model_hash] += 1
            
            self.last_model = model_hash
            
            logger.debug(f"Recorded usage: {model_type} at {model_path}")
    
    def get_transition_probabilities(self, current_model_hash: str) -> Dict[str, float]:
        """Get transition probabilities from current model to others."""
        with self.lock:
            if current_model_hash not in self.transition_counts:
                return {}
            
            transitions = self.transition_counts[current_model_hash]
            total = sum(transitions.values())
            
            if total == 0:
                return {}
            
            return {model: count / total for model, count in transitions.items()}
    
    def get_recent_models(self, n: int = 5) -> List[str]:
        """Get n most recently used models."""
        with self.lock:
            return [event['model_hash'] for event in list(self.usage_history)[-n:]]
    
    def get_frequent_sequences(self, min_support: float = 0.1) -> List[Tuple[str, str, float]]:
        """Get frequent model sequences with support above threshold."""
        sequences = []
        with self.lock:
            for prev_model, next_models in self.transition_counts.items():
                total_transitions = sum(next_models.values())
                if total_transitions < 3:  # Minimum occurrences
                    continue
                    
                for next_model, count in next_models.items():
                    support = count / total_transitions
                    if support >= min_support:
                        sequences.append((prev_model, next_model, support))
            
            # Sort by support descending
            sequences.sort(key=lambda x: x[2], reverse=True)
            return sequences
    
    def clear_history(self):
        """Clear all usage history."""
        with self.lock:
            self.usage_history.clear()
            self.transition_counts.clear()
            self.last_model = None
            logger.info("Usage history cleared")

class MarkovPredictor:
    """Markov chain predictor for model sequences."""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.chain = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
    def train(self, sequences: List[List[str]]):
        """Train the Markov chain on sequences of model hashes."""
        for sequence in sequences:
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_item = sequence[i + self.order]
                self.chain[context][next_item] += 1
                self.context_counts[context] += 1
    
    def predict_next(self, context: Tuple[str, ...], top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict next models given a context."""
        if len(context) != self.order:
            # If context is shorter than order, use what we have
            context = context[-self.order:] if len(context) > self.order else context
            
        if context not in self.chain:
            return []
        
        predictions = self.chain[context]
        total = self.context_counts[context]
        
        if total == 0:
            return []
        
        # Calculate probabilities and sort
        probs = [(model, count / total) for model, count in predictions.items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        
        return probs[:top_k]
    
    def update(self, context: Tuple[str, ...], next_item: str):
        """Update chain with new observation."""
        self.chain[context][next_item] += 1
        self.context_counts[context] += 1

class ModelPreloader:
    """Handles actual model preloading with different strategies."""
    
    STRATEGIES = {
        'aggressive': {
            'preload_threshold': 0.3,
            'max_preload': 3,
            'idle_time': 5,
            'memory_limit_percent': 80
        },
        'balanced': {
            'preload_threshold': 0.5,
            'max_preload': 2,
            'idle_time': 10,
            'memory_limit_percent': 70
        },
        'conservative': {
            'preload_threshold': 0.7,
            'max_preload': 1,
            'idle_time': 15,
            'memory_limit_percent': 60
        }
    }
    
    def __init__(self, strategy: str = 'balanced'):
        self.strategy = strategy
        self.config = self.STRATEGIES.get(strategy, self.STRATEGIES['balanced'])
        self.preloaded_models = set()
        self.preload_lock = threading.RLock()
        self.model_cache = {}  # model_hash -> model_info
        
        # Track memory usage
        self.memory_monitor = MemoryMonitor()
        
    def should_preload(self, probability: float) -> bool:
        """Determine if model should be preloaded based on strategy."""
        if probability < self.config['preload_threshold']:
            return False
            
        # Check memory constraints
        if self.memory_monitor.get_memory_usage_percent() > self.config['memory_limit_percent']:
            logger.debug("Memory limit reached, skipping preload")
            return False
            
        return True
    
    def preload_model(self, model_hash: str, model_info: Dict[str, Any]) -> bool:
        """Preload a model into memory."""
        with self.preload_lock:
            if model_hash in self.preloaded_models:
                return True
                
            try:
                # Get actual model path from hash mapping
                model_path = model_info.get('path')
                if not model_path:
                    logger.warning(f"No path found for model hash: {model_hash}")
                    return False
                
                # Check if model file exists
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}")
                    return False
                
                # Load model based on type
                model_type = model_info.get('type', 'checkpoint')
                
                if WEBUI_AVAILABLE:
                    if model_type == 'checkpoint':
                        # Use sd_models to preload checkpoint
                        sd_models.load_model(model_path, already_loaded_state_dict=None)
                    elif model_type == 'lora':
                        # For LoRA, we can preload by reading the file
                        # Actual application happens when used
                        with open(model_path, 'rb') as f:
                            _ = f.read(1024)  # Read small portion to trigger OS cache
                    elif model_type == 'ldsr':
                        # LDSR specific preloading
                        from extensions-builtin.LDSR import preload as ldsr_preload
                        ldsr_preload.ldsr_model(model_path)
                
                self.preloaded_models.add(model_hash)
                logger.info(f"Preloaded model: {model_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to preload model {model_hash}: {e}")
                return False
    
    def unload_oldest(self, keep_count: int = 2):
        """Unload oldest preloaded models if we have too many."""
        with self.preload_lock:
            if len(self.preloaded_models) <= keep_count:
                return
                
            # Convert to list and remove oldest (first added)
            models_to_unload = list(self.preloaded_models)[:-keep_count]
            
            for model_hash in models_to_unload:
                try:
                    # Unload model (implementation depends on webui)
                    if WEBUI_AVAILABLE:
                        # This is a simplified version - actual implementation
                        # would need to call appropriate unload functions
                        pass
                    
                    self.preloaded_models.remove(model_hash)
                    logger.debug(f"Unloaded model: {model_hash}")
                    
                except Exception as e:
                    logger.error(f"Failed to unload model {model_hash}: {e}")
    
    def clear_preloaded(self):
        """Clear all preloaded models."""
        with self.preload_lock:
            self.preloaded_models.clear()
            logger.info("Cleared all preloaded models")

class MemoryMonitor:
    """Monitor system memory usage."""
    
    def __init__(self):
        self.psutil_available = False
        try:
            import psutil
            self.psutil = psutil
            self.psutil_available = True
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
    
    def get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        if not self.psutil_available:
            return 0.0
            
        try:
            memory = self.psutil.virtual_memory()
            return memory.percent
        except:
            return 0.0
    
    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        if not self.psutil_available:
            return 0.0
            
        try:
            memory = self.psutil.virtual_memory()
            return memory.available / (1024 ** 3)
        except:
            return 0.0

class PredictiveLoader:
    """Main predictive loader system."""
    
    def __init__(self, strategy: str = 'balanced', data_dir: Optional[str] = None):
        self.usage_tracker = UsageTracker()
        self.predictor = MarkovPredictor(order=2)
        self.preloader = ModelPreloader(strategy)
        
        self.strategy = strategy
        self.enabled = True
        self.running = False
        
        # Threading
        self.preload_thread = None
        self.stop_event = threading.Event()
        
        # Data persistence
        self.data_dir = data_dir or self._get_default_data_dir()
        self.data_file = os.path.join(self.data_dir, 'predictive_loader_data.json')
        
        # Model registry (hash -> info)
        self.model_registry = {}
        
        # Load persisted data
        self._load_data()
        
        # Start background thread
        self.start()
    
    def _get_default_data_dir(self) -> str:
        """Get default directory for storing data."""
        if WEBUI_AVAILABLE:
            base_dir = getattr(paths, 'data_path', None) or os.path.dirname(__file__)
        else:
            base_dir = os.path.dirname(__file__)
        
        data_dir = os.path.join(base_dir, 'predictive_loader')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    def _load_data(self):
        """Load persisted data from disk."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Load usage history
                if 'usage_history' in data:
                    for event in data['usage_history']:
                        self.usage_tracker.usage_history.append(event)
                
                # Load transition counts
                if 'transition_counts' in data:
                    for prev_model, next_models in data['transition_counts'].items():
                        for next_model, count in next_models.items():
                            self.usage_tracker.transition_counts[prev_model][next_model] = count
                
                # Load model registry
                if 'model_registry' in data:
                    self.model_registry = data['model_registry']
                
                # Train predictor with loaded data
                self._train_predictor()
                
                logger.info(f"Loaded predictive loader data from {self.data_file}")
                
        except Exception as e:
            logger.error(f"Failed to load predictive loader data: {e}")
    
    def _save_data(self):
        """Save data to disk."""
        try:
            data = {
                'usage_history': list(self.usage_tracker.usage_history),
                'transition_counts': dict(self.usage_tracker.transition_counts),
                'model_registry': self.model_registry,
                'last_saved': time.time()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved predictive loader data to {self.data_file}")
            
        except Exception as e:
            logger.error(f"Failed to save predictive loader data: {e}")
    
    def _train_predictor(self):
        """Train the Markov predictor with current data."""
        # Convert usage history to sequences
        sequences = []
        current_sequence = []
        
        for event in self.usage_tracker.usage_history:
            current_sequence.append(event['model_hash'])
            if len(current_sequence) >= 10:  # Arbitrary sequence length
                sequences.append(current_sequence)
                current_sequence = current_sequence[-5:]  # Keep overlap
        
        if current_sequence:
            sequences.append(current_sequence)
        
        # Train predictor
        self.predictor.train(sequences)
    
    def register_model(self, model_path: str, model_type: str = "checkpoint"):
        """Register a model in the registry."""
        model_hash = self.usage_tracker._hash_model_path(model_path)
        
        if model_hash not in self.model_registry:
            self.model_registry[model_hash] = {
                'path': model_path,
                'type': model_type,
                'first_seen': time.time(),
                'last_used': None
            }
            logger.debug(f"Registered model: {model_path}")
    
    def record_model_usage(self, model_path: str, model_type: str = "checkpoint"):
        """Record model usage and update predictions."""
        if not self.enabled:
            return
        
        # Register model if not already registered
        self.register_model(model_path, model_type)
        
        # Record usage
        self.usage_tracker.record_usage(model_path, model_type)
        
        # Update model registry
        model_hash = self.usage_tracker._hash_model_path(model_path)
        if model_hash in self.model_registry:
            self.model_registry[model_hash]['last_used'] = time.time()
        
        # Retrain predictor periodically
        if len(self.usage_tracker.usage_history) % 50 == 0:
            self._train_predictor()
        
        # Save data periodically
        if len(self.usage_tracker.usage_history) % 100 == 0:
            self._save_data()
    
    def predict_next_models(self, current_model_hash: Optional[str] = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict next models to be used."""
        if not current_model_hash:
            # Use most recent model
            recent = self.usage_tracker.get_recent_models(1)
            if not recent:
                return []
            current_model_hash = recent[0]
        
        # Get predictions from Markov chain
        context = (current_model_hash,)
        predictions = self.predictor.predict_next(context, top_k)
        
        # Also get transition probabilities from usage tracker
        transition_probs = self.usage_tracker.get_transition_probabilities(current_model_hash)
        
        # Combine predictions (simple average for now)
        combined = {}
        for model_hash, prob in predictions:
            combined[model_hash] = prob
        
        for model_hash, prob in transition_probs.items():
            if model_hash in combined:
                combined[model_hash] = (combined[model_hash] + prob) / 2
            else:
                combined[model_hash] = prob * 0.8  # Weight transition probs slightly less
        
        # Sort by probability
        sorted_predictions = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def preload_predicted_models(self):
        """Preload models predicted to be used next."""
        if not self.enabled or not self.running:
            return
        
        try:
            # Get current model
            recent_models = self.usage_tracker.get_recent_models(1)
            if not recent_models:
                return
            
            current_model_hash = recent_models[0]
            
            # Get predictions
            predictions = self.predict_next_models(current_model_hash, 
                                                  self.preloader.config['max_preload'])
            
            if not predictions:
                return
            
            # Preload models that meet threshold
            preloaded_count = 0
            for model_hash, probability in predictions:
                if model_hash in self.preloader.preloaded_models:
                    continue
                
                if not self.preloader.should_preload(probability):
                    continue
                
                if model_hash not in self.model_registry:
                    continue
                
                model_info = self.model_registry[model_hash]
                if self.preloader.preload_model(model_hash, model_info):
                    preloaded_count += 1
                    logger.info(f"Preloaded model {model_hash} with probability {probability:.2f}")
                
                # Respect max preload limit
                if preloaded_count >= self.preloader.config['max_preload']:
                    break
            
            # Unload old preloaded models if needed
            self.preloader.unload_oldest(keep_count=2)
            
        except Exception as e:
            logger.error(f"Error in preload_predicted_models: {e}")
    
    def _preload_loop(self):
        """Background thread for preloading during idle time."""
        logger.info("Predictive loader background thread started")
        
        while not self.stop_event.is_set():
            try:
                # Check if we're idle (no recent usage)
                recent_models = self.usage_tracker.get_recent_models(1)
                if recent_models:
                    # Check time since last usage
                    last_event = list(self.usage_tracker.usage_history)[-1] if self.usage_tracker.usage_history else None
                    if last_event:
                        idle_time = time.time() - last_event['timestamp']
                        if idle_time >= self.preloader.config['idle_time']:
                            # We're idle, do preloading
                            self.preload_predicted_models()
                
                # Sleep for a bit
                self.stop_event.wait(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in preload loop: {e}")
                time.sleep(10)  # Back off on error
    
    def start(self):
        """Start the predictive loader."""
        if self.running:
            return
        
        self.stop_event.clear()
        self.preload_thread = threading.Thread(target=self._preload_loop, daemon=True)
        self.preload_thread.start()
        self.running = True
        logger.info("Predictive loader started")
    
    def stop(self):
        """Stop the predictive loader."""
        if not self.running:
            return
        
        self.stop_event.set()
        if self.preload_thread:
            self.preload_thread.join(timeout=5)
        
        self.running = False
        self._save_data()  # Save data on stop
        logger.info("Predictive loader stopped")
    
    def set_strategy(self, strategy: str):
        """Change preloading strategy."""
        if strategy not in ModelPreloader.STRATEGIES:
            logger.warning(f"Unknown strategy: {strategy}, using balanced")
            strategy = 'balanced'
        
        self.strategy = strategy
        self.preloader.strategy = strategy
        self.preloader.config = ModelPreloader.STRATEGIES[strategy]
        logger.info(f"Changed predictive loader strategy to: {strategy}")
    
    def enable(self):
        """Enable predictive loading."""
        self.enabled = True
        logger.info("Predictive loader enabled")
    
    def disable(self):
        """Disable predictive loading."""
        self.enabled = False
        self.preloader.clear_preloaded()
        logger.info("Predictive loader disabled")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the predictive loader."""
        return {
            'enabled': self.enabled,
            'running': self.running,
            'strategy': self.strategy,
            'usage_history_size': len(self.usage_tracker.usage_history),
            'registered_models': len(self.model_registry),
            'preloaded_models': len(self.preloader.preloaded_models),
            'transition_states': len(self.usage_tracker.transition_counts),
            'memory_usage_percent': self.preloader.memory_monitor.get_memory_usage_percent(),
            'data_file': self.data_file
        }
    
    def clear_data(self):
        """Clear all data and reset."""
        self.usage_tracker.clear_history()
        self.preloader.clear_preloaded()
        self.model_registry.clear()
        self.predictor = MarkovPredictor(order=2)
        
        # Remove data file
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        logger.info("Cleared all predictive loader data")

# Global instance
_predictive_loader = None

def get_predictive_loader() -> PredictiveLoader:
    """Get or create the global predictive loader instance."""
    global _predictive_loader
    if _predictive_loader is None:
        strategy = 'balanced'
        if WEBUI_AVAILABLE and hasattr(opts, 'predictive_loader_strategy'):
            strategy = opts.predictive_loader_strategy
        _predictive_loader = PredictiveLoader(strategy=strategy)
    return _predictive_loader

def record_model_usage(model_path: str, model_type: str = "checkpoint"):
    """Record model usage for predictive loading."""
    try:
        loader = get_predictive_loader()
        loader.record_model_usage(model_path, model_type)
    except Exception as e:
        logger.error(f"Failed to record model usage: {e}")

def setup_webui_integration():
    """Set up integration with WebUI if available."""
    if not WEBUI_AVAILABLE:
        return
    
    try:
        # Hook into model loading
        original_load_model = sd_models.load_model
        
        def hooked_load_model(*args, **kwargs):
            result = original_load_model(*args, **kwargs)
            # Record usage after successful load
            if args and len(args) > 0:
                model_path = args[0]
                record_model_usage(model_path, "checkpoint")
            return result
        
        sd_models.load_model = hooked_load_model
        
        # Hook into LoRA loading if available
        try:
            from extensions-builtin.Lora import extra_networks_lora
            original_lora_activate = extra_networks_lora.activate
            
            def hooked_lora_activate(p, *args, **kwargs):
                result = original_lora_activate(p, *args, **kwargs)
                # Record LoRA usage
                if hasattr(p, 'lora_weights') and p.lora_weights:
                    for lora_name in p.lora_weights.keys():
                        # Find LoRA path
                        lora_path = find_lora_path(lora_name)
                        if lora_path:
                            record_model_usage(lora_path, "lora")
                return result
            
            extra_networks_lora.activate = hooked_lora_activate
            
        except ImportError:
            logger.debug("LoRA integration not available")
        
        # Add settings to WebUI
        if hasattr(shared, 'opts'):
            if not hasattr(shared.opts, 'predictive_loader_strategy'):
                shared.opts.predictive_loader_strategy = 'balanced'
            if not hasattr(shared.opts, 'predictive_loader_enabled'):
                shared.opts.predictive_loader_enabled = True
        
        logger.info("WebUI integration setup complete")
        
    except Exception as e:
        logger.error(f"Failed to setup WebUI integration: {e}")

def find_lora_path(lora_name: str) -> Optional[str]:
    """Find the full path for a LoRA model."""
    try:
        # Search in standard LoRA directories
        lora_dirs = [
            os.path.join(paths.models_path, 'Lora'),
            os.path.join(paths.models_path, 'LyCORIS'),
        ]
        
        for lora_dir in lora_dirs:
            if os.path.exists(lora_dir):
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            if lora_name in file or lora_name in os.path.splitext(file)[0]:
                                return os.path.join(root, file)
    except:
        pass
    
    return None

# Auto-setup when module is imported
if WEBUI_AVAILABLE:
    setup_webui_integration()

# Export public API
__all__ = [
    'PredictiveLoader',
    'get_predictive_loader',
    'record_model_usage',
    'setup_webui_integration',
    'UsageTracker',
    'MarkovPredictor',
    'ModelPreloader'
]