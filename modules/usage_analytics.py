"""Predictive Model Preloading for Stable Diffusion WebUI

This module implements intelligent model preloading based on user usage patterns.
It uses Markov chain predictors to anticipate which models will be needed next
and preloads them during idle time to reduce generation latency.
"""

import json
import threading
import time
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class PreloadingStrategy(Enum):
    """Preloading strategy configurations."""
    AGGRESSIVE = "aggressive"    # Preloads more models, higher memory usage
    BALANCED = "balanced"        # Balanced approach (default)
    CONSERVATIVE = "conservative" # Minimal preloading, lower memory usage

@dataclass
class UsagePattern:
    """Represents a model usage pattern with transition probabilities."""
    model_name: str
    transition_counts: Dict[str, int]  # next_model -> count
    total_transitions: int
    
    def get_transition_probability(self, next_model: str) -> float:
        """Calculate transition probability to next_model."""
        if self.total_transitions == 0:
            return 0.0
        return self.transition_counts.get(next_model, 0) / self.total_transitions

@dataclass
class AnalyticsConfig:
    """Configuration for usage analytics and preloading."""
    enabled: bool = True
    strategy: PreloadingStrategy = PreloadingStrategy.BALANCED
    max_preload_memory_mb: int = 2048  # Maximum memory for preloaded models
    idle_threshold_seconds: float = 2.0  # Time before considering system idle
    pattern_history_size: int = 100  # Number of patterns to remember
    min_pattern_confidence: float = 0.3  # Minimum probability to trigger preload
    privacy_mode: bool = True  # Hash model names for privacy
    data_retention_days: int = 30  # Days to keep usage data

class PrivacyPreservingAnalytics:
    """Handles usage tracking with privacy considerations."""
    
    def __init__(self, privacy_mode: bool = True):
        self.privacy_mode = privacy_mode
        self._salt = self._generate_salt()
        
    def _generate_salt(self) -> str:
        """Generate a random salt for hashing."""
        import secrets
        return secrets.token_hex(16)
    
    def anonymize_model_name(self, model_name: str) -> str:
        """Anonymize model name if privacy mode is enabled."""
        if not self.privacy_mode:
            return model_name
        
        # Create a consistent hash for the same model name
        hash_input = f"{model_name}:{self._salt}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_data_path(self) -> Path:
        """Get path for storing analytics data."""
        data_dir = Path("data/usage_analytics")
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "usage_patterns.json"

class MarkovPredictor:
    """Markov chain-based model usage predictor."""
    
    def __init__(self, history_size: int = 100):
        self.patterns: Dict[str, UsagePattern] = {}
        self.recent_sequence: deque = deque(maxlen=5)  # Last 5 models used
        self.history_size = history_size
        self._lock = threading.RLock()
        
    def update_pattern(self, current_model: str, next_model: str) -> None:
        """Update transition pattern between models."""
        with self._lock:
            if current_model not in self.patterns:
                self.patterns[current_model] = UsagePattern(
                    model_name=current_model,
                    transition_counts=defaultdict(int),
                    total_transitions=0
                )
            
            pattern = self.patterns[current_model]
            pattern.transition_counts[next_model] += 1
            pattern.total_transitions += 1
            
            # Update recent sequence
            if not self.recent_sequence or self.recent_sequence[-1] != current_model:
                self.recent_sequence.append(current_model)
    
    def predict_next_models(self, current_model: str, n: int = 3) -> List[Tuple[str, float]]:
        """Predict next n most likely models with probabilities."""
        with self._lock:
            if current_model not in self.patterns:
                return []
            
            pattern = self.patterns[current_model]
            predictions = []
            
            for model, count in pattern.transition_counts.items():
                probability = count / pattern.total_transitions
                predictions.append((model, probability))
            
            # Sort by probability descending
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n]
    
    def get_high_confidence_predictions(self, current_model: str, 
                                      min_confidence: float = 0.3) -> List[str]:
        """Get predictions with confidence above threshold."""
        predictions = self.predict_next_models(current_model, n=5)
        return [model for model, prob in predictions if prob >= min_confidence]
    
    def get_sequence_predictions(self, n: int = 3) -> List[Tuple[str, float]]:
        """Predict based on recent sequence of models."""
        with self._lock:
            if len(self.recent_sequence) < 2:
                return []
            
            # Use last model for prediction
            last_model = self.recent_sequence[-1]
            return self.predict_next_models(last_model, n)
    
    def to_dict(self) -> dict:
        """Serialize predictor state to dictionary."""
        with self._lock:
            return {
                "patterns": {
                    name: {
                        "model_name": pattern.model_name,
                        "transition_counts": dict(pattern.transition_counts),
                        "total_transitions": pattern.total_transitions
                    }
                    for name, pattern in self.patterns.items()
                },
                "recent_sequence": list(self.recent_sequence)
            }
    
    def from_dict(self, data: dict) -> None:
        """Load predictor state from dictionary."""
        with self._lock:
            self.patterns.clear()
            
            for name, pattern_data in data.get("patterns", {}).items():
                self.patterns[name] = UsagePattern(
                    model_name=pattern_data["model_name"],
                    transition_counts=defaultdict(int, pattern_data["transition_counts"]),
                    total_transitions=pattern_data["total_transitions"]
                )
            
            self.recent_sequence = deque(
                data.get("recent_sequence", []),
                maxlen=5
            )

class ModelMemoryManager:
    """Manages model memory usage and preloading decisions."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.loaded_models: Dict[str, int] = {}  # model_name -> estimated_size_bytes
        self.current_memory_usage = 0
        self._lock = threading.RLock()
        
    def estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in bytes (simplified)."""
        # In production, this would query actual model size
        # For now, use heuristic based on model type
        if "ldsr" in model_name.lower():
            return 2 * 1024 * 1024 * 1024  # 2GB for LDSR
        elif "lora" in model_name.lower():
            return 100 * 1024 * 1024  # 100MB for LoRA
        else:
            return 500 * 1024 * 1024  # 500MB default
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if model can be loaded within memory constraints."""
        with self._lock:
            model_size = self.estimate_model_size(model_name)
            return (self.current_memory_usage + model_size) <= self.max_memory_bytes
    
    def register_loaded_model(self, model_name: str) -> None:
        """Register a model as loaded."""
        with self._lock:
            if model_name not in self.loaded_models:
                model_size = self.estimate_model_size(model_name)
                self.loaded_models[model_name] = model_size
                self.current_memory_usage += model_size
                logger.debug(f"Registered loaded model: {model_name} ({model_size} bytes)")
    
    def unregister_loaded_model(self, model_name: str) -> None:
        """Unregister a model as loaded."""
        with self._lock:
            if model_name in self.loaded_models:
                model_size = self.loaded_models[model_name]
                self.current_memory_usage -= model_size
                del self.loaded_models[model_name]
                logger.debug(f"Unregistered model: {model_name}")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.current_memory_usage / (1024 * 1024)
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return (self.max_memory_bytes - self.current_memory_usage) / (1024 * 1024)

class UsageAnalytics:
    """Main class for usage analytics and predictive preloading."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize usage analytics system."""
        if self._initialized:
            return
            
        self.config = AnalyticsConfig()
        self.analytics = PrivacyPreservingAnalytics(self.config.privacy_mode)
        self.predictor = MarkovPredictor(self.config.pattern_history_size)
        self.memory_manager = ModelMemoryManager(self.config.max_preload_memory_mb)
        
        self.last_model_used: Optional[str] = None
        self.last_activity_time = time.time()
        self.idle_timer: Optional[threading.Timer] = None
        self.is_preloading = False
        
        self._lock = threading.RLock()
        self._load_data()
        
        # Start idle monitoring thread
        self._start_idle_monitor()
        
        self._initialized = True
        logger.info("Usage analytics initialized with strategy: %s", self.config.strategy.value)
    
    def _start_idle_monitor(self) -> None:
        """Start background thread to monitor idle time."""
        def monitor_idle():
            while True:
                time.sleep(1.0)
                with self._lock:
                    idle_time = time.time() - self.last_activity_time
                    
                    if (idle_time >= self.config.idle_threshold_seconds and 
                        not self.is_preloading and 
                        self.config.enabled):
                        self._trigger_preloading()
        
        monitor_thread = threading.Thread(
            target=monitor_idle,
            daemon=True,
            name="UsageAnalyticsIdleMonitor"
        )
        monitor_thread.start()
    
    def _trigger_preloading(self) -> None:
        """Trigger preloading based on current strategy."""
        if not self.last_model_used:
            return
            
        self.is_preloading = True
        
        try:
            # Get predictions based on strategy
            predictions = self._get_strategy_predictions()
            
            # Preload models
            for model_name in predictions:
                if self.memory_manager.can_load_model(model_name):
                    self._preload_model(model_name)
                else:
                    logger.warning(f"Cannot preload {model_name}: memory limit reached")
                    break
        finally:
            self.is_preloading = False
    
    def _get_strategy_predictions(self) -> List[str]:
        """Get predictions based on configured strategy."""
        if not self.last_model_used:
            return []
        
        if self.config.strategy == PreloadingStrategy.AGGRESSIVE:
            # Preload top 3 models with confidence > 0.2
            predictions = self.predictor.predict_next_models(
                self.last_model_used, n=3
            )
            return [model for model, prob in predictions if prob >= 0.2]
            
        elif self.config.strategy == PreloadingStrategy.BALANCED:
            # Preload top 2 models with confidence > 0.3
            predictions = self.predictor.predict_next_models(
                self.last_model_used, n=2
            )
            return [model for model, prob in predictions if prob >= 0.3]
            
        else:  # CONSERVATIVE
            # Preload only top model with confidence > 0.5
            predictions = self.predictor.predict_next_models(
                self.last_model_used, n=1
            )
            return [model for model, prob in predictions if prob >= 0.5]
    
    def _preload_model(self, model_name: str) -> None:
        """Preload a model (integration point with actual preloading)."""
        logger.info(f"Preloading model: {model_name}")
        
        # This is where integration with actual model loading would happen
        # For now, we just register it as loaded in our memory manager
        # In production, this would call the appropriate model loading function
        
        try:
            # Simulate model loading
            self.memory_manager.register_loaded_model(model_name)
            logger.info(f"Successfully preloaded: {model_name}")
            
            # Update activity time to prevent continuous preloading
            self.last_activity_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to preload {model_name}: {e}")
    
    def record_model_usage(self, model_name: str) -> None:
        """Record usage of a model for pattern learning."""
        if not self.config.enabled:
            return
            
        with self._lock:
            # Anonymize model name for storage
            anonymized_name = self.analytics.anonymize_model_name(model_name)
            
            # Update predictor if we have a previous model
            if self.last_model_used:
                anonymized_last = self.analytics.anonymize_model_name(self.last_model_used)
                self.predictor.update_pattern(anonymized_last, anonymized_name)
            
            # Update last used model
            self.last_model_used = model_name
            self.last_activity_time = time.time()
            
            # Register as loaded
            self.memory_manager.register_loaded_model(model_name)
            
            # Save data periodically
            self._save_data_if_needed()
            
            logger.debug(f"Recorded usage: {model_name}")
    
    def _save_data_if_needed(self) -> None:
        """Save analytics data periodically."""
        # Save every 10 recordings
        if not hasattr(self, '_save_counter'):
            self._save_counter = 0
            
        self._save_counter += 1
        if self._save_counter >= 10:
            self._save_data()
            self._save_counter = 0
    
    def _save_data(self) -> None:
        """Save analytics data to disk."""
        try:
            data_path = self.analytics.get_data_path()
            
            data = {
                "config": asdict(self.config),
                "predictor": self.predictor.to_dict(),
                "last_save_time": time.time()
            }
            
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved analytics data")
            
        except Exception as e:
            logger.error(f"Failed to save analytics data: {e}")
    
    def _load_data(self) -> None:
        """Load analytics data from disk."""
        try:
            data_path = self.analytics.get_data_path()
            
            if not data_path.exists():
                logger.info("No existing analytics data found")
                return
                
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Load config
            config_data = data.get("config", {})
            self.config.enabled = config_data.get("enabled", True)
            self.config.strategy = PreloadingStrategy(
                config_data.get("strategy", "balanced")
            )
            self.config.max_preload_memory_mb = config_data.get(
                "max_preload_memory_mb", 2048
            )
            self.config.idle_threshold_seconds = config_data.get(
                "idle_threshold_seconds", 2.0
            )
            self.config.pattern_history_size = config_data.get(
                "pattern_history_size", 100
            )
            self.config.min_pattern_confidence = config_data.get(
                "min_pattern_confidence", 0.3
            )
            self.config.privacy_mode = config_data.get("privacy_mode", True)
            self.config.data_retention_days = config_data.get(
                "data_retention_days", 30
            )
            
            # Load predictor
            predictor_data = data.get("predictor", {})
            self.predictor.from_dict(predictor_data)
            
            # Update memory manager config
            self.memory_manager.max_memory_bytes = (
                self.config.max_preload_memory_mb * 1024 * 1024
            )
            
            logger.info("Loaded analytics data")
            
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
                    # Update dependent components
                    if key == "max_preload_memory_mb":
                        self.memory_manager.max_memory_bytes = value * 1024 * 1024
                    elif key == "privacy_mode":
                        self.analytics.privacy_mode = value
            
            self._save_data()
            logger.info("Updated analytics configuration")
    
    def get_statistics(self) -> Dict:
        """Get usage statistics and predictions."""
        with self._lock:
            stats = {
                "enabled": self.config.enabled,
                "strategy": self.config.strategy.value,
                "memory_usage_mb": self.memory_manager.get_memory_usage_mb(),
                "available_memory_mb": self.memory_manager.get_available_memory_mb(),
                "patterns_learned": len(self.predictor.patterns),
                "last_model": self.last_model_used,
                "predictions": []
            }
            
            if self.last_model_used:
                anonymized = self.analytics.anonymize_model_name(self.last_model_used)
                predictions = self.predictor.predict_next_models(anonymized, n=5)
                stats["predictions"] = [
                    {"model": model, "probability": prob}
                    for model, prob in predictions
                ]
            
            return stats
    
    def clear_data(self) -> None:
        """Clear all analytics data."""
        with self._lock:
            self.predictor.patterns.clear()
            self.predictor.recent_sequence.clear()
            self.memory_manager.loaded_models.clear()
            self.memory_manager.current_memory_usage = 0
            self.last_model_used = None
            
            # Delete data file
            data_path = self.analytics.get_data_path()
            if data_path.exists():
                data_path.unlink()
            
            logger.info("Cleared all analytics data")
    
    def set_strategy(self, strategy: str) -> None:
        """Set preloading strategy by name."""
        try:
            strategy_enum = PreloadingStrategy(strategy.lower())
            self.update_config(strategy=strategy_enum.value)
        except ValueError:
            logger.error(f"Invalid strategy: {strategy}")

# Global instance for easy access
usage_analytics = UsageAnalytics()

# Integration hooks for existing modules
def on_model_loaded(model_name: str) -> None:
    """Hook to call when a model is loaded."""
    usage_analytics.record_model_usage(model_name)

def on_model_unloaded(model_name: str) -> None:
    """Hook to call when a model is unloaded."""
    usage_analytics.memory_manager.unregister_loaded_model(model_name)

def get_predictions(current_model: str) -> List[str]:
    """Get predicted next models for a given model."""
    anonymized = usage_analytics.analytics.anonymize_model_name(current_model)
    predictions = usage_analytics.predictor.predict_next_models(anonymized, n=3)
    return [model for model, _ in predictions]

def configure_analytics(**kwargs) -> None:
    """Configure analytics settings."""
    usage_analytics.update_config(**kwargs)

def get_analytics_stats() -> Dict:
    """Get current analytics statistics."""
    return usage_analytics.get_statistics()

# Example integration with existing LDSR module
def integrate_with_ldsr():
    """Example integration with LDSR module."""
    try:
        # This would be called from the actual LDSR module
        from modules import sd_models
        
        # Hook into model loading
        original_load_model = sd_models.load_model
        
        def hooked_load_model(*args, **kwargs):
            result = original_load_model(*args, **kwargs)
            if hasattr(sd_models, 'current_model') and sd_models.current_model:
                on_model_loaded(sd_models.current_model.name)
            return result
        
        sd_models.load_model = hooked_load_model
        
        logger.info("Integrated usage analytics with model loading")
        
    except ImportError:
        logger.warning("Could not integrate with sd_models module")

# Auto-integrate when module is loaded
try:
    integrate_with_ldsr()
except Exception as e:
    logger.warning(f"Auto-integration failed: {e}")