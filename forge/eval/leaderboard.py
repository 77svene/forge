import os
import json
import time
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import numpy as np

from forge.data import get_template
from forge.train import train
from forge.llm import load_model, get_template
from forge.data import get_template
from forge.utils import get_current_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Benchmark Registry Constants
BENCHMARK_REGISTRY = {
    "mmlu": {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding",
        "dataset_name": "HuggingFaceH4/mmlu",
        "metric": "acc",
        "category": "reasoning",
        "max_samples": 1000
    },
    "human_eval": {
        "name": "HumanEval",
        "description": "Python function generation",
        "dataset_name": "openai/human_eval",
        "metric": "pass@1",
        "category": "coding",
        "max_samples": 100
    },
    "ceval": {
        "name": "C-Eval",
        "description": "Comprehensive Evaluation for Chinese LLMs",
        "dataset_name": "ceval-benchmark/ceval",
        "metric": "acc",
        "category": "knowledge",
        "max_samples": 1000
    },
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade School Math 8K",
        "dataset_name": "openai/gsm8k",
        "metric": "acc",
        "category": "reasoning",
        "max_samples": 500
    },
    "mbpp": {
        "name": "MBPP",
        "description": "Mostly Basic Python Programming",
        "dataset_name": "google-research-datasets/mbpp",
        "metric": "acc",
        "category": "coding",
        "max_samples": 1000
    }
}

@dataclass
class BenchmarkConfig:
    name: str
    model_path: str
    device: str = "auto"
    batch_size: int = 1
    max_samples: int = -1  # -1 means all
    output_dir: str = "./eval_results"
    trust_remote_code: bool = False
    dataset_cache_dir: str = "./cache"
    template: str = "default"
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return asdict(self)

@dataclass
class EvaluationResult:
    benchmark_name: str
    model_name: str
    timestamp: str
    score: float
    metric: str
    raw_output: Optional[str] = None
    details: Optional[Dict] = None

    def to_dict(self):
        d = asdict(self)
        if self.raw_output:
            d["raw_output"] = self.raw_output[:1000]  # truncate for storage
        return d

class BenchmarkRegistry:
    """Singleton registry for benchmark configurations."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.registry = BENCHMARK_REGISTRY
        return cls._instance

    @staticmethod
    def register(name: str, config: Dict[str, Any]):
        """Register a new benchmark configuration."""
        BenchmarkRegistry._instance.registry[name] = config
        logger.info(f"Registered benchmark: {name}")

    @staticmethod
    def get(name: str) -> Dict[str, Any]:
        """Get benchmark configuration by name."""
        if name not in BenchmarkRegistry._instance.registry:
            raise ValueError(f"Benchmark '{name}' not found in registry.")
        return BenchmarkRegistry._instance.registry[name]

    @staticmethod
    def list_benchmarks() -> List[str]:
        return list(BenchmarkRegistry._instance.registry.keys())

class LeaderboardRunner:
    """Main class to orchestrate benchmarking and leaderboard generation."""

    def __init__(self, model_path: str, device: str = "auto", batch_size: int = 1):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.results: List[EvaluationResult] = []
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                device_map="auto" if self.device == "auto" else self.device
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto" if self.device == "auto" else self.device
            )
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_prompt(self, prompt: str, template_name: str = "default") -> str:
        """Format prompt based on template."""
        try:
            template = get_template(template_name)
            return template.apply(prompt)
        except Exception:
            return prompt

    def _evaluate_single(self, benchmark_name: str, prompt: str) -> str:
        """Run inference for a single sample."""
        if self.model_path:
            if self.model is None:
                self._load_model()
            
            self.model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        return ""

    def run_benchmark(self, benchmark_name: str, max_samples: Optional[int] = None) -> List[EvaluationResult]:
        """Run a specific benchmark."""
        config = BenchmarkRegistry.get(benchmark_name)
        dataset = load_dataset(config["dataset_name"], split="test")
        
        # Limit samples
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Starting benchmark: {benchmark_name} with {len(dataset)} samples.")
        
        results = []
        for idx, row in dataset.iterrows():
            prompt = row["question"] if "question" in row else row["input"]
            start_time = time.time()
            response = self._evaluate_single(benchmark_name, prompt)
            # Simple accuracy check logic placeholder
            # In production, specific parsers are needed for MMLU/HumanEval
            score = self._calculate_metric(response, row)
            
            result = EvaluationResult(
                benchmark_name=benchmark_name,
                model_name=self.model_path,
                timestamp=get_current_timestamp(),
                score=score,
                metric=config["metric"],
                raw_output=response
            )
            results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples.")
        
        self.results.extend(results)
        return results

    def _calculate_metric(self, response: str, row: Dict) -> float:
        """Calculate metric based on benchmark type."""
        # Placeholder for actual metric calculation logic
        # For MMLU/CEval, parse the correct option
        # For HumanEval, run code and check output
        if "answer" in row:
            if response.strip().lower() in row["answer"].lower():
                return 1.0
            return 0.0
        return 0.5  # Default for demo

    def run_all_benchmarks(self, model_path: Optional[str] = None, batch_size: Optional[int] = None) -> List[EvaluationResult]:
        """Run all registered benchmarks for a model."""
        if model_path:
            self.model_path = model_path
            self.batch_size = batch_size or self.batch_size
        elif not self.model_path:
            raise ValueError("Model path must be provided if not set during init.")
        
        self.results = []
        for name in BenchmarkRegistry.list_benchmarks():
            try:
                logger.info(f"Running {name}...")
                self.run_benchmark(name)
            except Exception as e:
                logger.warning(f"Failed to run {name}: {e}")
        return self.results

    def save_results(self, output_dir: Optional[str] = None):
        """Save results to CSV and JSON."""
        if output_dir:
            self.results[0].output_dir = Path(output_dir)