#!/usr/bin/env python3
"""
Unified Multi-Modal Evaluation Framework for forge
Replaces basic BLEU/ROUGE evaluation with comprehensive benchmarking suite
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import importlib
import inspect
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    from peft import PeftModel
except ImportError:
    logger.warning("transformers or peft not installed. Some functionality may be limited.")

@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results"""
    benchmark_name: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "num_predictions": len(self.predictions)
        }
    
    def save(self, path: str):
        """Save results to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BenchmarkResult':
        """Load results from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class BenchmarkPlugin(ABC):
    """Abstract base class for benchmark plugins"""
    
    def __init__(self, name: str, cache_dir: Optional[str] = None):
        self.name = name
        self.cache_dir = cache_dir or os.path.join(project_root, "cache", "benchmarks")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    @abstractmethod
    def load_dataset(self, split: str = "test") -> Dataset:
        """Load and preprocess the benchmark dataset"""
        pass
    
    @abstractmethod
    def evaluate(self, model, tokenizer, dataset: Dataset, **kwargs) -> BenchmarkResult:
        """Run evaluation on the dataset"""
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        pass
    
    def get_cache_key(self, model_name: str, dataset_name: str) -> str:
        """Generate cache key for results"""
        key_str = f"{self.name}_{model_name}_{dataset_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache_results(self, result: BenchmarkResult):
        """Cache evaluation results"""
        cache_key = self.get_cache_key(result.model_name, result.dataset_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        result.save(cache_path)
        logger.info(f"Cached results to {cache_path}")
    
    def load_cached_results(self, model_name: str, dataset_name: str) -> Optional[BenchmarkResult]:
        """Load cached results if available"""
        cache_key = self.get_cache_key(model_name, dataset_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached results from {cache_path}")
            return BenchmarkResult.load(cache_path)
        return None

class MMMUBenchmark(BenchmarkPlugin):
    """MMMU (Massive Multi-discipline Multimodal Understanding) Benchmark"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("MMMU", cache_dir)
        self.subjects = [
            "Accounting", "Agriculture", "Architecture_and_Engineering",
            "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
            "Chemistry", "Clinical_Medicine", "Computer_Science",
            "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
            "Electronics", "Energy_and_Power", "Finance", "Geography",
            "History", "Literature", "Manage", "Marketing", "Materials",
            "Math", "Mechanical_Engineering", "Music", "Pharmacy",
            "Physics", "Psychology", "Public_Health", "Sociology"
        ]
        
    def load_dataset(self, split: str = "validation") -> Dataset:
        """Load MMMU dataset from Hugging Face"""
        try:
            # Try to load from Hugging Face datasets
            dataset = load_dataset("MMMU/MMMU", split=split, trust_remote_code=True)
            logger.info(f"Loaded MMMU dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load MMMU dataset: {e}")
            raise
    
    def preprocess_example(self, example: Dict) -> Dict:
        """Preprocess a single MMMU example"""
        # MMMU format: image(s), question, options, answer
        processed = {
            "id": example.get("id", ""),
            "question": example["question"],
            "options": example["options"],
            "answer": example["answer"],
            "subject": example.get("subject", "unknown"),
            "image": example.get("image", None),
            "images": example.get("images", [])
        }
        
        # Handle multiple images
        if not processed["image"] and processed["images"]:
            processed["image"] = processed["images"][0] if processed["images"] else None
            
        return processed
    
    def format_prompt(self, example: Dict) -> str:
        """Format MMMU example into model prompt"""
        prompt = f"Question: {example['question']}\n\n"
        
        if example.get("options"):
            prompt += "Options:\n"
            for i, option in enumerate(example["options"]):
                prompt += f"{chr(65 + i)}. {option}\n"
        
        prompt += "\nAnswer with the letter of the correct option."
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract answer letter from model response"""
        response = response.strip().upper()
        # Look for single letter answer
        for char in response:
            if char in "ABCDEFGH":
                return char
        # Fallback: look for "A", "B", etc. in text
        for letter in "ABCDEFGH":
            if f" {letter} " in f" {response} " or response.startswith(letter):
                return letter
        return response[0] if response else ""
    
    def evaluate(self, model, tokenizer, dataset: Dataset, 
                 batch_size: int = 4, max_new_tokens: int = 512,
                 device: str = "auto", **kwargs) -> BenchmarkResult:
        """Run MMMU evaluation"""
        
        model_name = getattr(model, 'name_or_path', 'unknown_model')
        cached = self.load_cached_results(model_name, "MMMU")
        if cached and not kwargs.get("force_rerun", False):
            return cached
        
        predictions = []
        references = []
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Starting MMMU evaluation on {len(dataset)} examples")
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating MMMU"):
            batch = dataset[i:i + batch_size]
            batch_examples = [self.preprocess_example({k: batch[k][j] for k in batch.keys()}) 
                            for j in range(len(batch[list(batch.keys())[0]]))]
            
            for example in batch_examples:
                try:
                    # Format prompt
                    prompt = self.format_prompt(example)
                    
                    # Handle image if present
                    if example.get("image") and hasattr(model, 'generate'):
                        # For vision-language models
                        inputs = self._prepare_vision_inputs(example, tokenizer, device)
                    else:
                        # Text-only model
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                         max_length=2048).to(device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], 
                                              skip_special_tokens=True)
                    
                    # Extract answer
                    predicted_answer = self.extract_answer(response)
                    
                    predictions.append({
                        "id": example["id"],
                        "question": example["question"],
                        "predicted": predicted_answer,
                        "response": response,
                        "subject": example["subject"]
                    })
                    
                    references.append({
                        "id": example["id"],
                        "answer": example["answer"],
                        "subject": example["subject"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing example {example.get('id', i)}: {e}")
                    predictions.append({
                        "id": example.get("id", str(i)),
                        "predicted": "",
                        "error": str(e)
                    })
                    references.append({
                        "id": example.get("id", str(i)),
                        "answer": example.get("answer", ""),
                        "subject": example.get("subject", "unknown")
                    })
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, references)
        
        # Create result
        result = BenchmarkResult(
            benchmark_name="MMMU",
            model_name=model_name,
            dataset_name="MMMU",
            metrics=metrics,
            predictions=predictions,
            metadata={
                "num_examples": len(dataset),
                "batch_size": batch_size,
                "device": device,
                "subjects": list(set(r["subject"] for r in references))
            }
        )
        
        # Cache results
        self.cache_results(result)
        
        return result
    
    def _prepare_vision_inputs(self, example: Dict, tokenizer, device: str) -> Dict:
        """Prepare inputs for vision-language models"""
        # This is a simplified version - actual implementation depends on model architecture
        prompt = self.format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                          max_length=2048).to(device)
        
        # Add image placeholder if model supports it
        if hasattr(tokenizer, 'image_token'):
            # Some models have special image tokens
            pass
            
        return inputs
    
    def compute_metrics(self, predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute MMMU metrics (accuracy by subject and overall)"""
        correct = 0
        total = 0
        subject_correct = defaultdict(int)
        subject_total = defaultdict(int)
        
        for pred, ref in zip(predictions, references):
            if "error" in pred:
                continue
                
            total += 1
            subject = ref.get("subject", "unknown")
            subject_total[subject] += 1
            
            if pred["predicted"].upper() == ref["answer"].upper():
                correct += 1
                subject_correct[subject] += 1
        
        # Calculate overall accuracy
        overall_accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-subject accuracy
        subject_accuracies = {}
        for subject in subject_total:
            if subject_total[subject] > 0:
                subject_accuracies[f"accuracy_{subject}"] = (
                    subject_correct[subject] / subject_total[subject]
                )
        
        metrics = {
            "accuracy": overall_accuracy,
            "total_examples": total,
            "correct_answers": correct,
            **subject_accuracies
        }
        
        return metrics

class MMLUBenchmark(BenchmarkPlugin):
    """MMLU (Massive Multitask Language Understanding) Benchmark"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("MMLU", cache_dir)
        
    def load_dataset(self, split: str = "test") -> Dataset:
        """Load MMLU dataset"""
        try:
            dataset = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
            logger.info(f"Loaded MMLU dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load MMLU dataset: {e}")
            raise
    
    def evaluate(self, model, tokenizer, dataset: Dataset, **kwargs) -> BenchmarkResult:
        """Run MMLU evaluation"""
        # Similar implementation to MMMU but for text-only
        model_name = getattr(model, 'name_or_path', 'unknown_model')
        cached = self.load_cached_results(model_name, "MMLU")
        if cached and not kwargs.get("force_rerun", False):
            return cached
        
        # Simplified implementation - would follow similar pattern to MMMU
        result = BenchmarkResult(
            benchmark_name="MMLU",
            model_name=model_name,
            dataset_name="MMLU",
            metrics={"accuracy": 0.0},  # Placeholder
            predictions=[],
            metadata={"status": "not_implemented"}
        )
        
        self.cache_results(result)
        return result
    
    def compute_metrics(self, predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute MMLU metrics"""
        return {"accuracy": 0.0}  # Placeholder

class HumanEvalBenchmark(BenchmarkPlugin):
    """HumanEval Benchmark for code generation"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("HumanEval", cache_dir)
        
    def load_dataset(self, split: str = "test") -> Dataset:
        """Load HumanEval dataset"""
        try:
            dataset = load_dataset("openai_humaneval", split=split, trust_remote_code=True)
            logger.info(f"Loaded HumanEval dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load HumanEval dataset: {e}")
            raise
    
    def evaluate(self, model, tokenizer, dataset: Dataset, **kwargs) -> BenchmarkResult:
        """Run HumanEval evaluation"""
        model_name = getattr(model, 'name_or_path', 'unknown_model')
        cached = self.load_cached_results(model_name, "HumanEval")
        if cached and not kwargs.get("force_rerun", False):
            return cached
        
        # Simplified implementation
        result = BenchmarkResult(
            benchmark_name="HumanEval",
            model_name=model_name,
            dataset_name="HumanEval",
            metrics={"pass@1": 0.0},  # Placeholder
            predictions=[],
            metadata={"status": "not_implemented"}
        )
        
        self.cache_results(result)
        return result
    
    def compute_metrics(self, predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute HumanEval metrics (pass@k)"""
        return {"pass@1": 0.0}  # Placeholder

class BenchmarkRegistry:
    """Registry for benchmark plugins"""
    
    def __init__(self):
        self._benchmarks = {}
        self._discover_benchmarks()
    
    def _discover_benchmarks(self):
        """Discover and register available benchmarks"""
        # Register built-in benchmarks
        self.register("MMMU", MMMUBenchmark)
        self.register("MMLU", MMLUBenchmark)
        self.register("HumanEval", HumanEvalBenchmark)
        
        # Try to discover additional benchmarks from plugins directory
        plugins_dir = Path(__file__).parent / "plugins"
        if plugins_dir.exists():
            self._load_plugins(plugins_dir)
    
    def _load_plugins(self, plugins_dir: Path):
        """Load benchmark plugins from directory"""
        for plugin_file in plugins_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"benchmark_plugin_{plugin_file.stem}",
                    plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for BenchmarkPlugin subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BenchmarkPlugin) and 
                        obj != BenchmarkPlugin):
                        self.register(name, obj)
                        logger.info(f"Loaded benchmark plugin: {name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {plugin_file}: {e}")
    
    def register(self, name: str, benchmark_class):
        """Register a benchmark class"""
        self._benchmarks[name] = benchmark_class
    
    def get(self, name: str) -> Optional[BenchmarkPlugin]:
        """Get benchmark instance by name"""
        if name in self._benchmarks:
            return self._benchmarks[name]()
        return None
    
    def list_available(self) -> List[str]:
        """List available benchmark names"""
        return list(self._benchmarks.keys())

class Leaderboard:
    """Leaderboard for comparing model performance across benchmarks"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(project_root, "leaderboard")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the leaderboard"""
        # Store individual result
        result_dir = os.path.join(self.storage_dir, result.benchmark_name)
        os.makedirs(result_dir, exist_ok=True)
        
        filename = f"{result.model_name.replace('/', '_')}_{int(result.timestamp)}.json"
        result.save(os.path.join(result_dir, filename))
        
        # Update aggregated leaderboard
        self._update_leaderboard(result)
    
    def _update_leaderboard(self, result: BenchmarkResult):
        """Update the aggregated leaderboard"""
        leaderboard_file = os.path.join(self.storage_dir, "leaderboard.json")
        
        if os.path.exists(leaderboard_file):
            with open(leaderboard_file, 'r') as f:
                leaderboard = json.load(f)
        else:
            leaderboard = {}
        
        model_name = result.model_name
        benchmark_name = result.benchmark_name
        
        if model_name not in leaderboard:
            leaderboard[model_name] = {}
        
        leaderboard[model_name][benchmark_name] = {
            "metrics": result.metrics,
            "timestamp": result.timestamp,
            "num_examples": result.metadata.get("num_examples", 0)
        }
        
        with open(leaderboard_file, 'w') as f:
            json.dump(leaderboard, f, indent=2)
    
    def get_leaderboard(self, benchmark: Optional[str] = None) -> Dict:
        """Get leaderboard data, optionally filtered by benchmark"""
        leaderboard_file = os.path.join(self.storage_dir, "leaderboard.json")
        
        if not os.path.exists(leaderboard_file):
            return {}
        
        with open(leaderboard_file, 'r') as f:
            leaderboard = json.load(f)
        
        if benchmark:
            filtered = {}
            for model, benchmarks in leaderboard.items():
                if benchmark in benchmarks:
                    filtered[model] = {benchmark: benchmarks[benchmark]}
            return filtered
        
        return leaderboard
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate a markdown report of the leaderboard"""
        leaderboard = self.get_leaderboard()
        
        if not leaderboard:
            logger.warning("No leaderboard data available")
            return
        
        report = ["# Model Evaluation Leaderboard\n"]
        
        # Get all benchmarks
        all_benchmarks = set()
        for model_benchmarks in leaderboard.values():
            all_benchmarks.update(model_benchmarks.keys())
        
        all_benchmarks = sorted(all_benchmarks)
        
        # Create table header
        header = ["Model"] + all_benchmarks
        table = [header]
        
        # Add model rows
        for model, benchmarks in sorted(leaderboard.items()):
            row = [model]
            for benchmark in all_benchmarks:
                if benchmark in benchmarks:
                    metrics = benchmarks[benchmark]["metrics"]
                    # Show primary metric (accuracy or first metric)
                    if "accuracy" in metrics:
                        row.append(f"{metrics['accuracy']:.3f}")
                    elif metrics:
                        first_metric = list(metrics.values())[0]
                        row.append(f"{first_metric:.3f}" if isinstance(first_metric, float) else str(first_metric))
                    else:
                        row.append("N/A")
                else:
                    row.append("-")
            table.append(row)
        
        # Convert table to markdown
        col_widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
        
        for i, row in enumerate(table):
            line = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
            report.append(line)
            if i == 0:  # Header separator
                report.append(" | ".join("-" * col_widths[j] for j in range(len(header))))
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Leaderboard report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text

class EvaluationFramework:
    """Main evaluation framework orchestrator"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.registry = BenchmarkRegistry()
        self.leaderboard = Leaderboard(cache_dir)
        self.cache_dir = cache_dir or os.path.join(project_root, "cache", "evaluations")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_model(self, model_path: str, **kwargs):
        """Load model and tokenizer"""
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Try to load as a Hugging Face model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Determine model class based on config
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                **kwargs
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run_benchmark(self, benchmark_name: str, model, tokenizer, 
                     dataset_split: str = "test", **kwargs) -> BenchmarkResult:
        """Run a specific benchmark"""
        benchmark = self.registry.get(benchmark_name)
        if not benchmark:
            raise ValueError(f"Benchmark '{benchmark_name}' not found. Available: {self.registry.list_available()}")
        
        logger.info(f"Running benchmark: {benchmark_name}")
        
        # Load dataset
        dataset = benchmark.load_dataset(split=dataset_split)
        
        # Run evaluation
        result = benchmark.evaluate(model, tokenizer, dataset, **kwargs)
        
        # Add to leaderboard
        self.leaderboard.add_result(result)
        
        return result
    
    def run_all_benchmarks(self, model, tokenizer, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run all available benchmarks"""
        results = {}
        
        for benchmark_name in self.registry.list_available():
            try:
                result = self.run_benchmark(benchmark_name, model, tokenizer, **kwargs)
                results[benchmark_name] = result
                logger.info(f"Completed {benchmark_name}: {result.metrics}")
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name}: {e}")
        
        return results
    
    def compare_models(self, model_paths: List[str], benchmark_names: Optional[List[str]] = None):
        """Compare multiple models on specified benchmarks"""
        if benchmark_names is None:
            benchmark_names = self.registry.list_available()
        
        comparison_results = {}
        
        for model_path in model_paths:
            try:
                model, tokenizer = self.load_model(model_path)
                model_name = getattr(model, 'name_or_path', model_path)
                
                model_results = {}
                for benchmark_name in benchmark_names:
                    try:
                        result = self.run_benchmark(benchmark_name, model, tokenizer)
                        model_results[benchmark_name] = result.metrics
                    except Exception as e:
                        logger.error(f"Failed to run {benchmark_name} for {model_name}: {e}")
                        model_results[benchmark_name] = {"error": str(e)}
                
                comparison_results[model_name] = model_results
                
                # Clean up
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_path}: {e}")
        
        return comparison_results

def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Evaluation Framework")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["MMMU"], help="Benchmarks to run")
    parser.add_argument("--all", action="store_true", help="Run all available benchmarks")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for results")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--leaderboard", action="store_true", help="Generate leaderboard report")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple models")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks")
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = EvaluationFramework(cache_dir=args.cache_dir)
    
    # List benchmarks if requested
    if args.list_benchmarks:
        print("Available benchmarks:")
        for benchmark in framework.registry.list_available():
            print(f"  - {benchmark}")
        return
    
    # Compare models if requested
    if args.compare:
        results = framework.compare_models(args.compare, args.benchmarks)
        print(json.dumps(results, indent=2))
        return
    
    # Generate leaderboard if requested
    if args.leaderboard:
        framework.leaderboard.generate_report(args.output)
        return
    
    # Load model
    model, tokenizer = framework.load_model(args.model)
    
    # Run benchmarks
    if args.all:
        results = framework.run_all_benchmarks(model, tokenizer, 
                                              batch_size=args.batch_size,
                                              split=args.split)
    else:
        results = {}
        for benchmark_name in args.benchmarks:
            result = framework.run_benchmark(benchmark_name, model, tokenizer,
                                           batch_size=args.batch_size,
                                           split=args.split)
            results[benchmark_name] = result
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print("\nEvaluation Results:")
        for benchmark_name, result in results.items():
            print(f"\n{benchmark_name}:")
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

if __name__ == "__main__":
    main()