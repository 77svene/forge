"""
Unified Multi-Modal Evaluation Framework for forge
Replaces basic BLEU/ROUGE evaluation with comprehensive benchmark suite
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import importlib
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path.home() / ".cache" / "forge" / "eval"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation"""
    name: str
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "test"
    max_samples: Optional[int] = None
    metrics: List[str] = field(default_factory=list)
    task_type: str = "text"  # text, vision, multimodal
    few_shot: int = 0
    cache_dataset: bool = True
    preprocessing_fn: Optional[str] = None
    evaluation_fn: Optional[str] = None
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

@dataclass 
class EvaluationResult:
    """Container for evaluation results"""
    benchmark: str
    model_name: str
    metrics: Dict[str, float]
    samples_evaluated: int
    timestamp: str
    config: BenchmarkConfig
    raw_predictions: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result
    
    def save(self, path: Path):
        """Save results to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")

class Benchmark(ABC):
    """Abstract base class for all benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset = None
        self.metrics = {}
        self._load_metrics()
        
    def _load_metrics(self):
        """Load evaluation metrics"""
        for metric_name in self.config.metrics:
            try:
                self.metrics[metric_name] = evaluate.load(metric_name)
            except Exception as e:
                logger.warning(f"Could not load metric {metric_name}: {e}")
    
    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and preprocess dataset"""
        pass
    
    @abstractmethod
    def evaluate(self, model, tokenizer, **kwargs) -> EvaluationResult:
        """Run evaluation on the benchmark"""
        pass
    
    def preprocess(self, examples: Dict) -> Dict:
        """Default preprocessing - can be overridden"""
        return examples
    
    def get_cache_path(self) -> Path:
        """Get cache path for preprocessed dataset"""
        cache_key = self.config.get_cache_key()
        return CACHE_DIR / f"{self.config.name}_{cache_key}.pt"
    
    def cache_dataset(self, dataset: Dataset):
        """Cache preprocessed dataset"""
        cache_path = self.get_cache_path()
        torch.save(dataset, cache_path)
        logger.info(f"Cached dataset to {cache_path}")
    
    def load_cached_dataset(self) -> Optional[Dataset]:
        """Load cached dataset if available"""
        cache_path = self.get_cache_path()
        if cache_path.exists() and self.config.cache_dataset:
            try:
                dataset = torch.load(cache_path)
                logger.info(f"Loaded cached dataset from {cache_path}")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

class MMLUBenchmark(Benchmark):
    """MMLU (Massive Multitask Language Understanding) benchmark"""
    
    def __init__(self, config: BenchmarkConfig):
        if not config.metrics:
            config.metrics = ["accuracy"]
        if not config.dataset_name:
            config.dataset_name = "cais/mmlu"
        super().__init__(config)
        self.subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]
    
    def load_dataset(self) -> Dataset:
        """Load MMLU dataset with caching"""
        cached = self.load_cached_dataset()
        if cached is not None:
            return cached
        
        logger.info(f"Loading MMLU dataset: {self.config.dataset_name}")
        
        # Load all subjects or specific subject
        if self.config.dataset_config:
            datasets = []
            for subject in self.subjects:
                try:
                    ds = load_dataset(
                        self.config.dataset_name,
                        subject,
                        split=self.config.split,
                        trust_remote_code=True
                    )
                    ds = ds.add_column("subject", [subject] * len(ds))
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load subject {subject}: {e}")
            
            if datasets:
                from datasets import concatenate_datasets
                dataset = concatenate_datasets(datasets)
            else:
                raise ValueError("No subjects loaded successfully")
        else:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split,
                trust_remote_code=True
            )
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        # Preprocess
        dataset = self.preprocess(dataset)
        
        # Cache
        self.cache_dataset(dataset)
        
        return dataset
    
    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess MMLU dataset"""
        def format_example(example):
            """Format MMLU example into prompt"""
            choices = [example['A'], example['B'], example['C'], example['D']]
            prompt = f"Question: {example['question']}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"
            return {
                "prompt": prompt,
                "choices": choices,
                "answer": example['answer'],
                "subject": example.get('subject', 'unknown')
            }
        
        return dataset.map(format_example)
    
    def evaluate(self, model, tokenizer, **kwargs) -> EvaluationResult:
        """Evaluate model on MMLU"""
        dataset = self.load_dataset()
        
        predictions = []
        references = []
        subjects = []
        
        logger.info(f"Evaluating on {len(dataset)} MMLU samples")
        
        for example in tqdm(dataset, desc="MMLU Evaluation"):
            prompt = example['prompt']
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract answer
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated = generated.strip().upper()
            
            # Extract first letter (A, B, C, D)
            pred = None
            for char in generated:
                if char in ['A', 'B', 'C', 'D']:
                    pred = char
                    break
            
            if pred is None:
                pred = 'A'  # Default fallback
            
            predictions.append(pred)
            references.append(example['answer'])
            subjects.append(example.get('subject', 'unknown'))
        
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        metrics['accuracy'] = correct / len(predictions)
        
        # Per-subject accuracy
        subject_correct = defaultdict(int)
        subject_total = defaultdict(int)
        for p, r, s in zip(predictions, references, subjects):
            subject_total[s] += 1
            if p == r:
                subject_correct[s] += 1
        
        for subject in subject_total:
            metrics[f'accuracy_{subject}'] = subject_correct[subject] / subject_total[subject]
        
        # Create result
        result = EvaluationResult(
            benchmark="MMLU",
            model_name=kwargs.get('model_name', 'unknown'),
            metrics=metrics,
            samples_evaluated=len(predictions),
            timestamp=datetime.now().isoformat(),
            config=self.config,
            raw_predictions=[{"pred": p, "ref": r, "subject": s} for p, r, s in zip(predictions, references, subjects)]
        )
        
        return result

class MMMUBenchmark(Benchmark):
    """MMMU (Massive Multi-discipline Multimodal Understanding) benchmark"""
    
    def __init__(self, config: BenchmarkConfig):
        if not config.metrics:
            config.metrics = ["accuracy"]
        if not config.dataset_name:
            config.dataset_name = "MMMU/MMMU"
        config.task_type = "multimodal"
        super().__init__(config)
    
    def load_dataset(self) -> Dataset:
        """Load MMMU dataset"""
        cached = self.load_cached_dataset()
        if cached is not None:
            return cached
        
        logger.info(f"Loading MMMU dataset: {self.config.dataset_name}")
        
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
            trust_remote_code=True
        )
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        dataset = self.preprocess(dataset)
        self.cache_dataset(dataset)
        
        return dataset
    
    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess MMMU dataset"""
        def format_multimodal_example(example):
            """Format multimodal example"""
            # MMMU has images and text questions
            prompt = f"Question: {example['question']}\n"
            
            if 'options' in example:
                options = example['options']
                if isinstance(options, list):
                    for i, option in enumerate(options):
                        prompt += f"{chr(65+i)}. {option}\n"
                else:
                    # Handle string format
                    options = options.split('\n')
                    for i, option in enumerate(options):
                        if option.strip():
                            prompt += f"{chr(65+i)}. {option.strip()}\n"
            
            prompt += "Answer:"
            
            return {
                "prompt": prompt,
                "image": example.get('image'),
                "answer": example['answer'],
                "subject": example.get('subject', 'unknown')
            }
        
        return dataset.map(format_multimodal_example)
    
    def evaluate(self, model, tokenizer, **kwargs) -> EvaluationResult:
        """Evaluate model on MMMU"""
        # Note: This is a simplified version. Full MMMU evaluation requires multimodal models
        dataset = self.load_dataset()
        
        logger.warning("MMMU evaluation requires multimodal model support. Using text-only evaluation.")
        
        predictions = []
        references = []
        
        for example in tqdm(dataset, desc="MMMU Evaluation"):
            prompt = example['prompt']
            
            # For now, use text-only evaluation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated = generated.strip().upper()
            
            pred = None
            for char in generated:
                if char in ['A', 'B', 'C', 'D']:
                    pred = char
                    break
            
            if pred is None:
                pred = 'A'
            
            predictions.append(pred)
            references.append(example['answer'])
        
        # Calculate accuracy
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        metrics = {'accuracy': correct / len(predictions)}
        
        result = EvaluationResult(
            benchmark="MMMU",
            model_name=kwargs.get('model_name', 'unknown'),
            metrics=metrics,
            samples_evaluated=len(predictions),
            timestamp=datetime.now().isoformat(),
            config=self.config,
            raw_predictions=[{"pred": p, "ref": r} for p, r in zip(predictions, references)]
        )
        
        return result

class HumanEvalBenchmark(Benchmark):
    """HumanEval benchmark for code generation"""
    
    def __init__(self, config: BenchmarkConfig):
        if not config.metrics:
            config.metrics = ["pass@1", "pass@10", "pass@100"]
        if not config.dataset_name:
            config.dataset_name = "openai_humaneval"
        super().__init__(config)
    
    def load_dataset(self) -> Dataset:
        """Load HumanEval dataset"""
        cached = self.load_cached_dataset()
        if cached is not None:
            return cached
        
        logger.info(f"Loading HumanEval dataset: {self.config.dataset_name}")
        
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.split,
            trust_remote_code=True
        )
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        dataset = self.preprocess(dataset)
        self.cache_dataset(dataset)
        
        return dataset
    
    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess HumanEval dataset"""
        def format_code_example(example):
            """Format code generation example"""
            prompt = example['prompt']
            
            # Add instruction for code completion
            full_prompt = f"Complete the following Python function:\n\n{prompt}"
            
            return {
                "prompt": full_prompt,
                "canonical_solution": example['canonical_solution'],
                "test": example['test'],
                "entry_point": example['entry_point'],
                "task_id": example['task_id']
            }
        
        return dataset.map(format_code_example)
    
    def evaluate(self, model, tokenizer, num_samples_per_task: int = 1, **kwargs) -> EvaluationResult:
        """Evaluate model on HumanEval"""
        dataset = self.load_dataset()
        
        logger.info(f"Evaluating on {len(dataset)} HumanEval tasks with {num_samples_per_task} samples each")
        
        all_results = []
        
        for example in tqdm(dataset, desc="HumanEval Evaluation"):
            prompt = example['prompt']
            task_id = example['task_id']
            
            # Generate multiple samples for pass@k calculation
            samples = []
            for _ in range(num_samples_per_task):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                samples.append(generated)
            
            # Evaluate samples (simplified - in practice you'd run tests)
            # This is a placeholder - real evaluation requires executing code
            all_results.append({
                "task_id": task_id,
                "samples": samples,
                "prompt": prompt
            })
        
        # Calculate pass@k (simplified - requires actual code execution)
        metrics = {}
        for k in [1, 10, 100]:
            if num_samples_per_task >= k:
                # Placeholder metric - real implementation would execute tests
                metrics[f"pass@{k}"] = 0.0  # Would be calculated from test results
        
        result = EvaluationResult(
            benchmark="HumanEval",
            model_name=kwargs.get('model_name', 'unknown'),
            metrics=metrics,
            samples_evaluated=len(dataset) * num_samples_per_task,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            raw_predictions=all_results
        )
        
        return result

class BenchmarkRegistry:
    """Registry for benchmark plugins"""
    
    _benchmarks = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a benchmark class"""
        def wrapper(benchmark_cls):
            cls._benchmarks[name] = benchmark_cls
            return benchmark_cls
        return wrapper
    
    @classmethod
    def get(cls, name: str) -> Optional[Benchmark]:
        """Get benchmark class by name"""
        return cls._benchmarks.get(name)
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmarks"""
        return list(cls._benchmarks.keys())

# Register built-in benchmarks
BenchmarkRegistry.register("mmlu")(MMLUBenchmark)
BenchmarkRegistry.register("mmmu")(MMMUBenchmark)
BenchmarkRegistry.register("humaneval")(HumanEvalBenchmark)

class EvaluationFramework:
    """Main evaluation framework"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, results_dir: Path = RESULTS_DIR):
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )
        
        model.eval()
        
        return model, tokenizer
    
    def run_benchmark(self, 
                     benchmark_name: str, 
                     model_path: str,
                     config: Optional[BenchmarkConfig] = None,
                     **kwargs) -> EvaluationResult:
        """Run a single benchmark"""
        
        # Get benchmark class
        benchmark_cls = BenchmarkRegistry.get(benchmark_name)
        if benchmark_cls is None:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {BenchmarkRegistry.list_benchmarks()}")
        
        # Create default config if not provided
        if config is None:
            config = BenchmarkConfig(
                name=benchmark_name,
                dataset_name="",
                metrics=[]
            )
        
        # Initialize benchmark
        benchmark = benchmark_cls(config)
        
        # Load model
        model, tokenizer = self.load_model(model_path, **kwargs)
        
        # Run evaluation
        result = benchmark.evaluate(
            model, 
            tokenizer, 
            model_name=model_path,
            **kwargs
        )
        
        # Save result
        self.save_result(result)
        
        return result
    
    def run_multiple_benchmarks(self,
                               benchmark_names: List[str],
                               model_path: str,
                               configs: Optional[Dict[str, BenchmarkConfig]] = None,
                               **kwargs) -> Dict[str, EvaluationResult]:
        """Run multiple benchmarks"""
        
        results = {}
        
        for benchmark_name in benchmark_names:
            try:
                config = configs.get(benchmark_name) if configs else None
                result = self.run_benchmark(benchmark_name, model_path, config, **kwargs)
                results[benchmark_name] = result
                logger.info(f"Completed {benchmark_name}: {result.metrics}")
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name}: {e}")
                results[benchmark_name] = None
        
        return results
    
    def save_result(self, result: EvaluationResult):
        """Save evaluation result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark}_{result.model_name.split('/')[-1]}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        result.save(filepath)
    
    def load_result(self, filepath: Path) -> EvaluationResult:
        """Load evaluation result from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = BenchmarkConfig(**data['config'])
        
        return EvaluationResult(
            benchmark=data['benchmark'],
            model_name=data['model_name'],
            metrics=data['metrics'],
            samples_evaluated=data['samples_evaluated'],
            timestamp=data['timestamp'],
            config=config,
            raw_predictions=data.get('raw_predictions')
        )
    
    def generate_leaderboard(self, 
                           benchmark_names: List[str],
                           model_paths: List[str],
                           output_path: Optional[Path] = None) -> Dict:
        """Generate leaderboard comparing multiple models"""
        
        leaderboard = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": benchmark_names,
            "models": {},
            "rankings": {}
        }
        
        all_results = {}
        
        # Run benchmarks for all models
        for model_path in model_paths:
            model_name = model_path.split('/')[-1]
            logger.info(f"Evaluating model: {model_name}")
            
            results = self.run_multiple_benchmarks(benchmark_names, model_path)
            all_results[model_name] = results
            
            # Store in leaderboard
            leaderboard["models"][model_name] = {
                "path": model_path,
                "results": {bn: r.metrics if r else None for bn, r in results.items()}
            }
        
        # Calculate rankings for each benchmark
        for benchmark_name in benchmark_names:
            scores = []
            for model_name, results in all_results.items():
                if results.get(benchmark_name):
                    # Use primary metric (first metric)
                    metrics = results[benchmark_name].metrics
                    primary_metric = list(metrics.keys())[0]
                    scores.append((model_name, metrics[primary_metric]))
            
            # Sort by score (descending for accuracy, etc.)
            scores.sort(key=lambda x: x[1], reverse=True)
            leaderboard["rankings"][benchmark_name] = [
                {"rank": i+1, "model": model, "score": score}
                for i, (model, score) in enumerate(scores)
            ]
        
        # Save leaderboard
        if output_path is None:
            output_path = self.results_dir / f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(leaderboard, f, indent=2)
        
        logger.info(f"Leaderboard saved to {output_path}")
        
        return leaderboard

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Evaluation Framework")
    
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["mmlu"],
                       help="Benchmarks to run")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="List available benchmarks")
    parser.add_argument("--generate-leaderboard", action="store_true",
                       help="Generate leaderboard for multiple models")
    parser.add_argument("--models", type=str, nargs="+",
                       help="Multiple models for leaderboard generation")
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = EvaluationFramework(results_dir=Path(args.output_dir))
    
    # List benchmarks if requested
    if args.list_benchmarks:
        print("Available benchmarks:")
        for benchmark in BenchmarkRegistry.list_benchmarks():
            print(f"  - {benchmark}")
        return
    
    # Generate leaderboard if requested
    if args.generate_leaderboard:
        if not args.models:
            parser.error("--models required for leaderboard generation")
        
        leaderboard = framework.generate_leaderboard(
            benchmark_names=args.benchmarks,
            model_paths=args.models
        )
        
        # Print summary
        print("\n=== LEADERBOARD SUMMARY ===")
        for benchmark in args.benchmarks:
            print(f"\n{benchmark.upper()}:")
            for entry in leaderboard["rankings"].get(benchmark, [])[:5]:
                print(f"  {entry['rank']}. {entry['model']}: {entry['score']:.4f}")
        return
    
    # Run benchmarks
    config = BenchmarkConfig(
        name="custom",
        dataset_name="",
        max_samples=args.max_samples
    )
    
    results = framework.run_multiple_benchmarks(
        benchmark_names=args.benchmarks,
        model_path=args.model
    )
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    for benchmark_name, result in results.items():
        if result:
            print(f"\n{benchmark_name}:")
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{benchmark_name}: FAILED")

if __name__ == "__main__":
    main()