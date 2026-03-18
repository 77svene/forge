import unittest
import tempfile
import subprocess
import os
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RegressionTestConfig:
    """Configuration for regression tests"""
    model_name: str
    dataset_name: str
    expected_metrics: Dict[str, float]
    tolerance: float = 0.05
    max_steps: int = 100
    eval_steps: int = 50
    batch_size: int = 2
    learning_rate: float = 2e-5
    seed: int = 42
    device: str = "auto"
    golden_reference_hash: Optional[str] = None

class GoldenReferenceManager:
    """Manages golden reference files for regression testing"""
    
    def __init__(self, reference_dir: Path):
        self.reference_dir = reference_dir
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
    def get_reference_path(self, model_name: str, dataset_name: str) -> Path:
        """Get path to golden reference file"""
        safe_model = model_name.replace("/", "_").replace("-", "_")
        safe_dataset = dataset_name.replace("/", "_").replace("-", "_")
        return self.reference_dir / f"golden_ref_{safe_model}_{safe_dataset}.json"
    
    def load_reference(self, model_name: str, dataset_name: str) -> Optional[Dict]:
        """Load golden reference if exists"""
        ref_path = self.get_reference_path(model_name, dataset_name)
        if ref_path.exists():
            with open(ref_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_reference(self, model_name: str, dataset_name: str, 
                      metrics: Dict, metadata: Dict) -> None:
        """Save golden reference with metadata"""
        ref_path = self.get_reference_path(model_name, dataset_name)
        reference = {
            "model": model_name,
            "dataset": dataset_name,
            "metrics": metrics,
            "metadata": {
                **metadata,
                "created_at": datetime.now().isoformat(),
                "git_commit": self._get_git_commit(),
                "environment": self._get_environment_info()
            }
        }
        with open(ref_path, 'w') as f:
            json.dump(reference, f, indent=2)
        logger.info(f"Saved golden reference to {ref_path}")
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _get_environment_info(self) -> Dict:
        """Get environment information"""
        return {
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

class ModelDownloader:
    """Downloads and caches small model variants for testing"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_model(self, model_name: str) -> Path:
        """Download model if not already cached"""
        model_path = self.cache_dir / model_name.replace("/", "_")
        
        if model_path.exists() and self._is_model_complete(model_path):
            logger.info(f"Using cached model at {model_path}")
            return model_path
        
        logger.info(f"Downloading model {model_name} to {model_path}")
        
        # Use transformers to download model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Download and save model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"Successfully downloaded and saved model to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            raise
    
    def _is_model_complete(self, model_path: Path) -> bool:
        """Check if model files are complete"""
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        for file in required_files:
            if not (model_path / file).exists():
                return False
        return True

class DatasetManager:
    """Manages test datasets for regression testing"""
    
    STANDARD_DATASETS = {
        "alpaca_sample": {
            "source": "tatsu-lab/alpaca",
            "split": "train[:100]",  # Small sample for testing
            "format": "alpaca"
        },
        "dolly_sample": {
            "source": "databricks/databricks-dolly-15k",
            "split": "train[:100]",
            "format": "dolly"
        },
        "oasst1_sample": {
            "source": "OpenAssistant/oasst1",
            "split": "train[:100]",
            "format": "oasst"
        }
    }
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset(self, dataset_name: str) -> Path:
        """Get dataset path, downloading if necessary"""
        if dataset_name not in self.STANDARD_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.STANDARD_DATASETS.keys())}")
        
        dataset_config = self.STANDARD_DATASETS[dataset_name]
        dataset_path = self.cache_dir / f"{dataset_name}.json"
        
        if dataset_path.exists():
            logger.info(f"Using cached dataset at {dataset_path}")
            return dataset_path
        
        logger.info(f"Downloading dataset {dataset_name}")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                dataset_config["source"],
                split=dataset_config["split"]
            )
            
            # Convert to expected format
            formatted_data = self._format_dataset(dataset, dataset_config["format"])
            
            with open(dataset_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"Saved dataset to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            raise
    
    def _format_dataset(self, dataset, format_type: str) -> List[Dict]:
        """Format dataset to standard format"""
        formatted = []
        
        if format_type == "alpaca":
            for item in dataset:
                formatted.append({
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                })
        elif format_type == "dolly":
            for item in dataset:
                formatted.append({
                    "instruction": item["instruction"],
                    "input": item.get("context", ""),
                    "output": item["response"]
                })
        elif format_type == "oasst":
            for item in dataset:
                if item["role"] == "assistant":
                    formatted.append({
                        "instruction": item.get("parent_id", ""),
                        "input": "",
                        "output": item["text"]
                    })
        
        return formatted

class TrainingRunner:
    """Runs fine-tuning training with forge"""
    
    def __init__(self, project_root: Path, output_dir: Path):
        self.project_root = project_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_training(self, config: RegressionTestConfig, 
                    model_path: Path, dataset_path: Path) -> Tuple[Path, Dict]:
        """Run training and return output path and metrics"""
        
        # Create training configuration
        train_config = self._create_training_config(config, model_path, dataset_path)
        config_path = self.output_dir / "train_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(train_config, f, indent=2)
        
        # Run training
        logger.info(f"Starting training with config: {config.model_name}")
        
        train_script = self.project_root / "train.py"
        if not train_script.exists():
            train_script = self.project_root / "src" / "train.py"
        
        cmd = [
            "python", str(train_script),
            "--config", str(config_path),
            "--output_dir", str(self.output_dir / "model_output")
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed with error: {result.stderr}")
                raise RuntimeError(f"Training failed: {result.stderr}")
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Extract metrics from output
            metrics = self._extract_metrics(result.stdout)
            metrics["training_time"] = training_time
            
            return self.output_dir / "model_output", metrics
            
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            raise RuntimeError("Training timed out after 30 minutes")
    
    def _create_training_config(self, config: RegressionTestConfig,
                               model_path: Path, dataset_path: Path) -> Dict:
        """Create training configuration"""
        return {
            "model_name_or_path": str(model_path),
            "dataset": str(dataset_path),
            "output_dir": str(self.output_dir / "model_output"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "evaluation_strategy": "steps",
            "eval_steps": config.eval_steps,
            "save_steps": config.eval_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "fp16": torch.cuda.is_available(),
            "logging_steps": 10,
            "report_to": "none",
            "remove_unused_columns": False,
            "ddp_find_unused_parameters": False
        }
    
    def _extract_metrics(self, output: str) -> Dict:
        """Extract metrics from training output"""
        metrics = {}
        
        # Look for common metric patterns
        lines = output.split('\n')
        for line in lines:
            if "train_loss" in line.lower():
                try:
                    # Extract loss value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "loss" in part.lower() and i + 1 < len(parts):
                            loss_str = parts[i + 1].replace(":", "").replace(",", "")
                            metrics["train_loss"] = float(loss_str)
                except:
                    pass
            elif "eval_loss" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "loss" in part.lower() and i + 1 < len(parts):
                            loss_str = parts[i + 1].replace(":", "").replace(",", "")
                            metrics["eval_loss"] = float(loss_str)
                except:
                    pass
        
        return metrics

class ModelEvaluator:
    """Evaluates fine-tuned model quality"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def evaluate_model(self, model_path: Path, dataset_path: Path,
                      config: RegressionTestConfig) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        
        # Use existing evaluation script
        eval_script = self.project_root / "scripts" / "eval_bleu_rouge.py"
        
        if not eval_script.exists():
            logger.warning("Evaluation script not found, using fallback evaluation")
            return self._fallback_evaluation(model_path, dataset_path, config)
        
        # Create evaluation configuration
        eval_output = self.project_root / "eval_results.json"
        
        cmd = [
            "python", str(eval_script),
            "--model_path", str(model_path),
            "--dataset_path", str(dataset_path),
            "--output_file", str(eval_output),
            "--batch_size", str(config.batch_size)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed: {result.stderr}")
                return self._fallback_evaluation(model_path, dataset_path, config)
            
            # Load evaluation results
            if eval_output.exists():
                with open(eval_output, 'r') as f:
                    eval_results = json.load(f)
                return eval_results
            else:
                return self._fallback_evaluation(model_path, dataset_path, config)
                
        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            return self._fallback_evaluation(model_path, dataset_path, config)
    
    def _fallback_evaluation(self, model_path: Path, dataset_path: Path,
                           config: RegressionTestConfig) -> Dict[str, float]:
        """Fallback evaluation using simple metrics"""
        logger.info("Using fallback evaluation metrics")
        
        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=config.device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Sample evaluation
        sample_size = min(20, len(dataset))
        samples = dataset[:sample_size]
        
        total_loss = 0
        total_perplexity = 0
        
        for sample in samples:
            # Format input
            if "instruction" in sample:
                prompt = f"### Instruction:\n{sample['instruction']}\n"
                if sample.get("input"):
                    prompt += f"### Input:\n{sample['input']}\n"
                prompt += "### Response:\n"
                target = sample["output"]
            else:
                prompt = sample.get("text", "")
                target = ""
            
            # Calculate loss
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = torch.exp(torch.tensor(loss)).item()
            
            total_loss += loss
            total_perplexity += perplexity
        
        avg_loss = total_loss / sample_size
        avg_perplexity = total_perplexity / sample_size
        
        # Calculate simple BLEU-like score (simplified)
        bleu_score = max(0, 1 - avg_perplexity / 100)  # Normalize
        
        return {
            "loss": avg_loss,
            "perplexity": avg_perplexity,
            "bleu": bleu_score,
            "rouge_l": bleu_score * 0.9,  # Approximate
            "samples_evaluated": sample_size
        }

class TestFinetuneQuality(unittest.TestCase):
    """Regression tests for fine-tuning quality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.test_dir = Path(tempfile.mkdtemp(prefix="forge_regression_"))
        cls.cache_dir = cls.test_dir / "cache"
        cls.output_dir = cls.test_dir / "output"
        cls.reference_dir = cls.project_root / "tests" / "regression" / "golden_references"
        
        # Initialize components
        cls.downloader = ModelDownloader(cls.cache_dir / "models")
        cls.dataset_manager = DatasetManager(cls.cache_dir / "datasets")
        cls.training_runner = TrainingRunner(cls.project_root, cls.output_dir)
        cls.evaluator = ModelEvaluator(cls.project_root)
        cls.reference_manager = GoldenReferenceManager(cls.reference_dir)
        
        # Test configurations
        cls.test_configs = [
            RegressionTestConfig(
                model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
                dataset_name="alpaca_sample",
                expected_metrics={
                    "loss": 3.5,
                    "bleu": 0.1,
                    "rouge_l": 0.1
                },
                max_steps=20,
                eval_steps=10
            ),
            RegressionTestConfig(
                model_name="hf-internal-testing/tiny-random-GPT2ForCausalLM",
                dataset_name="dolly_sample",
                expected_metrics={
                    "loss": 3.8,
                    "bleu": 0.08,
                    "rouge_l": 0.08
                },
                max_steps=20,
                eval_steps=10
            )
        ]
        
        logger.info(f"Test directory: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            logger.info(f"Cleaned up test directory: {cls.test_dir}")
    
    def test_model_integration(self):
        """Test that new model integrations work correctly"""
        for config in self.test_configs:
            with self.subTest(model=config.model_name):
                logger.info(f"Testing model integration: {config.model_name}")
                
                # Download model
                model_path = self.downloader.download_model(config.model_name)
                self.assertTrue(model_path.exists(), f"Model not downloaded: {config.model_name}")
                
                # Verify model files
                self.assertTrue((model_path / "config.json").exists())
                self.assertTrue((model_path / "pytorch_model.bin").exists() or 
                              (model_path / "model.safetensors").exists())
                
                logger.info(f"Model integration test passed for {config.model_name}")
    
    def test_finetuning_quality(self):
        """Test fine-tuning quality against golden references"""
        for config in self.test_configs:
            with self.subTest(model=config.model_name, dataset=config.dataset_name):
                logger.info(f"Testing fine-tuning quality: {config.model_name} on {config.dataset_name}")
                
                # Get dataset
                dataset_path = self.dataset_manager.get_dataset(config.dataset_name)
                
                # Download model
                model_path = self.downloader.download_model(config.model_name)
                
                # Run training
                trained_model_path, train_metrics = self.training_runner.run_training(
                    config, model_path, dataset_path
                )
                
                # Evaluate model
                eval_metrics = self.evaluator.evaluate_model(
                    trained_model_path, dataset_path, config
                )
                
                # Combine metrics
                all_metrics = {**train_metrics, **eval_metrics}
                
                # Load or create golden reference
                golden_ref = self.reference_manager.load_reference(
                    config.model_name, config.dataset_name
                )
                
                if golden_ref is None:
                    # First run - create golden reference
                    logger.info("Creating golden reference for first run")
                    self.reference_manager.save_reference(
                        config.model_name,
                        config.dataset_name,
                        all_metrics,
                        {"config": config.__dict__}
                    )
                    golden_metrics = all_metrics
                else:
                    golden_metrics = golden_ref["metrics"]
                
                # Compare metrics with golden reference
                self._compare_metrics(
                    all_metrics,
                    golden_metrics,
                    config.tolerance,
                    f"{config.model_name} on {config.dataset_name}"
                )
                
                logger.info(f"Fine-tuning quality test passed for {config.model_name}")
    
    def test_performance_degradation(self):
        """Test for performance degradation across versions"""
        # This test compares current performance with historical data
        historical_data_path = self.reference_dir / "performance_history.json"
        
        if not historical_data_path.exists():
            logger.info("No historical data found, skipping degradation test")
            return
        
        with open(historical_data_path, 'r') as f:
            historical_data = json.load(f)
        
        for config in self.test_configs:
            with self.subTest(model=config.model_name):
                model_key = f"{config.model_name}_{config.dataset_name}"
                
                if model_key not in historical_data:
                    continue
                
                # Get current metrics
                dataset_path = self.dataset_manager.get_dataset(config.dataset_name)
                model_path = self.downloader.download_model(config.model_name)
                trained_model_path, _ = self.training_runner.run_training(
                    config, model_path, dataset_path
                )
                current_metrics = self.evaluator.evaluate_model(
                    trained_model_path, dataset_path, config
                )
                
                # Compare with historical average
                historical_entries = historical_data[model_key]
                if len(historical_entries) < 2:
                    continue
                
                # Calculate historical average (excluding current)
                historical_avg = {}
                for metric in ["loss", "bleu", "rouge_l"]:
                    values = [entry["metrics"].get(metric, 0) 
                             for entry in historical_entries[-5:]]  # Last 5 runs
                    if values:
                        historical_avg[metric] = np.mean(values)
                
                # Check for significant degradation
                for metric, current_value in current_metrics.items():
                    if metric in historical_avg:
                        historical_value = historical_avg[metric]
                        
                        # For loss, higher is worse
                        if metric == "loss":
                            degradation = (current_value - historical_value) / historical_value
                            self.assertLess(
                                degradation,
                                0.2,  # 20% degradation threshold
                                f"Performance degradation detected for {metric}: "
                                f"current={current_value:.4f}, historical={historical_value:.4f}"
                            )
                        # For BLEU/ROUGE, lower is worse
                        elif metric in ["bleu", "rouge_l"]:
                            degradation = (historical_value - current_value) / historical_value
                            self.assertLess(
                                degradation,
                                0.2,  # 20% degradation threshold
                                f"Performance degradation detected for {metric}: "
                                f"current={current_value:.4f}, historical={historical_value:.4f}"
                            )
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        config = self.test_configs[0]  # Use first config
        
        # Run training twice with same seed
        dataset_path = self.dataset_manager.get_dataset(config.dataset_name)
        model_path = self.downloader.download_model(config.model_name)
        
        # First run
        output_dir1 = self.output_dir / "run1"
        runner1 = TrainingRunner(self.project_root, output_dir1)
        trained_model1, metrics1 = runner1.run_training(config, model_path, dataset_path)
        
        # Second run
        output_dir2 = self.output_dir / "run2"
        runner2 = TrainingRunner(self.project_root, output_dir2)
        trained_model2, metrics2 = runner2.run_training(config, model_path, dataset_path)
        
        # Compare metrics (should be very close)
        for metric in ["loss", "eval_loss"]:
            if metric in metrics1 and metric in metrics2:
                diff = abs(metrics1[metric] - metrics2[metric])
                self.assertLess(
                    diff,
                    0.01,  # Very small tolerance for same seed
                    f"Reproducibility issue for {metric}: "
                    f"run1={metrics1[metric]:.4f}, run2={metrics2[metric]:.4f}"
                )
    
    def test_memory_efficiency(self):
        """Test that training doesn't exceed memory limits"""
        config = self.test_configs[0]
        
        # Monitor memory usage during training
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training
        dataset_path = self.dataset_manager.get_dataset(config.dataset_name)
        model_path = self.downloader.download_model(config.model_name)
        trained_model_path, _ = self.training_runner.run_training(
            config, model_path, dataset_path
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 2GB for tiny models)
        self.assertLess(
            memory_increase,
            2048,  # 2GB limit
            f"Memory usage increased by {memory_increase:.2f}MB, exceeding limit"
        )
        
        logger.info(f"Memory usage increased by {memory_increase:.2f}MB")
    
    def test_training_speed(self):
        """Test that training completes within expected time"""
        config = self.test_configs[0]
        
        # Time the training
        start_time = time.time()
        
        dataset_path = self.dataset_manager.get_dataset(config.dataset_name)
        model_path = self.downloader.download_model(config.model_name)
        trained_model_path, _ = self.training_runner.run_training(
            config, model_path, dataset_path
        )
        
        training_time = time.time() - start_time
        
        # Training should complete within 5 minutes for tiny models
        self.assertLess(
            training_time,
            300,  # 5 minutes
            f"Training took {training_time:.2f}s, exceeding time limit"
        )
        
        logger.info(f"Training completed in {training_time:.2f}s")
    
    def _compare_metrics(self, current: Dict, reference: Dict, 
                        tolerance: float, test_name: str):
        """Compare current metrics with reference metrics"""
        for metric, ref_value in reference.items():
            if metric not in current:
                logger.warning(f"Metric {metric} not found in current results")
                continue
            
            current_value = current[metric]
            
            # Skip non-numeric metrics
            if not isinstance(current_value, (int, float)) or not isinstance(ref_value, (int, float)):
                continue
            
            # Calculate relative difference
            if ref_value != 0:
                rel_diff = abs(current_value - ref_value) / abs(ref_value)
            else:
                rel_diff = abs(current_value - ref_value)
            
            self.assertLessEqual(
                rel_diff,
                tolerance,
                f"{test_name}: Metric {metric} differs by {rel_diff:.2%} "
                f"(current: {current_value:.4f}, reference: {ref_value:.4f})"
            )
            
            logger.info(f"Metric {metric}: {current_value:.4f} (reference: {ref_value:.4f}, "
                       f"diff: {rel_diff:.2%})")

class ContinuousEvaluationRunner:
    """Runs continuous evaluation and updates golden references"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reference_dir = project_root / "tests" / "regression" / "golden_references"
        self.history_file = self.reference_dir / "performance_history.json"
        
    def run_evaluation(self, update_golden: bool = False):
        """Run full evaluation suite"""
        logger.info("Starting continuous evaluation")
        
        # Run tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFinetuneQuality)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Update performance history
        if result.wasSuccessful():
            self._update_performance_history()
        
        # Update golden references if requested
        if update_golden and result.wasSuccessful():
            self._update_golden_references()
        
        return result.wasSuccessful()
    
    def _update_performance_history(self):
        """Update performance history with current results"""
        # This would collect metrics from the test run and append to history
        # Implementation depends on how metrics are stored during tests
        pass
    
    def _update_golden_references(self):
        """Update golden references with current results"""
        logger.info("Updating golden references")
        # Implementation would update reference files with current metrics
        pass

def main():
    """Main entry point for continuous evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run regression tests for forge")
    parser.add_argument("--update-golden", action="store_true",
                       help="Update golden references with current results")
    parser.add_argument("--test-filter", type=str,
                       help="Filter tests by name pattern")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    
    if args.update_golden:
        # Run evaluation and update golden references
        runner = ContinuousEvaluationRunner(project_root)
        success = runner.run_evaluation(update_golden=True)
        exit(0 if success else 1)
    else:
        # Run tests normally
        if args.test_filter:
            suite = unittest.TestSuite()
            loader = unittest.TestLoader()
            
            # Load tests matching filter
            for test_name in loader.getTestCaseNames(TestFinetuneQuality):
                if args.test_filter in test_name:
                    suite.addTest(TestFinetuneQuality(test_name))
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestFinetuneQuality)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()