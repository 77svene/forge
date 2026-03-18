# Copyright 2025 the forge team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import fire
import torch
import numpy as np
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Qwen2_5_VLProcessor
import evaluate
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import HfApi, create_repo

from forge.extras.constants import IGNORE_INDEX
from forge.hparams import get_train_args
from forge.model import load_model, load_tokenizer
from forge.train.callbacks import LogCallback
from forge.train.sft.trainer import CustomSeq2SeqTrainer


# ==================== Benchmark Registry ====================
class BenchmarkRegistry:
    """Registry for standardized benchmark tasks."""
    
    BENCHMARKS = {
        "mmlu": {
            "name": "MMLU",
            "description": "Massive Multitask Language Understanding",
            "dataset": "cais/mmlu",
            "split": "test",
            "metric": "accuracy",
            "categories": ["stem", "humanities", "social_sciences", "other"],
            "input_key": "question",
            "output_key": "answer",
            "choices_key": "choices",
            "max_samples": 1000,
        },
        "humaneval": {
            "name": "HumanEval",
            "description": "Code generation benchmark",
            "dataset": "openai_humaneval",
            "split": "test",
            "metric": "pass@k",
            "categories": ["coding"],
            "input_key": "prompt",
            "output_key": "canonical_solution",
            "test_key": "test",
            "max_samples": 164,
        },
        "gsm8k": {
            "name": "GSM8K",
            "description": "Grade School Math 8K",
            "dataset": "gsm8k",
            "split": "test",
            "metric": "accuracy",
            "categories": ["math"],
            "input_key": "question",
            "output_key": "answer",
            "max_samples": 1319,
        },
        "hellaswag": {
            "name": "HellaSwag",
            "description": "Commonsense reasoning",
            "dataset": "hellaswag",
            "split": "validation",
            "metric": "accuracy",
            "categories": ["commonsense"],
            "input_key": "ctx",
            "output_key": "label",
            "choices_key": "endings",
            "max_samples": 10042,
        },
        "arc": {
            "name": "ARC",
            "description": "AI2 Reasoning Challenge",
            "dataset": "ai2_arc",
            "split": "test",
            "metric": "accuracy",
            "categories": ["science"],
            "input_key": "question",
            "output_key": "answerKey",
            "choices_key": "choices",
            "max_samples": 1172,
        },
        "truthfulqa": {
            "name": "TruthfulQA",
            "description": "Truthfulness benchmark",
            "dataset": "truthful_qa",
            "split": "validation",
            "metric": "mc2",
            "categories": ["truthfulness"],
            "input_key": "question",
            "output_key": "mc2_targets",
            "max_samples": 817,
        },
        "winogrande": {
            "name": "Winogrande",
            "description": "Commonsense reasoning",
            "dataset": "winogrande",
            "split": "validation",
            "metric": "accuracy",
            "categories": ["commonsense"],
            "input_key": "sentence",
            "output_key": "answer",
            "choices_key": "options",
            "max_samples": 1267,
        },
    }
    
    @classmethod
    def get_benchmark(cls, name: str) -> Dict[str, Any]:
        """Get benchmark configuration by name."""
        if name not in cls.BENCHMARKS:
            raise ValueError(f"Benchmark {name} not found. Available: {list(cls.BENCHMARKS.keys())}")
        return cls.BENCHMARKS[name]
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all available benchmarks."""
        return list(cls.BENCHMARKS.keys())


# ==================== Benchmark Dataset ====================
class BenchmarkDataset(Dataset):
    """Dataset wrapper for benchmark tasks."""
    
    def __init__(
        self,
        benchmark_name: str,
        tokenizer: Any,
        processor: Any = None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        template: str = "qwen2_vl",
    ):
        self.benchmark_name = benchmark_name
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.template = template
        
        # Load benchmark configuration
        self.config = BenchmarkRegistry.get_benchmark(benchmark_name)
        
        # Load dataset
        self.dataset = load_dataset(
            self.config["dataset"],
            split=self.config["split"],
            trust_remote_code=True,
        )
        
        # Limit samples if specified
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        elif self.config.get("max_samples"):
            self.dataset = self.dataset.select(range(min(self.config["max_samples"], len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples for {self.config['name']}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # Format input based on benchmark type
        if self.benchmark_name == "mmlu":
            prompt = self._format_mmlu(item)
        elif self.benchmark_name == "humaneval":
            prompt = self._format_humaneval(item)
        elif self.benchmark_name == "gsm8k":
            prompt = self._format_gsm8k(item)
        elif self.benchmark_name == "hellaswag":
            prompt = self._format_hellaswag(item)
        elif self.benchmark_name == "arc":
            prompt = self._format_arc(item)
        elif self.benchmark_name == "truthfulqa":
            prompt = self._format_truthfulqa(item)
        elif self.benchmark_name == "winogrande":
            prompt = self._format_winogrande(item)
        else:
            prompt = item[self.config["input_key"]]
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        
        # Handle multimodal models
        if self.processor and hasattr(self.processor, "image_processor"):
            # For multimodal models, we need to handle images/videos
            # This is a simplified version - in practice you'd need to load actual images
            encoding["pixel_values"] = torch.zeros((1, 3, 224, 224))  # Placeholder
            encoding["image_grid_thw"] = torch.tensor([[1, 14, 14]])
        
        # Prepare labels for evaluation
        if "output_key" in self.config:
            target = item[self.config["output_key"]]
            encoding["labels"] = self.tokenizer(
                str(target),
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze()
        
        # Store original item for metric calculation
        encoding["original_item"] = item
        encoding["benchmark"] = self.benchmark_name
        
        return {k: v.squeeze() if isinstance(v, torch.Tensor) and k != "labels" else v 
                for k, v in encoding.items()}
    
    def _format_mmlu(self, item: Dict) -> str:
        """Format MMLU prompt."""
        choices = item[self.config["choices_key"]]
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{item['question']}\n\n{choices_text}\n\nAnswer:"
    
    def _format_humaneval(self, item: Dict) -> str:
        """Format HumanEval prompt."""
        return item["prompt"]
    
    def _format_gsm8k(self, item: Dict) -> str:
        """Format GSM8K prompt."""
        return f"Solve this math problem step by step:\n{item['question']}\n\nAnswer:"
    
    def _format_hellaswag(self, item: Dict) -> str:
        """Format HellaSwag prompt."""
        endings = item["endings"]
        endings_text = "\n".join([f"{i+1}. {ending}" for i, ending in enumerate(endings)])
        return f"{item['ctx']}\n\nChoose the most plausible continuation:\n{endings_text}\n\nAnswer:"
    
    def _format_arc(self, item: Dict) -> str:
        """Format ARC prompt."""
        choices = item["choices"]
        choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])
        return f"{item['question']}\n\n{choices_text}\n\nAnswer:"
    
    def _format_truthfulqa(self, item: Dict) -> str:
        """Format TruthfulQA prompt."""
        return f"{item['question']}\n\nAnswer truthfully:"
    
    def _format_winogrande(self, item: Dict) -> str:
        """Format Winogrande prompt."""
        options = item["options"]
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"{item['sentence']}\n\nChoose the correct option:\n{options_text}\n\nAnswer:"


# ==================== Metric Calculator ====================
class MetricCalculator:
    """Calculate metrics for different benchmark types."""
    
    @staticmethod
    def calculate(predictions: List[Any], references: List[Any], benchmark_name: str) -> Dict[str, float]:
        """Calculate metrics for a specific benchmark."""
        config = BenchmarkRegistry.get_benchmark(benchmark_name)
        metric_name = config["metric"]
        
        if metric_name == "accuracy":
            return MetricCalculator._calculate_accuracy(predictions, references)
        elif metric_name == "pass@k":
            return MetricCalculator._calculate_pass_at_k(predictions, references, k=[1, 10, 100])
        elif metric_name == "mc2":
            return MetricCalculator._calculate_mc2(predictions, references)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    @staticmethod
    def _calculate_accuracy(predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """Calculate accuracy."""
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        accuracy = correct / len(predictions) if predictions else 0
        return {"accuracy": accuracy, "correct": correct, "total": len(predictions)}
    
    @staticmethod
    def _calculate_pass_at_k(predictions: List[str], references: List[str], k: List[int] = [1, 10, 100]) -> Dict[str, float]:
        """Calculate pass@k for code generation."""
        # Simplified implementation - in practice you'd execute code
        results = {}
        for k_val in k:
            # This is a placeholder - real implementation would run tests
            results[f"pass@{k_val}"] = 0.0
        return results
    
    @staticmethod
    def _calculate_mc2(predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """Calculate MC2 metric for TruthfulQA."""
        # Simplified implementation
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        mc2 = correct / len(predictions) if predictions else 0
        return {"mc2": mc2}


# ==================== Report Generator ====================
class ReportGenerator:
    """Generate benchmark reports in various formats."""
    
    @staticmethod
    def generate_html_report(results: Dict[str, Dict[str, float]], output_path: str, model_name: str):
        """Generate HTML report."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {{ model_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .metric { font-weight: bold; color: #2c3e50; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Benchmark Report: {{ model_name }}</h1>
    <p class="timestamp">Generated: {{ timestamp }}</p>
    
    <h2>Summary</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Score</th>
            <th>Details</th>
        </tr>
        {% for benchmark, metrics in results.items() %}
        <tr>
            <td>{{ benchmark }}</td>
            <td class="metric">{{ "%.4f"|format(metrics.get('accuracy', metrics.get('mc2', metrics.get('pass@1', 0.0)))) }}</td>
            <td>{{ metrics|tojson }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Detailed Results</h2>
    {% for benchmark, metrics in results.items() %}
    <h3>{{ benchmark }}</h3>
    <ul>
        {% for metric, value in metrics.items() %}
        <li><strong>{{ metric }}:</strong> {{ value }}</li>
        {% endfor %}
    </ul>
    {% endfor %}
    
    <footer>
        <p>Generated by forge Benchmarking Framework</p>
    </footer>
</body>
</html>
        """
        
        template = Template(template_str)
        html_content = template.render(
            model_name=model_name,
            results=results,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
    
    @staticmethod
    def generate_latex_report(results: Dict[str, Dict[str, float]], output_path: str, model_name: str):
        """Generate LaTeX report."""
        latex_str = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}

\title{Benchmark Report: """ + model_name + r"""}
\author{forge Benchmarking Framework}
\date{""" + datetime.datetime.now().strftime("%Y-%m-%d") + r"""}

\begin{document}

\maketitle

\section{Summary}
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Benchmark} & \textbf{Score} & \textbf{Details} \\
\midrule
"""
        
        for benchmark, metrics in results.items():
            main_score = metrics.get('accuracy', metrics.get('mc2', metrics.get('pass@1', 0.0)))
            latex_str += f"{benchmark} & {main_score:.4f} & {list(metrics.keys())} \\\\\n"
        
        latex_str += r"""\bottomrule
\end{tabular}
\caption{Benchmark Results}
\end{table}

\section{Detailed Results}
"""
        
        for benchmark, metrics in results.items():
            latex_str += f"\\subsection{{{benchmark}}}\n"
            latex_str += "\\begin{itemize}\n"
            for metric, value in metrics.items():
                latex_str += f"\\item \\textbf{{{metric}}}: {value}\n"
            latex_str += "\\end{itemize}\n\n"
        
        latex_str += r"""\end{document}"""
        
        with open(output_path, "w") as f:
            f.write(latex_str)
        
        print(f"LaTeX report saved to {output_path}")
    
    @staticmethod
    def generate_leaderboard_entry(results: Dict[str, Dict[str, float]], model_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry for public leaderboard."""
        entry = {
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
            "model_info": model_info,
            "average_score": np.mean([
                metrics.get('accuracy', metrics.get('mc2', metrics.get('pass@1', 0.0)))
                for metrics in results.values()
            ])
        }
        return entry


# ==================== Leaderboard Integration ====================
class LeaderboardManager:
    """Manage integration with Hugging Face Spaces leaderboard."""
    
    def __init__(self, repo_id: str = "forge/leaderboard", token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
    
    def submit_results(self, entry: Dict[str, Any], branch: str = "main") -> str:
        """Submit results to leaderboard."""
        try:
            # Create repo if it doesn't exist
            try:
                create_repo(self.repo_id, token=self.token, repo_type="space", space_sdk="gradio")
            except Exception:
                pass  # Repo already exists
            
            # Prepare results file
            results_file = f"results/{entry['model_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Upload to Hugging Face
            self.api.upload_file(
                path_or_fileobj=json.dumps(entry, indent=2).encode(),
                path_in_repo=results_file,
                repo_id=self.repo_id,
                repo_type="space",
                token=self.token,
            )
            
            print(f"Results submitted to {self.repo_id}/{results_file}")
            return results_file
            
        except Exception as e:
            print(f"Failed to submit to leaderboard: {e}")
            return ""


# ==================== Main Benchmark Runner ====================
class BenchmarkRunner:
    """Run benchmarks and generate reports."""
    
    def __init__(
        self,
        model_name_or_path: str,
        benchmarks: List[str] = None,
        batch_size: int = 4,
        max_length: int = 2048,
        output_dir: str = "benchmark_results",
        template: str = "qwen2_vl",
        device: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.benchmarks = benchmarks or BenchmarkRegistry.list_benchmarks()
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.template = template
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        print(f"Loading model: {model_name_or_path}")
        self.model, self.tokenizer, self.processor = self._load_model()
        
        # Results storage
        self.results = {}
        self.predictions = {}
    
    def _load_model(self) -> Tuple[Any, Any, Any]:
        """Load model, tokenizer, and processor."""
        # Prepare args for model loading
        args = {
            "model_name_or_path": self.model_name_or_path,
            "stage": "sft",
            "do_train": False,
            "finetuning_type": "full",
            "template": self.template,
            "output_dir": str(self.output_dir / "model_cache"),
            "bf16": True,
            "report_to": "none",
        }
        
        model_args, _, training_args, finetuning_args, _ = get_train_args(args)
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        processor = tokenizer_module.get("processor")
        
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
        
        # Move model to device
        if self.device == "auto":
            if torch.cuda.is_available():
                model = model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                model = model.to("mps")
        else:
            model = model.to(self.device)
        
        model.eval()
        
        return model, tokenizer, processor
    
    def run_benchmark(self, benchmark_name: str) -> Dict[str, float]:
        """Run a single benchmark."""
        print(f"\n{'='*60}")
        print(f"Running benchmark: {benchmark_name}")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = BenchmarkDataset(
            benchmark_name=benchmark_name,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_length=self.max_length,
            template=self.template,
        )
        
        # Create data collator
        if self.processor and hasattr(self.processor, "image_processor"):
            # Multimodal model
            data_collator = MultiModalDataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                pad_to_multiple_of=8,
                label_pad_token_id=IGNORE_INDEX,
            )
        else:
            # Text model
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                pad_to_multiple_of=8,
                label_pad_token_id=IGNORE_INDEX,
            )
        
        # Run inference
        predictions = []
        references = []
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                )
                
                # Decode predictions
                for i in range(len(outputs)):
                    pred_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                    predictions.append(pred_text)
                    
                    # Get reference if available
                    if "labels" in batch:
                        ref_text = self.tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
                        references.append(ref_text)
        
        # Calculate metrics
        metrics = MetricCalculator.calculate(predictions, references, benchmark_name)
        
        # Store results
        self.results[benchmark_name] = metrics
        self.predictions[benchmark_name] = predictions
        
        print(f"Results for {benchmark_name}: {metrics}")
        
        return metrics
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all specified benchmarks."""
        print(f"Starting benchmark suite for {self.model_name_or_path}")
        print(f"Benchmarks to run: {', '.join(self.benchmarks)}")
        
        for benchmark in self.benchmarks:
            try:
                self.run_benchmark(benchmark)
            except Exception as e:
                print(f"Error running benchmark {benchmark}: {e}")
                self.results[benchmark] = {"error": str(e)}
        
        # Generate reports
        self.generate_reports()
        
        return self.results
    
    def generate_reports(self):
        """Generate all reports."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = self.model_name_or_path.split("/")[-1]
        
        # HTML report
        html_path = self.output_dir / f"report_{model_short_name}_{timestamp}.html"
        ReportGenerator.generate_html_report(self.results, str(html_path), self.model_name_or_path)
        
        # LaTeX report
        latex_path = self.output_dir / f"report_{model_short_name}_{timestamp}.tex"
        ReportGenerator.generate_latex_report(self.results, str(latex_path), self.model_name_or_path)
        
        # JSON results
        json_path = self.output_dir / f"results_{model_short_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "model": self.model_name_or_path,
                "timestamp": timestamp,
                "results": self.results,
                "benchmarks": self.benchmarks,
            }, f, indent=2)
        
        print(f"\nReports generated in {self.output_dir}")
        print(f"HTML: {html_path}")
        print(f"LaTeX: {latex_path}")
        print(f"JSON: {json_path}")
        
        # Generate leaderboard entry
        entry = ReportGenerator.generate_leaderboard_entry(
            self.results,
            self.model_name_or_path,
            {
                "model_type": "multimodal" if self.processor else "text",
                "template": self.template,
                "benchmarks": self.benchmarks,
            }
        )
        
        # Save leaderboard entry
        entry_path = self.output_dir / f"leaderboard_entry_{model_short_name}_{timestamp}.json"
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)
        
        print(f"Leaderboard entry: {entry_path}")


# ==================== Original Qwen2-VL Code (Preserved) ====================
class DummyDataset(Dataset):
    def __init__(self, size: int = 1000, seq_length: int = 1024, processor: Qwen2_5_VLProcessor = None):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768
        self.processor = processor

        image_token_num = 18 * 18 // (2 * 2)
        image_t = 2

        self.text_seqlen = seq_length // 4  # 25% text
        video_seq_length = self.seq_length - self.text_seqlen - image_t * image_token_num
        video_t = video_seq_length // image_token_num

        self.image_size = [18 * 18 * image_t, 1176]
        self.image_grid_thw = torch.tensor([[1, 18, 18]] * image_t, dtype=torch.long)
        self.image_seqlen = image_t * image_token_num

        self.video_size = [18 * 18 * video_t, 1176]
        self.video_grid_thw = torch.tensor([[video_t, 18, 18]], dtype=torch.long)
        self.video_seqlen = video_t * image_token_num

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        input_ids[: self.image_seqlen] = self.processor.image_token_id
        input_ids[self.image_seqlen : self.image_seqlen + self.video_seqlen] = self.processor.video_token_id

        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        labels[: self.image_seqlen + self.video_seqlen] = IGNORE_INDEX
        pixel_values = torch.rand(self.image_size, dtype=torch.float32)
        pixel_values_videos = torch.rand(self.video_size, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": self.image_grid_thw,
            "video_grid_thw": self.video_grid_thw,
        }


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __post_init__(self):
        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_pixel_values = [feature.pop("pixel_values") for feature in features]
        batch_pixel_values_videos = [feature.pop("pixel_values_videos") for feature in features]
        batch_image_grid_thw = [feature.pop("image_grid_thw") for feature in features]
        batch_video_grid_thw = [feature.pop("video_grid_thw") for feature in features]

        batch: dict[str, torch.Tensor] = super().__call__(features)

        batch["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
        batch["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
        batch["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
        batch["video_grid_thw"] = torch.cat(batch_video_grid_thw, dim=0)

        if self.get_rope_func is not None:
            rope_index_kwargs = {
                "input_ids": batch["input_ids"],
                "image_grid_thw": batch["image_grid_thw"],
                "video_grid_thw": batch["video_grid_thw"],
                "attention_mask": (batch["attention_mask"] >= 1).float(),
            }
            batch["position_ids"], batch["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

        if "position_ids" not in batch or batch["position_ids"].dim() != 3:
            raise ValueError("Qwen2VL requires 3D position ids for mrope.")

        return batch


def bench_qwen(
    model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    batch_size: int = 1,
    seq_length: int = 2048,
    liger_kernel: bool = False,
    deepspeed_stage: int = 3,
    benchmarks: str = "mmlu,humaneval,gsm8k",  # New parameter
    output_dir: str = "benchmark_results",  # New parameter
    run_original: bool = False,  # New parameter to run original training
    submit_to_leaderboard: bool = False,  # New parameter
    leaderboard_repo: str = "forge/leaderboard",  # New parameter
    hf_token: str = None,  # New parameter
):
    """
    Run benchmarking suite for Qwen2-VL models.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model ID
        batch_size: Batch size for inference
        seq_length: Maximum sequence length
        liger_kernel: Whether to use Liger kernel
        deepspeed_stage: DeepSpeed stage (2 or 3)
        benchmarks: Comma-separated list of benchmarks to run
        output_dir: Directory to save results
        run_original: Whether to run original training benchmark
        submit_to_leaderboard: Whether to submit results to leaderboard
        leaderboard_repo: HuggingFace Space repo for leaderboard
        hf_token: HuggingFace token for leaderboard submission
    """
    
    # Parse benchmarks
    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    
    # Run original training benchmark if requested
    if run_original:
        print("Running original training benchmark...")
        _run_original_training_benchmark(
            model_name_or_path, batch_size, seq_length, liger_kernel, deepspeed_stage
        )
    
    # Run evaluation benchmarks
    print(f"\nRunning evaluation benchmarks: {benchmark_list}")
    
    runner = BenchmarkRunner(
        model_name_or_path=model_name_or_path,
        benchmarks=benchmark_list,
        batch_size=batch_size,
        max_length=seq_length,
        output_dir=output_dir,
        template="qwen2_vl",
    )
    
    results = runner.run_all_benchmarks()
    
    # Submit to leaderboard if requested
    if submit_to_leaderboard and hf_token:
        print("\nSubmitting results to leaderboard...")
        leaderboard = LeaderboardManager(repo_id=leaderboard_repo, token=hf_token)
        
        # Load the leaderboard entry we just generated
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = model_name_or_path.split("/")[-1]
        entry_path = Path(output_dir) / f"leaderboard_entry_{model_short_name}_{timestamp}.json"
        
        if entry_path.exists():
            with open(entry_path, "r") as f:
                entry = json.load(f)
            
            leaderboard.submit_results(entry)
    
    print("\nBenchmarking complete!")
    return results


def _run_original_training_benchmark(
    model_name_or_path: str,
    batch_size: int,
    seq_length: int,
    liger_kernel: bool,
    deepspeed_stage: int,
):
    """Run the original training benchmark (preserved for backward compatibility)."""
    os.environ["LLAMABOARD_ENABLED"] = "true"
    os.environ["LLAMABOARD_WORKDIR"] = "output/dummy_dir"
    args = {
        "model_name_or_path": model_name_or_path,
        "enable_liger_kernel": liger_kernel,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": "alpaca_en_demo",
        "template": "qwen2_vl",
        "cutoff_len": seq_length,
        "output_dir": "output/dummy_dir",
        "logging_steps": 10,
        "save_strategy": "no",
        "save_only_model": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": batch_size,
        "max_steps": 1000,
        "bf16": True,
        "include_num_input_tokens_seen": True,
        "report_to": "none",
    }
    if deepspeed_stage in [2, 3]:
        args["deepspeed"] = f"examples/deepspeed/ds_z{deepspeed_stage}_config.json"

    model_args, _, training_args, finetuning_args, _ = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    trainset = DummyDataset(size=100000, seq_length=seq_length, processor=tokenizer_module["processor"])
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, pad_to_multiple_of=8, label_pad_token_id=IGNORE_INDEX
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=trainset,
        **tokenizer_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    fire.Fire(bench_qwen)