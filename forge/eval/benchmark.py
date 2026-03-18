"""
forge Unified Benchmarking & Leaderboard Framework

This module provides a comprehensive benchmarking system for evaluating language models
across standardized tasks, generating comparison reports, and submitting results to
public leaderboards.
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset, Dataset
import evaluate
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    SAFETY = "safety"
    MULTILINGUAL = "multilingual"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark task."""
    name: str
    type: BenchmarkType
    dataset_name: str
    dataset_split: str = "test"
    metric: str = "accuracy"
    few_shot_examples: int = 0
    max_samples: Optional[int] = None
    description: str = ""
    url: str = ""
    requires_gpu: bool = False
    batch_size: int = 8
    max_length: int = 2048


@dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation."""
    benchmark_name: str
    model_name: str
    score: float
    metrics: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_outputs: Optional[List[Any]] = None


class BenchmarkRegistry:
    """Registry of available benchmarks."""
    
    _benchmarks: Dict[str, BenchmarkConfig] = {}
    
    @classmethod
    def register(cls, config: BenchmarkConfig) -> None:
        """Register a benchmark configuration."""
        cls._benchmarks[config.name] = config
        logger.info(f"Registered benchmark: {config.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BenchmarkConfig]:
        """Get benchmark configuration by name."""
        return cls._benchmarks.get(name)
    
    @classmethod
    def list_benchmarks(cls, benchmark_type: Optional[BenchmarkType] = None) -> List[str]:
        """List all registered benchmarks, optionally filtered by type."""
        if benchmark_type is None:
            return list(cls._benchmarks.keys())
        return [name for name, config in cls._benchmarks.items() 
                if config.type == benchmark_type]
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, BenchmarkConfig]:
        """Get all benchmark configurations."""
        return cls._benchmarks.copy()


class BaseBenchmark:
    """Base class for all benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metric = evaluate.load(config.metric)
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the benchmark dataset."""
        try:
            self.dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split
            )
            if self.config.max_samples:
                self.dataset = self.dataset.select(range(self.config.max_samples))
            logger.info(f"Loaded dataset {self.config.dataset_name} with {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset {self.config.dataset_name}: {e}")
            raise
    
    def preprocess(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single example. Override in subclasses."""
        return example
    
    def postprocess(self, prediction: Any, example: Dict[str, Any]) -> Any:
        """Postprocess model prediction. Override in subclasses."""
        return prediction
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> BenchmarkResult:
        """Run evaluation on the benchmark."""
        raise NotImplementedError("Subclasses must implement evaluate method")


class MMLUBenchmark(BaseBenchmark):
    """Massive Multitask Language Understanding benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.subjects = self._get_subjects()
    
    def _get_subjects(self) -> List[str]:
        """Get list of MMLU subjects."""
        if hasattr(self.dataset, 'features') and 'subject' in self.dataset.features:
            return list(set(self.dataset['subject']))
        return ["default"]
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> BenchmarkResult:
        """Evaluate model on MMLU."""
        results_by_subject = defaultdict(list)
        predictions = []
        references = []
        
        for example in self.dataset:
            subject = example.get('subject', 'default')
            prompt = self._format_prompt(example)
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                             max_length=self.config.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False
                )
            
            prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                        skip_special_tokens=True)
            prediction = self.postprocess(prediction, example)
            
            # Calculate metrics
            score = self._calculate_score(prediction, example)
            results_by_subject[subject].append(score)
            
            predictions.append(prediction)
            references.append(example['answer'])
        
        # Calculate overall and per-subject scores
        overall_score = np.mean([s for scores in results_by_subject.values() for s in scores])
        subject_scores = {subject: np.mean(scores) for subject, scores in results_by_subject.items()}
        
        return BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=model.config.name_or_path,
            score=overall_score,
            metrics={
                "overall_accuracy": overall_score,
                **{f"subject_{k}": v for k, v in subject_scores.items()}
            },
            timestamp=datetime.now().isoformat(),
            metadata={
                "num_subjects": len(self.subjects),
                "num_examples": len(self.dataset),
                "subject_scores": subject_scores
            }
        )
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format MMLU prompt."""
        choices = [example['A'], example['B'], example['C'], example['D']]
        prompt = f"Question: {example['question']}\n"
        for i, choice in enumerate(['A', 'B', 'C', 'D']):
            prompt += f"{choice}. {choices[i]}\n"
        prompt += "Answer:"
        return prompt
    
    def _calculate_score(self, prediction: str, example: Dict[str, Any]) -> float:
        """Calculate score for MMLU."""
        correct_answer = example['answer']
        return 1.0 if prediction.strip().upper() == correct_answer.strip().upper() else 0.0


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval code generation benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.code_metric = evaluate.load("code_eval")
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> BenchmarkResult:
        """Evaluate model on HumanEval."""
        predictions = []
        references = []
        
        for example in self.dataset:
            prompt = self._format_prompt(example)
            
            # Generate code
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.config.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95
                )
            
            generated_code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True)
            
            # Extract function implementation
            code = self._extract_function(generated_code, example)
            predictions.append(code)
            references.append(example['canonical_solution'])
        
        # Evaluate using pass@k
        results = self.code_metric.compute(
            references=references,
            predictions=predictions,
            k=[1, 10, 100]
        )
        
        return BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=model.config.name_or_path,
            score=results['pass@1'],
            metrics={
                "pass@1": results['pass@1'],
                "pass@10": results['pass@10'],
                "pass@100": results['pass@100']
            },
            timestamp=datetime.now().isoformat(),
            metadata={
                "num_examples": len(self.dataset),
                "language": "python"
            }
        )
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format HumanEval prompt."""
        return f"{example['prompt']}\n"
    
    def _extract_function(self, generated_code: str, example: Dict[str, Any]) -> str:
        """Extract function implementation from generated code."""
        # Simple extraction - in production, use more robust parsing
        lines = generated_code.split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
            if 'def ' in line and example['entry_point'] in line:
                in_function = True
            if in_function:
                function_lines.append(line)
                if line.strip() == '' and len(function_lines) > 1:
                    break
        
        return '\n'.join(function_lines) if function_lines else generated_code


class GSM8KBenchmark(BaseBenchmark):
    """Grade School Math 8K benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.math_metric = evaluate.load("exact_match")
    
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> BenchmarkResult:
        """Evaluate model on GSM8K."""
        predictions = []
        references = []
        
        for example in self.dataset:
            prompt = self._format_prompt(example)
            
            # Generate answer
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.config.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False
                )
            
            prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True)
            prediction = self._extract_answer(prediction)
            
            predictions.append(prediction)
            references.append(example['answer'])
        
        # Calculate exact match
        results = self.math_metric.compute(
            predictions=predictions,
            references=references
        )
        
        return BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=model.config.name_or_path,
            score=results['exact_match'],
            metrics={
                "exact_match": results['exact_match']
            },
            timestamp=datetime.now().isoformat(),
            metadata={
                "num_examples": len(self.dataset),
                "grade_level": "grade_school"
            }
        )
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format GSM8K prompt."""
        return f"Question: {example['question']}\nAnswer:"
    
    def _extract_answer(self, text: str) -> str:
        """Extract numerical answer from text."""
        # Look for pattern "#### <number>"
        import re
        match = re.search(r'####\s*(\d+)', text)
        if match:
            return match.group(1)
        
        # Fallback: find last number in text
        numbers = re.findall(r'\d+', text)
        return numbers[-1] if numbers else "0"


class ReportGenerator:
    """Generate benchmark reports in various formats."""
    
    @staticmethod
    def generate_html_report(results: List[BenchmarkResult], 
                           output_path: str,
                           title: str = "Benchmark Results") -> None:
        """Generate HTML report from benchmark results."""
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .score { font-weight: bold; }
                .timestamp { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <p>Generated on: {{ generation_time }}</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Benchmark</th>
                    <th>Score</th>
                    <th>Timestamp</th>
                </tr>
                {% for result in results %}
                <tr>
                    <td>{{ result.model_name }}</td>
                    <td>{{ result.benchmark_name }}</td>
                    <td class="score">{{ "%.4f"|format(result.score) }}</td>
                    <td class="timestamp">{{ result.timestamp }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Detailed Metrics</h2>
            {% for result in results %}
            <h3>{{ result.model_name }} - {{ result.benchmark_name }}</h3>
            <table>
                {% for metric, value in result.metrics.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td>{{ "%.4f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endfor %}
        </body>
        </html>
        """)
        
        html_content = html_template.render(
            title=title,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=results
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated at {output_path}")
    
    @staticmethod
    def generate_latex_report(results: List[BenchmarkResult],
                            output_path: str,
                            title: str = "Benchmark Results") -> None:
        """Generate LaTeX report from benchmark results."""
        latex_template = Template(r"""
        \documentclass{article}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{hyperref}
        \usepackage{xcolor}
        
        \title{ {{ title }} }
        \author{forge Benchmark Framework}
        \date{ {{ generation_time }} }
        
        \begin{document}
        
        \maketitle
        
        \section{Summary}
        \begin{table}[h]
        \centering
        \begin{tabular}{lllr}
        \toprule
        Model & Benchmark & Score & Timestamp \\
        \midrule
        {% for result in results %}
        {{ result.model_name }} & {{ result.benchmark_name }} & {{ "%.4f"|format(result.score) }} & {{ result.timestamp }} \\
        {% endfor %}
        \bottomrule
        \end{tabular}
        \caption{Benchmark Results Summary}
        \end{table}
        
        \section{Detailed Results}
        {% for result in results %}
        \subsection{ {{ result.model_name }} - {{ result.benchmark_name }} }
        \begin{table}[h]
        \centering
        \begin{tabular}{lr}
        \toprule
        Metric & Value \\
        \midrule
        {% for metric, value in result.metrics.items() %}
        {{ metric }} & {{ "%.4f"|format(value) }} \\
        {% endfor %}
        \bottomrule
        \end{tabular}
        \end{table}
        {% endfor %}
        
        \end{document}
        """)
        
        latex_content = latex_template.render(
            title=title,
            generation_time=datetime.now().strftime("%Y-%m-%d"),
            results=results
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX report generated at {output_path}")
    
    @staticmethod
    def generate_comparison_chart(results: List[BenchmarkResult],
                                output_path: str,
                                title: str = "Model Comparison") -> None:
        """Generate comparison chart from benchmark results."""
        # Prepare data
        data = []
        for result in results:
            data.append({
                'Model': result.model_name,
                'Benchmark': result.benchmark_name,
                'Score': result.score
            })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot(index='Model', columns='Benchmark', values='Score')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f',
                   linewidths=.5, cbar_kws={'label': 'Score'})
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart generated at {output_path}")


class LeaderboardSubmitter:
    """Submit results to public leaderboards."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
    
    def submit_to_huggingface(self, results: List[BenchmarkResult],
                            space_name: str,
                            username: str) -> bool:
        """Submit results to Hugging Face Spaces leaderboard."""
        try:
            from huggingface_hub import HfApi, create_repo, upload_file
            
            api = HfApi(token=self.api_token)
            
            # Create or get the space
            repo_id = f"{username}/{space_name}"
            try:
                create_repo(repo_id, repo_type="space", space_sdk="gradio")
                logger.info(f"Created new space: {repo_id}")
            except Exception:
                logger.info(f"Space {repo_id} already exists")
            
            # Prepare results data
            results_data = []
            for result in results:
                results_data.append({
                    "model": result.model_name,
                    "benchmark": result.benchmark_name,
                    "score": result.score,
                    "metrics": result.metrics,
                    "timestamp": result.timestamp,
                    "metadata": result.metadata
                })
            
            # Save results to JSON
            results_file = "benchmark_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Upload to space
            api.upload_file(
                path_or_fileobj=results_file,
                path_in_repo="data/results.json",
                repo_id=repo_id,
                repo_type="space"
            )
            
            # Create or update the app file
            app_content = self._generate_gradio_app()
            api.upload_file(
                path_or_fileobj=app_content.encode(),
                path_in_repo="app.py",
                repo_id=repo_id,
                repo_type="space"
            )
            
            # Clean up
            os.remove(results_file)
            
            logger.info(f"Successfully submitted results to {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit to Hugging Face: {e}")
            return False
    
    def _generate_gradio_app(self) -> str:
        """Generate Gradio app for leaderboard."""
        return """
import gradio as gr
import json
import pandas as pd
from datetime import datetime

def load_results():
    try:
        with open("data/results.json", "r") as f:
            results = json.load(f)
        return results
    except:
        return []

def create_leaderboard():
    results = load_results()
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='model',
        columns='benchmark',
        values='score',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate average score
    pivot['Average'] = pivot.iloc[:, 1:].mean(axis=1)
    pivot = pivot.sort_values('Average', ascending=False)
    
    return pivot

def update_leaderboard():
    df = create_leaderboard()
    return df

# Create Gradio interface
with gr.Blocks(title="forge Leaderboard") as demo:
    gr.Markdown("# forge Model Leaderboard")
    gr.Markdown("Benchmark results for various language models")
    
    leaderboard_table = gr.Dataframe(
        value=create_leaderboard(),
        interactive=False,
        label="Leaderboard"
    )
    
    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(
        fn=update_leaderboard,
        outputs=leaderboard_table
    )
    
    gr.Markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    demo.launch()
"""


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, model_name_or_path: str, 
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.float16):
        self.model_name_or_path = model_name_or_path
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def run_benchmark(self, benchmark_name: str, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        config = BenchmarkRegistry.get(benchmark_name)
        if not config:
            raise ValueError(f"Benchmark {benchmark_name} not found in registry")
        
        # Create benchmark instance based on type
        benchmark_class = self._get_benchmark_class(config)
        benchmark = benchmark_class(config)
        
        logger.info(f"Running benchmark: {benchmark_name}")
        result = benchmark.evaluate(self.model, self.tokenizer)
        
        return result
    
    def run_benchmarks(self, benchmark_names: List[str], **kwargs) -> List[BenchmarkResult]:
        """Run multiple benchmarks."""
        results = []
        for name in benchmark_names:
            try:
                result = self.run_benchmark(name, **kwargs)
                results.append(result)
                logger.info(f"Completed {name}: Score = {result.score:.4f}")
            except Exception as e:
                logger.error(f"Failed to run benchmark {name}: {e}")
        
        return results
    
    def _get_benchmark_class(self, config: BenchmarkConfig) -> BaseBenchmark:
        """Get appropriate benchmark class for configuration."""
        benchmark_classes = {
            "mmlu": MMLUBenchmark,
            "humaneval": HumanEvalBenchmark,
            "gsm8k": GSM8KBenchmark,
            # Add more benchmark classes here
        }
        
        benchmark_type = config.name.lower()
        for key, cls in benchmark_classes.items():
            if key in benchmark_type:
                return cls
        
        # Default to base class
        return BaseBenchmark


def register_default_benchmarks() -> None:
    """Register default benchmarks."""
    benchmarks = [
        BenchmarkConfig(
            name="mmlu",
            type=BenchmarkType.KNOWLEDGE,
            dataset_name="hails/mmlu_no_train",
            dataset_split="test",
            metric="accuracy",
            description="Massive Multitask Language Understanding",
            url="https://arxiv.org/abs/2009.03300",
            max_samples=100  # For quick testing
        ),
        BenchmarkConfig(
            name="humaneval",
            type=BenchmarkType.CODE,
            dataset_name="openai_humaneval",
            dataset_split="test",
            metric="code_eval",
            description="HumanEval code generation benchmark",
            url="https://arxiv.org/abs/2107.03374",
            requires_gpu=True
        ),
        BenchmarkConfig(
            name="gsm8k",
            type=BenchmarkType.MATH,
            dataset_name="gsm8k",
            dataset_split="test",
            metric="exact_match",
            description="Grade School Math 8K benchmark",
            url="https://arxiv.org/abs/2110.14168"
        ),
        # Add more benchmarks as needed
    ]
    
    for config in benchmarks:
        BenchmarkRegistry.register(config)


def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Run benchmarks on language models")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["mmlu", "gsm8k"],
                       help="Benchmarks to run")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu, mps)")
    parser.add_argument("--submit", action="store_true",
                       help="Submit results to leaderboard")
    parser.add_argument("--hf_username", type=str,
                       help="Hugging Face username for submission")
    parser.add_argument("--hf_space", type=str, default="forge-leaderboard",
                       help="Hugging Face space name")
    
    args = parser.parse_args()
    
    # Register default benchmarks
    register_default_benchmarks()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize runner
    runner = BenchmarkRunner(
        model_name_or_path=args.model,
        device=args.device
    )
    
    # Load model
    runner.load_model()
    
    # Run benchmarks
    results = runner.run_benchmarks(args.benchmarks)
    
    if not results:
        logger.error("No benchmarks were successfully run")
        return
    
    # Generate reports
    report_generator = ReportGenerator()
    
    # HTML report
    html_path = os.path.join(args.output_dir, "benchmark_report.html")
    report_generator.generate_html_report(results, html_path)
    
    # LaTeX report
    latex_path = os.path.join(args.output_dir, "benchmark_report.tex")
    report_generator.generate_latex_report(results, latex_path)
    
    # Comparison chart
    chart_path = os.path.join(args.output_dir, "comparison_chart.png")
    report_generator.generate_comparison_chart(results, chart_path)
    
    # Save raw results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")
    
    # Submit to leaderboard if requested
    if args.submit:
        if not args.hf_username:
            logger.error("Hugging Face username required for submission")
            return
        
        submitter = LeaderboardSubmitter()
        success = submitter.submit_to_huggingface(
            results=results,
            space_name=args.hf_space,
            username=args.hf_username
        )
        
        if success:
            logger.info("Results submitted to leaderboard successfully")
        else:
            logger.error("Failed to submit results to leaderboard")


if __name__ == "__main__":
    main()