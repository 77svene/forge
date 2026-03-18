import os
import sys

# Add parent dir to path to allow importing conf.py
sys.path.insert(0, os.path.abspath(".."))

from conf import *  # noqa: F403

# Language settings
language = "zh_CN"
html_search_language = "zh"

# Static files
# Point to the root _static directory
html_static_path = ["../_static"]

# Add custom JS for language switcher and interactive features
html_js_files = [
    "js/switcher.js",
    "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",
    "js/interactive-playground.js",
    "js/model-comparison.js",
]

# Add custom CSS for interactive components
html_css_files = [
    "css/interactive-docs.css",
]

# Enable MyST-NB for Jupyter notebook execution in documentation
extensions.append("myst_nb")

# Configure MyST-NB for interactive execution
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = False

# Configure interactive widgets
interactive_widgets = {
    "enable": True,
    "default_kernel": "python3",
    "allowed_languages": ["python"],
    "max_output_lines": 1000,
    "timeout": 30,
}

# Pyodide configuration for browser-based Python execution
pyodide_config = {
    "indexURL": "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
    "packages": [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
    ],
}

# Model comparison tool configuration
model_comparison_config = {
    "enable": True,
    "default_models": [
        "llama2-7b",
        "llama2-13b",
        "mistral-7b",
        "falcon-7b",
    ],
    "metrics": ["perplexity", "accuracy", "latency", "memory"],
    "max_comparison_models": 4,
}

# Interactive playground configuration
playground_config = {
    "enable": True,
    "default_framework": "forge",
    "presets": {
        "quick_start": {
            "model_name": "llama2-7b-hf",
            "dataset": "alpaca_gpt4_en",
            "template": "llama2",
            "finetuning_type": "lora",
            "lora_rank": 8,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 4,
        },
        "qlora_optimized": {
            "model_name": "llama2-7b-hf",
            "dataset": "alpaca_gpt4_en",
            "template": "llama2",
            "finetuning_type": "qlora",
            "quantization_bit": 4,
            "lora_rank": 16,
            "learning_rate": 1e-4,
            "num_epochs": 5,
            "batch_size": 2,
        },
        "full_finetune": {
            "model_name": "llama2-7b-hf",
            "dataset": "alpaca_gpt4_en",
            "template": "llama2",
            "finetuning_type": "full",
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "batch_size": 8,
        },
    },
    "available_models": [
        "llama2-7b-hf",
        "llama2-13b-hf",
        "mistral-7b-v0.1",
        "falcon-7b",
        "qwen-7b",
        "baichuan2-7b",
    ],
    "available_datasets": [
        "alpaca_gpt4_en",
        "alpaca_gpt4_zh",
        "self_cognition",
        "sharegpt",
        "openassistant",
    ],
}

# Custom HTML context for interactive components
html_context = {
    **html_context,
    "pyodide_config": pyodide_config,
    "interactive_widgets": interactive_widgets,
    "model_comparison_config": model_comparison_config,
    "playground_config": playground_config,
}

# Enable live code execution in documentation
suppress_warnings = ["mystnb.unknown_mime_type"]

# Configure notebook output
nb_output_stderr = "remove"
nb_merge_streams = True

# Add custom directives for interactive components
rst_prolog = """
.. |try_it| raw:: html

   <div class="try-it-button" onclick="openPlayground(this)">Try It Live</div>

.. |compare_models| raw:: html

   <div class="compare-models-button" onclick="openModelComparison()">Compare Models</div>

.. |interactive_widget| raw:: html

   <div class="interactive-widget"></div>
"""