# **Forge: Craft, Train, Deploy.**

**Stop scrolling through terminal logs.** Visualize your LLM training in real-time and deploy with a single click. Forge is the next evolution of LlamaFactory, transforming it from a powerful CLI tool into a unified, visual platform for the entire model development lifecycle.

---

## **Why Forge? A Quantum Leap from LlamaFactory**

While LlamaFactory provides an excellent foundation for fine-tuning, Forge supercharges it with production-grade features that eliminate manual overhead and accelerate your path from model to API.

| Feature | LlamaFactory | **Forge** |
| :--- | :--- | :--- |
| **Training Monitoring** | Terminal logs, TensorBoard | **Real-time dashboard** with live metrics, cost estimation, and hyperparameter visualization |
| **Deployment** | Manual export & serving setup | **One-click pipeline** that optimizes, quantizes, and serves models via auto-scaling API endpoints |
| **Data Handling** | Basic preprocessing | **Intelligent preprocessing** with quality scoring, deduplication, and synthetic data augmentation suggestions |
| **Infrastructure** | Single-machine or manual setup | **Multi-cloud orchestration** that dynamically allocates workloads for optimal cost/speed |
| **Evaluation** | Manual script execution | **Built-in evaluation suite** with automated benchmarking and model card generation |
| **Workflow** | Disconnected steps | **Unified lifecycle** from dataset to deployed API in one platform |

---

## **Quickstart: From Zero to Deployed Model in 5 Minutes**

### 1. Installation
```bash
pip install forge-llm
```

### 2. Craft Your Training Configuration
Create a `config.yaml` file:
```yaml
model:
  name: meta-llama/Llama-3-8b
  method: qlora

dataset:
  path: your_dataset.jsonl
  preprocess: true  # Enable intelligent preprocessing

training:
  epochs: 3
  learning_rate: 2e-4
  dashboard: true  # Launch real-time dashboard

deployment:
  auto_deploy: true
  cloud: aws  # or gcp, azure, hybrid
  instance_type: auto
```

### 3. Launch Training & Monitor Visually
```python
from forge import Forge

# Initialize the platform
forge = Forge(config="config.yaml")

# Start training with live dashboard
forge.train()

# The dashboard opens automatically at http://localhost:8080
# Watch metrics update in real-time, adjust hyperparameters on the fly
```

### 4. Deploy with One Command
```python
# After training completes, deploy immediately
endpoint = forge.deploy()

print(f"Model deployed at: {endpoint.url}")
# API is live with automatic scaling, monitoring, and cost tracking
```

---

## **Architecture Overview**

```
┌─────────────────────────────────────────────────────────┐
│                    Forge Platform                        │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  Dashboard  │  Pipeline   │  Optimizer  │  Orchestrator│
│  (React)    │  (Airflow)  │  (AutoML)   │  (K8s/Multi) │
├─────────────┴─────────────┴─────────────┴──────────────┤
│                   Core Engine (Python)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Dataset │  │ Training│  │Evaluation│  │Deployment│   │
│  │Processor│  │  Engine │  │  Suite   │  │  Manager │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
├─────────────────────────────────────────────────────────┤
│              Model Zoo & Plugin System                  │
│  (100+ LLMs/VLMs, Custom Architectures, Extensions)    │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Real-time Dashboard**: WebSocket-powered visualization of loss, gradients, and system metrics
- **Intelligent Preprocessor**: Automated data quality analysis and enhancement suggestions
- **Multi-Cloud Orchestrator**: Dynamically provisions GPUs across AWS, GCP, Azure, and local clusters
- **Deployment Manager**: Handles quantization, optimization, and serving with built-in A/B testing

---

## **Installation**

### From PyPI (Recommended)
```bash
pip install forge-llm
```

### From Source
```bash
git clone https://github.com/sovereign-ai/forge.git
cd forge
pip install -e .
```

### Docker
```bash
docker pull sovereignai/forge:latest
docker run -p 8080:8080 sovereignai/forge
```

### Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)

---

## **Migration from LlamaFactory**

Switching is seamless—your existing LlamaFactory configs work out of the box:

```bash
# Convert existing LlamaFactory project
forge migrate --from llamafactory --config your_old_config.yaml

# Or run directly with LlamaFactory format
forge train --config llamafactory_style.yaml --dashboard
```

---

## **Advanced Features**

### Multi-Cloud Training
```python
# Automatically find cheapest/fastest GPU availability
forge.train(
    cloud_strategy="cost_optimized",  # or "speed_optimized"
    fallback_providers=["aws", "gcp", "lambda"]
)
```

### Real-time Hyperparameter Tuning
Adjust learning rates, batch sizes, and other parameters during training without restarting.

### Built-in Model Cards
Automatically generate comprehensive model cards with:
- Training metrics and comparisons
- Bias and fairness evaluations
- Carbon footprint estimates
- Deployment recommendations

---

## **Community & Support**

- **Discord**: [Join our community](https://discord.gg/forge)
- **GitHub Discussions**: Ask questions and share your projects
- **Documentation**: [Full documentation](https://docs.forge-llm.ai)
- **Examples**: [Community fine-tuned models](https://huggingface.co/forge)

---

## **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development installation
git clone https://github.com/sovereign-ai/forge.git
cd forge
pip install -e ".[dev]"
pre-commit install
```

---

## **License**

Forge is released under the [Apache 2.0 License](LICENSE).

---

**Stop managing infrastructure. Start building intelligence.**  
⭐ **Star us on GitHub** to support the project and stay updated!

[![GitHub Stars](https://img.shields.io/github/stars/sovereign-ai/forge?style=social)](https://github.com/sovereign-ai/forge)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord)](https://discord.gg/forge)
[![Twitter](https://img.shields.io/twitter/follow/forge_llm?style=social)](https://twitter.com/forge_llm)