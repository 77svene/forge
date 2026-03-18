<div align="center">

# 🔥 FORGE
### *The Visual-First LLM Toolkit That Tunes Itself*

[![GitHub Stars](https://img.shields.io/github/stars/forge-ai/forge?style=flat-square)](https://github.com/forge-ai/forge)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Discord](https://img.shields.io/discord/123456789?style=flat-square&logo=discord)](https://discord.gg/forge)

**Stop fine-tuning like it's 2023.**

Forge is the first LLM toolkit with a visual builder, one-click deployment, and AI that tunes itself.

**Visual fine-tuning. Autonomous optimization. One-click deployment.**

[Quickstart](#-quickstart) • [Features](#-features) • [Documentation](https://docs.forge.ai) • [Discord](https://discord.gg/forge)

</div>

---

## 🚀 Why Forge > LlamaFactory?

LlamaFactory was great for 2023. Forge is built for 2025.

| Feature | LlamaFactory | Forge |
|---------|--------------|-------|
| **Dataset Handling** | Manual YAML configs | 🎨 **Visual drag-and-drop builder** |
| **Training Monitoring** | Logs & TensorBoard | 📊 **Real-time metrics dashboard** |
| **Deployment** | Manual export scripts | 🚀 **One-click to HF/Ollama/vLLM** |
| **Optimization** | Manual hyperparameter search | 🤖 **Autonomous AI agent** |
| **Quantization** | Manual configuration | ⚡ **Automatic & optimized** |
| **UI/UX** | CLI only | 🖥️ **Full visual web interface** |
| **Time to Production** | Hours/days | **Minutes** |

## ✨ Features

### 🎨 Visual Builder
- Drag-and-drop dataset creation
- Real-time training metrics visualization
- Model comparison dashboards
- Interactive hyperparameter tuning

### 🚀 One-Click Deployment
- **Hugging Face Hub** - Direct push with model cards
- **Ollama** - Local deployment with custom Modelfiles
- **vLLM** - Production-ready serving with automatic quantization
- **GGUF/GGML** - Optimized formats for edge deployment

### 🤖 Autonomous Agent Framework
- Self-optimizing hyperparameters (learning rate, batch size, epochs)
- Automatic architecture search for your dataset
- Early stopping with intelligent fallback strategies
- Multi-objective optimization (speed vs. accuracy)

### 📊 Real-time Monitoring
- Live loss/accuracy curves
- GPU utilization tracking
- Memory usage optimization alerts
- Training cost estimation

## ⚡ Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/forge-ai/forge.git
cd forge

# Install with pip (recommended)
pip install -e .

# Or install with all extras
pip install -e ".[full]"
```

### Launch Visual Interface

```bash
forge ui --port 7860
```

Open `http://localhost:7860` in your browser.

### Python API Example

```python
from forge import ForgeTrainer, ForgeConfig

# Configure your fine-tuning job
config = ForgeConfig(
    model="meta-llama/Llama-3-8b",
    dataset="your_dataset.jsonl",
    task="text-generation",
    visual=True,  # Enable visual monitoring
    auto_optimize=True,  # Let AI tune hyperparameters
)

# Create and run trainer
trainer = ForgeTrainer(config)
trainer.train()

# One-click deployment
trainer.deploy(
    target="huggingface",
    repo_id="your-username/your-model",
    quantization="gptq"  # Automatic quantization
)
```

### CLI Quick Commands

```bash
# Fine-tune with visual interface
forge train --model llama3 --dataset data.jsonl --visual

# Deploy to Ollama
forge deploy --target ollama --model ./output --name my-model

# Start autonomous optimization
forge optimize --model llama3 --dataset data.jsonl --budget 100
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FORGE VISUAL UI                       │
├─────────────────────────────────────────────────────────┤
│  Dataset Builder  │  Training Dashboard  │  Deployment  │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                  CORE ENGINE                             │
├─────────────────────────────────────────────────────────┤
│  Autonomous Agent  │  Training Loop  │  Optimization    │
│  (Hyperparameter   │  (PyTorch/      │  (Multi-objective│
│   Search)          │   DeepSpeed)    │   Bayesian)      │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│               MODEL & DATA LAYER                        │
├─────────────────────────────────────────────────────────┤
│  100+ Models  │  Dataset  │  Quantization  │  Export    │
│  (LLMs/VLMs)  │  Loaders │  (AutoGPTQ,    │  Formats   │
│               │          │   AWQ, GGUF)   │            │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation Options

### Basic Installation
```bash
pip install forge-ai
```

### With All Features
```bash
pip install forge-ai[full]
```

### Docker
```bash
docker run -p 7860:7860 forgeai/forge:latest
```

### From Source
```bash
git clone https://github.com/forge-ai/forge.git
cd forge
pip install -e ".[dev]"
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md).

## 📜 License

Forge is released under the [Apache 2.0 License](LICENSE).

## 🔗 Links

- [Documentation](https://docs.forge.ai)
- [Discord Community](https://discord.gg/forge)
- [Twitter/X](https://twitter.com/forge_ai)
- [Blog](https://blog.forge.ai)

---

<div align="center">

**Built with ❤️ by the Forge Team**

*Star us on GitHub if Forge saves you time!*

</div>