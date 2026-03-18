# 🔥 **forge** — Your local foundry for open models.
### *Stop stitching together a dozen tools. Train, serve, and deploy any open model from a single, lightning-fast local UI.*

[![GitHub Stars](https://img.shields.io/github/stars/forge/forge?style=social)](https://github.com/forge/forge)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord&logoColor=white)](https://discord.gg/forge)
[![Twitter](https://img.shields.io/twitter/follow/forge?style=social)](https://twitter.com/forge)

---

**forge** is a production-ready, upgraded fork of [unsloth](https://github.com/unsloth-ai/unsloth) (55k+ ⭐) that transforms your local machine into a complete model foundry. No more juggling training scripts, inference servers, and deployment pipelines. **One interface. One workflow. Zero friction.**

---

## 🚀 **Why forge? The Upgrade That Changes Everything.**

unsloth gave us blazing-fast training. **forge** gives you the entire lifecycle.

| Feature | unsloth (Original) | **forge** (This Fork) |
|---------|-------------------|----------------------|
| **Training** | ✅ Fast LoRA/QLoRA | ✅ Fast LoRA/QLoRA + **Full fine-tuning** |
| **Inference** | ❌ Basic generation | ✅ **Production inference server** with OpenAI-compatible API |
| **Model Formats** | Limited | ✅ **GGUF, GPTQ, AWQ, EXL2** quantized serving |
| **Model Registry** | ❌ None | ✅ **Integrated version control** with HF Hub sync |
| **Deployment** | ❌ Manual | ✅ **One-click cloud deployment** (AWS/GCP/Azure) with auto-scaling |
| **UI/UX** | ❌ Notebook-only | ✅ **Full local web UI** for all operations |
| **Monitoring** | ❌ Basic logs | ✅ **Real-time metrics dashboard** |

---

## ⚡ **Quickstart: From Zero to Deployed in 5 Minutes**

### 1. Install forge
```bash
pip install forge-ai
# Or with CUDA 12.1 support
pip install forge-ai[cu121]
```

### 2. Launch the Local UI
```bash
forge serve --ui
# Opens at http://localhost:7860
```

### 3. Train Your First Model (Python API)
```python
from forge import Trainer, ModelRegistry

# Load and train
trainer = Trainer(
    model="meta-llama/Llama-3-8B",
    dataset="your_dataset.jsonl",
    method="qlora",  # or "lora", "full"
    epochs=3,
)

model_path = trainer.run()

# Register and deploy
registry = ModelRegistry()
model_id = registry.register(model_path, name="my-llama-3-finetune")

# One-click deploy to cloud
deployer = registry.deploy(
    model_id,
    provider="aws",
    instance="g5.xlarge",
    scaling={"min": 1, "max": 5}
)
print(f"Deployed at: {deployer.endpoint}")
```

### 4. Use the Inference Server (OpenAI-Compatible)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # forge local server
    api_key="forge"
)

response = client.chat.completions.create(
    model="my-llama-3-finetune",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────┐
│                    forge Web UI                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Training│  │ Serving │  │Registry │  │ Deploy  │    │
│  │ Studio  │  │ Control │  │ Browser │  │ Manager │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
└───────┼────────────┼────────────┼────────────┼──────────┘
        │            │            │            │
┌───────▼────────────▼────────────▼────────────▼──────────┐
│                  forge Core Engine                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Trainer    │  │  Inference  │  │  Deployment │     │
│  │  (unsloth++)│  │  Server     │  │  Orchestrator│    │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────────────────────────────────────┐       │
│  │          Model Registry & Hub Sync          │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
        │            │            │            │
┌───────▼────────────▼────────────▼────────────▼──────────┐
│              Local/Cloud Infrastructure                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │  GPU    │  │ Storage │  │  Cloud  │  │Monitoring│    │
│  │  Pool   │  │ (Hub)   │  │  API    │  │  Stack   │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Training Engine**: unsloth's optimized core + full fine-tuning support
- **Inference Server**: Production-ready with quantized model serving (GGUF/GPTQ/AWQ)
- **Model Registry**: Version control, sharing, and automatic Hugging Face Hub sync
- **Deployment Orchestrator**: One-click to AWS/GCP/Azure with auto-scaling
- **Web UI**: Unified interface for all operations

---

## 📦 **Installation**

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Option 1: pip (Recommended)
```bash
# Basic installation
pip install forge-ai

# With CUDA 12.1 support
pip install forge-ai[cu121]

# With all optional dependencies
pip install forge-ai[all]
```

### Option 2: Docker
```bash
docker run -p 7860:7860 -p 8000:8000 \
  --gpus all \
  forge/forge:latest
```

### Option 3: From Source
```bash
git clone https://github.com/forge/forge.git
cd forge
pip install -e .
```

### Verify Installation
```bash
forge --version
forge doctor  # Checks system requirements
```

---

## 🎯 **Migrating from unsloth**

Switching is seamless. Your existing unsloth code works with minimal changes:

```python
# Old unsloth code
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)

# New forge code (same API, more power)
from forge import Trainer
trainer = Trainer(model="...", ...)  # Now includes serving & deployment
```

**Migration benefits:**
- ✅ **Same training speed** (unsloth optimizations preserved)
- ✅ **+ Production inference server**
- ✅ **+ Integrated model registry**
- ✅ **+ One-click deployment**
- ✅ **+ Beautiful web UI**

---

## 🌟 **What's Coming Next**

- [ ] **Multi-node training** across consumer GPUs
- [ ] **Model merging** studio in UI
- [ ] **Dataset marketplace** integration
- [ ] **Mobile app** for monitoring deployments
- [ ] **Plugin system** for custom transformations

---

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas:**
1. New model architecture support
2. Cloud provider integrations
3. UI/UX improvements
4. Performance optimizations

---

## 📄 **License**

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## 🙏 **Credits**

Built on the incredible work of:
- [unsloth](https://github.com/unsloth-ai/unsloth) team for the training optimizations
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF support
- [vLLM](https://github.com/vllm-project/vllm) for inference inspiration
- [Hugging Face](https://huggingface.co) for the model ecosystem

---

**Ready to build your model foundry?**

```bash
pip install forge-ai && forge serve --ui
```

**⭐ Star us on GitHub** if you believe in democratizing AI infrastructure.

[![Star History Chart](https://api.star-history.com/svg?repos=forge/forge&type=Date)](https://star-history.com/#forge/forge&Date)