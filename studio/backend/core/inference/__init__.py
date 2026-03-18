# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference submodule - Production-grade model serving with vLLM integration

The default get_inference_backend() returns an InferenceOrchestrator that
delegates to a subprocess running vLLM with continuous batching, PagedAttention,
and quantization support. Includes OpenAI-compatible API endpoints.
"""

from .orchestrator import InferenceOrchestrator, get_inference_backend
from .llama_cpp import LlamaCppBackend
from .vllm_engine import VLLMBackend, VLLMEngineConfig
from .openai_api import OpenAICompatibleServer

# Expose InferenceOrchestrator as InferenceBackend for backward compat
InferenceBackend = InferenceOrchestrator

# Default to vLLM backend for production-grade serving
def get_production_inference_backend(
    model_path: str,
    quantization: str = None,
    tensor_parallel_size: int = 1,
    max_batch_size: int = 64,
    **kwargs
):
    """
    Get production-grade inference backend with vLLM.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        quantization: Quantization method ('awq', 'gptq', or None)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_batch_size: Maximum batch size for continuous batching
        **kwargs: Additional arguments for vLLM engine
    """
    config = VLLMEngineConfig(
        model_path=model_path,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        max_batch_size=max_batch_size,
        **kwargs
    )
    return VLLMBackend(config)

__all__ = [
    "InferenceBackend",
    "InferenceOrchestrator", 
    "get_inference_backend",
    "get_production_inference_backend",
    "LlamaCppBackend",
    "VLLMBackend",
    "VLLMEngineConfig",
    "OpenAICompatibleServer",
]