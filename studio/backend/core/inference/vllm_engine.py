"""vLLM Engine for Production-Grade Model Serving

This module implements a high-performance inference engine using vLLM with continuous batching,
PagedAttention, model quantization support, and OpenAI-compatible API endpoints.

Features:
- Continuous batching for high throughput
- PagedAttention for efficient memory management
- Support for AWQ/GPTQ quantization
- CUDA graph capture for optimized inference
- OpenAI-compatible API endpoints
- Async request handling
- Token streaming support

Usage:
    engine = VLLMEngine(model_name="meta-llama/Llama-2-7b-hf")
    await engine.start()
    result = await engine.generate("Hello, world!")
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import torch
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import vLLM - gracefully handle if not installed
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with: pip install vllm")


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLN = "squeezelln"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str
    quantization: QuantizationMethod = QuantizationMethod.NONE
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    cuda_graph_max_batch_size: int = 256


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class OpenAICompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class VLLMEngine:
    """Production-grade vLLM inference engine with continuous batching."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the vLLM engine.
        
        Args:
            config: Model configuration including quantization and parallel settings
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required but not installed. Install with: pip install vllm")
        
        self.config = config
        self.engine = None
        self.model_name = config.model_name
        self.is_running = False
        self.request_count = 0
        self.total_tokens_generated = 0
        
        logger.info(f"Initializing vLLM engine for model: {config.model_name}")
        logger.info(f"Quantization: {config.quantization.value}")
        logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
    
    async def start(self):
        """Start the vLLM engine asynchronously."""
        try:
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                quantization=self.config.quantization.value if self.config.quantization != QuantizationMethod.NONE else None,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
                enable_prefix_caching=self.config.enable_prefix_caching,
                enforce_eager=self.config.enforce_eager,
                max_context_len_to_capture=self.config.max_context_len_to_capture,
                # CUDA graph configuration
                max_seq_len_to_capture=self.config.max_context_len_to_capture,
                cuda_graph_max_batch_size=self.config.cuda_graph_max_batch_size,
            )
            
            # Initialize the async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_running = True
            
            logger.info(f"vLLM engine started successfully for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM engine: {e}")
            raise
    
    async def stop(self):
        """Stop the vLLM engine."""
        if self.engine:
            # vLLM doesn't have an explicit stop method, but we can clean up references
            self.engine = None
            self.is_running = False
            logger.info("vLLM engine stopped")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None
    ) -> Union[GenerationResponse, AsyncGenerator[str, None]]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stop: Stop sequences
            stream: Whether to stream output
            seed: Random seed for reproducibility
            
        Returns:
            GenerationResponse or async generator for streaming
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")
        
        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
            seed=seed,
        )
        
        if stream:
            return self._generate_stream(prompt, sampling_params, request_id)
        else:
            return await self._generate_complete(prompt, sampling_params, request_id)
    
    async def _generate_complete(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> GenerationResponse:
        """Generate complete response without streaming."""
        start_time = time.time()
        
        try:
            # Generate using vLLM
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            )
            
            # Collect all outputs
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if not final_output:
                raise RuntimeError("No output generated")
            
            # Calculate metrics
            generation_time = time.time() - start_time
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens
            
            # Update statistics
            self.request_count += 1
            self.total_tokens_generated += completion_tokens
            
            # Build response
            response = GenerationResponse(
                id=request_id,
                model=self.model_name,
                choices=[{
                    "text": final_output.outputs[0].text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": final_output.outputs[0].finish_reason or "stop"
                }],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            logger.debug(
                f"Generated {completion_tokens} tokens in {generation_time:.2f}s "
                f"({completion_tokens/generation_time:.1f} tokens/s)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def _generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            )
            
            async for request_output in results_generator:
                if request_output.outputs:
                    text = request_output.outputs[0].text
                    if text:
                        # Format as SSE for OpenAI compatibility
                        chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": self.model_name,
                            "choices": [{
                                "text": text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": None
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate chat completion from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream output
            
        Returns:
            Chat completion response or async generator for streaming
        """
        # Convert messages to prompt format
        # This is a simple implementation - adjust based on model requirements
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Generate completion
        if stream:
            return self._chat_completion_stream(
                prompt, messages, max_tokens, temperature, top_p, stop
            )
        else:
            response = await self.generate(
                prompt=prompt,
                max_tokens=max_tokens or 256,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=False
            )
            
            # Format as chat completion
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0]["text"]
                    },
                    "finish_reason": response.choices[0]["finish_reason"]
                }],
                "usage": response.usage
            }
    
    async def _chat_completion_stream(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion."""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens or 256,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id
        )
        
        async for request_output in results_generator:
            if request_output.outputs:
                text = request_output.outputs[0].text
                if text:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": text
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_running:
            return {"status": "not_running"}
        
        return {
            "model_name": self.model_name,
            "quantization": self.config.quantization.value,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "is_running": self.is_running,
            "requests_processed": self.request_count,
            "total_tokens_generated": self.total_tokens_generated
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check engine health."""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "model": self.model_name,
            "timestamp": int(time.time())
        }


class VLLMEngineManager:
    """Manager for multiple vLLM engine instances."""
    
    def __init__(self):
        """Initialize the engine manager."""
        self.engines: Dict[str, VLLMEngine] = {}
        self.default_engine: Optional[VLLMEngine] = None
    
    async def load_model(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
        set_default: bool = True
    ) -> VLLMEngine:
        """Load a model with vLLM engine.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            set_default: Whether to set this as the default engine
            
        Returns:
            Initialized vLLM engine
        """
        if model_name in self.engines:
            logger.warning(f"Model {model_name} already loaded")
            return self.engines[model_name]
        
        if config is None:
            config = ModelConfig(model_name=model_name)
        
        engine = VLLMEngine(config)
        await engine.start()
        
        self.engines[model_name] = engine
        
        if set_default or self.default_engine is None:
            self.default_engine = engine
            logger.info(f"Set {model_name} as default engine")
        
        return engine
    
    async def unload_model(self, model_name: str):
        """Unload a model."""
        if model_name in self.engines:
            engine = self.engines[model_name]
            await engine.stop()
            del self.engines[model_name]
            
            if self.default_engine == engine:
                self.default_engine = next(iter(self.engines.values())) if self.engines else None
            
            logger.info(f"Unloaded model: {model_name}")
    
    def get_engine(self, model_name: Optional[str] = None) -> VLLMEngine:
        """Get an engine by model name or the default engine."""
        if model_name:
            if model_name not in self.engines:
                raise ValueError(f"Model {model_name} not loaded")
            return self.engines[model_name]
        
        if not self.default_engine:
            raise ValueError("No models loaded")
        
        return self.default_engine
    
    async def shutdown(self):
        """Shutdown all engines."""
        for model_name in list(self.engines.keys()):
            await self.unload_model(model_name)


# FastAPI Router for OpenAI-compatible API
def create_vllm_router(engine_manager: VLLMEngineManager) -> APIRouter:
    """Create FastAPI router with OpenAI-compatible endpoints.
    
    Args:
        engine_manager: VLLM engine manager instance
        
    Returns:
        Configured FastAPI router
    """
    router = APIRouter(prefix="/v1", tags=["vllm"])
    
    @router.get("/models")
    async def list_models():
        """List available models."""
        models = []
        for model_name, engine in engine_manager.engines.items():
            info = await engine.get_model_info()
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permission": [],
                "root": model_name,
                "parent": None,
                **info
            })
        
        return {
            "object": "list",
            "data": models
        }
    
    @router.get("/models/{model_name}")
    async def get_model(model_name: str):
        """Get model information."""
        if model_name not in engine_manager.engines:
            raise HTTPException(status_code=404, detail="Model not found")
        
        engine = engine_manager.engines[model_name]
        info = await engine.get_model_info()
        
        return {
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user",
            "permission": [],
            "root": model_name,
            "parent": None,
            **info
        }
    
    @router.post("/completions")
    async def create_completion(request: OpenAICompletionRequest):
        """Create a completion (OpenAI-compatible)."""
        try:
            engine = engine_manager.get_engine(request.model)
            
            # Convert OpenAI request to our format
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            
            results = []
            for prompt in prompts:
                response = await engine.generate(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                    stream=False
                )
                
                # Convert to OpenAI format
                results.append({
                    "id": response.id,
                    "object": "text_completion",
                    "created": response.created,
                    "model": response.model,
                    "choices": [{
                        "text": choice["text"],
                        "index": i,
                        "logprobs": None,
                        "finish_reason": choice["finish_reason"]
                    } for i, choice in enumerate(response.choices)],
                    "usage": response.usage
                })
            
            if len(results) == 1:
                return results[0]
            else:
                return {
                    "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "data": results
                }
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @router.post("/chat/completions")
    async def create_chat_completion(request: OpenAIChatCompletionRequest):
        """Create a chat completion (OpenAI-compatible)."""
        try:
            engine = engine_manager.get_engine(request.model)
            
            if request.stream:
                # Return streaming response
                async def generate_stream():
                    async for chunk in engine.chat_completion(
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                        stream=True
                    ):
                        yield chunk
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream"
                )
            else:
                # Return complete response
                response = await engine.chat_completion(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                    stream=False
                )
                return response
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        if not engine_manager.engines:
            return {"status": "no_models_loaded"}
        
        health_status = {}
        for model_name, engine in engine_manager.engines.items():
            health_status[model_name] = await engine.health_check()
        
        return {
            "status": "healthy",
            "models": health_status,
            "timestamp": int(time.time())
        }
    
    return router


# Integration with existing backend
def integrate_with_backend(app, model_configs: Optional[List[ModelConfig]] = None):
    """Integrate vLLM engine with existing FastAPI backend.
    
    Args:
        app: FastAPI application instance
        model_configs: List of model configurations to load at startup
    """
    # Create engine manager
    engine_manager = VLLMEngineManager()
    
    # Store in app state
    app.state.vllm_engine_manager = engine_manager
    
    # Add startup event to load models
    @app.on_event("startup")
    async def startup_vllm_engines():
        if model_configs:
            for config in model_configs:
                try:
                    await engine_manager.load_model(config.model_name, config)
                    logger.info(f"Loaded model: {config.model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {config.model_name}: {e}")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_vllm_engines():
        await engine_manager.shutdown()
    
    # Include the vLLM router
    router = create_vllm_router(engine_manager)
    app.include_router(router)
    
    return engine_manager


# Example usage
async def example_usage():
    """Example of how to use the vLLM engine."""
    # Configure model
    config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        quantization=QuantizationMethod.AWQ,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
    
    # Create and start engine
    engine = VLLMEngine(config)
    await engine.start()
    
    # Generate text
    response = await engine.generate(
        prompt="Explain quantum computing in simple terms:",
        max_tokens=256,
        temperature=0.7
    )
    
    print(f"Generated: {response.choices[0]['text']}")
    print(f"Tokens: {response.usage['total_tokens']}")
    
    # Stream generation
    print("\nStreaming output:")
    async for chunk in engine.generate(
        prompt="Write a short poem about AI:",
        max_tokens=100,
        stream=True
    ):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            data = json.loads(chunk[6:])
            if data["choices"][0]["text"]:
                print(data["choices"][0]["text"], end="", flush=True)
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    chat_response = await engine.chat_completion(messages=messages)
    print(f"\n\nChat response: {chat_response['choices'][0]['message']['content']}")
    
    # Get model info
    info = await engine.get_model_info()
    print(f"\nModel info: {json.dumps(info, indent=2)}")
    
    # Stop engine
    await engine.stop()


if __name__ == "__main__":
    # Run example if executed directly
    asyncio.run(example_usage())