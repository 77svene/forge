"""
studio/backend/core/inference/batch_scheduler.py
Production-grade model serving with vLLM integration for continuous batching,
PagedAttention, AWQ/GPTQ quantization, and OpenAI-compatible API endpoints.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Try to import vLLM - graceful fallback if not installed
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.outputs import RequestOutput
    from vllm.transformers_utils.tokenizer import get_tokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not installed. Install with: pip install vllm")

# Try to import TGI as fallback
try:
    from text_generation import AsyncClient
    TGI_AVAILABLE = True
except ImportError:
    TGI_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceBackend(str, Enum):
    """Supported inference backends."""
    VLLM = "vllm"
    TGI = "tgi"
    HUGGINGFACE = "huggingface"


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for model serving."""
    model_name_or_path: str
    backend: InferenceBackend = InferenceBackend.VLLM
    quantization: QuantizationMethod = QuantizationMethod.NONE
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    enable_prefix_caching: bool = True
    enable_cuda_graph: bool = True
    max_num_batched_tokens: int = 2560
    max_num_seqs: int = 256
    scheduler_policy: str = "fcfs"
    tgi_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""
    request_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: float = 0.0
    total_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class BatchScheduler:
    """
    Production-grade batch scheduler for model inference.
    
    Features:
    - Continuous batching with vLLM or TGI backend
    - PagedAttention for efficient memory management
    - AWQ/GPTQ quantization support
    - CUDA graph capture for reduced overhead
    - OpenAI-compatible API endpoints
    - Request queuing and prioritization
    - Real-time metrics collection
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.engine = None
        self.tokenizer = None
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.is_initialized = False
        self._background_tasks = set()
        
        if not VLLM_AVAILABLE and config.backend == InferenceBackend.VLLM:
            raise ImportError("vLLM is required for VLLM backend. Install with: pip install vllm")
        
        if not TGI_AVAILABLE and config.backend == InferenceBackend.TGI:
            raise ImportError("text-generation-client is required for TGI backend")
    
    async def initialize(self):
        """Initialize the inference engine."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.config.backend.value} backend for {self.config.model_name_or_path}")
        
        if self.config.backend == InferenceBackend.VLLM:
            await self._initialize_vllm()
        elif self.config.backend == InferenceBackend.TGI:
            await self._initialize_tgi()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        self.is_initialized = True
        logger.info(f"Model {self.config.model_name_or_path} initialized successfully")
    
    async def _initialize_vllm(self):
        """Initialize vLLM engine with optimized settings."""
        quantization = None
        if self.config.quantization != QuantizationMethod.NONE:
            quantization = self.config.quantization.value
        
        engine_args = AsyncEngineArgs(
            model=self.config.model_name_or_path,
            quantization=quantization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            enable_prefix_caching=self.config.enable_prefix_caching,
            enable_cuda_graph=self.config.enable_cuda_graph,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            scheduler_policy=self.config.scheduler_policy,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Load tokenizer
        self.tokenizer = get_tokenizer(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
    
    async def _initialize_tgi(self):
        """Initialize TGI client."""
        if not self.config.tgi_endpoint:
            raise ValueError("TGI endpoint URL is required for TGI backend")
        
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.engine = AsyncClient(
            base_url=self.config.tgi_endpoint,
            headers=headers,
            timeout=600
        )
    
    async def generate(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Generate text from prompt with continuous batching.
        
        Args:
            prompt: Input text prompt
            request_id: Optional request ID for tracking
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stop_sequences: Sequences to stop generation at
            stream: Whether to stream results
            
        Returns:
            Generated text or async iterator for streaming
        """
        if not self.is_initialized:
            await self.initialize()
        
        request_id = request_id or str(uuid.uuid4())
        metrics = RequestMetrics(request_id=request_id)
        self.active_requests[request_id] = metrics
        
        try:
            if self.config.backend == InferenceBackend.VLLM:
                return await self._generate_vllm(
                    prompt=prompt,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    metrics=metrics,
                    **kwargs
                )
            elif self.config.backend == InferenceBackend.TGI:
                return await self._generate_tgi(
                    prompt=prompt,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    metrics=metrics,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Generation failed for request {request_id}: {e}")
            raise
        finally:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def _generate_vllm(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Generate using vLLM backend."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
            **kwargs
        )
        
        if stream:
            return self._stream_vllm(prompt, request_id, sampling_params, metrics)
        else:
            return await self._complete_vllm(prompt, request_id, sampling_params, metrics)
    
    async def _complete_vllm(
        self,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams,
        metrics: RequestMetrics
    ) -> Dict[str, Any]:
        """Non-streaming vLLM generation."""
        start_time = time.time()
        metrics.start_time = start_time
        
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            raise RuntimeError("No output generated")
        
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - start_time
        
        # Extract token counts
        if final_output.outputs:
            output = final_output.outputs[0]
            metrics.prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
            metrics.completion_tokens = len(output.token_ids) if output.token_ids else 0
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
            
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.config.model_name_or_path,
                "choices": [{
                    "text": output.text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": output.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "metrics": {
                    "time_to_first_token": metrics.time_to_first_token,
                    "total_time": metrics.total_time,
                    "tokens_per_second": metrics.completion_tokens / metrics.total_time if metrics.total_time > 0 else 0,
                }
            }
        
        raise RuntimeError("No output generated")
    
    async def _stream_vllm(
        self,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams,
        metrics: RequestMetrics
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming vLLM generation."""
        start_time = time.time()
        metrics.start_time = start_time
        first_token_received = False
        
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )
        
        async for request_output in results_generator:
            if not first_token_received and request_output.outputs:
                metrics.time_to_first_token = time.time() - start_time
                first_token_received = True
            
            if request_output.outputs:
                output = request_output.outputs[0]
                chunk = {
                    "id": request_id,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": self.config.model_name_or_path,
                    "choices": [{
                        "text": output.text[-1] if output.text else "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": output.finish_reason,
                    }],
                }
                yield chunk
        
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - start_time
    
    async def _generate_tgi(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Generate using TGI backend."""
        start_time = time.time()
        metrics.start_time = start_time
        
        if stream:
            return self._stream_tgi(
                prompt=prompt,
                request_id=request_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                metrics=metrics,
                **kwargs
            )
        else:
            response = await self.engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences or [],
                **kwargs
            )
            
            metrics.end_time = time.time()
            metrics.total_time = metrics.end_time - start_time
            metrics.prompt_tokens = response.details.generated_tokens
            metrics.completion_tokens = response.details.generated_tokens
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
            
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.config.model_name_or_path,
                "choices": [{
                    "text": response.generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "metrics": {
                    "time_to_first_token": metrics.time_to_first_token,
                    "total_time": metrics.total_time,
                    "tokens_per_second": metrics.completion_tokens / metrics.total_time if metrics.total_time > 0 else 0,
                }
            }
    
    async def _stream_tgi(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
        metrics: RequestMetrics,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming TGI generation."""
        start_time = time.time()
        metrics.start_time = start_time
        first_token_received = False
        
        async for response in self.engine.generate_stream(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or [],
            **kwargs
        ):
            if not first_token_received:
                metrics.time_to_first_token = time.time() - start_time
                first_token_received = True
            
            if response.token:
                chunk = {
                    "id": request_id,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": self.config.model_name_or_path,
                    "choices": [{
                        "text": response.token.text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop" if response.token.special else None,
                    }],
                }
                yield chunk
        
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - start_time
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current scheduler metrics."""
        return {
            "active_requests": len(self.active_requests),
            "queue_size": self.request_queue.qsize(),
            "model": self.config.model_name_or_path,
            "backend": self.config.backend.value,
            "quantization": self.config.quantization.value,
            "is_initialized": self.is_initialized,
        }
    
    async def shutdown(self):
        """Shutdown the scheduler gracefully."""
        logger.info("Shutting down batch scheduler...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for active requests to complete
        timeout = 30
        start_time = time.time()
        while self.active_requests and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if self.active_requests:
            logger.warning(f"Forcefully terminating {len(self.active_requests)} active requests")
        
        self.is_initialized = False
        logger.info("Batch scheduler shutdown complete")


# Pydantic models for OpenAI-compatible API
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = Field(..., description="Model name")
    prompt: Union[str, List[str]] = Field(..., description="Input prompt")
    suffix: Optional[str] = Field(None, description="Suffix for completion")
    max_tokens: int = Field(100, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    n: int = Field(1, ge=1, le=10, description="Number of completions")
    stream: bool = Field(False, description="Whether to stream results")
    logprobs: Optional[int] = Field(None, ge=0, le=5, description="Log probabilities")
    echo: bool = Field(False, description="Echo prompt in completion")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    best_of: int = Field(1, ge=1, description="Best of n completions")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model name")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    n: int = Field(1, ge=1, le=10, description="Number of completions")
    stream: bool = Field(False, description="Whether to stream results")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    metrics: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    metrics: Optional[Dict[str, Any]] = None


def create_inference_router(
    scheduler: BatchScheduler,
    prefix: str = "/v1"
) -> APIRouter:
    """
    Create FastAPI router with OpenAI-compatible endpoints.
    
    Args:
        scheduler: BatchScheduler instance
        prefix: URL prefix for endpoints
        
    Returns:
        Configured FastAPI router
    """
    router = APIRouter(prefix=prefix, tags=["inference"])
    
    @router.post("/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        """OpenAI-compatible text completion endpoint."""
        if not scheduler.is_initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        try:
            # Handle multiple prompts (simplified - only handles first)
            prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
            
            result = await scheduler.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                stream=False,
                repetition_penalty=1.0 + (request.frequency_penalty + request.presence_penalty) / 2.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint."""
        if not scheduler.is_initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        try:
            # Convert chat messages to prompt
            prompt = _convert_messages_to_prompt(request.messages, scheduler.tokenizer)
            
            result = await scheduler.generate(
                prompt=prompt,
                max_tokens=request.max_tokens or 100,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                stream=False,
                repetition_penalty=1.0 + (request.frequency_penalty + request.presence_penalty) / 2.0
            )
            
            # Convert to chat format
            chat_result = {
                "id": result["id"],
                "object": "chat.completion",
                "created": result["created"],
                "model": result["model"],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["choices"][0]["text"],
                    },
                    "finish_reason": result["choices"][0]["finish_reason"],
                }],
                "usage": result["usage"],
                "metrics": result.get("metrics"),
            }
            
            return chat_result
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/chat/completions/stream")
    async def create_chat_completion_stream(request: ChatCompletionRequest):
        """Streaming chat completion endpoint."""
        if not scheduler.is_initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        try:
            prompt = _convert_messages_to_prompt(request.messages, scheduler.tokenizer)
            
            async def generate_stream():
                async for chunk in await scheduler.generate(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 100,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                    stream=True,
                    repetition_penalty=1.0 + (request.frequency_penalty + request.presence_penalty) / 2.0
                ):
                    # Convert to SSE format
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models")
    async def list_models():
        """List available models."""
        return {
            "data": [{
                "id": scheduler.config.model_name_or_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sovereign",
                "permission": [],
                "root": scheduler.config.model_name_or_path,
                "parent": None,
            }]
        }
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        metrics = await scheduler.get_metrics()
        return {
            "status": "healthy" if scheduler.is_initialized else "initializing",
            "model": scheduler.config.model_name_or_path,
            "backend": scheduler.config.backend.value,
            **metrics
        }
    
    return router


def _convert_messages_to_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Any
) -> str:
    """Convert chat messages to model prompt format."""
    if hasattr(tokenizer, "apply_chat_template"):
        # Use tokenizer's chat template if available
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Fallback: simple concatenation
    prompt_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(f"{role}: {content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


# Import json for SSE streaming
import json


# Factory function for easy integration
def create_batch_scheduler(
    model_name_or_path: str,
    backend: str = "vllm",
    quantization: str = "none",
    **kwargs
) -> BatchScheduler:
    """
    Create and configure a batch scheduler.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        backend: Inference backend (vllm, tgi, huggingface)
        quantization: Quantization method (awq, gptq, squeezellm, none)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BatchScheduler instance
    """
    config = ModelConfig(
        model_name_or_path=model_name_or_path,
        backend=InferenceBackend(backend),
        quantization=QuantizationMethod(quantization),
        **kwargs
    )
    return BatchScheduler(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_scheduler():
        # Example configuration
        scheduler = create_batch_scheduler(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            backend="vllm",
            quantization="awq",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        
        await scheduler.initialize()
        
        # Test generation
        result = await scheduler.generate(
            prompt="Explain quantum computing in simple terms:",
            max_tokens=100,
            temperature=0.7,
        )
        
        print("Generation result:", result)
        
        # Get metrics
        metrics = await scheduler.get_metrics()
        print("Scheduler metrics:", metrics)
        
        await scheduler.shutdown()
    
    # Run test
    asyncio.run(test_scheduler())