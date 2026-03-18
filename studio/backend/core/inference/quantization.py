"""
Production-grade model serving with vLLM integration.
Supports continuous batching, PagedAttention, AWQ/GPTQ quantization, and OpenAI-compatible API.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, AsyncIterator

import torch
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Conditional imports for vLLM
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.transformers_utils.tokenizer import get_tokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    AsyncLLMEngine = None
    SamplingParams = None

from studio.backend.auth.authentication import get_current_user
from studio.backend.core.data_recipe.jobs.types import ModelConfig

logger = logging.getLogger(__name__)

# Quantization methods supported
class QuantizationMethod(str, Enum):
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLN = "squeezelln"

# Model precision options
class ModelPrecision(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    AUTO = "auto"

@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_path: str
    quantization: QuantizationMethod = QuantizationMethod.NONE
    precision: ModelPrecision = ModelPrecision.AUTO
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    use_v2_block_manager: bool = True
    download_dir: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    quantization_param_path: Optional[str] = None
    device: str = "auto"
    
    def to_engine_args(self) -> Dict[str, Any]:
        """Convert to vLLM engine arguments."""
        args = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "use_v2_block_manager": self.use_v2_block_manager,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "device": self.device,
        }
        
        if self.max_model_len:
            args["max_model_len"] = self.max_model_len
        
        if self.quantization != QuantizationMethod.NONE:
            args["quantization"] = self.quantization.value
            if self.quantization_param_path:
                args["quantization_param_path"] = self.quantization_param_path
        
        if self.download_dir:
            args["download_dir"] = self.download_dir
        
        return args

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = Field(..., description="Model ID")
    prompt: Union[str, List[str]] = Field(..., description="Input prompt(s)")
    suffix: Optional[str] = Field(None, description="Suffix for completion")
    max_tokens: int = Field(16, description="Maximum tokens to generate", ge=1)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Top-p sampling", ge=0.0, le=1.0)
    n: int = Field(1, description="Number of completions", ge=1)
    stream: bool = Field(False, description="Stream partial progress")
    logprobs: Optional[int] = Field(None, description="Include log probabilities")
    echo: bool = Field(False, description="Echo prompt in completion")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: float = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(None, description="Best of n completions")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model ID")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Top-p sampling", ge=0.0, le=1.0)
    n: int = Field(1, description="Number of completions", ge=1)
    stream: bool = Field(False, description="Stream partial progress")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens", ge=1)
    presence_penalty: float = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")

class ModelEngine:
    """Production-grade model serving engine using vLLM."""
    
    def __init__(self, config: ServingConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for production serving. Install with: pip install vllm")
        
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None
        self._initialized = False
        self._request_count = 0
        self._total_tokens_generated = 0
        self._start_time = time.time()
        
    async def initialize(self):
        """Initialize the vLLM engine."""
        if self._initialized:
            return
        
        logger.info(f"Initializing vLLM engine with model: {self.config.model_path}")
        logger.info(f"Quantization: {self.config.quantization.value}")
        logger.info(f"Tensor parallel size: {self.config.tensor_parallel_size}")
        
        try:
            engine_args = AsyncEngineArgs(**self.config.to_engine_args())
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Initialize tokenizer
            self.tokenizer = get_tokenizer(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self._initialized = True
            logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def generate_completion(self, request: CompletionRequest) -> Dict[str, Any]:
        """Generate completion using vLLM."""
        if not self._initialized:
            await self.initialize()
        
        self._request_count += 1
        
        # Convert OpenAI request to vLLM sampling params
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            stop=request.stop,
        )
        
        # Generate request ID
        request_id = f"cmpl-{uuid.uuid4().hex}"
        
        try:
            # Start generation
            results_generator = self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Collect all results
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if not final_output:
                raise HTTPException(status_code=500, detail="Generation failed")
            
            # Convert to OpenAI format
            choices = []
            for i, output in enumerate(final_output.outputs):
                choice = {
                    "index": i,
                    "text": output.text,
                    "logprobs": self._convert_logprobs(output.logprobs) if output.logprobs else None,
                    "finish_reason": output.finish_reason or "length",
                }
                choices.append(choice)
            
            # Update stats
            self._total_tokens_generated += sum(
                len(output.token_ids) for output in final_output.outputs
            )
            
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": len(final_output.prompt_token_ids),
                    "completion_tokens": sum(len(output.token_ids) for output in final_output.outputs),
                    "total_tokens": len(final_output.prompt_token_ids) + 
                                   sum(len(output.token_ids) for output in final_output.outputs),
                }
            }
            
        except Exception as e:
            logger.error(f"Completion generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Generate chat completion using vLLM."""
        if not self._initialized:
            await self.initialize()
        
        # Convert chat messages to prompt
        prompt = self._convert_messages_to_prompt(request.messages)
        
        # Create completion request
        completion_request = CompletionRequest(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            user=request.user,
        )
        
        result = await self.generate_completion(completion_request)
        
        # Convert to chat format
        result["object"] = "chat.completion"
        for choice in result["choices"]:
            choice["message"] = {
                "role": "assistant",
                "content": choice["text"],
            }
            del choice["text"]
        
        return result
    
    async def stream_completion(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Stream completion tokens."""
        if not self._initialized:
            await self.initialize()
        
        self._request_count += 1
        request_id = f"cmpl-{uuid.uuid4().hex}"
        
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            stop=request.stop,
        )
        
        try:
            results_generator = self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            async for request_output in results_generator:
                for output in request_output.outputs:
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "text": output.text[-1] if output.text else "",
                            "logprobs": self._convert_logprobs(output.logprobs) if output.logprobs else None,
                            "finish_reason": output.finish_reason,
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        prompt = self._convert_messages_to_prompt(request.messages)
        
        completion_request = CompletionRequest(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=True,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            user=request.user,
        )
        
        async for chunk in self.stream_completion(completion_request):
            # Convert to chat format
            if chunk.startswith("data: [DONE]"):
                yield chunk
            elif chunk.startswith("data:"):
                data = json.loads(chunk[5:].strip())
                data["object"] = "chat.completion.chunk"
                for choice in data["choices"]:
                    choice["delta"] = {
                        "content": choice["text"],
                    }
                    del choice["text"]
                yield f"data: {json.dumps(data)}\n\n"
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to model prompt format."""
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}</s>")
        
        # Add assistant prompt for generation
        prompt_parts.append("<|assistant|>\n")
        return "\n".join(prompt_parts)
    
    def _convert_logprobs(self, logprobs) -> Optional[Dict]:
        """Convert vLLM logprobs to OpenAI format."""
        if not logprobs:
            return None
        
        try:
            return {
                "tokens": logprobs.tokens,
                "token_logprobs": logprobs.token_logprobs,
                "top_logprobs": [
                    {str(k): v for k, v in top.items()} 
                    for top in logprobs.top_logprobs
                ] if logprobs.top_logprobs else None,
                "text_offset": logprobs.text_offset,
            }
        except Exception:
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "model": self.config.model_path,
            "quantization": self.config.quantization.value,
            "precision": self.config.precision.value,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "requests_processed": self._request_count,
            "total_tokens_generated": self._total_tokens_generated,
            "uptime_seconds": time.time() - self._start_time,
            "initialized": self._initialized,
        }
        
        if self.engine:
            try:
                # Get scheduler stats if available
                if hasattr(self.engine, 'scheduler'):
                    scheduler_stats = self.engine.scheduler.get_stats()
                    stats.update({
                        "running_requests": scheduler_stats.running,
                        "waiting_requests": scheduler_stats.waiting,
                        "swapped_requests": scheduler_stats.swapped,
                    })
            except Exception:
                pass
        
        return stats
    
    async def shutdown(self):
        """Shutdown the engine gracefully."""
        if self.engine:
            logger.info("Shutting down vLLM engine")
            # vLLM doesn't have explicit shutdown, but we can clean up
            self.engine = None
            self._initialized = False

class ModelManager:
    """Manages multiple model engines."""
    
    def __init__(self):
        self.engines: Dict[str, ModelEngine] = {}
        self.default_model: Optional[str] = None
    
    async def load_model(self, model_id: str, config: ServingConfig) -> ModelEngine:
        """Load a model with given configuration."""
        if model_id in self.engines:
            logger.warning(f"Model {model_id} already loaded")
            return self.engines[model_id]
        
        engine = ModelEngine(config)
        await engine.initialize()
        
        self.engines[model_id] = engine
        if not self.default_model:
            self.default_model = model_id
        
        logger.info(f"Model {model_id} loaded successfully")
        return engine
    
    async def unload_model(self, model_id: str):
        """Unload a model."""
        if model_id not in self.engines:
            raise KeyError(f"Model {model_id} not found")
        
        await self.engines[model_id].shutdown()
        del self.engines[model_id]
        
        if self.default_model == model_id:
            self.default_model = next(iter(self.engines), None)
        
        logger.info(f"Model {model_id} unloaded")
    
    def get_engine(self, model_id: Optional[str] = None) -> ModelEngine:
        """Get model engine by ID or default."""
        model_id = model_id or self.default_model
        if not model_id or model_id not in self.engines:
            raise KeyError(f"Model {model_id} not found")
        return self.engines[model_id]
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        models = []
        for model_id, engine in self.engines.items():
            stats = await engine.get_stats()
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(engine._start_time),
                "owned_by": "studio",
                "permission": [],
                "root": engine.config.model_path,
                "parent": None,
                "stats": stats,
            })
        return models

# Global model manager
model_manager = ModelManager()

# FastAPI Router
router = APIRouter(prefix="/v1", tags=["inference"])

@router.on_event("startup")
async def startup_event():
    """Initialize default model on startup."""
    # Load default model from config if available
    default_model_path = "meta-llama/Llama-2-7b-chat-hf"  # Default model
    try:
        config = ServingConfig(
            model_path=default_model_path,
            quantization=QuantizationMethod.AWQ,
            precision=ModelPrecision.FLOAT16,
            tensor_parallel_size=1 if torch.cuda.device_count() == 1 else torch.cuda.device_count(),
        )
        await model_manager.load_model("default", config)
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown all models on app shutdown."""
    for model_id in list(model_manager.engines.keys()):
        await model_manager.unload_model(model_id)

@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models."""
    models = await model_manager.list_models()
    return {"data": models}

@router.get("/models/{model_id}")
async def get_model(model_id: str, current_user: dict = Depends(get_current_user)):
    """Get model information."""
    try:
        engine = model_manager.get_engine(model_id)
        stats = await engine.get_stats()
        return {
            "id": model_id,
            "object": "model",
            "created": int(engine._start_time),
            "owned_by": "studio",
            "permission": [],
            "root": engine.config.model_path,
            "parent": None,
            "stats": stats,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

@router.post("/completions")
async def create_completion(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create completion (OpenAI-compatible)."""
    try:
        engine = model_manager.get_engine(request.model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    if request.stream:
        return StreamingResponse(
            engine.stream_completion(request),
            media_type="text/event-stream"
        )
    else:
        return await engine.generate_completion(request)

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create chat completion (OpenAI-compatible)."""
    try:
        engine = model_manager.get_engine(request.model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    if request.stream:
        return StreamingResponse(
            engine.stream_chat_completion(request),
            media_type="text/event-stream"
        )
    else:
        return await engine.generate_chat_completion(request)

@router.post("/models/load")
async def load_model(
    model_id: str,
    config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Load a new model."""
    try:
        serving_config = ServingConfig(**config)
        engine = await model_manager.load_model(model_id, serving_config)
        return {"status": "success", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str, current_user: dict = Depends(get_current_user)):
    """Unload a model."""
    try:
        await model_manager.unload_model(model_id)
        return {"status": "success", "model_id": model_id}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "models_loaded": len(model_manager.engines),
        "vllm_available": VLLM_AVAILABLE,
    }

# Utility functions for quantization
def quantize_model(
    model_path: str,
    output_dir: str,
    quantization_method: QuantizationMethod,
    dataset: Optional[str] = None,
    group_size: int = 128,
    bits: int = 4,
    **kwargs
) -> str:
    """
    Quantize a model using specified method.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Output directory for quantized model
        quantization_method: Quantization method (awq, gptq, squeezelln)
        dataset: Dataset for calibration (optional)
        group_size: Group size for quantization
        bits: Number of bits for quantization
    
    Returns:
        Path to quantized model
    """
    if quantization_method == QuantizationMethod.AWQ:
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
            
            model = AutoAWQForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": bits,
                "version": "GEMM"
            }
            
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=dataset or "pileval",
                split="train",
                text_column="text"
            )
            
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            return output_dir
            
        except ImportError:
            raise ImportError("AWQ quantization requires: pip install autoawq")
    
    elif quantization_method == QuantizationMethod.GPTQ:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=True,
            )
            
            model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                quantize_config=quantize_config,
                trust_remote_code=True
            )
            
            # Load calibration dataset
            if dataset:
                from datasets import load_dataset
                data = load_dataset(dataset, split="train")
                examples = [tokenizer(example["text"]) for example in data.select(range(128))]
            else:
                # Use default calibration
                examples = [tokenizer("The quick brown fox jumps over the lazy dog.") for _ in range(128)]
            
            model.quantize(examples)
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            return output_dir
            
        except ImportError:
            raise ImportError("GPTQ quantization requires: pip install auto-gptq")
    
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}")

# Integration with existing data recipes
def create_serving_config_from_recipe(recipe_config: ModelConfig) -> ServingConfig:
    """Create serving config from data recipe model config."""
    quantization = QuantizationMethod.NONE
    if hasattr(recipe_config, 'quantization'):
        if recipe_config.quantization == "awq":
            quantization = QuantizationMethod.AWQ
        elif recipe_config.quantization == "gptq":
            quantization = QuantizationMethod.GPTQ
    
    precision = ModelPrecision.AUTO
    if hasattr(recipe_config, 'precision'):
        if recipe_config.precision == "float16":
            precision = ModelPrecision.FLOAT16
        elif recipe_config.precision == "bfloat16":
            precision = ModelPrecision.BFLOAT16
    
    return ServingConfig(
        model_path=recipe_config.model_name_or_path,
        quantization=quantization,
        precision=precision,
        tensor_parallel_size=getattr(recipe_config, 'tensor_parallel_size', 1),
        trust_remote_code=getattr(recipe_config, 'trust_remote_code', False),
    )

# Export main components
__all__ = [
    "router",
    "ModelEngine",
    "ModelManager",
    "ServingConfig",
    "QuantizationMethod",
    "ModelPrecision",
    "model_manager",
    "quantize_model",
    "create_serving_config_from_recipe",
]