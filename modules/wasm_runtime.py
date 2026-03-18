"""
WebAssembly Upscaling Pipeline for forge
Offloads upscaling operations (LDSR, ESRGAN) to WebAssembly for CPU parallelization
and reduced GPU memory pressure. Enables 4x+ upscaling on systems with limited VRAM.
"""

import os
import sys
import time
import asyncio
import threading
import numpy as np
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
import json

try:
    from PIL import Image
except ImportError:
    Image = None

# WebAssembly runtime dependencies
try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False

try:
    import wasmer
    WASMER_AVAILABLE = True
except ImportError:
    WASMER_AVAILABLE = False

# WebGPU support for browser environments
try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False

# Import existing modules for integration
from modules import devices, shared, images, sd_models
from modules.shared import opts, cmd_opts
from modules.upscaler import Upscaler, UpscalerData
from modules.devices import torch_gc, get_optimal_device


class UpscaleBackend(Enum):
    """Available upscaling backend options"""
    GPU = "gpu"
    CPU = "cpu"
    WASM = "wasm"
    WEBGPU = "webgpu"
    AUTO = "auto"


@dataclass
class UpscaleJob:
    """Represents a single upscaling job"""
    image: np.ndarray
    scale_factor: int
    model_name: str
    backend: UpscaleBackend
    priority: int = 0
    callback: Optional[callable] = None
    job_id: str = ""
    created_at: float = 0.0
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"upscale_{int(time.time()*1000)}"
        if not self.created_at:
            self.created_at = time.time()


class WasmModule:
    """Represents a compiled WebAssembly upscaling module"""
    
    def __init__(self, wasm_path: str, model_path: Optional[str] = None):
        self.wasm_path = Path(wasm_path)
        self.model_path = Path(model_path) if model_path else None
        self.instance = None
        self.memory = None
        self.exports = {}
        self.model_loaded = False
        self.module_size = 0
        self._lock = threading.Lock()
        
    def load(self, engine: str = "wasmtime") -> bool:
        """Load the WASM module with specified engine"""
        if not self.wasm_path.exists():
            raise FileNotFoundError(f"WASM module not found: {self.wasm_path}")
            
        try:
            if engine == "wasmtime" and WASMTIME_AVAILABLE:
                return self._load_wasmtime()
            elif engine == "wasmer" and WASMER_AVAILABLE:
                return self._load_wasmer()
            else:
                # Try available engines
                if WASMTIME_AVAILABLE:
                    return self._load_wasmtime()
                elif WASMER_AVAILABLE:
                    return self._load_wasmer()
                else:
                    raise RuntimeError("No WASM runtime available")
        except Exception as e:
            print(f"Error loading WASM module: {e}")
            return False
    
    def _load_wasmtime(self) -> bool:
        """Load using wasmtime engine"""
        try:
            engine = wasmtime.Engine()
            module = wasmtime.Module.from_file(engine, str(self.wasm_path))
            store = wasmtime.Store(engine)
            linker = wasmtime.Linker(engine)
            
            # Define memory imports if needed
            memory = wasmtime.Memory(store, wasmtime.MemoryType(256, 65536))
            linker.define(store, "env", "memory", memory)
            
            instance = linker.instantiate(store, module)
            self.instance = instance
            self.memory = memory
            self.exports = instance.exports(store)
            self.module_size = self.wasm_path.stat().st_size
            return True
        except Exception as e:
            print(f"Wasmtime loading failed: {e}")
            return False
    
    def _load_wasmer(self) -> bool:
        """Load using wasmer engine"""
        try:
            from wasmer import engine, Store, Module, Instance, Memory, MemoryType
            
            store = Store()
            module = Module.from_file(store, str(self.wasm_path))
            
            # Create memory
            memory = Memory(store, MemoryType(256, 65536))
            
            # Import object with memory
            import_object = {"env": {"memory": memory}}
            instance = Instance(module, import_object)
            
            self.instance = instance
            self.memory = memory
            self.exports = instance.exports
            self.module_size = self.wasm_path.stat().st_size
            return True
        except Exception as e:
            print(f"Wasmer loading failed: {e}")
            return False
    
    def upscale(self, image: np.ndarray, scale_factor: int = 4) -> Optional[np.ndarray]:
        """Perform upscaling using the WASM module"""
        if not self.instance:
            raise RuntimeError("WASM module not loaded")
            
        with self._lock:
            try:
                # Get export functions
                if "upscale" not in self.exports:
                    raise RuntimeError("WASM module missing 'upscale' export")
                
                upscale_func = self.exports["upscale"]
                alloc_func = self.exports.get("allocate_buffer")
                free_func = self.exports.get("free_buffer")
                
                # Prepare image data
                if image.ndim == 2:
                    # Grayscale
                    h, w = image.shape
                    c = 1
                    img_data = image.tobytes()
                else:
                    # RGB/RGBA
                    h, w, c = image.shape
                    img_data = image.tobytes()
                
                # Allocate memory in WASM if allocation functions available
                if alloc_func and free_func:
                    input_ptr = alloc_func(len(img_data))
                    output_size = h * w * c * scale_factor * scale_factor
                    output_ptr = alloc_func(output_size)
                else:
                    # Use fixed memory locations (unsafe, for simple modules)
                    input_ptr = 0
                    output_ptr = h * w * c
                
                # Write image data to WASM memory
                if hasattr(self.memory, 'data_ptr'):
                    # Wasmtime style
                    memory_data = self.memory.data_ptr
                    memory_data[input_ptr:input_ptr + len(img_data)] = img_data
                else:
                    # Wasmer style
                    self.memory.write(input_ptr, img_data)
                
                # Call upscale function
                result = upscale_func(
                    input_ptr,
                    output_ptr,
                    w,
                    h,
                    c,
                    scale_factor
                )
                
                # Read result from WASM memory
                output_size = h * w * c * scale_factor * scale_factor
                if hasattr(self.memory, 'data_ptr'):
                    output_data = bytes(self.memory.data_ptr[output_ptr:output_ptr + output_size])
                else:
                    output_data = self.memory.read(output_ptr, output_size)
                
                # Convert to numpy array
                if c == 1:
                    result_array = np.frombuffer(output_data, dtype=np.uint8).reshape(
                        h * scale_factor, w * scale_factor
                    )
                else:
                    result_array = np.frombuffer(output_data, dtype=np.uint8).reshape(
                        h * scale_factor, w * scale_factor, c
                    )
                
                # Free allocated memory
                if alloc_func and free_func:
                    free_func(input_ptr)
                    free_func(output_ptr)
                
                return result_array
                
            except Exception as e:
                print(f"WASM upscale error: {e}")
                return None


class WasmUpscalePipeline:
    """Main WebAssembly upscaling pipeline manager"""
    
    def __init__(self):
        self.modules: Dict[str, WasmModule] = {}
        self.job_queue = Queue()
        self.worker_threads: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.backend_priority = [UpscaleBackend.GPU, UpscaleBackend.WASM, UpscaleBackend.CPU]
        self.device_stats = {}
        self.wasm_dir = Path(__file__).parent.parent / "wasm" / "upscalers"
        self.wasm_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            "jobs_processed": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "gpu_time": 0.0,
            "wasm_time": 0.0,
            "cpu_time": 0.0
        }
        
        # Initialize
        self._check_dependencies()
        self._load_available_modules()
        self._start_workers()
    
    def _check_dependencies(self):
        """Check for required dependencies"""
        if not WASMTIME_AVAILABLE and not WASMER_AVAILABLE:
            print("Warning: No WASM runtime available. Install wasmtime or wasmer.")
        
        if Image is None:
            print("Warning: PIL not available. Some image operations may fail.")
    
    def _load_available_modules(self):
        """Load all available WASM upscaler modules"""
        if not self.wasm_dir.exists():
            print(f"WASM directory not found: {self.wasm_dir}")
            return
        
        for wasm_file in self.wasm_dir.glob("*.wasm"):
            module_name = wasm_file.stem
            model_file = self.wasm_dir / f"{module_name}.bin"
            
            try:
                module = WasmModule(
                    str(wasm_file),
                    str(model_file) if model_file.exists() else None
                )
                
                if module.load():
                    self.modules[module_name] = module
                    print(f"Loaded WASM module: {module_name}")
                else:
                    print(f"Failed to load WASM module: {module_name}")
            except Exception as e:
                print(f"Error loading WASM module {module_name}: {e}")
    
    def _start_workers(self):
        """Start background worker threads"""
        self.running = True
        for i in range(2):  # 2 worker threads
            worker = threading.Thread(
                target=self._process_jobs,
                name=f"WasmUpscaleWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def _process_jobs(self):
        """Process upscaling jobs from the queue"""
        while self.running:
            try:
                job = self.job_queue.get(timeout=1.0)
                if job is None:
                    break
                
                start_time = time.time()
                result = self._execute_job(job)
                elapsed = time.time() - start_time
                
                # Update stats
                self.stats["jobs_processed"] += 1
                self.stats["total_time"] += elapsed
                self.stats["avg_time"] = self.stats["total_time"] / self.stats["jobs_processed"]
                
                if job.backend == UpscaleBackend.WASM:
                    self.stats["wasm_time"] += elapsed
                elif job.backend == UpscaleBackend.GPU:
                    self.stats["gpu_time"] += elapsed
                else:
                    self.stats["cpu_time"] += elapsed
                
                # Call callback with result
                if job.callback and result is not None:
                    job.callback(result, job.job_id, elapsed)
                
                self.job_queue.task_done()
                
            except Exception as e:
                print(f"Error processing upscale job: {e}")
    
    def _execute_job(self, job: UpscaleJob) -> Optional[np.ndarray]:
        """Execute a single upscaling job"""
        try:
            if job.backend == UpscaleBackend.WASM:
                return self._upscale_wasm(job)
            elif job.backend == UpscaleBackend.GPU:
                return self._upscale_gpu(job)
            elif job.backend == UpscaleBackend.CPU:
                return self._upscale_cpu(job)
            elif job.backend == UpscaleBackend.AUTO:
                return self._upscale_auto(job)
            else:
                print(f"Unknown backend: {job.backend}")
                return None
        except Exception as e:
            print(f"Upscale execution error: {e}")
            return None
    
    def _upscale_wasm(self, job: UpscaleJob) -> Optional[np.ndarray]:
        """Upscale using WASM backend"""
        module_name = job.model_name.lower().replace(" ", "_")
        
        if module_name not in self.modules:
            # Try to find a matching module
            for name in self.modules:
                if module_name in name or name in module_name:
                    module_name = name
                    break
            else:
                print(f"No WASM module found for: {job.model_name}")
                return None
        
        module = self.modules[module_name]
        return module.upscale(job.image, job.scale_factor)
    
    def _upscale_gpu(self, job: UpscaleJob) -> Optional[np.ndarray]:
        """Upscale using GPU backend (existing implementation)"""
        # This would integrate with existing upscalers
        # For now, return a placeholder
        try:
            from modules.upscaler import run_upscaler
            # Convert numpy to PIL if needed
            if Image is not None:
                if job.image.ndim == 2:
                    pil_image = Image.fromarray(job.image, mode='L')
                else:
                    pil_image = Image.fromarray(job.image, mode='RGB')
                
                # Use existing upscaler
                result = run_upscaler(
                    pil_image,
                    upscaler_name=job.model_name,
                    upscale_factor=job.scale_factor
                )
                
                if result:
                    return np.array(result)
            return None
        except Exception as e:
            print(f"GPU upscale error: {e}")
            return None
    
    def _upscale_cpu(self, job: UpscaleJob) -> Optional[np.ndarray]:
        """Upscale using CPU fallback (basic interpolation)"""
        try:
            from PIL import Image
            
            if job.image.ndim == 2:
                pil_image = Image.fromarray(job.image, mode='L')
            else:
                pil_image = Image.fromarray(job.image, mode='RGB')
            
            new_size = (
                job.image.shape[1] * job.scale_factor,
                job.image.shape[0] * job.scale_factor
            )
            
            # Use Lanczos resampling for better quality
            result = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(result)
        except Exception as e:
            print(f"CPU upscale error: {e}")
            return None
    
    def _upscale_auto(self, job: UpscaleJob) -> Optional[np.ndarray]:
        """Automatically select best backend based on resources"""
        # Check GPU memory
        gpu_available = False
        try:
            if devices.device.type == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_free = gpu_memory - gpu_allocated
                
                # Estimate memory needed (rough)
                h, w = job.image.shape[:2]
                bytes_needed = h * w * 3 * job.scale_factor * job.scale_factor * 4  # float32
                
                gpu_available = gpu_free > bytes_needed * 1.5  # 50% safety margin
        except:
            gpu_available = False
        
        # Check WASM availability
        wasm_available = len(self.modules) > 0 and (
            WASMTIME_AVAILABLE or WASMER_AVAILABLE
        )
        
        # Decision logic
        if gpu_available and job.scale_factor <= 2:
            # Small scale, use GPU
            job.backend = UpscaleBackend.GPU
        elif wasm_available and job.scale_factor >= 4:
            # Large scale, use WASM to save GPU memory
            job.backend = UpscaleBackend.WASM
        elif gpu_available:
            # GPU available, use it
            job.backend = UpscaleBackend.GPU
        elif wasm_available:
            # Use WASM
            job.backend = UpscaleBackend.WASM
        else:
            # Fallback to CPU
            job.backend = UpscaleBackend.CPU
        
        return self._execute_job(job)
    
    def submit_job(
        self,
        image: np.ndarray,
        scale_factor: int,
        model_name: str,
        backend: UpscaleBackend = UpscaleBackend.AUTO,
        priority: int = 0,
        callback: Optional[callable] = None
    ) -> str:
        """Submit an upscaling job to the pipeline"""
        job = UpscaleJob(
            image=image,
            scale_factor=scale_factor,
            model_name=model_name,
            backend=backend,
            priority=priority,
            callback=callback
        )
        
        self.job_queue.put(job)
        return job.job_id
    
    def upscale_immediate(
        self,
        image: np.ndarray,
        scale_factor: int,
        model_name: str,
        backend: UpscaleBackend = UpscaleBackend.AUTO
    ) -> Optional[np.ndarray]:
        """Perform upscaling immediately (blocking)"""
        job = UpscaleJob(
            image=image,
            scale_factor=scale_factor,
            model_name=model_name,
            backend=backend
        )
        
        return self._execute_job(job)
    
    def get_available_models(self) -> List[str]:
        """Get list of available WASM upscaling models"""
        return list(self.modules.keys())
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends"""
        status = {
            "gpu": {
                "available": devices.device.type == 'cuda',
                "device": str(devices.device),
                "memory_allocated": 0,
                "memory_reserved": 0
            },
            "wasm": {
                "available": len(self.modules) > 0,
                "runtime": "wasmtime" if WASMTIME_AVAILABLE else "wasmer" if WASMER_AVAILABLE else "none",
                "modules_loaded": len(self.modules),
                "module_names": list(self.modules.keys())
            },
            "cpu": {
                "available": True,
                "cores": os.cpu_count()
            },
            "webgpu": {
                "available": WGPU_AVAILABLE,
                "supported": False  # Would need browser environment
            }
        }
        
        # Add GPU memory info if available
        try:
            if devices.device.type == 'cuda':
                status["gpu"]["memory_allocated"] = torch.cuda.memory_allocated(0)
                status["gpu"]["memory_reserved"] = torch.cuda.memory_reserved(0)
        except:
            pass
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.stats,
            "queue_size": self.job_queue.qsize(),
            "workers_active": len([t for t in self.worker_threads if t.is_alive()])
        }
    
    def compile_model_to_wasm(
        self,
        model_path: str,
        output_dir: Optional[str] = None,
        optimization_level: int = 2
    ) -> Optional[str]:
        """
        Compile a PyTorch upscaling model to WebAssembly
        This is a placeholder - actual implementation would require
        PyTorch -> ONNX -> WASM compilation pipeline
        """
        print(f"WASM compilation not yet implemented for: {model_path}")
        print("Use pre-compiled WASM modules or implement using Emscripten/WASI")
        
        # In a real implementation, this would:
        # 1. Export PyTorch model to ONNX
        # 2. Use Emscripten or similar to compile to WASM
        # 3. Optimize with wasm-opt
        # 4. Package with model weights
        
        return None
    
    def shutdown(self):
        """Shutdown the pipeline gracefully"""
        self.running = False
        
        # Stop workers
        for _ in self.worker_threads:
            self.job_queue.put(None)
        
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        # Clear modules
        self.modules.clear()


# Global pipeline instance
_pipeline: Optional[WasmUpscalePipeline] = None


def get_pipeline() -> WasmUpscalePipeline:
    """Get or create the global WASM upscaling pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = WasmUpscalePipeline()
    return _pipeline


# Integration with existing upscaler system
class WasmUpscaler(Upscaler):
    """WebAssembly-based upscaler for integration with existing system"""
    
    def __init__(self, name: str = "WASM Upscaler", paths: List[str] = None):
        super().__init__(name)
        self.pipeline = get_pipeline()
        self.scalers = []
        
        # Register available WASM models
        for model_name in self.pipeline.get_available_models():
            scaler = UpscalerData(
                name=f"WASM {model_name}",
                path=None,
                upscaler=self,
                scale=None,  # Scale is determined by WASM module
                model_name=model_name
            )
            self.scalers.append(scaler)
    
    def do_upscale(self, img: Image.Image, selected_model: str = None) -> Image.Image:
        """Perform upscaling using WASM pipeline"""
        try:
            # Convert PIL to numpy
            img_array = np.array(img)
            
            # Determine scale factor from model name or use default
            scale_factor = 4  # Default scale
            if selected_model and "x" in selected_model:
                try:
                    scale_factor = int(selected_model.split("x")[0].split()[-1])
                except:
                    pass
            
            # Use WASM pipeline
            result = self.pipeline.upscale_immediate(
                img_array,
                scale_factor=scale_factor,
                model_name=selected_model or "default",
                backend=UpscaleBackend.WASM
            )
            
            if result is not None:
                return Image.fromarray(result)
            else:
                # Fallback to CPU
                return img.resize(
                    (img.width * scale_factor, img.height * scale_factor),
                    Image.Resampling.LANCZOS
                )
                
        except Exception as e:
            print(f"WASM upscale error: {e}")
            return img
    
    def load_model(self, path: str) -> bool:
        """Load a WASM model"""
        # Models are loaded automatically by the pipeline
        return True


# WebGPU support for browser environments
class WebGPUUpscaler:
    """WebGPU-based upscaler for browser environments"""
    
    def __init__(self):
        self.device = None
        self.adapter = None
        self.compute_pipeline = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize WebGPU device and pipeline"""
        if not WGPU_AVAILABLE:
            raise RuntimeError("WebGPU not available")
        
        try:
            # Request adapter
            self.adapter = await wgpu.request_adapter(power_preference="high-performance")
            self.device = await self.adapter.request_device()
            
            # Load compute shader for upscaling
            shader_code = self._get_upscale_shader()
            shader_module = self.device.create_shader_module(code=shader_code)
            
            # Create compute pipeline
            self.compute_pipeline = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader_module, "entry_point": "main"}
            )
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"WebGPU initialization failed: {e}")
            return False
    
    def _get_upscale_shader(self) -> str:
        """Get WGSL shader for upscaling"""
        # Simple bilinear upscaling shader
        return """
        @group(0) @binding(0) var input_texture: texture_2d<f32>;
        @group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
        @group(0) @binding(2) var<uniform> scale_factor: u32;
        
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let output_coord = vec2<i32>(global_id.xy);
            let input_size = textureDimensions(input_texture);
            let output_size = input_size * scale_factor;
            
            if (output_coord.x >= i32(output_size.x) || output_coord.y >= i32(output_size.y)) {
                return;
            }
            
            // Calculate input coordinates (bilinear interpolation)
            let input_coord_f = vec2<f32>(output_coord) / vec2<f32>(scale_factor);
            let input_coord = vec2<i32>(input_coord_f);
            let fract_coord = fract(input_coord_f);
            
            // Sample 4 nearest pixels
            let p00 = textureLoad(input_texture, input_coord, 0);
            let p10 = textureLoad(input_texture, input_coord + vec2<i32>(1, 0), 0);
            let p01 = textureLoad(input_texture, input_coord + vec2<i32>(0, 1), 0);
            let p11 = textureLoad(input_texture, input_coord + vec2<i32>(1, 1), 0);
            
            // Bilinear interpolation
            let top = mix(p00, p10, fract_coord.x);
            let bottom = mix(p01, p11, fract_coord.x);
            let result = mix(top, bottom, fract_coord.y);
            
            textureStore(output_texture, output_coord, result);
        }
        """
    
    async def upscale(self, image: np.ndarray, scale_factor: int = 4) -> Optional[np.ndarray]:
        """Upscale image using WebGPU"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert image to texture
            height, width, channels = image.shape
            
            # Create input texture
            input_texture = self.device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
            )
            
            # Create output texture
            output_width = width * scale_factor
            output_height = height * scale_factor
            output_texture = self.device.create_texture(
                size=(output_width, output_height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING
            )
            
            # Upload image data
            if channels == 3:
                # Convert RGB to RGBA
                rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_image[:, :, :3] = image
                rgba_image[:, :, 3] = 255
                image_data = rgba_image.tobytes()
            else:
                image_data = image.tobytes()
            
            self.device.queue.write_texture(
                {"texture": input_texture, "mip_level": 0, "origin": (0, 0, 0)},
                image_data,
                {"offset": 0, "bytes_per_row": width * 4},
                (width, height, 1)
            )
            
            # Create uniform buffer for scale factor
            scale_buffer = self.device.create_buffer_with_data(
                data=np.array([scale_factor], dtype=np.uint32).tobytes(),
                usage=wgpu.BufferUsage.UNIFORM
            )
            
            # Create bind group
            bind_group = self.device.create_bind_group(
                layout=self.compute_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": input_texture.create_view()},
                    {"binding": 1, "resource": output_texture.create_view()},
                    {"binding": 2, "resource": {"buffer": scale_buffer}}
                ]
            )
            
            # Dispatch compute shader
            command_encoder = self.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.compute_pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(
                (output_width + 7) // 8,
                (output_height + 7) // 8,
                1
            )
            compute_pass.end()
            
            # Submit commands
            self.device.queue.submit([command_encoder.finish()])
            
            # Read back result
            output_size = output_width * output_height * 4
            readback_buffer = self.device.create_buffer(
                size=output_size,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
            
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                {"texture": output_texture, "mip_level": 0, "origin": (0, 0, 0)},
                {"buffer": readback_buffer, "offset": 0, "bytes_per_row": output_width * 4},
                (output_width, output_height, 1)
            )
            self.device.queue.submit([command_encoder.finish()])
            
            # Map buffer and read data
            await readback_buffer.map_async(wgpu.MapMode.READ)
            result_data = readback_buffer.read_mapped(0, output_size)
            
            # Convert to numpy array
            result_array = np.frombuffer(result_data, dtype=np.uint8).reshape(
                output_height, output_width, 4
            )
            
            # Remove alpha channel if original was RGB
            if channels == 3:
                result_array = result_array[:, :, :3]
            
            return result_array
            
        except Exception as e:
            print(f"WebGPU upscale error: {e}")
            return None


# API endpoints for web interface
def api_wasm_upscale(
    image_data: str,
    scale_factor: int = 4,
    model_name: str = "default",
    backend: str = "auto"
) -> Dict[str, Any]:
    """
    API endpoint for WASM upscaling
    Used by web interface to offload upscaling
    """
    try:
        import base64
        import io
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get pipeline
        pipeline = get_pipeline()
        
        # Convert backend string to enum
        backend_enum = UpscaleBackend(backend.lower())
        
        # Perform upscaling
        result = pipeline.upscale_immediate(
            image_array,
            scale_factor=scale_factor,
            model_name=model_name,
            backend=backend_enum
        )
        
        if result is None:
            return {"success": False, "error": "Upscaling failed"}
        
        # Convert result to base64
        result_image = Image.fromarray(result)
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image": result_base64,
            "width": result.shape[1],
            "height": result.shape[0],
            "backend_used": backend_enum.value
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Initialize on module load
def initialize():
    """Initialize the WASM upscaling pipeline"""
    try:
        pipeline = get_pipeline()
        
        # Register with existing upscaler system if not already registered
        from modules import upscaler
        if not any(isinstance(u, WasmUpscaler) for u in upscaler.Upscaler.__subclasses__()):
            upscaler.Upscaler.providers.append(WasmUpscaler)
        
        print(f"WASM Upscaling Pipeline initialized")
        print(f"Available backends: {pipeline.get_backend_status()}")
        
    except Exception as e:
        print(f"Failed to initialize WASM pipeline: {e}")


# Auto-initialize when module is imported
initialize()

# Cleanup on exit
import atexit
atexit.register(lambda: get_pipeline().shutdown() if _pipeline else None)