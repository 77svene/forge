import os
import logging
import importlib
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
import tempfile
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

@dataclass
class DataConfig:
    """Configuration for data processing pipeline"""
    source: str
    source_type: str = "local"  # "local", "huggingface", "s3"
    modality: ModalityType = ModalityType.TEXT
    format: Optional[str] = None
    streaming: bool = False
    batch_size: int = 1
    num_workers: int = 0
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    tokenization: Dict[str, Any] = field(default_factory=dict)
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None

@dataclass
class ProcessedSample:
    """Standardized output from any processor"""
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: ModalityType = ModalityType.TEXT
    
    def to_tensor(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Convert all data to tensors"""
        result = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value)
            elif isinstance(value, Image.Image):
                result[key] = torch.from_numpy(np.array(value))
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                result[key] = torch.tensor(value)
            else:
                result[key] = value
                
        if device:
            result = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in result.items()}
        return result

class BaseProcessor(ABC):
    """Abstract base class for all modality processors"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._cache = {}
        
    @abstractmethod
    def process(self, data: Any) -> ProcessedSample:
        """Process raw data into standardized format"""
        pass
    
    @abstractmethod
    def detect_format(self, data: Any) -> str:
        """Detect format of input data"""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate if data can be processed"""
        try:
            self.detect_format(data)
            return True
        except Exception:
            return False
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        if isinstance(data, (str, Path)):
            return hashlib.md5(str(data).encode()).hexdigest()
        elif isinstance(data, bytes):
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()

class TextProcessor(BaseProcessor):
    """Processor for text data"""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.tokenizer = None
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load tokenizer based on config"""
        try:
            from transformers import AutoTokenizer
            tokenizer_name = self.config.tokenization.get("tokenizer", "bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            logger.warning("transformers not installed, using basic tokenization")
            self.tokenizer = None
            
    def process(self, data: Union[str, List[str]]) -> ProcessedSample:
        """Process text data"""
        if isinstance(data, str):
            data = [data]
            
        processed = {}
        
        if self.tokenizer:
            # Use HuggingFace tokenizer
            encoding = self.tokenizer(
                data,
                padding=self.config.tokenization.get("padding", True),
                truncation=self.config.tokenization.get("truncation", True),
                max_length=self.config.tokenization.get("max_length", 512),
                return_tensors="pt"
            )
            processed.update(encoding)
        else:
            # Basic tokenization
            processed["input_ids"] = [list(map(ord, text)) for text in data]
            
        # Apply augmentation if specified
        if self.config.augmentation.get("random_mask", False):
            processed = self._apply_random_mask(processed)
            
        return ProcessedSample(
            data=processed,
            metadata={"num_samples": len(data)},
            modality=ModalityType.TEXT
        )
    
    def _apply_random_mask(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random masking augmentation"""
        mask_prob = self.config.augmentation.get("mask_prob", 0.15)
        if "input_ids" in processed:
            input_ids = processed["input_ids"]
            mask = torch.rand(input_ids.shape) < mask_prob
            processed["masked_input_ids"] = input_ids.clone()
            processed["masked_input_ids"][mask] = self.tokenizer.mask_token_id if self.tokenizer else 0
            processed["mask_labels"] = mask
        return processed
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, str):
            return "plain_text"
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            return "text_list"
        else:
            raise ValueError(f"Unsupported text format: {type(data)}")

class ImageProcessor(BaseProcessor):
    """Processor for image data"""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.feature_extractor = None
        self._load_feature_extractor()
        
    def _load_feature_extractor(self):
        """Load image feature extractor"""
        try:
            from transformers import AutoFeatureExtractor
            model_name = self.config.preprocessing.get("model", "google/vit-base-patch16-224")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except ImportError:
            logger.warning("transformers not installed, using PIL for basic processing")
            
    def process(self, data: Union[str, Path, Image.Image, np.ndarray]) -> ProcessedSample:
        """Process image data"""
        # Load image
        if isinstance(data, (str, Path)):
            image = Image.open(data).convert("RGB")
        elif isinstance(data, np.ndarray):
            image = Image.fromarray(data).convert("RGB")
        elif isinstance(data, Image.Image):
            image = data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(data)}")
            
        processed = {}
        
        if self.feature_extractor:
            # Use feature extractor
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            processed.update(inputs)
        else:
            # Basic processing
            processed["pixel_values"] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        # Apply augmentations
        if self.config.augmentation:
            processed = self._apply_augmentations(processed, image)
            
        # Apply normalization
        if self.config.normalization:
            processed = self._apply_normalization(processed)
            
        return ProcessedSample(
            data=processed,
            metadata={
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            },
            modality=ModalityType.IMAGE
        )
    
    def _apply_augmentations(self, processed: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
        """Apply image augmentations"""
        try:
            import torchvision.transforms as T
            
            augmentations = []
            if self.config.augmentation.get("random_crop", False):
                augmentations.append(T.RandomCrop(self.config.augmentation.get("crop_size", 224)))
            if self.config.augmentation.get("random_flip", False):
                augmentations.append(T.RandomHorizontalFlip())
            if self.config.augmentation.get("color_jitter", False):
                augmentations.append(T.ColorJitter(
                    brightness=self.config.augmentation.get("brightness", 0.2),
                    contrast=self.config.augmentation.get("contrast", 0.2),
                    saturation=self.config.augmentation.get("saturation", 0.2),
                    hue=self.config.augmentation.get("hue", 0.1)
                ))
                
            if augmentations:
                transform = T.Compose(augmentations)
                if "pixel_values" in processed:
                    processed["augmented_pixel_values"] = transform(processed["pixel_values"])
                    
        except ImportError:
            logger.warning("torchvision not installed, skipping augmentations")
            
        return processed
    
    def _apply_normalization(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization to image tensors"""
        if "pixel_values" in processed:
            mean = self.config.normalization.get("mean", [0.485, 0.456, 0.406])
            std = self.config.normalization.get("std", [0.229, 0.224, 0.225])
            
            mean = torch.tensor(mean).view(3, 1, 1)
            std = torch.tensor(std).view(3, 1, 1)
            
            processed["normalized_pixel_values"] = (processed["pixel_values"] - mean) / std
            
        return processed
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, (str, Path)):
            ext = Path(data).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return f"image_{ext[1:]}"
        elif isinstance(data, Image.Image):
            return "pil_image"
        elif isinstance(data, np.ndarray):
            return "numpy_array"
        raise ValueError(f"Unsupported image format: {type(data)}")

class VideoProcessor(BaseProcessor):
    """Processor for video data"""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.frame_extractor = None
        self._load_video_utils()
        
    def _load_video_utils(self):
        """Load video processing utilities"""
        try:
            import decord
            self.frame_extractor = decord
        except ImportError:
            logger.warning("decord not installed, video processing may be limited")
            
    def process(self, data: Union[str, Path, List[np.ndarray]]) -> ProcessedSample:
        """Process video data"""
        frames = []
        metadata = {}
        
        if isinstance(data, (str, Path)):
            # Load video file
            if self.frame_extractor:
                vr = self.frame_extractor.VideoReader(str(data))
                metadata["num_frames"] = len(vr)
                metadata["fps"] = vr.get_avg_fps()
                
                # Sample frames
                num_frames = self.config.preprocessing.get("num_frames", 16)
                indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
                frames = vr.get_batch(indices).asnumpy()
            else:
                # Fallback to OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(data))
                    metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    metadata["num_frames"] = frame_count
                    
                    num_frames = self.config.preprocessing.get("num_frames", 16)
                    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
                    
                    frames = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    frames = np.array(frames)
                except ImportError:
                    raise ImportError("Either decord or opencv-python required for video processing")
                    
        elif isinstance(data, list):
            frames = np.array(data)
            metadata["num_frames"] = len(frames)
            
        # Process frames as images
        processed_frames = []
        image_processor = ImageProcessor(self.config)
        
        for frame in frames:
            processed_frame = image_processor.process(frame)
            processed_frames.append(processed_frame.data)
            
        # Stack frames
        processed = {}
        for key in processed_frames[0].keys():
            processed[key] = torch.stack([f[key] for f in processed_frames])
            
        return ProcessedSample(
            data=processed,
            metadata=metadata,
            modality=ModalityType.VIDEO
        )
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, (str, Path)):
            ext = Path(data).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return f"video_{ext[1:]}"
        elif isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data):
            return "frame_list"
        raise ValueError(f"Unsupported video format: {type(data)}")

class AudioProcessor(BaseProcessor):
    """Processor for audio data"""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.feature_extractor = None
        self._load_audio_utils()
        
    def _load_audio_utils(self):
        """Load audio processing utilities"""
        try:
            import torchaudio
            self.torchaudio = torchaudio
        except ImportError:
            logger.warning("torchaudio not installed, audio processing may be limited")
            
    def process(self, data: Union[str, Path, np.ndarray, Tuple[np.ndarray, int]]) -> ProcessedSample:
        """Process audio data"""
        waveform = None
        sample_rate = None
        metadata = {}
        
        if isinstance(data, (str, Path)):
            # Load audio file
            if hasattr(self, 'torchaudio'):
                waveform, sample_rate = self.torchaudio.load(str(data))
                metadata["sample_rate"] = sample_rate
                metadata["num_channels"] = waveform.shape[0]
                metadata["num_samples"] = waveform.shape[1]
            else:
                # Fallback to librosa
                try:
                    import librosa
                    waveform, sample_rate = librosa.load(str(data), sr=None)
                    waveform = torch.from_numpy(waveform).unsqueeze(0)
                    metadata["sample_rate"] = sample_rate
                    metadata["num_channels"] = 1
                    metadata["num_samples"] = len(waveform)
                except ImportError:
                    raise ImportError("Either torchaudio or librosa required for audio processing")
                    
        elif isinstance(data, tuple) and len(data) == 2:
            waveform, sample_rate = data
            waveform = torch.from_numpy(waveform) if isinstance(waveform, np.ndarray) else waveform
            metadata["sample_rate"] = sample_rate
            
        elif isinstance(data, np.ndarray):
            waveform = torch.from_numpy(data)
            sample_rate = self.config.preprocessing.get("sample_rate", 16000)
            metadata["sample_rate"] = sample_rate
            
        # Resample if needed
        target_sample_rate = self.config.preprocessing.get("target_sample_rate", 16000)
        if sample_rate != target_sample_rate and hasattr(self, 'torchaudio'):
            resampler = self.torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            metadata["resampled_rate"] = target_sample_rate
            
        # Extract features
        processed = {}
        
        if self.config.preprocessing.get("extract_mel", False):
            # Extract mel spectrogram
            if hasattr(self, 'torchaudio'):
                mel_transform = self.torchaudio.transforms.MelSpectrogram(
                    sample_rate=target_sample_rate,
                    n_mels=self.config.preprocessing.get("n_mels", 80)
                )
                mel_spec = mel_transform(waveform)
                processed["mel_spectrogram"] = mel_spec
            else:
                logger.warning("Mel spectrogram extraction requires torchaudio")
                
        # Add waveform
        processed["waveform"] = waveform
        
        return ProcessedSample(
            data=processed,
            metadata=metadata,
            modality=ModalityType.AUDIO
        )
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, (str, Path)):
            ext = Path(data).suffix.lower()
            if ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                return f"audio_{ext[1:]}"
        elif isinstance(data, tuple) and len(data) == 2:
            return "waveform_tuple"
        elif isinstance(data, np.ndarray):
            return "numpy_waveform"
        raise ValueError(f"Unsupported audio format: {type(data)}")

class MultiModalProcessor(BaseProcessor):
    """Processor for multimodal data"""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.processors = {}
        self._init_processors()
        
    def _init_processors(self):
        """Initialize modality-specific processors"""
        modality_map = {
            ModalityType.TEXT: TextProcessor,
            ModalityType.IMAGE: ImageProcessor,
            ModalityType.VIDEO: VideoProcessor,
            ModalityType.AUDIO: AudioProcessor
        }
        
        for modality, processor_class in modality_map.items():
            # Create config for each modality
            modality_config = DataConfig(
                source=self.config.source,
                source_type=self.config.source_type,
                modality=modality,
                preprocessing=self.config.preprocessing.get(modality.value, {}),
                augmentation=self.config.augmentation.get(modality.value, {}),
                normalization=self.config.normalization.get(modality.value, {}),
                tokenization=self.config.tokenization.get(modality.value, {})
            )
            self.processors[modality] = processor_class(modality_config)
            
    def process(self, data: Dict[str, Any]) -> ProcessedSample:
        """Process multimodal data"""
        processed = {}
        metadata = {}
        
        for modality_key, modality_data in data.items():
            # Determine modality from key
            try:
                modality = ModalityType(modality_key)
            except ValueError:
                # Try to detect from data
                modality = self._detect_modality(modality_data)
                
            if modality in self.processors:
                processor = self.processors[modality]
                result = processor.process(modality_data)
                processed[modality.value] = result.data
                metadata[modality.value] = result.metadata
                
        return ProcessedSample(
            data=processed,
            metadata=metadata,
            modality=ModalityType.MULTIMODAL
        )
    
    def _detect_modality(self, data: Any) -> ModalityType:
        """Detect modality from data"""
        if isinstance(data, str):
            if any(data.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                return ModalityType.IMAGE
            elif any(data.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov']):
                return ModalityType.VIDEO
            elif any(data.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                return ModalityType.AUDIO
            else:
                return ModalityType.TEXT
        elif isinstance(data, Image.Image):
            return ModalityType.IMAGE
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 3 and data.shape[2] in [1, 3, 4]:
                return ModalityType.IMAGE
            elif len(data.shape) == 2:
                return ModalityType.AUDIO
        return ModalityType.TEXT
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, dict):
            formats = []
            for key, value in data.items():
                try:
                    modality = ModalityType(key)
                    processor = self.processors.get(modality)
                    if processor:
                        formats.append(f"{key}:{processor.detect_format(value)}")
                except ValueError:
                    formats.append(f"unknown:{type(value).__name__}")
            return "|".join(formats)
        return "multimodal_dict"

class ProcessorRegistry:
    """Registry for managing modality processors"""
    
    _instance = None
    _processors: Dict[ModalityType, Type[BaseProcessor]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize registry with default processors"""
        self.register(ModalityType.TEXT, TextProcessor)
        self.register(ModalityType.IMAGE, ImageProcessor)
        self.register(ModalityType.VIDEO, VideoProcessor)
        self.register(ModalityType.AUDIO, AudioProcessor)
        self.register(ModalityType.MULTIMODAL, MultiModalProcessor)
        
    def register(self, modality: ModalityType, processor_class: Type[BaseProcessor]):
        """Register a processor for a modality"""
        self._processors[modality] = processor_class
        logger.info(f"Registered processor {processor_class.__name__} for {modality.value}")
        
    def get_processor(self, modality: ModalityType, config: DataConfig) -> BaseProcessor:
        """Get processor instance for modality"""
        if modality not in self._processors:
            raise ValueError(f"No processor registered for modality: {modality}")
        return self._processors[modality](config)
    
    def auto_detect_processor(self, data: Any, config: DataConfig) -> BaseProcessor:
        """Automatically detect and return appropriate processor"""
        # Try each processor to see which can handle the data
        for modality, processor_class in self._processors.items():
            if modality == ModalityType.MULTIMODAL:
                continue  # Skip multimodal for auto-detection
                
            processor = processor_class(config)
            if processor.validate(data):
                return processor
                
        # Fallback to text processor
        return self._processors[ModalityType.TEXT](config)
    
    def list_processors(self) -> List[ModalityType]:
        """List all registered modalities"""
        return list(self._processors.keys())

class StreamingDataset(IterableDataset):
    """Streaming dataset with automatic preprocessing"""
    
    def __init__(
        self,
        config: DataConfig,
        processor: Optional[BaseProcessor] = None,
        transform: Optional[Callable] = None
    ):
        self.config = config
        self.processor = processor or ProcessorRegistry().get_processor(config.modality, config)
        self.transform = transform
        self._data_source = None
        self._init_data_source()
        
    def _init_data_source(self):
        """Initialize data source based on config"""
        if self.config.source_type == "huggingface":
            self._init_huggingface_source()
        elif self.config.source_type == "s3":
            self._init_s3_source()
        else:
            self._init_local_source()
            
    def _init_huggingface_source(self):
        """Initialize HuggingFace dataset source"""
        try:
            from datasets import load_dataset
            dataset_name = self.config.source
            split = self.config.preprocessing.get("split", "train")
            
            self._data_source = load_dataset(
                dataset_name,
                split=split,
                streaming=self.config.streaming
            )
            
            if self.config.max_samples:
                self._data_source = self._data_source.take(self.config.max_samples)
                
        except ImportError:
            raise ImportError("datasets library required for HuggingFace source")
            
    def _init_s3_source(self):
        """Initialize S3 data source"""
        try:
            import s3fs
            fs = s3fs.S3FileSystem()
            
            # Parse S3 path
            if self.config.source.startswith("s3://"):
                path = self.config.source[5:]
            else:
                path = self.config.source
                
            # List files
            files = fs.ls(path)
            self._data_source = iter(files)
            
        except ImportError:
            raise ImportError("s3fs library required for S3 source")
            
    def _init_local_source(self):
        """Initialize local file source"""
        path = Path(self.config.source)
        
        if path.is_file():
            # Single file
            self._data_source = iter([path])
        elif path.is_dir():
            # Directory of files
            files = list(path.glob("*"))
            if self.config.max_samples:
                files = files[:self.config.max_samples]
            self._data_source = iter(files)
        else:
            raise ValueError(f"Invalid source path: {self.config.source}")
            
    def __iter__(self):
        """Iterate over dataset"""
        for item in self._data_source:
            try:
                # Load data based on source type
                if self.config.source_type == "huggingface":
                    data = item
                else:
                    data = self._load_file(item)
                    
                # Process data
                processed = self.processor.process(data)
                
                # Apply transform if provided
                if self.transform:
                    processed = self.transform(processed)
                    
                yield processed
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
                
    def _load_file(self, path: Path) -> Any:
        """Load file based on extension"""
        ext = path.suffix.lower()
        
        if ext in ['.txt', '.json', '.jsonl']:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return Image.open(path)
        elif ext in ['.wav', '.mp3', '.flac']:
            return str(path)  # Let processor handle loading
        elif ext in ['.mp4', '.avi', '.mov']:
            return str(path)  # Let processor handle loading
        else:
            # Try to read as text
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return str(path)

class MultiModalDataPipeline:
    """Main pipeline for multimodal data processing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.registry = ProcessorRegistry()
        self.processor = None
        self.dataset = None
        self._init_pipeline()
        
    def _init_pipeline(self):
        """Initialize the pipeline"""
        if self.config.modality == ModalityType.MULTIMODAL:
            self.processor = MultiModalProcessor(self.config)
        else:
            self.processor = self.registry.get_processor(self.config.modality, self.config)
            
        if self.config.streaming:
            self.dataset = StreamingDataset(self.config, self.processor)
            
    def process_batch(self, batch: List[Any]) -> List[ProcessedSample]:
        """Process a batch of data"""
        results = []
        for item in batch:
            try:
                processed = self.processor.process(item)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
        return results
    
    def create_dataloader(self, **kwargs):
        """Create a DataLoader from the dataset"""
        if not self.dataset:
            raise ValueError("Dataset not initialized. Set streaming=True in config.")
            
        from torch.utils.data import DataLoader
        
        default_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "collate_fn": self._collate_fn
        }
        default_kwargs.update(kwargs)
        
        return DataLoader(self.dataset, **default_kwargs)
    
    def _collate_fn(self, batch: List[ProcessedSample]) -> Dict[str, Any]:
        """Custom collate function for processed samples"""
        collated = {}
        
        for sample in batch:
            for key, value in sample.data.items():
                if key not in collated:
                    collated[key] = []
                collated[key].append(value)
                
        # Stack tensors
        for key in collated:
            if all(isinstance(v, torch.Tensor) for v in collated[key]):
                collated[key] = torch.stack(collated[key])
            elif all(isinstance(v, dict) for v in collated[key]):
                # Handle nested dicts (for multimodal)
                collated[key] = self._collate_nested_dicts(collated[key])
                
        return collated
    
    def _collate_nested_dicts(self, dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate nested dictionaries"""
        result = {}
        for d in dicts:
            for key, value in d.items():
                if key not in result:
                    result[key] = []
                result[key].append(value)
                
        # Stack tensors in nested dicts
        for key in result:
            if all(isinstance(v, torch.Tensor) for v in result[key]):
                result[key] = torch.stack(result[key])
                
        return result
    
    def register_processor(self, modality: ModalityType, processor_class: Type[BaseProcessor]):
        """Register a custom processor"""
        self.registry.register(modality, processor_class)
        
    @staticmethod
    def from_config_file(config_path: Union[str, Path]) -> 'MultiModalDataPipeline':
        """Create pipeline from config file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Convert modality string to enum
        if 'modality' in config_dict:
            config_dict['modality'] = ModalityType(config_dict['modality'])
            
        config = DataConfig(**config_dict)
        return MultiModalDataPipeline(config)

# Factory function for easy creation
def create_data_pipeline(
    source: str,
    modality: Union[str, ModalityType] = "text",
    **kwargs
) -> MultiModalDataPipeline:
    """Create a multimodal data pipeline"""
    if isinstance(modality, str):
        modality = ModalityType(modality)
        
    config = DataConfig(
        source=source,
        modality=modality,
        **kwargs
    )
    return MultiModalDataPipeline(config)

# Export main components
__all__ = [
    "ModalityType",
    "DataConfig",
    "ProcessedSample",
    "BaseProcessor",
    "TextProcessor",
    "ImageProcessor",
    "VideoProcessor",
    "AudioProcessor",
    "MultiModalProcessor",
    "ProcessorRegistry",
    "StreamingDataset",
    "MultiModalDataPipeline",
    "create_data_pipeline"
]