"""
src/forge/data/multimodal/transforms.py

Multi-Modal Data Pipeline with Automatic Preprocessing
Unified data loading system for text, image, video, and audio data with
automatic format detection, augmentation, normalization, and streaming support.
"""

import io
import os
import json
import logging
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.io import read_image, read_video
import torchaudio
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality processor."""
    modality: ModalityType
    extensions: List[str] = field(default_factory=list)
    mime_types: List[str] = field(default_factory=list)
    processor_class: Optional[str] = None
    transform_pipeline: Optional[List[Dict]] = None
    normalize: bool = True
    augmentation: bool = False


@dataclass
class MultiModalConfig:
    """Global configuration for multi-modal pipeline."""
    tokenizer_name: str = "bert-base-uncased"
    image_processor_name: str = "google/vit-base-patch16-224"
    audio_processor_name: str = "facebook/wav2vec2-base-960h"
    video_processor_name: str = "MCG-NJU/videomae-base"
    max_text_length: int = 512
    image_size: Tuple[int, int] = (224, 224)
    audio_sample_rate: int = 16000
    video_fps: float = 1.0
    streaming_buffer_size: int = 1000
    cache_dir: Optional[str] = None


class ModalityProcessor(ABC):
    """Abstract base class for modality processors."""
    
    def __init__(self, config: ModalityConfig, global_config: MultiModalConfig):
        self.config = config
        self.global_config = global_config
        self._setup_transforms()
    
    @abstractmethod
    def _setup_transforms(self) -> None:
        """Setup transformation pipeline for this modality."""
        pass
    
    @abstractmethod
    def process(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process raw data into model-ready format."""
        pass
    
    @abstractmethod
    def detect_format(self, data: Any) -> bool:
        """Detect if data matches this processor's format."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "modality": self.config.modality.value,
            "processed_count": getattr(self, '_processed_count', 0),
            "error_count": getattr(self, '_error_count', 0)
        }


class TextProcessor(ModalityProcessor):
    """Processor for text data with tokenization and augmentation."""
    
    def __init__(self, config: ModalityConfig, global_config: MultiModalConfig):
        super().__init__(config, global_config)
        self.tokenizer = AutoTokenizer.from_pretrained(global_config.tokenizer_name)
        self._processed_count = 0
        self._error_count = 0
    
    def _setup_transforms(self) -> None:
        """Setup text transformation pipeline."""
        self.transforms = []
        
        # Add augmentation if enabled
        if self.config.augmentation:
            self.transforms.extend([
                self._random_deletion,
                self._random_swap,
                self._synonym_replacement
            ])
        
        # Add normalization transforms
        if self.config.normalize:
            self.transforms.append(self._normalize_text)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove extra spaces."""
        return " ".join(text.lower().split())
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words from text."""
        words = text.split()
        if len(words) == 1:
            return text
        return " ".join([w for w in words if np.random.random() > p])
    
    def _random_swap(self, text: str, n_swaps: int = 1) -> str:
        """Randomly swap words in text."""
        words = text.split()
        for _ in range(min(n_swaps, len(words) // 2)):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)
    
    def _synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement (placeholder for actual implementation)."""
        # In production, integrate with WordNet or similar
        return text
    
    def process(self, data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Tokenize and process text."""
        try:
            # Apply transforms
            processed_text = data
            for transform in self.transforms:
                processed_text = transform(processed_text)
            
            # Tokenize
            encoding = self.tokenizer(
                processed_text,
                max_length=self.global_config.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            self._processed_count += 1
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(),
                "original_text": data,
                "modality": ModalityType.TEXT.value,
                "metadata": metadata or {}
            }
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error processing text: {e}")
            raise
    
    def detect_format(self, data: Any) -> bool:
        """Check if data is text."""
        return isinstance(data, str)


class ImageProcessor(ModalityProcessor):
    """Processor for image data with augmentation and normalization."""
    
    def __init__(self, config: ModalityConfig, global_config: MultiModalConfig):
        super().__init__(config, global_config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            global_config.image_processor_name
        )
        self._processed_count = 0
        self._error_count = 0
    
    def _setup_transforms(self) -> None:
        """Setup image transformation pipeline."""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.global_config.image_size))
        
        # Add augmentation if enabled
        if self.config.augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.config.normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def _load_image(self, data: Any) -> torch.Tensor:
        """Load image from various sources."""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # File path or URL
            if data.startswith(('http://', 'https://')):
                # Download from URL
                import requests
                from PIL import Image
                response = requests.get(data)
                img = Image.open(io.BytesIO(response.content))
                return transforms.ToTensor()(img)
            else:
                # Local file
                return read_image(data)
        elif isinstance(data, bytes):
            # Raw bytes
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            return transforms.ToTensor()(img)
        else:
            raise ValueError(f"Unsupported image data type: {type(data)}")
    
    def process(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process and transform image."""
        try:
            # Load image
            image_tensor = self._load_image(data)
            
            # Apply transforms
            transformed = self.transform(image_tensor)
            
            # Extract features using processor
            inputs = self.feature_extractor(
                images=transformed,
                return_tensors="pt"
            )
            
            self._processed_count += 1
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "original_image": data,
                "modality": ModalityType.IMAGE.value,
                "metadata": metadata or {},
                "image_shape": transformed.shape
            }
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error processing image: {e}")
            raise
    
    def detect_format(self, data: Any) -> bool:
        """Check if data is an image."""
        if isinstance(data, str):
            ext = Path(data).suffix.lower()
            return ext in self.config.extensions
        elif isinstance(data, bytes):
            # Try to detect from magic bytes
            return data[:8] == b'\x89PNG\r\n\x1a\n' or data[:2] == b'\xff\xd8'
        return False


class AudioProcessor(ModalityProcessor):
    """Processor for audio data with resampling and augmentation."""
    
    def __init__(self, config: ModalityConfig, global_config: MultiModalConfig):
        super().__init__(config, global_config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            global_config.audio_processor_name
        )
        self._processed_count = 0
        self._error_count = 0
    
    def _setup_transforms(self) -> None:
        """Setup audio transformation pipeline."""
        self.transforms = []
        
        # Resample to target sample rate
        self.transforms.append(self._resample)
        
        # Add augmentation if enabled
        if self.config.augmentation:
            self.transforms.extend([
                self._add_noise,
                self._time_stretch,
                self._pitch_shift
            ])
        
        # Normalize if enabled
        if self.config.normalize:
            self.transforms.append(self._normalize_audio)
    
    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Resample audio to target sample rate."""
        if sample_rate != self.global_config.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.global_config.audio_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.global_config.audio_sample_rate
        return waveform, sample_rate
    
    def _add_noise(self, waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add random noise to audio."""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def _time_stretch(self, waveform: torch.Tensor, rate: float = 1.1) -> torch.Tensor:
        """Time stretch audio (placeholder)."""
        # In production, use torchaudio.transforms.TimeStretch
        return waveform
    
    def _pitch_shift(self, waveform: torch.Tensor, sample_rate: int, n_steps: int = 2) -> torch.Tensor:
        """Pitch shift audio (placeholder)."""
        # In production, use torchaudio.transforms.PitchShift
        return waveform
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        return waveform / (waveform.abs().max() + 1e-8)
    
    def _load_audio(self, data: Any) -> Tuple[torch.Tensor, int]:
        """Load audio from various sources."""
        if isinstance(data, tuple) and len(data) == 2:
            # Already (waveform, sample_rate)
            return data
        elif isinstance(data, str):
            # File path or URL
            if data.startswith(('http://', 'https://')):
                # Download from URL
                import requests
                response = requests.get(data)
                waveform, sample_rate = torchaudio.load(io.BytesIO(response.content))
                return waveform, sample_rate
            else:
                # Local file
                return torchaudio.load(data)
        elif isinstance(data, bytes):
            # Raw bytes
            waveform, sample_rate = torchaudio.load(io.BytesIO(data))
            return waveform, sample_rate
        else:
            raise ValueError(f"Unsupported audio data type: {type(data)}")
    
    def process(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process and transform audio."""
        try:
            # Load audio
            waveform, sample_rate = self._load_audio(data)
            
            # Apply transforms
            for transform in self.transforms:
                if transform == self._resample:
                    waveform, sample_rate = transform(waveform, sample_rate)
                else:
                    waveform = transform(waveform)
            
            # Extract features
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            self._processed_count += 1
            
            return {
                "input_values": inputs["input_values"].squeeze(),
                "original_audio": data,
                "modality": ModalityType.AUDIO.value,
                "metadata": metadata or {},
                "sample_rate": sample_rate,
                "duration": waveform.shape[1] / sample_rate
            }
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error processing audio: {e}")
            raise
    
    def detect_format(self, data: Any) -> bool:
        """Check if data is audio."""
        if isinstance(data, str):
            ext = Path(data).suffix.lower()
            return ext in self.config.extensions
        elif isinstance(data, bytes):
            # Check for common audio file signatures
            return data[:4] == b'RIFF' or data[:3] == b'ID3'
        return False


class VideoProcessor(ModalityProcessor):
    """Processor for video data with frame extraction and augmentation."""
    
    def __init__(self, config: ModalityConfig, global_config: MultiModalConfig):
        super().__init__(config, global_config)
        self.processor = AutoProcessor.from_pretrained(global_config.video_processor_name)
        self._processed_count = 0
        self._error_count = 0
    
    def _setup_transforms(self) -> None:
        """Setup video transformation pipeline."""
        self.frame_transform = transforms.Compose([
            transforms.Resize(self.global_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) if self.config.normalize else transforms.Lambda(lambda x: x)
        ])
    
    def _load_video(self, data: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Load video from various sources."""
        if isinstance(data, tuple) and len(data) == 3:
            # Already (video_frames, audio_frames, info)
            return data
        elif isinstance(data, str):
            # File path or URL
            if data.startswith(('http://', 'https://')):
                # Download from URL
                import tempfile
                import requests
                response = requests.get(data)
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name
                try:
                    video_frames, audio_frames, info = read_video(temp_path)
                finally:
                    os.unlink(temp_path)
                return video_frames, audio_frames, info
            else:
                # Local file
                return read_video(data)
        else:
            raise ValueError(f"Unsupported video data type: {type(data)}")
    
    def _extract_frames(self, video_frames: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
        """Extract uniformly sampled frames from video."""
        total_frames = video_frames.shape[0]
        if total_frames <= num_frames:
            # Repeat frames if video is too short
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
        else:
            # Uniform sampling
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
        
        frames = video_frames[indices]
        
        # Apply frame transforms
        processed_frames = []
        for frame in frames:
            # Convert from (T, H, W, C) to (C, H, W) and normalize
            frame = frame.permute(2, 0, 1).float() / 255.0
            processed_frame = self.frame_transform(frame)
            processed_frames.append(processed_frame)
        
        return torch.stack(processed_frames)
    
    def process(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process and transform video."""
        try:
            # Load video
            video_frames, audio_frames, info = self._load_video(data)
            
            # Extract frames
            num_frames = min(16, video_frames.shape[0])
            processed_frames = self._extract_frames(video_frames, num_frames)
            
            # Process with video processor
            inputs = self.processor(
                videos=processed_frames.unsqueeze(0),  # Add batch dimension
                return_tensors="pt"
            )
            
            self._processed_count += 1
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "original_video": data,
                "modality": ModalityType.VIDEO.value,
                "metadata": metadata or {},
                "video_info": info,
                "num_frames": num_frames,
                "duration": video_frames.shape[0] / info.get("video_fps", 30)
            }
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error processing video: {e}")
            raise
    
    def detect_format(self, data: Any) -> bool:
        """Check if data is video."""
        if isinstance(data, str):
            ext = Path(data).suffix.lower()
            return ext in self.config.extensions
        elif isinstance(data, bytes):
            # Check for common video file signatures
            return data[:4] == b'\x00\x00\x00\x18' or data[:4] == b'ftyp'
        return False


class ModalityRegistry:
    """Registry for modality processors with automatic detection."""
    
    def __init__(self, global_config: MultiModalConfig):
        self.global_config = global_config
        self._processors: Dict[ModalityType, ModalityProcessor] = {}
        self._extension_map: Dict[str, ModalityType] = {}
        self._mime_map: Dict[str, ModalityType] = {}
        self._setup_default_processors()
    
    def _setup_default_processors(self) -> None:
        """Setup default processors for each modality."""
        # Text processor
        text_config = ModalityConfig(
            modality=ModalityType.TEXT,
            extensions=['.txt', '.json', '.jsonl', '.csv', '.tsv'],
            mime_types=['text/plain', 'application/json']
        )
        self.register_processor(TextProcessor(text_config, self.global_config))
        
        # Image processor
        image_config = ModalityConfig(
            modality=ModalityType.IMAGE,
            extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            mime_types=['image/jpeg', 'image/png', 'image/bmp']
        )
        self.register_processor(ImageProcessor(image_config, self.global_config))
        
        # Audio processor
        audio_config = ModalityConfig(
            modality=ModalityType.AUDIO,
            extensions=['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
            mime_types=['audio/wav', 'audio/mpeg', 'audio/flac']
        )
        self.register_processor(AudioProcessor(audio_config, self.global_config))
        
        # Video processor
        video_config = ModalityConfig(
            modality=ModalityType.VIDEO,
            extensions=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            mime_types=['video/mp4', 'video/avi', 'video/quicktime']
        )
        self.register_processor(VideoProcessor(video_config, self.global_config))
    
    def register_processor(self, processor: ModalityProcessor) -> None:
        """Register a processor for a modality."""
        self._processors[processor.config.modality] = processor
        
        # Register extensions
        for ext in processor.config.extensions:
            self._extension_map[ext.lower()] = processor.config.modality
        
        # Register MIME types
        for mime in processor.config.mime_types:
            self._mime_map[mime] = processor.config.modality
    
    def detect_modality(self, data: Any, filename: Optional[str] = None) -> ModalityType:
        """Automatically detect modality of data."""
        # Try filename extension first
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self._extension_map:
                return self._extension_map[ext]
        
        # Try MIME type detection
        if isinstance(data, str) and os.path.isfile(data):
            mime_type, _ = mimetypes.guess_type(data)
            if mime_type and mime_type in self._mime_map:
                return self._mime_map[mime_type]
        
        # Try content-based detection
        for processor in self._processors.values():
            if processor.detect_format(data):
                return processor.config.modality
        
        return ModalityType.UNKNOWN
    
    def get_processor(self, modality: ModalityType) -> Optional[ModalityProcessor]:
        """Get processor for a specific modality."""
        return self._processors.get(modality)
    
    def process_data(self, data: Any, modality: Optional[ModalityType] = None, 
                     metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process data with automatic modality detection."""
        if modality is None:
            modality = self.detect_modality(data, metadata.get('filename') if metadata else None)
        
        processor = self.get_processor(modality)
        if processor is None:
            raise ValueError(f"No processor registered for modality: {modality}")
        
        return processor.process(data, metadata)


class StreamingMultiModalDataset(IterableDataset):
    """Streaming dataset that supports multiple modalities and data sources."""
    
    def __init__(
        self,
        data_source: Union[str, List[str]],
        registry: ModalityRegistry,
        source_type: str = "auto",
        transform: Optional[Callable] = None,
        shuffle: bool = False,
        buffer_size: int = 1000
    ):
        """
        Args:
            data_source: Path, URL, or HuggingFace dataset identifier
            registry: Modality registry for processing
            source_type: 'huggingface', 's3', 'local', or 'auto'
            transform: Optional transform to apply after processing
            shuffle: Whether to shuffle data
            buffer_size: Buffer size for streaming
        """
        self.data_source = data_source
        self.registry = registry
        self.source_type = source_type
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self._setup_source()
    
    def _setup_source(self) -> None:
        """Setup data source based on type."""
        if self.source_type == "auto":
            self.source_type = self._detect_source_type()
        
        if self.source_type == "huggingface":
            self._setup_huggingface_source()
        elif self.source_type == "s3":
            self._setup_s3_source()
        elif self.source_type == "local":
            self._setup_local_source()
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
    
    def _detect_source_type(self) -> str:
        """Automatically detect source type."""
        if isinstance(self.data_source, str):
            parsed = urlparse(self.data_source)
            if parsed.scheme in ('http', 'https') and 'huggingface.co' in parsed.netloc:
                return "huggingface"
            elif parsed.scheme == 's3':
                return "s3"
            elif os.path.exists(self.data_source):
                return "local"
        return "local"
    
    def _setup_huggingface_source(self) -> None:
        """Setup HuggingFace dataset source."""
        from datasets import load_dataset
        
        # Parse dataset name and config
        if ':' in self.data_source:
            dataset_name, config_name = self.data_source.split(':', 1)
        else:
            dataset_name = self.data_source
            config_name = None
        
        self.dataset = load_dataset(
            dataset_name,
            config_name,
            streaming=True,
            split="train"
        )
        
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=self.buffer_size)
    
    def _setup_s3_source(self) -> None:
        """Setup S3 data source."""
        try:
            import s3fs
            self.fs = s3fs.S3FileSystem()
            
            # Parse S3 path
            parsed = urlparse(self.data_source)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip('/')
            
            # List files
            self.files = self.fs.ls(f"{bucket}/{prefix}")
            
            if self.shuffle:
                np.random.shuffle(self.files)
        except ImportError:
            raise ImportError("s3fs is required for S3 support. Install with: pip install s3fs")
    
    def _setup_local_source(self) -> None:
        """Setup local file source."""
        if os.path.isfile(self.data_source):
            self.files = [self.data_source]
        elif os.path.isdir(self.data_source):
            # Recursively find all files
            self.files = []
            for root, _, filenames in os.walk(self.data_source):
                for filename in filenames:
                    self.files.append(os.path.join(root, filename))
        else:
            # Assume it's a pattern or list
            from glob import glob
            self.files = glob(self.data_source)
        
        if self.shuffle:
            np.random.shuffle(self.files)
    
    def _load_file(self, file_path: str) -> Any:
        """Load file content based on source type."""
        if self.source_type == "huggingface":
            # Already loaded by datasets library
            return file_path  # This will be the dataset item
        
        elif self.source_type == "s3":
            with self.fs.open(file_path, 'rb') as f:
                return f.read()
        
        elif self.source_type == "local":
            with open(file_path, 'rb') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset."""
        if self.source_type == "huggingface":
            for item in self.dataset:
                # Process each field according to its modality
                processed_item = {}
                for key, value in item.items():
                    # Detect modality for each field
                    modality = self.registry.detect_modality(value, key)
                    if modality != ModalityType.UNKNOWN:
                        processed = self.registry.process_data(
                            value, 
                            modality, 
                            metadata={"field": key, "source": "huggingface"}
                        )
                        processed_item[key] = processed
                    else:
                        processed_item[key] = value
                
                if self.transform:
                    processed_item = self.transform(processed_item)
                
                yield processed_item
        
        else:
            # For file-based sources
            for file_path in self.files:
                try:
                    content = self._load_file(file_path)
                    
                    # Detect modality
                    modality = self.registry.detect_modality(content, file_path)
                    
                    if modality != ModalityType.UNKNOWN:
                        processed = self.registry.process_data(
                            content,
                            modality,
                            metadata={
                                "filename": os.path.basename(file_path),
                                "source": self.source_type,
                                "path": file_path
                            }
                        )
                        
                        if self.transform:
                            processed = self.transform(processed)
                        
                        yield processed
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue


class MultiModalDataLoader:
    """Unified data loader for multi-modal data with automatic preprocessing."""
    
    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False
    ):
        self.config = config or MultiModalConfig()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # Initialize registry
        self.registry = ModalityRegistry(self.config)
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "modality_counts": {m.value: 0 for m in ModalityType},
            "errors": 0
        }
    
    def load_dataset(
        self,
        data_source: Union[str, List[str]],
        source_type: str = "auto",
        transform: Optional[Callable] = None
    ) -> StreamingMultiModalDataset:
        """Load a streaming multi-modal dataset."""
        return StreamingMultiModalDataset(
            data_source=data_source,
            registry=self.registry,
            source_type=source_type,
            transform=transform,
            shuffle=self.shuffle,
            buffer_size=self.config.streaming_buffer_size
        )
    
    def create_dataloader(
        self,
        dataset: StreamingMultiModalDataset,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """Create a DataLoader from a dataset."""
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def _default_collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Default collate function that handles multi-modal data."""
        if not batch:
            return {}
        
        # Group by modality
        collated = {}
        
        # Get all keys from first item
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            # Check if values are tensors or dicts
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], dict):
                # Recursively collate nested dicts
                collated[key] = self._default_collate_fn(values)
            else:
                collated[key] = values
        
        return collated
    
    def process_single(self, data: Any, modality: Optional[ModalityType] = None) -> Dict[str, Any]:
        """Process a single data item."""
        result = self.registry.process_data(data, modality)
        
        # Update statistics
        self.stats["total_processed"] += 1
        self.stats["modality_counts"][result["modality"]] += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        processor_stats = {}
        for modality, processor in self.registry._processors.items():
            processor_stats[modality.value] = processor.get_stats()
        
        return {
            "global_stats": self.stats,
            "processor_stats": processor_stats
        }


# Factory function for easy instantiation
def create_multimodal_pipeline(
    config: Optional[MultiModalConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> MultiModalDataLoader:
    """Create a multi-modal data pipeline with default configuration."""
    return MultiModalDataLoader(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_multimodal_pipeline()
    
    # Example 1: Process text
    text_result = pipeline.process_single("This is a sample text for processing.")
    print(f"Text processing result keys: {text_result.keys()}")
    
    # Example 2: Process image (if available)
    try:
        # This would require an actual image file
        # image_result = pipeline.process_single("path/to/image.jpg")
        pass
    except Exception as e:
        print(f"Image processing example skipped: {e}")
    
    # Example 3: Load streaming dataset
    try:
        # Load from HuggingFace
        dataset = pipeline.load_dataset(
            "wikitext",
            source_type="huggingface",
            transform=lambda x: {k: v for k, v in x.items() if k != "text"}  # Example transform
        )
        
        # Create dataloader
        dataloader = pipeline.create_dataloader(dataset)
        
        # Get first batch
        for batch in dataloader:
            print(f"Batch keys: {batch.keys()}")
            break
    except Exception as e:
        print(f"Streaming dataset example skipped: {e}")
    
    # Print statistics
    print(f"Pipeline statistics: {pipeline.get_stats()}")