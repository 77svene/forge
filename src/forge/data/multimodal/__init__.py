"""
Multi-Modal Data Pipeline with Automatic Preprocessing
Unified data loading system that automatically handles text, image, video, and audio data
with built-in augmentation, normalization, and format conversion.
Supports streaming datasets from HuggingFace, S3, and local storage with automatic
tokenization and feature extraction.
"""

import os
import io
import json
import logging
import mimetypes
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable
from urllib.parse import urlparse

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchaudio
import librosa
import cv2
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from datasets import load_dataset, Dataset, IterableDataset
import boto3
from botocore.exceptions import ClientError
import smart_open

from ..utils import logging as forge_logging

logger = forge_logging.get_logger(__name__)


class ModalityType(Enum):
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class DataFormat(Enum):
    """Supported data formats."""
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    IMAGE_FOLDER = "imagefolder"
    AUDIO_FOLDER = "audiofolder"
    VIDEO_FOLDER = "videofolder"
    HUGGINGFACE = "huggingface"
    S3 = "s3"
    LOCAL = "local"


@dataclass
class ModalityConfig:
    """Configuration for modality processing."""
    modality: ModalityType
    enabled: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    feature_extractor: Optional[str] = None
    tokenizer: Optional[str] = None


@dataclass
class DataPipelineConfig:
    """Configuration for the multi-modal data pipeline."""
    modalities: Dict[ModalityType, ModalityConfig] = field(default_factory=dict)
    streaming: bool = True
    batch_size: int = 32
    num_workers: int = 4
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    format_detection: bool = True
    auto_convert: bool = True


class ModalityProcessor(ABC):
    """Abstract base class for modality processors."""
    
    def __init__(self, config: ModalityConfig):
        self.config = config
        self.feature_extractor = None
        self.tokenizer = None
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize feature extractors and tokenizers."""
        if self.config.feature_extractor:
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.config.feature_extractor
                )
            except Exception as e:
                logger.warning(f"Failed to load feature extractor: {e}")
        
        if self.config.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer
                )
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
    
    @abstractmethod
    def load(self, data: Any) -> Any:
        """Load raw data into memory."""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Apply preprocessing to the data."""
        pass
    
    @abstractmethod
    def augment(self, data: Any) -> Any:
        """Apply data augmentation."""
        pass
    
    @abstractmethod
    def normalize(self, data: Any) -> Any:
        """Normalize the data."""
        pass
    
    @abstractmethod
    def extract_features(self, data: Any) -> Dict[str, torch.Tensor]:
        """Extract features from the data."""
        pass
    
    def process(self, data: Any) -> Dict[str, torch.Tensor]:
        """Full processing pipeline."""
        if not self.config.enabled:
            return {}
        
        loaded = self.load(data)
        preprocessed = self.preprocess(loaded)
        augmented = self.augment(preprocessed)
        normalized = self.normalize(augmented)
        features = self.extract_features(normalized)
        return features


class TextProcessor(ModalityProcessor):
    """Processor for text data."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        self.max_length = config.preprocessing.get("max_length", 512)
        self.truncation = config.preprocessing.get("truncation", True)
        self.padding = config.preprocessing.get("padding", "max_length")
    
    def load(self, data: Any) -> str:
        """Load text data."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict) and "text" in data:
            return data["text"]
        else:
            raise ValueError(f"Unsupported text data format: {type(data)}")
    
    def preprocess(self, text: str) -> str:
        """Preprocess text."""
        if not self.config.preprocessing:
            return text
        
        # Apply text preprocessing options
        if self.config.preprocessing.get("lowercase", False):
            text = text.lower()
        if self.config.preprocessing.get("strip_whitespace", False):
            text = text.strip()
        if self.config.preprocessing.get("remove_special_chars", False):
            import re
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def augment(self, text: str) -> str:
        """Apply text augmentation."""
        if not self.config.augmentation:
            return text
        
        # Simple text augmentation techniques
        if self.config.augmentation.get("random_deletion", False):
            import random
            words = text.split()
            if len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
                text = " ".join(words)
        
        if self.config.augmentation.get("synonym_replacement", False):
            # This would require a synonym dictionary
            pass
        
        return text
    
    def normalize(self, text: str) -> str:
        """Normalize text (no-op for text)."""
        return text
    
    def extract_features(self, text: str) -> Dict[str, torch.Tensor]:
        """Extract text features using tokenizer."""
        if not self.tokenizer:
            # Return basic tokenization if no tokenizer specified
            return {"text": text}
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0)
        }


class ImageProcessor(ModalityProcessor):
    """Processor for image data."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        self.size = config.preprocessing.get("size", (224, 224))
        self.mean = config.normalization.get("mean", [0.485, 0.456, 0.406])
        self.std = config.normalization.get("std", [0.229, 0.224, 0.225])
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        self.base_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def load(self, data: Any) -> Image.Image:
        """Load image data."""
        if isinstance(data, Image.Image):
            return data
        elif isinstance(data, str):
            # Load from file path or URL
            if data.startswith(('http://', 'https://')):
                import requests
                response = requests.get(data)
                return Image.open(io.BytesIO(response.content))
            else:
                return Image.open(data)
        elif isinstance(data, bytes):
            return Image.open(io.BytesIO(data))
        elif isinstance(data, np.ndarray):
            return Image.fromarray(data)
        elif isinstance(data, torch.Tensor):
            # Convert tensor to PIL Image
            if data.dim() == 3:
                if data.shape[0] in [1, 3]:  # CHW format
                    data = data.permute(1, 2, 0)
            return Image.fromarray(data.numpy().astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image data format: {type(data)}")
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image."""
        if not self.config.preprocessing:
            return image
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def augment(self, image: Image.Image) -> Image.Image:
        """Apply image augmentation."""
        if self.config.augmentation and self.config.augmentation.get("enabled", False):
            return self.augmentation_transform(image)
        return image
    
    def normalize(self, image: Image.Image) -> torch.Tensor:
        """Normalize image."""
        if self.config.augmentation and self.config.augmentation.get("enabled", False):
            # Already normalized in augmentation transform
            return self.augmentation_transform(image)
        return self.base_transform(image)
    
    def extract_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract image features."""
        if self.feature_extractor:
            # Use HuggingFace feature extractor
            inputs = self.feature_extractor(
                images=image_tensor,
                return_tensors="pt"
            )
            return {"pixel_values": inputs["pixel_values"].squeeze(0)}
        
        # Return normalized tensor
        return {"pixel_values": image_tensor}


class AudioProcessor(ModalityProcessor):
    """Processor for audio data."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        self.sample_rate = config.preprocessing.get("sample_rate", 16000)
        self.max_length = config.preprocessing.get("max_length", 10)  # seconds
        self.n_mels = config.preprocessing.get("n_mels", 80)
        self.n_fft = config.preprocessing.get("n_fft", 400)
        self.hop_length = config.preprocessing.get("hop_length", 160)
    
    def load(self, data: Any) -> Tuple[torch.Tensor, int]:
        """Load audio data."""
        if isinstance(data, tuple) and len(data) == 2:
            # Already loaded as (waveform, sample_rate)
            return data
        elif isinstance(data, str):
            # Load from file path or URL
            if data.startswith(('http://', 'https://')):
                import requests
                response = requests.get(data)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name
                waveform, sample_rate = torchaudio.load(temp_path)
                os.unlink(temp_path)
            else:
                waveform, sample_rate = torchaudio.load(data)
            return waveform, sample_rate
        elif isinstance(data, bytes):
            # Load from bytes
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(data)
                temp_path = f.name
            waveform, sample_rate = torchaudio.load(temp_path)
            os.unlink(temp_path)
            return waveform, sample_rate
        elif isinstance(data, np.ndarray):
            # Convert numpy array to tensor
            if len(data.shape) == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data)
            return waveform, self.sample_rate
        else:
            raise ValueError(f"Unsupported audio data format: {type(data)}")
    
    def preprocess(self, audio_data: Tuple[torch.Tensor, int]) -> torch.Tensor:
        """Preprocess audio."""
        waveform, sample_rate = audio_data
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim or pad to max length
        max_samples = self.sample_rate * self.max_length
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation."""
        if not self.config.augmentation:
            return waveform
        
        augmented = waveform.clone()
        
        # Add noise
        if self.config.augmentation.get("add_noise", False):
            noise_level = self.config.augmentation.get("noise_level", 0.005)
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
        
        # Time stretching
        if self.config.augmentation.get("time_stretch", False):
            rate = self.config.augmentation.get("stretch_rate", 1.1)
            augmented = torchaudio.transforms.TimeStretch()(augmented, rate)
        
        return augmented
    
    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform."""
        # Normalize to [-1, 1]
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio features."""
        if self.feature_extractor:
            # Use HuggingFace feature extractor
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            return {"input_values": inputs["input_values"].squeeze(0)}
        
        # Extract mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec = mel_transform(waveform)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-9)
        
        return {"mel_spectrogram": log_mel_spec.squeeze(0)}


class VideoProcessor(ModalityProcessor):
    """Processor for video data."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        self.frame_rate = config.preprocessing.get("frame_rate", 30)
        self.max_frames = config.preprocessing.get("max_frames", 100)
        self.size = config.preprocessing.get("size", (224, 224))
        self.mean = config.normalization.get("mean", [0.485, 0.456, 0.406])
        self.std = config.normalization.get("std", [0.229, 0.224, 0.225])
    
    def load(self, data: Any) -> List[Image.Image]:
        """Load video data."""
        if isinstance(data, list) and all(isinstance(frame, Image.Image) for frame in data):
            return data
        elif isinstance(data, str):
            # Load from file path or URL
            if data.startswith(('http://', 'https://')):
                import requests
                response = requests.get(data)
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name
                frames = self._extract_frames(temp_path)
                os.unlink(temp_path)
            else:
                frames = self._extract_frames(data)
            return frames
        elif isinstance(data, bytes):
            # Load from bytes
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(data)
                temp_path = f.name
            frames = self._extract_frames(temp_path)
            os.unlink(temp_path)
            return frames
        else:
            raise ValueError(f"Unsupported video data format: {type(data)}")
    
    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        frame_interval = max(1, int(original_fps / self.frame_rate))
        
        frame_count = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def preprocess(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Preprocess video frames."""
        processed_frames = []
        for frame in frames:
            # Convert to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            processed_frames.append(frame)
        return processed_frames
    
    def augment(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply video augmentation."""
        if not self.config.augmentation:
            return frames
        
        augmented_frames = []
        for frame in frames:
            # Apply same augmentation to all frames
            if self.config.augmentation.get("random_horizontal_flip", False):
                if np.random.random() > 0.5:
                    frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.config.augmentation.get("color_jitter", False):
                import torchvision.transforms.functional as F
                frame = F.adjust_brightness(frame, np.random.uniform(0.8, 1.2))
                frame = F.adjust_contrast(frame, np.random.uniform(0.8, 1.2))
                frame = F.adjust_saturation(frame, np.random.uniform(0.8, 1.2))
            
            augmented_frames.append(frame)
        
        return augmented_frames
    
    def normalize(self, frames: List[Image.Image]) -> torch.Tensor:
        """Normalize video frames."""
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        normalized_frames = []
        for frame in frames:
            normalized_frames.append(transform(frame))
        
        # Stack frames: (T, C, H, W)
        video_tensor = torch.stack(normalized_frames, dim=0)
        
        # Rearrange to (C, T, H, W) for 3D CNNs
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor
    
    def extract_features(self, video_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract video features."""
        if self.feature_extractor:
            # Use HuggingFace feature extractor
            inputs = self.feature_extractor(
                list(video_tensor.permute(1, 0, 2, 3)),  # List of frames
                return_tensors="pt"
            )
            return {"pixel_values": inputs["pixel_values"].squeeze(0)}
        
        return {"video_frames": video_tensor}


class ModalityProcessorRegistry:
    """Registry for modality processors."""
    
    _processors = {
        ModalityType.TEXT: TextProcessor,
        ModalityType.IMAGE: ImageProcessor,
        ModalityType.AUDIO: AudioProcessor,
        ModalityType.VIDEO: VideoProcessor,
    }
    
    @classmethod
    def register(cls, modality: ModalityType, processor_class: type):
        """Register a new modality processor."""
        cls._processors[modality] = processor_class
    
    @classmethod
    def get_processor(cls, modality: ModalityType, config: ModalityConfig) -> ModalityProcessor:
        """Get processor instance for modality."""
        if modality not in cls._processors:
            raise ValueError(f"No processor registered for modality: {modality}")
        
        processor_class = cls._processors[modality]
        return processor_class(config)


class DataLoader:
    """Unified data loader for multiple sources."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.s3_client = None
        self._setup_s3_client()
    
    def _setup_s3_client(self):
        """Setup S3 client if needed."""
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    
    def detect_format(self, path: str) -> DataFormat:
        """Detect data format from path."""
        if self.config.format_detection:
            parsed = urlparse(path)
            
            if parsed.scheme in ['s3']:
                return DataFormat.S3
            elif parsed.scheme in ['http', 'https']:
                if 'huggingface.co' in parsed.netloc:
                    return DataFormat.HUGGINGFACE
                else:
                    # Try to detect from file extension
                    ext = Path(parsed.path).suffix.lower()
                    return self._extension_to_format(ext)
            else:
                # Local file
                path_obj = Path(path)
                if path_obj.is_dir():
                    # Check directory contents
                    files = list(path_obj.glob('*'))
                    if any(f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for f in files):
                        return DataFormat.IMAGE_FOLDER
                    elif any(f.suffix.lower() in ['.wav', '.mp3', '.flac'] for f in files):
                        return DataFormat.AUDIO_FOLDER
                    elif any(f.suffix.lower() in ['.mp4', '.avi', '.mov'] for f in files):
                        return DataFormat.VIDEO_FOLDER
                    else:
                        return DataFormat.LOCAL
                else:
                    ext = path_obj.suffix.lower()
                    return self._extension_to_format(ext)
        else:
            return DataFormat.LOCAL
    
    def _extension_to_format(self, ext: str) -> DataFormat:
        """Convert file extension to data format."""
        format_map = {
            '.jsonl': DataFormat.JSONL,
            '.json': DataFormat.JSON,
            '.csv': DataFormat.CSV,
            '.parquet': DataFormat.PARQUET,
            '.arrow': DataFormat.ARROW,
        }
        return format_map.get(ext, DataFormat.LOCAL)
    
    def load_from_huggingface(self, dataset_name: str, **kwargs) -> Union[Dataset, IterableDataset]:
        """Load dataset from HuggingFace Hub."""
        try:
            if self.config.streaming:
                return load_dataset(dataset_name, streaming=True, **kwargs)
            else:
                return load_dataset(dataset_name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
            raise
    
    def load_from_s3(self, s3_path: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Load data from S3 with streaming."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        try:
            # Check if it's a single file or directory
            if key.endswith('/'):
                # Directory - list objects
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=key)
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        obj_key = obj['Key']
                        if not obj_key.endswith('/'):
                            yield self._load_s3_object(bucket, obj_key)
            else:
                # Single file
                yield self._load_s3_object(bucket, key)
        except ClientError as e:
            logger.error(f"Failed to load from S3 {s3_path}: {e}")
            raise
    
    def _load_s3_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Load a single object from S3."""
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        
        # Determine content type
        content_type = response.get('ContentType', '')
        if not content_type:
            content_type = mimetypes.guess_type(key)[0] or 'application/octet-stream'
        
        return {
            'content': content,
            'content_type': content_type,
            'key': key,
            'bucket': bucket
        }
    
    def load_from_local(self, path: str, **kwargs) -> Union[Dataset, Iterator[Dict[str, Any]]]:
        """Load data from local storage."""
        path_obj = Path(path)
        
        if path_obj.is_dir():
            # Load directory as dataset
            if self.config.streaming:
                return self._stream_local_directory(path_obj)
            else:
                return self._load_local_directory(path_obj)
        else:
            # Load single file
            return self._load_local_file(path_obj)
    
    def _stream_local_directory(self, directory: Path) -> Iterator[Dict[str, Any]]:
        """Stream files from local directory."""
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                yield self._load_local_file(file_path)
    
    def _load_local_directory(self, directory: Path) -> Dataset:
        """Load local directory as HuggingFace Dataset."""
        # This is a simplified implementation
        # In production, you'd want to handle different data formats properly
        data = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                data.append(self._load_local_file(file_path))
        
        return Dataset.from_list(data)
    
    def _load_local_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single local file."""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        content_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
        
        return {
            'content': content,
            'content_type': content_type,
            'path': str(file_path)
        }
    
    def load(self, source: str, **kwargs) -> Union[Dataset, IterableDataset, Iterator[Dict[str, Any]]]:
        """Load data from any source."""
        format_type = self.detect_format(source)
        
        if format_type == DataFormat.HUGGINGFACE:
            return self.load_from_huggingface(source, **kwargs)
        elif format_type == DataFormat.S3:
            return self.load_from_s3(source, **kwargs)
        else:
            return self.load_from_local(source, **kwargs)


class MultiModalDataset:
    """Unified multi-modal dataset with automatic preprocessing."""
    
    def __init__(
        self,
        data_source: str,
        config: DataPipelineConfig,
        modality_mapping: Optional[Dict[str, ModalityType]] = None
    ):
        self.data_source = data_source
        self.config = config
        self.modality_mapping = modality_mapping or {}
        self.data_loader = DataLoader(config)
        self.processors = {}
        self._initialize_processors()
        self._dataset = None
        self._iterator = None
    
    def _initialize_processors(self):
        """Initialize modality processors."""
        for modality, modality_config in self.config.modalities.items():
            if modality_config.enabled:
                self.processors[modality] = ModalityProcessorRegistry.get_processor(
                    modality, modality_config
                )
    
    def _detect_modalities(self, sample: Dict[str, Any]) -> List[ModalityType]:
        """Detect modalities present in a sample."""
        modalities = []
        
        for key, value in sample.items():
            if key in self.modality_mapping:
                modalities.append(self.modality_mapping[key])
            else:
                # Auto-detect based on content type or key name
                if isinstance(value, str):
                    if any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        modalities.append(ModalityType.IMAGE)
                    elif any(ext in value.lower() for ext in ['.wav', '.mp3', '.flac']):
                        modalities.append(ModalityType.AUDIO)
                    elif any(ext in value.lower() for ext in ['.mp4', '.avi', '.mov']):
                        modalities.append(ModalityType.VIDEO)
                    else:
                        modalities.append(ModalityType.TEXT)
                elif isinstance(value, (Image.Image, np.ndarray, torch.Tensor)):
                    modalities.append(ModalityType.IMAGE)
                elif isinstance(value, bytes):
                    # Try to detect from content
                    content_type = self._detect_content_type(value)
                    if 'image' in content_type:
                        modalities.append(ModalityType.IMAGE)
                    elif 'audio' in content_type:
                        modalities.append(ModalityType.AUDIO)
                    elif 'video' in content_type:
                        modalities.append(ModalityType.VIDEO)
        
        return list(set(modalities))  # Remove duplicates
    
    def _detect_content_type(self, content: bytes) -> str:
        """Detect content type from bytes."""
        # Simple magic number detection
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        elif content[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        elif content[:4] in [b'RIFF', b'FORM']:
            return 'audio/wav'
        elif content[:3] == b'ID3' or content[:2] == b'\xff\xfb':
            return 'audio/mp3'
        else:
            return 'application/octet-stream'
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample through all modality processors."""
        features = {}
        
        # Detect modalities in this sample
        modalities = self._detect_modalities(sample)
        
        for modality in modalities:
            if modality in self.processors:
                processor = self.processors[modality]
                
                # Find data for this modality
                modality_data = None
                for key, value in sample.items():
                    detected_modality = self.modality_mapping.get(key)
                    if detected_modality == modality or (detected_modality is None and modality in self._detect_modalities({key: value})):
                        modality_data = value
                        break
                
                if modality_data is not None:
                    modality_features = processor.process(modality_data)
                    # Prefix feature keys with modality name
                    for feature_key, feature_value in modality_features.items():
                        features[f"{modality.value}_{feature_key}"] = feature_value
        
        return features
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        if self.config.streaming:
            yield from self._streaming_iter()
        else:
            yield from self._standard_iter()
    
    def _streaming_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Streaming iteration over dataset."""
        data_iter = self.data_loader.load(self.data_source)
        
        count = 0
        for sample in data_iter:
            if self.config.max_samples and count >= self.config.max_samples:
                break
            
            try:
                features = self._process_sample(sample)
                if features:  # Only yield if we extracted features
                    yield features
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
    
    def _standard_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Standard iteration over dataset."""
        if self._dataset is None:
            self._dataset = self.data_loader.load(self.data_source)
        
        if isinstance(self._dataset, Dataset):
            # HuggingFace Dataset
            if self.config.shuffle:
                self._dataset = self._dataset.shuffle(seed=self.config.seed)
            
            for i, sample in enumerate(self._dataset):
                if self.config.max_samples and i >= self.config.max_samples:
                    break
                
                try:
                    features = self._process_sample(sample)
                    if features:
                        yield features
                except Exception as e:
                    logger.warning(f"Failed to process sample {i}: {e}")
                    continue
        else:
            # Iterable or other format
            count = 0
            for sample in self._dataset:
                if self.config.max_samples and count >= self.config.max_samples:
                    break
                
                try:
                    features = self._process_sample(sample)
                    if features:
                        yield features
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to process sample: {e}")
                    continue
    
    def get_dataloader(self, batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        """Get PyTorch DataLoader for the dataset."""
        batch_size = batch_size or self.config.batch_size
        
        def collate_fn(batch):
            """Custom collate function for multi-modal data."""
            collated = {}
            
            for key in batch[0].keys():
                values = [item[key] for item in batch]
                
                if isinstance(values[0], torch.Tensor):
                    # Stack tensors
                    try:
                        collated[key] = torch.stack(values, dim=0)
                    except:
                        # If stacking fails, keep as list
                        collated[key] = values
                else:
                    # Keep as list for non-tensor data
                    collated[key] = values
            
            return collated
        
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            shuffle=self.config.shuffle and not self.config.streaming
        )


# Factory function for easy creation
def create_multimodal_dataset(
    data_source: str,
    modalities: Optional[List[ModalityType]] = None,
    **kwargs
) -> MultiModalDataset:
    """Create a multi-modal dataset with default configuration."""
    
    # Create default modality configs
    modality_configs = {}
    if modalities:
        for modality in modalities:
            modality_configs[modality] = ModalityConfig(
                modality=modality,
                enabled=True,
                preprocessing={},
                augmentation={},
                normalization={}
            )
    
    config = DataPipelineConfig(
        modalities=modality_configs,
        **kwargs
    )
    
    return MultiModalDataset(data_source, config)


# Export main classes and functions
__all__ = [
    "ModalityType",
    "DataFormat",
    "ModalityConfig",
    "DataPipelineConfig",
    "ModalityProcessor",
    "TextProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "ModalityProcessorRegistry",
    "DataLoader",
    "MultiModalDataset",
    "create_multimodal_dataset",
]