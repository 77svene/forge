# Copyright 2025 the forge team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import json
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from PIL import Image

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


# Multi-modal processor registry
MODALITY_PROCESSORS = {}


@dataclass
class ModalityConfig:
    """Configuration for modality processing."""
    augmentation: bool = False
    normalization: bool = True
    target_format: Optional[str] = None
    max_samples: Optional[int] = None


def register_modality_processor(modality: str):
    """Decorator to register a modality processor."""
    def decorator(cls):
        MODALITY_PROCESSORS[modality] = cls
        return cls
    return decorator


class BaseModalityProcessor(ABC):
    """Base class for modality-specific processors."""
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        self.config = config or ModalityConfig()
    
    @abstractmethod
    def detect_format(self, data: Any) -> str:
        """Detect the format of the input data."""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any, format: str) -> Any:
        """Preprocess data with augmentation and normalization."""
        pass
    
    @abstractmethod
    def extract_features(self, data: Any, format: str) -> Dict[str, Any]:
        """Extract features from preprocessed data."""
        pass
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Full processing pipeline."""
        format = self.detect_format(data)
        preprocessed = self.preprocess(data, format)
        return self.extract_features(preprocessed, format)


@register_modality_processor("text")
class TextModalityProcessor(BaseModalityProcessor):
    """Processor for text data."""
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, str):
            return "text"
        elif isinstance(data, dict) and "text" in data:
            return "text_dict"
        else:
            raise ValueError(f"Unsupported text format: {type(data)}")
    
    def preprocess(self, data: Any, format: str) -> str:
        if format == "text":
            text = data
        elif format == "text_dict":
            text = data["text"]
        
        # Basic text normalization
        if self.config.normalization:
            text = text.strip()
            # Remove extra whitespace
            text = " ".join(text.split())
        
        return text
    
    def extract_features(self, data: str, format: str) -> Dict[str, Any]:
        return {"text": data, "modality": "text"}


@register_modality_processor("image")
class ImageModalityProcessor(BaseModalityProcessor):
    """Processor for image data."""
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        super().__init__(config)
        self.target_size = (224, 224)  # Default target size
        self.supported_formats = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, Image.Image):
            return "pil"
        elif isinstance(data, np.ndarray):
            return "numpy"
        elif isinstance(data, bytes):
            return "bytes"
        elif isinstance(data, str):
            ext = Path(data).suffix.lower().lstrip(".")
            if ext in self.supported_formats:
                return ext
        elif isinstance(data, dict):
            if "image" in data:
                return "dict"
            elif "path" in data:
                return "path_dict"
        
        raise ValueError(f"Unsupported image format: {type(data)}")
    
    def preprocess(self, data: Any, format: str) -> Image.Image:
        # Convert to PIL Image
        if format == "pil":
            image = data
        elif format == "numpy":
            image = Image.fromarray(data)
        elif format == "bytes":
            image = Image.open(io.BytesIO(data))
        elif format in self.supported_formats:
            image = Image.open(data)
        elif format == "dict":
            image_data = data["image"]
            if isinstance(image_data, Image.Image):
                image = image_data
            elif isinstance(image_data, str):
                image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError(f"Unsupported image data in dict: {type(image_data)}")
        elif format == "path_dict":
            image = Image.open(data["path"])
        else:
            raise ValueError(f"Unsupported format for preprocessing: {format}")
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Normalization (convert to numpy array and normalize)
        if self.config.normalization:
            img_array = np.array(image, dtype=np.float32) / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Augmentation placeholder
        if self.config.augmentation:
            # Add your augmentation logic here
            pass
        
        return image
    
    def extract_features(self, data: Image.Image, format: str) -> Dict[str, Any]:
        img_array = np.array(data, dtype=np.float32)
        return {
            "image": img_array,
            "image_shape": img_array.shape,
            "modality": "image",
        }


@register_modality_processor("audio")
class AudioModalityProcessor(BaseModalityProcessor):
    """Processor for audio data."""
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        super().__init__(config)
        self.target_sr = 16000  # Target sample rate
        self.supported_formats = {"wav", "mp3", "flac", "ogg", "m4a"}
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, np.ndarray):
            return "numpy"
        elif isinstance(data, bytes):
            return "bytes"
        elif isinstance(data, str):
            ext = Path(data).suffix.lower().lstrip(".")
            if ext in self.supported_formats:
                return ext
        elif isinstance(data, dict):
            if "array" in data:
                return "numpy_dict"
            elif "path" in data:
                return "path_dict"
            elif "bytes" in data:
                return "bytes_dict"
        
        raise ValueError(f"Unsupported audio format: {type(data)}")
    
    def preprocess(self, data: Any, format: str) -> np.ndarray:
        # Try to import audio processing libraries
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            logger.warning("librosa or soundfile not installed. Audio processing limited.")
            librosa = None
            sf = None
        
        if format == "numpy":
            audio_array = data
            sr = getattr(data, "sample_rate", self.target_sr)
        elif format == "numpy_dict":
            audio_array = data["array"]
            sr = data.get("sampling_rate", self.target_sr)
        elif format == "bytes":
            if sf is None:
                raise ImportError("soundfile is required for bytes audio processing")
            audio_array, sr = sf.read(io.BytesIO(data))
        elif format == "bytes_dict":
            if sf is None:
                raise ImportError("soundfile is required for bytes audio processing")
            audio_array, sr = sf.read(io.BytesIO(data["bytes"]))
        elif format in self.supported_formats:
            if librosa is None:
                raise ImportError("librosa is required for file audio processing")
            audio_array, sr = librosa.load(data, sr=None)
        elif format == "path_dict":
            if librosa is None:
                raise ImportError("librosa is required for file audio processing")
            audio_array, sr = librosa.load(data["path"], sr=None)
        else:
            raise ValueError(f"Unsupported format for preprocessing: {format}")
        
        # Resample if needed
        if librosa and sr != self.target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.target_sr)
        
        # Normalization
        if self.config.normalization and len(audio_array) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Augmentation placeholder
        if self.config.augmentation:
            # Add your augmentation logic here (e.g., noise injection, time stretching)
            pass
        
        return audio_array
    
    def extract_features(self, data: np.ndarray, format: str) -> Dict[str, Any]:
        return {
            "audio": data,
            "audio_length": len(data),
            "modality": "audio",
        }


@register_modality_processor("video")
class VideoModalityProcessor(BaseModalityProcessor):
    """Processor for video data."""
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        super().__init__(config)
        self.target_fps = 30
        self.max_frames = 100
        self.frame_size = (224, 224)
    
    def detect_format(self, data: Any) -> str:
        if isinstance(data, str):
            ext = Path(data).suffix.lower().lstrip(".")
            if ext in {"mp4", "avi", "mov", "mkv", "webm"}:
                return ext
        elif isinstance(data, dict):
            if "frames" in data:
                return "frames_dict"
            elif "path" in data:
                return "path_dict"
            elif "bytes" in data:
                return "bytes_dict"
        
        raise ValueError(f"Unsupported video format: {type(data)}")
    
    def preprocess(self, data: Any, format: str) -> List[np.ndarray]:
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for video processing")
        
        frames = []
        
        if format in {"mp4", "avi", "mov", "mkv", "webm"}:
            cap = cv2.VideoCapture(data)
        elif format == "path_dict":
            cap = cv2.VideoCapture(data["path"])
        elif format == "bytes_dict":
            # Save bytes to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(data["bytes"])
                temp_path = f.name
            cap = cv2.VideoCapture(temp_path)
            os.unlink(temp_path)
        elif format == "frames_dict":
            frames_data = data["frames"]
            for frame in frames_data:
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
                elif isinstance(frame, Image.Image):
                    frames.append(np.array(frame))
            cap = None
        else:
            raise ValueError(f"Unsupported format for preprocessing: {format}")
        
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.target_fps))
            
            frame_count = 0
            while cap.isOpened() and len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize
                    frame = cv2.resize(frame, self.frame_size)
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
        
        # Normalization
        if self.config.normalization:
            frames = [frame.astype(np.float32) / 255.0 for frame in frames]
        
        # Augmentation placeholder
        if self.config.augmentation:
            # Add your augmentation logic here
            pass
        
        return frames
    
    def extract_features(self, data: List[np.ndarray], format: str) -> Dict[str, Any]:
        if len(data) == 0:
            return {"video": np.array([]), "video_frames": 0, "modality": "video"}
        
        video_array = np.stack(data, axis=0)
        return {
            "video": video_array,
            "video_frames": len(data),
            "video_shape": video_array.shape,
            "modality": "video",
        }


class MultiModalDataLoader:
    """Unified data loader for multi-modal datasets with streaming support."""
    
    def __init__(
        self,
        data_args: "DataArguments",
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.modality_processors = {}
        self._init_processors()
    
    def _init_processors(self):
        """Initialize modality processors based on configuration."""
        for modality, processor_cls in MODALITY_PROCESSORS.items():
            config = ModalityConfig(
                augmentation=self.data_args.augmentation,
                normalization=self.data_args.normalization,
            )
            self.modality_processors[modality] = processor_cls(config)
    
    def detect_modality(self, sample: Dict[str, Any]) -> List[str]:
        """Detect modalities present in a sample."""
        modalities = []
        
        for key, value in sample.items():
            if key == "text" or (isinstance(value, str) and len(value) > 0):
                modalities.append("text")
            elif key in ["image", "images"] or self._is_image_data(value):
                modalities.append("image")
            elif key in ["audio", "audios"] or self._is_audio_data(value):
                modalities.append("audio")
            elif key in ["video", "videos"] or self._is_video_data(value):
                modalities.append("video")
        
        return list(set(modalities))
    
    def _is_image_data(self, data: Any) -> bool:
        """Check if data is image-like."""
        if isinstance(data, Image.Image):
            return True
        if isinstance(data, np.ndarray) and len(data.shape) in [2, 3]:
            return True
        if isinstance(data, bytes):
            try:
                Image.open(io.BytesIO(data))
                return True
            except:
                return False
        return False
    
    def _is_audio_data(self, data: Any) -> bool:
        """Check if data is audio-like."""
        if isinstance(data, np.ndarray) and len(data.shape) == 1:
            return True
        if isinstance(data, bytes):
            # Basic check for audio file signatures
            audio_signatures = [b'RIFF', b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2']
            return any(data.startswith(sig) for sig in audio_signatures)
        return False
    
    def _is_video_data(self, data: Any) -> bool:
        """Check if data is video-like."""
        if isinstance(data, str):
            ext = Path(data).suffix.lower()
            return ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if isinstance(data, bytes):
            # Basic check for video file signatures
            video_signatures = [b'\x00\x00\x00\x18ftypmp4', b'\x00\x00\x00\x20ftypisom']
            return any(data.startswith(sig) for sig in video_signatures)
        return False
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample with all modalities."""
        processed = {}
        modalities = self.detect_modality(sample)
        
        for modality in modalities:
            if modality in self.modality_processors:
                processor = self.modality_processors[modality]
                
                # Extract data for this modality
                if modality == "text":
                    data = sample.get("text", "")
                elif modality == "image":
                    data = sample.get("image", sample.get("images"))
                elif modality == "audio":
                    data = sample.get("audio", sample.get("audios"))
                elif modality == "video":
                    data = sample.get("video", sample.get("videos"))
                else:
                    continue
                
                if data is not None:
                    try:
                        features = processor.process(data)
                        processed.update(features)
                    except Exception as e:
                        logger.warning(f"Failed to process {modality} data: {e}")
        
        # Copy over other fields
        for key, value in sample.items():
            if key not in processed and key not in ["text", "image", "images", "audio", "audios", "video", "videos"]:
                processed[key] = value
        
        return processed
    
    def load_dataset(
        self,
        dataset_attr: "DatasetAttr",
        streaming: Optional[bool] = None,
    ) -> Union[Dataset, IterableDataset]:
        """Load dataset with streaming support."""
        streaming = streaming if streaming is not None else self.data_args.streaming
        
        # Load raw dataset
        dataset = self._load_raw_dataset(dataset_attr, streaming)
        
        # Process dataset with modality processors
        if streaming:
            dataset = self._process_streaming_dataset(dataset)
        else:
            dataset = self._process_dataset(dataset)
        
        return dataset
    
    def _load_raw_dataset(
        self,
        dataset_attr: "DatasetAttr",
        streaming: bool,
    ) -> Union[Dataset, IterableDataset]:
        """Load raw dataset from various sources."""
        data_path, data_name, data_dir, data_files = None, None, None, None
        
        if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
            data_path = dataset_attr.dataset_name
            data_name = dataset_attr.subset
            data_dir = dataset_attr.folder
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(self.data_args.dataset_dir, dataset_attr.dataset_name)
            data_name = dataset_attr.subset
            data_dir = dataset_attr.folder
        elif dataset_attr.load_from == "cloud_file":
            data_path = dataset_attr.dataset_name
        elif dataset_attr.load_from == "file":
            data_files = []
            local_path = os.path.join(self.data_args.dataset_dir, dataset_attr.dataset_name)
            if os.path.isdir(local_path):
                for file_name in os.listdir(local_path):
                    data_files.append(os.path.join(local_path, file_name))
            elif os.path.isfile(local_path):
                data_files.append(local_path)
            else:
                raise ValueError(f"File {local_path} not found.")
            
            data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
            if data_path is None:
                raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
            
            if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
                raise ValueError("File types should be identical.")
        else:
            raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")
        
        # Load from different sources
        if dataset_attr.load_from == "ms_hub":
            check_version("modelscope>=1.14.0", mandatory=True)
            from modelscope import MsDataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            
            cache_dir = self.model_args.cache_dir or MS_DATASETS_CACHE
            dataset = MsDataset.load(
                dataset_name=data_path,
                subset_name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=dataset_attr.split,
                cache_dir=cache_dir,
                token=self.model_args.ms_hub_token,
                use_streaming=streaming,
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        
        elif dataset_attr.load_from == "om_hub":
            check_version("openmind>=0.8.0", mandatory=True)
            from openmind import OmDataset
            from openmind.utils.hub import OM_DATASETS_CACHE
            
            cache_dir = self.model_args.cache_dir or OM_DATASETS_CACHE
            dataset = OmDataset.load_dataset(
                path=data_path,
                name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=dataset_attr.split,
                cache_dir=cache_dir,
                token=self.model_args.om_hub_token,
                streaming=streaming,
            )
        
        elif dataset_attr.load_from == "cloud_file":
            dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
        
        else:
            dataset = load_dataset(
                path=data_path,
                name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=dataset_attr.split,
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
                num_proc=self.data_args.preprocessing_num_workers,
                streaming=streaming and dataset_attr.load_from != "file",
            )
            if streaming and dataset_attr.load_from == "file":
                dataset = dataset.to_iterable_dataset(num_shards=self.training_args.dataloader_num_workers)
        
        # Handle sampling
        if dataset_attr.num_samples is not None and not streaming:
            target_num = dataset_attr.num_samples
            indexes = np.random.permutation(len(dataset))[:target_num]
            target_num -= len(indexes)
            if target_num > 0:
                expand_indexes = np.random.choice(len(dataset), target_num)
                indexes = np.concatenate((indexes, expand_indexes), axis=0)
            
            assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
            dataset = dataset.select(indexes)
            logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")
        
        if self.data_args.max_samples is not None:
            if streaming:
                logger.warning("max_samples is ignored for streaming datasets")
            else:
                max_samples = min(self.data_args.max_samples, len(dataset))
                dataset = dataset.select(range(max_samples))
        
        return dataset
    
    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a regular (non-streaming) dataset."""
        def process_batch(batch):
            processed_batch = {key: [] for key in batch.keys()}
            
            for i in range(len(batch[next(iter(batch))])):
                sample = {key: batch[key][i] for key in batch.keys()}
                processed_sample = self.process_sample(sample)
                
                for key, value in processed_sample.items():
                    if key not in processed_batch:
                        processed_batch[key] = [None] * i
                    processed_batch[key].append(value)
                
                # Ensure all keys have the same length
                for key in processed_batch:
                    if len(processed_batch[key]) < i + 1:
                        processed_batch[key].append(None)
            
            return processed_batch
        
        return dataset.map(
            process_batch,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            num_proc=self.data_args.preprocessing_num_workers,
        )
    
    def _process_streaming_dataset(self, dataset: IterableDataset) -> IterableDataset:
        """Process a streaming dataset."""
        def process_generator():
            for sample in dataset:
                yield self.process_sample(sample)
        
        return IterableDataset.from_generator(process_generator)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    
    # Use multi-modal loader
    loader = MultiModalDataLoader(data_args, model_args, training_args)
    dataset = loader.load_dataset(dataset_attr)
    
    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor = PretrainDatasetProcessor(template, tokenizer, data_args, do_generate)
    elif stage == "sft":
        if data_args.packing:
            dataset_processor = PackedSupervisedDatasetProcessor(template, tokenizer, data_args, do_generate)
        else:
            dataset_processor = SupervisedDatasetProcessor(template, tokenizer, data_args, do_generate)
    elif stage == "rm":
        dataset_processor = PairwiseDatasetProcessor(template, tokenizer, data_args)
    elif stage == "ppo":
        dataset_processor = UnsupervisedDatasetProcessor(template, tokenizer, data_args)
    elif stage == "kto":
        dataset_processor = FeedbackDatasetProcessor(template, tokenizer, data_args)
    else:
        raise NotImplementedError(f"Unknown stage: {stage}.")

    return dataset_processor