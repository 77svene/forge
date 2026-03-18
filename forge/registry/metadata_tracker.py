"""Model Registry and Versioning System for forge

This module provides comprehensive model management with Git-like versioning,
automatic metadata extraction, model card generation, and a central registry
for fine-tuned models. Designed for scalability and integration with the
existing forge ecosystem.
"""

import os
import json
import hashlib
import datetime
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Status of a model version"""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATION = "validation"
    READY = "ready"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class BranchType(Enum):
    """Types of branches in the model registry"""
    MAIN = "main"
    DEVELOPMENT = "development"
    EXPERIMENTAL = "experimental"
    RELEASE = "release"
    HOTFIX = "hotfix"


@dataclass
class TrainingMetadata:
    """Metadata extracted from training runs"""
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    training_duration: Optional[float] = None
    total_steps: Optional[int] = None
    total_epochs: Optional[int] = None
    checkpoint_path: Optional[str] = None
    base_model: Optional[str] = None
    adapter_type: Optional[str] = None
    quantization: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelCard:
    """Automatically generated model card"""
    model_id: str
    model_name: str
    version: str
    description: str = ""
    intended_use: str = ""
    limitations: str = ""
    ethical_considerations: str = ""
    training_data: str = ""
    evaluation_data: str = ""
    training_procedure: str = ""
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    citation: str = ""
    license: str = "apache-2.0"
    language: List[str] = field(default_factory=lambda: ["en"])
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_markdown(self) -> str:
        """Generate markdown model card"""
        md = f"""# {self.model_name} - {self.version}

## Model Description
{self.description}

## Intended Use
{self.intended_use}

## Limitations
{self.limitations}

## Ethical Considerations
{self.ethical_considerations}

## Training Data
{self.training_data}

## Evaluation Data
{self.evaluation_data}

## Training Procedure
{self.training_procedure}

## Evaluation Results
{self._format_metrics()}

## Citation
{self.citation}

## License
{self.license}

## Language
{', '.join(self.language)}

## Tags
{', '.join(self.tags)}

## Metadata
- **Model ID**: {self.model_id}
- **Created**: {self.created_at}
- **Updated**: {self.updated_at}
"""
        return md
    
    def _format_metrics(self) -> str:
        """Format evaluation metrics for markdown"""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        lines = []
        for dataset, metrics in self.evaluation_results.items():
            lines.append(f"### {dataset}")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {metric}: {value:.4f}")
                else:
                    lines.append(f"- {metric}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCard':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a version of a model with metadata and relationships"""
    version_id: str
    model_id: str
    version_number: str
    parent_versions: List[str] = field(default_factory=list)
    branch: str = "main"
    status: VersionStatus = VersionStatus.DRAFT
    metadata: Optional[TrainingMetadata] = None
    model_card: Optional[ModelCard] = None
    file_paths: Dict[str, str] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    created_by: Optional[str] = None
    
    def compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum for a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def add_file(self, file_path: str, logical_name: str) -> None:
        """Add a file to this version with checksum"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_paths[logical_name] = file_path
        self.checksums[logical_name] = self.compute_checksum(file_path)
        self.updated_at = datetime.datetime.now().isoformat()
    
    def verify_integrity(self) -> bool:
        """Verify all files have correct checksums"""
        for logical_name, file_path in self.file_paths.items():
            if not os.path.exists(file_path):
                return False
            current_checksum = self.compute_checksum(file_path)
            if current_checksum != self.checksums.get(logical_name):
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['status'] = self.status.value
        if self.metadata:
            data['metadata'] = self.metadata.to_dict()
        if self.model_card:
            data['model_card'] = self.model_card.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        # Convert string back to enum
        data['status'] = VersionStatus(data['status'])
        
        # Convert nested objects
        if data.get('metadata'):
            data['metadata'] = TrainingMetadata.from_dict(data['metadata'])
        if data.get('model_card'):
            data['model_card'] = ModelCard.from_dict(data['model_card'])
        
        return cls(**data)


@dataclass
class ModelBranch:
    """Represents a branch in the model version tree"""
    name: str
    model_id: str
    branch_type: BranchType = BranchType.DEVELOPMENT
    head_version: Optional[str] = None
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['branch_type'] = self.branch_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelBranch':
        """Create from dictionary"""
        data['branch_type'] = BranchType(data['branch_type'])
        return cls(**data)


@dataclass
class Model:
    """Represents a model in the registry"""
    model_id: str
    name: str
    description: str = ""
    base_model: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    default_branch: str = "main"
    branches: Dict[str, ModelBranch] = field(default_factory=dict)
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    created_by: Optional[str] = None
    is_public: bool = True
    
    def create_branch(self, name: str, branch_type: BranchType = BranchType.DEVELOPMENT, 
                     from_version: Optional[str] = None, description: str = "") -> ModelBranch:
        """Create a new branch"""
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")
        
        # If from_version is specified, use it as head, otherwise use default branch head
        if from_version:
            if from_version not in self.versions:
                raise ValueError(f"Version '{from_version}' not found")
            head_version = from_version
        else:
            default_branch = self.branches.get(self.default_branch)
            head_version = default_branch.head_version if default_branch else None
        
        branch = ModelBranch(
            name=name,
            model_id=self.model_id,
            branch_type=branch_type,
            head_version=head_version,
            description=description
        )
        
        self.branches[name] = branch
        self.updated_at = datetime.datetime.now().isoformat()
        return branch
    
    def create_version(self, version_number: str, branch: str = "main", 
                      parent_versions: Optional[List[str]] = None,
                      metadata: Optional[TrainingMetadata] = None,
                      model_card: Optional[ModelCard] = None) -> ModelVersion:
        """Create a new version of the model"""
        if branch not in self.branches:
            raise ValueError(f"Branch '{branch}' not found")
        
        # Generate version ID
        version_id = str(uuid.uuid4())
        
        # Set parent versions
        if parent_versions is None:
            # Default to current head of the branch
            branch_obj = self.branches[branch]
            if branch_obj.head_version:
                parent_versions = [branch_obj.head_version]
            else:
                parent_versions = []
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            model_id=self.model_id,
            version_number=version_number,
            parent_versions=parent_versions,
            branch=branch,
            metadata=metadata,
            model_card=model_card
        )
        
        # Update branch head
        self.branches[branch].head_version = version_id
        self.branches[branch].updated_at = datetime.datetime.now().isoformat()
        
        # Add version to model
        self.versions[version_id] = version
        self.updated_at = datetime.datetime.now().isoformat()
        
        return version
    
    def merge_branches(self, source_branch: str, target_branch: str, 
                      merge_message: str = "") -> ModelVersion:
        """Merge source branch into target branch"""
        if source_branch not in self.branches:
            raise ValueError(f"Source branch '{source_branch}' not found")
        if target_branch not in self.branches:
            raise ValueError(f"Target branch '{target_branch}' not found")
        
        source_head = self.branches[source_branch].head_version
        target_head = self.branches[target_branch].head_version
        
        if not source_head:
            raise ValueError(f"Source branch '{source_branch}' has no versions")
        
        # Create merge version
        parent_versions = [target_head] if target_head else []
        if source_head not in parent_versions:
            parent_versions.append(source_head)
        
        # Generate version number for merge
        current_version = self.versions[target_head].version_number if target_head else "0.0.0"
        version_parts = current_version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        merge_version = ".".join(version_parts)
        
        # Create merge version
        merge_version_obj = self.create_version(
            version_number=merge_version,
            branch=target_branch,
            parent_versions=parent_versions
        )
        
        # Copy metadata from source if target doesn't have it
        if not merge_version_obj.metadata and source_head in self.versions:
            source_version = self.versions[source_head]
            if source_version.metadata:
                merge_version_obj.metadata = source_version.metadata
        
        return merge_version_obj
    
    def get_version_history(self, branch: Optional[str] = None, 
                           limit: Optional[int] = None) -> List[ModelVersion]:
        """Get version history, optionally filtered by branch"""
        versions = list(self.versions.values())
        
        if branch:
            versions = [v for v in versions if v.branch == branch]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert branches and versions
        data['branches'] = {k: v.to_dict() for k, v in self.branches.items()}
        data['versions'] = {k: v.to_dict() for k, v in self.versions.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Create from dictionary"""
        # Convert branches and versions back from dicts
        branches_data = data.pop('branches', {})
        versions_data = data.pop('versions', {})
        
        model = cls(**data)
        
        # Reconstruct branches
        for branch_name, branch_data in branches_data.items():
            model.branches[branch_name] = ModelBranch.from_dict(branch_data)
        
        # Reconstruct versions
        for version_id, version_data in versions_data.items():
            model.versions[version_id] = ModelVersion.from_dict(version_data)
        
        return model


class MetadataTracker:
    """
    Main class for model registry and versioning system.
    
    Provides Git-like versioning for models with branching and merging,
    automatic metadata extraction from training runs, model card generation,
    and support for model sharing and discovery.
    """
    
    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize the metadata tracker.
        
        Args:
            registry_path: Path to the registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "cache").mkdir(exist_ok=True)
        (self.registry_path / "exports").mkdir(exist_ok=True)
        
        # Load registry index
        self.index_file = self.registry_path / "registry_index.json"
        self.models: Dict[str, Model] = {}
        self._load_index()
        
        logger.info(f"Initialized MetadataTracker at {self.registry_path}")
    
    def _load_index(self) -> None:
        """Load registry index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                
                for model_id, model_data in index_data.get('models', {}).items():
                    self.models[model_id] = Model.from_dict(model_data)
                
                logger.info(f"Loaded {len(self.models)} models from registry index")
            except Exception as e:
                logger.error(f"Failed to load registry index: {e}")
                self.models = {}
    
    def _save_index(self) -> None:
        """Save registry index to disk"""
        try:
            index_data = {
                'version': '1.0',
                'updated_at': datetime.datetime.now().isoformat(),
                'models': {model_id: model.to_dict() for model_id, model in self.models.items()}
            }
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            logger.debug("Saved registry index")
        except Exception as e:
            logger.error(f"Failed to save registry index: {e}")
    
    def _get_model_dir(self, model_id: str) -> Path:
        """Get directory for a specific model"""
        model_dir = self.registry_path / "models" / model_id
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    def _get_version_dir(self, model_id: str, version_id: str) -> Path:
        """Get directory for a specific version"""
        version_dir = self._get_model_dir(model_id) / "versions" / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir
    
    def register_model(self, name: str, description: str = "", 
                      base_model: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      created_by: Optional[str] = None) -> Model:
        """
        Register a new model in the registry.
        
        Args:
            name: Name of the model
            description: Description of the model
            base_model: Base model used for fine-tuning
            tags: Tags for categorization
            created_by: Creator identifier
            
        Returns:
            Created Model object
        """
        # Generate model ID
        model_id = hashlib.sha256(f"{name}:{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Create model
        model = Model(
            model_id=model_id,
            name=name,
            description=description,
            base_model=base_model,
            tags=tags or [],
            created_by=created_by
        )
        
        # Create default main branch
        model.create_branch("main", BranchType.MAIN, description="Main development branch")
        
        # Add to registry
        self.models[model_id] = model
        self._save_index()
        
        logger.info(f"Registered model '{name}' with ID {model_id}")
        return model
    
    def extract_training_metadata(self, training_run_dir: Union[str, Path]) -> TrainingMetadata:
        """
        Extract metadata from a training run directory.
        
        Args:
            training_run_dir: Path to training run directory
            
        Returns:
            Extracted TrainingMetadata
        """
        training_run_dir = Path(training_run_dir)
        metadata = TrainingMetadata()
        
        # Look for common training output files
        config_files = [
            "training_config.json",
            "config.json",
            "trainer_config.json",
            "hyperparameters.json"
        ]
        
        for config_file in config_files:
            config_path = training_run_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    if "training_config" in config_file:
                        metadata.training_config = config_data
                    elif "hyperparameters" in config_file:
                        metadata.hyperparameters = config_data
                    else:
                        metadata.training_config.update(config_data)
                except Exception as e:
                    logger.warning(f"Failed to load {config_file}: {e}")
        
        # Look for training logs
        log_files = list(training_run_dir.glob("*.log")) + list(training_run_dir.glob("events.out.tfevents.*"))
        if log_files:
            # Simple log parsing for metrics
            for log_file in log_files[:1]:  # Just check first log file
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                    
                    # Extract metrics from logs (simplified)
                    if "train/loss" in content:
                        metadata.training_metrics["final_loss"] = 0.0  # Placeholder
                except:
                    pass
        
        # Look for evaluation results
        eval_files = [
            "eval_results.json",
            "evaluation.json",
            "metrics.json"
        ]
        
        for eval_file in eval_files:
            eval_path = training_run_dir / eval_file
            if eval_path.exists():
                try:
                    with open(eval_path, 'r') as f:
                        eval_data = json.load(f)
                    metadata.evaluation_metrics = eval_data
                except Exception as e:
                    logger.warning(f"Failed to load {eval_file}: {e}")
        
        # Extract hardware info if available
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            metadata.hardware_info["gpu"] = os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Set checkpoint path if exists
        checkpoint_dirs = list(training_run_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            metadata.checkpoint_path = str(checkpoint_dirs[0])
        
        return metadata
    
    def create_version_from_training(self, model_id: str, 
                                   training_run_dir: Union[str, Path],
                                   version_number: str,
                                   branch: str = "main",
                                   generate_card: bool = True) -> ModelVersion:
        """
        Create a new version from a training run.
        
        Args:
            model_id: ID of the model
            training_run_dir: Path to training run directory
            version_number: Version number (e.g., "1.0.0")
            branch: Branch to create version on
            generate_card: Whether to generate a model card
            
        Returns:
            Created ModelVersion
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Extract metadata
        metadata = self.extract_training_metadata(training_run_dir)
        metadata.base_model = model.base_model
        
        # Create model card if requested
        model_card = None
        if generate_card:
            model_card = self._generate_model_card(model, version_number, metadata)
        
        # Create version
        version = model.create_version(
            version_number=version_number,
            branch=branch,
            metadata=metadata,
            model_card=model_card
        )
        
        # Copy model files to version directory
        training_run_dir = Path(training_run_dir)
        version_dir = self._get_version_dir(model_id, version.version_id)
        
        # Copy important files
        important_files = [
            "adapter_model.bin",
            "adapter_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
            "model.safetensors"
        ]
        
        for file_name in important_files:
            src_file = training_run_dir / file_name
            if src_file.exists():
                dst_file = version_dir / file_name
                shutil.copy2(src_file, dst_file)
                version.add_file(str(dst_file), file_name)
        
        # Save metadata and model card
        if version.metadata:
            metadata_file = version_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(version.metadata.to_dict(), f, indent=2)
            version.add_file(str(metadata_file), "metadata.json")
        
        if version.model_card:
            card_file = version_dir / "model_card.md"
            with open(card_file, 'w') as f:
                f.write(version.model_card.to_markdown())
            version.add_file(str(card_file), "model_card.md")
        
        # Update registry
        self._save_index()
        
        logger.info(f"Created version {version_number} for model {model_id}")
        return version
    
    def _generate_model_card(self, model: Model, version_number: str, 
                           metadata: TrainingMetadata) -> ModelCard:
        """
        Generate a model card from metadata.
        
        Args:
            model: Model object
            version_number: Version number
            metadata: Training metadata
            
        Returns:
            Generated ModelCard
        """
        # Format training data description
        training_data_desc = ""
        if metadata.dataset_info:
            dataset_name = metadata.dataset_info.get("name", "Unknown dataset")
            dataset_size = metadata.dataset_info.get("size", "Unknown size")
            training_data_desc = f"Trained on {dataset_name} ({dataset_size} examples)"
        
        # Format training procedure
        training_procedure = ""
        if metadata.hyperparameters:
            training_procedure = "## Training Hyperparameters\n"
            for key, value in metadata.hyperparameters.items():
                training_procedure += f"- **{key}**: {value}\n"
        
        # Format evaluation results
        evaluation_results = {}
        if metadata.evaluation_metrics:
            evaluation_results["validation"] = metadata.evaluation_metrics
        
        # Create model card
        model_card = ModelCard(
            model_id=model.model_id,
            model_name=model.name,
            version=version_number,
            description=model.description,
            intended_use="This model is intended for research and experimentation.",
            limitations="This model has not been tested in production environments.",
            ethical_considerations="Users should be aware of potential biases in the training data.",
            training_data=training_data_desc,
            training_procedure=training_procedure,
            evaluation_results=evaluation_results,
            tags=model.tags,
            metrics=metadata.training_metrics
        )
        
        return model_card
    
    def create_branch(self, model_id: str, branch_name: str, 
                     branch_type: BranchType = BranchType.DEVELOPMENT,
                     from_version: Optional[str] = None,
                     description: str = "") -> ModelBranch:
        """
        Create a new branch for a model.
        
        Args:
            model_id: ID of the model
            branch_name: Name of the new branch
            branch_type: Type of branch
            from_version: Version to branch from (optional)
            description: Branch description
            
        Returns:
            Created ModelBranch
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        branch = model.create_branch(
            name=branch_name,
            branch_type=branch_type,
            from_version=from_version,
            description=description
        )
        
        self._save_index()
        logger.info(f"Created branch '{branch_name}' for model {model_id}")
        return branch
    
    def merge_branches(self, model_id: str, source_branch: str, 
                      target_branch: str, merge_message: str = "") -> ModelVersion:
        """
        Merge one branch into another.
        
        Args:
            model_id: ID of the model
            source_branch: Source branch name
            target_branch: Target branch name
            merge_message: Merge commit message
            
        Returns:
            Created merge version
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        merge_version = model.merge_branches(source_branch, target_branch, merge_message)
        
        self._save_index()
        logger.info(f"Merged branch '{source_branch}' into '{target_branch}' for model {model_id}")
        return merge_version
    
    def tag_version(self, model_id: str, version_id: str, tag: str) -> None:
        """
        Add a tag to a version.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version
            tag: Tag to add
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        if version_id not in model.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = model.versions[version_id]
        if tag not in version.tags:
            version.tags.append(tag)
            version.updated_at = datetime.datetime.now().isoformat()
            
            self._save_index()
            logger.info(f"Tagged version {version_id} with '{tag}'")
    
    def search_models(self, query: str = "", tags: Optional[List[str]] = None,
                     base_model: Optional[str] = None,
                     limit: int = 10) -> List[Model]:
        """
        Search for models in the registry.
        
        Args:
            query: Search query (matches name and description)
            tags: Filter by tags
            base_model: Filter by base model
            limit: Maximum number of results
            
        Returns:
            List of matching models
        """
        results = []
        
        for model in self.models.values():
            # Check visibility
            if not model.is_public:
                continue
            
            # Check query match
            if query:
                query_lower = query.lower()
                if (query_lower not in model.name.lower() and 
                    query_lower not in model.description.lower()):
                    continue
            
            # Check tags
            if tags:
                if not all(tag in model.tags for tag in tags):
                    continue
            
            # Check base model
            if base_model and model.base_model != base_model:
                continue
            
            results.append(model)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_model_card(self, model_id: str, version_id: Optional[str] = None) -> Optional[str]:
        """
        Get model card for a specific version.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version (uses latest if not specified)
            
        Returns:
            Model card markdown or None
        """
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        if version_id:
            if version_id not in model.versions:
                return None
            version = model.versions[version_id]
        else:
            # Get latest version from main branch
            main_branch = model.branches.get("main")
            if not main_branch or not main_branch.head_version:
                return None
            version = model.versions.get(main_branch.head_version)
            if not version:
                return None
        
        if not version.model_card:
            return None
        
        return version.model_card.to_markdown()
    
    def export_model(self, model_id: str, version_id: str, 
                    export_dir: Union[str, Path]) -> Path:
        """
        Export a model version to a directory.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version
            export_dir: Directory to export to
            
        Returns:
            Path to exported directory
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        if version_id not in model.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = model.versions[version_id]
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for logical_name, file_path in version.file_paths.items():
            src_path = Path(file_path)
            if src_path.exists():
                dst_path = export_dir / src_path.name
                shutil.copy2(src_path, dst_path)
        
        # Save metadata
        metadata_file = export_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Save model card
        if version.model_card:
            card_file = export_dir / "README.md"
            with open(card_file, 'w') as f:
                f.write(version.model_card.to_markdown())
        
        logger.info(f"Exported model {model_id} version {version_id} to {export_dir}")
        return export_dir
    
    def get_version_tree(self, model_id: str) -> Dict[str, Any]:
        """
        Get the version tree for a model (for visualization).
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary representing the version tree
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        tree = {
            "model_id": model_id,
            "model_name": model.name,
            "branches": {},
            "versions": {},
            "tags": {}
        }
        
        # Add branches
        for branch_name, branch in model.branches.items():
            tree["branches"][branch_name] = {
                "type": branch.branch_type.value,
                "head": branch.head_version,
                "description": branch.description
            }
        
        # Add versions
        for version_id, version in model.versions.items():
            tree["versions"][version_id] = {
                "version_number": version.version_number,
                "branch": version.branch,
                "parents": version.parent_versions,
                "status": version.status.value,
                "created_at": version.created_at,
                "tags": version.tags
            }
            
            # Add tags
            for tag in version.tags:
                if tag not in tree["tags"]:
                    tree["tags"][tag] = []
                tree["tags"][tag].append(version_id)
        
        return tree
    
    def verify_integrity(self, model_id: str, version_id: str) -> bool:
        """
        Verify the integrity of a model version.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version
            
        Returns:
            True if integrity check passes
        """
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        if version_id not in model.versions:
            return False
        
        version = model.versions[version_id]
        return version.verify_integrity()
    
    def cleanup_old_versions(self, model_id: str, keep_last: int = 5) -> int:
        """
        Clean up old versions, keeping only the most recent ones.
        
        Args:
            model_id: ID of the model
            keep_last: Number of recent versions to keep per branch
            
        Returns:
            Number of versions removed
        """
        if model_id not in self.models:
            return 0
        
        model = self.models[model_id]
        removed_count = 0
        
        # Group versions by branch
        branch_versions = defaultdict(list)
        for version_id, version in model.versions.items():
            branch_versions[version.branch].append((version_id, version.created_at))
        
        # Sort each branch by creation time (newest first)
        for branch in branch_versions:
            branch_versions[branch].sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the most recent versions
            to_keep = set()
            for version_id, _ in branch_versions[branch][:keep_last]:
                to_keep.add(version_id)
            
            # Remove old versions
            for version_id, _ in branch_versions[branch][keep_last:]:
                if version_id in model.versions:
                    # Remove version directory
                    version_dir = self._get_version_dir(model_id, version_id)
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    
                    # Remove from model
                    del model.versions[version_id]
                    removed_count += 1
        
        if removed_count > 0:
            self._save_index()
            logger.info(f"Cleaned up {removed_count} old versions for model {model_id}")
        
        return removed_count


# Factory function for easy integration
def create_metadata_tracker(registry_path: Optional[Union[str, Path]] = None) -> MetadataTracker:
    """
    Create a MetadataTracker instance.
    
    Args:
        registry_path: Path to registry directory. If None, uses default location.
        
    Returns:
        MetadataTracker instance
    """
    if registry_path is None:
        # Use default location in user's home directory
        registry_path = Path.home() / ".forge" / "model_registry"
    
    return MetadataTracker(registry_path)


# Integration with existing forge training
def integrate_with_trainer(trainer, model_name: str, registry_path: Optional[Union[str, Path]] = None):
    """
    Integrate metadata tracking with a forge trainer.
    
    Args:
        trainer: forge trainer instance
        model_name: Name for the model in registry
        registry_path: Path to registry directory
    """
    tracker = create_metadata_tracker(registry_path)
    
    # Register model if not exists
    models = tracker.search_models(query=model_name)
    if not models:
        model = tracker.register_model(
            name=model_name,
            description=f"Fine-tuned model based on {trainer.model.config.model_type}",
            base_model=trainer.model.config.model_type,
            tags=["fine-tuned", trainer.model.config.model_type]
        )
        model_id = model.model_id
    else:
        model_id = models[0].model_id
    
    # Hook into training completion
    original_save = trainer.save_model
    
    def save_with_metadata(output_dir: str, **kwargs):
        # Call original save
        original_save(output_dir, **kwargs)
        
        # Create version in registry
        try:
            version_number = f"1.0.{len(tracker.models[model_id].versions)}"
            tracker.create_version_from_training(
                model_id=model_id,
                training_run_dir=output_dir,
                version_number=version_number,
                branch="main",
                generate_card=True
            )
            logger.info(f"Created version {version_number} in registry")
        except Exception as e:
            logger.error(f"Failed to create version in registry: {e}")
    
    # Replace save method
    trainer.save_model = save_with_metadata
    
    return tracker, model_id


# Example usage and CLI integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Model Registry CLI")
    parser.add_argument("--registry-path", type=str, help="Path to registry directory")
    parser.add_argument("--action", type=str, required=True,
                       choices=["register", "list", "version", "branch", "merge", "tag", "export", "card"],
                       help="Action to perform")
    
    # Action-specific arguments
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--model-id", type=str, help="Model ID")
    parser.add_argument("--version", type=str, help="Version number")
    parser.add_argument("--branch", type=str, help="Branch name")
    parser.add_argument("--tag", type=str, help="Tag name")
    parser.add_argument("--training-dir", type=str, help="Training run directory")
    parser.add_argument("--export-dir", type=str, help="Export directory")
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = create_metadata_tracker(args.registry_path)
    
    if args.action == "register":
        if not args.model_name:
            print("Error: --model-name is required for register action")
        else:
            model = tracker.register_model(args.model_name)
            print(f"Registered model: {model.model_id}")
    
    elif args.action == "list":
        models = tracker.search_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  {model.model_id}: {model.name}")
    
    elif args.action == "version":
        if not args.model_id or not args.training_dir or not args.version:
            print("Error: --model-id, --training-dir, and --version are required")
        else:
            version = tracker.create_version_from_training(
                args.model_id,
                args.training_dir,
                args.version
            )
            print(f"Created version: {version.version_id}")
    
    elif args.action == "branch":
        if not args.model_id or not args.branch:
            print("Error: --model-id and --branch are required")
        else:
            branch = tracker.create_branch(args.model_id, args.branch)
            print(f"Created branch: {branch.name}")
    
    elif args.action == "merge":
        if not args.model_id or not args.branch:
            print("Error: --model-id and --branch are required")
        else:
            # Assuming branch is source, main is target
            version = tracker.merge_branches(args.model_id, args.branch, "main")
            print(f"Merged branch, created version: {version.version_id}")
    
    elif args.action == "tag":
        if not args.model_id or not args.tag:
            print("Error: --model-id and --tag are required")
        else:
            # Tag the latest version
            model = tracker.models[args.model_id]
            main_branch = model.branches.get("main")
            if main_branch and main_branch.head_version:
                tracker.tag_version(args.model_id, main_branch.head_version, args.tag)
                print(f"Tagged version with: {args.tag}")
    
    elif args.action == "export":
        if not args.model_id or not args.export_dir:
            print("Error: --model-id and --export-dir are required")
        else:
            model = tracker.models[args.model_id]
            main_branch = model.branches.get("main")
            if main_branch and main_branch.head_version:
                export_path = tracker.export_model(
                    args.model_id,
                    main_branch.head_version,
                    args.export_dir
                )
                print(f"Exported to: {export_path}")
    
    elif args.action == "card":
        if not args.model_id:
            print("Error: --model-id is required")
        else:
            card = tracker.get_model_card(args.model_id)
            if card:
                print(card)
            else:
                print("No model card available")