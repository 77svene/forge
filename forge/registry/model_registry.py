"""
Model Registry and Versioning System for forge
Provides Git-like versioning for models with branching, merging, and automated model card generation.
"""

import os
import json
import hashlib
import datetime
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
import yaml
import logging

# Integration with existing forge modules
try:
    from ..extras.logging import get_logger
    from ..extras.misc import get_current_device, count_parameters
    from ..model import load_model_and_tokenizer
    from ..train import TrainingArguments
    from ..hparams import get_train_args
except ImportError:
    # Fallback for standalone usage
    def get_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    model_id: str
    version: str
    base_model: str
    model_type: str
    created_at: str
    created_by: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    model_size: Optional[int] = None
    parameters_count: Optional[int] = None
    training_dataset: Optional[str] = None
    training_steps: Optional[int] = None
    training_time: Optional[float] = None
    license: str = "apache-2.0"
    language: List[str] = field(default_factory=lambda: ["en"])
    framework: str = "pytorch"
    parent_version: Optional[str] = None
    branch: str = "main"
    commit_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a version of a model."""
    metadata: ModelMetadata
    model_path: str
    model_card_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None
    training_logs_path: Optional[str] = None
    evaluation_results_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "model_path": self.model_path,
            "model_card_path": self.model_card_path,
            "tokenizer_path": self.tokenizer_path,
            "config_path": self.config_path,
            "training_logs_path": self.training_logs_path,
            "evaluation_results_path": self.evaluation_results_path,
        }


class ModelCardGenerator:
    """Generates model cards with training details and performance metrics."""
    
    @staticmethod
    def generate_model_card(
        metadata: ModelMetadata,
        training_args: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None
    ) -> str:
        """
        Generate a model card in markdown format.
        
        Args:
            metadata: Model metadata
            training_args: Training arguments used
            evaluation_results: Evaluation results
            template: Custom template string
        
        Returns:
            Model card as markdown string
        """
        if template:
            return template.format(
                model_name=metadata.model_id,
                version=metadata.version,
                base_model=metadata.base_model,
                description=metadata.description,
                created_at=metadata.created_at,
                created_by=metadata.created_by,
                tags=", ".join(metadata.tags),
                license=metadata.license,
                language=", ".join(metadata.language),
                parameters=metadata.parameters_count,
                training_steps=metadata.training_steps,
                training_time=metadata.training_time,
                training_dataset=metadata.training_dataset,
                performance_metrics=json.dumps(metadata.performance_metrics, indent=2),
                training_config=json.dumps(metadata.training_config, indent=2),
            )
        
        # Default template
        model_card = f"""---
license: {metadata.license}
language: {metadata.language}
tags:
{chr(10).join(f'- {tag}' for tag in metadata.tags)}
datasets:
{f'- {metadata.training_dataset}' if metadata.training_dataset else ''}
metrics:
{chr(10).join(f'- {metric}: {value}' for metric, value in metadata.performance_metrics.items())}
model-index:
- name: {metadata.model_id}
  results:
{chr(10).join(f'  - task: {{name: {metric}}}\n    metrics:\n    - type: {metric}\n      value: {value}' for metric, value in metadata.performance_metrics.items())}
---

# {metadata.model_id}

## Model Description

**Model Name:** {metadata.model_id}  
**Version:** {metadata.version}  
**Base Model:** {metadata.base_model}  
**Model Type:** {metadata.model_type}  
**Created:** {metadata.created_at}  
**Created By:** {metadata.created_by}  
**License:** {metadata.license}  
**Language:** {', '.join(metadata.language)}  

{metadata.description}

## Model Details

- **Parameters:** {metadata.parameters_count:,} parameters
- **Model Size:** {metadata.model_size} bytes
- **Framework:** {metadata.framework}
- **Branch:** {metadata.branch}

## Training Details

"""
        
        if training_args:
            model_card += f"""### Training Configuration

```json
{json.dumps(training_args, indent=2)}
```

"""
        
        if metadata.training_steps:
            model_card += f"""### Training Statistics

- **Training Steps:** {metadata.training_steps:,}
- **Training Time:** {metadata.training_time:.2f} hours
- **Training Dataset:** {metadata.training_dataset}

"""
        
        if evaluation_results:
            model_card += f"""### Evaluation Results

```json
{json.dumps(evaluation_results, indent=2)}
```

"""
        
        if metadata.performance_metrics:
            model_card += f"""### Performance Metrics

| Metric | Value |
|--------|-------|
{chr(10).join(f'| {metric} | {value} |' for metric, value in metadata.performance_metrics.items())}

"""
        
        model_card += f"""## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{metadata.model_id}")
tokenizer = AutoTokenizer.from_pretrained("{metadata.model_id}")

# Example usage
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{{{metadata.model_id.replace('/', '_')},
  author = {{{metadata.created_by}}},
  title = {{{metadata.model_id}}},
  year = {{{metadata.created_at[:4]}}},
  version = {{{metadata.version}}},
  url = {{https://huggingface.co/{metadata.model_id}}},
}}
```

## License

This model is licensed under {metadata.license}.

---
*Generated by forge Model Registry*
"""
        
        return model_card


class ModelRegistry:
    """
    Git-like model registry with versioning, branching, and merging capabilities.
    
    Features:
    - Git-like version control for models
    - Branching and merging support
    - Automatic metadata extraction
    - Model card generation
    - Central registry for model sharing and discovery
    """
    
    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry directory
        """
        self.registry_path = Path(registry_path)
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.branches_dir = self.registry_path / "branches"
        self.refs_dir = self.registry_path / "refs"
        self.config_path = self.registry_path / "config.json"
        
        # Create directory structure
        self._init_registry()
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _init_registry(self):
        """Initialize registry directory structure."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.branches_dir.mkdir(exist_ok=True)
        self.refs_dir.mkdir(exist_ok=True)
        
        # Create default branches
        main_branch_dir = self.branches_dir / "main"
        main_branch_dir.mkdir(exist_ok=True)
        
        # Create HEAD file pointing to main branch
        head_path = self.registry_path / "HEAD"
        if not head_path.exists():
            head_path.write_text("ref: refs/heads/main")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load registry configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "registry_version": "1.0.0",
                "default_branch": "main",
                "auto_generate_model_cards": True,
                "store_training_logs": True,
                "max_versions_per_model": 100,
                "compression": "none"
            }
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save registry configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_current_branch(self) -> str:
        """Get current branch name."""
        head_path = self.registry_path / "HEAD"
        if head_path.exists():
            head_content = head_path.read_text().strip()
            if head_content.startswith("ref: refs/heads/"):
                return head_content.split("/")[-1]
        return self.config["default_branch"]
    
    def _get_branch_commit(self, branch: str) -> Optional[str]:
        """Get the latest commit hash for a branch."""
        branch_file = self.refs_dir / branch
        if branch_file.exists():
            return branch_file.read_text().strip()
        return None
    
    def _set_branch_commit(self, branch: str, commit_hash: str):
        """Set the commit hash for a branch."""
        branch_file = self.refs_dir / branch
        branch_file.write_text(commit_hash)
    
    def _generate_commit_hash(self, content: str) -> str:
        """Generate a commit hash from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_model_dir(self, model_id: str) -> Path:
        """Get directory for a specific model."""
        # Convert model_id to filesystem-safe name
        safe_name = model_id.replace("/", "_")
        return self.models_dir / safe_name
    
    def _get_version_dir(self, model_id: str, version: str) -> Path:
        """Get directory for a specific model version."""
        model_dir = self._get_model_dir(model_id)
        return model_dir / "versions" / version
    
    def _get_metadata_path(self, model_id: str, version: str) -> Path:
        """Get path to metadata file for a model version."""
        version_dir = self._get_version_dir(model_id, version)
        return version_dir / "metadata.json"
    
    def _get_model_card_path(self, model_id: str, version: str) -> Path:
        """Get path to model card file for a model version."""
        version_dir = self._get_version_dir(model_id, version)
        return version_dir / "model_card.md"
    
    def _extract_metadata_from_training(
        self,
        model_id: str,
        model_path: str,
        training_args: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ModelMetadata:
        """
        Extract metadata from a trained model.
        
        Args:
            model_id: Model identifier
            model_path: Path to the model
            training_args: Training arguments
            performance_metrics: Performance metrics
            **kwargs: Additional metadata
        
        Returns:
            ModelMetadata object
        """
        model_path = Path(model_path)
        
        # Extract basic model information
        model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        
        # Try to load model config to get parameter count
        config_path = model_path / "config.json"
        parameters_count = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Estimate parameter count from config
                    if "num_parameters" in config:
                        parameters_count = config["num_parameters"]
                    elif "hidden_size" in config and "num_hidden_layers" in config:
                        # Rough estimate for transformer models
                        hidden_size = config.get("hidden_size", 0)
                        num_layers = config.get("num_hidden_layers", 0)
                        vocab_size = config.get("vocab_size", 0)
                        parameters_count = (hidden_size * hidden_size * 4 * num_layers + 
                                          hidden_size * vocab_size)
            except Exception as e:
                logger.warning(f"Could not extract parameter count: {e}")
        
        # Extract training information from kwargs
        training_steps = kwargs.get("training_steps")
        training_time = kwargs.get("training_time")
        training_dataset = kwargs.get("training_dataset")
        base_model = kwargs.get("base_model", "unknown")
        model_type = kwargs.get("model_type", "causal_lm")
        description = kwargs.get("description", "")
        tags = kwargs.get("tags", [])
        
        # Generate version hash
        version_content = f"{model_id}{datetime.datetime.now().isoformat()}"
        version_hash = self._generate_commit_hash(version_content)
        
        # Get current branch
        current_branch = self._get_current_branch()
        
        # Get parent version if exists
        parent_version = None
        if current_branch:
            parent_commit = self._get_branch_commit(current_branch)
            if parent_commit:
                parent_version = parent_commit
        
        metadata = ModelMetadata(
            model_id=model_id,
            version=version_hash,
            base_model=base_model,
            model_type=model_type,
            created_at=datetime.datetime.now().isoformat(),
            created_by=kwargs.get("created_by", "forge_user"),
            description=description,
            tags=tags,
            training_config=training_args or {},
            performance_metrics=performance_metrics or {},
            model_size=model_size,
            parameters_count=parameters_count,
            training_dataset=training_dataset,
            training_steps=training_steps,
            training_time=training_time,
            license=kwargs.get("license", "apache-2.0"),
            language=kwargs.get("language", ["en"]),
            framework=kwargs.get("framework", "pytorch"),
            parent_version=parent_version,
            branch=current_branch,
            commit_hash=version_hash
        )
        
        return metadata
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        training_args: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Register a new model or a new version of an existing model.
        
        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b")
            model_path: Path to the model directory
            tokenizer_path: Path to the tokenizer directory
            training_args: Training arguments used
            performance_metrics: Performance metrics from evaluation
            description: Model description
            tags: List of tags for the model
            **kwargs: Additional metadata
        
        Returns:
            Version hash of the registered model
        """
        logger.info(f"Registering model {model_id}")
        
        # Extract metadata
        metadata = self._extract_metadata_from_training(
            model_id=model_id,
            model_path=model_path,
            training_args=training_args,
            performance_metrics=performance_metrics,
            description=description,
            tags=tags or [],
            **kwargs
        )
        
        # Create version directory
        version_dir = self._get_version_dir(model_id, metadata.version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_dest = version_dir / "model"
        if Path(model_path).exists():
            if model_dest.exists():
                shutil.rmtree(model_dest)
            shutil.copytree(model_path, model_dest)
        
        # Copy tokenizer if provided
        tokenizer_dest = None
        if tokenizer_path and Path(tokenizer_path).exists():
            tokenizer_dest = version_dir / "tokenizer"
            if tokenizer_dest.exists():
                shutil.rmtree(tokenizer_dest)
            shutil.copytree(tokenizer_path, tokenizer_dest)
        
        # Save metadata
        metadata_path = self._get_metadata_path(model_id, metadata.version)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Generate and save model card if enabled
        model_card_path = None
        if self.config.get("auto_generate_model_cards", True):
            model_card_content = ModelCardGenerator.generate_model_card(
                metadata=metadata,
                training_args=training_args,
                evaluation_results=performance_metrics
            )
            model_card_path = self._get_model_card_path(model_id, metadata.version)
            with open(model_card_path, 'w') as f:
                f.write(model_card_content)
        
        # Update branch reference
        current_branch = self._get_current_branch()
        self._set_branch_commit(current_branch, metadata.version)
        
        # Create model version object
        model_version = ModelVersion(
            metadata=metadata,
            model_path=str(model_dest),
            model_card_path=str(model_card_path) if model_card_path else None,
            tokenizer_path=str(tokenizer_dest) if tokenizer_dest else None
        )
        
        # Save version index
        self._update_version_index(model_id, metadata.version, model_version)
        
        logger.info(f"Model {model_id} registered with version {metadata.version}")
        return metadata.version
    
    def _update_version_index(self, model_id: str, version: str, model_version: ModelVersion):
        """Update the version index for a model."""
        model_dir = self._get_model_dir(model_id)
        index_path = model_dir / "index.json"
        
        # Load existing index or create new
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"model_id": model_id, "versions": {}}
        
        # Add new version
        index["versions"][version] = model_version.to_dict()
        index["latest_version"] = version
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        """
        Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            from_branch: Source branch to create from (defaults to current branch)
        
        Returns:
            True if successful, False otherwise
        """
        if from_branch is None:
            from_branch = self._get_current_branch()
        
        # Get commit hash from source branch
        source_commit = self._get_branch_commit(from_branch)
        if not source_commit:
            logger.error(f"Source branch {from_branch} has no commits")
            return False
        
        # Create new branch pointing to same commit
        self._set_branch_commit(branch_name, source_commit)
        
        logger.info(f"Created branch {branch_name} from {from_branch}")
        return True
    
    def checkout_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Name of the branch to checkout
        
        Returns:
            True if successful, False otherwise
        """
        # Check if branch exists
        branch_file = self.refs_dir / branch_name
        if not branch_file.exists():
            logger.error(f"Branch {branch_name} does not exist")
            return False
        
        # Update HEAD
        head_path = self.registry_path / "HEAD"
        head_path.write_text(f"ref: refs/heads/{branch_name}")
        
        logger.info(f"Switched to branch {branch_name}")
        return True
    
    def get_model_version(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version hash (if None, returns latest version)
        
        Returns:
            ModelVersion object or None if not found
        """
        model_dir = self._get_model_dir(model_id)
        index_path = model_dir / "index.json"
        
        if not index_path.exists():
            return None
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        if version is None:
            version = index.get("latest_version")
            if not version:
                return None
        
        version_data = index["versions"].get(version)
        if not version_data:
            return None
        
        return ModelVersion(
            metadata=ModelMetadata.from_dict(version_data["metadata"]),
            model_path=version_data["model_path"],
            model_card_path=version_data.get("model_card_path"),
            tokenizer_path=version_data.get("tokenizer_path"),
            config_path=version_data.get("config_path"),
            training_logs_path=version_data.get("training_logs_path"),
            evaluation_results_path=version_data.get("evaluation_results_path")
        )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                index_path = model_dir / "index.json"
                if index_path.exists():
                    with open(index_path, 'r') as f:
                        index = json.load(f)
                    
                    latest_version = index.get("latest_version")
                    if latest_version:
                        version_data = index["versions"].get(latest_version, {})
                        metadata = version_data.get("metadata", {})
                        
                        models.append({
                            "model_id": index["model_id"],
                            "latest_version": latest_version,
                            "description": metadata.get("description", ""),
                            "tags": metadata.get("tags", []),
                            "created_at": metadata.get("created_at"),
                            "parameters_count": metadata.get("parameters_count"),
                            "performance_metrics": metadata.get("performance_metrics", {})
                        })
        
        return models
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_parameters: Optional[int] = None,
        max_parameters: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for models in the registry.
        
        Args:
            query: Search query for model name or description
            tags: Filter by tags
            min_parameters: Minimum parameter count
            max_parameters: Maximum parameter count
        
        Returns:
            List of matching models
        """
        all_models = self.list_models()
        results = []
        
        for model in all_models:
            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in model["model_id"].lower() and 
                    query_lower not in model.get("description", "").lower()):
                    continue
            
            # Filter by tags
            if tags:
                model_tags = set(model.get("tags", []))
                if not set(tags).intersection(model_tags):
                    continue
            
            # Filter by parameter count
            params = model.get("parameters_count")
            if params is not None:
                if min_parameters and params < min_parameters:
                    continue
                if max_parameters and params > max_parameters:
                    continue
            
            results.append(model)
        
        return results
    
    def merge_branches(
        self,
        source_branch: str,
        target_branch: str,
        strategy: str = "theirs"
    ) -> bool:
        """
        Merge one branch into another.
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            strategy: Merge strategy ("theirs", "ours", or "manual")
        
        Returns:
            True if successful, False otherwise
        """
        source_commit = self._get_branch_commit(source_branch)
        target_commit = self._get_branch_commit(target_branch)
        
        if not source_commit or not target_commit:
            logger.error("One or both branches have no commits")
            return False
        
        # For simplicity, we'll implement a basic merge strategy
        # In a production system, this would involve more complex conflict resolution
        
        if strategy == "theirs":
            # Use source branch's commit
            self._set_branch_commit(target_branch, source_commit)
            logger.info(f"Merged {source_branch} into {target_branch} using 'theirs' strategy")
            return True
        elif strategy == "ours":
            # Keep target branch's commit (no-op merge)
            logger.info(f"Merged {source_branch} into {target_branch} using 'ours' strategy (no changes)")
            return True
        else:
            logger.error(f"Merge strategy '{strategy}' not implemented")
            return False
    
    def get_model_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            List of version history entries
        """
        model_dir = self._get_model_dir(model_id)
        index_path = model_dir / "index.json"
        
        if not index_path.exists():
            return []
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        history = []
        for version, version_data in index["versions"].items():
            metadata = version_data["metadata"]
            history.append({
                "version": version,
                "created_at": metadata["created_at"],
                "description": metadata["description"],
                "branch": metadata["branch"],
                "parent_version": metadata.get("parent_version"),
                "performance_metrics": metadata.get("performance_metrics", {})
            })
        
        # Sort by creation date
        history.sort(key=lambda x: x["created_at"], reverse=True)
        return history
    
    def export_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        export_path: Optional[str] = None,
        format: str = "huggingface"
    ) -> str:
        """
        Export a model to a specific format.
        
        Args:
            model_id: Model identifier
            version: Version to export (defaults to latest)
            export_path: Path to export to
            format: Export format ("huggingface", "pytorch", "onnx")
        
        Returns:
            Path to exported model
        """
        model_version = self.get_model_version(model_id, version)
        if not model_version:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        if export_path is None:
            export_path = f"./exported_models/{model_id}_{model_version.metadata.version}"
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "huggingface":
            # Copy model files
            model_source = Path(model_version.model_path)
            if model_source.exists():
                for item in model_source.iterdir():
                    if item.is_file():
                        shutil.copy2(item, export_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, export_dir / item.name, dirs_exist_ok=True)
            
            # Copy tokenizer if available
            if model_version.tokenizer_path:
                tokenizer_source = Path(model_version.tokenizer_path)
                if tokenizer_source.exists():
                    for item in tokenizer_source.iterdir():
                        if item.is_file():
                            shutil.copy2(item, export_dir / item.name)
            
            # Save model card
            if model_version.model_card_path:
                model_card_source = Path(model_version.model_card_path)
                if model_card_source.exists():
                    shutil.copy2(model_card_source, export_dir / "README.md")
        
        elif format == "pytorch":
            # For PyTorch format, we'd need to convert the model
            # This is a simplified implementation
            logger.warning("PyTorch export format not fully implemented")
            shutil.copytree(model_version.model_path, export_dir, dirs_exist_ok=True)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Model exported to {export_dir}")
        return str(export_dir)
    
    def delete_model_version(self, model_id: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version to delete
        
        Returns:
            True if successful, False otherwise
        """
        model_dir = self._get_model_dir(model_id)
        version_dir = self._get_version_dir(model_id, version)
        
        if not version_dir.exists():
            logger.error(f"Version {version} not found for model {model_id}")
            return False
        
        # Remove version directory
        shutil.rmtree(version_dir)
        
        # Update index
        index_path = model_dir / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            
            if version in index["versions"]:
                del index["versions"][version]
                
                # Update latest version if needed
                if index.get("latest_version") == version:
                    if index["versions"]:
                        # Find the most recent version
                        latest = max(
                            index["versions"].items(),
                            key=lambda x: x[1]["metadata"]["created_at"]
                        )
                        index["latest_version"] = latest[0]
                    else:
                        index["latest_version"] = None
                
                with open(index_path, 'w') as f:
                    json.dump(index, f, indent=2)
        
        logger.info(f"Deleted version {version} of model {model_id}")
        return True


# Integration with existing forge training
def register_trained_model(
    model,
    tokenizer,
    training_args: TrainingArguments,
    output_dir: str,
    performance_metrics: Optional[Dict[str, float]] = None,
    registry_path: str = "./model_registry",
    **kwargs
) -> str:
    """
    Register a trained model with the registry.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        training_args: Training arguments
        output_dir: Output directory where model is saved
        performance_metrics: Performance metrics from evaluation
        registry_path: Path to model registry
        **kwargs: Additional metadata
    
    Returns:
        Version hash of registered model
    """
    # Initialize registry
    registry = ModelRegistry(registry_path)
    
    # Extract model ID from training args or kwargs
    model_id = kwargs.get("model_id", training_args.output_dir.split("/")[-1])
    
    # Prepare training arguments for metadata
    training_config = {
        "model_name_or_path": training_args.model_name_or_path,
        "finetuning_type": training_args.finetuning_type,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "bf16": training_args.bf16,
        "lora_rank": training_args.lora_rank,
        "lora_alpha": training_args.lora_alpha,
        "lora_dropout": training_args.lora_dropout,
        # Add more training args as needed
    }
    
    # Register model
    version = registry.register_model(
        model_id=model_id,
        model_path=output_dir,
        tokenizer_path=output_dir,  # Tokenizer is usually in the same directory
        training_args=training_config,
        performance_metrics=performance_metrics,
        base_model=training_args.model_name_or_path,
        model_type="causal_lm",
        **kwargs
    )
    
    return version


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry("./my_model_registry")
    
    # Create a new branch
    registry.create_branch("experiment-1")
    
    # Register a model (simulated)
    version = registry.register_model(
        model_id="my-org/my-fine-tuned-model",
        model_path="./output/my-model",
        description="Fine-tuned Llama model for code generation",
        tags=["code", "llama", "fine-tuned"],
        training_steps=10000,
        training_time=24.5,
        training_dataset="codeparrot/github-code",
        performance_metrics={"bleu": 45.2, "rouge-l": 52.1}
    )
    
    # List all models
    models = registry.list_models()
    print(f"Found {len(models)} models")
    
    # Search for models
    results = registry.search_models(query="code", tags=["fine-tuned"])
    print(f"Search results: {results}")
    
    # Get model history
    history = registry.get_model_history("my-org/my-fine-tuned-model")
    print(f"Model history: {len(history)} versions")
    
    # Export model
    export_path = registry.export_model(
        model_id="my-org/my-fine-tuned-model",
        format="huggingface"
    )
    print(f"Model exported to: {export_path}")