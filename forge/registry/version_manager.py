"""
Model Registry and Versioning System for forge
Provides Git-like versioning for models with branching, merging, metadata tracking,
automated model card generation, and central registry for model sharing/discovery.
"""

import os
import json
import hashlib
import time
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import yaml
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """Status of a model version"""
    DRAFT = "draft"
    TRAINING = "training"
    COMPLETED = "completed"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class MergeStrategy(Enum):
    """Merge strategies for model versions"""
    FAST_FORWARD = "fast_forward"
    THREE_WAY = "three_way"
    REBASE = "rebase"
    SQUASH = "squash"

@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    model_name: str
    base_model: str
    dataset_info: Dict[str, Any]
    training_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    hardware_info: Dict[str, Any]
    training_duration: float
    training_steps: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    description: str = ""
    license: str = "apache-2.0"
    authors: List[str] = field(default_factory=list)
    language: List[str] = field(default_factory=lambda: ["en"])
    tasks: List[str] = field(default_factory=lambda: ["text-generation"])
    library_name: str = "transformers"
    model_type: str = ""
    architectures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        return cls(**data)

@dataclass
class ModelVersion:
    """Represents a specific version of a model"""
    version_id: str
    model_name: str
    branch: str
    commit_hash: str
    parent_hashes: List[str]
    metadata: ModelMetadata
    model_path: str
    status: VersionStatus = VersionStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""
    diff_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['metadata'] = self.metadata.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create version from dictionary"""
        data['status'] = VersionStatus(data['status'])
        data['metadata'] = ModelMetadata.from_dict(data['metadata'])
        return cls(**data)

@dataclass
class ModelBranch:
    """Represents a branch in the model repository"""
    name: str
    head_version_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    is_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert branch to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelBranch':
        """Create branch from dictionary"""
        return cls(**data)

@dataclass
class ModelRepository:
    """Represents a model repository with all versions and branches"""
    model_name: str
    description: str
    owner: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    default_branch: str = "main"
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    branches: Dict[str, ModelBranch] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)  # tag -> version_id
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert repository to dictionary"""
        data = asdict(self)
        data['versions'] = {k: v.to_dict() for k, v in self.versions.items()}
        data['branches'] = {k: v.to_dict() for k, v in self.branches.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRepository':
        """Create repository from dictionary"""
        versions = {k: ModelVersion.from_dict(v) for k, v in data.get('versions', {}).items()}
        branches = {k: ModelBranch.from_dict(v) for k, v in data.get('branches', {}).items()}
        data['versions'] = versions
        data['branches'] = branches
        return cls(**data)

class VersionManager:
    """
    Git-like version manager for forge models.
    Provides comprehensive model management with version control, metadata tracking,
    and automated model card generation.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the VersionManager.
        
        Args:
            registry_path: Path to the model registry. If None, uses ~/.forge/registry
        """
        if registry_path is None:
            registry_path = os.path.join(os.path.expanduser("~"), ".forge", "registry")
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry index
        self.index_path = self.registry_path / "index.json"
        self.repositories: Dict[str, ModelRepository] = {}
        self._load_index()
        
        # Thread lock for concurrent operations
        self._lock = threading.RLock()
        
        # Model card template
        self.model_card_template = self._get_default_model_card_template()
        
        logger.info(f"VersionManager initialized with registry at {self.registry_path}")
    
    def _load_index(self) -> None:
        """Load the registry index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    index_data = json.load(f)
                
                for model_name, repo_data in index_data.get('repositories', {}).items():
                    self.repositories[model_name] = ModelRepository.from_dict(repo_data)
                
                logger.info(f"Loaded {len(self.repositories)} repositories from index")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self.repositories = {}
    
    def _save_index(self) -> None:
        """Save the registry index to disk"""
        with self._lock:
            index_data = {
                'repositories': {name: repo.to_dict() for name, repo in self.repositories.items()},
                'updated_at': datetime.now().isoformat()
            }
            
            # Atomic write using temporary file
            temp_path = self.index_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            temp_path.replace(self.index_path)
    
    def _get_repository_path(self, model_name: str) -> Path:
        """Get the path to a model repository"""
        return self.registry_path / model_name
    
    def _get_version_path(self, model_name: str, version_id: str) -> Path:
        """Get the path to a specific version"""
        return self._get_repository_path(model_name) / "versions" / version_id
    
    def _get_branch_path(self, model_name: str, branch_name: str) -> Path:
        """Get the path to a branch reference"""
        return self._get_repository_path(model_name) / "refs" / "heads" / branch_name
    
    def _compute_commit_hash(self, model_path: str, metadata: Dict[str, Any], parent_hashes: List[str]) -> str:
        """Compute a unique hash for a commit"""
        # Create a deterministic representation
        content = {
            'model_path': model_path,
            'metadata': metadata,
            'parent_hashes': sorted(parent_hashes),
            'timestamp': time.time()
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _extract_training_metadata(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from training configuration.
        Integrates with forge's training pipeline.
        """
        metadata = {
            'training_config': training_config,
            'hyperparameters': {},
            'dataset_info': {},
            'hardware_info': {}
        }
        
        # Extract hyperparameters
        if 'learning_rate' in training_config:
            metadata['hyperparameters']['learning_rate'] = training_config['learning_rate']
        if 'batch_size' in training_config:
            metadata['hyperparameters']['batch_size'] = training_config['batch_size']
        if 'num_epochs' in training_config:
            metadata['hyperparameters']['num_epochs'] = training_config['num_epochs']
        if 'warmup_steps' in training_config:
            metadata['hyperparameters']['warmup_steps'] = training_config['warmup_steps']
        if 'weight_decay' in training_config:
            metadata['hyperparameters']['weight_decay'] = training_config['weight_decay']
        
        # Extract dataset info
        if 'dataset' in training_config:
            metadata['dataset_info']['name'] = training_config['dataset']
        if 'dataset_path' in training_config:
            metadata['dataset_info']['path'] = training_config['dataset_path']
        if 'dataset_samples' in training_config:
            metadata['dataset_info']['samples'] = training_config['dataset_samples']
        
        # Extract hardware info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                metadata['hardware_info']['gpu'] = torch.cuda.get_device_name(0)
                metadata['hardware_info']['gpu_count'] = torch.cuda.device_count()
        except ImportError:
            pass
        
        return metadata
    
    def _generate_model_card(self, version: ModelVersion) -> str:
        """
        Generate a model card for a given version.
        Uses Hugging Face model card format with forge-specific details.
        """
        metadata = version.metadata
        
        # Prepare template variables
        template_vars = {
            'model_name': metadata.model_name,
            'version_id': version.version_id,
            'base_model': metadata.base_model,
            'description': metadata.description,
            'created_at': version.created_at,
            'training_duration': f"{metadata.training_duration:.2f} hours",
            'training_steps': metadata.training_steps,
            'performance_metrics': metadata.performance_metrics,
            'hyperparameters': metadata.hyperparameters,
            'dataset_info': metadata.dataset_info,
            'hardware_info': metadata.hardware_info,
            'tags': ', '.join(metadata.tags),
            'license': metadata.license,
            'authors': ', '.join(metadata.authors),
            'language': ', '.join(metadata.language),
            'tasks': ', '.join(metadata.tasks),
            'library_name': metadata.library_name,
            'model_type': metadata.model_type,
            'architectures': ', '.join(metadata.architectures),
            'branch': version.branch,
            'commit_hash': version.commit_hash,
            'message': version.message
        }
        
        # Generate model card from template
        model_card = self.model_card_template.format(**template_vars)
        return model_card
    
    def _get_default_model_card_template(self) -> str:
        """Get the default model card template"""
        return """---
language:
{language}
tags:
{tags}
license: {license}
datasets:
- {dataset_info_name}
metrics:
- accuracy
model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
    metrics:
      - name: accuracy
        type: accuracy
        value: {accuracy}
---

# {model_name} - Version {version_id}

{description}

## Model Details

- **Base Model**: {base_model}
- **Version**: {version_id}
- **Branch**: {branch}
- **Commit**: {commit_hash}
- **Created**: {created_at}
- **License**: {license}

## Training Details

- **Training Duration**: {training_duration}
- **Training Steps**: {training_steps}
- **Dataset**: {dataset_info_name}
- **Hardware**: {hardware_info_gpu}

### Hyperparameters

{hyperparameters_table}

## Performance Metrics

{performance_metrics_table}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Configuration

```json
{training_config}
```

## Citation

If you use this model, please cite:

```bibtex
@software{{forge_{model_name},
  author = {{{authors}}},
  title = {{{model_name}}},
  year = {{{year}}},
  publisher = {{forge}},
  journal = {{GitHub repository}},
  howpublished = {{\\url{{https://github.com/yourusername/forge}}}}
}}
```

---
*This model card was automatically generated by forge Version Manager*
"""
    
    def create_repository(self, model_name: str, description: str = "", 
                         owner: str = "anonymous") -> ModelRepository:
        """
        Create a new model repository.
        
        Args:
            model_name: Name of the model/repository
            description: Description of the model
            owner: Owner of the repository
            
        Returns:
            Created ModelRepository
        """
        with self._lock:
            if model_name in self.repositories:
                raise ValueError(f"Repository {model_name} already exists")
            
            # Create repository directory structure
            repo_path = self._get_repository_path(model_name)
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "versions").mkdir(exist_ok=True)
            (repo_path / "refs" / "heads").mkdir(parents=True, exist_ok=True)
            (repo_path / "objects").mkdir(exist_ok=True)
            
            # Create repository
            repo = ModelRepository(
                model_name=model_name,
                description=description,
                owner=owner
            )
            
            # Create default branch
            default_branch = ModelBranch(
                name="main",
                head_version_id="",
                is_default=True,
                description="Default branch"
            )
            repo.branches["main"] = default_branch
            
            # Save repository
            self.repositories[model_name] = repo
            self._save_index()
            
            # Save repository metadata
            repo_meta_path = repo_path / "repository.json"
            with open(repo_meta_path, 'w') as f:
                json.dump(repo.to_dict(), f, indent=2)
            
            logger.info(f"Created repository {model_name}")
            return repo
    
    def commit(self, model_name: str, model_path: str, training_config: Dict[str, Any],
              branch: str = "main", message: str = "", 
              performance_metrics: Optional[Dict[str, float]] = None,
              tags: Optional[List[str]] = None) -> ModelVersion:
        """
        Commit a new version of a model.
        
        Args:
            model_name: Name of the model repository
            model_path: Path to the model files
            training_config: Training configuration
            branch: Branch to commit to
            message: Commit message
            performance_metrics: Performance metrics
            tags: Tags for the version
            
        Returns:
            Created ModelVersion
        """
        with self._lock:
            if model_name not in self.repositories:
                raise ValueError(f"Repository {model_name} does not exist")
            
            repo = self.repositories[model_name]
            
            if branch not in repo.branches:
                raise ValueError(f"Branch {branch} does not exist")
            
            branch_obj = repo.branches[branch]
            
            # Extract metadata from training config
            extracted_metadata = self._extract_training_metadata(training_config)
            
            # Get parent hash
            parent_hashes = []
            if branch_obj.head_version_id:
                parent_hashes.append(branch_obj.head_version_id)
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                base_model=training_config.get('base_model', 'unknown'),
                dataset_info=extracted_metadata['dataset_info'],
                training_config=training_config,
                hyperparameters=extracted_metadata['hyperparameters'],
                performance_metrics=performance_metrics or {},
                hardware_info=extracted_metadata['hardware_info'],
                training_duration=training_config.get('training_duration', 0.0),
                training_steps=training_config.get('training_steps', 0),
                tags=tags or [],
                description=training_config.get('description', ''),
                model_type=training_config.get('model_type', ''),
                architectures=training_config.get('architectures', [])
            )
            
            # Compute commit hash
            commit_hash = self._compute_commit_hash(
                model_path, metadata.to_dict(), parent_hashes
            )
            
            # Create version ID
            version_id = f"v{len(repo.versions) + 1}-{commit_hash[:8]}"
            
            # Create version directory
            version_path = self._get_version_path(model_name, version_id)
            version_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model files to version directory
            if os.path.exists(model_path):
                if os.path.isdir(model_path):
                    shutil.copytree(model_path, version_path / "model")
                else:
                    shutil.copy2(model_path, version_path / "model")
            
            # Create version
            version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                branch=branch,
                commit_hash=commit_hash,
                parent_hashes=parent_hashes,
                metadata=metadata,
                model_path=str(version_path / "model"),
                status=VersionStatus.COMPLETED,
                message=message
            )
            
            # Update repository
            repo.versions[version_id] = version
            branch_obj.head_version_id = version_id
            branch_obj.updated_at = datetime.now().isoformat()
            repo.updated_at = datetime.now().isoformat()
            
            # Save version metadata
            version_meta_path = version_path / "version.json"
            with open(version_meta_path, 'w') as f:
                json.dump(version.to_dict(), f, indent=2)
            
            # Generate and save model card
            model_card = self._generate_model_card(version)
            model_card_path = version_path / "MODEL_CARD.md"
            with open(model_card_path, 'w') as f:
                f.write(model_card)
            
            # Update index
            self._save_index()
            
            logger.info(f"Committed version {version_id} to {model_name}/{branch}")
            return version
    
    def create_branch(self, model_name: str, branch_name: str, 
                     source_branch: str = "main") -> ModelBranch:
        """
        Create a new branch from an existing branch.
        
        Args:
            model_name: Name of the model repository
            branch_name: Name of the new branch
            source_branch: Source branch to create from
            
        Returns:
            Created ModelBranch
        """
        with self._lock:
            if model_name not in self.repositories:
                raise ValueError(f"Repository {model_name} does not exist")
            
            repo = self.repositories[model_name]
            
            if branch_name in repo.branches:
                raise ValueError(f"Branch {branch_name} already exists")
            
            if source_branch not in repo.branches:
                raise ValueError(f"Source branch {source_branch} does not exist")
            
            source_branch_obj = repo.branches[source_branch]
            
            # Create new branch
            new_branch = ModelBranch(
                name=branch_name,
                head_version_id=source_branch_obj.head_version_id,
                description=f"Branched from {source_branch}"
            )
            
            # Update repository
            repo.branches[branch_name] = new_branch
            repo.updated_at = datetime.now().isoformat()
            
            # Save branch reference
            branch_path = self._get_branch_path(model_name, branch_name)
            branch_path.parent.mkdir(parents=True, exist_ok=True)
            with open(branch_path, 'w') as f:
                f.write(source_branch_obj.head_version_id)
            
            # Update index
            self._save_index()
            
            logger.info(f"Created branch {branch_name} from {source_branch} in {model_name}")
            return new_branch
    
    def merge_branch(self, model_name: str, source_branch: str, target_branch: str,
                    strategy: MergeStrategy = MergeStrategy.THREE_WAY,
                    message: str = "") -> ModelVersion:
        """
        Merge one branch into another.
        
        Args:
            model_name: Name of the model repository
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            strategy: Merge strategy
            message: Merge commit message
            
        Returns:
            Merge commit version
        """
        with self._lock:
            if model_name not in self.repositories:
                raise ValueError(f"Repository {model_name} does not exist")
            
            repo = self.repositories[model_name]
            
            if source_branch not in repo.branches:
                raise ValueError(f"Source branch {source_branch} does not exist")
            
            if target_branch not in repo.branches:
                raise ValueError(f"Target branch {target_branch} does not exist")
            
            source_branch_obj = repo.branches[source_branch]
            target_branch_obj = repo.branches[target_branch]
            
            # Get versions
            source_version_id = source_branch_obj.head_version_id
            target_version_id = target_branch_obj.head_version_id
            
            if not source_version_id or not target_version_id:
                raise ValueError("Cannot merge empty branches")
            
            source_version = repo.versions[source_version_id]
            target_version = repo.versions[target_version_id]
            
            # Simple merge strategy: create a new version with combined metadata
            # In a real implementation, this would handle actual model merging
            
            # Create merged metadata
            merged_metadata = ModelMetadata(
                model_name=model_name,
                base_model=target_version.metadata.base_model,
                dataset_info={
                    **target_version.metadata.dataset_info,
                    **source_version.metadata.dataset_info
                },
                training_config=target_version.metadata.training_config,
                hyperparameters={
                    **target_version.metadata.hyperparameters,
                    **source_version.metadata.hyperparameters
                },
                performance_metrics={
                    **target_version.metadata.performance_metrics,
                    **source_version.metadata.performance_metrics
                },
                hardware_info=target_version.metadata.hardware_info,
                training_duration=max(
                    target_version.metadata.training_duration,
                    source_version.metadata.training_duration
                ),
                training_steps=max(
                    target_version.metadata.training_steps,
                    source_version.metadata.training_steps
                ),
                tags=list(set(target_version.metadata.tags + source_version.metadata.tags)),
                description=f"Merged from {source_branch} into {target_branch}",
                model_type=target_version.metadata.model_type,
                architectures=target_version.metadata.architectures
            )
            
            # Create merge commit
            merge_config = {
                'base_model': target_version.metadata.base_model,
                'merge_strategy': strategy.value,
                'source_branch': source_branch,
                'target_branch': target_branch,
                'source_version': source_version_id,
                'target_version': target_version_id
            }
            
            # Create a temporary directory for merged model
            with tempfile.TemporaryDirectory() as temp_dir:
                # In a real implementation, this would actually merge the models
                # For now, we'll just copy the target model
                if os.path.exists(target_version.model_path):
                    if os.path.isdir(target_version.model_path):
                        shutil.copytree(target_version.model_path, os.path.join(temp_dir, "model"))
                    else:
                        shutil.copy2(target_version.model_path, os.path.join(temp_dir, "model"))
                
                # Commit the merged version
                merged_version = self.commit(
                    model_name=model_name,
                    model_path=os.path.join(temp_dir, "model"),
                    training_config=merge_config,
                    branch=target_branch,
                    message=message or f"Merge {source_branch} into {target_branch}",
                    performance_metrics=merged_metadata.performance_metrics,
                    tags=merged_metadata.tags
                )
            
            logger.info(f"Merged {source_branch} into {target_branch} in {model_name}")
            return merged_version
    
    def get_version(self, model_name: str, version_id: str) -> Optional[ModelVersion]:
        """
        Get a specific version of a model.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID
            
        Returns:
            ModelVersion if found, None otherwise
        """
        if model_name not in self.repositories:
            return None
        
        repo = self.repositories[model_name]
        return repo.versions.get(version_id)
    
    def get_branch_head(self, model_name: str, branch: str) -> Optional[ModelVersion]:
        """
        Get the head version of a branch.
        
        Args:
            model_name: Name of the model repository
            branch: Branch name
            
        Returns:
            Head ModelVersion if found, None otherwise
        """
        if model_name not in self.repositories:
            return None
        
        repo = self.repositories[model_name]
        
        if branch not in repo.branches:
            return None
        
        branch_obj = repo.branches[branch]
        if not branch_obj.head_version_id:
            return None
        
        return repo.versions.get(branch_obj.head_version_id)
    
    def list_versions(self, model_name: str, branch: Optional[str] = None,
                     status: Optional[VersionStatus] = None) -> List[ModelVersion]:
        """
        List versions of a model, optionally filtered by branch and status.
        
        Args:
            model_name: Name of the model repository
            branch: Filter by branch
            status: Filter by status
            
        Returns:
            List of ModelVersions
        """
        if model_name not in self.repositories:
            return []
        
        repo = self.repositories[model_name]
        versions = list(repo.versions.values())
        
        if branch:
            versions = [v for v in versions if v.branch == branch]
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def list_branches(self, model_name: str) -> List[ModelBranch]:
        """
        List all branches of a model.
        
        Args:
            model_name: Name of the model repository
            
        Returns:
            List of ModelBranches
        """
        if model_name not in self.repositories:
            return []
        
        repo = self.repositories[model_name]
        return list(repo.branches.values())
    
    def list_models(self) -> List[str]:
        """
        List all models in the registry.
        
        Returns:
            List of model names
        """
        return list(self.repositories.keys())
    
    def search_models(self, query: str, tags: Optional[List[str]] = None,
                     base_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search models in the registry.
        
        Args:
            query: Search query
            tags: Filter by tags
            base_model: Filter by base model
            
        Returns:
            List of search results
        """
        results = []
        
        for model_name, repo in self.repositories.items():
            # Get latest version
            latest_version = None
            if repo.branches and repo.default_branch in repo.branches:
                default_branch = repo.branches[repo.default_branch]
                if default_branch.head_version_id:
                    latest_version = repo.versions.get(default_branch.head_version_id)
            
            if not latest_version:
                continue
            
            # Apply filters
            if base_model and latest_version.metadata.base_model != base_model:
                continue
            
            if tags and not all(tag in latest_version.metadata.tags for tag in tags):
                continue
            
            # Search in metadata
            search_text = f"{model_name} {repo.description} {latest_version.metadata.description}"
            if query.lower() in search_text.lower():
                results.append({
                    'model_name': model_name,
                    'description': repo.description,
                    'latest_version': latest_version.version_id,
                    'base_model': latest_version.metadata.base_model,
                    'tags': latest_version.metadata.tags,
                    'performance_metrics': latest_version.metadata.performance_metrics,
                    'created_at': latest_version.created_at
                })
        
        return results
    
    def tag_version(self, model_name: str, version_id: str, tag: str) -> None:
        """
        Tag a specific version.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID to tag
            tag: Tag name
        """
        with self._lock:
            if model_name not in self.repositories:
                raise ValueError(f"Repository {model_name} does not exist")
            
            repo = self.repositories[model_name]
            
            if version_id not in repo.versions:
                raise ValueError(f"Version {version_id} does not exist")
            
            repo.tags[tag] = version_id
            repo.updated_at = datetime.now().isoformat()
            
            # Update index
            self._save_index()
            
            logger.info(f"Tagged {version_id} as {tag} in {model_name}")
    
    def get_model_card(self, model_name: str, version_id: str) -> Optional[str]:
        """
        Get the model card for a specific version.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID
            
        Returns:
            Model card content if found, None otherwise
        """
        version_path = self._get_version_path(model_name, version_id)
        model_card_path = version_path / "MODEL_CARD.md"
        
        if model_card_path.exists():
            with open(model_card_path, 'r') as f:
                return f.read()
        
        return None
    
    def export_model(self, model_name: str, version_id: str, 
                    export_path: str, format: str = "huggingface") -> str:
        """
        Export a model version to a specific format.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID
            export_path: Path to export to
            format: Export format (huggingface, gguf, etc.)
            
        Returns:
            Path to exported model
        """
        version = self.get_version(model_name, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if os.path.exists(version.model_path):
            if os.path.isdir(version.model_path):
                shutil.copytree(version.model_path, export_path / "model", dirs_exist_ok=True)
            else:
                shutil.copy2(version.model_path, export_path / "model")
        
        # Copy model card
        model_card = self.get_model_card(model_name, version_id)
        if model_card:
            with open(export_path / "MODEL_CARD.md", 'w') as f:
                f.write(model_card)
        
        # Save version metadata
        with open(export_path / "version.json", 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        logger.info(f"Exported {model_name}/{version_id} to {export_path}")
        return str(export_path)
    
    def import_model(self, model_name: str, import_path: str, 
                    branch: str = "main", message: str = "") -> ModelVersion:
        """
        Import a model into the registry.
        
        Args:
            model_name: Name for the model repository
            import_path: Path to import from
            branch: Branch to import into
            message: Import message
            
        Returns:
            Imported ModelVersion
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise ValueError(f"Import path {import_path} does not exist")
        
        # Create repository if it doesn't exist
        if model_name not in self.repositories:
            self.create_repository(model_name, description=f"Imported from {import_path}")
        
        # Load metadata if available
        version_meta_path = import_path / "version.json"
        training_config = {}
        
        if version_meta_path.exists():
            with open(version_meta_path, 'r') as f:
                version_data = json.load(f)
                training_config = version_data.get('metadata', {}).get('training_config', {})
        
        # Import the model
        model_path = import_path / "model" if (import_path / "model").exists() else import_path
        
        return self.commit(
            model_name=model_name,
            model_path=str(model_path),
            training_config=training_config,
            branch=branch,
            message=message or f"Imported from {import_path}"
        )
    
    def delete_version(self, model_name: str, version_id: str, force: bool = False) -> bool:
        """
        Delete a specific version.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID to delete
            force: Force deletion even if it's the head of a branch
            
        Returns:
            True if deleted, False otherwise
        """
        with self._lock:
            if model_name not in self.repositories:
                return False
            
            repo = self.repositories[model_name]
            
            if version_id not in repo.versions:
                return False
            
            version = repo.versions[version_id]
            
            # Check if version is head of any branch
            for branch in repo.branches.values():
                if branch.head_version_id == version_id and not force:
                    logger.warning(f"Cannot delete {version_id}: it's the head of branch {branch.name}")
                    return False
            
            # Delete version directory
            version_path = self._get_version_path(model_name, version_id)
            if version_path.exists():
                shutil.rmtree(version_path)
            
            # Remove from repository
            del repo.versions[version_id]
            
            # Update any branches that point to this version
            for branch in repo.branches.values():
                if branch.head_version_id == version_id:
                    # Find parent version
                    if version.parent_hashes:
                        branch.head_version_id = version.parent_hashes[0]
                    else:
                        branch.head_version_id = ""
            
            # Remove from tags
            tags_to_remove = [tag for tag, vid in repo.tags.items() if vid == version_id]
            for tag in tags_to_remove:
                del repo.tags[tag]
            
            repo.updated_at = datetime.now().isoformat()
            
            # Update index
            self._save_index()
            
            logger.info(f"Deleted version {version_id} from {model_name}")
            return True
    
    def get_version_history(self, model_name: str, version_id: str, 
                           max_depth: int = 10) -> List[ModelVersion]:
        """
        Get the commit history for a version.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ModelVersions in history
        """
        if model_name not in self.repositories:
            return []
        
        repo = self.repositories[model_name]
        history = []
        current_id = version_id
        depth = 0
        
        while current_id and depth < max_depth:
            version = repo.versions.get(current_id)
            if not version:
                break
            
            history.append(version)
            
            # Move to parent
            if version.parent_hashes:
                current_id = version.parent_hashes[0]
            else:
                break
            
            depth += 1
        
        return history
    
    def validate_version(self, model_name: str, version_id: str) -> Dict[str, Any]:
        """
        Validate a model version.
        
        Args:
            model_name: Name of the model repository
            version_id: Version ID
            
        Returns:
            Validation results
        """
        version = self.get_version(model_name, version_id)
        if not version:
            return {'valid': False, 'error': 'Version not found'}
        
        validation_results = {
            'valid': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        # Check model files exist
        model_path = Path(version.model_path)
        if not model_path.exists():
            validation_results['valid'] = False
            validation_results['errors'].append(f"Model path does not exist: {model_path}")
        else:
            validation_results['checks'].append("Model files exist")
        
        # Check metadata completeness
        metadata = version.metadata
        if not metadata.base_model:
            validation_results['warnings'].append("Base model not specified")
        
        if not metadata.dataset_info:
            validation_results['warnings'].append("Dataset info not specified")
        
        if not metadata.performance_metrics:
            validation_results['warnings'].append("Performance metrics not specified")
        
        # Update version status if validation passes
        if validation_results['valid'] and not validation_results['errors']:
            version.status = VersionStatus.VALIDATED
            version.updated_at = datetime.now().isoformat()
            self._save_index()
        
        return validation_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Registry statistics
        """
        stats = {
            'total_models': len(self.repositories),
            'total_versions': sum(len(repo.versions) for repo in self.repositories.values()),
            'total_branches': sum(len(repo.branches) for repo in self.repositories.values()),
            'models_by_status': {},
            'models_by_base_model': {},
            'recent_activity': []
        }
        
        # Count by status
        for repo in self.repositories.values():
            for version in repo.versions.values():
                status = version.status.value
                stats['models_by_status'][status] = stats['models_by_status'].get(status, 0) + 1
                
                base_model = version.metadata.base_model
                stats['models_by_base_model'][base_model] = stats['models_by_base_model'].get(base_model, 0) + 1
        
        # Get recent activity (last 10 versions)
        all_versions = []
        for repo in self.repositories.values():
            all_versions.extend(repo.versions.values())
        
        all_versions.sort(key=lambda v: v.created_at, reverse=True)
        stats['recent_activity'] = [
            {
                'model_name': v.model_name,
                'version_id': v.version_id,
                'branch': v.branch,
                'created_at': v.created_at,
                'message': v.message
            }
            for v in all_versions[:10]
        ]
        
        return stats

# Integration with forge training pipeline
class TrainingMetadataExtractor:
    """
    Extracts metadata from forge training runs.
    Integrates with existing training scripts.
    """
    
    @staticmethod
    def extract_from_training_script(script_path: str) -> Dict[str, Any]:
        """
        Extract training configuration from a forge training script.
        
        Args:
            script_path: Path to training script
            
        Returns:
            Extracted configuration
        """
        # This would parse the training script and extract configuration
        # For now, return a placeholder
        return {
            'base_model': 'unknown',
            'dataset': 'unknown',
            'training_steps': 0,
            'training_duration': 0.0
        }
    
    @staticmethod
    def extract_from_log_file(log_path: str) -> Dict[str, Any]:
        """
        Extract training metrics from log files.
        
        Args:
            log_path: Path to log file
            
        Returns:
            Extracted metrics
        """
        # This would parse log files and extract metrics
        # For now, return a placeholder
        return {
            'loss': 0.0,
            'accuracy': 0.0,
            'perplexity': 0.0
        }

# CLI Integration
def create_cli_commands(version_manager: VersionManager) -> Dict[str, callable]:
    """
    Create CLI commands for the version manager.
    
    Args:
        version_manager: VersionManager instance
        
    Returns:
        Dictionary of command names to functions
    """
    commands = {}
    
    def init_repo(args):
        """Initialize a new repository"""
        version_manager.create_repository(
            model_name=args.model_name,
            description=args.description,
            owner=args.owner
        )
        print(f"Created repository {args.model_name}")
    
    def commit_model(args):
        """Commit a new version"""
        version = version_manager.commit(
            model_name=args.model_name,
            model_path=args.model_path,
            training_config=json.loads(args.config) if args.config else {},
            branch=args.branch,
            message=args.message,
            tags=args.tags.split(',') if args.tags else []
        )
        print(f"Committed version {version.version_id}")
    
    def list_versions(args):
        """List versions"""
        versions = version_manager.list_versions(
            model_name=args.model_name,
            branch=args.branch
        )
        for v in versions:
            print(f"{v.version_id} ({v.branch}) - {v.created_at} - {v.message}")
    
    def create_branch(args):
        """Create a new branch"""
        branch = version_manager.create_branch(
            model_name=args.model_name,
            branch_name=args.branch_name,
            source_branch=args.source
        )
        print(f"Created branch {branch.name}")
    
    def merge_branch(args):
        """Merge branches"""
        version = version_manager.merge_branch(
            model_name=args.model_name,
            source_branch=args.source,
            target_branch=args.target,
            message=args.message
        )
        print(f"Merged into {version.version_id}")
    
    def search_models(args):
        """Search models"""
        results = version_manager.search_models(
            query=args.query,
            tags=args.tags.split(',') if args.tags else None,
            base_model=args.base_model
        )
        for r in results:
            print(f"{r['model_name']}: {r['description']}")
    
    commands['init'] = init_repo
    commands['commit'] = commit_model
    commands['list'] = list_versions
    commands['branch'] = create_branch
    commands['merge'] = merge_branch
    commands['search'] = search_models
    
    return commands

# Example usage
if __name__ == "__main__":
    # Initialize version manager
    vm = VersionManager()
    
    # Create a repository
    repo = vm.create_repository(
        model_name="llama2-7b-finetuned",
        description="Fine-tuned Llama2 7B model",
        owner="forge-user"
    )
    
    # Commit a version
    version = vm.commit(
        model_name="llama2-7b-finetuned",
        model_path="./models/llama2-7b-finetuned",
        training_config={
            "base_model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 3,
            "training_steps": 1000,
            "training_duration": 2.5
        },
        branch="main",
        message="Initial fine-tuning on Alpaca dataset",
        performance_metrics={
            "accuracy": 0.85,
            "loss": 0.45,
            "perplexity": 12.3
        },
        tags=["llama2", "alpaca", "fine-tuned"]
    )
    
    # Create a branch
    vm.create_branch(
        model_name="llama2-7b-finetuned",
        branch_name="experimental",
        source_branch="main"
    )
    
    # List versions
    versions = vm.list_versions("llama2-7b-finetuned")
    for v in versions:
        print(f"{v.version_id}: {v.message}")
    
    # Search models
    results = vm.search_models("llama2")
    print(f"Found {len(results)} models")