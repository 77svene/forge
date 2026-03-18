"""
Model Registry & Versioning with Git-like capabilities, lineage tracking, and A/B testing.
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path
import tempfile
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus


class ModelVersionStatus(Enum):
    DRAFT = "draft"
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ExperimentStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Represents a specific version of a model with full lineage."""
    version_id: str
    model_name: str
    version_number: str
    commit_hash: str
    parent_version_id: Optional[str] = None
    dataset_version_id: Optional[str] = None
    training_job_id: Optional[str] = None
    status: ModelVersionStatus = ModelVersionStatus.DRAFT
    metadata: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    artifacts: Dict[str, str] = None  # artifact_name -> content_hash
    created_at: str = None
    updated_at: str = None
    created_by: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class DatasetVersion:
    """Represents a specific version of a dataset."""
    version_id: str
    dataset_name: str
    version_number: str
    commit_hash: str
    parent_version_id: Optional[str] = None
    data_recipe_id: Optional[str] = None
    preprocessing_job_id: Optional[str] = None
    status: ModelVersionStatus = ModelVersionStatus.DRAFT
    metadata: Dict[str, Any] = None
    statistics: Dict[str, Any] = None
    artifacts: Dict[str, str] = None
    created_at: str = None
    updated_at: str = None
    created_by: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.statistics is None:
            self.statistics = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class Experiment:
    """A/B testing experiment comparing multiple model versions."""
    experiment_id: str
    name: str
    description: str
    model_versions: List[str]  # List of version_ids
    traffic_allocation: Dict[str, float]  # version_id -> traffic percentage
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    metrics: Dict[str, Dict[str, float]] = None  # version_id -> metric_name -> value
    start_time: str = None
    end_time: Optional[str] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}
        if self.start_time is None:
            self.start_time = datetime.utcnow().isoformat()


@dataclass
class MarketplaceListing:
    """Model listing in the marketplace."""
    listing_id: str
    model_name: str
    version_id: str
    description: str
    author: str
    license: str
    tags: List[str]
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    created_at: str = None
    updated_at: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class ContentAddressableStorage:
    """Git-like content-addressable storage for model artifacts."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.objects_path = self.storage_path / "objects"
        self.objects_path.mkdir(exist_ok=True)
        self.refs_path = self.storage_path / "refs"
        self.refs_path.mkdir(exist_ok=True)
        self HEAD_path = self.storage_path / "HEAD"
        
        # Initialize HEAD if it doesn't exist
        if not self HEAD_path.exists():
            self HEAD_path.write_text("ref: refs/heads/main\n")
    
    def _hash_content(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def store_object(self, content: bytes) -> str:
        """Store content and return its hash."""
        content_hash = self._hash_content(content)
        object_dir = self.objects_path / content_hash[:2]
        object_dir.mkdir(exist_ok=True)
        object_path = object_dir / content_hash[2:]
        
        if not object_path.exists():
            object_path.write_bytes(content)
        
        return content_hash
    
    def retrieve_object(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content by its hash."""
        object_dir = self.objects_path / content_hash[:2]
        object_path = object_dir / content_hash[2:]
        
        if object_path.exists():
            return object_path.read_bytes()
        return None
    
    def store_json(self, data: Dict[str, Any]) -> str:
        """Store JSON data and return its hash."""
        content = json.dumps(data, sort_keys=True).encode('utf-8')
        return self.store_object(content)
    
    def retrieve_json(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve JSON data by its hash."""
        content = self.retrieve_object(content_hash)
        if content:
            return json.loads(content.decode('utf-8'))
        return None
    
    def create_ref(self, ref_name: str, commit_hash: str):
        """Create or update a reference."""
        ref_path = self.refs_path / ref_name
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(commit_hash)
    
    def get_ref(self, ref_name: str) -> Optional[str]:
        """Get the commit hash a reference points to."""
        ref_path = self.refs_path / ref_name
        if ref_path.exists():
            return ref_path.read_text().strip()
        return None
    
    def create_branch(self, branch_name: str, commit_hash: str):
        """Create a new branch."""
        self.create_ref(f"heads/{branch_name}", commit_hash)
    
    def get_branch(self, branch_name: str) -> Optional[str]:
        """Get the commit hash a branch points to."""
        return self.get_ref(f"heads/{branch_name}")
    
    def create_tag(self, tag_name: str, commit_hash: str):
        """Create a new tag."""
        self.create_ref(f"tags/{tag_name}", commit_hash)
    
    def get_tag(self, tag_name: str) -> Optional[str]:
        """Get the commit hash a tag points to."""
        return self.get_ref(f"tags/{tag_name}")


class ModelRegistry:
    """
    Git-like model registry with versioning, lineage tracking, and A/B testing.
    
    Features:
    - Content-addressable storage for models and datasets
    - Automatic lineage tracking from data to deployment
    - Branching and tagging like Git
    - A/B testing framework
    - Model marketplace
    """
    
    def __init__(self, registry_path: str = "~/.forge/registry"):
        self.registry_path = Path(os.path.expanduser(registry_path))
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.storage = ContentAddressableStorage(str(self.registry_path / "objects"))
        
        # Initialize database
        self.db_path = self.registry_path / "registry.db"
        self._init_database()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Job manager for training jobs
        self.job_manager = JobManager()
        
        # Cache for frequently accessed items
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version_number TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                parent_version_id TEXT,
                dataset_version_id TEXT,
                training_job_id TEXT,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL,
                metrics TEXT NOT NULL,
                artifacts TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT,
                tags TEXT NOT NULL,
                UNIQUE(model_name, version_number)
            )
        ''')
        
        # Dataset versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                version_number TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                parent_version_id TEXT,
                data_recipe_id TEXT,
                preprocessing_job_id TEXT,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL,
                statistics TEXT NOT NULL,
                artifacts TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT,
                tags TEXT NOT NULL,
                UNIQUE(dataset_name, version_number)
            )
        ''')
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                model_versions TEXT NOT NULL,
                traffic_allocation TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                created_by TEXT,
                metadata TEXT NOT NULL
            )
        ''')
        
        # Marketplace listings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marketplace_listings (
                listing_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version_id TEXT NOT NULL,
                description TEXT NOT NULL,
                author TEXT NOT NULL,
                license TEXT NOT NULL,
                tags TEXT NOT NULL,
                download_count INTEGER NOT NULL,
                rating REAL NOT NULL,
                rating_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
            )
        ''')
        
        # Lineage graph table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lineage_graph (
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_type, source_id, target_type, target_id, relationship)
            )
        ''')
        
        # Create indices for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON model_versions(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_name ON dataset_versions(dataset_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_status ON experiments(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_marketplace_model ON marketplace_listings(model_name)')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def _generate_version_number(self, name: str, is_model: bool = True) -> str:
        """Generate a semantic version number for a model or dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        table = "model_versions" if is_model else "dataset_versions"
        name_column = "model_name" if is_model else "dataset_name"
        
        cursor.execute(f'''
            SELECT version_number FROM {table}
            WHERE {name_column} = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return "1.0.0"
        
        last_version = result[0]
        try:
            major, minor, patch = map(int, last_version.split('.'))
            # Auto-increment patch version for new versions
            return f"{major}.{minor}.{patch + 1}"
        except:
            # If version format is unexpected, start fresh
            return "1.0.0"
    
    def _create_commit(self, data: Dict[str, Any], parent_hash: Optional[str] = None) -> str:
        """Create a commit object and return its hash."""
        commit = {
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "parent": parent_hash
        }
        return self.storage.store_json(commit)
    
    def _update_lineage(self, source_type: str, source_id: str, 
                       target_type: str, target_id: str, 
                       relationship: str, metadata: Dict[str, Any] = None):
        """Update lineage graph."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO lineage_graph 
            (source_type, source_id, target_type, target_id, relationship, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_type, source_id, target_type, target_id, relationship,
            json.dumps(metadata or {}),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def register_model(self, model_name: str, model_data: Dict[str, Any], 
                      dataset_version_id: Optional[str] = None,
                      parent_version_id: Optional[str] = None,
                      training_job_id: Optional[str] = None,
                      metadata: Dict[str, Any] = None,
                      artifacts: Dict[str, bytes] = None,
                      created_by: Optional[str] = None,
                      tags: List[str] = None) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_data: Model configuration and weights
            dataset_version_id: ID of dataset version used for training
            parent_version_id: ID of parent model version (for fine-tuning)
            training_job_id: ID of training job
            metadata: Additional metadata
            artifacts: Dictionary of artifact_name -> artifact_content
            created_by: User who created this version
            tags: List of tags
            
        Returns:
            ModelVersion object
        """
        # Store artifacts in content-addressable storage
        artifact_hashes = {}
        if artifacts:
            for artifact_name, content in artifacts.items():
                artifact_hashes[artifact_name] = self.storage.store_object(content)
        
        # Create model data with artifacts
        model_commit_data = {
            "model_data": model_data,
            "artifacts": artifact_hashes,
            "metadata": metadata or {}
        }
        
        # Get parent commit hash if exists
        parent_hash = None
        if parent_version_id:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT commit_hash FROM model_versions WHERE version_id = ?', 
                         (parent_version_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                parent_hash = result[0]
        
        # Create commit
        commit_hash = self._create_commit(model_commit_data, parent_hash)
        
        # Generate version
        version_number = self._generate_version_number(model_name, is_model=True)
        version_id = str(uuid.uuid4())
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version_number=version_number,
            commit_hash=commit_hash,
            parent_version_id=parent_version_id,
            dataset_version_id=dataset_version_id,
            training_job_id=training_job_id,
            status=ModelVersionStatus.DRAFT,
            metadata=metadata or {},
            artifacts=artifact_hashes,
            created_by=created_by,
            tags=tags or []
        )
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_versions 
            (version_id, model_name, version_number, commit_hash, parent_version_id,
             dataset_version_id, training_job_id, status, metadata, metrics, artifacts,
             created_at, updated_at, created_by, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            version_id, model_name, version_number, commit_hash, parent_version_id,
            dataset_version_id, training_job_id, model_version.status.value,
            json.dumps(model_version.metadata), json.dumps(model_version.metrics),
            json.dumps(model_version.artifacts), model_version.created_at,
            model_version.updated_at, created_by, json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
        
        # Update lineage
        if parent_version_id:
            self._update_lineage(
                "model_version", version_id,
                "model_version", parent_version_id,
                "derived_from"
            )
        
        if dataset_version_id:
            self._update_lineage(
                "model_version", version_id,
                "dataset_version", dataset_version_id,
                "trained_on"
            )
        
        # Create branch for this model if it doesn't exist
        branch_name = f"{model_name}/main"
        if not self.storage.get_branch(branch_name):
            self.storage.create_branch(branch_name, commit_hash)
        
        return model_version
    
    def register_dataset(self, dataset_name: str, dataset_data: Dict[str, Any],
                        data_recipe_id: Optional[str] = None,
                        preprocessing_job_id: Optional[str] = None,
                        parent_version_id: Optional[str] = None,
                        metadata: Dict[str, Any] = None,
                        statistics: Dict[str, Any] = None,
                        artifacts: Dict[str, bytes] = None,
                        created_by: Optional[str] = None,
                        tags: List[str] = None) -> DatasetVersion:
        """
        Register a new dataset version.
        
        Args:
            dataset_name: Name of the dataset
            dataset_data: Dataset configuration and metadata
            data_recipe_id: ID of data recipe used to create dataset
            preprocessing_job_id: ID of preprocessing job
            parent_version_id: ID of parent dataset version
            metadata: Additional metadata
            statistics: Dataset statistics
            artifacts: Dictionary of artifact_name -> artifact_content
            created_by: User who created this version
            tags: List of tags
            
        Returns:
            DatasetVersion object
        """
        # Store artifacts
        artifact_hashes = {}
        if artifacts:
            for artifact_name, content in artifacts.items():
                artifact_hashes[artifact_name] = self.storage.store_object(content)
        
        # Create dataset commit data
        dataset_commit_data = {
            "dataset_data": dataset_data,
            "artifacts": artifact_hashes,
            "statistics": statistics or {},
            "metadata": metadata or {}
        }
        
        # Get parent commit hash
        parent_hash = None
        if parent_version_id:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT commit_hash FROM dataset_versions WHERE version_id = ?', 
                         (parent_version_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                parent_hash = result[0]
        
        # Create commit
        commit_hash = self._create_commit(dataset_commit_data, parent_hash)
        
        # Generate version
        version_number = self._generate_version_number(dataset_name, is_model=False)
        version_id = str(uuid.uuid4())
        
        # Create dataset version
        dataset_version = DatasetVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            version_number=version_number,
            commit_hash=commit_hash,
            parent_version_id=parent_version_id,
            data_recipe_id=data_recipe_id,
            preprocessing_job_id=preprocessing_job_id,
            status=ModelVersionStatus.DRAFT,
            metadata=metadata or {},
            statistics=statistics or {},
            artifacts=artifact_hashes,
            created_by=created_by,
            tags=tags or []
        )
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO dataset_versions 
            (version_id, dataset_name, version_number, commit_hash, parent_version_id,
             data_recipe_id, preprocessing_job_id, status, metadata, statistics, artifacts,
             created_at, updated_at, created_by, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            version_id, dataset_name, version_number, commit_hash, parent_version_id,
            data_recipe_id, preprocessing_job_id, dataset_version.status.value,
            json.dumps(dataset_version.metadata), json.dumps(dataset_version.statistics),
            json.dumps(dataset_version.artifacts), dataset_version.created_at,
            dataset_version.updated_at, created_by, json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
        
        # Update lineage
        if parent_version_id:
            self._update_lineage(
                "dataset_version", version_id,
                "dataset_version", parent_version_id,
                "derived_from"
            )
        
        if data_recipe_id:
            self._update_lineage(
                "dataset_version", version_id,
                "data_recipe", data_recipe_id,
                "created_from"
            )
        
        # Create branch
        branch_name = f"{dataset_name}/main"
        if not self.storage.get_branch(branch_name):
            self.storage.create_branch(branch_name, commit_hash)
        
        return dataset_version
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a model version by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_id, model_name, version_number, commit_hash, parent_version_id,
                   dataset_version_id, training_job_id, status, metadata, metrics, artifacts,
                   created_at, updated_at, created_by, tags
            FROM model_versions WHERE version_id = ?
        ''', (version_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return ModelVersion(
            version_id=result[0],
            model_name=result[1],
            version_number=result[2],
            commit_hash=result[3],
            parent_version_id=result[4],
            dataset_version_id=result[5],
            training_job_id=result[6],
            status=ModelVersionStatus(result[7]),
            metadata=json.loads(result[8]),
            metrics=json.loads(result[9]),
            artifacts=json.loads(result[10]),
            created_at=result[11],
            updated_at=result[12],
            created_by=result[13],
            tags=json.loads(result[14])
        )
    
    def get_dataset_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a dataset version by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_id, dataset_name, version_number, commit_hash, parent_version_id,
                   data_recipe_id, preprocessing_job_id, status, metadata, statistics, artifacts,
                   created_at, updated_at, created_by, tags
            FROM dataset_versions WHERE version_id = ?
        ''', (version_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return DatasetVersion(
            version_id=result[0],
            dataset_name=result[1],
            version_number=result[2],
            commit_hash=result[3],
            parent_version_id=result[4],
            data_recipe_id=result[5],
            preprocessing_job_id=result[6],
            status=ModelVersionStatus(result[7]),
            metadata=json.loads(result[8]),
            statistics=json.loads(result[9]),
            artifacts=json.loads(result[10]),
            created_at=result[11],
            updated_at=result[12],
            created_by=result[13],
            tags=json.loads(result[14])
        )
    
    def list_model_versions(self, model_name: Optional[str] = None, 
                          status: Optional[ModelVersionStatus] = None,
                          limit: int = 100, offset: int = 0) -> List[ModelVersion]:
        """List model versions with optional filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT version_id, model_name, version_number, commit_hash, parent_version_id,
                   dataset_version_id, training_job_id, status, metadata, metrics, artifacts,
                   created_at, updated_at, created_by, tags
            FROM model_versions
            WHERE 1=1
        '''
        params = []
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        if status:
            query += ' AND status = ?'
            params.append(status.value)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            ModelVersion(
                version_id=row[0],
                model_name=row[1],
                version_number=row[2],
                commit_hash=row[3],
                parent_version_id=row[4],
                dataset_version_id=row[5],
                training_job_id=row[6],
                status=ModelVersionStatus(row[7]),
                metadata=json.loads(row[8]),
                metrics=json.loads(row[9]),
                artifacts=json.loads(row[10]),
                created_at=row[11],
                updated_at=row[12],
                created_by=row[13],
                tags=json.loads(row[14])
            )
            for row in results
        ]
    
    def create_branch(self, model_name: str, branch_name: str, source_version_id: str):
        """Create a new branch for a model."""
        version = self.get_model_version(source_version_id)
        if not version:
            raise ValueError(f"Model version {source_version_id} not found")
        
        if version.model_name != model_name:
            raise ValueError(f"Model version belongs to {version.model_name}, not {model_name}")
        
        full_branch_name = f"{model_name}/{branch_name}"
        self.storage.create_branch(full_branch_name, version.commit_hash)
    
    def checkout_branch(self, model_name: str, branch_name: str) -> Optional[ModelVersion]:
        """Get the model version a branch points to."""
        full_branch_name = f"{model_name}/{branch_name}"
        commit_hash = self.storage.get_branch(full_branch_name)
        
        if not commit_hash:
            return None
        
        # Find model version by commit hash
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_id FROM model_versions 
            WHERE commit_hash = ? AND model_name = ?
            LIMIT 1
        ''', (commit_hash, model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return self.get_model_version(result[0])
        return None
    
    def create_tag(self, model_name: str, tag_name: str, version_id: str):
        """Create a tag for a specific model version."""
        version = self.get_model_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        if version.model_name != model_name:
            raise ValueError(f"Model version belongs to {version.model_name}, not {model_name}")
        
        full_tag_name = f"{model_name}/{tag_name}"
        self.storage.create_tag(full_tag_name, version.commit_hash)
    
    def get_tagged_version(self, model_name: str, tag_name: str) -> Optional[ModelVersion]:
        """Get the model version a tag points to."""
        full_tag_name = f"{model_name}/{tag_name}"
        commit_hash = self.storage.get_tag(full_tag_name)
        
        if not commit_hash:
            return None
        
        # Find model version by commit hash
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_id FROM model_versions 
            WHERE commit_hash = ? AND model_name = ?
            LIMIT 1
        ''', (commit_hash, model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return self.get_model_version(result[0])
        return None
    
    def update_model_status(self, version_id: str, status: ModelVersionStatus, 
                           metrics: Optional[Dict[str, float]] = None):
        """Update model version status and optionally metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        update_fields = ['status = ?', 'updated_at = ?']
        params = [status.value, datetime.utcnow().isoformat()]
        
        if metrics is not None:
            update_fields.append('metrics = ?')
            params.append(json.dumps(metrics))
        
        params.append(version_id)
        
        cursor.execute(f'''
            UPDATE model_versions 
            SET {', '.join(update_fields)}
            WHERE version_id = ?
        ''', params)
        
        conn.commit()
        conn.close()
    
    def get_lineage(self, version_id: str, direction: str = "both") -> Dict[str, List[Dict[str, Any]]]:
        """
        Get lineage for a model or dataset version.
        
        Args:
            version_id: ID of the version
            direction: "ancestors", "descendants", or "both"
            
        Returns:
            Dictionary with ancestors and descendants
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        result = {"ancestors": [], "descendants": []}
        
        if direction in ["ancestors", "both"]:
            # Get ancestors (what this version was derived from)
            cursor.execute('''
                SELECT target_type, target_id, relationship, metadata
                FROM lineage_graph
                WHERE source_id = ? AND source_type IN ('model_version', 'dataset_version')
                ORDER BY created_at DESC
            ''', (version_id,))
            
            for row in cursor.fetchall():
                result["ancestors"].append({
                    "type": row[0],
                    "id": row[1],
                    "relationship": row[2],
                    "metadata": json.loads(row[3])
                })
        
        if direction in ["descendants", "both"]:
            # Get descendants (what was derived from this version)
            cursor.execute('''
                SELECT source_type, source_id, relationship, metadata
                FROM lineage_graph
                WHERE target_id = ? AND target_type IN ('model_version', 'dataset_version')
                ORDER BY created_at DESC
            ''', (version_id,))
            
            for row in cursor.fetchall():
                result["descendants"].append({
                    "type": row[0],
                    "id": row[1],
                    "relationship": row[2],
                    "metadata": json.loads(row[3])
                })
        
        conn.close()
        return result
    
    def create_experiment(self, name: str, description: str, 
                         model_version_ids: List[str],
                         traffic_allocation: Optional[Dict[str, float]] = None,
                         created_by: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Experiment:
        """
        Create an A/B testing experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            model_version_ids: List of model version IDs to compare
            traffic_allocation: Traffic percentage for each version (must sum to 100)
            created_by: User who created the experiment
            metadata: Additional metadata
            
        Returns:
            Experiment object
        """
        # Validate model versions exist
        for version_id in model_version_ids:
            version = self.get_model_version(version_id)
            if not version:
                raise ValueError(f"Model version {version_id} not found")
        
        # Default traffic allocation (equal distribution)
        if traffic_allocation is None:
            equal_share = 100.0 / len(model_version_ids)
            traffic_allocation = {vid: equal_share for vid in model_version_ids}
        
        # Validate traffic allocation
        if abs(sum(traffic_allocation.values()) - 100.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 100%")
        
        if set(traffic_allocation.keys()) != set(model_version_ids):
            raise ValueError("Traffic allocation keys must match model version IDs")
        
        experiment_id = str(uuid.uuid4())
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            model_versions=model_version_ids,
            traffic_allocation=traffic_allocation,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments 
            (experiment_id, name, description, model_versions, traffic_allocation,
             status, metrics, start_time, end_time, created_by, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, name, description,
            json.dumps(model_version_ids), json.dumps(traffic_allocation),
            experiment.status.value, json.dumps(experiment.metrics),
            experiment.start_time, experiment.end_time,
            created_by, json.dumps(experiment.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        return experiment
    
    def update_experiment_metrics(self, experiment_id: str, 
                                 version_id: str, 
                                 metrics: Dict[str, float]):
        """Update metrics for a specific model version in an experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get current experiment
        cursor.execute('SELECT metrics FROM experiments WHERE experiment_id = ?', 
                      (experiment_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Experiment {experiment_id} not found")
        
        current_metrics = json.loads(result[0])
        current_metrics[version_id] = metrics
        
        cursor.execute('''
            UPDATE experiments 
            SET metrics = ?, updated_at = ?
            WHERE experiment_id = ?
        ''', (json.dumps(current_metrics), datetime.utcnow().isoformat(), experiment_id))
        
        conn.commit()
        conn.close()
    
    def complete_experiment(self, experiment_id: str, winning_version_id: Optional[str] = None):
        """Mark an experiment as completed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        update_fields = ['status = ?', 'end_time = ?']
        params = [ExperimentStatus.COMPLETED.value, datetime.utcnow().isoformat()]
        
        if winning_version_id:
            # Update metadata with winner
            cursor.execute('SELECT metadata FROM experiments WHERE experiment_id = ?', 
                          (experiment_id,))
            result = cursor.fetchone()
            if result:
                metadata = json.loads(result[0])
                metadata['winning_version_id'] = winning_version_id
                update_fields.append('metadata = ?')
                params.append(json.dumps(metadata))
        
        params.append(experiment_id)
        
        cursor.execute(f'''
            UPDATE experiments 
            SET {', '.join(update_fields)}
            WHERE experiment_id = ?
        ''', params)
        
        conn.commit()
        conn.close()
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None,
                        limit: int = 100, offset: int = 0) -> List[Experiment]:
        """List experiments with optional filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT experiment_id, name, description, model_versions, traffic_allocation,
                   status, metrics, start_time, end_time, created_by, metadata
            FROM experiments
            WHERE 1=1
        '''
        params = []
        
        if status:
            query += ' AND status = ?'
            params.append(status.value)
        
        query += ' ORDER BY start_time DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            Experiment(
                experiment_id=row[0],
                name=row[1],
                description=row[2],
                model_versions=json.loads(row[3]),
                traffic_allocation=json.loads(row[4]),
                status=ExperimentStatus(row[5]),
                metrics=json.loads(row[6]),
                start_time=row[7],
                end_time=row[8],
                created_by=row[9],
                metadata=json.loads(row[10])
            )
            for row in results
        ]
    
    def publish_to_marketplace(self, model_name: str, version_id: str,
                              description: str, author: str, license: str,
                              tags: List[str] = None,
                              metadata: Dict[str, Any] = None) -> MarketplaceListing:
        """
        Publish a model version to the marketplace.
        
        Args:
            model_name: Name of the model
            version_id: ID of the model version to publish
            description: Model description
            author: Model author
            license: License type
            tags: List of tags
            metadata: Additional metadata
            
        Returns:
            MarketplaceListing object
        """
        version = self.get_model_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        if version.model_name != model_name:
            raise ValueError(f"Model version belongs to {version.model_name}, not {model_name}")
        
        if version.status != ModelVersionStatus.READY:
            raise ValueError("Only models with READY status can be published")
        
        listing_id = str(uuid.uuid4())
        listing = MarketplaceListing(
            listing_id=listing_id,
            model_name=model_name,
            version_id=version_id,
            description=description,
            author=author,
            license=license,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO marketplace_listings 
            (listing_id, model_name, version_id, description, author, license,
             tags, download_count, rating, rating_count, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            listing_id, model_name, version_id, description, author, license,
            json.dumps(tags or []), 0, 0.0, 0,
            listing.created_at, listing.updated_at, json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        return listing
    
    def download_from_marketplace(self, listing_id: str) -> Tuple[ModelVersion, Dict[str, bytes]]:
        """
        Download a model from the marketplace.
        
        Args:
            listing_id: ID of the marketplace listing
            
        Returns:
            Tuple of (ModelVersion, artifacts_dict)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get listing
        cursor.execute('''
            SELECT version_id, download_count 
            FROM marketplace_listings 
            WHERE listing_id = ?
        ''', (listing_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            raise ValueError(f"Listing {listing_id} not found")
        
        version_id, download_count = result
        
        # Update download count
        cursor.execute('''
            UPDATE marketplace_listings 
            SET download_count = download_count + 1, updated_at = ?
            WHERE listing_id = ?
        ''', (datetime.utcnow().isoformat(), listing_id))
        
        conn.commit()
        conn.close()
        
        # Get model version
        version = self.get_model_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        # Download artifacts
        artifacts = {}
        for artifact_name, artifact_hash in version.artifacts.items():
            content = self.storage.retrieve_object(artifact_hash)
            if content:
                artifacts[artifact_name] = content
        
        return version, artifacts
    
    def rate_model(self, listing_id: str, rating: float):
        """Rate a model in the marketplace."""
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get current rating
        cursor.execute('''
            SELECT rating, rating_count 
            FROM marketplace_listings 
            WHERE listing_id = ?
        ''', (listing_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            raise ValueError(f"Listing {listing_id} not found")
        
        current_rating, rating_count = result
        
        # Calculate new rating
        new_rating_count = rating_count + 1
        new_rating = ((current_rating * rating_count) + rating) / new_rating_count
        
        cursor.execute('''
            UPDATE marketplace_listings 
            SET rating = ?, rating_count = ?, updated_at = ?
            WHERE listing_id = ?
        ''', (new_rating, new_rating_count, datetime.utcnow().isoformat(), listing_id))
        
        conn.commit()
        conn.close()
    
    def search_marketplace(self, query: str = None, tags: List[str] = None,
                          min_rating: float = None, limit: int = 100, 
                          offset: int = 0) -> List[MarketplaceListing]:
        """Search marketplace listings."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sql_query = '''
            SELECT listing_id, model_name, version_id, description, author, license,
                   tags, download_count, rating, rating_count, created_at, updated_at, metadata
            FROM marketplace_listings
            WHERE 1=1
        '''
        params = []
        
        if query:
            sql_query += ' AND (model_name LIKE ? OR description LIKE ?)'
            params.extend([f'%{query}%', f'%{query}%'])
        
        if tags:
            # SQLite doesn't have great JSON support, so we'll filter in Python
            # For production, consider using a proper JSON database or full-text search
            pass
        
        if min_rating is not None:
            sql_query += ' AND rating >= ?'
            params.append(min_rating)
        
        sql_query += ' ORDER BY download_count DESC, rating DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        conn.close()
        
        listings = []
        for row in results:
            listing_tags = json.loads(row[6])
            
            # Filter by tags if specified
            if tags and not set(tags).issubset(set(listing_tags)):
                continue
            
            listings.append(MarketplaceListing(
                listing_id=row[0],
                model_name=row[1],
                version_id=row[2],
                description=row[3],
                author=row[4],
                license=row[5],
                tags=listing_tags,
                download_count=row[7],
                rating=row[8],
                rating_count=row[9],
                created_at=row[10],
                updated_at=row[11],
                metadata=json.loads(row[12])
            ))
        
        return listings
    
    def export_model(self, version_id: str, export_path: str):
        """Export a model version to a directory."""
        version = self.get_model_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metadata
        metadata = {
            "model_name": version.model_name,
            "version": version.version_number,
            "version_id": version.version_id,
            "commit_hash": version.commit_hash,
            "parent_version_id": version.parent_version_id,
            "dataset_version_id": version.dataset_version_id,
            "status": version.status.value,
            "metadata": version.metadata,
            "metrics": version.metrics,
            "created_at": version.created_at,
            "created_by": version.created_by,
            "tags": version.tags
        }
        
        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Export artifacts
        artifacts_dir = export_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        for artifact_name, artifact_hash in version.artifacts.items():
            content = self.storage.retrieve_object(artifact_hash)
            if content:
                artifact_path = artifacts_dir / artifact_name
                artifact_path.write_bytes(content)
    
    def import_model(self, import_path: str, model_name: Optional[str] = None,
                    created_by: Optional[str] = None) -> ModelVersion:
        """Import a model from an exported directory."""
        import_dir = Path(import_path)
        
        # Read metadata
        metadata_path = import_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError("No metadata.json found in import directory")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Use provided model name or original
        if model_name is None:
            model_name = metadata["model_name"]
        
        # Read artifacts
        artifacts = {}
        artifacts_dir = import_dir / "artifacts"
        if artifacts_dir.exists():
            for artifact_path in artifacts_dir.iterdir():
                if artifact_path.is_file():
                    artifacts[artifact_path.name] = artifact_path.read_bytes()
        
        # Register as new version
        return self.register_model(
            model_name=model_name,
            model_data={"imported": True, "original_metadata": metadata},
            metadata=metadata.get("metadata", {}),
            artifacts=artifacts,
            created_by=created_by,
            tags=metadata.get("tags", []) + ["imported"]
        )
    
    def start_training_job(self, model_name: str, dataset_version_id: str,
                          training_config: Dict[str, Any],
                          parent_version_id: Optional[str] = None,
                          created_by: Optional[str] = None) -> str:
        """
        Start a training job and track it in the registry.
        
        Args:
            model_name: Name of the model to train
            dataset_version_id: ID of dataset version to use
            training_config: Training configuration
            parent_version_id: Optional parent model version for fine-tuning
            created_by: User starting the job
            
        Returns:
            Job ID
        """
        # Create initial model version in DRAFT status
        model_version = self.register_model(
            model_name=model_name,
            model_data={"training_config": training_config},
            dataset_version_id=dataset_version_id,
            parent_version_id=parent_version_id,
            metadata={"training_config": training_config},
            created_by=created_by,
            tags=["training"]
        )
        
        # Update status to TRAINING
        self.update_model_status(model_version.version_id, ModelVersionStatus.TRAINING)
        
        # Start actual training job using job manager
        job_id = self.job_manager.submit_job(
            job_type="model_training",
            config={
                "model_version_id": model_version.version_id,
                "model_name": model_name,
                "dataset_version_id": dataset_version_id,
                "training_config": training_config,
                "parent_version_id": parent_version_id
            },
            created_by=created_by
        )
        
        # Update model version with job ID
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model_versions 
            SET training_job_id = ?, updated_at = ?
            WHERE version_id = ?
        ''', (job_id, datetime.utcnow().isoformat(), model_version.version_id))
        conn.commit()
        conn.close()
        
        return job_id
    
    def check_training_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of a training job."""
        job_status = self.job_manager.get_job_status(job_id)
        
        # Find associated model version
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT version_id, status FROM model_versions 
            WHERE training_job_id = ?
        ''', (job_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {"job_status": job_status, "model_version": None}
        
        version_id, model_status = result
        
        # Update model status based on job status
        if job_status["status"] == JobStatus.COMPLETED.value:
            self.update_model_status(version_id, ModelVersionStatus.READY)
        elif job_status["status"] == JobStatus.FAILED.value:
            self.update_model_status(version_id, ModelVersionStatus.FAILED)
        
        return {
            "job_status": job_status,
            "model_version": {
                "version_id": version_id,
                "status": model_status
            }
        }
    
    def get_model_artifact(self, version_id: str, artifact_name: str) -> Optional[bytes]:
        """Get a specific artifact for a model version."""
        version = self.get_model_version(version_id)
        if not version:
            return None
        
        artifact_hash = version.artifacts.get(artifact_name)
        if not artifact_hash:
            return None
        
        return self.storage.retrieve_object(artifact_hash)
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        version1 = self.get_model_version(version_id1)
        version2 = self.get_model_version(version_id2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "version1": {
                "version_id": version1.version_id,
                "model_name": version1.model_name,
                "version": version1.version_number,
                "status": version1.status.value,
                "metrics": version1.metrics,
                "created_at": version1.created_at
            },
            "version2": {
                "version_id": version2.version_id,
                "model_name": version2.model_name,
                "version": version2.version_number,
                "status": version2.status.value,
                "metrics": version2.metrics,
                "created_at": version2.created_at
            },
            "metrics_comparison": {},
            "lineage": {
                "common_ancestor": None,
                "relationship": None
            }
        }
        
        # Compare metrics
        all_metrics = set(version1.metrics.keys()) | set(version2.metrics.keys())
        for metric in all_metrics:
            val1 = version1.metrics.get(metric)
            val2 = version2.metrics.get(metric)
            
            if val1 is not None and val2 is not None:
                comparison["metrics_comparison"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else None
                }
        
        # Check lineage relationship
        lineage1 = self.get_lineage(version_id1, "ancestors")
        lineage2 = self.get_lineage(version_id2, "ancestors")
        
        ancestors1 = {a["id"] for a in lineage1["ancestors"]}
        ancestors2 = {a["id"] for a in lineage2["ancestors"]}
        
        common_ancestors = ancestors1.intersection(ancestors2)
        if common_ancestors:
            comparison["lineage"]["common_ancestor"] = list(common_ancestors)[0]
            comparison["lineage"]["relationship"] = "siblings"
        elif version_id1 in ancestors2:
            comparison["lineage"]["relationship"] = "version1 is ancestor of version2"
        elif version_id2 in ancestors1:
            comparison["lineage"]["relationship"] = "version2 is ancestor of version1"
        
        return comparison
    
    def cleanup_old_versions(self, model_name: str, keep_last_n: int = 10):
        """Clean up old model versions, keeping only the last N."""
        versions = self.list_model_versions(model_name=model_name, limit=1000)
        
        if len(versions) <= keep_last_n:
            return
        
        # Sort by creation date
        versions.sort(key=lambda v: v.created_at)
        
        # Archive old versions
        for version in versions[:-keep_last_n]:
            if version.status not in [ModelVersionStatus.DEPLOYED, ModelVersionStatus.ARCHIVED]:
                self.update_model_status(version.version_id, ModelVersionStatus.ARCHIVED)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Model version counts by status
        cursor.execute('''
            SELECT status, COUNT(*) FROM model_versions GROUP BY status
        ''')
        stats["model_versions_by_status"] = dict(cursor.fetchall())
        
        # Dataset version counts
        cursor.execute('SELECT COUNT(*) FROM dataset_versions')
        stats["total_dataset_versions"] = cursor.fetchone()[0]
        
        # Experiment counts by status
        cursor.execute('''
            SELECT status, COUNT(*) FROM experiments GROUP BY status
        ''')
        stats["experiments_by_status"] = dict(cursor.fetchall())
        
        # Marketplace statistics
        cursor.execute('SELECT COUNT(*) FROM marketplace_listings')
        stats["marketplace_listings"] = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(download_count) FROM marketplace_listings')
        stats["total_downloads"] = cursor.fetchone()[0] or 0
        
        # Storage statistics
        cursor.execute('SELECT COUNT(*) FROM model_versions')
        stats["total_model_versions"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# Global registry instance
_registry = None

def get_registry(registry_path: str = "~/.forge/registry") -> ModelRegistry:
    """Get or create the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_path)
    return _registry


# Example usage and CLI integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--model", help="Filter by model name")
    list_parser.add_argument("--status", help="Filter by status")
    
    # Show model details
    show_parser = subparsers.add_parser("show", help="Show model version details")
    show_parser.add_argument("version_id", help="Version ID to show")
    
    # Create branch
    branch_parser = subparsers.add_parser("branch", help="Create a branch")
    branch_parser.add_argument("model_name", help="Model name")
    branch_parser.add_argument("branch_name", help="Branch name")
    branch_parser.add_argument("source_version", help="Source version ID")
    
    # List experiments
    exp_parser = subparsers.add_parser("experiments", help="List experiments")
    exp_parser.add_argument("--status", help="Filter by status")
    
    # Marketplace search
    search_parser = subparsers.add_parser("search", help="Search marketplace")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--min-rating", type=float, help="Minimum rating")
    
    args = parser.parse_args()
    
    registry = get_registry()
    
    if args.command == "list":
        versions = registry.list_model_versions(
            model_name=args.model,
            status=ModelVersionStatus(args.status) if args.status else None
        )
        for v in versions:
            print(f"{v.version_id[:8]}... {v.model_name} v{v.version_number} [{v.status.value}]")
    
    elif args.command == "show":
        version = registry.get_model_version(args.version_id)
        if version:
            print(json.dumps(asdict(version), indent=2))
        else:
            print(f"Version {args.version_id} not found")
    
    elif args.command == "branch":
        registry.create_branch(args.model_name, args.branch_name, args.source_version)
        print(f"Created branch {args.model_name}/{args.branch_name}")
    
    elif args.command == "experiments":
        experiments = registry.list_experiments(
            status=ExperimentStatus(args.status) if args.status else None
        )
        for exp in experiments:
            print(f"{exp.experiment_id[:8]}... {exp.name} [{exp.status.value}]")
    
    elif args.command == "search":
        listings = registry.search_marketplace(
            query=args.query,
            min_rating=args.min_rating
        )
        for listing in listings:
            print(f"{listing.model_name} - {listing.description[:50]}... (⭐{listing.rating:.1f})")