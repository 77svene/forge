"""Model Registry & Versioning - Git-like versioning for models with content-addressable storage, lineage tracking, and A/B testing."""

import os
import json
import hashlib
import shutil
import sqlite3
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from studio.backend.core.data_recipe.huggingface import HuggingFaceDatasetLoader
from studio.backend.core.data_recipe.jobs.manager import JobManager


class ModelStatus(Enum):
    """Status of a model version."""
    PENDING = "pending"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ABTestStatus(Enum):
    """Status of an A/B test."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a versioned model with full lineage tracking."""
    model_id: str
    version_id: str
    name: str
    description: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    parent_version_id: Optional[str] = None
    dataset_version_id: Optional[str] = None
    training_job_id: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[ModelMetrics] = None
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    artifact_size: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        data = data.copy()
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('metrics'):
            data['metrics'] = ModelMetrics.from_dict(data['metrics'])
        return cls(**data)


@dataclass
class ABTest:
    """A/B test configuration between model versions."""
    test_id: str
    name: str
    description: str
    status: ABTestStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    model_variants: List[Dict[str, Any]]  # List of {model_id, version_id, traffic_percentage}
    target_metric: str
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTest':
        """Create from dictionary."""
        data = data.copy()
        data['status'] = ABTestStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


class ContentAddressableStorage:
    """Content-addressable storage for model artifacts."""
    
    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self.objects_dir = storage_root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def store(self, data: bytes, filename: str = None) -> Tuple[str, Path]:
        """Store data and return (hash, path)."""
        content_hash = self._compute_hash(data)
        hash_dir = self.objects_dir / content_hash[:2]
        hash_dir.mkdir(exist_ok=True)
        
        if filename:
            object_path = hash_dir / f"{content_hash}_{filename}"
        else:
            object_path = hash_dir / content_hash
        
        if not object_path.exists():
            with open(object_path, 'wb') as f:
                f.write(data)
        
        return content_hash, object_path
    
    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Retrieve data by hash."""
        hash_dir = self.objects_dir / content_hash[:2]
        if not hash_dir.exists():
            return None
        
        for file in hash_dir.glob(f"{content_hash}*"):
            if file.is_file():
                with open(file, 'rb') as f:
                    return f.read()
        
        return None
    
    def exists(self, content_hash: str) -> bool:
        """Check if object exists in storage."""
        hash_dir = self.objects_dir / content_hash[:2]
        if not hash_dir.exists():
            return False
        
        return any(hash_dir.glob(f"{content_hash}*"))


class ModelRegistry:
    """Model Registry with Git-like versioning, lineage tracking, and A/B testing."""
    
    def __init__(self, storage_root: Union[str, Path], db_path: Optional[Union[str, Path]] = None):
        """
        Initialize Model Registry.
        
        Args:
            storage_root: Root directory for storage
            db_path: Path to SQLite database (defaults to storage_root/registry.db)
        """
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        if db_path is None:
            self.db_path = self.storage_root / "registry.db"
        else:
            self.db_path = Path(db_path)
        
        self.content_storage = ContentAddressableStorage(self.storage_root / "artifacts")
        self._init_database()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Integration with existing modules
        self.job_manager = JobManager()
        self.dataset_loader = HuggingFaceDatasetLoader()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Model versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    parent_version_id TEXT,
                    dataset_version_id TEXT,
                    training_job_id TEXT,
                    hyperparameters TEXT DEFAULT '{}',
                    metrics TEXT,
                    artifact_path TEXT,
                    artifact_hash TEXT,
                    artifact_size INTEGER,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (model_id) REFERENCES models (model_id),
                    FOREIGN KEY (parent_version_id) REFERENCES model_versions (version_id)
                )
            ''')
            
            # A/B tests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    model_variants TEXT NOT NULL,
                    target_metric TEXT NOT NULL,
                    minimum_sample_size INTEGER DEFAULT 1000,
                    confidence_level REAL DEFAULT 0.95,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Lineage tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lineage (
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (source_type, source_id, target_type, target_id, relationship)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON model_versions (model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage (source_type, source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage (target_type, target_id)')
            
            conn.commit()
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())
    
    def _compute_content_hash(self, content: bytes) -> str:
        """Compute content hash for version ID."""
        return hashlib.sha256(content).hexdigest()[:12]
    
    def create_model(self, name: str, description: str = "", created_by: str = "system", 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new model in the registry.
        
        Args:
            name: Model name
            description: Model description
            created_by: Creator identifier
            metadata: Additional metadata
            
        Returns:
            model_id: Unique model identifier
        """
        with self._lock:
            model_id = self._generate_id()
            now = datetime.utcnow()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO models (model_id, name, description, created_at, updated_at, created_by, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    name,
                    description,
                    now.isoformat(),
                    now.isoformat(),
                    created_by,
                    json.dumps(metadata or {})
                ))
                conn.commit()
            
            return model_id
    
    def create_version(self, model_id: str, name: str, description: str = "", 
                       created_by: str = "system", parent_version_id: Optional[str] = None,
                       dataset_version_id: Optional[str] = None, 
                       hyperparameters: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new version of a model.
        
        Args:
            model_id: Parent model ID
            name: Version name
            description: Version description
            created_by: Creator identifier
            parent_version_id: Parent version ID for lineage tracking
            dataset_version_id: Dataset version used for training
            hyperparameters: Training hyperparameters
            tags: Version tags
            metadata: Additional metadata
            
        Returns:
            version_id: Unique version identifier
        """
        with self._lock:
            version_id = self._generate_id()
            now = datetime.utcnow()
            
            # Validate parent version exists if provided
            if parent_version_id:
                parent_version = self.get_version(parent_version_id)
                if not parent_version:
                    raise ValueError(f"Parent version {parent_version_id} not found")
                if parent_version.model_id != model_id:
                    raise ValueError("Parent version must belong to the same model")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_versions 
                    (version_id, model_id, name, description, status, created_at, updated_at, 
                     created_by, parent_version_id, dataset_version_id, hyperparameters, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    version_id,
                    model_id,
                    name,
                    description,
                    ModelStatus.PENDING.value,
                    now.isoformat(),
                    now.isoformat(),
                    created_by,
                    parent_version_id,
                    dataset_version_id,
                    json.dumps(hyperparameters or {}),
                    json.dumps(tags or []),
                    json.dumps(metadata or {})
                ))
                
                # Track lineage to parent version
                if parent_version_id:
                    self._track_lineage(
                        cursor, "model_version", parent_version_id, 
                        "model_version", version_id, "parent_child"
                    )
                
                # Track lineage to dataset
                if dataset_version_id:
                    self._track_lineage(
                        cursor, "dataset_version", dataset_version_id,
                        "model_version", version_id, "trained_on"
                    )
                
                # Track lineage to model
                self._track_lineage(
                    cursor, "model", model_id,
                    "model_version", version_id, "has_version"
                )
                
                conn.commit()
            
            return version_id
    
    def _track_lineage(self, cursor, source_type: str, source_id: str, 
                      target_type: str, target_id: str, relationship: str,
                      metadata: Optional[Dict[str, Any]] = None):
        """Track lineage relationship between entities."""
        cursor.execute('''
            INSERT OR REPLACE INTO lineage 
            (source_type, source_id, target_type, target_id, relationship, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            source_type, source_id, target_type, target_id, relationship,
            json.dumps(metadata or {})
        ))
    
    def update_version_status(self, version_id: str, status: ModelStatus, 
                              error_message: Optional[str] = None) -> bool:
        """
        Update the status of a model version.
        
        Args:
            version_id: Version ID
            status: New status
            error_message: Error message if status is FAILED
            
        Returns:
            bool: True if update successful
        """
        with self._lock:
            now = datetime.utcnow()
            metadata_update = {}
            
            if error_message:
                metadata_update['error_message'] = error_message
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current metadata
                cursor.execute('SELECT metadata FROM model_versions WHERE version_id = ?', (version_id,))
                result = cursor.fetchone()
                if not result:
                    return False
                
                current_metadata = json.loads(result[0])
                current_metadata.update(metadata_update)
                
                cursor.execute('''
                    UPDATE model_versions 
                    SET status = ?, updated_at = ?, metadata = ?
                    WHERE version_id = ?
                ''', (status.value, now.isoformat(), json.dumps(current_metadata), version_id))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def store_model_artifact(self, version_id: str, artifact_data: bytes, 
                            filename: Optional[str] = None) -> bool:
        """
        Store model artifact with content-addressable storage.
        
        Args:
            version_id: Version ID
            artifact_data: Model artifact bytes
            filename: Optional filename
            
        Returns:
            bool: True if storage successful
        """
        with self._lock:
            content_hash, artifact_path = self.content_storage.store(artifact_data, filename)
            relative_path = artifact_path.relative_to(self.storage_root)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE model_versions 
                    SET artifact_path = ?, artifact_hash = ?, artifact_size = ?
                    WHERE version_id = ?
                ''', (str(relative_path), content_hash, len(artifact_data), version_id))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def get_model_artifact(self, version_id: str) -> Optional[Tuple[bytes, str]]:
        """
        Retrieve model artifact.
        
        Args:
            version_id: Version ID
            
        Returns:
            Tuple of (artifact_data, filename) or None if not found
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT artifact_path, artifact_hash FROM model_versions 
                    WHERE version_id = ?
                ''', (version_id,))
                result = cursor.fetchone()
                
                if not result or not result[0]:
                    return None
                
                artifact_path, artifact_hash = result
                full_path = self.storage_root / artifact_path
                
                if not full_path.exists():
                    # Try to retrieve from content storage
                    data = self.content_storage.retrieve(artifact_hash)
                    if data:
                        return data, full_path.name
                    return None
                
                with open(full_path, 'rb') as f:
                    return f.read(), full_path.name
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM model_versions WHERE version_id = ?
                ''', (version_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return ModelVersion.from_dict(dict(row))
    
    def get_latest_version(self, model_id: str, status: Optional[ModelStatus] = None) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM model_versions 
                    WHERE model_id = ?
                '''
                params = [model_id]
                
                if status:
                    query += ' AND status = ?'
                    params.append(status.value)
                
                query += ' ORDER BY created_at DESC LIMIT 1'
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return ModelVersion.from_dict(dict(row))
    
    def list_versions(self, model_id: str, limit: int = 100, offset: int = 0,
                     status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """List versions of a model."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM model_versions 
                    WHERE model_id = ?
                '''
                params = [model_id]
                
                if status:
                    query += ' AND status = ?'
                    params.append(status.value)
                
                query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [ModelVersion.from_dict(dict(row)) for row in rows]
    
    def get_lineage(self, version_id: str, direction: str = "both", 
                   depth: int = 10) -> Dict[str, Any]:
        """
        Get lineage tree for a model version.
        
        Args:
            version_id: Starting version ID
            direction: "parents", "children", or "both"
            depth: Maximum depth to traverse
            
        Returns:
            Lineage tree as nested dictionary
        """
        def traverse(current_id: str, current_depth: int, visited: set) -> Dict[str, Any]:
            if current_depth >= depth or current_id in visited:
                return {}
            
            visited.add(current_id)
            result = {}
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get current version info
                cursor.execute('SELECT * FROM model_versions WHERE version_id = ?', (current_id,))
                version_row = cursor.fetchone()
                if not version_row:
                    return {}
                
                version_data = dict(version_row)
                result['version'] = ModelVersion.from_dict(version_data).to_dict()
                
                # Get parents
                if direction in ["parents", "both"]:
                    cursor.execute('''
                        SELECT source_id FROM lineage 
                        WHERE target_type = 'model_version' AND target_id = ? 
                        AND relationship = 'parent_child'
                    ''', (current_id,))
                    parent_rows = cursor.fetchall()
                    
                    if parent_rows:
                        result['parents'] = []
                        for parent_row in parent_rows:
                            parent_id = parent_row['source_id']
                            parent_data = traverse(parent_id, current_depth + 1, visited.copy())
                            if parent_data:
                                result['parents'].append(parent_data)
                
                # Get children
                if direction in ["children", "both"]:
                    cursor.execute('''
                        SELECT target_id FROM lineage 
                        WHERE source_type = 'model_version' AND source_id = ? 
                        AND relationship = 'parent_child'
                    ''', (current_id,))
                    child_rows = cursor.fetchall()
                    
                    if child_rows:
                        result['children'] = []
                        for child_row in child_rows:
                            child_id = child_row['target_id']
                            child_data = traverse(child_id, current_depth + 1, visited.copy())
                            if child_data:
                                result['children'].append(child_data)
                
                # Get dataset lineage
                cursor.execute('''
                    SELECT source_id FROM lineage 
                    WHERE target_type = 'model_version' AND target_id = ? 
                    AND relationship = 'trained_on'
                ''', (current_id,))
                dataset_rows = cursor.fetchall()
                
                if dataset_rows:
                    result['datasets'] = []
                    for dataset_row in dataset_rows:
                        dataset_id = dataset_row['source_id']
                        result['datasets'].append({'dataset_version_id': dataset_id})
            
            return result
        
        return traverse(version_id, 0, set())
    
    def create_ab_test(self, name: str, description: str, created_by: str,
                      model_variants: List[Dict[str, Any]], target_metric: str,
                      minimum_sample_size: int = 1000, confidence_level: float = 0.95,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an A/B test between model versions.
        
        Args:
            name: Test name
            description: Test description
            created_by: Creator identifier
            model_variants: List of dicts with model_id, version_id, traffic_percentage
            target_metric: Metric to optimize
            minimum_sample_size: Minimum samples per variant
            confidence_level: Statistical confidence level
            metadata: Additional metadata
            
        Returns:
            test_id: Unique test identifier
        """
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.get('traffic_percentage', 0) for v in model_variants)
        if abs(total_traffic - 100) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")
        
        # Validate all versions exist
        for variant in model_variants:
            version = self.get_version(variant['version_id'])
            if not version:
                raise ValueError(f"Version {variant['version_id']} not found")
            if version.model_id != variant['model_id']:
                raise ValueError(f"Version {variant['version_id']} does not belong to model {variant['model_id']}")
        
        with self._lock:
            test_id = self._generate_id()
            now = datetime.utcnow()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ab_tests 
                    (test_id, name, description, status, created_at, updated_at, created_by,
                     model_variants, target_metric, minimum_sample_size, confidence_level, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_id,
                    name,
                    description,
                    ABTestStatus.DRAFT.value,
                    now.isoformat(),
                    now.isoformat(),
                    created_by,
                    json.dumps(model_variants),
                    target_metric,
                    minimum_sample_size,
                    confidence_level,
                    json.dumps(metadata or {})
                ))
                
                # Track lineage from versions to test
                for variant in model_variants:
                    self._track_lineage(
                        cursor, "model_version", variant['version_id'],
                        "ab_test", test_id, "participates_in"
                    )
                
                conn.commit()
            
            return test_id
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test."""
        with self._lock:
            now = datetime.utcnow()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE ab_tests 
                    SET status = ?, start_time = ?, updated_at = ?
                    WHERE test_id = ? AND status = ?
                ''', (ABTestStatus.RUNNING.value, now.isoformat(), now.isoformat(),
                     test_id, ABTestStatus.DRAFT.value))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def get_ab_test(self, test_id: str) -> Optional[ABTest]:
        """Get A/B test details."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ab_tests WHERE test_id = ?', (test_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return ABTest.from_dict(dict(row))
    
    def list_ab_tests(self, status: Optional[ABTestStatus] = None, 
                     limit: int = 100, offset: int = 0) -> List[ABTest]:
        """List A/B tests."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM ab_tests'
                params = []
                
                if status:
                    query += ' WHERE status = ?'
                    params.append(status.value)
                
                query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [ABTest.from_dict(dict(row)) for row in rows]
    
    def get_variant_for_request(self, test_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model variant for a specific request using consistent hashing.
        
        Args:
            test_id: A/B test ID
            request_id: Unique request identifier
            
        Returns:
            Variant dict with model_id, version_id, traffic_percentage
        """
        test = self.get_ab_test(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        # Use consistent hashing to assign request to variant
        hash_input = f"{test_id}:{request_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0.00 to 99.99
        
        cumulative = 0.0
        for variant in test.model_variants:
            cumulative += variant.get('traffic_percentage', 0)
            if percentage < cumulative:
                return variant
        
        return test.model_variants[-1] if test.model_variants else None
    
    def record_ab_test_result(self, test_id: str, version_id: str, 
                             request_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Record result for an A/B test variant.
        
        Args:
            test_id: A/B test ID
            version_id: Model version ID
            request_id: Request identifier
            metrics: Performance metrics
            
        Returns:
            bool: True if recording successful
        """
        # In a production system, this would store results in a time-series database
        # For now, we'll just validate the test exists and is running
        test = self.get_ab_test(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return False
        
        # Validate version is part of the test
        version_ids = [v['version_id'] for v in test.model_variants]
        if version_id not in version_ids:
            return False
        
        # Store result (simplified - in production would use proper analytics storage)
        result_key = f"ab_result:{test_id}:{version_id}:{request_id}"
        # This would typically go to Redis or a similar store
        
        return True
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get aggregated results for an A/B test.
        
        Args:
            test_id: A/B test ID
            
        Returns:
            Dictionary with aggregated metrics per variant
        """
        # In production, this would query the analytics database
        # For now, return a placeholder structure
        test = self.get_ab_test(test_id)
        if not test:
            return {}
        
        results = {
            'test_id': test_id,
            'name': test.name,
            'status': test.status.value,
            'target_metric': test.target_metric,
            'variants': []
        }
        
        for variant in test.model_variants:
            version = self.get_version(variant['version_id'])
            results['variants'].append({
                'model_id': variant['model_id'],
                'version_id': variant['version_id'],
                'version_name': version.name if version else 'Unknown',
                'traffic_percentage': variant['traffic_percentage'],
                'sample_size': 0,  # Would be populated from analytics
                'metrics': {}  # Would be populated from analytics
            })
        
        return results
    
    def search_models(self, query: str, tags: Optional[List[str]] = None,
                     status: Optional[ModelStatus] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search models by name, description, or tags.
        
        Args:
            query: Search query
            tags: Filter by tags
            status: Filter by status
            limit: Maximum results
            
        Returns:
            List of model summaries with versions
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Search models
                cursor.execute('''
                    SELECT * FROM models 
                    WHERE name LIKE ? OR description LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
                
                model_rows = cursor.fetchall()
                results = []
                
                for model_row in model_rows:
                    model_data = dict(model_row)
                    
                    # Get latest version
                    cursor.execute('''
                        SELECT * FROM model_versions 
                        WHERE model_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    ''', (model_data['model_id'],))
                    
                    version_row = cursor.fetchone()
                    version_data = dict(version_row) if version_row else None
                    
                    # Filter by tags if provided
                    if tags and version_data:
                        version_tags = json.loads(version_data.get('tags', '[]'))
                        if not any(tag in version_tags for tag in tags):
                            continue
                    
                    # Filter by status if provided
                    if status and version_data:
                        if version_data['status'] != status.value:
                            continue
                    
                    results.append({
                        'model': model_data,
                        'latest_version': version_data
                    })
                
                return results
    
    def export_model(self, version_id: str, export_path: Union[str, Path]) -> bool:
        """
        Export a model version to a directory.
        
        Args:
            version_id: Version ID to export
            export_path: Directory to export to
            
        Returns:
            bool: True if export successful
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        version = self.get_version(version_id)
        if not version:
            return False
        
        # Get artifact
        artifact_data, artifact_filename = self.get_model_artifact(version_id)
        if not artifact_data:
            return False
        
        # Save artifact
        artifact_path = export_path / artifact_filename
        with open(artifact_path, 'wb') as f:
            f.write(artifact_data)
        
        # Save metadata
        metadata_path = export_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Save lineage
        lineage = self.get_lineage(version_id, direction="parents")
        lineage_path = export_path / "lineage.json"
        with open(lineage_path, 'w') as f:
            json.dump(lineage, f, indent=2)
        
        return True
    
    def import_model(self, import_path: Union[str, Path], model_id: Optional[str] = None,
                    created_by: str = "system") -> Optional[str]:
        """
        Import a model from an exported directory.
        
        Args:
            import_path: Directory containing exported model
            model_id: Optional model ID to add version to (creates new model if None)
            created_by: Creator identifier
            
        Returns:
            version_id: Imported version ID or None if failed
        """
        import_path = Path(import_path)
        
        # Load metadata
        metadata_path = import_path / "model_metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Find artifact file
        artifact_files = list(import_path.glob("*"))
        artifact_files = [f for f in artifact_files if f.name != "model_metadata.json" and f.name != "lineage.json"]
        
        if not artifact_files:
            return None
        
        artifact_path = artifact_files[0]
        
        # Read artifact
        with open(artifact_path, 'rb') as f:
            artifact_data = f.read()
        
        # Create or get model
        if model_id is None:
            model_id = self.create_model(
                name=metadata.get('name', 'Imported Model'),
                description=metadata.get('description', ''),
                created_by=created_by
            )
        
        # Create version
        version_id = self.create_version(
            model_id=model_id,
            name=metadata.get('name', 'Imported Version'),
            description=metadata.get('description', ''),
            created_by=created_by,
            hyperparameters=metadata.get('hyperparameters', {}),
            tags=metadata.get('tags', []),
            metadata=metadata.get('metadata', {})
        )
        
        # Store artifact
        self.store_model_artifact(version_id, artifact_data, artifact_path.name)
        
        # Update status to ready
        self.update_version_status(version_id, ModelStatus.READY)
        
        return version_id
    
    def get_marketplace_models(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get models available in the marketplace (public models with READY status).
        
        Args:
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of marketplace model summaries
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get models with at least one READY version
                cursor.execute('''
                    SELECT DISTINCT m.* FROM models m
                    JOIN model_versions mv ON m.model_id = mv.model_id
                    WHERE mv.status = ?
                    ORDER BY m.updated_at DESC
                    LIMIT ? OFFSET ?
                ''', (ModelStatus.READY.value, limit, offset))
                
                model_rows = cursor.fetchall()
                results = []
                
                for model_row in model_rows:
                    model_data = dict(model_row)
                    
                    # Get all ready versions
                    cursor.execute('''
                        SELECT * FROM model_versions 
                        WHERE model_id = ? AND status = ?
                        ORDER BY created_at DESC
                    ''', (model_data['model_id'], ModelStatus.READY.value))
                    
                    version_rows = cursor.fetchall()
                    versions = [ModelVersion.from_dict(dict(row)).to_dict() for row in version_rows]
                    
                    results.append({
                        'model': model_data,
                        'versions': versions,
                        'version_count': len(versions)
                    })
                
                return results
    
    def clone_version(self, source_version_id: str, new_name: Optional[str] = None,
                     created_by: str = "system") -> Optional[str]:
        """
        Clone a model version.
        
        Args:
            source_version_id: Source version ID to clone
            new_name: Optional new name (defaults to source name + " clone")
            created_by: Creator identifier
            
        Returns:
            version_id: Cloned version ID or None if failed
        """
        source_version = self.get_version(source_version_id)
        if not source_version:
            return None
        
        # Create new version
        new_name = new_name or f"{source_version.name} clone"
        new_version_id = self.create_version(
            model_id=source_version.model_id,
            name=new_name,
            description=f"Cloned from {source_version_id}",
            created_by=created_by,
            parent_version_id=source_version_id,
            dataset_version_id=source_version.dataset_version_id,
            hyperparameters=source_version.hyperparameters.copy(),
            tags=source_version.tags.copy() + ["cloned"],
            metadata=source_version.metadata.copy()
        )
        
        # Copy artifact if exists
        artifact_data, artifact_filename = self.get_model_artifact(source_version_id)
        if artifact_data:
            self.store_model_artifact(new_version_id, artifact_data, artifact_filename)
            self.update_version_status(new_version_id, ModelStatus.READY)
        
        return new_version_id
    
    def cleanup_old_versions(self, model_id: str, keep_last_n: int = 5,
                            keep_statuses: Optional[List[ModelStatus]] = None) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.
        
        Args:
            model_id: Model ID to clean up
            keep_last_n: Number of recent versions to keep
            keep_statuses: Statuses to always keep (e.g., DEPLOYED)
            
        Returns:
            int: Number of versions archived
        """
        if keep_statuses is None:
            keep_statuses = [ModelStatus.DEPLOYED]
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all versions ordered by creation date
                cursor.execute('''
                    SELECT version_id, status FROM model_versions 
                    WHERE model_id = ? 
                    ORDER BY created_at DESC
                ''', (model_id,))
                
                versions = cursor.fetchall()
                archived_count = 0
                
                # Keep track of versions to preserve
                preserve_versions = set()
                
                # Always keep versions with specified statuses
                for version in versions:
                    if ModelStatus(version['status']) in keep_statuses:
                        preserve_versions.add(version['version_id'])
                
                # Keep the last N versions
                for i, version in enumerate(versions[:keep_last_n]):
                    preserve_versions.add(version['version_id'])
                
                # Archive old versions not in preserve list
                for version in versions:
                    version_id = version['version_id']
                    if version_id not in preserve_versions:
                        self.update_version_status(version_id, ModelStatus.ARCHIVED)
                        archived_count += 1
                
                return archived_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count models
                cursor.execute('SELECT COUNT(*) FROM models')
                stats['total_models'] = cursor.fetchone()[0]
                
                # Count versions by status
                cursor.execute('''
                    SELECT status, COUNT(*) as count 
                    FROM model_versions 
                    GROUP BY status
                ''')
                stats['versions_by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Count A/B tests by status
                cursor.execute('''
                    SELECT status, COUNT(*) as count 
                    FROM ab_tests 
                    GROUP BY status
                ''')
                stats['ab_tests_by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Storage usage
                cursor.execute('SELECT SUM(artifact_size) FROM model_versions WHERE artifact_size IS NOT NULL')
                total_size = cursor.fetchone()[0]
                stats['total_artifact_size_bytes'] = total_size or 0
                stats['total_artifact_size_mb'] = (total_size or 0) / (1024 * 1024)
                
                return stats


# Singleton instance for global access
_registry_instance = None
_registry_lock = threading.Lock()


def get_model_registry(storage_root: Optional[Union[str, Path]] = None) -> ModelRegistry:
    """
    Get or create the global model registry instance.
    
    Args:
        storage_root: Storage root directory (uses default if None)
        
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    
    with _registry_lock:
        if _registry_instance is None:
            if storage_root is None:
                # Use default storage location
                storage_root = Path.home() / ".forge" / "model_registry"
            
            _registry_instance = ModelRegistry(storage_root)
        
        return _registry_instance


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the registry
    registry = ModelRegistry("./test_registry")
    
    # Create a model
    model_id = registry.create_model(
        name="llama-3-8b",
        description="Llama 3 8B parameter model",
        created_by="user@example.com"
    )
    
    # Create a version
    version_id = registry.create_version(
        model_id=model_id,
        name="v1.0",
        description="Initial version",
        created_by="user@example.com",
        hyperparameters={"learning_rate": 0.001, "epochs": 3}
    )
    
    # Store artifact (simulated)
    artifact_data = b"fake model weights data"
    registry.store_model_artifact(version_id, artifact_data, "model.bin")
    
    # Update status
    registry.update_version_status(version_id, ModelStatus.READY)
    
    # Create A/B test
    test_id = registry.create_ab_test(
        name="Llama v1 vs v2",
        description="Testing new fine-tuning approach",
        created_by="user@example.com",
        model_variants=[
            {"model_id": model_id, "version_id": version_id, "traffic_percentage": 50},
            {"model_id": model_id, "version_id": "future-version-id", "traffic_percentage": 50}
        ],
        target_metric="accuracy"
    )
    
    print(f"Created model: {model_id}")
    print(f"Created version: {version_id}")
    print(f"Created A/B test: {test_id}")
    print(f"Registry stats: {registry.get_statistics()}")