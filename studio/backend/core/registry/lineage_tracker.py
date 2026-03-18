"""Model Registry & Versioning — DVC-like model versioning with lineage tracking, A/B testing, and marketplace."""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import shutil
from contextlib import contextmanager

from studio.backend.auth.storage import get_db_connection
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.huggingface import HuggingFaceDataset


class ModelStatus(Enum):
    """Model version status in the registry."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ABTestStatus(Enum):
    """A/B test status."""
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
    evaluation_dataset_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LineageNode:
    """Represents a node in the lineage graph."""
    node_id: str
    node_type: str  # "dataset", "model", "recipe", "deployment"
    name: str
    version: str
    hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: Optional[str] = None


@dataclass
class ModelVersion:
    """Represents a versioned model in the registry."""
    model_id: str
    version: str
    name: str
    description: str
    status: ModelStatus
    model_hash: str
    config_hash: str
    dataset_hashes: List[str]
    recipe_id: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    lineage: List[LineageNode] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: Optional[str] = None
    deployed_at: Optional[str] = None
    artifact_path: Optional[str] = None


@dataclass
class ABTest:
    """A/B test configuration between model versions."""
    test_id: str
    name: str
    description: str
    status: ABTestStatus
    model_versions: List[str]  # List of model version IDs
    traffic_allocation: Dict[str, float]  # model_version_id -> traffic percentage
    metrics_to_track: List[str]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


class ContentAddressableStorage:
    """Content-addressable storage for models and datasets."""
    
    def __init__(self, base_path: str = ".forge/registry"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.datasets_path = self.base_path / "datasets"
        self.configs_path = self.base_path / "configs"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.models_path, self.datasets_path, self.configs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def store_artifact(self, data: bytes, artifact_type: str = "model") -> str:
        """Store artifact and return its hash."""
        artifact_hash = self.compute_hash(data)
        
        if artifact_type == "model":
            path = self.models_path / artifact_hash
        elif artifact_type == "dataset":
            path = self.datasets_path / artifact_hash
        elif artifact_type == "config":
            path = self.configs_path / artifact_hash
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        if not path.exists():
            path.write_bytes(data)
        
        return artifact_hash
    
    def get_artifact(self, artifact_hash: str, artifact_type: str = "model") -> Optional[bytes]:
        """Retrieve artifact by hash."""
        if artifact_type == "model":
            path = self.models_path / artifact_hash
        elif artifact_type == "dataset":
            path = self.datasets_path / artifact_hash
        elif artifact_type == "config":
            path = self.configs_path / artifact_hash
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        if path.exists():
            return path.read_bytes()
        return None
    
    def delete_artifact(self, artifact_hash: str, artifact_type: str = "model") -> bool:
        """Delete artifact by hash."""
        if artifact_type == "model":
            path = self.models_path / artifact_hash
        elif artifact_type == "dataset":
            path = self.datasets_path / artifact_hash
        elif artifact_type == "config":
            path = self.configs_path / artifact_hash
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        if path.exists():
            path.unlink()
            return True
        return False


class LineageTracker:
    """Tracks model lineage from data to deployment."""
    
    def __init__(self, storage: ContentAddressableStorage):
        self.storage = storage
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for lineage tracking."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    dataset_hashes TEXT NOT NULL,
                    recipe_id TEXT,
                    metrics TEXT,
                    lineage TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    deployed_at TEXT,
                    artifact_path TEXT,
                    UNIQUE(model_id)
                )
            """)
            
            # A/B tests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    model_versions TEXT NOT NULL,
                    traffic_allocation TEXT NOT NULL,
                    metrics_to_track TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    results TEXT,
                    UNIQUE(test_id)
                )
            """)
            
            # Lineage relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage_relationships (
                    parent_hash TEXT NOT NULL,
                    child_hash TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (parent_hash, child_hash, relationship_type)
                )
            """)
            
            # Model marketplace table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_marketplace (
                    listing_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT NOT NULL,
                    license TEXT,
                    price REAL DEFAULT 0.0,
                    downloads INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0.0,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
                )
            """)
            
            conn.commit()
    
    def create_model_version(
        self,
        name: str,
        description: str,
        model_data: bytes,
        config_data: bytes,
        dataset_hashes: List[str],
        recipe_id: Optional[str] = None,
        metrics: Optional[ModelMetrics] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> ModelVersion:
        """Create a new model version with lineage tracking."""
        
        # Store artifacts in content-addressable storage
        model_hash = self.storage.store_artifact(model_data, "model")
        config_hash = self.storage.store_artifact(config_data, "config")
        
        # Generate model ID and version
        model_id = str(uuid.uuid4())
        version = self._generate_version(name)
        
        # Build lineage nodes
        lineage = []
        
        # Add dataset nodes
        for dataset_hash in dataset_hashes:
            lineage.append(LineageNode(
                node_id=dataset_hash,
                node_type="dataset",
                name=f"Dataset-{dataset_hash[:8]}",
                version="1.0",
                hash=dataset_hash,
                created_by=created_by
            ))
        
        # Add config node
        lineage.append(LineageNode(
            node_id=config_hash,
            node_type="config",
            name=f"Config-{config_hash[:8]}",
            version="1.0",
            hash=config_hash,
            created_by=created_by
        ))
        
        # Add model node
        lineage.append(LineageNode(
            node_id=model_hash,
            node_type="model",
            name=name,
            version=version,
            hash=model_hash,
            metadata={"model_id": model_id},
            created_by=created_by
        ))
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            status=ModelStatus.READY,
            model_hash=model_hash,
            config_hash=config_hash,
            dataset_hashes=dataset_hashes,
            recipe_id=recipe_id,
            metrics=metrics,
            lineage=lineage,
            tags=tags or [],
            metadata=metadata or {},
            created_by=created_by
        )
        
        # Store in database
        self._store_model_version(model_version)
        
        # Store lineage relationships
        self._store_lineage_relationships(model_version)
        
        return model_version
    
    def _generate_version(self, name: str) -> str:
        """Generate semantic version based on existing versions."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM model_versions WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,)
            )
            result = cursor.fetchone()
            
            if not result:
                return "1.0.0"
            
            last_version = result[0]
            try:
                major, minor, patch = map(int, last_version.split('.'))
                return f"{major}.{minor}.{patch + 1}"
            except ValueError:
                # If version doesn't follow semver, use timestamp
                return datetime.utcnow().strftime("%Y%m%d.%H%M%S")
    
    def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_versions 
                (model_id, version, name, description, status, model_hash, config_hash, 
                 dataset_hashes, recipe_id, metrics, lineage, tags, metadata, 
                 created_at, created_by, deployed_at, artifact_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_version.model_id,
                    model_version.version,
                    model_version.name,
                    model_version.description,
                    model_version.status.value,
                    model_version.model_hash,
                    model_version.config_hash,
                    json.dumps(model_version.dataset_hashes),
                    model_version.recipe_id,
                    json.dumps(asdict(model_version.metrics)) if model_version.metrics else None,
                    json.dumps([asdict(node) for node in model_version.lineage]),
                    json.dumps(model_version.tags),
                    json.dumps(model_version.metadata),
                    model_version.created_at,
                    model_version.created_by,
                    model_version.deployed_at,
                    model_version.artifact_path
                )
            )
            conn.commit()
    
    def _store_lineage_relationships(self, model_version: ModelVersion):
        """Store lineage relationships in database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Store dataset -> model relationships
            for dataset_hash in model_version.dataset_hashes:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO lineage_relationships 
                    (parent_hash, child_hash, relationship_type, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_hash,
                        model_version.model_hash,
                        "trained_on",
                        json.dumps({"model_id": model_version.model_id}),
                        datetime.utcnow().isoformat()
                    )
                )
            
            # Store config -> model relationship
            cursor.execute(
                """
                INSERT OR REPLACE INTO lineage_relationships 
                (parent_hash, child_hash, relationship_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_version.config_hash,
                    model_version.model_hash,
                    "configured_by",
                    json.dumps({"model_id": model_version.model_id}),
                    datetime.utcnow().isoformat()
                )
            )
            
            conn.commit()
    
    def get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        """Retrieve model version by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_id, version, name, description, status, model_hash, 
                       config_hash, dataset_hashes, recipe_id, metrics, lineage, 
                       tags, metadata, created_at, created_by, deployed_at, artifact_path
                FROM model_versions WHERE model_id = ?
                """,
                (model_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            return self._row_to_model_version(result)
    
    def _row_to_model_version(self, row: tuple) -> ModelVersion:
        """Convert database row to ModelVersion object."""
        (
            model_id, version, name, description, status, model_hash,
            config_hash, dataset_hashes_json, recipe_id, metrics_json,
            lineage_json, tags_json, metadata_json, created_at, created_by,
            deployed_at, artifact_path
        ) = row
        
        # Parse JSON fields
        dataset_hashes = json.loads(dataset_hashes_json)
        metrics_data = json.loads(metrics_json) if metrics_json else None
        lineage_data = json.loads(lineage_json) if lineage_json else []
        tags = json.loads(tags_json)
        metadata = json.loads(metadata_json)
        
        # Reconstruct objects
        metrics = ModelMetrics(**metrics_data) if metrics_data else None
        lineage = [LineageNode(**node_data) for node_data in lineage_data]
        
        return ModelVersion(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            status=ModelStatus(status),
            model_hash=model_hash,
            config_hash=config_hash,
            dataset_hashes=dataset_hashes,
            recipe_id=recipe_id,
            metrics=metrics,
            lineage=lineage,
            tags=tags,
            metadata=metadata,
            created_at=created_at,
            created_by=created_by,
            deployed_at=deployed_at,
            artifact_path=artifact_path
        )
    
    def list_model_versions(
        self,
        name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ModelVersion]:
        """List model versions with filtering."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT model_id, version, name, description, status, model_hash, 
                       config_hash, dataset_hashes, recipe_id, metrics, lineage, 
                       tags, metadata, created_at, created_by, deployed_at, artifact_path
                FROM model_versions
                WHERE 1=1
            """
            params = []
            
            if name:
                query += " AND name LIKE ?"
                params.append(f"%{name}%")
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [self._row_to_model_version(row) for row in results]
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model version status."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            update_fields = {"status": status.value}
            if status == ModelStatus.DEPLOYED:
                update_fields["deployed_at"] = datetime.utcnow().isoformat()
            
            set_clause = ", ".join([f"{field} = ?" for field in update_fields.keys()])
            values = list(update_fields.values())
            values.append(model_id)
            
            cursor.execute(
                f"UPDATE model_versions SET {set_clause} WHERE model_id = ?",
                values
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    def get_lineage_tree(self, model_hash: str, depth: int = 10) -> Dict[str, Any]:
        """Get lineage tree for a model."""
        def build_tree(current_hash: str, current_depth: int, visited: Set[str]) -> Dict[str, Any]:
            if current_depth >= depth or current_hash in visited:
                return {"hash": current_hash, "children": []}
            
            visited.add(current_hash)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT parent_hash, relationship_type, metadata 
                    FROM lineage_relationships 
                    WHERE child_hash = ?
                    """,
                    (current_hash,)
                )
                parents = cursor.fetchall()
            
            children = []
            for parent_hash, rel_type, metadata_json in parents:
                metadata = json.loads(metadata_json) if metadata_json else {}
                child_tree = build_tree(parent_hash, current_depth + 1, visited)
                child_tree["relationship"] = rel_type
                child_tree["metadata"] = metadata
                children.append(child_tree)
            
            return {"hash": current_hash, "children": children}
        
        return build_tree(model_hash, 0, set())
    
    def find_similar_models(self, model_hash: str, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find models with similar lineage (shared datasets/configs)."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get datasets and config for the target model
            cursor.execute(
                """
                SELECT parent_hash, relationship_type 
                FROM lineage_relationships 
                WHERE child_hash = ?
                """,
                (model_hash,)
            )
            target_relationships = cursor.fetchall()
            
            target_hashes = {rel[0] for rel in target_relationships}
            
            # Find other models sharing these relationships
            similar_models = []
            
            for parent_hash, rel_type in target_relationships:
                cursor.execute(
                    """
                    SELECT DISTINCT child_hash, metadata 
                    FROM lineage_relationships 
                    WHERE parent_hash = ? AND child_hash != ?
                    """,
                    (parent_hash, model_hash)
                )
                
                for child_hash, metadata_json in cursor.fetchall():
                    # Get model info
                    cursor.execute(
                        """
                        SELECT model_id, name, version, metrics 
                        FROM model_versions 
                        WHERE model_hash = ?
                        """,
                        (child_hash,)
                    )
                    model_info = cursor.fetchone()
                    
                    if model_info:
                        model_id, name, version, metrics_json = model_info
                        metrics = json.loads(metrics_json) if metrics_json else {}
                        
                        # Calculate similarity score
                        cursor.execute(
                            """
                            SELECT COUNT(*) as shared_count
                            FROM lineage_relationships 
                            WHERE child_hash = ? AND parent_hash IN (
                                SELECT parent_hash 
                                FROM lineage_relationships 
                                WHERE child_hash = ?
                            )
                            """,
                            (child_hash, model_hash)
                        )
                        shared_count = cursor.fetchone()[0]
                        
                        similarity = shared_count / len(target_hashes) if target_hashes else 0
                        
                        if similarity >= similarity_threshold:
                            similar_models.append({
                                "model_id": model_id,
                                "name": name,
                                "version": version,
                                "similarity": similarity,
                                "metrics": metrics,
                                "shared_components": shared_count
                            })
            
            # Remove duplicates and sort by similarity
            seen = set()
            unique_models = []
            for model in similar_models:
                if model["model_id"] not in seen:
                    seen.add(model["model_id"])
                    unique_models.append(model)
            
            return sorted(unique_models, key=lambda x: x["similarity"], reverse=True)


class ABTestManager:
    """Manages A/B testing between model versions."""
    
    def __init__(self, lineage_tracker: LineageTracker):
        self.lineage_tracker = lineage_tracker
    
    def create_ab_test(
        self,
        name: str,
        description: str,
        model_version_ids: List[str],
        traffic_allocation: Dict[str, float],
        metrics_to_track: List[str],
        created_by: Optional[str] = None
    ) -> ABTest:
        """Create a new A/B test."""
        
        # Validate model versions exist
        for model_id in model_version_ids:
            model = self.lineage_tracker.get_model_version(model_id)
            if not model:
                raise ValueError(f"Model version {model_id} not found")
            if model.status != ModelStatus.READY:
                raise ValueError(f"Model version {model_id} is not in READY state")
        
        # Validate traffic allocation
        total_traffic = sum(traffic_allocation.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_traffic}")
        
        for model_id in traffic_allocation.keys():
            if model_id not in model_version_ids:
                raise ValueError(f"Model {model_id} not in test model versions")
        
        # Create A/B test
        test_id = str(uuid.uuid4())
        ab_test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            status=ABTestStatus.DRAFT,
            model_versions=model_version_ids,
            traffic_allocation=traffic_allocation,
            metrics_to_track=metrics_to_track,
            created_by=created_by
        )
        
        # Store in database
        self._store_ab_test(ab_test)
        
        return ab_test
    
    def _store_ab_test(self, ab_test: ABTest):
        """Store A/B test in database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ab_tests 
                (test_id, name, description, status, model_versions, 
                 traffic_allocation, metrics_to_track, start_time, end_time,
                 created_at, created_by, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ab_test.test_id,
                    ab_test.name,
                    ab_test.description,
                    ab_test.status.value,
                    json.dumps(ab_test.model_versions),
                    json.dumps(ab_test.traffic_allocation),
                    json.dumps(ab_test.metrics_to_track),
                    ab_test.start_time,
                    ab_test.end_time,
                    ab_test.created_at,
                    ab_test.created_by,
                    json.dumps(ab_test.results)
                )
            )
            conn.commit()
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check current status
            cursor.execute("SELECT status FROM ab_tests WHERE test_id = ?", (test_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            current_status = ABTestStatus(result[0])
            if current_status != ABTestStatus.DRAFT:
                raise ValueError(f"Test must be in DRAFT status to start, current: {current_status}")
            
            # Update status and start time
            cursor.execute(
                """
                UPDATE ab_tests 
                SET status = ?, start_time = ?
                WHERE test_id = ?
                """,
                (ABTestStatus.RUNNING.value, datetime.utcnow().isoformat(), test_id)
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    def record_metric(
        self,
        test_id: str,
        model_version_id: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric for an A/B test."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get current results
            cursor.execute("SELECT results FROM ab_tests WHERE test_id = ?", (test_id,))
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"Test {test_id} not found")
            
            results = json.loads(result[0]) if result[0] else {}
            
            # Initialize structure if needed
            if model_version_id not in results:
                results[model_version_id] = {}
            
            if metric_name not in results[model_version_id]:
                results[model_version_id][metric_name] = []
            
            # Record metric
            metric_record = {
                "value": metric_value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            results[model_version_id][metric_name].append(metric_record)
            
            # Update database
            cursor.execute(
                "UPDATE ab_tests SET results = ? WHERE test_id = ?",
                (json.dumps(results), test_id)
            )
            conn.commit()
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get aggregated results for an A/B test."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_versions, traffic_allocation, results
                FROM ab_tests WHERE test_id = ?
                """,
                (test_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {}
            
            model_versions_json, traffic_json, results_json = result
            model_versions = json.loads(model_versions_json)
            traffic_allocation = json.loads(traffic_json)
            results = json.loads(results_json) if results_json else {}
            
            # Aggregate metrics
            aggregated = {}
            for model_id in model_versions:
                if model_id in results:
                    model_metrics = results[model_id]
                    aggregated[model_id] = {
                        "traffic_allocation": traffic_allocation.get(model_id, 0),
                        "metrics": {}
                    }
                    
                    for metric_name, records in model_metrics.items():
                        values = [r["value"] for r in records]
                        if values:
                            aggregated[model_id]["metrics"][metric_name] = {
                                "mean": sum(values) / len(values),
                                "min": min(values),
                                "max": max(values),
                                "count": len(values),
                                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0
                            }
            
            return aggregated
    
    def declare_winner(self, test_id: str, winning_model_id: str, confidence: float = 0.95) -> bool:
        """Declare a winner for the A/B test."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Update test status
            cursor.execute(
                """
                UPDATE ab_tests 
                SET status = ?, end_time = ?
                WHERE test_id = ?
                """,
                (ABTestStatus.COMPLETED.value, datetime.utcnow().isoformat(), test_id)
            )
            
            # Update results with winner
            cursor.execute("SELECT results FROM ab_tests WHERE test_id = ?", (test_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                results = json.loads(result[0])
                results["winner"] = {
                    "model_id": winning_model_id,
                    "confidence": confidence,
                    "declared_at": datetime.utcnow().isoformat()
                }
                
                cursor.execute(
                    "UPDATE ab_tests SET results = ? WHERE test_id = ?",
                    (json.dumps(results), test_id)
                )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def list_ab_tests(
        self,
        status: Optional[ABTestStatus] = None,
        limit: int = 100
    ) -> List[ABTest]:
        """List A/B tests with filtering."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT test_id, name, description, status, model_versions,
                       traffic_allocation, metrics_to_track, start_time, end_time,
                       created_at, created_by, results
                FROM ab_tests
                WHERE 1=1
            """
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            ab_tests = []
            for row in results:
                (
                    test_id, name, description, status_str, model_versions_json,
                    traffic_json, metrics_json, start_time, end_time,
                    created_at, created_by, results_json
                ) = row
                
                ab_test = ABTest(
                    test_id=test_id,
                    name=name,
                    description=description,
                    status=ABTestStatus(status_str),
                    model_versions=json.loads(model_versions_json),
                    traffic_allocation=json.loads(traffic_json),
                    metrics_to_track=json.loads(metrics_json),
                    start_time=start_time,
                    end_time=end_time,
                    created_at=created_at,
                    created_by=created_by,
                    results=json.loads(results_json) if results_json else {}
                )
                ab_tests.append(ab_test)
            
            return ab_tests


class ModelMarketplace:
    """Model marketplace for sharing and discovering models."""
    
    def __init__(self, lineage_tracker: LineageTracker):
        self.lineage_tracker = lineage_tracker
    
    def list_model(
        self,
        model_id: str,
        name: str,
        description: str,
        author: str,
        license_type: str = "Apache-2.0",
        price: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """List a model in the marketplace."""
        
        # Verify model exists
        model = self.lineage_tracker.get_model_version(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        listing_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_marketplace 
                (listing_id, model_id, version, name, description, author,
                 license, price, downloads, rating, tags, metadata,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    listing_id,
                    model_id,
                    model.version,
                    name,
                    description,
                    author,
                    license_type,
                    price,
                    0,  # downloads
                    0.0,  # rating
                    json.dumps(tags or []),
                    json.dumps(metadata or {}),
                    now,
                    now
                )
            )
            conn.commit()
        
        return listing_id
    
    def search_models(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None,
        sort_by: str = "downloads",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search models in the marketplace."""
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            sql_query = """
                SELECT listing_id, model_id, version, name, description, author,
                       license, price, downloads, rating, tags, metadata,
                       created_at, updated_at
                FROM model_marketplace
                WHERE 1=1
            """
            params = []
            
            if query:
                sql_query += " AND (name LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if author:
                sql_query += " AND author = ?"
                params.append(author)
            
            if min_rating is not None:
                sql_query += " AND rating >= ?"
                params.append(min_rating)
            
            if max_price is not None:
                sql_query += " AND price <= ?"
                params.append(max_price)
            
            # Sorting
            sort_options = {
                "downloads": "downloads DESC",
                "rating": "rating DESC",
                "newest": "created_at DESC",
                "price_low": "price ASC",
                "price_high": "price DESC"
            }
            
            order_by = sort_options.get(sort_by, "downloads DESC")
            sql_query += f" ORDER BY {order_by} LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            models = []
            for row in results:
                (
                    listing_id, model_id, version, name, description, author,
                    license_type, price, downloads, rating, tags_json,
                    metadata_json, created_at, updated_at
                ) = row
                
                model_data = {
                    "listing_id": listing_id,
                    "model_id": model_id,
                    "version": version,
                    "name": name,
                    "description": description,
                    "author": author,
                    "license": license_type,
                    "price": price,
                    "downloads": downloads,
                    "rating": rating,
                    "tags": json.loads(tags_json),
                    "metadata": json.loads(metadata_json),
                    "created_at": created_at,
                    "updated_at": updated_at
                }
                
                # Add model lineage info
                model = self.lineage_tracker.get_model_version(model_id)
                if model:
                    model_data["lineage_summary"] = {
                        "dataset_count": len(model.dataset_hashes),
                        "has_config": bool(model.config_hash),
                        "metrics": model.metrics
                    }
                
                models.append(model_data)
            
            return models
    
    def download_model(self, listing_id: str) -> Optional[bytes]:
        """Download a model from the marketplace."""
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get model ID from listing
            cursor.execute(
                "SELECT model_id FROM model_marketplace WHERE listing_id = ?",
                (listing_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            model_id = result[0]
            
            # Increment download count
            cursor.execute(
                """
                UPDATE model_marketplace 
                SET downloads = downloads + 1, updated_at = ?
                WHERE listing_id = ?
                """,
                (datetime.utcnow().isoformat(), listing_id)
            )
            conn.commit()
            
            # Get model artifact
            model = self.lineage_tracker.get_model_version(model_id)
            if model:
                return self.lineage_tracker.storage.get_artifact(model.model_hash, "model")
            
            return None
    
    def rate_model(self, listing_id: str, rating: float, user_id: Optional[str] = None) -> bool:
        """Rate a model in the marketplace."""
        
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get current rating info
            cursor.execute(
                "SELECT rating, downloads FROM model_marketplace WHERE listing_id = ?",
                (listing_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
            
            current_rating, downloads = result
            
            # Calculate new average rating
            # Simple average for now - could be weighted by recency
            new_rating = (current_rating * downloads + rating) / (downloads + 1) if downloads > 0 else rating
            
            cursor.execute(
                """
                UPDATE model_marketplace 
                SET rating = ?, updated_at = ?
                WHERE listing_id = ?
                """,
                (new_rating, datetime.utcnow().isoformat(), listing_id)
            )
            conn.commit()
            
            return cursor.rowcount > 0


# Main registry class that combines all components
class ModelRegistry:
    """Main model registry with versioning, lineage tracking, and marketplace."""
    
    def __init__(self, storage_path: str = ".forge/registry"):
        self.storage = ContentAddressableStorage(storage_path)
        self.lineage_tracker = LineageTracker(self.storage)
        self.ab_test_manager = ABTestManager(self.lineage_tracker)
        self.marketplace = ModelMarketplace(self.lineage_tracker)
    
    def create_model_from_recipe(
        self,
        recipe_id: str,
        name: str,
        description: str,
        model_data: bytes,
        config_data: bytes,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> ModelVersion:
        """Create a model version from a data recipe."""
        
        # Get dataset hashes from recipe
        # This would integrate with the existing data_recipe module
        dataset_hashes = []  # Would be populated from recipe
        
        return self.lineage_tracker.create_model_version(
            name=name,
            description=description,
            model_data=model_data,
            config_data=config_data,
            dataset_hashes=dataset_hashes,
            recipe_id=recipe_id,
            tags=tags,
            metadata=metadata,
            created_by=created_by
        )
    
    def deploy_model(self, model_id: str, deployment_config: Optional[Dict[str, Any]] = None) -> bool:
        """Deploy a model version."""
        
        # Update status to deployed
        success = self.lineage_tracker.update_model_status(model_id, ModelStatus.DEPLOYED)
        
        if success and deployment_config:
            # Store deployment metadata
            model = self.lineage_tracker.get_model_version(model_id)
            if model:
                model.metadata["deployment"] = deployment_config
                self.lineage_tracker._store_model_version(model)
        
        return success
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple model versions."""
        
        models = []
        for model_id in model_ids:
            model = self.lineage_tracker.get_model_version(model_id)
            if model:
                models.append(model)
        
        if not models:
            return {}
        
        # Compare metrics
        comparison = {
            "models": [],
            "metrics_comparison": {},
            "lineage_comparison": {}
        }
        
        for model in models:
            model_info = {
                "model_id": model.model_id,
                "name": model.name,
                "version": model.version,
                "status": model.status.value,
                "created_at": model.created_at,
                "metrics": asdict(model.metrics) if model.metrics else {},
                "dataset_count": len(model.dataset_hashes),
                "tags": model.tags
            }
            comparison["models"].append(model_info)
        
        # Compare common metrics
        all_metrics = set()
        for model in models:
            if model.metrics:
                all_metrics.update(asdict(model.metrics).keys())
        
        for metric in all_metrics:
            if metric in ["timestamp", "evaluation_dataset_hash", "custom_metrics"]:
                continue
            
            values = []
            for model in models:
                if model.metrics:
                    metric_value = getattr(model.metrics, metric, None)
                    if metric_value is not None:
                        values.append({
                            "model_id": model.model_id,
                            "value": metric_value
                        })
            
            if values:
                comparison["metrics_comparison"][metric] = {
                    "values": values,
                    "best": max(values, key=lambda x: x["value"]) if metric != "loss" else min(values, key=lambda x: x["value"])
                }
        
        return comparison


# Integration with existing modules
def integrate_with_data_recipe(registry: ModelRegistry, job_manager: JobManager):
    """Integrate model registry with data recipe system."""
    
    # Hook into job completion to create model versions
    original_complete_job = job_manager.complete_job
    
    def complete_job_with_registry(job_id: str, result_data: Dict[str, Any]):
        # Call original method
        original_complete_job(job_id, result_data)
        
        # Check if job produced a model
        if "model_artifact" in result_data and "config_artifact" in result_data:
            # Create model version
            registry.create_model_from_recipe(
                recipe_id=result_data.get("recipe_id"),
                name=result_data.get("model_name", f"Model-{job_id}"),
                description=result_data.get("description", "Auto-created from data recipe"),
                model_data=result_data["model_artifact"],
                config_data=result_data["config_artifact"],
                tags=result_data.get("tags", ["auto-created"]),
                metadata={"job_id": job_id}
            )
    
    job_manager.complete_job = complete_job_with_registry


# CLI integration would go here
def setup_cli_commands():
    """Setup CLI commands for model registry."""
    # This would integrate with cli.py
    pass


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # Example: Create a model version
    model_data = b"dummy model data"
    config_data = b'{"learning_rate": 0.001}'
    
    model_version = registry.lineage_tracker.create_model_version(
        name="test-model",
        description="A test model",
        model_data=model_data,
        config_data=config_data,
        dataset_hashes=["abc123", "def456"],
        tags=["test", "example"],
        created_by="test-user"
    )
    
    print(f"Created model: {model_version.model_id}")
    
    # Example: Create A/B test
    ab_test = registry.ab_test_manager.create_ab_test(
        name="Model Comparison Test",
        description="Comparing v1 vs v2",
        model_version_ids=[model_version.model_id],
        traffic_allocation={model_version.model_id: 1.0},
        metrics_to_track=["accuracy", "latency"]
    )
    
    print(f"Created A/B test: {ab_test.test_id}")