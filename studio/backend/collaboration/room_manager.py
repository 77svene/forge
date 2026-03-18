"""Real-time Collaboration System for SOVEREIGN Studio.

Implements WebSocket-based collaboration with CRDTs for conflict-free model editing,
live training metrics streaming, and permissioned team workspaces with RBAC.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs

from studio.backend.auth.authentication import verify_token, get_user_permissions
from studio.backend.auth.storage import get_user_by_id
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class CollaborationError(Exception):
    """Base exception for collaboration system errors."""
    pass


class PermissionError(CollaborationError):
    """Raised when user lacks required permissions."""
    pass


class RoomNotFoundError(CollaborationError):
    """Raised when requested room doesn't exist."""
    pass


class CRDTConflictError(CollaborationError):
    """Raised when CRDT merge conflicts occur."""
    pass


class UserRole(Enum):
    """User roles within collaboration rooms."""
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"


class MessageType(Enum):
    """WebSocket message types."""
    # Connection management
    AUTH = "auth"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    HEARTBEAT = "heartbeat"
    
    # Model collaboration
    MODEL_UPDATE = "model_update"
    MODEL_SYNC = "model_sync"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    
    # Training metrics
    METRICS_UPDATE = "metrics_update"
    METRICS_SUBSCRIBE = "metrics_subscribe"
    METRICS_UNSUBSCRIBE = "metrics_unsubscribe"
    
    # Workspace management
    WORKSPACE_CREATE = "workspace_create"
    WORKSPACE_UPDATE = "workspace_update"
    WORKSPACE_DELETE = "workspace_delete"
    PERMISSION_UPDATE = "permission_update"
    
    # System messages
    ERROR = "error"
    NOTIFICATION = "notification"
    STATE_SYNC = "state_sync"


@dataclass
class VectorClock:
    """Vector clock for CRDT conflict resolution."""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str) -> None:
        """Increment clock for given node."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def merge(self, other: 'VectorClock') -> None:
        """Merge with another vector clock (take maximum of each)."""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)
    
    def compare(self, other: 'VectorClock') -> int:
        """Compare vector clocks. Returns: -1 if self < other, 0 if concurrent, 1 if self > other."""
        less = greater = False
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        for node in all_nodes:
            s = self.clocks.get(node, 0)
            o = other.clocks.get(node, 0)
            if s < o:
                less = True
            elif s > o:
                greater = True
        
        if less and not greater:
            return -1
        elif greater and not less:
            return 1
        return 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return self.clocks.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        """Create from dictionary."""
        return cls(clocks=data)


@dataclass
class CRDTRegister:
    """Last-Writer-Wins Register CRDT for model parameters."""
    value: Any = None
    timestamp: float = 0.0
    vector_clock: VectorClock = field(default_factory=VectorClock)
    node_id: str = ""
    
    def update(self, new_value: Any, node_id: str) -> None:
        """Update register value with new timestamp."""
        self.value = new_value
        self.timestamp = time.time()
        self.node_id = node_id
        self.vector_clock.increment(node_id)
    
    def merge(self, other: 'CRDTRegister') -> None:
        """Merge with another register using LWW strategy."""
        # Compare vector clocks first
        comparison = self.vector_clock.compare(other.vector_clock)
        
        if comparison == -1:  # other is newer
            self.value = other.value
            self.timestamp = other.timestamp
            self.node_id = other.node_id
        elif comparison == 0:  # concurrent - use timestamp
            if other.timestamp > self.timestamp:
                self.value = other.value
                self.timestamp = other.timestamp
                self.node_id = other.node_id
        
        # Always merge vector clocks
        self.vector_clock.merge(other.vector_clock)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "vector_clock": self.vector_clock.to_dict(),
            "node_id": self.node_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTRegister':
        """Create from dictionary."""
        reg = cls()
        reg.value = data.get("value")
        reg.timestamp = data.get("timestamp", 0.0)
        reg.vector_clock = VectorClock.from_dict(data.get("vector_clock", {}))
        reg.node_id = data.get("node_id", "")
        return reg


@dataclass
class CRDTMap:
    """CRDT Map for collaborative model editing."""
    registers: Dict[str, CRDTRegister] = field(default_factory=dict)
    vector_clock: VectorClock = field(default_factory=VectorClock)
    
    def update(self, key: str, value: Any, node_id: str) -> None:
        """Update a key in the map."""
        if key not in self.registers:
            self.registers[key] = CRDTRegister()
        self.registers[key].update(value, node_id)
        self.vector_clock.increment(node_id)
    
    def merge(self, other: 'CRDTMap') -> None:
        """Merge with another CRDT map."""
        # Merge all registers
        for key, other_reg in other.registers.items():
            if key not in self.registers:
                self.registers[key] = CRDTRegister()
            self.registers[key].merge(other_reg)
        
        # Merge vector clocks
        self.vector_clock.merge(other.vector_clock)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for key."""
        if key in self.registers:
            return self.registers[key].value
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "registers": {k: v.to_dict() for k, v in self.registers.items()},
            "vector_clock": self.vector_clock.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTMap':
        """Create from dictionary."""
        crdt_map = cls()
        crdt_map.registers = {
            k: CRDTRegister.from_dict(v) 
            for k, v in data.get("registers", {}).items()
        }
        crdt_map.vector_clock = VectorClock.from_dict(
            data.get("vector_clock", {})
        )
        return crdt_map


@dataclass
class UserSession:
    """Represents a connected user session."""
    user_id: str
    username: str
    websocket: Any  # WebSocket connection
    room_id: str
    role: UserRole
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    subscribed_metrics: Set[str] = field(default_factory=set)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required permission level."""
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.EDITOR: 1,
            UserRole.ADMIN: 2,
            UserRole.OWNER: 3
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(required_role, 0)


@dataclass
class TrainingMetrics:
    """Real-time training metrics with time-series data."""
    job_id: str
    metrics: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    max_history: int = 1000
    
    def add_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Add a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append((timestamp, value))
        
        # Trim history if needed
        if len(self.metrics[metric_name]) > self.max_history:
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_history:]
        
        self.last_updated = timestamp
    
    def get_metrics(self, metric_names: Optional[List[str]] = None, 
                   since_timestamp: float = 0.0) -> Dict[str, List[Tuple[float, float]]]:
        """Get metrics, optionally filtered by name and time."""
        result = {}
        
        for name, values in self.metrics.items():
            if metric_names and name not in metric_names:
                continue
            
            filtered = [(ts, val) for ts, val in values if ts >= since_timestamp]
            if filtered:
                result[name] = filtered
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "metrics": self.metrics,
            "last_updated": self.last_updated
        }


@dataclass
class CollaborationRoom:
    """Represents a collaboration workspace/room."""
    room_id: str
    name: str
    description: str
    owner_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Collaboration state
    model_state: CRDTMap = field(default_factory=CRDTMap)
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_sessions: Dict[str, UserSession] = field(default_factory=dict)
    permissions: Dict[str, UserRole] = field(default_factory=dict)
    
    # Training metrics
    training_metrics: Dict[str, TrainingMetrics] = field(default_factory=dict)
    metrics_subscriptions: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Configuration
    max_participants: int = 50
    is_public: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def add_session(self, session: UserSession) -> None:
        """Add a user session to the room."""
        self.active_sessions[session.user_id] = session
        self.updated_at = time.time()
    
    def remove_session(self, user_id: str) -> None:
        """Remove a user session from the room."""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            self.updated_at = time.time()
    
    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get user session by ID."""
        return self.active_sessions.get(user_id)
    
    def update_model_state(self, updates: Dict[str, Any], node_id: str) -> None:
        """Update model state using CRDT merge."""
        for key, value in updates.items():
            self.model_state.update(key, value, node_id)
        self.updated_at = time.time()
    
    def merge_model_state(self, other_state: Dict[str, Any]) -> None:
        """Merge external model state."""
        other_crdt = CRDTMap.from_dict(other_state)
        self.model_state.merge(other_crdt)
        self.updated_at = time.time()
    
    def save_checkpoint(self, checkpoint_id: str, checkpoint_data: Dict[str, Any], 
                       user_id: str) -> None:
        """Save a model checkpoint."""
        self.checkpoints[checkpoint_id] = {
            "data": checkpoint_data,
            "saved_by": user_id,
            "saved_at": time.time(),
            "model_state": self.model_state.to_dict()
        }
        self.updated_at = time.time()
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint."""
        if checkpoint_id in self.checkpoints:
            checkpoint = self.checkpoints[checkpoint_id]
            self.model_state = CRDTMap.from_dict(checkpoint["model_state"])
            self.updated_at = time.time()
            return checkpoint["data"]
        return None
    
    def update_metrics(self, job_id: str, metrics: Dict[str, float]) -> None:
        """Update training metrics for a job."""
        if job_id not in self.training_metrics:
            self.training_metrics[job_id] = TrainingMetrics(job_id=job_id)
        
        timestamp = time.time()
        for name, value in metrics.items():
            self.training_metrics[job_id].add_metric(name, value, timestamp)
        
        # Notify subscribed users
        self._notify_metrics_subscribers(job_id, metrics, timestamp)
    
    def subscribe_to_metrics(self, user_id: str, job_id: str) -> None:
        """Subscribe user to metrics updates for a job."""
        self.metrics_subscriptions[job_id].add(user_id)
    
    def unsubscribe_from_metrics(self, user_id: str, job_id: str) -> None:
        """Unsubscribe user from metrics updates."""
        if job_id in self.metrics_subscriptions:
            self.metrics_subscriptions[job_id].discard(user_id)
    
    def _notify_metrics_subscribers(self, job_id: str, metrics: Dict[str, float], 
                                   timestamp: float) -> None:
        """Notify all subscribers about metrics update."""
        if job_id not in self.metrics_subscriptions:
            return
        
        message = {
            "type": MessageType.METRICS_UPDATE.value,
            "job_id": job_id,
            "metrics": metrics,
            "timestamp": timestamp
        }
        
        # Note: Actual sending is handled by RoomManager
        # This just prepares the message
        self.updated_at = time.time()
    
    def update_permissions(self, user_id: str, role: UserRole, 
                          updater_id: str) -> bool:
        """Update user permissions. Returns True if successful."""
        updater_session = self.get_session(updater_id)
        if not updater_session:
            return False
        
        # Check if updater has permission
        if not updater_session.has_permission(UserRole.ADMIN):
            return False
        
        # Cannot change owner's role
        if user_id == self.owner_id and role != UserRole.OWNER:
            return False
        
        self.permissions[user_id] = role
        
        # Update active session if user is connected
        session = self.get_session(user_id)
        if session:
            session.role = role
        
        self.updated_at = time.time()
        return True
    
    def get_user_role(self, user_id: str) -> UserRole:
        """Get user's role in this room."""
        if user_id == self.owner_id:
            return UserRole.OWNER
        return self.permissions.get(user_id, UserRole.VIEWER)
    
    def to_dict(self, include_state: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "room_id": self.room_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "participant_count": len(self.active_sessions),
            "max_participants": self.max_participants,
            "is_public": self.is_public,
            "settings": self.settings,
            "checkpoints": list(self.checkpoints.keys()),
            "active_jobs": list(self.training_metrics.keys())
        }
        
        if include_state:
            data["model_state"] = self.model_state.to_dict()
            data["permissions"] = {k: v.value for k, v in self.permissions.items()}
        
        return data


class RoomManager:
    """Manages collaboration rooms and WebSocket connections."""
    
    def __init__(self, job_manager: Optional[JobManager] = None):
        self.rooms: Dict[str, CollaborationRoom] = {}
        self.user_sessions: Dict[str, UserSession] = {}  # websocket -> session
        self.user_rooms: Dict[str, Set[str]] = defaultdict(set)  # user_id -> room_ids
        self.room_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.job_manager = job_manager
        
        # Start background tasks
        self._cleanup_task = None
        self._metrics_task = None
        self._running = False
        
        logger.info("RoomManager initialized")
    
    async def start(self) -> None:
        """Start background tasks."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        self._metrics_task = asyncio.create_task(self._stream_training_metrics())
        logger.info("RoomManager background tasks started")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Close all WebSocket connections
        for session in list(self.user_sessions.values()):
            try:
                await session.websocket.close()
            except Exception:
                pass
        
        logger.info("RoomManager stopped")
    
    async def create_room(self, name: str, description: str, owner_id: str,
                         settings: Optional[Dict[str, Any]] = None) -> CollaborationRoom:
        """Create a new collaboration room."""
        room_id = str(uuid.uuid4())
        
        room = CollaborationRoom(
            room_id=room_id,
            name=name,
            description=description,
            owner_id=owner_id,
            settings=settings or {}
        )
        
        # Set owner permissions
        room.permissions[owner_id] = UserRole.OWNER
        
        self.rooms[room_id] = room
        logger.info(f"Created room {room_id}: {name}")
        
        return room
    
    async def delete_room(self, room_id: str, user_id: str) -> bool:
        """Delete a collaboration room."""
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return False
            
            # Check permissions
            if room.owner_id != user_id:
                user_role = room.get_user_role(user_id)
                if user_role != UserRole.OWNER:
                    return False
            
            # Notify all participants
            message = {
                "type": MessageType.NOTIFICATION.value,
                "message": f"Room '{room.name}' has been deleted",
                "room_id": room_id
            }
            
            for session in room.active_sessions.values():
                try:
                    await self._send_message(session.websocket, message)
                    await session.websocket.close()
                except Exception:
                    pass
            
            # Clean up
            for user_id in list(room.active_sessions.keys()):
                self.user_rooms[user_id].discard(room_id)
            
            del self.rooms[room_id]
            logger.info(f"Deleted room {room_id}")
            
            return True
    
    async def handle_websocket(self, websocket: Any, path: str) -> None:
        """Handle incoming WebSocket connection."""
        session = None
        try:
            # Parse query parameters for authentication
            query_params = parse_qs(path.split('?')[1] if '?' in path else '')
            token = query_params.get('token', [None])[0]
            
            if not token:
                await self._send_error(websocket, "Authentication token required")
                await websocket.close()
                return
            
            # Verify token and get user
            user_data = await self._authenticate_user(token)
            if not user_data:
                await self._send_error(websocket, "Invalid authentication token")
                await websocket.close()
                return
            
            # Create session
            session = UserSession(
                user_id=user_data["id"],
                username=user_data.get("username", "Unknown"),
                websocket=websocket,
                room_id="",  # Will be set when joining a room
                role=UserRole.VIEWER
            )
            
            self.user_sessions[websocket] = session
            logger.info(f"User {session.username} connected")
            
            # Send authentication success
            await self._send_message(websocket, {
                "type": MessageType.AUTH.value,
                "status": "success",
                "user_id": session.user_id,
                "username": session.username,
                "node_id": session.node_id
            })
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(session, message)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if session:
                await self._handle_disconnect(session)
        finally:
            if session:
                await self._handle_disconnect(session)
    
    async def _authenticate_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user from token."""
        try:
            # Verify token using existing auth system
            user_id = verify_token(token)
            if not user_id:
                return None
            
            # Get user details
            user = get_user_by_id(user_id)
            if not user:
                return None
            
            return {
                "id": user_id,
                "username": user.get("username", "Unknown"),
                "email": user.get("email"),
                "permissions": get_user_permissions(user_id)
            }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def _handle_message(self, session: UserSession, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(raw_message)
            message_type = message.get("type")
            
            if not message_type:
                await self._send_error(session.websocket, "Message type required")
                return
            
            session.update_activity()
            
            # Route message based on type
            handlers = {
                MessageType.JOIN_ROOM.value: self._handle_join_room,
                MessageType.LEAVE_ROOM.value: self._handle_leave_room,
                MessageType.HEARTBEAT.value: self._handle_heartbeat,
                MessageType.MODEL_UPDATE.value: self._handle_model_update,
                MessageType.MODEL_SYNC.value: self._handle_model_sync,
                MessageType.CHECKPOINT_SAVE.value: self._handle_checkpoint_save,
                MessageType.CHECKPOINT_LOAD.value: self._handle_checkpoint_load,
                MessageType.METRICS_SUBSCRIBE.value: self._handle_metrics_subscribe,
                MessageType.METRICS_UNSUBSCRIBE.value: self._handle_metrics_unsubscribe,
                MessageType.WORKSPACE_CREATE.value: self._handle_workspace_create,
                MessageType.WORKSPACE_UPDATE.value: self._handle_workspace_update,
                MessageType.PERMISSION_UPDATE.value: self._handle_permission_update,
            }
            
            handler = handlers.get(message_type)
            if handler:
                await handler(session, message)
            else:
                await self._send_error(session.websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(session.websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await self._send_error(session.websocket, f"Internal server error: {str(e)}")
    
    async def _handle_join_room(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle join room request."""
        room_id = message.get("room_id")
        requested_role = message.get("role", "viewer")
        
        if not room_id:
            await self._send_error(session.websocket, "Room ID required")
            return
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                await self._send_error(session.websocket, "Room not found")
                return
            
            # Check if room is full
            if len(room.active_sessions) >= room.max_participants:
                await self._send_error(session.websocket, "Room is full")
                return
            
            # Check permissions
            user_role = room.get_user_role(session.user_id)
            if user_role == UserRole.VIEWER and requested_role != "viewer":
                # Check if user has permission to request higher role
                if not room.is_public:
                    await self._send_error(session.websocket, "Insufficient permissions")
                    return
            
            # Set role
            try:
                role = UserRole(requested_role)
            except ValueError:
                role = UserRole.VIEWER
            
            # Update session
            session.room_id = room_id
            session.role = role
            
            # Add to room
            room.add_session(session)
            self.user_rooms[session.user_id].add(room_id)
            
            # Send room state
            room_state = room.to_dict(include_state=True)
            await self._send_message(session.websocket, {
                "type": MessageType.STATE_SYNC.value,
                "room": room_state,
                "your_role": role.value,
                "your_node_id": session.node_id
            })
            
            # Notify other participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.NOTIFICATION.value,
                "message": f"{session.username} joined the room",
                "user_id": session.user_id,
                "username": session.username,
                "role": role.value
            }, exclude_user_id=session.user_id)
            
            logger.info(f"User {session.username} joined room {room_id} as {role.value}")
    
    async def _handle_leave_room(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle leave room request."""
        room_id = session.room_id
        if not room_id:
            return
        
        await self._remove_user_from_room(session, room_id)
    
    async def _handle_heartbeat(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle heartbeat message."""
        await self._send_message(session.websocket, {
            "type": MessageType.HEARTBEAT.value,
            "timestamp": time.time()
        })
    
    async def _handle_model_update(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle model update using CRDTs."""
        room_id = session.room_id
        if not room_id:
            await self._send_error(session.websocket, "Not in a room")
            return
        
        # Check permissions
        if not session.has_permission(UserRole.EDITOR):
            await self._send_error(session.websocket, "Insufficient permissions for model editing")
            return
        
        updates = message.get("updates", {})
        if not updates:
            return
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return
            
            # Update model state with CRDT merge
            room.update_model_state(updates, session.node_id)
            
            # Broadcast update to other participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.MODEL_UPDATE.value,
                "updates": updates,
                "user_id": session.user_id,
                "node_id": session.node_id,
                "timestamp": time.time()
            }, exclude_user_id=session.user_id)
    
    async def _handle_model_sync(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle model state synchronization request."""
        room_id = session.room_id
        if not room_id:
            return
        
        room = self.rooms.get(room_id)
        if not room:
            return
        
        # Send current model state
        await self._send_message(session.websocket, {
            "type": MessageType.MODEL_SYNC.value,
            "model_state": room.model_state.to_dict(),
            "timestamp": time.time()
        })
    
    async def _handle_checkpoint_save(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle checkpoint save request."""
        room_id = session.room_id
        if not room_id:
            await self._send_error(session.websocket, "Not in a room")
            return
        
        # Check permissions
        if not session.has_permission(UserRole.EDITOR):
            await self._send_error(session.websocket, "Insufficient permissions")
            return
        
        checkpoint_name = message.get("name", f"checkpoint_{int(time.time())}")
        checkpoint_data = message.get("data", {})
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return
            
            checkpoint_id = str(uuid.uuid4())
            room.save_checkpoint(checkpoint_id, checkpoint_data, session.user_id)
            
            # Notify participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.NOTIFICATION.value,
                "message": f"Checkpoint '{checkpoint_name}' saved by {session.username}",
                "checkpoint_id": checkpoint_id,
                "checkpoint_name": checkpoint_name,
                "user_id": session.user_id
            })
            
            logger.info(f"Checkpoint {checkpoint_id} saved in room {room_id}")
    
    async def _handle_checkpoint_load(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle checkpoint load request."""
        room_id = session.room_id
        if not room_id:
            await self._send_error(session.websocket, "Not in a room")
            return
        
        # Check permissions
        if not session.has_permission(UserRole.EDITOR):
            await self._send_error(session.websocket, "Insufficient permissions")
            return
        
        checkpoint_id = message.get("checkpoint_id")
        if not checkpoint_id:
            await self._send_error(session.websocket, "Checkpoint ID required")
            return
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return
            
            checkpoint_data = room.load_checkpoint(checkpoint_id)
            if not checkpoint_data:
                await self._send_error(session.websocket, "Checkpoint not found")
                return
            
            # Send checkpoint data to requesting user
            await self._send_message(session.websocket, {
                "type": MessageType.CHECKPOINT_LOAD.value,
                "checkpoint_id": checkpoint_id,
                "data": checkpoint_data,
                "model_state": room.model_state.to_dict()
            })
            
            # Notify other participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.NOTIFICATION.value,
                "message": f"{session.username} loaded checkpoint {checkpoint_id}",
                "checkpoint_id": checkpoint_id,
                "user_id": session.user_id
            }, exclude_user_id=session.user_id)
    
    async def _handle_metrics_subscribe(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle metrics subscription request."""
        room_id = session.room_id
        if not room_id:
            return
        
        job_id = message.get("job_id")
        if not job_id:
            return
        
        room = self.rooms.get(room_id)
        if not room:
            return
        
        room.subscribe_to_metrics(session.user_id, job_id)
        session.subscribed_metrics.add(job_id)
        
        # Send existing metrics
        if job_id in room.training_metrics:
            metrics_data = room.training_metrics[job_id].to_dict()
            await self._send_message(session.websocket, {
                "type": MessageType.METRICS_UPDATE.value,
                "job_id": job_id,
                "metrics": metrics_data["metrics"],
                "initial": True
            })
    
    async def _handle_metrics_unsubscribe(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle metrics unsubscription request."""
        room_id = session.room_id
        if not room_id:
            return
        
        job_id = message.get("job_id")
        if not job_id:
            return
        
        room = self.rooms.get(room_id)
        if not room:
            return
        
        room.unsubscribe_from_metrics(session.user_id, job_id)
        session.subscribed_metrics.discard(job_id)
    
    async def _handle_workspace_create(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle workspace creation request."""
        name = message.get("name")
        description = message.get("description", "")
        settings = message.get("settings", {})
        
        if not name:
            await self._send_error(session.websocket, "Workspace name required")
            return
        
        # Create room
        room = await self.create_room(name, description, session.user_id, settings)
        
        # Auto-join the creator
        join_message = {
            "type": MessageType.JOIN_ROOM.value,
            "room_id": room.room_id,
            "role": "owner"
        }
        await self._handle_join_room(session, join_message)
        
        # Send confirmation
        await self._send_message(session.websocket, {
            "type": MessageType.WORKSPACE_CREATE.value,
            "status": "success",
            "room": room.to_dict()
        })
    
    async def _handle_workspace_update(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle workspace update request."""
        room_id = message.get("room_id")
        updates = message.get("updates", {})
        
        if not room_id:
            await self._send_error(session.websocket, "Room ID required")
            return
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                await self._send_error(session.websocket, "Room not found")
                return
            
            # Check permissions
            if not session.has_permission(UserRole.ADMIN):
                if room.owner_id != session.user_id:
                    await self._send_error(session.websocket, "Insufficient permissions")
                    return
            
            # Update allowed fields
            allowed_fields = ["name", "description", "max_participants", "is_public", "settings"]
            for field, value in updates.items():
                if field in allowed_fields and hasattr(room, field):
                    setattr(room, field, value)
            
            room.updated_at = time.time()
            
            # Notify participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.NOTIFICATION.value,
                "message": f"Room settings updated by {session.username}",
                "updates": updates
            })
    
    async def _handle_permission_update(self, session: UserSession, message: Dict[str, Any]) -> None:
        """Handle permission update request."""
        room_id = session.room_id
        if not room_id:
            await self._send_error(session.websocket, "Not in a room")
            return
        
        target_user_id = message.get("user_id")
        new_role = message.get("role")
        
        if not target_user_id or not new_role:
            await self._send_error(session.websocket, "User ID and role required")
            return
        
        try:
            role = UserRole(new_role)
        except ValueError:
            await self._send_error(session.websocket, f"Invalid role: {new_role}")
            return
        
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return
            
            # Update permissions
            success = room.update_permissions(target_user_id, role, session.user_id)
            
            if success:
                # Notify affected user and room
                target_session = room.get_session(target_user_id)
                if target_session:
                    await self._send_message(target_session.websocket, {
                        "type": MessageType.PERMISSION_UPDATE.value,
                        "new_role": role.value,
                        "updated_by": session.user_id
                    })
                
                await self._broadcast_to_room(room_id, {
                    "type": MessageType.NOTIFICATION.value,
                    "message": f"User permissions updated by {session.username}",
                    "user_id": target_user_id,
                    "new_role": role.value
                }, exclude_user_id=target_user_id)
            else:
                await self._send_error(session.websocket, "Failed to update permissions")
    
    async def _remove_user_from_room(self, session: UserSession, room_id: str) -> None:
        """Remove user from room."""
        async with self.room_locks[room_id]:
            room = self.rooms.get(room_id)
            if not room:
                return
            
            # Remove from metrics subscriptions
            for job_id in list(room.metrics_subscriptions.keys()):
                room.metrics_subscriptions[job_id].discard(session.user_id)
            
            # Remove session
            room.remove_session(session.user_id)
            self.user_rooms[session.user_id].discard(room_id)
            
            # Notify other participants
            await self._broadcast_to_room(room_id, {
                "type": MessageType.NOTIFICATION.value,
                "message": f"{session.username} left the room",
                "user_id": session.user_id,
                "username": session.username
            })
            
            session.room_id = ""
            logger.info(f"User {session.username} left room {room_id}")
    
    async def _handle_disconnect(self, session: UserSession) -> None:
        """Handle WebSocket disconnection."""
        # Remove from all rooms
        for room_id in list(self.user_rooms.get(session.user_id, set())):
            await self._remove_user_from_room(session, room_id)
        
        # Remove session
        if session.websocket in self.user_sessions:
            del self.user_sessions[session.websocket]
        
        logger.info(f"User {session.username} disconnected")
    
    async def _broadcast_to_room(self, room_id: str, message: Dict[str, Any],
                                exclude_user_id: Optional[str] = None) -> None:
        """Broadcast message to all users in a room."""
        room = self.rooms.get(room_id)
        if not room:
            return
        
        for user_id, session in room.active_sessions.items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            
            try:
                await self._send_message(session.websocket, message)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
    
    async def _send_message(self, websocket: Any, message: Dict[str, Any]) -> None:
        """Send JSON message to WebSocket."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            raise
    
    async def _send_error(self, websocket: Any, error_message: str) -> None:
        """Send error message to WebSocket."""
        await self._send_message(websocket, {
            "type": MessageType.ERROR.value,
            "error": error_message,
            "timestamp": time.time()
        })
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to clean up inactive sessions."""
        while self._running:
            try:
                current_time = time.time()
                inactive_threshold = 300  # 5 minutes
                
                for websocket, session in list(self.user_sessions.items()):
                    if current_time - session.last_activity > inactive_threshold:
                        logger.info(f"Cleaning up inactive session for {session.username}")
                        await self._handle_disconnect(session)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(10)
    
    async def _stream_training_metrics(self) -> None:
        """Background task to stream training metrics from JobManager."""
        while self._running:
            try:
                if self.job_manager:
                    # Get active jobs and their metrics
                    active_jobs = self.job_manager.get_active_jobs()
                    
                    for job in active_jobs:
                        job_id = job.get("id")
                        if not job_id:
                            continue
                        
                        # Get metrics for this job
                        metrics = self.job_manager.get_job_metrics(job_id)
                        if not metrics:
                            continue
                        
                        # Find rooms monitoring this job
                        for room_id, room in self.rooms.items():
                            if job_id in room.training_metrics:
                                # Update metrics
                                room.update_metrics(job_id, metrics)
                                
                                # Broadcast to subscribers
                                subscribers = room.metrics_subscriptions.get(job_id, set())
                                for user_id in subscribers:
                                    session = room.get_session(user_id)
                                    if session:
                                        try:
                                            await self._send_message(session.websocket, {
                                                "type": MessageType.METRICS_UPDATE.value,
                                                "job_id": job_id,
                                                "metrics": metrics,
                                                "timestamp": time.time()
                                            })
                                        except Exception:
                                            pass
                
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics streaming error: {e}")
                await asyncio.sleep(5)
    
    async def update_job_metrics(self, job_id: str, metrics: Dict[str, float],
                               room_id: Optional[str] = None) -> None:
        """Update metrics for a job (called by external systems)."""
        if room_id:
            # Update specific room
            room = self.rooms.get(room_id)
            if room and job_id in room.training_metrics:
                room.update_metrics(job_id, metrics)
        else:
            # Update all rooms monitoring this job
            for room in self.rooms.values():
                if job_id in room.training_metrics:
                    room.update_metrics(job_id, metrics)
    
    def get_room(self, room_id: str) -> Optional[CollaborationRoom]:
        """Get room by ID."""
        return self.rooms.get(room_id)
    
    def get_user_rooms(self, user_id: str) -> List[CollaborationRoom]:
        """Get all rooms a user is participating in."""
        room_ids = self.user_rooms.get(user_id, set())
        return [self.rooms[rid] for rid in room_ids if rid in self.rooms]
    
    def get_public_rooms(self) -> List[CollaborationRoom]:
        """Get all public rooms."""
        return [room for room in self.rooms.values() if room.is_public]
    
    async def get_room_participants(self, room_id: str) -> List[Dict[str, Any]]:
        """Get list of participants in a room."""
        room = self.rooms.get(room_id)
        if not room:
            return []
        
        participants = []
        for session in room.active_sessions.values():
            participants.append({
                "user_id": session.user_id,
                "username": session.username,
                "role": session.role.value,
                "connected_at": session.connected_at,
                "last_activity": session.last_activity
            })
        
        return participants


# Global instance for easy import
room_manager = RoomManager()


# Integration with existing job system
def setup_job_integration(job_manager: JobManager) -> None:
    """Set up integration with JobManager for metrics streaming."""
    room_manager.job_manager = job_manager
    logger.info("JobManager integration configured")


# WebSocket server factory for integration with web frameworks
async def websocket_handler(websocket: Any, path: str) -> None:
    """WebSocket handler for use with websockets library."""
    await room_manager.handle_websocket(websocket, path)


# Example usage with FastAPI/Starlette
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from studio.backend.collaboration.room_manager import room_manager

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await room_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    await room_manager.stop()

@app.websocket("/ws/collaboration")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await room_manager.handle_websocket(websocket, websocket.url.path)
    except WebSocketDisconnect:
        pass
"""