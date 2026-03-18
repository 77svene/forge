"""
Real-time Collaboration System for Unsloth Studio
WebSocket-based collaboration with CRDTs, live metrics, and team workspaces
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import existing authentication and storage modules
from .auth.authentication import get_current_user, User, verify_token
from .auth.storage import StorageManager
from .core.data_recipe.jobs.manager import JobManager


class CollaborationError(Exception):
    """Base exception for collaboration system errors"""
    pass


class PermissionError(CollaborationError):
    """Raised when user lacks required permissions"""
    pass


class CRDTConflictError(CollaborationError):
    """Raised when CRDT conflict resolution fails"""
    pass


class WorkspaceRole(str, Enum):
    """Role-based access control for team workspaces"""
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"


class MessageType(str, Enum):
    """WebSocket message types for collaboration"""
    JOIN_WORKSPACE = "join_workspace"
    LEAVE_WORKSPACE = "leave_workspace"
    MODEL_UPDATE = "model_update"
    CHECKPOINT_SYNC = "checkpoint_sync"
    METRICS_UPDATE = "metrics_update"
    CURSOR_MOVE = "cursor_move"
    CHAT_MESSAGE = "chat_message"
    PERMISSION_UPDATE = "permission_update"
    ERROR = "error"
    ACK = "ack"
    HEARTBEAT = "heartbeat"


@dataclass
class CRDTOperation:
    """Conflict-free Replicated Data Type operation for model editing"""
    operation_id: str
    operation_type: str  # "insert", "delete", "update"
    position: int
    value: Any
    vector_clock: Dict[str, int]
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    workspace_id: str = ""


@dataclass
class ModelCheckpoint:
    """Shared model checkpoint with CRDT metadata"""
    checkpoint_id: str
    model_state: Dict[str, Any]
    crdt_state: Dict[str, Any]
    version_vector: Dict[str, int]
    created_at: datetime
    updated_at: datetime
    created_by: str
    workspace_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveMetrics:
    """Real-time training metrics"""
    metrics_id: str
    job_id: str
    workspace_id: str
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float]
    learning_rate: float
    memory_usage: float
    gpu_utilization: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TeamWorkspace:
    """Team workspace with permissions and shared state"""
    workspace_id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    max_members: int = 10
    checkpoints: Dict[str, ModelCheckpoint] = field(default_factory=dict)
    active_users: Set[str] = field(default_factory=set)
    permissions: Dict[str, WorkspaceRole] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebSocketConnection:
    """WebSocket connection wrapper with user context"""
    websocket: WebSocket
    user_id: str
    workspace_id: str
    connection_id: str
    connected_at: datetime
    last_heartbeat: datetime
    cursor_position: Optional[Dict[str, Any]] = None
    is_authenticated: bool = False


class CRDTManager:
    """
    Conflict-free Replicated Data Type manager for model editing
    Implements operational transformation for concurrent edits
    """
    
    def __init__(self):
        self.operations_log: Dict[str, List[CRDTOperation]] = defaultdict(list)
        self.vector_clocks: Dict[str, Dict[str, int]] = defaultdict(dict)
    
    def apply_operation(self, workspace_id: str, operation: CRDTOperation) -> Dict[str, Any]:
        """Apply CRDT operation and return transformed state"""
        if workspace_id not in self.operations_log:
            self.operations_log[workspace_id] = []
            self.vector_clocks[workspace_id] = {}
        
        # Update vector clock
        user_clock = self.vector_clocks[workspace_id].get(operation.user_id, 0)
        self.vector_clocks[workspace_id][operation.user_id] = max(
            user_clock, operation.vector_clock.get(operation.user_id, 0)
        ) + 1
        
        # Apply operation transformation if needed
        transformed_op = self._transform_operation(workspace_id, operation)
        
        # Log operation
        self.operations_log[workspace_id].append(transformed_op)
        
        # Return current state
        return self.get_current_state(workspace_id)
    
    def _transform_operation(self, workspace_id: str, operation: CRDTOperation) -> CRDTOperation:
        """Transform operation against concurrent operations"""
        # Simple implementation - in production, use proper OT algorithms
        concurrent_ops = [
            op for op in self.operations_log[workspace_id]
            if op.timestamp > operation.timestamp - 5.0  # Last 5 seconds
        ]
        
        # Adjust position based on concurrent insertions/deletions
        position_offset = 0
        for cop in concurrent_ops:
            if cop.position <= operation.position:
                if cop.operation_type == "insert":
                    position_offset += 1
                elif cop.operation_type == "delete":
                    position_offset -= 1
        
        operation.position = max(0, operation.position + position_offset)
        return operation
    
    def get_current_state(self, workspace_id: str) -> Dict[str, Any]:
        """Reconstruct current state from CRDT operations"""
        if workspace_id not in self.operations_log:
            return {}
        
        # Simple state reconstruction - in production, maintain snapshots
        state = {}
        for operation in sorted(self.operations_log[workspace_id], key=lambda x: x.timestamp):
            if operation.operation_type == "update":
                state[operation.position] = operation.value
            elif operation.operation_type == "insert":
                state[operation.position] = operation.value
            elif operation.operation_type == "delete":
                state.pop(operation.position, None)
        
        return state
    
    def merge_states(self, workspace_id: str, remote_state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge remote state with local CRDT state"""
        # Last-write-wins merge strategy
        local_state = self.get_current_state(workspace_id)
        merged = {**local_state, **remote_state}
        return merged


class WebSocketManager:
    """
    Manages WebSocket connections, rooms, and real-time collaboration
    """
    
    def __init__(self, storage_manager: StorageManager, job_manager: JobManager):
        self.storage = storage_manager
        self.job_manager = job_manager
        self.crdt_manager = CRDTManager()
        
        # Connection management
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.workspace_connections: Dict[str, Set[str]] = defaultdict(set)
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Workspace state
        self.workspaces: Dict[str, TeamWorkspace] = {}
        self.metrics_streams: Dict[str, asyncio.Task] = {}
        
        # Security
        self.security = HTTPBearer()
        
        # Load existing workspaces
        self._load_workspaces()
    
    def _load_workspaces(self):
        """Load existing workspaces from storage"""
        try:
            workspace_data = self.storage.load_data("workspaces.json")
            if workspace_data:
                for ws_id, ws_dict in workspace_data.items():
                    self.workspaces[ws_id] = TeamWorkspace(**ws_dict)
        except Exception as e:
            print(f"Failed to load workspaces: {e}")
    
    def _save_workspaces(self):
        """Save workspaces to storage"""
        try:
            workspace_data = {
                ws_id: asdict(ws) for ws_id, ws in self.workspaces.items()
            }
            self.storage.save_data("workspaces.json", workspace_data)
        except Exception as e:
            print(f"Failed to save workspaces: {e}")
    
    async def authenticate_websocket(
        self, 
        websocket: WebSocket, 
        token: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate WebSocket connection using JWT token"""
        if not token:
            # Try to get token from query params
            token = websocket.query_params.get("token")
        
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
        
        try:
            user = verify_token(token)
            return user
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
    
    def check_permission(
        self, 
        user_id: str, 
        workspace_id: str, 
        required_role: WorkspaceRole
    ) -> bool:
        """Check if user has required permission level"""
        if workspace_id not in self.workspaces:
            return False
        
        workspace = self.workspaces[workspace_id]
        user_role = workspace.permissions.get(user_id)
        
        if not user_role:
            return False
        
        role_hierarchy = {
            WorkspaceRole.VIEWER: 0,
            WorkspaceRole.EDITOR: 1,
            WorkspaceRole.ADMIN: 2,
            WorkspaceRole.OWNER: 3
        }
        
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)
    
    async def connect(self, websocket: WebSocket, user: User, workspace_id: str) -> str:
        """Accept WebSocket connection and add to workspace"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user.id,
            workspace_id=workspace_id,
            connection_id=connection_id,
            connected_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        self.active_connections[connection_id] = connection
        self.workspace_connections[workspace_id].add(connection_id)
        self.user_connections[user.id].add(connection_id)
        
        # Initialize workspace if not exists
        if workspace_id not in self.workspaces:
            self.workspaces[workspace_id] = TeamWorkspace(
                workspace_id=workspace_id,
                name=f"Workspace {workspace_id}",
                description="Auto-created workspace",
                owner_id=user.id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                permissions={user.id: WorkspaceRole.OWNER}
            )
            self._save_workspaces()
        
        # Add user to workspace active users
        self.workspaces[workspace_id].active_users.add(user.id)
        
        # Start metrics streaming for this workspace if not already running
        if workspace_id not in self.metrics_streams:
            self.metrics_streams[workspace_id] = asyncio.create_task(
                self._stream_metrics(workspace_id)
            )
        
        # Notify other users in workspace
        await self._broadcast_to_workspace(
            workspace_id,
            {
                "type": MessageType.JOIN_WORKSPACE,
                "user_id": user.id,
                "username": user.username,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=user.id
        )
        
        # Send current workspace state to new connection
        await self._send_workspace_state(connection_id, workspace_id)
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        workspace_id = connection.workspace_id
        user_id = connection.user_id
        
        # Remove from tracking
        self.workspace_connections[workspace_id].discard(connection_id)
        self.user_connections[user_id].discard(connection_id)
        del self.active_connections[connection_id]
        
        # Remove from workspace active users if no other connections
        if not self.user_connections[user_id]:
            self.workspaces[workspace_id].active_users.discard(user_id)
        
        # Notify other users
        await self._broadcast_to_workspace(
            workspace_id,
            {
                "type": MessageType.LEAVE_WORKSPACE,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=user_id
        )
        
        # Stop metrics streaming if no connections in workspace
        if not self.workspace_connections[workspace_id]:
            if workspace_id in self.metrics_streams:
                self.metrics_streams[workspace_id].cancel()
                del self.metrics_streams[workspace_id]
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        message_type = message.get("type")
        
        try:
            if message_type == MessageType.HEARTBEAT:
                connection.last_heartbeat = datetime.utcnow()
                await self._send_ack(connection_id, "heartbeat")
            
            elif message_type == MessageType.MODEL_UPDATE:
                await self._handle_model_update(connection, message)
            
            elif message_type == MessageType.CURSOR_MOVE:
                await self._handle_cursor_move(connection, message)
            
            elif message_type == MessageType.CHAT_MESSAGE:
                await self._handle_chat_message(connection, message)
            
            elif message_type == MessageType.CHECKPOINT_SYNC:
                await self._handle_checkpoint_sync(connection, message)
            
            else:
                await self._send_error(connection_id, f"Unknown message type: {message_type}")
        
        except Exception as e:
            await self._send_error(connection_id, str(e))
    
    async def _handle_model_update(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle model update with CRDT conflict resolution"""
        workspace_id = connection.workspace_id
        user_id = connection.user_id
        
        # Check edit permission
        if not self.check_permission(user_id, workspace_id, WorkspaceRole.EDITOR):
            raise PermissionError("User does not have edit permission")
        
        # Create CRDT operation
        operation = CRDTOperation(
            operation_id=str(uuid.uuid4()),
            operation_type=message.get("operation_type", "update"),
            position=message.get("position", 0),
            value=message.get("value"),
            vector_clock=message.get("vector_clock", {}),
            user_id=user_id,
            workspace_id=workspace_id
        )
        
        # Apply CRDT operation
        new_state = self.crdt_manager.apply_operation(workspace_id, operation)
        
        # Update workspace checkpoint
        checkpoint_id = message.get("checkpoint_id", f"checkpoint_{int(time.time())}")
        if checkpoint_id in self.workspaces[workspace_id].checkpoints:
            checkpoint = self.workspaces[workspace_id].checkpoints[checkpoint_id]
            checkpoint.model_state = new_state
            checkpoint.updated_at = datetime.utcnow()
            checkpoint.version_vector = self.crdt_manager.vector_clocks[workspace_id]
        else:
            checkpoint = ModelCheckpoint(
                checkpoint_id=checkpoint_id,
                model_state=new_state,
                crdt_state={},
                version_vector=self.crdt_manager.vector_clocks[workspace_id],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=user_id,
                workspace_id=workspace_id
            )
            self.workspaces[workspace_id].checkpoints[checkpoint_id] = checkpoint
        
        # Broadcast update to all users in workspace
        await self._broadcast_to_workspace(
            workspace_id,
            {
                "type": MessageType.MODEL_UPDATE,
                "operation": asdict(operation),
                "checkpoint_id": checkpoint_id,
                "new_state": new_state,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=user_id
        )
        
        # Save workspace state
        self._save_workspaces()
        
        await self._send_ack(connection.connection_id, "model_update")
    
    async def _handle_cursor_move(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle cursor position update"""
        connection.cursor_position = message.get("position")
        
        await self._broadcast_to_workspace(
            connection.workspace_id,
            {
                "type": MessageType.CURSOR_MOVE,
                "user_id": connection.user_id,
                "position": connection.cursor_position,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=connection.user_id
        )
    
    async def _handle_chat_message(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle chat message in workspace"""
        await self._broadcast_to_workspace(
            connection.workspace_id,
            {
                "type": MessageType.CHAT_MESSAGE,
                "user_id": connection.user_id,
                "message": message.get("content"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _handle_checkpoint_sync(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle checkpoint synchronization request"""
        workspace_id = connection.workspace_id
        checkpoint_id = message.get("checkpoint_id")
        
        if checkpoint_id in self.workspaces[workspace_id].checkpoints:
            checkpoint = self.workspaces[workspace_id].checkpoints[checkpoint_id]
            await self._send_to_connection(
                connection.connection_id,
                {
                    "type": MessageType.CHECKPOINT_SYNC,
                    "checkpoint": asdict(checkpoint),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        else:
            await self._send_error(connection.connection_id, f"Checkpoint {checkpoint_id} not found")
    
    async def _stream_metrics(self, workspace_id: str):
        """Stream live training metrics to workspace"""
        try:
            while True:
                # Get active jobs for this workspace
                # In production, this would query the job manager
                metrics = LiveMetrics(
                    metrics_id=str(uuid.uuid4()),
                    job_id=f"job_{workspace_id}",
                    workspace_id=workspace_id,
                    epoch=1,
                    step=100,
                    loss=0.5,
                    accuracy=0.85,
                    learning_rate=0.001,
                    memory_usage=4.5,
                    gpu_utilization=75.0,
                    custom_metrics={"f1_score": 0.82}
                )
                
                # Broadcast metrics to workspace
                await self._broadcast_to_workspace(
                    workspace_id,
                    {
                        "type": MessageType.METRICS_UPDATE,
                        "metrics": asdict(metrics),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                await asyncio.sleep(5)  # Update every 5 seconds
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Metrics streaming error for workspace {workspace_id}: {e}")
    
    async def _broadcast_to_workspace(
        self, 
        workspace_id: str, 
        message: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """Broadcast message to all connections in a workspace"""
        if workspace_id not in self.workspace_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection_id in self.workspace_connections[workspace_id].copy():
            if connection_id not in self.active_connections:
                disconnected.append(connection_id)
                continue
            
            connection = self.active_connections[connection_id]
            
            # Skip excluded user
            if exclude_user and connection.user_id == exclude_user:
                continue
            
            try:
                await connection.websocket.send_text(message_str)
            except (ConnectionClosedOK, ConnectionClosedError):
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        try:
            await connection.websocket.send_text(json.dumps(message))
        except (ConnectionClosedOK, ConnectionClosedError):
            await self.disconnect(connection_id)
    
    async def _send_workspace_state(self, connection_id: str, workspace_id: str):
        """Send current workspace state to a connection"""
        if workspace_id not in self.workspaces:
            return
        
        workspace = self.workspaces[workspace_id]
        state_message = {
            "type": "workspace_state",
            "workspace": {
                "id": workspace.workspace_id,
                "name": workspace.name,
                "description": workspace.description,
                "active_users": list(workspace.active_users),
                "checkpoints": list(workspace.checkpoints.keys()),
                "permissions": {k: v.value for k, v in workspace.permissions.items()}
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_to_connection(connection_id, state_message)
    
    async def _send_ack(self, connection_id: str, operation: str):
        """Send acknowledgment to connection"""
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.ACK,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.ERROR,
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def cleanup_stale_connections(self, timeout_seconds: int = 300):
        """Clean up connections that haven't sent heartbeat"""
        current_time = datetime.utcnow()
        stale_connections = []
        
        for connection_id, connection in self.active_connections.items():
            time_since_heartbeat = (current_time - connection.last_heartbeat).total_seconds()
            if time_since_heartbeat > timeout_seconds:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            await self.disconnect(connection_id)
    
    async def update_workspace_permissions(
        self, 
        workspace_id: str, 
        admin_user_id: str,
        target_user_id: str, 
        new_role: WorkspaceRole
    ):
        """Update user permissions in workspace"""
        if workspace_id not in self.workspaces:
            raise CollaborationError(f"Workspace {workspace_id} not found")
        
        workspace = self.workspaces[workspace_id]
        
        # Check if admin has permission to update roles
        if not self.check_permission(admin_user_id, workspace_id, WorkspaceRole.ADMIN):
            raise PermissionError("Insufficient permissions to update roles")
        
        # Update permission
        workspace.permissions[target_user_id] = new_role
        workspace.updated_at = datetime.utcnow()
        
        # Notify workspace of permission change
        await self._broadcast_to_workspace(
            workspace_id,
            {
                "type": MessageType.PERMISSION_UPDATE,
                "user_id": target_user_id,
                "new_role": new_role.value,
                "updated_by": admin_user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        self._save_workspaces()


# FastAPI endpoint integration
async def collaboration_websocket_endpoint(
    websocket: WebSocket,
    workspace_id: str,
    token: Optional[str] = None,
    manager: WebSocketManager = Depends()
):
    """FastAPI WebSocket endpoint for real-time collaboration"""
    # Authenticate user
    user = await manager.authenticate_websocket(websocket, token)
    if not user:
        return
    
    # Connect to workspace
    connection_id = await manager.connect(websocket, user, workspace_id)
    
    try:
        # Main message loop
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await manager.handle_message(connection_id, message)
    
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)


# Dependency injection for FastAPI
def get_websocket_manager(
    storage_manager: StorageManager = Depends(),
    job_manager: JobManager = Depends()
) -> WebSocketManager:
    """Dependency to get WebSocket manager instance"""
    return WebSocketManager(storage_manager, job_manager)