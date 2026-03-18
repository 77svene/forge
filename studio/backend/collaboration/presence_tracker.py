import asyncio
import time
import json
import logging
from typing import Dict, Set, Optional, Any, List, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import weakref

# Integration with existing auth system
from studio.backend.auth.authentication import verify_token, get_current_user
from studio.backend.auth.storage import UserDB

logger = logging.getLogger(__name__)


class PresenceStatus(str, Enum):
    """User presence status in a workspace."""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"


class CollaborationEventType(str, Enum):
    """Types of collaboration events."""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    STATUS_CHANGED = "status_changed"
    CURSOR_MOVED = "cursor_moved"
    CHECKPOINT_UPDATED = "checkpoint_updated"
    METRICS_UPDATED = "metrics_updated"
    WORKSPACE_UPDATED = "workspace_updated"
    CONFLICT_RESOLVED = "conflict_resolved"


@dataclass
class UserPresence:
    """Presence information for a user in a workspace."""
    user_id: str
    username: str
    status: PresenceStatus
    last_active: float
    cursor_position: Optional[Dict[str, Any]] = None
    current_file: Optional[str] = None
    current_line: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class WorkspaceState:
    """State of a collaborative workspace."""
    workspace_id: str
    name: str
    owner_id: str
    participants: Dict[str, UserPresence] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metrics_stream: List[Dict[str, Any]] = field(default_factory=list)
    crdt_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    permissions: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'workspace_id': self.workspace_id,
            'name': self.name,
            'owner_id': self.owner_id,
            'participants': {uid: p.to_dict() for uid, p in self.participants.items()},
            'checkpoint_count': len(self.checkpoints),
            'metrics_count': len(self.metrics_stream),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
        
        if include_sensitive:
            data['checkpoints'] = self.checkpoints[-10:]  # Last 10 checkpoints
            data['metrics'] = self.metrics_stream[-100:]  # Last 100 metrics
            data['crdt_state'] = self.crdt_state
        
        return data


class CRDTMergeEngine:
    """CRDT-based conflict resolution for collaborative editing."""
    
    @staticmethod
    def merge_states(local_state: Dict[str, Any], remote_state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two CRDT states using last-write-wins with vector clocks."""
        merged = {}
        
        for key in set(local_state.keys()) | set(remote_state.keys()):
            local_val = local_state.get(key)
            remote_val = remote_state.get(key)
            
            if local_val is None:
                merged[key] = remote_val
            elif remote_val is None:
                merged[key] = local_val
            else:
                # Compare vector clocks or timestamps
                local_clock = local_val.get('vector_clock', {})
                remote_clock = remote_val.get('vector_clock', {})
                
                # Simple last-write-wins for now
                local_ts = local_val.get('timestamp', 0)
                remote_ts = remote_val.get('timestamp', 0)
                
                if remote_ts > local_ts:
                    merged[key] = remote_val
                else:
                    merged[key] = local_val
        
        return merged
    
    @staticmethod
    def create_operation(operation_type: str, path: str, value: Any, user_id: str) -> Dict[str, Any]:
        """Create a CRDT operation."""
        return {
            'type': operation_type,
            'path': path,
            'value': value,
            'user_id': user_id,
            'timestamp': time.time(),
            'vector_clock': {user_id: int(time.time() * 1000)}
        }


class PresenceTracker:
    """
    Real-time presence tracking and collaboration system.
    Manages WebSocket connections, user presence, and collaborative state.
    """
    
    def __init__(self):
        self.workspaces: Dict[str, WorkspaceState] = {}
        self.user_workspaces: Dict[str, Set[str]] = defaultdict(set)  # user_id -> workspace_ids
        self.connections: Dict[str, Dict[str, Any]] = {}  # connection_id -> connection_info
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> connection_ids
        
        # Event handlers
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = defaultdict(list)
        
        # Metrics streaming
        self.metrics_subscribers: Dict[str, Set[str]] = defaultdict(set)  # workspace_id -> connection_ids
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_broadcast_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.presence_timeout = 300  # 5 minutes
        self.max_metrics_history = 1000
        self.max_checkpoints_per_workspace = 50
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start background tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_inactive_presence())
        self.metrics_broadcast_task = asyncio.create_task(self._broadcast_metrics_periodic())
        logger.info("Presence tracker started")
    
    async def stop(self):
        """Stop background tasks."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.metrics_broadcast_task:
            self.metrics_broadcast_task.cancel()
        
        # Close all WebSocket connections
        async with self._lock:
            for conn_id, conn_info in list(self.connections.items()):
                try:
                    if 'websocket' in conn_info:
                        await conn_info['websocket'].close()
                except Exception:
                    pass
        
        logger.info("Presence tracker stopped")
    
    async def connect(self, websocket, workspace_id: str, token: str) -> bool:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
            workspace_id: ID of the workspace to join
            token: Authentication token
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Verify token and get user
            user_data = await self._authenticate_user(token)
            if not user_data:
                await websocket.close(code=4001, reason="Authentication failed")
                return False
            
            user_id = user_data['user_id']
            username = user_data.get('username', user_id)
            
            # Check workspace permissions
            if not await self._check_workspace_permission(user_id, workspace_id, 'view'):
                await websocket.close(code=4003, reason="Permission denied")
                return False
            
            async with self._lock:
                # Create connection ID
                conn_id = f"{user_id}_{workspace_id}_{id(websocket)}"
                
                # Store connection info
                self.connections[conn_id] = {
                    'websocket': websocket,
                    'user_id': user_id,
                    'workspace_id': workspace_id,
                    'connected_at': time.time(),
                    'last_activity': time.time()
                }
                
                # Update user connections
                self.user_connections[user_id].add(conn_id)
                
                # Create or get workspace
                if workspace_id not in self.workspaces:
                    self.workspaces[workspace_id] = WorkspaceState(
                        workspace_id=workspace_id,
                        name=f"Workspace {workspace_id}",
                        owner_id=user_id
                    )
                
                workspace = self.workspaces[workspace_id]
                
                # Update user presence
                presence = UserPresence(
                    user_id=user_id,
                    username=username,
                    status=PresenceStatus.ONLINE,
                    last_active=time.time(),
                    metadata={'connection_id': conn_id}
                )
                workspace.participants[user_id] = presence
                workspace.updated_at = time.time()
                
                # Update user-workspace mapping
                self.user_workspaces[user_id].add(workspace_id)
                
                # Set permissions if owner
                if user_id == workspace.owner_id:
                    workspace.permissions[user_id] = {'view', 'edit', 'manage', 'delete'}
            
            # Send initial state
            await self._send_initial_state(conn_id, workspace_id)
            
            # Notify others
            await self._broadcast_event(
                workspace_id,
                CollaborationEventType.USER_JOINED,
                {
                    'user_id': user_id,
                    'username': username,
                    'status': PresenceStatus.ONLINE.value,
                    'timestamp': time.time()
                },
                exclude_user=user_id
            )
            
            logger.info(f"User {user_id} connected to workspace {workspace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            try:
                await websocket.close(code=4000, reason="Internal error")
            except Exception:
                pass
            return False
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            # Remove connection
            del self.connections[connection_id]
            self.user_connections[user_id].discard(connection_id)
            
            # Update workspace if no more connections for this user
            if not self.user_connections[user_id]:
                if workspace_id in self.workspaces:
                    workspace = self.workspaces[workspace_id]
                    if user_id in workspace.participants:
                        workspace.participants[user_id].status = PresenceStatus.OFFLINE
                        workspace.participants[user_id].last_active = time.time()
                        workspace.updated_at = time.time()
                
                # Notify others
                await self._broadcast_event(
                    workspace_id,
                    CollaborationEventType.USER_LEFT,
                    {
                        'user_id': user_id,
                        'timestamp': time.time()
                    }
                )
            
            # Remove from metrics subscribers
            self.metrics_subscribers[workspace_id].discard(connection_id)
            
            logger.info(f"Connection {connection_id} disconnected")
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        try:
            async with self._lock:
                if connection_id not in self.connections:
                    return
                
                conn_info = self.connections[connection_id]
                conn_info['last_activity'] = time.time()
                
                user_id = conn_info['user_id']
                workspace_id = conn_info['workspace_id']
                
                # Update presence last active time
                if workspace_id in self.workspaces:
                    workspace = self.workspaces[workspace_id]
                    if user_id in workspace.participants:
                        workspace.participants[user_id].last_active = time.time()
            
            message_type = message.get('type')
            
            if message_type == 'status_update':
                await self._handle_status_update(connection_id, message)
            elif message_type == 'cursor_move':
                await self._handle_cursor_move(connection_id, message)
            elif message_type == 'checkpoint_update':
                await self._handle_checkpoint_update(connection_id, message)
            elif message_type == 'metrics_update':
                await self._handle_metrics_update(connection_id, message)
            elif message_type == 'crdt_operation':
                await self._handle_crdt_operation(connection_id, message)
            elif message_type == 'subscribe_metrics':
                await self._handle_metrics_subscription(connection_id, True)
            elif message_type == 'unsubscribe_metrics':
                await self._handle_metrics_subscription(connection_id, False)
            elif message_type == 'ping':
                await self._send_to_connection(connection_id, {'type': 'pong'})
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_status_update(self, connection_id: str, message: Dict[str, Any]):
        """Handle user status update."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            if user_id not in workspace.participants:
                return
            
            new_status = PresenceStatus(message.get('status', PresenceStatus.ONLINE.value))
            workspace.participants[user_id].status = new_status
            workspace.updated_at = time.time()
        
        await self._broadcast_event(
            workspace_id,
            CollaborationEventType.STATUS_CHANGED,
            {
                'user_id': user_id,
                'status': new_status.value,
                'timestamp': time.time()
            }
        )
    
    async def _handle_cursor_move(self, connection_id: str, message: Dict[str, Any]):
        """Handle cursor position update."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            if user_id not in workspace.participants:
                return
            
            workspace.participants[user_id].cursor_position = message.get('position')
            workspace.participants[user_id].current_file = message.get('file')
            workspace.participants[user_id].current_line = message.get('line')
            workspace.updated_at = time.time()
        
        await self._broadcast_event(
            workspace_id,
            CollaborationEventType.CURSOR_MOVED,
            {
                'user_id': user_id,
                'position': message.get('position'),
                'file': message.get('file'),
                'line': message.get('line'),
                'timestamp': time.time()
            },
            exclude_user=user_id
        )
    
    async def _handle_checkpoint_update(self, connection_id: str, message: Dict[str, Any]):
        """Handle model checkpoint update."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            # Check edit permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'edit'):
                await self._send_error(connection_id, "Permission denied for checkpoint update")
                return
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            
            checkpoint_data = {
                'checkpoint_id': message.get('checkpoint_id', f"ckpt_{int(time.time())}"),
                'user_id': user_id,
                'timestamp': time.time(),
                'metadata': message.get('metadata', {}),
                'metrics': message.get('metrics', {}),
                'model_hash': message.get('model_hash'),
                'file_path': message.get('file_path')
            }
            
            workspace.checkpoints.append(checkpoint_data)
            
            # Limit checkpoints
            if len(workspace.checkpoints) > self.max_checkpoints_per_workspace:
                workspace.checkpoints = workspace.checkpoints[-self.max_checkpoints_per_workspace:]
            
            workspace.updated_at = time.time()
        
        await self._broadcast_event(
            workspace_id,
            CollaborationEventType.CHECKPOINT_UPDATED,
            checkpoint_data
        )
    
    async def _handle_metrics_update(self, connection_id: str, message: Dict[str, Any]):
        """Handle training metrics update."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            # Check edit permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'edit'):
                return
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            
            metrics_data = {
                'user_id': user_id,
                'timestamp': time.time(),
                'metrics': message.get('metrics', {}),
                'step': message.get('step'),
                'epoch': message.get('epoch'),
                'loss': message.get('loss'),
                'accuracy': message.get('accuracy'),
                'learning_rate': message.get('learning_rate')
            }
            
            workspace.metrics_stream.append(metrics_data)
            
            # Limit metrics history
            if len(workspace.metrics_stream) > self.max_metrics_history:
                workspace.metrics_stream = workspace.metrics_stream[-self.max_metrics_history:]
            
            workspace.updated_at = time.time()
        
        # Broadcast to metrics subscribers
        await self._broadcast_to_metrics_subscribers(workspace_id, metrics_data)
    
    async def _handle_crdt_operation(self, connection_id: str, message: Dict[str, Any]):
        """Handle CRDT operation for conflict-free editing."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            workspace_id = conn_info['workspace_id']
            
            # Check edit permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'edit'):
                return
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            
            operation = CRDTMergeEngine.create_operation(
                operation_type=message.get('operation_type', 'update'),
                path=message.get('path', ''),
                value=message.get('value'),
                user_id=user_id
            )
            
            # Merge with existing state
            current_state = workspace.crdt_state.get(message.get('path', ''), {})
            new_state = CRDTMergeEngine.merge_states(
                current_state,
                {'value': message.get('value'), 'timestamp': time.time(), 'user_id': user_id}
            )
            
            workspace.crdt_state[message.get('path', '')] = new_state
            workspace.updated_at = time.time()
        
        await self._broadcast_event(
            workspace_id,
            CollaborationEventType.CONFLICT_RESOLVED,
            {
                'operation': operation,
                'resolved_state': new_state,
                'timestamp': time.time()
            }
        )
    
    async def _handle_metrics_subscription(self, connection_id: str, subscribe: bool):
        """Handle metrics stream subscription."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            workspace_id = conn_info['workspace_id']
            
            if subscribe:
                self.metrics_subscribers[workspace_id].add(connection_id)
            else:
                self.metrics_subscribers[workspace_id].discard(connection_id)
    
    async def _send_initial_state(self, connection_id: str, workspace_id: str):
        """Send initial workspace state to newly connected client."""
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            conn_info = self.connections[connection_id]
            user_id = conn_info['user_id']
            
            # Get permissions for this user
            permissions = list(workspace.permissions.get(user_id, set()))
            
            state_data = {
                'type': 'initial_state',
                'workspace': workspace.to_dict(include_sensitive='manage' in permissions),
                'user_id': user_id,
                'permissions': permissions,
                'timestamp': time.time()
            }
            
            await self._send_to_connection(connection_id, state_data)
    
    async def _broadcast_event(self, workspace_id: str, event_type: CollaborationEventType, 
                              data: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast event to all participants in a workspace."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return
            
            workspace = self.workspaces[workspace_id]
            message = {
                'type': event_type.value,
                'workspace_id': workspace_id,
                'data': data,
                'timestamp': time.time()
            }
            
            tasks = []
            for user_id, presence in workspace.participants.items():
                if user_id == exclude_user:
                    continue
                
                for conn_id in self.user_connections.get(user_id, set()):
                    if conn_id in self.connections:
                        tasks.append(self._send_to_connection(conn_id, message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _broadcast_to_metrics_subscribers(self, workspace_id: str, metrics_data: Dict[str, Any]):
        """Broadcast metrics to subscribers."""
        async with self._lock:
            if workspace_id not in self.metrics_subscribers:
                return
            
            message = {
                'type': 'metrics_update',
                'workspace_id': workspace_id,
                'data': metrics_data,
                'timestamp': time.time()
            }
            
            tasks = []
            for conn_id in self.metrics_subscribers[workspace_id]:
                if conn_id in self.connections:
                    tasks.append(self._send_to_connection(conn_id, message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_connection(self, connection_id: str, data: Dict[str, Any]):
        """Send data to a specific connection."""
        try:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            websocket = conn_info.get('websocket')
            
            if websocket and hasattr(websocket, 'send_json'):
                await websocket.send_json(data)
            elif websocket and hasattr(websocket, 'send'):
                await websocket.send(json.dumps(data))
                
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection."""
        await self._send_to_connection(connection_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': time.time()
        })
    
    async def _authenticate_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user using existing auth system."""
        try:
            # Use existing authentication module
            user_data = await verify_token(token)
            if user_data:
                return {
                    'user_id': user_data.get('user_id') or user_data.get('sub'),
                    'username': user_data.get('username', ''),
                    'email': user_data.get('email', '')
                }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
        
        return None
    
    async def _check_workspace_permission(self, user_id: str, workspace_id: str, permission: str) -> bool:
        """Check if user has permission for workspace."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Owner has all permissions
            if user_id == workspace.owner_id:
                return True
            
            # Check specific permission
            user_permissions = workspace.permissions.get(user_id, set())
            return permission in user_permissions or 'manage' in user_permissions
    
    async def _cleanup_inactive_presence(self):
        """Background task to clean up inactive presence."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = time.time()
                
                async with self._lock:
                    for workspace_id, workspace in list(self.workspaces.items()):
                        users_to_remove = []
                        
                        for user_id, presence in workspace.participants.items():
                            if (presence.status != PresenceStatus.OFFLINE and 
                                current_time - presence.last_active > self.presence_timeout):
                                presence.status = PresenceStatus.AWAY
                                users_to_remove.append(user_id)
                        
                        # Notify about status changes
                        for user_id in users_to_remove:
                            await self._broadcast_event(
                                workspace_id,
                                CollaborationEventType.STATUS_CHANGED,
                                {
                                    'user_id': user_id,
                                    'status': PresenceStatus.AWAY.value,
                                    'timestamp': current_time
                                }
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _broadcast_metrics_periodic(self):
        """Background task to broadcast aggregated metrics periodically."""
        while True:
            try:
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
                async with self._lock:
                    for workspace_id, workspace in self.workspaces.items():
                        if not workspace.metrics_stream:
                            continue
                        
                        # Aggregate recent metrics
                        recent_metrics = workspace.metrics_stream[-10:]  # Last 10 metrics
                        if not recent_metrics:
                            continue
                        
                        aggregated = {
                            'avg_loss': sum(m.get('loss', 0) for m in recent_metrics) / len(recent_metrics),
                            'avg_accuracy': sum(m.get('accuracy', 0) for m in recent_metrics) / len(recent_metrics),
                            'latest_step': max(m.get('step', 0) for m in recent_metrics),
                            'timestamp': time.time()
                        }
                        
                        await self._broadcast_to_metrics_subscribers(workspace_id, {
                            'type': 'aggregated_metrics',
                            'workspace_id': workspace_id,
                            'data': aggregated
                        })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics broadcast task: {e}")
    
    # Public API methods for external integration
    
    async def create_workspace(self, user_id: str, name: str, workspace_id: Optional[str] = None) -> str:
        """Create a new workspace."""
        async with self._lock:
            if workspace_id is None:
                workspace_id = f"ws_{user_id}_{int(time.time())}"
            
            if workspace_id in self.workspaces:
                raise ValueError(f"Workspace {workspace_id} already exists")
            
            workspace = WorkspaceState(
                workspace_id=workspace_id,
                name=name,
                owner_id=user_id
            )
            
            # Set owner permissions
            workspace.permissions[user_id] = {'view', 'edit', 'manage', 'delete'}
            
            self.workspaces[workspace_id] = workspace
            self.user_workspaces[user_id].add(workspace_id)
            
            logger.info(f"Workspace {workspace_id} created by user {user_id}")
            return workspace_id
    
    async def delete_workspace(self, user_id: str, workspace_id: str) -> bool:
        """Delete a workspace."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Check permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'delete'):
                return False
            
            # Notify participants
            await self._broadcast_event(
                workspace_id,
                CollaborationEventType.WORKSPACE_UPDATED,
                {'action': 'deleted', 'workspace_id': workspace_id}
            )
            
            # Clean up
            del self.workspaces[workspace_id]
            
            # Clean up user-workspace mappings
            for uid in list(self.user_workspaces.keys()):
                self.user_workspaces[uid].discard(workspace_id)
            
            logger.info(f"Workspace {workspace_id} deleted by user {user_id}")
            return True
    
    async def add_workspace_permission(self, owner_id: str, workspace_id: str, 
                                      target_user_id: str, permissions: Set[str]) -> bool:
        """Add permissions for a user in a workspace."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Only owner or users with manage permission can add permissions
            if not await self._check_workspace_permission(owner_id, workspace_id, 'manage'):
                return False
            
            workspace.permissions[target_user_id] = permissions
            workspace.updated_at = time.time()
            
            logger.info(f"Permissions {permissions} added for user {target_user_id} in workspace {workspace_id}")
            return True
    
    async def get_workspace_presence(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get current presence in a workspace."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return []
            
            workspace = self.workspaces[workspace_id]
            return [p.to_dict() for p in workspace.participants.values()]
    
    async def get_user_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all workspaces for a user."""
        async with self._lock:
            workspace_ids = self.user_workspaces.get(user_id, set())
            workspaces = []
            
            for ws_id in workspace_ids:
                if ws_id in self.workspaces:
                    workspace = self.workspaces[ws_id]
                    workspaces.append({
                        'workspace_id': ws_id,
                        'name': workspace.name,
                        'owner_id': workspace.owner_id,
                        'participant_count': len(workspace.participants),
                        'updated_at': workspace.updated_at
                    })
            
            return workspaces
    
    async def update_training_metrics(self, workspace_id: str, user_id: str, 
                                     metrics: Dict[str, Any]) -> bool:
        """Update training metrics for a workspace (for external systems)."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Check edit permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'edit'):
                return False
            
            metrics_data = {
                'user_id': user_id,
                'timestamp': time.time(),
                'metrics': metrics,
                'source': 'external'
            }
            
            workspace.metrics_stream.append(metrics_data)
            
            # Limit metrics history
            if len(workspace.metrics_stream) > self.max_metrics_history:
                workspace.metrics_stream = workspace.metrics_stream[-self.max_metrics_history:]
            
            workspace.updated_at = time.time()
            
            # Broadcast to subscribers
            await self._broadcast_to_metrics_subscribers(workspace_id, metrics_data)
            
            return True
    
    async def save_checkpoint(self, workspace_id: str, user_id: str, 
                             checkpoint_data: Dict[str, Any]) -> bool:
        """Save a model checkpoint for a workspace (for external systems)."""
        async with self._lock:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Check edit permission
            if not await self._check_workspace_permission(user_id, workspace_id, 'edit'):
                return False
            
            checkpoint = {
                'checkpoint_id': checkpoint_data.get('checkpoint_id', f"ckpt_{int(time.time())}"),
                'user_id': user_id,
                'timestamp': time.time(),
                'metadata': checkpoint_data.get('metadata', {}),
                'metrics': checkpoint_data.get('metrics', {}),
                'model_hash': checkpoint_data.get('model_hash'),
                'file_path': checkpoint_data.get('file_path'),
                'source': 'external'
            }
            
            workspace.checkpoints.append(checkpoint)
            
            # Limit checkpoints
            if len(workspace.checkpoints) > self.max_checkpoints_per_workspace:
                workspace.checkpoints = workspace.checkpoints[-self.max_checkpoints_per_workspace:]
            
            workspace.updated_at = time.time()
            
            # Notify participants
            await self._broadcast_event(
                workspace_id,
                CollaborationEventType.CHECKPOINT_UPDATED,
                checkpoint
            )
            
            return True


# Singleton instance for global access
presence_tracker = PresenceTracker()


# Integration with existing CLI and backend systems
async def initialize_collaboration_system():
    """Initialize the collaboration system."""
    await presence_tracker.start()
    logger.info("Collaboration system initialized")


async def shutdown_collaboration_system():
    """Shutdown the collaboration system."""
    await presence_tracker.stop()
    logger.info("Collaboration system shutdown")


# FastAPI/Starlette integration helpers
async def websocket_endpoint(websocket, workspace_id: str, token: str):
    """WebSocket endpoint for collaboration."""
    connection_id = None
    try:
        # Accept connection
        if hasattr(websocket, 'accept'):
            await websocket.accept()
        
        # Connect to presence tracker
        success = await presence_tracker.connect(websocket, workspace_id, token)
        if not success:
            return
        
        # Get connection ID
        user_data = await presence_tracker._authenticate_user(token)
        if user_data:
            connection_id = f"{user_data['user_id']}_{workspace_id}_{id(websocket)}"
        
        # Message loop
        while True:
            try:
                if hasattr(websocket, 'receive_json'):
                    data = await websocket.receive_json()
                elif hasattr(websocket, 'receive'):
                    data = await websocket.receive()
                    if data.get('type') == 'websocket.disconnect':
                        break
                    data = json.loads(data.get('text', '{}'))
                else:
                    break
                
                await presence_tracker.handle_message(connection_id, data)
                
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            await presence_tracker.disconnect(connection_id)


# Example usage with existing auth system
"""
from fastapi import FastAPI, WebSocket, Depends
from studio.backend.auth.authentication import get_current_user_ws

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await initialize_collaboration_system()

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_collaboration_system()

@app.websocket("/ws/collaborate/{workspace_id}")
async def websocket_collaborate(
    websocket: WebSocket,
    workspace_id: str,
    token: str = Depends(get_current_user_ws)
):
    await websocket_endpoint(websocket, workspace_id, token)
"""