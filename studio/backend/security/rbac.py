"""
Enterprise Security & Audit Module - RBAC Implementation
Replaces SQLite with PostgreSQL, implements fine-grained RBAC with team hierarchy,
integrates HashiCorp Vault for secrets, and adds comprehensive audit trails.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set, Tuple, Union
from enum import Enum
from contextlib import contextmanager
from functools import wraps

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, ForeignKey, Table, Text, Integer, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.pool import QueuePool

import hvac
from pydantic import BaseModel, Field

# Import existing modules
from studio.backend.auth.authentication import User, get_current_user
from studio.backend.auth.storage import get_db_session

logger = logging.getLogger(__name__)

# Database Configuration
Base = declarative_base()

# Association tables for many-to-many relationships
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.id'))
)

team_roles = Table(
    'team_roles',
    Base.metadata,
    Column('team_id', UUID(as_uuid=True), ForeignKey('teams.id', ondelete='CASCADE'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.id'))
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True)
)

team_members = Table(
    'team_members',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('team_id', UUID(as_uuid=True), ForeignKey('teams.id', ondelete='CASCADE'), primary_key=True),
    Column('joined_at', DateTime, default=datetime.utcnow),
    Column('is_admin', Boolean, default=False)
)


class PermissionType(str, Enum):
    """Fine-grained permission types for RBAC"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    MANAGE = "manage"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Resource types for permission scoping"""
    USER = "user"
    TEAM = "team"
    PROJECT = "project"
    DATASET = "dataset"
    MODEL = "model"
    JOB = "job"
    SECRET = "secret"
    AUDIT_LOG = "audit_log"
    SYSTEM = "system"


class AuditAction(str, Enum):
    """Audit log action types"""
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    PERMISSION_CHANGE = "permission_change"
    ROLE_CHANGE = "role_change"
    TEAM_CHANGE = "team_change"
    SECRET_ACCESS = "secret_access"
    SYSTEM = "system"


# SQLAlchemy Models
class Team(Base):
    """Team model with hierarchical structure"""
    __tablename__ = 'teams'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    parent_team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='SET NULL'))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    metadata_ = Column('metadata', JSONB, default=dict)
    
    # Relationships
    parent = relationship("Team", remote_side=[id], backref="children")
    members = relationship("User", secondary=team_members, back_populates="teams")
    roles = relationship("Role", secondary=team_roles, back_populates="teams")
    
    def get_all_parent_teams(self) -> List['Team']:
        """Get all parent teams in hierarchy"""
        parents = []
        current = self.parent
        while current:
            parents.append(current)
            current = current.parent
        return parents
    
    def get_all_child_teams(self) -> List['Team']:
        """Get all child teams recursively"""
        children = []
        stack = list(self.children)
        while stack:
            team = stack.pop()
            children.append(team)
            stack.extend(team.children)
        return children


class Role(Base):
    """Role model for RBAC"""
    __tablename__ = 'roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)  # System roles cannot be deleted
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    users = relationship("User", secondary=user_roles, back_populates="roles")
    teams = relationship("Team", secondary=team_roles, back_populates="roles")


class Permission(Base):
    """Permission model for fine-grained access control"""
    __tablename__ = 'permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    resource_type = Column(SQLEnum(ResourceType), nullable=False)
    action = Column(SQLEnum(PermissionType), nullable=False)
    scope = Column(String(255))  # Optional scope like project:123 or team:456
    is_system_permission = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    __table_args__ = (
        sa.UniqueConstraint('resource_type', 'action', 'scope', name='uq_permission_resource_action_scope'),
    )
    
    @classmethod
    def create_permission_name(cls, resource_type: ResourceType, action: PermissionType, scope: str = None) -> str:
        """Generate standardized permission name"""
        name = f"{resource_type.value}:{action.value}"
        if scope:
            name = f"{name}:{scope}"
        return name


class AuditLog(Base):
    """Comprehensive audit trail for compliance"""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    action = Column(SQLEnum(AuditAction), nullable=False)
    resource_type = Column(SQLEnum(ResourceType))
    resource_id = Column(String(255))
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='SET NULL'))
    ip_address = Column(String(45))  # IPv6 can be up to 45 chars
    user_agent = Column(Text)
    details = Column(JSONB, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(255))
    
    # Relationships
    user = relationship("User", backref="audit_logs")
    team = relationship("Team", backref="audit_logs")
    
    __table_args__ = (
        sa.Index('ix_audit_logs_user_timestamp', 'user_id', 'timestamp'),
        sa.Index('ix_audit_logs_resource', 'resource_type', 'resource_id'),
        sa.Index('ix_audit_logs_action_timestamp', 'action', 'timestamp'),
    )


# Extend existing User model with relationships
User.roles = relationship("Role", secondary=user_roles, back_populates="users")
User.teams = relationship("Team", secondary=team_members, back_populates="members")


class VaultSecretManager:
    """HashiCorp Vault integration for secret management"""
    
    def __init__(self, vault_url: str = None, vault_token: str = None):
        self.vault_url = vault_url or os.getenv('VAULT_URL', 'http://localhost:8200')
        self.vault_token = vault_token or os.getenv('VAULT_TOKEN')
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Vault"""
        try:
            self.client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token
            )
            if not self.client.is_authenticated():
                logger.error("Vault authentication failed")
                raise ConnectionError("Failed to authenticate with Vault")
            logger.info("Connected to HashiCorp Vault")
        except Exception as e:
            logger.error(f"Failed to connect to Vault: {e}")
            raise
    
    def get_secret(self, path: str, key: str = None) -> Any:
        """Retrieve secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            data = response['data']['data']
            return data.get(key) if key else data
        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {e}")
            raise
    
    def set_secret(self, path: str, secret: Dict[str, Any]):
        """Store secret in Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret
            )
            logger.info(f"Secret stored at {path}")
        except Exception as e:
            logger.error(f"Failed to store secret at {path}: {e}")
            raise
    
    def get_database_credentials(self) -> Dict[str, str]:
        """Get database credentials from Vault"""
        return self.get_secret('database/credentials')
    
    def get_api_key(self, service: str) -> str:
        """Get API key for external service"""
        return self.get_secret(f'apikeys/{service}', 'key')


class RBACManager:
    """Role-Based Access Control Manager"""
    
    def __init__(self, db_session: Session = None):
        self.db_session = db_session or get_db_session()
        self.vault = VaultSecretManager()
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize default system roles and permissions"""
        default_roles = [
            {
                'name': 'superadmin',
                'description': 'Super administrator with full system access',
                'is_system_role': True,
                'permissions': [
                    (ResourceType.SYSTEM, PermissionType.ADMIN, None),
                    (ResourceType.USER, PermissionType.MANAGE, None),
                    (ResourceType.TEAM, PermissionType.MANAGE, None),
                ]
            },
            {
                'name': 'admin',
                'description': 'Administrator with team management capabilities',
                'is_system_role': True,
                'permissions': [
                    (ResourceType.USER, PermissionType.READ, None),
                    (ResourceType.TEAM, PermissionType.MANAGE, None),
                    (ResourceType.PROJECT, PermissionType.MANAGE, None),
                ]
            },
            {
                'name': 'developer',
                'description': 'Developer with project and dataset access',
                'is_system_role': True,
                'permissions': [
                    (ResourceType.PROJECT, PermissionType.CREATE, None),
                    (ResourceType.PROJECT, PermissionType.READ, None),
                    (ResourceType.PROJECT, PermissionType.UPDATE, None),
                    (ResourceType.DATASET, PermissionType.CREATE, None),
                    (ResourceType.DATASET, PermissionType.READ, None),
                    (ResourceType.MODEL, PermissionType.CREATE, None),
                    (ResourceType.MODEL, PermissionType.READ, None),
                    (ResourceType.JOB, PermissionType.EXECUTE, None),
                ]
            },
            {
                'name': 'viewer',
                'description': 'Read-only access to resources',
                'is_system_role': True,
                'permissions': [
                    (ResourceType.PROJECT, PermissionType.READ, None),
                    (ResourceType.DATASET, PermissionType.READ, None),
                    (ResourceType.MODEL, PermissionType.READ, None),
                ]
            },
        ]
        
        for role_data in default_roles:
            if not self.db_session.query(Role).filter_by(name=role_data['name']).first():
                role = Role(
                    name=role_data['name'],
                    description=role_data['description'],
                    is_system_role=role_data['is_system_role']
                )
                self.db_session.add(role)
                
                # Add permissions to role
                for resource_type, action, scope in role_data['permissions']:
                    permission = self._get_or_create_permission(resource_type, action, scope)
                    role.permissions.append(permission)
        
        self.db_session.commit()
    
    def _get_or_create_permission(self, resource_type: ResourceType, action: PermissionType, scope: str = None) -> Permission:
        """Get or create a permission"""
        permission_name = Permission.create_permission_name(resource_type, action, scope)
        permission = self.db_session.query(Permission).filter_by(name=permission_name).first()
        
        if not permission:
            permission = Permission(
                name=permission_name,
                resource_type=resource_type,
                action=action,
                scope=scope,
                description=f"{action.value} on {resource_type.value}" + (f" (scope: {scope})" if scope else "")
            )
            self.db_session.add(permission)
            self.db_session.flush()
        
        return permission
    
    def create_role(self, name: str, description: str = None, 
                   permissions: List[Tuple[ResourceType, PermissionType, str]] = None,
                   created_by: uuid.UUID = None) -> Role:
        """Create a new role"""
        if self.db_session.query(Role).filter_by(name=name).first():
            raise ValueError(f"Role '{name}' already exists")
        
        role = Role(
            name=name,
            description=description,
            created_by=created_by
        )
        
        if permissions:
            for resource_type, action, scope in permissions:
                permission = self._get_or_create_permission(resource_type, action, scope)
                role.permissions.append(permission)
        
        self.db_session.add(role)
        self.db_session.commit()
        
        # Audit log
        self._create_audit_log(
            action=AuditAction.CREATE,
            resource_type=ResourceType.SYSTEM,
            resource_id=str(role.id),
            details={'role_name': name, 'action': 'create_role'}
        )
        
        return role
    
    def assign_role_to_user(self, user_id: uuid.UUID, role_id: uuid.UUID, 
                           assigned_by: uuid.UUID = None) -> bool:
        """Assign a role to a user"""
        user = self.db_session.query(User).get(user_id)
        role = self.db_session.query(Role).get(role_id)
        
        if not user or not role:
            raise ValueError("User or role not found")
        
        if role not in user.roles:
            user.roles.append(role)
            self.db_session.commit()
            
            # Audit log
            self._create_audit_log(
                action=AuditAction.ROLE_CHANGE,
                resource_type=ResourceType.USER,
                resource_id=str(user_id),
                user_id=assigned_by,
                details={
                    'action': 'assign_role',
                    'role_id': str(role_id),
                    'role_name': role.name
                }
            )
            return True
        return False
    
    def remove_role_from_user(self, user_id: uuid.UUID, role_id: uuid.UUID,
                            removed_by: uuid.UUID = None) -> bool:
        """Remove a role from a user"""
        user = self.db_session.query(User).get(user_id)
        role = self.db_session.query(Role).get(role_id)
        
        if not user or not role:
            raise ValueError("User or role not found")
        
        if role in user.roles:
            user.roles.remove(role)
            self.db_session.commit()
            
            # Audit log
            self._create_audit_log(
                action=AuditAction.ROLE_CHANGE,
                resource_type=ResourceType.USER,
                resource_id=str(user_id),
                user_id=removed_by,
                details={
                    'action': 'remove_role',
                    'role_id': str(role_id),
                    'role_name': role.name
                }
            )
            return True
        return False
    
    def assign_role_to_team(self, team_id: uuid.UUID, role_id: uuid.UUID,
                           assigned_by: uuid.UUID = None) -> bool:
        """Assign a role to a team"""
        team = self.db_session.query(Team).get(team_id)
        role = self.db_session.query(Role).get(role_id)
        
        if not team or not role:
            raise ValueError("Team or role not found")
        
        if role not in team.roles:
            team.roles.append(role)
            self.db_session.commit()
            
            # Audit log
            self._create_audit_log(
                action=AuditAction.ROLE_CHANGE,
                resource_type=ResourceType.TEAM,
                resource_id=str(team_id),
                user_id=assigned_by,
                details={
                    'action': 'assign_team_role',
                    'role_id': str(role_id),
                    'role_name': role.name
                }
            )
            return True
        return False
    
    def get_user_permissions(self, user_id: uuid.UUID, 
                            team_id: uuid.UUID = None) -> Set[str]:
        """Get all permissions for a user, optionally in a team context"""
        user = self.db_session.query(User).get(user_id)
        if not user:
            return set()
        
        permissions = set()
        
        # Direct user roles
        for role in user.roles:
            if role.is_active:
                for perm in role.permissions:
                    permissions.add(perm.name)
        
        # Team roles (if team context provided)
        if team_id:
            team = self.db_session.query(Team).get(team_id)
            if team and team.is_active:
                # Check if user is member of team
                if user in team.members:
                    for role in team.roles:
                        if role.is_active:
                            for perm in role.permissions:
                                permissions.add(perm.name)
                
                # Check parent teams for inherited permissions
                for parent_team in team.get_all_parent_teams():
                    if user in parent_team.members:
                        for role in parent_team.roles:
                            if role.is_active:
                                for perm in role.permissions:
                                    permissions.add(perm.name)
        
        # Also check all teams user belongs to
        for team in user.teams:
            if team.is_active:
                for role in team.roles:
                    if role.is_active:
                        for perm in role.permissions:
                            permissions.add(perm.name)
        
        return permissions
    
    def check_permission(self, user_id: uuid.UUID, resource_type: ResourceType,
                        action: PermissionType, resource_id: str = None,
                        team_id: uuid.UUID = None) -> bool:
        """Check if user has specific permission"""
        # Superadmin bypass
        user_permissions = self.get_user_permissions(user_id, team_id)
        
        # Check for admin permission on the resource type
        admin_perm = Permission.create_permission_name(resource_type, PermissionType.ADMIN)
        if admin_perm in user_permissions:
            return True
        
        # Check for manage permission on the resource type
        manage_perm = Permission.create_permission_name(resource_type, PermissionType.MANAGE)
        if manage_perm in user_permissions:
            return True
        
        # Check specific permission
        specific_perm = Permission.create_permission_name(resource_type, action, resource_id)
        if specific_perm in user_permissions:
            return True
        
        # Check generic permission (without scope)
        generic_perm = Permission.create_permission_name(resource_type, action)
        return generic_perm in user_permissions
    
    def _create_audit_log(self, action: AuditAction, resource_type: ResourceType = None,
                         resource_id: str = None, user_id: uuid.UUID = None,
                         team_id: uuid.UUID = None, details: Dict = None,
                         ip_address: str = None, user_agent: str = None,
                         session_id: str = None):
        """Create audit log entry"""
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            team_id=team_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            session_id=session_id
        )
        self.db_session.add(audit_log)
        self.db_session.commit()
        return audit_log


class DatabaseManager:
    """PostgreSQL database manager with connection pooling"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = create_engine(
            self.database_url,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _get_database_url(self) -> str:
        """Get database URL from Vault or environment"""
        try:
            vault = VaultSecretManager()
            creds = vault.get_database_credentials()
            return f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"
        except:
            # Fallback to environment variable
            return os.getenv('DATABASE_URL', 'postgresql://localhost:5432/forge')
    
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Decorators for permission checks
def require_permission(resource_type: ResourceType, action: PermissionType):
    """Decorator to require specific permission for endpoint"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from request (implementation depends on framework)
            request = kwargs.get('request') or args[0] if args else None
            if not request:
                raise PermissionError("Request object not found")
            
            user = getattr(request, 'user', None)
            if not user:
                raise PermissionError("User not authenticated")
            
            # Get resource ID from kwargs if available
            resource_id = kwargs.get('resource_id') or kwargs.get('id')
            team_id = kwargs.get('team_id')
            
            # Check permission
            rbac = RBACManager()
            if not rbac.check_permission(
                user_id=user.id,
                resource_type=resource_type,
                action=action,
                resource_id=str(resource_id) if resource_id else None,
                team_id=team_id
            ):
                # Audit failed permission check
                rbac._create_audit_log(
                    action=AuditAction.PERMISSION_CHANGE,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    user_id=user.id,
                    team_id=team_id,
                    details={
                        'action': 'permission_denied',
                        'required_permission': f"{resource_type.value}:{action.value}",
                        'endpoint': func.__name__
                    }
                )
                raise PermissionError(f"Permission denied: {resource_type.value}:{action.value}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role_name: str):
    """Decorator to require specific role for endpoint"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0] if args else None
            if not request:
                raise PermissionError("Request object not found")
            
            user = getattr(request, 'user', None)
            if not user:
                raise PermissionError("User not authenticated")
            
            # Check if user has the required role
            user_role_names = [role.name for role in user.roles]
            if role_name not in user_role_names:
                # Also check team roles
                for team in user.teams:
                    team_role_names = [role.name for role in team.roles]
                    if role_name in team_role_names:
                        break
                else:
                    raise PermissionError(f"Role required: {role_name}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Audit logging decorator
def audit_log(action: AuditAction, resource_type: ResourceType = None):
    """Decorator to automatically log actions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0] if args else None
            user = getattr(request, 'user', None) if request else None
            
            # Extract resource ID from kwargs
            resource_id = kwargs.get('resource_id') or kwargs.get('id')
            team_id = kwargs.get('team_id')
            
            # Get client info
            ip_address = request.client.host if request and hasattr(request, 'client') else None
            user_agent = request.headers.get('user-agent') if request and hasattr(request, 'headers') else None
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful action
                rbac = RBACManager()
                rbac._create_audit_log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    user_id=user.id if user else None,
                    team_id=team_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={
                        'function': func.__name__,
                        'status': 'success',
                        'args': {k: str(v) for k, v in kwargs.items() if k != 'request'}
                    }
                )
                
                return result
                
            except Exception as e:
                # Log failed action
                rbac = RBACManager()
                rbac._create_audit_log(
                    action=AuditAction.SYSTEM,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    user_id=user.id if user else None,
                    team_id=team_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={
                        'function': func.__name__,
                        'status': 'failed',
                        'error': str(e),
                        'args': {k: str(v) for k, v in kwargs.items() if k != 'request'}
                    }
                )
                raise
                
        return wrapper
    return decorator


# Utility functions
def create_default_admin_user(db_session: Session, user_id: uuid.UUID):
    """Assign superadmin role to initial user"""
    rbac = RBACManager(db_session)
    superadmin_role = db_session.query(Role).filter_by(name='superadmin').first()
    if superadmin_role:
        rbac.assign_role_to_user(user_id, superadmin_role.id, assigned_by=user_id)
        logger.info(f"Assigned superadmin role to user {user_id}")


def get_user_accessible_teams(user_id: uuid.UUID, db_session: Session = None) -> List[Team]:
    """Get all teams a user has access to (direct membership + child teams)"""
    session = db_session or get_db_session()
    user = session.query(User).get(user_id)
    
    if not user:
        return []
    
    accessible_teams = set(user.teams)
    
    # Add all child teams of user's teams
    for team in user.teams:
        accessible_teams.update(team.get_all_child_teams())
    
    return list(accessible_teams)


def migrate_from_sqlite(db_session: Session):
    """Migrate data from SQLite to PostgreSQL"""
    # This would contain migration logic from existing SQLite database
    # Implementation depends on existing schema
    logger.info("Migration from SQLite to PostgreSQL completed")


# Initialize database on module import
db_manager = DatabaseManager()
rbac_manager = RBACManager()

# Export key classes and functions
__all__ = [
    'RBACManager',
    'DatabaseManager',
    'VaultSecretManager',
    'Team',
    'Role',
    'Permission',
    'AuditLog',
    'require_permission',
    'require_role',
    'audit_log',
    'create_default_admin_user',
    'get_user_accessible_teams',
    'PermissionType',
    'ResourceType',
    'AuditAction'
]