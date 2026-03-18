# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
PostgreSQL storage for authentication data with RBAC, audit logging, and secret management.
Extended for real-time collaboration with team workspaces, shared checkpoints,
and live training metrics.
"""

import hashlib
import secrets
import os
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
import json

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float, LargeBinary, JSON, Enum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
import psycopg2
from psycopg2.extras import RealDictCursor

from utils.paths import auth_db_path, ensure_dir
import hvac  # HashiCorp Vault client

Base = declarative_base()

# Vault configuration
VAULT_ADDR = os.getenv('VAULT_ADDR', 'http://localhost:8200')
VAULT_TOKEN = os.getenv('VAULT_TOKEN')
VAULT_SECRET_PATH = os.getenv('VAULT_SECRET_PATH', 'secret/data/forge')

DEFAULT_ADMIN_USERNAME = "forge"

# Plaintext bootstrap password file — lives beside auth.db, deleted on
# first password change so the credential never lingers on disk.
_BOOTSTRAP_PW_PATH = auth_db_path().parent / ".bootstrap_password"

# In-process cache so we don't re-read the file on every HTML serve.
_bootstrap_password: Optional[str] = None


class VaultSecretManager:
    """Manages secrets using HashiCorp Vault."""
    
    def __init__(self):
        self.client = None
        self._initialize_vault()
    
    def _initialize_vault(self):
        """Initialize Vault client."""
        try:
            self.client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
            if not self.client.is_authenticated():
                print("Warning: Vault authentication failed")
        except Exception as e:
            print(f"Warning: Could not connect to Vault: {e}")
            self.client = None
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from Vault."""
        if not self.client:
            return default
        
        try:
            secret_path = f"{VAULT_SECRET_PATH}/{key}"
            result = self.client.read(secret_path)
            if result and 'data' in result and 'data' in result['data']:
                return result['data']['data'].get('value', default)
        except Exception as e:
            print(f"Error reading secret {key} from Vault: {e}")
        
        return default
    
    def set_secret(self, key: str, value: str) -> bool:
        """Store secret in Vault."""
        if not self.client:
            return False
        
        try:
            secret_path = f"{VAULT_SECRET_PATH}/{key}"
            self.client.write(secret_path, data={'value': value})
            return True
        except Exception as e:
            print(f"Error writing secret {key} to Vault: {e}")
            return False


# SQLAlchemy Models
class AuthUser(Base):
    __tablename__ = 'auth_user'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    password_salt = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    jwt_secret = Column(String(255), nullable=False)
    must_change_password = Column(Boolean, default=False)
    email = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    workspace_memberships = relationship("WorkspaceMember", back_populates="user")
    created_workspaces = relationship("Workspace", back_populates="creator")
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = 'refresh_tokens'
    
    id = Column(Integer, primary_key=True)
    token_hash = Column(String(255), nullable=False, index=True)
    username = Column(String(255), ForeignKey('auth_user.username', ondelete='CASCADE'), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    revoked = Column(Boolean, default=False)
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    
    # Relationships
    user = relationship("AuthUser", back_populates="refresh_tokens")


class Workspace(Base):
    __tablename__ = 'workspaces'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_by = Column(String(255), ForeignKey('auth_user.username'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_public = Column(Boolean, default=False)
    max_members = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    creator = relationship("AuthUser", back_populates="created_workspaces")
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    checkpoints = relationship("SharedCheckpoint", back_populates="workspace", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="workspace", cascade="all, delete-orphan")
    invitations = relationship("WorkspaceInvitation", back_populates="workspace", cascade="all, delete-orphan")


class WorkspaceMember(Base):
    __tablename__ = 'workspace_members'
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    username = Column(String(255), ForeignKey('auth_user.username'), nullable=False)
    role = Column(Enum('owner', 'admin', 'editor', 'viewer', name='workspace_role_enum'), nullable=False)
    joined_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_active = Column(DateTime(timezone=True))
    
    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("AuthUser", back_populates="workspace_memberships")
    
    __table_args__ = (
        Index('idx_workspace_members_username', 'username'),
        Index('idx_workspace_members_workspace', 'workspace_id'),
        Index('idx_workspace_members_role', 'role'),
    )


class SharedCheckpoint(Base):
    __tablename__ = 'shared_checkpoints'
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    checkpoint_path = Column(String(500), nullable=False)
    model_type = Column(String(100), nullable=False)
    created_by = Column(String(255), ForeignKey('auth_user.username'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    crdt_state = Column(LargeBinary)
    version = Column(Integer, default=1)
    metadata_json = Column(JSON)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="checkpoints")
    creator = relationship("AuthUser")
    
    __table_args__ = (
        Index('idx_shared_checkpoints_workspace', 'workspace_id'),
        Index('idx_shared_checkpoints_active', 'is_active'),
    )


class TrainingRun(Base):
    __tablename__ = 'training_runs'
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    checkpoint_id = Column(Integer, ForeignKey('shared_checkpoints.id'))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(Enum('pending', 'running', 'paused', 'completed', 'failed', name='training_status_enum'), nullable=False)
    started_by = Column(String(255), ForeignKey('auth_user.username'), nullable=False)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True))
    config_json = Column(JSON)
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="training_runs")
    checkpoint = relationship("SharedCheckpoint")
    starter = relationship("AuthUser")
    metrics = relationship("TrainingMetric", back_populates="run", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_training_runs_workspace', 'workspace_id'),
        Index('idx_training_runs_status', 'status'),
    )


class TrainingMetric(Base):
    __tablename__ = 'training_metrics'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    epoch = Column(Integer)
    batch = Column(Integer)
    
    # Relationships
    run = relationship("TrainingRun", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_training_metrics_run', 'run_id'),
        Index('idx_training_metrics_step', 'step'),
        Index('idx_training_metrics_name', 'metric_name'),
    )


class WorkspaceInvitation(Base):
    __tablename__ = 'workspace_invitations'
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    email = Column(String(255), nullable=False)
    role = Column(Enum('admin', 'editor', 'viewer', name='invitation_role_enum'), nullable=False)
    invited_by = Column(String(255), ForeignKey('auth_user.username'), nullable=False)
    invited_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    accepted = Column(Boolean, default=False)
    accepted_at = Column(DateTime(timezone=True))
    
    # Relationships
    workspace = relationship("Workspace", back_populates="invitations")
    inviter = relationship("AuthUser")
    
    __table_args__ = (
        Index('idx_workspace_invitations_email', 'email'),
        Index('idx_workspace_invitations_token', 'token'),
    )


# RBAC Models
class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSON, nullable=False)  # JSON object with permission keys
    is_system = Column(Boolean, default=False)  # System roles cannot be deleted
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    __tablename__ = 'user_roles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('auth_user.id', ondelete='CASCADE'), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id', ondelete='CASCADE'), nullable=False)
    assigned_by = Column(String(255), ForeignKey('auth_user.username'))
    assigned_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("AuthUser", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")
    assigner = relationship("AuthUser", foreign_keys=[assigned_by])
    
    __table_args__ = (
        Index('idx_user_roles_user', 'user_id'),
        Index('idx_user_roles_role', 'role_id'),
        {'schema': 'rbac'},
    )


class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('auth_user.id', ondelete='SET NULL'))
    username = Column(String(255))
    action = Column(String(100), nullable=False)  # e.g., 'login', 'create_workspace', 'delete_checkpoint'
    resource_type = Column(String(100))  # e.g., 'workspace', 'checkpoint', 'user'
    resource_id = Column(String(255))
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    details = Column(JSON)  # Additional context about the action
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("AuthUser")
    
    __table_args__ = (
        Index('idx_audit_logs_user', 'user_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
    )


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.vault = VaultSecretManager()
        self.connection_string = connection_string or self._get_connection_string()
        self.engine = create_engine(
            self.connection_string,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initialize_database()
    
    def _get_connection_string(self) -> str:
        """Get database connection string from Vault or environment."""
        # Try to get from Vault first
        db_url = self.vault.get_secret('database_url')
        if db_url:
            return db_url
        
        # Fallback to environment variables
        db_user = os.getenv('POSTGRES_USER', 'forge')
        db_password = os.getenv('POSTGRES_PASSWORD', 'password')
        db_host = os.getenv('POSTGRES_HOST', 'localhost')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DB', 'forge_auth')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def _initialize_database(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        
        # Create default roles if they don't exist
        self._create_default_roles()
    
    def _create_default_roles(self):
        """Create default RBAC roles."""
        session = self.SessionLocal()
        try:
            default_roles = [
                {
                    'name': 'superadmin',
                    'description': 'Full system access',
                    'permissions': json.dumps({
                        'users': ['create', 'read', 'update', 'delete'],
                        'workspaces': ['create', 'read', 'update', 'delete', 'manage_members'],
                        'checkpoints': ['create', 'read', 'update', 'delete'],
                        'training': ['create', 'read', 'update', 'delete'],
                        'system': ['manage_roles', 'view_audit_logs']
                    }),
                    'is_system': True
                },
                {
                    'name': 'admin',
                    'description': 'Administrative access',
                    'permissions': json.dumps({
                        'users': ['read', 'update'],
                        'workspaces': ['create', 'read', 'update', 'delete', 'manage_members'],
                        'checkpoints': ['create', 'read', 'update', 'delete'],
                        'training': ['create', 'read', 'update', 'delete']
                    }),
                    'is_system': True
                },
                {
                    'name': 'user',
                    'description': 'Standard user access',
                    'permissions': json.dumps({
                        'workspaces': ['create', 'read'],
                        'checkpoints': ['create', 'read'],
                        'training': ['create', 'read']
                    }),
                    'is_system': True
                }
            ]
            
            for role_data in default_roles:
                existing_role = session.query(Role).filter_by(name=role_data['name']).first()
                if not existing_role:
                    role = Role(**role_data)
                    session.add(role)
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error creating default roles: {e}")
        finally:
            session.close()
    
    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()
    
    def get_connection(self):
        """Get a raw database connection (for backward compatibility)."""
        return self.engine.raw_connection()


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def generate_bootstrap_password() -> str:
    """Generate a 4-word diceware passphrase and persist it to disk.

    The passphrase is written to ``_BOOTSTRAP_PW_PATH`` so that it
    survives server restarts (the DB only stores the *hash*).  On
    subsequent calls / restarts, the persisted value is returned.
    """
    global _bootstrap_password

    # 1. Already cached in this process?
    if _bootstrap_password is not None:
        return _bootstrap_password

    # 2. Already persisted from a previous run?
    if _BOOTSTRAP_PW_PATH.is_file():
        _bootstrap_password = _BOOTSTRAP_PW_PATH.read_text().strip()
        if _bootstrap_password:
            return _bootstrap_password

    # 3. First-ever startup — generate a fresh passphrase.
    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the *same* passphrase is used if the server restarts
    # before the user changes the password.
    ensure_dir(_BOOTSTRAP_PW_PATH.parent)
    _BOOTSTRAP_PW_PATH.write_text(_bootstrap_password)

    return _bootstrap_password


def get_bootstrap_password() -> Optional[str]:
    """Return the cached bootstrap password, or None if not yet generated."""
    return _bootstrap_password


def clear_bootstrap_password() -> None:
    """Delete the persisted bootstrap password file (called after password change)."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper used for refresh token storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_connection():
    """Get a connection to the PostgreSQL database, creating tables if needed."""
    db_manager = get_db_manager()
    return db_manager.get_connection()


def get_session():
    """Get a SQLAlchemy session for database operations."""
    db_manager = get_db_manager()
    return db_manager.get_session()


def log_audit_event(
    user_id: Optional[int],
    username: Optional[str],
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Log an audit event to the database."""
    session = get_session()
    try:
        audit_log = AuditLog(
            user_id=user_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            success=success,
            error_message=error_message
        )
        session.add(audit_log)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error logging audit event: {e}")
    finally:
        session.close()


def check_permission(user_id: int, resource: str, action: str) -> bool:
    """Check if a user has permission to perform an action on a resource."""
    session = get_session()
    try:
        # Get user's roles
        user_roles = session.query(UserRole).filter_by(user_id=user_id).all()
        
        for user_role in user_roles:
            role = session.query(Role).filter_by(id=user_role.role_id).first()
            if role and role.permissions:
                permissions = json.loads(role.permissions) if isinstance(role.permissions, str) else role.permissions
                if resource in permissions and action in permissions[resource]:
                    return True
        
        return False
    except Exception as e:
        print(f"Error checking permission: {e}")
        return False
    finally:
        session.close()


def assign_role_to_user(user_id: int, role_name: str, assigned_by: str) -> bool:
    """Assign a role to a user."""
    session = get_session()
    try:
        role = session.query(Role).filter_by(name=role_name).first()
        if not role:
            return False
        
        # Check if user already has this role
        existing = session.query(UserRole).filter_by(
            user_id=user_id,
            role_id=role.id
        ).first()
        
        if existing:
            return True  # Already assigned
        
        user_role = UserRole(
            user_id=user_id,
            role_id=role.id,
            assigned_by=assigned_by
        )
        session.add(user_role)
        session.commit()
        
        # Log the role assignment
        log_audit_event(
            user_id=user_id,
            username=assigned_by,
            action='assign_role',
            resource_type='user',
            resource_id=str(user_id),
            details={'role': role_name}
        )
        
        return True
    except Exception as e:
        session.rollback()
        print(f"Error assigning role: {e}")
        return False
    finally:
        session.close()


def get_user_permissions(user_id: int) -> Dict[str, List[str]]:
    """Get all permissions for a user based on their roles."""
    session = get_session()
    try:
        user_roles = session.query(UserRole).filter_by(user_id=user_id).all()
        all_permissions = {}
        
        for user_role in user_roles:
            role = session.query(Role).filter_by(id=user_role.role_id).first()
            if role and role.permissions:
                permissions = json.loads(role.permissions) if isinstance(role.permissions, str) else role.permissions
                for resource, actions in permissions.items():
                    if resource not in all_permissions:
                        all_permissions[resource] = []
                    all_permissions[resource].extend(actions)
        
        # Remove duplicates
        for resource in all_permissions:
            all_permissions[resource] = list(set(all_permissions[resource]))
        
        return all_permissions
    except Exception as e:
        print(f"Error getting user permissions: {e}")
        return {}
    finally:
        session.close()


def get_jwt_secret() -> str:
    """Get JWT secret from Vault or generate and store a new one."""
    vault = VaultSecretManager()
    
    # Try to get from Vault
    secret = vault.get_secret('jwt_secret')
    if secret:
        return secret
    
    # Generate new secret
    new_secret = secrets.token_urlsafe(64)
    
    # Try to store in Vault
    if vault.set_secret('jwt_secret', new_secret):
        return new_secret
    
    # Fallback: store in environment variable
    os.environ['JWT_SECRET'] = new_secret
    return new_secret


# Legacy compatibility functions (these maintain the same interface as the SQLite version)
def create_user(username: str, password_hash: str, password_salt: str, must_change_password: bool = False) -> bool:
    """Create a new user in the database."""
    session = get_session()
    try:
        jwt_secret = get_jwt_secret()
        user = AuthUser(
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
            jwt_secret=jwt_secret,
            must_change_password=must_change_password
        )
        session.add(user)
        session.commit()
        
        # Assign default user role
        assign_role_to_user(user.id, 'user', 'system')
        
        # Log user creation
        log_audit_event(
            user_id=user.id,
            username='system',
            action='create_user',
            resource_type='user',
            resource_id=str(user.id)
        )
        
        return True
    except IntegrityError:
        session.rollback()
        return False
    except Exception as e:
        session.rollback()
        print(f"Error creating user: {e}")
        return False
    finally:
        session.close()


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username."""
    session = get_session()
    try:
        user = session.query(AuthUser).filter_by(username=username).first()
        if user:
            return {
                'id': user.id,
                'username': user.username,
                'password_hash': user.password_hash,
                'password_salt': user.password_salt,
                'jwt_secret': user.jwt_secret,
                'must_change_password': user.must_change_password,
                'email': user.email,
                'is_active': user.is_active,
                'created_at': user.created_at,
                'last_login': user.last_login
            }
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None
    finally:
        session.close()


def update_user_password(username: str, password_hash: str, password_salt: str) -> bool:
    """Update user password and clear bootstrap password."""
    session = get_session()
    try:
        user = session.query(AuthUser).filter_by(username=username).first()
        if user:
            user.password_hash = password_hash
            user.password_salt = password_salt
            user.must_change_password = False
            user.updated_at = datetime.now(timezone.utc)
            session.commit()
            
            # Clear bootstrap password if this is the admin user
            if username == DEFAULT_ADMIN_USERNAME:
                clear_bootstrap_password()
            
            # Log password change
            log_audit_event(
                user_id=user.id,
                username=username,
                action='change_password',
                resource_type='user',
                resource_id=str(user.id)
            )
            
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error updating user password: {e}")
        return False
    finally:
        session.close()


def store_refresh_token(token_hash: str, username: str, expires_at: str, user_agent: str = None, ip_address: str = None) -> bool:
    """Store a refresh token in the database."""
    session = get_session()
    try:
        refresh_token = RefreshToken(
            token_hash=token_hash,
            username=username,
            expires_at=datetime.fromisoformat(expires_at.replace('Z', '+00:00')),
            user_agent=user_agent,
            ip_address=ip_address
        )
        session.add(refresh_token)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error storing refresh token: {e}")
        return False
    finally:
        session.close()


def validate_refresh_token(token_hash: str, username: str) -> bool:
    """Validate a refresh token."""
    session = get_session()
    try:
        token = session.query(RefreshToken).filter_by(
            token_hash=token_hash,
            username=username,
            revoked=False
        ).first()
        
        if token and token.expires_at > datetime.now(timezone.utc):
            return True
        return False
    except Exception as e:
        print(f"Error validating refresh token: {e}")
        return False
    finally:
        session.close()


def revoke_refresh_token(token_hash: str) -> bool:
    """Revoke a refresh token."""
    session = get_session()
    try:
        token = session.query(RefreshToken).filter_by(token_hash=token_hash).first()
        if token:
            token.revoked = True
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error revoking refresh token: {e}")
        return False
    finally:
        session.close()


def cleanup_expired_tokens() -> int:
    """Remove expired refresh tokens."""
    session = get_session()
    try:
        expired = session.query(RefreshToken).filter(
            RefreshToken.expires_at < datetime.now(timezone.utc)
        ).delete()
        session.commit()
        return expired
    except Exception as e:
        session.rollback()
        print(f"Error cleaning up expired tokens: {e}")
        return 0
    finally:
        session.close()


def create_workspace(name: str, description: str, created_by: str, is_public: bool = False, max_members: int = 10) -> Optional[int]:
    """Create a new workspace."""
    session = get_session()
    try:
        workspace = Workspace(
            name=name,
            description=description,
            created_by=created_by,
            is_public=is_public,
            max_members=max_members
        )
        session.add(workspace)
        session.flush()  # Get the ID
        
        # Add creator as owner
        member = WorkspaceMember(
            workspace_id=workspace.id,
            username=created_by,
            role='owner'
        )
        session.add(member)
        session.commit()
        
        # Log workspace creation
        log_audit_event(
            user_id=None,
            username=created_by,
            action='create_workspace',
            resource_type='workspace',
            resource_id=str(workspace.id),
            details={'name': name, 'is_public': is_public}
        )
        
        return workspace.id
    except Exception as e:
        session.rollback()
        print(f"Error creating workspace: {e}")
        return None
    finally:
        session.close()


def get_user_workspaces(username: str) -> List[Dict[str, Any]]:
    """Get all workspaces a user is a member of."""
    session = get_session()
    try:
        memberships = session.query(WorkspaceMember).filter_by(username=username).all()
        workspaces = []
        
        for membership in memberships:
            workspace = session.query(Workspace).filter_by(id=membership.workspace_id).first()
            if workspace and workspace.is_active:
                workspaces.append({
                    'id': workspace.id,
                    'name': workspace.name,
                    'description': workspace.description,
                    'created_by': workspace.created_by,
                    'created_at': workspace.created_at,
                    'is_public': workspace.is_public,
                    'role': membership.role,
                    'member_count': session.query(WorkspaceMember).filter_by(workspace_id=workspace.id).count()
                })
        
        return workspaces
    except Exception as e:
        print(f"Error getting user workspaces: {e}")
        return []
    finally:
        session.close()


def add_workspace_member(workspace_id: int, username: str, role: str, invited_by: str) -> bool:
    """Add a member to a workspace."""
    session = get_session()
    try:
        # Check if workspace exists
        workspace = session.query(Workspace).filter_by(id=workspace_id).first()
        if not workspace:
            return False
        
        # Check if user is already a member
        existing = session.query(WorkspaceMember).filter_by(
            workspace_id=workspace_id,
            username=username
        ).first()
        
        if existing:
            return False
        
        # Check member limit
        current_members = session.query(WorkspaceMember).filter_by(workspace_id=workspace_id).count()
        if current_members >= workspace.max_members:
            return False
        
        member = WorkspaceMember(
            workspace_id=workspace_id,
            username=username,
            role=role
        )
        session.add(member)
        session.commit()
        
        # Log member addition
        log_audit_event(
            user_id=None,
            username=invited_by,
            action='add_workspace_member',
            resource_type='workspace',
            resource_id=str(workspace_id),
            details={'added_user': username, 'role': role}
        )
        
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding workspace member: {e}")
        return False
    finally:
        session.close()


def create_training_run(workspace_id: int, name: str, started_by: str, 
                       checkpoint_id: Optional[int] = None, 
                       description: str = None,
                       config: Dict[str, Any] = None) -> Optional[int]:
    """Create a new training run."""
    session = get_session()
    try:
        run = TrainingRun(
            workspace_id=workspace_id,
            checkpoint_id=checkpoint_id,
            name=name,
            description=description,
            status='pending',
            started_by=started_by,
            config_json=config
        )
        session.add(run)
        session.flush()
        
        # Log training run creation
        log_audit_event(
            user_id=None,
            username=started_by,
            action='create_training_run',
            resource_type='training_run',
            resource_id=str(run.id),
            details={'workspace_id': workspace_id, 'name': name}
        )
        
        session.commit()
        return run.id
    except Exception as e:
        session.rollback()
        print(f"Error creating training run: {e}")
        return None
    finally:
        session.close()


def update_training_run_status(run_id: int, status: str, error_message: str = None) -> bool:
    """Update training run status."""
    session = get_session()
    try:
        run = session.query(TrainingRun).filter_by(id=run_id).first()
        if run:
            run.status = status
            if status in ['completed', 'failed']:
                run.ended_at = datetime.now(timezone.utc)
            if error_message:
                run.error_message = error_message
            
            session.commit()
            
            # Log status update
            log_audit_event(
                user_id=None,
                username='system',
                action='update_training_run_status',
                resource_type='training_run',
                resource_id=str(run_id),
                details={'status': status, 'error': error_message}
            )
            
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error updating training run status: {e}")
        return False
    finally:
        session.close()


def record_training_metric(run_id: int, metric_name: str, metric_value: float, 
                          step: int, epoch: int = None, batch: int = None) -> bool:
    """Record a training metric."""
    session = get_session()
    try:
        metric = TrainingMetric(
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            step=step,
            epoch=epoch,
            batch=batch
        )
        session.add(metric)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error recording training metric: {e}")
        return False
    finally:
        session.close()


def get_training_metrics(run_id: int, metric_name: str = None, 
                        limit: int = 1000) -> List[Dict[str, Any]]:
    """Get training metrics for a run."""
    session = get_session()
    try:
        query = session.query(TrainingMetric).filter_by(run_id=run_id)
        
        if metric_name:
            query = query.filter_by(metric_name=metric_name)
        
        metrics = query.order_by(TrainingMetric.step).limit(limit).all()
        
        return [{
            'id': m.id,
            'metric_name': m.metric_name,
            'metric_value': m.metric_value,
            'step': m.step,
            'epoch': m.epoch,
            'batch': m.batch,
            'timestamp': m.timestamp
        } for m in metrics]
    except Exception as e:
        print(f"Error getting training metrics: {e}")
        return []
    finally:
        session.close()


def create_workspace_invitation(workspace_id: int, email: str, role: str, 
                               invited_by: str, expires_hours: int = 168) -> Optional[str]:
    """Create a workspace invitation."""
    session = get_session()
    try:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        
        invitation = WorkspaceInvitation(
            workspace_id=workspace_id,
            email=email,
            role=role,
            invited_by=invited_by,
            expires_at=expires_at,
            token=token
        )
        session.add(invitation)
        session.commit()
        
        # Log invitation creation
        log_audit_event(
            user_id=None,
            username=invited_by,
            action='create_workspace_invitation',
            resource_type='workspace_invitation',
            resource_id=str(invitation.id),
            details={'workspace_id': workspace_id, 'email': email, 'role': role}
        )
        
        return token
    except Exception as e:
        session.rollback()
        print(f"Error creating workspace invitation: {e}")
        return None
    finally:
        session.close()


def accept_workspace_invitation(token: str, username: str) -> bool:
    """Accept a workspace invitation."""
    session = get_session()
    try:
        invitation = session.query(WorkspaceInvitation).filter_by(
            token=token,
            accepted=False
        ).first()
        
        if not invitation or invitation.expires_at < datetime.now(timezone.utc):
            return False
        
        # Add user to workspace
        success = add_workspace_member(
            workspace_id=invitation.workspace_id,
            username=username,
            role=invitation.role,
            invited_by=invitation.invited_by
        )
        
        if success:
            invitation.accepted = True
            invitation.accepted_at = datetime.now(timezone.utc)
            session.commit()
            
            # Log invitation acceptance
            log_audit_event(
                user_id=None,
                username=username,
                action='accept_workspace_invitation',
                resource_type='workspace_invitation',
                resource_id=str(invitation.id),
                details={'workspace_id': invitation.workspace_id}
            )
            
            return True
        
        return False
    except Exception as e:
        session.rollback()
        print(f"Error accepting workspace invitation: {e}")
        return False
    finally:
        session.close()


def get_audit_logs(user_id: Optional[int] = None, action: Optional[str] = None,
                  resource_type: Optional[str] = None, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get audit logs with filtering."""
    session = get_session()
    try:
        query = session.query(AuditLog)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        if action:
            query = query.filter_by(action=action)
        if resource_type:
            query = query.filter_by(resource_type=resource_type)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
        
        return [{
            'id': log.id,
            'user_id': log.user_id,
            'username': log.username,
            'action': log.action,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'timestamp': log.timestamp,
            'ip_address': log.ip_address,
            'user_agent': log.user_agent,
            'details': log.details,
            'success': log.success,
            'error_message': log.error_message
        } for log in logs]
    except Exception as e:
        print(f"Error getting audit logs: {e}")
        return []
    finally:
        session.close()


# Initialize database on module import
def initialize_database():
    """Initialize the database and create default data."""
    try:
        db_manager = get_db_manager()
        print("Database initialized successfully")
        
        # Create default admin user if it doesn't exist
        session = db_manager.get_session()
        try:
            admin_user = session.query(AuthUser).filter_by(username=DEFAULT_ADMIN_USERNAME).first()
            if not admin_user:
                # Generate bootstrap password
                bootstrap_pw = generate_bootstrap_password()
                
                # Hash the password
                import bcrypt
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(bootstrap_pw.encode('utf-8'), salt).decode('utf-8')
                
                # Create admin user
                create_user(
                    username=DEFAULT_ADMIN_USERNAME,
                    password_hash=password_hash,
                    password_salt=salt.decode('utf-8'),
                    must_change_password=True
                )
                print(f"Created admin user with bootstrap password: {bootstrap_pw}")
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error initializing database: {e}")


# Auto-initialize when module is imported
initialize_database()