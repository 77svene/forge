"""Enterprise Security & Audit Logger with PostgreSQL, RBAC, and Vault Integration."""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

import hvac
from pydantic import BaseModel, Field

# Import existing modules for integration
from studio.backend.auth.authentication import get_current_user, User
from studio.backend.auth.storage import get_db_session

logger = logging.getLogger(__name__)
Base = declarative_base()


class AuditAction(str, Enum):
    """Enumeration of auditable actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    EXPORT = "export"
    IMPORT = "import"
    EXECUTE = "execute"
    GRANT = "grant"
    REVOKE = "revoke"
    APPROVE = "approve"
    REJECT = "reject"


class ResourceType(str, Enum):
    """Enumeration of resource types for RBAC."""
    USER = "user"
    PROJECT = "project"
    MODEL = "model"
    DATASET = "dataset"
    JOB = "job"
    SECRET = "secret"
    API_KEY = "api_key"
    SYSTEM = "system"
    TEAM = "team"
    ROLE = "role"
    PERMISSION = "permission"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLog(Base):
    """SQLAlchemy model for audit logs in PostgreSQL."""
    __tablename__ = "audit_logs"
    __table_args__ = {
        "schema": "security",
        "comment": "Enterprise audit trail for compliance and security monitoring"
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    user_id = Column(String(255), index=True)
    user_email = Column(String(255))
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    resource_type = Column(SQLEnum(ResourceType), nullable=False, index=True)
    resource_id = Column(String(255), index=True)
    resource_name = Column(String(255))
    severity = Column(SQLEnum(AuditSeverity), default=AuditSeverity.LOW)
    ip_address = Column(String(45))  # IPv6 max length
    user_agent = Column(String(512))
    request_id = Column(String(255), index=True)
    session_id = Column(String(255))
    details = Column(JSON)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    duration_ms = Column(Integer)
    old_value = Column(JSON)
    new_value = Column(JSON)
    metadata_ = Column("metadata", JSON)
    team_id = Column(String(255), index=True)
    organization_id = Column(String(255), index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "action": self.action.value if self.action else None,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "severity": self.severity.value if self.severity else None,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata_,
            "team_id": self.team_id,
            "organization_id": self.organization_id,
        }


class Role(Base):
    """SQLAlchemy model for RBAC roles."""
    __tablename__ = "roles"
    __table_args__ = {"schema": "security"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSON, default=list)
    is_system_role = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    """SQLAlchemy model for user-role assignments with team hierarchy."""
    __tablename__ = "user_roles"
    __table_args__ = (
        {"schema": "security"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("security.roles.id"), nullable=False)
    team_id = Column(String(255), index=True)
    organization_id = Column(String(255), index=True)
    granted_by = Column(String(255))
    granted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)

    # Relationships
    role = relationship("Role", back_populates="user_roles")


class AuditEvent(BaseModel):
    """Pydantic model for audit event validation."""
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    action: AuditAction
    resource_type: ResourceType
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.LOW
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    team_id: Optional[str] = None
    organization_id: Optional[str] = None


class VaultSecretManager:
    """Integration with HashiCorp Vault for secret management."""

    def __init__(self, vault_url: str = None, token: str = None):
        self.vault_url = vault_url or os.getenv("VAULT_URL", "http://localhost:8200")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.client = None

    def connect(self):
        """Establish connection to Vault."""
        try:
            self.client = hvac.Client(url=self.vault_url, token=self.token)
            if not self.client.is_authenticated():
                raise ConnectionError("Failed to authenticate with Vault")
            logger.info("Successfully connected to HashiCorp Vault")
        except Exception as e:
            logger.error(f"Vault connection failed: {str(e)}")
            raise

    def get_secret(self, path: str, key: str = None) -> Union[Dict[str, Any], Any]:
        """Retrieve secret from Vault."""
        if not self.client:
            self.connect()

        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            data = response["data"]["data"]
            return data.get(key) if key else data
        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {str(e)}")
            raise

    def get_database_credentials(self, role: str = "studio") -> Dict[str, str]:
        """Get dynamic database credentials from Vault."""
        if not self.client:
            self.connect()

        try:
            response = self.client.database.generate_credentials(name=role)
            return {
                "username": response["data"]["username"],
                "password": response["data"]["password"],
            }
        except Exception as e:
            logger.error(f"Failed to generate database credentials: {str(e)}")
            raise


class AuditLogger:
    """Enterprise audit logging system with PostgreSQL backend."""

    def __init__(self, database_url: str = None, vault_enabled: bool = False):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.SessionLocal = None
        self.vault_manager = VaultSecretManager() if vault_enabled else None
        self._initialize_database()

    def _get_database_url(self) -> str:
        """Get database URL from environment or Vault."""
        if self.vault_manager:
            try:
                creds = self.vault_manager.get_database_credentials()
                base_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/forge")
                # Replace credentials in URL
                if "@" in base_url:
                    protocol, rest = base_url.split("://", 1)
                    _, host_part = rest.split("@", 1)
                    return f"{protocol}://{creds['username']}:{creds['password']}@{host_part}"
            except Exception as e:
                logger.warning(f"Failed to get credentials from Vault: {str(e)}")

        return os.getenv(
            "DATABASE_URL",
            "postgresql://forge:forge@localhost:5432/forge"
        )

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true",
            )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Create schema and tables
            with self.engine.connect() as conn:
                conn.execute("CREATE SCHEMA IF NOT EXISTS security")
                conn.commit()

            Base.metadata.create_all(bind=self.engine)
            logger.info("Audit database initialized successfully")

        except SQLAlchemyError as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_event(self, event: AuditEvent) -> Optional[int]:
        """Log an audit event to PostgreSQL."""
        try:
            with self.get_session() as session:
                audit_log = AuditLog(
                    user_id=event.user_id,
                    user_email=event.user_email,
                    action=event.action,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    resource_name=event.resource_name,
                    severity=event.severity,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    request_id=event.request_id,
                    session_id=event.session_id,
                    details=event.details,
                    success=event.success,
                    error_message=event.error_message,
                    duration_ms=event.duration_ms,
                    old_value=event.old_value,
                    new_value=event.new_value,
                    metadata_=event.metadata,
                    team_id=event.team_id,
                    organization_id=event.organization_id,
                )

                session.add(audit_log)
                session.flush()  # Get the ID
                return audit_log.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to log audit event: {str(e)}")
            # Fallback to file logging if database fails
            self._log_to_fallback(event, str(e))
            return None

    def _log_to_fallback(self, event: AuditEvent, error: str):
        """Fallback logging when database is unavailable."""
        fallback_logger = logging.getLogger("audit_fallback")
        fallback_data = event.dict()
        fallback_data["error"] = error
        fallback_data["fallback_timestamp"] = datetime.now(timezone.utc).isoformat()
        fallback_logger.error(json.dumps(fallback_data))

    def log_from_request(
        self,
        request: Any,
        action: AuditAction,
        resource_type: ResourceType,
        resource_id: str = None,
        resource_name: str = None,
        success: bool = True,
        error_message: str = None,
        details: Dict[str, Any] = None,
        severity: AuditSeverity = AuditSeverity.LOW,
        old_value: Dict[str, Any] = None,
        new_value: Dict[str, Any] = None,
    ) -> Optional[int]:
        """Convenience method to log from HTTP request context."""
        user = get_current_user(request) if hasattr(request, 'user') else None
        user_id = str(user.id) if user else None
        user_email = user.email if user else None

        # Extract request metadata
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent") if hasattr(request, 'headers') else None
        request_id = request.headers.get("x-request-id") if hasattr(request, 'headers') else None
        session_id = request.cookies.get("session_id") if hasattr(request, 'cookies') else None

        event = AuditEvent(
            user_id=user_id,
            user_email=user_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            session_id=session_id,
            success=success,
            error_message=error_message,
            details=details,
            old_value=old_value,
            new_value=new_value,
            metadata={"source": "http_request"},
        )

        return self.log_event(event)

    def _get_client_ip(self, request: Any) -> str:
        """Extract client IP from request headers."""
        if hasattr(request, 'headers'):
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            return request.headers.get("x-real-ip", request.client.host if hasattr(request, 'client') else None)
        return None

    def query_logs(
        self,
        user_id: str = None,
        action: AuditAction = None,
        resource_type: ResourceType = None,
        resource_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        team_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters."""
        try:
            with self.get_session() as session:
                query = session.query(AuditLog)

                if user_id:
                    query = query.filter(AuditLog.user_id == user_id)
                if action:
                    query = query.filter(AuditLog.action == action)
                if resource_type:
                    query = query.filter(AuditLog.resource_type == resource_type)
                if resource_id:
                    query = query.filter(AuditLog.resource_id == resource_id)
                if start_time:
                    query = query.filter(AuditLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(AuditLog.timestamp <= end_time)
                if team_id:
                    query = query.filter(AuditLog.team_id == team_id)

                query = query.order_by(AuditLog.timestamp.desc())
                query = query.limit(limit).offset(offset)

                logs = query.all()
                return [log.to_dict() for log in logs]

        except SQLAlchemyError as e:
            logger.error(f"Failed to query audit logs: {str(e)}")
            return []

    def get_user_activity(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent activity for a specific user."""
        from datetime import timedelta
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        return self.query_logs(user_id=user_id, start_time=start_time)

    def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        action: AuditAction,
        resource_id: str = None,
        team_id: str = None,
    ) -> bool:
        """Check if user has permission to perform action on resource."""
        try:
            with self.get_session() as session:
                # Get active roles for user
                user_roles = session.query(UserRole).filter(
                    UserRole.user_id == user_id,
                    UserRole.is_active == True,
                    (UserRole.expires_at == None) | (UserRole.expires_at > datetime.now(timezone.utc))
                )

                if team_id:
                    user_roles = user_roles.filter(UserRole.team_id == team_id)

                user_roles = user_roles.all()

                # Check permissions in each role
                for user_role in user_roles:
                    role = session.query(Role).filter(Role.id == user_role.role_id).first()
                    if role and role.permissions:
                        for permission in role.permissions:
                            # Permission format: "resource_type:action" or "resource_type:action:resource_id"
                            parts = permission.split(":")
                            if len(parts) >= 2:
                                perm_resource = parts[0]
                                perm_action = parts[1]
                                perm_resource_id = parts[2] if len(parts) > 2 else None

                                if (perm_resource == resource_type.value and 
                                    perm_action == action.value and
                                    (perm_resource_id is None or perm_resource_id == resource_id)):
                                    return True

                return False

        except SQLAlchemyError as e:
            logger.error(f"Permission check failed: {str(e)}")
            return False

    def create_role(
        self,
        name: str,
        permissions: List[str],
        description: str = None,
        is_system_role: bool = False,
    ) -> Optional[int]:
        """Create a new RBAC role."""
        try:
            with self.get_session() as session:
                role = Role(
                    name=name,
                    description=description,
                    permissions=permissions,
                    is_system_role=is_system_role,
                )
                session.add(role)
                session.flush()
                return role.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to create role: {str(e)}")
            return None

    def assign_role_to_user(
        self,
        user_id: str,
        role_id: int,
        granted_by: str = None,
        team_id: str = None,
        organization_id: str = None,
        expires_at: datetime = None,
    ) -> bool:
        """Assign a role to a user."""
        try:
            with self.get_session() as session:
                user_role = UserRole(
                    user_id=user_id,
                    role_id=role_id,
                    granted_by=granted_by,
                    team_id=team_id,
                    organization_id=organization_id,
                    expires_at=expires_at,
                )
                session.add(user_role)

                # Log the role assignment
                self.log_event(AuditEvent(
                    user_id=granted_by,
                    action=AuditAction.GRANT,
                    resource_type=ResourceType.ROLE,
                    resource_id=str(role_id),
                    resource_name=f"Role assignment to user {user_id}",
                    details={"user_id": user_id, "role_id": role_id, "team_id": team_id},
                    team_id=team_id,
                    organization_id=organization_id,
                ))

                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to assign role: {str(e)}")
            return False

    def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        organization_id: str = None,
    ) -> Dict[str, Any]:
        """Generate compliance report for audit logs."""
        try:
            with self.get_session() as session:
                query = session.query(AuditLog).filter(
                    AuditLog.timestamp.between(start_date, end_date)
                )

                if organization_id:
                    query = query.filter(AuditLog.organization_id == organization_id)

                total_events = query.count()
                successful_events = query.filter(AuditLog.success == True).count()
                failed_events = query.filter(AuditLog.success == False).count()

                # Group by action
                action_counts = {}
                for action in AuditAction:
                    count = query.filter(AuditLog.action == action).count()
                    if count > 0:
                        action_counts[action.value] = count

                # Group by resource type
                resource_counts = {}
                for resource in ResourceType:
                    count = query.filter(AuditLog.resource_type == resource).count()
                    if count > 0:
                        resource_counts[resource.value] = count

                # Critical events
                critical_events = query.filter(
                    AuditLog.severity == AuditSeverity.CRITICAL
                ).count()

                return {
                    "report_period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                    },
                    "summary": {
                        "total_events": total_events,
                        "successful_events": successful_events,
                        "failed_events": failed_events,
                        "success_rate": successful_events / total_events if total_events > 0 else 0,
                        "critical_events": critical_events,
                    },
                    "by_action": action_counts,
                    "by_resource_type": resource_counts,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to generate compliance report: {str(e)}")
            return {"error": str(e)}


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        vault_enabled = os.getenv("VAULT_ENABLED", "false").lower() == "true"
        _audit_logger = AuditLogger(vault_enabled=vault_enabled)
    return _audit_logger


def audit_event(
    action: AuditAction,
    resource_type: ResourceType,
    resource_id: str = None,
    resource_name: str = None,
    success: bool = True,
    error_message: str = None,
    details: Dict[str, Any] = None,
    severity: AuditSeverity = AuditSeverity.LOW,
):
    """Decorator for automatic audit logging of function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)
            error = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                logger = get_audit_logger()

                # Try to extract user from function arguments
                user_id = None
                for arg in args:
                    if hasattr(arg, 'user'):
                        user_id = str(arg.user.id) if arg.user else None
                        break

                logger.log_event(AuditEvent(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_name=resource_name,
                    success=success and error is None,
                    error_message=error or error_message,
                    details=details,
                    severity=severity,
                    duration_ms=int(duration),
                    metadata={"function": func.__name__, "module": func.__module__},
                ))

        return wrapper
    return decorator


# Integration with existing CLI
def setup_audit_commands(cli_app):
    """Add audit-related commands to CLI."""
    @cli_app.command()
    def audit_report(
        start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
        end_date: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
        organization_id: str = typer.Option(None, help="Filter by organization"),
    ):
        """Generate audit compliance report."""
        from datetime import datetime
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        logger = get_audit_logger()
        report = logger.get_compliance_report(start, end, organization_id)
        print(json.dumps(report, indent=2))

    @cli_app.command()
    def check_permission(
        user_id: str = typer.Argument(..., help="User ID to check"),
        resource: ResourceType = typer.Argument(..., help="Resource type"),
        action: AuditAction = typer.Argument(..., help="Action to check"),
        resource_id: str = typer.Option(None, help="Specific resource ID"),
        team_id: str = typer.Option(None, help="Team context"),
    ):
        """Check user permissions."""
        logger = get_audit_logger()
        has_permission = logger.check_permission(user_id, resource, action, resource_id, team_id)
        print(f"Permission {'GRANTED' if has_permission else 'DENIED'}")


# Initialize on import if configured
if os.getenv("AUTO_INIT_AUDIT", "true").lower() == "true":
    try:
        get_audit_logger()
    except Exception as e:
        logger.warning(f"Failed to auto-initialize audit logger: {str(e)}")