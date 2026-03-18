"""
Enterprise Security & Audit Module for SOVEREIGN Studio
Implements secret management integration with HashiCorp Vault
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from functools import lru_cache
import hvac
from hvac.exceptions import VaultError, InvalidPath
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from studio.backend.auth.storage import get_db
from studio.backend.core.data_recipe.jobs.constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class VaultConfig(BaseModel):
    """Configuration for HashiCorp Vault connection"""
    vault_url: str = Field(
        default_factory=lambda: os.getenv("VAULT_URL", "http://localhost:8200")
    )
    vault_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_TOKEN")
    )
    vault_role_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_ROLE_ID")
    )
    vault_secret_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_SECRET_ID")
    )
    vault_namespace: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_NAMESPACE")
    )
    mount_point: str = Field(
        default_factory=lambda: os.getenv("VAULT_MOUNT_POINT", "secret")
    )
    kv_version: int = Field(
        default_factory=lambda: int(os.getenv("VAULT_KV_VERSION", "2"))
    )
    default_lease_ttl: int = Field(
        default_factory=lambda: int(os.getenv("VAULT_DEFAULT_LEASE_TTL", "3600"))
    )
    max_lease_ttl: int = Field(
        default_factory=lambda: int(os.getenv("VAULT_MAX_LEASE_TTL", "86400"))
    )

    @validator('kv_version')
    def validate_kv_version(cls, v):
        if v not in [1, 2]:
            raise ValueError("KV version must be 1 or 2")
        return v


class SecretAuditLog(BaseModel):
    """Audit log entry for secret access"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    action: str  # "read", "write", "delete", "list"
    secret_path: str
    success: bool
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecretsManager:
    """
    Enterprise-grade secrets management with HashiCorp Vault integration.
    Provides secure storage, retrieval, and audit logging for sensitive data.
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        self.config = config or VaultConfig()
        self._client: Optional[hvac.Client] = None
        self._audit_logs: List[SecretAuditLog] = []
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Vault client with configuration"""
        try:
            client_kwargs = {
                "url": self.config.vault_url,
                "namespace": self.config.vault_namespace
            }

            # Authenticate using token or AppRole
            if self.config.vault_token:
                client_kwargs["token"] = self.config.vault_token
            elif self.config.vault_role_id and self.config.vault_secret_id:
                # AppRole authentication will be handled after client creation
                pass
            else:
                raise ValueError(
                    "Either VAULT_TOKEN or both VAULT_ROLE_ID and VAULT_SECRET_ID must be provided"
                )

            self._client = hvac.Client(**client_kwargs)

            # AppRole authentication if token not provided
            if not self.config.vault_token and self.config.vault_role_id:
                self._authenticate_approle()

            # Verify authentication
            if not self._client.is_authenticated():
                raise VaultError("Failed to authenticate with Vault")

            # Configure KV secrets engine
            self._configure_kv_engine()

            logger.info(f"Successfully connected to Vault at {self.config.vault_url}")

        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {str(e)}")
            raise

    def _authenticate_approle(self) -> None:
        """Authenticate using AppRole method"""
        try:
            auth_response = self._client.auth.approle.login(
                role_id=self.config.vault_role_id,
                secret_id=self.config.vault_secret_id
            )
            self._client.token = auth_response["auth"]["client_token"]
            logger.info("Successfully authenticated using AppRole")
        except Exception as e:
            logger.error(f"AppRole authentication failed: {str(e)}")
            raise

    def _configure_kv_engine(self) -> None:
        """Configure KV secrets engine if needed"""
        try:
            # Check if KV engine exists
            mounts = self._client.sys.list_mounted_secrets_engines()
            mount_path = f"{self.config.mount_point}/"

            if mount_path not in mounts["data"]:
                # Enable KV secrets engine
                self._client.sys.enable_secrets_engine(
                    backend_type="kv",
                    path=self.config.mount_point,
                    options={"version": self.config.kv_version}
                )
                logger.info(f"Enabled KV v{self.config.kv_version} secrets engine at {self.config.mount_point}")
        except Exception as e:
            logger.warning(f"Could not verify/configure KV engine: {str(e)}")

    def _log_audit_event(
        self,
        action: str,
        secret_path: str,
        success: bool,
        user_id: Optional[str] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log audit event for secret access"""
        audit_entry = SecretAuditLog(
            user_id=user_id,
            action=action,
            secret_path=secret_path,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self._audit_logs.append(audit_entry)

        # In production, this would also persist to database
        logger.info(
            f"Audit: {action} on {secret_path} by user {user_id} - "
            f"{'SUCCESS' if success else 'FAILED'}"
        )

    def _get_full_path(self, path: str) -> str:
        """Construct full secret path based on KV version"""
        if self.config.kv_version == 2:
            return f"{self.config.mount_point}/data/{path}"
        return f"{self.config.mount_point}/{path}"

    def get_secret(
        self,
        path: str,
        key: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Union[Dict[str, Any], Any, None]:
        """
        Retrieve secret from Vault
        
        Args:
            path: Path to the secret
            key: Specific key within the secret (optional)
            user_id: User requesting the secret (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            Secret value or dictionary of secrets
        """
        try:
            full_path = self._get_full_path(path)

            if self.config.kv_version == 2:
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.config.mount_point
                )
                secret_data = response["data"]["data"]
            else:
                response = self._client.secrets.kv.v1.read_secret(
                    path=path,
                    mount_point=self.config.mount_point
                )
                secret_data = response["data"]

            self._log_audit_event(
                action="read",
                secret_path=path,
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            if key:
                return secret_data.get(key)
            return secret_data

        except InvalidPath:
            self._log_audit_event(
                action="read",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message="Secret not found",
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        except Exception as e:
            self._log_audit_event(
                action="read",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to read secret at {path}: {str(e)}")
            raise

    def set_secret(
        self,
        path: str,
        secret: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        cas_required: bool = False
    ) -> bool:
        """
        Store secret in Vault
        
        Args:
            path: Path to store the secret
            secret: Dictionary of secret data
            user_id: User storing the secret (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            cas_required: Require Check-And-Set operation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.kv_version == 2:
                self._client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=secret,
                    mount_point=self.config.mount_point,
                    cas=0 if cas_required else None
                )
            else:
                self._client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=secret,
                    mount_point=self.config.mount_point
                )

            self._log_audit_event(
                action="write",
                secret_path=path,
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            logger.info(f"Successfully stored secret at {path}")
            return True

        except Exception as e:
            self._log_audit_event(
                action="write",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to store secret at {path}: {str(e)}")
            return False

    def delete_secret(
        self,
        path: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        versions: Optional[List[int]] = None
    ) -> bool:
        """
        Delete secret from Vault
        
        Args:
            path: Path to the secret
            user_id: User deleting the secret (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            versions: Specific versions to delete (KV v2 only)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.kv_version == 2:
                if versions:
                    # Delete specific versions
                    self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                        path=path,
                        mount_point=self.config.mount_point
                    )
                else:
                    # Delete latest version
                    self._client.secrets.kv.v2.delete_latest_version_of_secret(
                        path=path,
                        mount_point=self.config.mount_point
                    )
            else:
                self._client.secrets.kv.v1.delete_secret(
                    path=path,
                    mount_point=self.config.mount_point
                )

            self._log_audit_event(
                action="delete",
                secret_path=path,
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            logger.info(f"Successfully deleted secret at {path}")
            return True

        except Exception as e:
            self._log_audit_event(
                action="delete",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to delete secret at {path}: {str(e)}")
            return False

    def list_secrets(
        self,
        path: str = "",
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> List[str]:
        """
        List secrets at a given path
        
        Args:
            path: Path to list (empty for root)
            user_id: User listing secrets (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            List of secret names
        """
        try:
            if self.config.kv_version == 2:
                response = self._client.secrets.kv.v2.list_secrets(
                    path=path,
                    mount_point=self.config.mount_point
                )
            else:
                response = self._client.secrets.kv.v1.list_secrets(
                    path=path,
                    mount_point=self.config.mount_point
                )

            self._log_audit_event(
                action="list",
                secret_path=path,
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            return response.get("data", {}).get("keys", [])

        except InvalidPath:
            self._log_audit_event(
                action="list",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message="Path not found",
                ip_address=ip_address,
                user_agent=user_agent
            )
            return []
        except Exception as e:
            self._log_audit_event(
                action="list",
                secret_path=path,
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to list secrets at {path}: {str(e)}")
            raise

    def rotate_secret(
        self,
        path: str,
        new_secret: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Rotate a secret by creating new version and optionally deleting old ones
        
        Args:
            path: Path to the secret
            new_secret: New secret data
            user_id: User rotating the secret (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, get current version if exists
            current = self.get_secret(path, user_id=user_id)
            
            # Store new version
            success = self.set_secret(
                path=path,
                secret=new_secret,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            if success and current and self.config.kv_version == 2:
                # Optionally delete old versions (keep last 3)
                metadata = self._client.secrets.kv.v2.read_secret_metadata(
                    path=path,
                    mount_point=self.config.mount_point
                )
                versions = metadata.get("data", {}).get("versions", {})
                if len(versions) > 3:
                    # Delete oldest versions, keep last 3
                    version_numbers = sorted(
                        [int(v) for v in versions.keys()],
                        reverse=True
                    )[3:]
                    if version_numbers:
                        self._client.secrets.kv.v2.delete_secret_versions(
                            path=path,
                            versions=version_numbers,
                            mount_point=self.config.mount_point
                        )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate secret at {path}: {str(e)}")
            return False

    def get_audit_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[SecretAuditLog]:
        """
        Retrieve audit logs with filtering
        
        Args:
            start_time: Filter logs after this time
            end_time: Filter logs before this time
            user_id: Filter by user ID
            action: Filter by action type
            limit: Maximum number of logs to return
            
        Returns:
            List of audit log entries
        """
        filtered_logs = self._audit_logs.copy()
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        if action:
            filtered_logs = [log for log in filtered_logs if log.action == action]
        
        # Sort by timestamp descending and limit
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]

    def health_check(self) -> Dict[str, Any]:
        """
        Check Vault health and connection status
        
        Returns:
            Health status dictionary
        """
        try:
            health = self._client.sys.read_health_status()
            return {
                "status": "healthy",
                "vault_version": health.get("version"),
                "cluster_id": health.get("cluster_id"),
                "cluster_name": health.get("cluster_name"),
                "initialized": health.get("initialized"),
                "sealed": health.get("sealed"),
                "standby": health.get("standby"),
                "performance_standby": health.get("performance_standby"),
                "replication_performance_mode": health.get("replication_performance_mode"),
                "replication_dr_mode": health.get("replication_dr_mode"),
                "server_time_utc": health.get("server_time_utc"),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Vault health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def create_transit_key(
        self,
        key_name: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Create a transit encryption key for encrypting/decrypting data
        
        Args:
            key_name: Name of the transit key
            user_id: User creating the key (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.secrets.transit.create_key(
                name=key_name,
                mount_point="transit"
            )
            
            self._log_audit_event(
                action="create_transit_key",
                secret_path=f"transit/keys/{key_name}",
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            logger.info(f"Created transit key: {key_name}")
            return True
            
        except Exception as e:
            self._log_audit_event(
                action="create_transit_key",
                secret_path=f"transit/keys/{key_name}",
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to create transit key {key_name}: {str(e)}")
            return False

    def encrypt_data(
        self,
        key_name: str,
        plaintext: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Encrypt data using transit encryption
        
        Args:
            key_name: Name of the transit key
            plaintext: Data to encrypt
            user_id: User encrypting data (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            Encrypted ciphertext or None on failure
        """
        try:
            # Convert plaintext to base64
            import base64
            plaintext_b64 = base64.b64encode(plaintext.encode()).decode()
            
            response = self._client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=plaintext_b64,
                mount_point="transit"
            )
            
            ciphertext = response["data"]["ciphertext"]
            
            self._log_audit_event(
                action="encrypt",
                secret_path=f"transit/keys/{key_name}",
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return ciphertext
            
        except Exception as e:
            self._log_audit_event(
                action="encrypt",
                secret_path=f"transit/keys/{key_name}",
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to encrypt data with key {key_name}: {str(e)}")
            return None

    def decrypt_data(
        self,
        key_name: str,
        ciphertext: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Decrypt data using transit encryption
        
        Args:
            key_name: Name of the transit key
            ciphertext: Encrypted data to decrypt
            user_id: User decrypting data (for audit)
            ip_address: Client IP address (for audit)
            user_agent: Client user agent (for audit)
            
        Returns:
            Decrypted plaintext or None on failure
        """
        try:
            response = self._client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext,
                mount_point="transit"
            )
            
            plaintext_b64 = response["data"]["plaintext"]
            
            # Decode from base64
            import base64
            plaintext = base64.b64decode(plaintext_b64).decode()
            
            self._log_audit_event(
                action="decrypt",
                secret_path=f"transit/keys/{key_name}",
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return plaintext
            
        except Exception as e:
            self._log_audit_event(
                action="decrypt",
                secret_path=f"transit/keys/{key_name}",
                success=False,
                user_id=user_id,
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            logger.error(f"Failed to decrypt data with key {key_name}: {str(e)}")
            return None


# Singleton instance for application-wide use
_secrets_manager_instance: Optional[SecretsManager] = None


@lru_cache(maxsize=1)
def get_secrets_manager() -> SecretsManager:
    """
    Get or create singleton SecretsManager instance
    
    Returns:
        SecretsManager instance
    """
    global _secrets_manager_instance
    if _secrets_manager_instance is None:
        _secrets_manager_instance = SecretsManager()
    return _secrets_manager_instance


def initialize_secrets_manager(config: Optional[VaultConfig] = None) -> SecretsManager:
    """
    Initialize SecretsManager with custom configuration
    
    Args:
        config: Vault configuration
        
    Returns:
        Initialized SecretsManager instance
    """
    global _secrets_manager_instance
    _secrets_manager_instance = SecretsManager(config)
    return _secrets_manager_instance


# Integration with existing authentication system
def get_user_secrets(
    user_id: str,
    secret_path: str,
    db: Session = None
) -> Optional[Dict[str, Any]]:
    """
    Get secrets for a specific user with RBAC checks
    
    Args:
        user_id: User ID
        secret_path: Path to the secret
        db: Database session
        
    Returns:
        Secret data or None if not authorized
    """
    from studio.backend.auth.storage import User
    from studio.backend.auth.authentication import get_current_user_permissions
    
    if db is None:
        db = next(get_db())
    
    # Check user permissions for this secret path
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"User {user_id} not found")
        return None
    
    # Get user permissions
    permissions = get_current_user_permissions(user_id, db)
    
    # Check if user has permission to read this secret path
    # This is a simplified example - in production, implement proper RBAC
    allowed_paths = permissions.get("secrets", {}).get("read", [])
    if not any(secret_path.startswith(path) for path in allowed_paths):
        logger.warning(f"User {user_id} not authorized to read {secret_path}")
        return None
    
    # Get the secret
    secrets_manager = get_secrets_manager()
    return secrets_manager.get_secret(
        path=secret_path,
        user_id=user_id
    )


def store_user_secret(
    user_id: str,
    secret_path: str,
    secret_data: Dict[str, Any],
    db: Session = None
) -> bool:
    """
    Store secrets for a specific user with RBAC checks
    
    Args:
        user_id: User ID
        secret_path: Path to store the secret
        secret_data: Secret data to store
        db: Database session
        
    Returns:
        True if successful, False otherwise
    """
    from studio.backend.auth.storage import User
    from studio.backend.auth.authentication import get_current_user_permissions
    
    if db is None:
        db = next(get_db())
    
    # Check user permissions for this secret path
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"User {user_id} not found")
        return False
    
    # Get user permissions
    permissions = get_current_user_permissions(user_id, db)
    
    # Check if user has permission to write this secret path
    allowed_paths = permissions.get("secrets", {}).get("write", [])
    if not any(secret_path.startswith(path) for path in allowed_paths):
        logger.warning(f"User {user_id} not authorized to write {secret_path}")
        return False
    
    # Store the secret
    secrets_manager = get_secrets_manager()
    return secrets_manager.set_secret(
        path=secret_path,
        secret=secret_data,
        user_id=user_id
    )


# Export main classes and functions
__all__ = [
    "SecretsManager",
    "VaultConfig",
    "SecretAuditLog",
    "get_secrets_manager",
    "initialize_secrets_manager",
    "get_user_secrets",
    "store_user_secret"
]