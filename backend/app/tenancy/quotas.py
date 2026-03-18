"""
Multi-tenant Isolation with Resource Quotas for DeerFlow
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Any
from enum import Enum
import logging
from collections import defaultdict
import threading

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be limited per tenant."""
    CONCURRENT_AGENTS = "concurrent_agents"
    COMPUTE_TIME_SECONDS = "compute_time_seconds"
    MEMORY_MB = "memory_mb"
    API_CALLS_PER_MINUTE = "api_calls_per_minute"
    MESSAGES_PER_HOUR = "messages_per_hour"


@dataclass
class ResourceQuota:
    """Resource quota configuration for a tenant."""
    resource_type: ResourceType
    limit: float
    window_seconds: int = 3600  # Default 1 hour window
    burst_limit: Optional[float] = None  # For token bucket burst capacity


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    tenant_id: str
    name: str
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: float, refill_rate: float, burst_capacity: Optional[float] = None):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            burst_capacity: Maximum burst tokens (if different from capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.burst_capacity = burst_capacity or capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.RLock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.burst_capacity, self.tokens + new_tokens)
        self.last_refill = now
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from bucket.
        
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def available_tokens(self) -> float:
        """Get number of available tokens."""
        with self._lock:
            self._refill()
            return self.tokens
    
    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate time to wait for tokens to become available."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            deficit = tokens - self.tokens
            return deficit / self.refill_rate


class TenantQuotaManager:
    """
    Manages resource quotas for multiple tenants with token bucket rate limiting.
    """
    
    def __init__(self, default_quotas: Optional[Dict[ResourceType, ResourceQuota]] = None):
        """
        Initialize quota manager.
        
        Args:
            default_quotas: Default quotas for new tenants
        """
        self.tenants: Dict[str, TenantConfig] = {}
        self.token_buckets: Dict[str, Dict[ResourceType, TokenBucket]] = defaultdict(dict)
        self.active_agents: Dict[str, Set[str]] = defaultdict(set)  # tenant_id -> set of agent_ids
        self._lock = threading.RLock()
        
        # Default quotas if none specified
        self.default_quotas = default_quotas or {
            ResourceType.CONCURRENT_AGENTS: ResourceQuota(
                ResourceType.CONCURRENT_AGENTS, 
                limit=10,
                window_seconds=3600
            ),
            ResourceType.COMPUTE_TIME_SECONDS: ResourceQuota(
                ResourceType.COMPUTE_TIME_SECONDS,
                limit=3600,  # 1 hour of compute per hour
                window_seconds=3600,
                burst_limit=7200  # Allow 2 hour burst
            ),
            ResourceType.API_CALLS_PER_MINUTE: ResourceQuota(
                ResourceType.API_CALLS_PER_MINUTE,
                limit=100,
                window_seconds=60
            ),
            ResourceType.MESSAGES_PER_HOUR: ResourceQuota(
                ResourceType.MESSAGES_PER_HOUR,
                limit=1000,
                window_seconds=3600
            )
        }
        
        # Initialize Prometheus metrics if available
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics if available."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            self.metrics_enabled = False
            return
        
        self.metrics_enabled = True
        
        # Quota usage metrics
        self.quota_usage = Gauge(
            'deerflow_quota_usage',
            'Current quota usage by tenant and resource',
            ['tenant_id', 'resource_type']
        )
        
        self.quota_limit = Gauge(
            'deerflow_quota_limit',
            'Configured quota limit by tenant and resource',
            ['tenant_id', 'resource_type']
        )
        
        self.quota_exceeded = Counter(
            'deerflow_quota_exceeded_total',
            'Total number of quota exceeded events',
            ['tenant_id', 'resource_type']
        )
        
        self.active_agents_gauge = Gauge(
            'deerflow_active_agents',
            'Number of active agents per tenant',
            ['tenant_id']
        )
        
        self.tenant_enabled = Gauge(
            'deerflow_tenant_enabled',
            'Whether tenant is enabled (1) or disabled (0)',
            ['tenant_id']
        )
    
    def register_tenant(self, tenant_id: str, name: str, 
                       quotas: Optional[Dict[ResourceType, ResourceQuota]] = None) -> TenantConfig:
        """
        Register a new tenant with specified quotas.
        
        Args:
            tenant_id: Unique tenant identifier
            name: Tenant display name
            quotas: Resource quotas (uses defaults if not specified)
            
        Returns:
            Created TenantConfig
        """
        with self._lock:
            if tenant_id in self.tenants:
                raise ValueError(f"Tenant {tenant_id} already registered")
            
            # Merge with default quotas
            merged_quotas = self.default_quotas.copy()
            if quotas:
                merged_quotas.update(quotas)
            
            tenant_config = TenantConfig(
                tenant_id=tenant_id,
                name=name,
                quotas=merged_quotas
            )
            
            self.tenants[tenant_id] = tenant_config
            
            # Initialize token buckets for this tenant
            for resource_type, quota in merged_quotas.items():
                refill_rate = quota.limit / quota.window_seconds
                self.token_buckets[tenant_id][resource_type] = TokenBucket(
                    capacity=quota.limit,
                    refill_rate=refill_rate,
                    burst_capacity=quota.burst_limit
                )
            
            # Update Prometheus metrics
            if self.metrics_enabled:
                self.tenant_enabled.labels(tenant_id=tenant_id).set(1)
                for resource_type, quota in merged_quotas.items():
                    self.quota_limit.labels(
                        tenant_id=tenant_id,
                        resource_type=resource_type.value
                    ).set(quota.limit)
            
            logger.info(f"Registered tenant {tenant_id} ({name}) with {len(merged_quotas)} quotas")
            return tenant_config
    
    def update_tenant_quotas(self, tenant_id: str, 
                            quotas: Dict[ResourceType, ResourceQuota]) -> None:
        """
        Update quotas for an existing tenant.
        
        Args:
            tenant_id: Tenant identifier
            quotas: New quota configuration
        """
        with self._lock:
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            tenant_config = self.tenants[tenant_id]
            tenant_config.quotas.update(quotas)
            tenant_config.updated_at = time.time()
            
            # Update token buckets
            for resource_type, quota in quotas.items():
                refill_rate = quota.limit / quota.window_seconds
                self.token_buckets[tenant_id][resource_type] = TokenBucket(
                    capacity=quota.limit,
                    refill_rate=refill_rate,
                    burst_capacity=quota.burst_limit
                )
                
                # Update Prometheus metrics
                if self.metrics_enabled:
                    self.quota_limit.labels(
                        tenant_id=tenant_id,
                        resource_type=resource_type.value
                    ).set(quota.limit)
            
            logger.info(f"Updated quotas for tenant {tenant_id}")
    
    def disable_tenant(self, tenant_id: str) -> None:
        """Disable a tenant, preventing new resource allocation."""
        with self._lock:
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            self.tenants[tenant_id].enabled = False
            if self.metrics_enabled:
                self.tenant_enabled.labels(tenant_id=tenant_id).set(0)
            
            logger.info(f"Disabled tenant {tenant_id}")
    
    def enable_tenant(self, tenant_id: str) -> None:
        """Re-enable a disabled tenant."""
        with self._lock:
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            self.tenants[tenant_id].enabled = True
            if self.metrics_enabled:
                self.tenant_enabled.labels(tenant_id=tenant_id).set(1)
            
            logger.info(f"Enabled tenant {tenant_id}")
    
    def check_quota(self, tenant_id: str, resource_type: ResourceType, 
                   amount: float = 1.0) -> bool:
        """
        Check if tenant has available quota for a resource.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource to check
            amount: Amount of resource to check
            
        Returns:
            True if quota is available, False otherwise
        """
        with self._lock:
            if tenant_id not in self.tenants:
                logger.warning(f"Quota check for unknown tenant {tenant_id}")
                return False
            
            tenant_config = self.tenants[tenant_id]
            if not tenant_config.enabled:
                logger.warning(f"Quota check for disabled tenant {tenant_id}")
                return False
            
            if resource_type not in self.token_buckets[tenant_id]:
                # No quota configured for this resource type
                return True
            
            bucket = self.token_buckets[tenant_id][resource_type]
            available = bucket.consume(amount)
            
            # Update Prometheus metrics
            if self.metrics_enabled:
                current_usage = bucket.capacity - bucket.available_tokens()
                self.quota_usage.labels(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value
                ).set(current_usage)
                
                if not available:
                    self.quota_exceeded.labels(
                        tenant_id=tenant_id,
                        resource_type=resource_type.value
                    ).inc()
            
            return available
    
    def acquire_agent_slot(self, tenant_id: str, agent_id: str) -> bool:
        """
        Acquire a concurrent agent slot for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            agent_id: Unique agent identifier
            
        Returns:
            True if slot was acquired, False if quota exceeded
        """
        with self._lock:
            if tenant_id not in self.tenants:
                logger.warning(f"Agent slot acquisition for unknown tenant {tenant_id}")
                return False
            
            tenant_config = self.tenants[tenant_id]
            if not tenant_config.enabled:
                return False
            
            # Check concurrent agents quota
            current_agents = len(self.active_agents[tenant_id])
            quota = tenant_config.quotas.get(ResourceType.CONCURRENT_AGENTS)
            
            if quota and current_agents >= quota.limit:
                if self.metrics_enabled:
                    self.quota_exceeded.labels(
                        tenant_id=tenant_id,
                        resource_type=ResourceType.CONCURRENT_AGENTS.value
                    ).inc()
                return False
            
            self.active_agents[tenant_id].add(agent_id)
            
            # Update Prometheus metrics
            if self.metrics_enabled:
                self.active_agents_gauge.labels(tenant_id=tenant_id).set(
                    len(self.active_agents[tenant_id])
                )
            
            return True
    
    def release_agent_slot(self, tenant_id: str, agent_id: str) -> None:
        """
        Release a concurrent agent slot.
        
        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier to release
        """
        with self._lock:
            if tenant_id in self.active_agents:
                self.active_agents[tenant_id].discard(agent_id)
                
                # Update Prometheus metrics
                if self.metrics_enabled:
                    self.active_agents_gauge.labels(tenant_id=tenant_id).set(
                        len(self.active_agents[tenant_id])
                    )
    
    def record_compute_time(self, tenant_id: str, seconds: float) -> bool:
        """
        Record compute time usage for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            seconds: Compute time in seconds
            
        Returns:
            True if recorded within quota, False if quota exceeded
        """
        return self.check_quota(tenant_id, ResourceType.COMPUTE_TIME_SECONDS, seconds)
    
    def record_api_call(self, tenant_id: str) -> bool:
        """
        Record an API call for rate limiting.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            True if within rate limit, False if exceeded
        """
        return self.check_quota(tenant_id, ResourceType.API_CALLS_PER_MINUTE)
    
    def record_message(self, tenant_id: str) -> bool:
        """
        Record a message for rate limiting.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            True if within rate limit, False if exceeded
        """
        return self.check_quota(tenant_id, ResourceType.MESSAGES_PER_HOUR)
    
    def get_tenant_usage(self, tenant_id: str) -> Dict[ResourceType, Dict[str, float]]:
        """
        Get current usage statistics for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with usage statistics per resource type
        """
        with self._lock:
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            usage = {}
            tenant_config = self.tenants[tenant_id]
            
            for resource_type, quota in tenant_config.quotas.items():
                if resource_type == ResourceType.CONCURRENT_AGENTS:
                    current = len(self.active_agents[tenant_id])
                elif resource_type in self.token_buckets[tenant_id]:
                    bucket = self.token_buckets[tenant_id][resource_type]
                    current = bucket.capacity - bucket.available_tokens()
                else:
                    current = 0
                
                usage[resource_type] = {
                    "current": current,
                    "limit": quota.limit,
                    "utilization": current / quota.limit if quota.limit > 0 else 0,
                    "window_seconds": quota.window_seconds
                }
            
            return usage
    
    def get_all_tenants(self) -> Dict[str, TenantConfig]:
        """Get configuration for all tenants."""
        with self._lock:
            return self.tenants.copy()
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get configuration for a specific tenant."""
        with self._lock:
            return self.tenants.get(tenant_id)


# Global quota manager instance
_quota_manager: Optional[TenantQuotaManager] = None


def get_quota_manager() -> TenantQuotaManager:
    """Get or create the global quota manager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = TenantQuotaManager()
    return _quota_manager


def initialize_quota_manager(default_quotas: Optional[Dict[ResourceType, ResourceQuota]] = None) -> TenantQuotaManager:
    """
    Initialize the global quota manager with custom defaults.
    
    Args:
        default_quotas: Default quotas for new tenants
        
    Returns:
        Initialized TenantQuotaManager
    """
    global _quota_manager
    _quota_manager = TenantQuotaManager(default_quotas)
    return _quota_manager


# Decorator for rate limiting functions
def rate_limit(resource_type: ResourceType, amount: float = 1.0):
    """
    Decorator to rate limit a function based on tenant quota.
    
    Args:
        resource_type: Type of resource to limit
        amount: Amount of resource consumed per call
    """
    def decorator(func):
        def wrapper(tenant_id: str, *args, **kwargs):
            quota_manager = get_quota_manager()
            if not quota_manager.check_quota(tenant_id, resource_type, amount):
                raise QuotaExceededError(
                    f"Quota exceeded for {resource_type.value} "
                    f"(tenant: {tenant_id}, amount: {amount})"
                )
            return func(tenant_id, *args, **kwargs)
        return wrapper
    return decorator


class QuotaExceededError(Exception):
    """Exception raised when a quota is exceeded."""
    pass


class TenantNotFoundError(Exception):
    """Exception raised when a tenant is not found."""
    pass


# Integration helpers for existing DeerFlow modules
class ChannelQuotaIntegration:
    """
    Helper class to integrate quota management with existing channel modules.
    """
    
    @staticmethod
    def extract_tenant_from_message(message: Dict[str, Any]) -> Optional[str]:
        """
        Extract tenant_id from a message.
        
        Args:
            message: Message dictionary
            
        Returns:
            tenant_id if found, None otherwise
        """
        # Try various common locations for tenant_id
        if "tenant_id" in message:
            return message["tenant_id"]
        
        # Check in metadata
        metadata = message.get("metadata", {})
        if "tenant_id" in metadata:
            return metadata["tenant_id"]
        
        # Check in headers (for HTTP-like messages)
        headers = message.get("headers", {})
        if "X-Tenant-ID" in headers:
            return headers["X-Tenant-ID"]
        
        # Check in channel-specific data
        channel_data = message.get("channel_data", {})
        if "tenant_id" in channel_data:
            return channel_data["tenant_id"]
        
        return None
    
    @staticmethod
    def add_tenant_to_message(message: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """
        Add tenant_id to a message.
        
        Args:
            message: Message dictionary
            tenant_id: Tenant identifier
            
        Returns:
            Updated message dictionary
        """
        if "metadata" not in message:
            message["metadata"] = {}
        
        message["metadata"]["tenant_id"] = tenant_id
        message["tenant_id"] = tenant_id
        
        return message
    
    @staticmethod
    def check_message_quota(tenant_id: str, message: Dict[str, Any]) -> bool:
        """
        Check if a message can be processed based on tenant quota.
        
        Args:
            tenant_id: Tenant identifier
            message: Message dictionary
            
        Returns:
            True if message can be processed, False otherwise
        """
        quota_manager = get_quota_manager()
        
        # Check message rate limit
        if not quota_manager.record_message(tenant_id):
            logger.warning(f"Message rate limit exceeded for tenant {tenant_id}")
            return False
        
        # Check API call rate limit
        if not quota_manager.record_api_call(tenant_id):
            logger.warning(f"API rate limit exceeded for tenant {tenant_id}")
            return False
        
        return True


# Example usage in existing modules:
"""
# In backend/app/channels/service.py:

from backend.app.tenancy.quotas import get_quota_manager, ChannelQuotaIntegration

class ChannelService:
    def __init__(self):
        self.quota_manager = get_quota_manager()
    
    async def process_message(self, message: Dict[str, Any]):
        # Extract tenant_id
        tenant_id = ChannelQuotaIntegration.extract_tenant_from_message(message)
        if not tenant_id:
            # Use default tenant or handle missing tenant
            tenant_id = "default"
        
        # Check quota before processing
        if not ChannelQuotaIntegration.check_message_quota(tenant_id, message):
            raise QuotaExceededError(f"Quota exceeded for tenant {tenant_id}")
        
        # Process message...
        # Record compute time
        start_time = time.time()
        try:
            # ... processing logic ...
            pass
        finally:
            compute_time = time.time() - start_time
            self.quota_manager.record_compute_time(tenant_id, compute_time)

# In backend/app/gateway/routers/channels.py:

from backend.app.tenancy.quotas import get_quota_manager, TenantNotFoundError, QuotaExceededError

router = APIRouter()

@router.post("/channels/{channel_id}/messages")
async def send_message(
    channel_id: str,
    message: MessageSchema,
    tenant_id: str = Header(..., alias="X-Tenant-ID")
):
    quota_manager = get_quota_manager()
    
    # Check if tenant exists and is enabled
    tenant_config = quota_manager.get_tenant_config(tenant_id)
    if not tenant_config:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if not tenant_config.enabled:
        raise HTTPException(status_code=403, detail="Tenant disabled")
    
    # Check quota
    if not quota_manager.record_api_call(tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Process message...
"""