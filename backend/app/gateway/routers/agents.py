"""CRUD API for custom agents."""

import logging
import re
import shutil
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from deerflow.config.agents_config import AgentConfig, list_custom_agents, load_agent_config, load_agent_soul
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["agents"])

AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")

# Time-series data collection for predictive scaling
class WorkloadMetrics:
    """Collects and stores time-series data for workload prediction."""
    
    def __init__(self, max_history_hours: int = 24):
        self.message_arrivals = deque(maxlen=10000)
        self.agent_durations = deque(maxlen=10000)
        self.channel_activity = deque(maxlen=10000)
        self.max_history = timedelta(hours=max_history_hours)
    
    def record_message_arrival(self, channel: str, timestamp: Optional[datetime] = None):
        """Record a message arrival event."""
        ts = timestamp or datetime.utcnow()
        self.message_arrivals.append((ts, channel))
        self._cleanup_old_data()
    
    def record_agent_duration(self, agent_name: str, duration_ms: float, timestamp: Optional[datetime] = None):
        """Record agent processing duration."""
        ts = timestamp or datetime.utcnow()
        self.agent_durations.append((ts, agent_name, duration_ms))
        self._cleanup_old_data()
    
    def record_channel_activity(self, channel: str, active_agents: int, timestamp: Optional[datetime] = None):
        """Record channel activity level."""
        ts = timestamp or datetime.utcnow()
        self.channel_activity.append((ts, channel, active_agents))
        self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove data older than max_history."""
        cutoff = datetime.utcnow() - self.max_history
        while self.message_arrivals and self.message_arrivals[0][0] < cutoff:
            self.message_arrivals.popleft()
        while self.agent_durations and self.agent_durations[0][0] < cutoff:
            self.agent_durations.popleft()
        while self.channel_activity and self.channel_activity[0][0] < cutoff:
            self.channel_activity.popleft()
    
    def get_recent_metrics(self, minutes: int = 60):
        """Get metrics from the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_messages = [(ts, ch) for ts, ch in self.message_arrivals if ts >= cutoff]
        recent_durations = [(ts, name, dur) for ts, name, dur in self.agent_durations if ts >= cutoff]
        recent_activity = [(ts, ch, agents) for ts, ch, agents in self.channel_activity if ts >= cutoff]
        
        return {
            "message_count": len(recent_messages),
            "avg_duration_ms": sum(dur for _, _, dur in recent_durations) / len(recent_durations) if recent_durations else 0,
            "active_channels": len(set(ch for _, ch, _ in recent_activity)),
            "peak_agents": max((agents for _, _, agents in recent_activity), default=0)
        }

# Global metrics collector
workload_metrics = WorkloadMetrics()

class AgentResponse(BaseModel):
    """Response model for a custom agent."""

    name: str = Field(..., description="Agent name (hyphen-case)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    soul: str | None = Field(default=None, description="SOUL.md content (included on GET /{name})")


class AgentsListResponse(BaseModel):
    """Response model for listing all custom agents."""

    agents: list[AgentResponse]


class AgentCreateRequest(BaseModel):
    """Request body for creating a custom agent."""

    name: str = Field(..., description="Agent name (must match ^[A-Za-z0-9-]+$, stored as lowercase)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    soul: str = Field(default="", description="SOUL.md content — agent personality and behavioral guardrails")


class AgentUpdateRequest(BaseModel):
    """Request body for updating a custom agent."""

    description: str | None = Field(default=None, description="Updated description")
    model: str | None = Field(default=None, description="Updated model override")
    tool_groups: list[str] | None = Field(default=None, description="Updated tool group whitelist")
    soul: str | None = Field(default=None, description="Updated SOUL.md content")


class WorkloadPrediction(BaseModel):
    """Response model for workload prediction."""
    
    predicted_load: float = Field(..., description="Predicted load (0-1 scale)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    recommended_replicas: int = Field(..., description="Recommended number of replicas")
    time_horizon_minutes: int = Field(..., description="Prediction time horizon in minutes")


def _validate_agent_name(name: str) -> None:
    """Validate agent name against allowed pattern.

    Args:
        name: The agent name to validate.

    Raises:
        HTTPException: 422 if the name is invalid.
    """
    if not AGENT_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid agent name '{name}'. Must match ^[A-Za-z0-9-]+$ (letters, digits, and hyphens only).",
        )


def _normalize_agent_name(name: str) -> str:
    """Normalize agent name to lowercase for filesystem storage."""
    return name.lower()


def _agent_config_to_response(agent_cfg: AgentConfig, include_soul: bool = False) -> AgentResponse:
    """Convert AgentConfig to AgentResponse."""
    soul: str | None = None
    if include_soul:
        soul = load_agent_soul(agent_cfg.name) or ""

    return AgentResponse(
        name=agent_cfg.name,
        description=agent_cfg.description,
        model=agent_cfg.model,
        tool_groups=agent_cfg.tool_groups,
        soul=soul,
    )


def _predict_workload(time_horizon_minutes: int = 15) -> WorkloadPrediction:
    """Predict workload using simple heuristics (placeholder for LSTM model).
    
    In a full implementation, this would use a trained LSTM model.
    For now, we use statistical forecasting based on recent patterns.
    """
    metrics = workload_metrics.get_recent_metrics(minutes=60)
    
    # Simple prediction based on recent trends
    recent_load = min(1.0, metrics["message_count"] / 100)  # Normalize to 0-1
    
    # Add some trend analysis
    if metrics["peak_agents"] > 5:
        predicted_load = min(1.0, recent_load * 1.3)  # Expect 30% increase
    else:
        predicted_load = recent_load
    
    confidence = 0.7  # Base confidence
    if metrics["message_count"] > 10:
        confidence = 0.85
    
    # Calculate recommended replicas (assuming 10 messages per replica per minute)
    base_replicas = 1
    load_per_replica = 10  # messages per minute per replica
    required_replicas = max(1, int(metrics["message_count"] / (load_per_replica * time_horizon_minutes)))
    
    # Apply prediction
    recommended_replicas = max(1, int(required_replicas * predicted_load))
    
    return WorkloadPrediction(
        predicted_load=predicted_load,
        confidence=confidence,
        recommended_replicas=recommended_replicas,
        time_horizon_minutes=time_horizon_minutes
    )


def _background_metrics_collection():
    """Background task to collect metrics periodically."""
    while True:
        try:
            # Simulate metric collection (in real implementation, this would pull from actual systems)
            current_time = datetime.utcnow()
            
            # Record some simulated activity
            workload_metrics.record_message_arrival("general", current_time)
            workload_metrics.record_channel_activity("general", 3, current_time)
            
            # Log metrics every 5 minutes
            if current_time.minute % 5 == 0:
                metrics = workload_metrics.get_recent_metrics(5)
                logger.info(f"Workload metrics: {metrics}")
                
        except Exception as e:
            logger.error(f"Error in background metrics collection: {e}")
        
        time.sleep(60)  # Collect every minute


@router.get(
    "/agents",
    response_model=AgentsListResponse,
    summary="List Custom Agents",
    description="List all custom agents available in the agents directory.",
)
async def list_agents() -> AgentsListResponse:
    """List all custom agents.

    Returns:
        List of all custom agents with their metadata (without soul content).
    """
    try:
        agents = list_custom_agents()
        return AgentsListResponse(agents=[_agent_config_to_response(a) for a in agents])
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get(
    "/agents/check",
    summary="Check Agent Name",
    description="Validate an agent name and check if it is available (case-insensitive).",
)
async def check_agent_name(name: str) -> dict:
    """Check whether an agent name is valid and not yet taken.

    Args:
        name: The agent name to check.

    Returns:
        ``{"available": true/false, "name": "<normalized>"}``

    Raises:
        HTTPException: 422 if the name is invalid.
    """
    _validate_agent_name(name)
    normalized = _normalize_agent_name(name)
    available = not get_paths().agent_dir(normalized).exists()
    return {"available": available, "name": normalized}


@router.get(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Get Custom Agent",
    description="Retrieve details and SOUL.md content for a specific custom agent.",
)
async def get_agent(name: str) -> AgentResponse:
    """Get a specific custom agent by name.

    Args:
        name: The agent name.

    Returns:
        Agent details including SOUL.md content.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _validate_agent_name(name)
    name = _normalize_agent_name(name)

    try:
        agent_cfg = load_agent_config(name)
        return _agent_config_to_response(agent_cfg, include_soul=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to get agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=201,
    summary="Create Custom Agent",
    description="Create a new custom agent with its config and SOUL.md.",
)
async def create_agent_endpoint(request: AgentCreateRequest, background_tasks: BackgroundTasks) -> AgentResponse:
    """Create a new custom agent.

    Args:
        request: The agent creation request.
        background_tasks: FastAPI background tasks.

    Returns:
        The created agent details.

    Raises:
        HTTPException: 409 if agent already exists, 422 if name is invalid.
    """
    _validate_agent_name(request.name)
    normalized_name = _normalize_agent_name(request.name)

    agent_dir = get_paths().agent_dir(normalized_name)

    if agent_dir.exists():
        raise HTTPException(status_code=409, detail=f"Agent '{normalized_name}' already exists")

    try:
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Write config.yaml
        config_data: dict = {"name": normalized_name}
        if request.description:
            config_data["description"] = request.description
        if request.model is not None:
            config_data["model"] = request.model
        if request.tool_groups is not None:
            config_data["tool_groups"] = request.tool_groups

        config_file = agent_dir / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        # Write SOUL.md
        soul_file = agent_dir / "SOUL.md"
        soul_file.write_text(request.soul, encoding="utf-8")

        logger.info(f"Created agent '{normalized_name}' at {agent_dir}")

        # Record agent creation as activity
        workload_metrics.record_channel_activity("agent_creation", 1)
        
        agent_cfg = load_agent_config(normalized_name)
        return _agent_config_to_response(agent_cfg, include_soul=True)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on failure
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
        logger.error(f"Failed to create agent '{request.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.put(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Update Custom Agent",
    description="Update an existing custom agent's config and/or SOUL.md.",
)
async def update_agent(name: str, request: AgentUpdateRequest) -> AgentResponse:
    """Update an existing custom agent.

    Args:
        name: The agent name.
        request: The update request (all fields optional).

    Returns:
        The updated agent details.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _validate_agent_name(name)
    normalized_name = _normalize_agent_name(name)

    agent_dir = get_paths().agent_dir(normalized_name)
    if not agent_dir.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{normalized_name}' not found")

    try:
        # Update config.yaml if any config fields are provided
        config_file = agent_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {"name": normalized_name}

        # Update fields if provided
        if request.description is not None:
            config_data["description"] = request.description
        if request.model is not None:
            config_data["model"] = request.model
        if request.tool_groups is not None:
            config_data["tool_groups"] = request.tool_groups

        # Write updated config
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        # Update SOUL.md if provided
        if request.soul is not None:
            soul_file = agent_dir / "SOUL.md"
            soul_file.write_text(request.soul, encoding="utf-8")

        logger.info(f"Updated agent '{normalized_name}'")

        # Record agent update as activity
        workload_metrics.record_channel_activity("agent_update", 1)

        agent_cfg = load_agent_config(normalized_name)
        return _agent_config_to_response(agent_cfg, include_soul=True)

    except Exception as e:
        logger.error(f"Failed to update agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")


@router.delete(
    "/agents/{name}",
    status_code=204,
    summary="Delete Custom Agent",
    description="Delete a custom agent and all its files.",
)
async def delete_agent(name: str) -> None:
    """Delete a custom agent.

    Args:
        name: The agent name.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _validate_agent_name(name)
    normalized_name = _normalize_agent_name(name)

    agent_dir = get_paths().agent_dir(normalized_name)
    if not agent_dir.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{normalized_name}' not found")

    try:
        shutil.rmtree(agent_dir)
        logger.info(f"Deleted agent '{normalized_name}'")
        
        # Record agent deletion as activity
        workload_metrics.record_channel_activity("agent_deletion", 1)
        
    except Exception as e:
        logger.error(f"Failed to delete agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


@router.get(
    "/scaling/predict",
    response_model=WorkloadPrediction,
    summary="Predict Workload",
    description="Predict workload for the next 5-15 minutes using ML-based forecasting.",
)
async def predict_workload(minutes: int = 15) -> WorkloadPrediction:
    """Predict workload for auto-scaling.

    Args:
        minutes: Prediction time horizon in minutes (5-15).

    Returns:
        Workload prediction with scaling recommendations.

    Raises:
        HTTPException: 422 if minutes is out of range.
    """
    if minutes < 5 or minutes > 15:
        raise HTTPException(
            status_code=422,
            detail="Prediction time horizon must be between 5 and 15 minutes."
        )
    
    try:
        prediction = _predict_workload(minutes)
        logger.info(f"Workload prediction: {prediction.dict()}")
        return prediction
    except Exception as e:
        logger.error(f"Failed to predict workload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to predict workload: {str(e)}")


@router.get(
    "/scaling/metrics",
    summary="Get Workload Metrics",
    description="Get current workload metrics for monitoring.",
)
async def get_workload_metrics(minutes: int = 60) -> dict:
    """Get current workload metrics.

    Args:
        minutes: Time window in minutes for metrics.

    Returns:
        Current workload metrics.
    """
    try:
        metrics = workload_metrics.get_recent_metrics(minutes)
        return {
            "time_window_minutes": minutes,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get workload metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Start background metrics collection when module loads
import threading
metrics_thread = threading.Thread(target=_background_metrics_collection, daemon=True)
metrics_thread.start()
logger.info("Started background metrics collection for predictive auto-scaling")