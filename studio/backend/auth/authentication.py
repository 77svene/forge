# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any

from fastapi import Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import (
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()  # Reads Authorization: Bearer <token>

# WebSocket connection manager for real-time collaboration
class WebSocketConnectionManager:
    def __init__(self):
        # Store active connections by workspace_id -> {user_id: websocket}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, workspace_id: str, user_id: str):
        await websocket.accept()
        if workspace_id not in self.active_connections:
            self.active_connections[workspace_id] = {}
        self.active_connections[workspace_id][user_id] = websocket
    
    def disconnect(self, workspace_id: str, user_id: str):
        if workspace_id in self.active_connections:
            self.active_connections[workspace_id].pop(user_id, None)
            if not self.active_connections[workspace_id]:
                del self.active_connections[workspace_id]
    
    async def broadcast_to_workspace(self, workspace_id: str, message: dict, exclude_user: Optional[str] = None):
        """Broadcast message to all users in a workspace except the excluded user."""
        if workspace_id in self.active_connections:
            for user_id, connection in self.active_connections[workspace_id].items():
                if user_id != exclude_user:
                    try:
                        await connection.send_json(message)
                    except:
                        # Connection might be closed, remove it
                        self.disconnect(workspace_id, user_id)

# Global WebSocket manager instance
websocket_manager = WebSocketConnectionManager()


def _get_secret_for_subject(subject: str) -> str:
    secret = get_jwt_secret(subject)
    if secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return secret


def _decode_subject_without_verification(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because the signing secret is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        _get_secret_for_subject(subject),
        algorithm=ALGORITHM,
    )


def create_refresh_token(subject: str) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs) and expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(token, subject, expires_at.isoformat())
    return token


def refresh_access_token(refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token itself is NOT consumed — it stays valid until expiry.
    Returns a new access_token or None if the refresh token is invalid/expired.
    """
    username = verify_refresh_token(refresh_token)
    if username is None:
        return None, None
    return create_access_token(subject=username), username


def reload_secret() -> None:
    """
    Keep legacy API compatibility for callers expecting auth storage init.

    Auth now resolves the current signing secret directly from SQLite.
    """
    load_jwt_secret()


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    return await _get_current_subject(
        credentials,
        allow_password_change=False,
    )


async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        credentials,
        allow_password_change=True,
    )


async def _get_current_subject(
    credentials: HTTPAuthorizationCredentials,
    *,
    allow_password_change: bool,
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.

    Use this as a dependency on routes that should be protected, e.g.:

        @router.get("/secure")
        async def secure_endpoint(current_subject: str = Depends(get_current_subject)):
            ...
    """
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    record = get_user_and_secret(subject)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        if must_change_password and not allow_password_change:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password change required",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def get_current_subject_from_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
) -> str:
    """
    Validate JWT from WebSocket connection.
    
    Token can be passed as:
    1. Query parameter: ws://host/ws?token=<jwt>
    2. First message after connection: {"type": "auth", "token": "<jwt>"}
    """
    # Try to get token from query parameters first
    if token is None:
        token = websocket.query_params.get("token")
    
    # If no token in query params, wait for auth message
    if token is None:
        try:
            # Wait for auth message (timeout after 10 seconds)
            data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            if data.get("type") == "auth" and "token" in data:
                token = data["token"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
        except (asyncio.TimeoutError, WebSocketDisconnect):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication timeout",
            )
    
    # Validate the token
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    record = get_user_and_secret(subject)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        if must_change_password:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password change required",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def validate_websocket_token(token: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a JWT token for WebSocket connections.
    
    Returns:
        Tuple of (is_valid, subject, error_message)
    """
    subject = _decode_subject_without_verification(token)
    if subject is None:
        return False, None, "Invalid token payload"

    record = get_user_and_secret(subject)
    if record is None:
        return False, None, "Invalid or expired token"

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            return False, None, "Invalid token payload"
        if must_change_password:
            return False, None, "Password change required"
        return True, subject, None
    except jwt.InvalidTokenError:
        return False, None, "Invalid or expired token"


# Import asyncio for timeout handling
import asyncio