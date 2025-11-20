"""Session management for browser-use MCP server.

This module handles BrowserSession lifecycle: creation, tracking, and cleanup.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Active sessions store
active_sessions: Dict[str, Dict[str, Any]] = {}


async def create_session(
    session_id: str,
    headless: bool = True,
    chrome_path: Optional[str] = None,
) -> Any:
    """Create a new BrowserSession.

    Args:
        session_id: Unique identifier for session
        headless: Run browser in headless mode
        chrome_path: Optional path to Chrome executable

    Returns:
        BrowserSession instance
    """
    from browser_use import BrowserSession
    from browser_use.browser import BrowserProfile

    logger.info(f"Creating browser session {session_id}")

    profile_kwargs: Dict[str, Any] = {
        "headless": headless,
        "is_local": True,
        "use_cloud": False,
    }

    if chrome_path:
        profile_kwargs["executable_path"] = chrome_path

    profile = BrowserProfile(**profile_kwargs)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    # Track session
    active_sessions[session_id] = {
        "id": session_id,
        "session": session,
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat(),
    }

    logger.info(f"Browser session {session_id} created and tracked")
    return session


async def get_session(session_id: str) -> Optional[Any]:
    """Get existing session by ID.

    Args:
        session_id: Session identifier

    Returns:
        BrowserSession instance or None if not found
    """
    if session_id in active_sessions:
        # Update last activity
        active_sessions[session_id]["last_activity"] = (
            datetime.now().isoformat()
        )
        return active_sessions[session_id]["session"]
    return None


async def close_session(session_id: str) -> bool:
    """Close and remove a session.

    Args:
        session_id: Session identifier

    Returns:
        True if session was closed, False if not found
    """
    if session_id not in active_sessions:
        return False

    session_data = active_sessions[session_id]
    session = session_data["session"]

    try:
        await session.stop()
        logger.info(f"Closed browser session {session_id}")
    except Exception as e:
        logger.error(f"Error closing session {session_id}: {e}")

    del active_sessions[session_id]
    return True


async def close_all_sessions() -> int:
    """Close all active sessions.

    Returns:
        Number of sessions closed
    """
    session_ids = list(active_sessions.keys())
    count = 0

    for session_id in session_ids:
        if await close_session(session_id):
            count += 1

    return count


def list_sessions() -> list[Dict[str, Any]]:
    """List all active sessions with metadata.

    Returns:
        List of session metadata dicts
    """
    return [
        {
            "id": session_id,
            "created_at": data["created_at"],
            "last_activity": data["last_activity"],
        }
        for session_id, data in active_sessions.items()
    ]
