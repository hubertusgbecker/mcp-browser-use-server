"""Tests for server/session.py module.

This test suite covers the browser session management functionality.
Following TDD principles - tests written first.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from server import session


class TestCreateSession:
    """Test suite for create_session function."""

    @pytest.mark.asyncio
    async def test_create_session_creates_browser_session(self):
        """Test that create_session creates and starts a BrowserSession.
        
        Verifies:
        - BrowserSession is instantiated with correct profile
        - Session.start() is called
        - Session is tracked in active_sessions
        - Returns the session instance
        """
        # Arrange
        session_id = "test-session-123"
        mock_browser_session = AsyncMock()
        mock_browser_session.start = AsyncMock()
        
        with patch("browser_use.BrowserSession") as mock_session_class, \
             patch("browser_use.browser.BrowserProfile") as mock_profile_class:
            
            mock_profile = MagicMock()
            mock_profile_class.return_value = mock_profile
            mock_session_class.return_value = mock_browser_session
            
            # Act
            result = await session.create_session(
                session_id=session_id,
                headless=True,
                chrome_path=None
            )
            
            # Assert
            assert result == mock_browser_session
            mock_profile_class.assert_called_once_with(
                headless=True,
                is_local=True,
                use_cloud=False
            )
            mock_session_class.assert_called_once_with(
                browser_profile=mock_profile
            )
            mock_browser_session.start.assert_called_once()
            
            # Verify session is tracked
            assert session_id in session.active_sessions
            assert session.active_sessions[session_id]["id"] == session_id
            assert session.active_sessions[session_id]["session"] == mock_browser_session
            assert "created_at" in session.active_sessions[session_id]
            assert "last_activity" in session.active_sessions[session_id]

    @pytest.mark.asyncio
    async def test_create_session_with_chrome_path(self):
        """Test that create_session uses custom chrome_path when provided."""
        # Arrange
        session_id = "test-session-chrome"
        chrome_path = "/custom/path/to/chrome"
        mock_browser_session = AsyncMock()
        
        with patch("browser_use.BrowserSession") as mock_session_class, \
             patch("browser_use.browser.BrowserProfile") as mock_profile_class:
            
            mock_session_class.return_value = mock_browser_session
            
            # Act
            await session.create_session(
                session_id=session_id,
                headless=False,
                chrome_path=chrome_path
            )
            
            # Assert
            mock_profile_class.assert_called_once_with(
                headless=False,
                is_local=True,
                use_cloud=False,
                executable_path=chrome_path
            )


class TestGetSession:
    """Test suite for get_session function."""

    @pytest.mark.asyncio
    async def test_get_session_returns_existing_session(self):
        """Test that get_session returns an existing session."""
        # Arrange
        session_id = "existing-session"
        mock_session = MagicMock()
        session.active_sessions[session_id] = {
            "id": session_id,
            "session": mock_session,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Act
        result = await session.get_session(session_id)
        
        # Assert
        assert result == mock_session
        # Verify last_activity was updated
        assert "last_activity" in session.active_sessions[session_id]

    @pytest.mark.asyncio
    async def test_get_session_returns_none_for_missing_session(self):
        """Test that get_session returns None when session doesn't exist."""
        # Arrange
        session_id = "nonexistent-session"
        
        # Act
        result = await session.get_session(session_id)
        
        # Assert
        assert result is None


class TestCloseSession:
    """Test suite for close_session function."""

    @pytest.mark.asyncio
    async def test_close_session_closes_and_removes_session(self):
        """Test that close_session stops the session and removes it from tracking."""
        # Arrange
        session_id = "session-to-close"
        mock_session = AsyncMock()
        mock_session.stop = AsyncMock()
        
        session.active_sessions[session_id] = {
            "id": session_id,
            "session": mock_session,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Act
        result = await session.close_session(session_id)
        
        # Assert
        assert result is True
        mock_session.stop.assert_called_once()
        assert session_id not in session.active_sessions

    @pytest.mark.asyncio
    async def test_close_session_returns_false_for_missing_session(self):
        """Test that close_session returns False when session doesn't exist."""
        # Arrange
        session_id = "nonexistent-session"
        
        # Act
        result = await session.close_session(session_id)
        
        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_close_session_handles_stop_error_gracefully(self):
        """Test that close_session handles errors during stop() gracefully."""
        # Arrange
        session_id = "error-session"
        mock_session = AsyncMock()
        mock_session.stop = AsyncMock(side_effect=Exception("Stop failed"))
        
        session.active_sessions[session_id] = {
            "id": session_id,
            "session": mock_session,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Act
        result = await session.close_session(session_id)
        
        # Assert
        assert result is True  # Still returns True and removes from tracking
        assert session_id not in session.active_sessions


class TestCloseAllSessions:
    """Test suite for close_all_sessions function."""

    @pytest.mark.asyncio
    async def test_close_all_sessions_closes_all_tracked_sessions(self):
        """Test that close_all_sessions closes all active sessions."""
        # Arrange
        mock_session1 = AsyncMock()
        mock_session2 = AsyncMock()
        mock_session3 = AsyncMock()
        
        session.active_sessions = {
            "session1": {
                "id": "session1",
                "session": mock_session1,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            },
            "session2": {
                "id": "session2",
                "session": mock_session2,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            },
            "session3": {
                "id": "session3",
                "session": mock_session3,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
        }
        
        # Act
        result = await session.close_all_sessions()
        
        # Assert
        assert result == 3
        assert len(session.active_sessions) == 0
        mock_session1.stop.assert_called_once()
        mock_session2.stop.assert_called_once()
        mock_session3.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_sessions_with_no_sessions(self):
        """Test that close_all_sessions returns 0 when no sessions exist."""
        # Arrange
        session.active_sessions = {}
        
        # Act
        result = await session.close_all_sessions()
        
        # Assert
        assert result == 0


class TestListSessions:
    """Test suite for list_sessions function."""

    def test_list_sessions_returns_session_metadata(self):
        """Test that list_sessions returns metadata for all active sessions."""
        # Arrange
        created_time = datetime.now().isoformat()
        activity_time = datetime.now().isoformat()
        
        session.active_sessions = {
            "session1": {
                "id": "session1",
                "session": MagicMock(),
                "created_at": created_time,
                "last_activity": activity_time
            },
            "session2": {
                "id": "session2",
                "session": MagicMock(),
                "created_at": created_time,
                "last_activity": activity_time
            }
        }
        
        # Act
        result = session.list_sessions()
        
        # Assert
        assert len(result) == 2
        assert {"id": "session1", "created_at": created_time, "last_activity": activity_time} in result
        assert {"id": "session2", "created_at": created_time, "last_activity": activity_time} in result

    def test_list_sessions_returns_empty_list_when_no_sessions(self):
        """Test that list_sessions returns empty list when no sessions exist."""
        # Arrange
        session.active_sessions = {}
        
        # Act
        result = session.list_sessions()
        
        # Assert
        assert result == []


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Cleanup sessions before and after each test."""
    session.active_sessions = {}
    yield
    session.active_sessions = {}
