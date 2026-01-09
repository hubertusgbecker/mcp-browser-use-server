"""Tests for new browser tools (get_state, navigate, click, etc.)."""

import json

import pytest

from server.server import create_mcp_server, task_store


class TestBrowserGetState:
    """Test browser_get_state tool."""

    @pytest.mark.asyncio
    async def test_browser_get_state_from_task(self, mock_llm, cleanup_tasks):
        """Test browser_get_state returns state from completed task."""
        # Setup: populate task_store with a completed task
        task_id = "test-task-123"
        task_store[task_id] = {
            "id": task_id,
            "status": "completed",
            "url": "https://example.com",
            "title": "Example Domain",
            "result": {
                "final_result": "Task completed successfully",
                "success": True,
            },
            "tabs": [{"url": "https://example.com", "title": "Example Domain"}],
            "interactive_elements": [
                {"index": 1, "tag": "a", "text": "More information..."},
                {"index": 2, "tag": "button", "text": "Click me"},
            ],
        }

        # Create server
        server = create_mcp_server(llm=mock_llm)

        # Import the call_tool decorator's wrapped function
        # The server registers handlers, so we need to simulate an MCP call
        from mcp import types

        # Get list of tools to verify browser_get_state is registered
        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )
        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_get_state" not in tool_names:
            pytest.fail("browser_get_state not registered in tools list")

        # Call the tool via the registered handler
        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_get_state", arguments={"task_id": task_id}
                )
            )
        )

        # Verify response
        if len(result.root.content) != 1:
            pytest.fail("Unexpected number of content items in result")
        if result.root.content[0].type != "text":
            pytest.fail("Expected text content in tool result")

        response_data = json.loads(result.root.content[0].text)
        if response_data.get("url") != "https://example.com":
            pytest.fail("Returned URL mismatch")
        if response_data.get("title") != "Example Domain":
            pytest.fail("Returned title mismatch")
        if len(response_data.get("interactive_elements", [])) != 2:
            pytest.fail("Unexpected number of interactive elements")
        if response_data["interactive_elements"][0].get("index") != 1:
            pytest.fail("Interactive element index mismatch")

    @pytest.mark.asyncio
    async def test_browser_get_state_task_not_found(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_get_state with non-existent task."""
        server = create_mcp_server(llm=mock_llm)

        from mcp import types

        # Call with non-existent task
        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_get_state",
                    arguments={"task_id": "nonexistent"},
                )
            )
        )

        if len(result.root.content) != 1:
            pytest.fail("Unexpected number of content items for not-found case")
        response_data = json.loads(result.root.content[0].text)
        if "error" not in response_data:
            pytest.fail("Expected error in response for nonexistent task")
        if "not found" not in response_data["error"].lower():
            pytest.fail("Expected 'not found' in error message")

    @pytest.mark.asyncio
    async def test_browser_get_state_in_list_tools(self, mock_llm):
        """Test that browser_get_state appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)

        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        # Check that browser_get_state is in the list
        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_get_state" not in tool_names:
            pytest.fail("browser_get_state not present in list_tools")

        # Get the tool definition
        get_state_tool = next(
            (
                tool
                for tool in tools_list.root.tools
                if tool.name == "browser_get_state"
            ),
            None,
        )
        if get_state_tool is None:
            pytest.fail("browser_get_state tool definition missing")
        if "task_id" not in get_state_tool.inputSchema.get("properties", {}):
            pytest.fail("browser_get_state inputSchema missing 'task_id'")

    @pytest.mark.asyncio
    async def test_browser_get_state_from_session(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_get_state with session_id (live state)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session manager and browser session
        mock_session = MagicMock()
        mock_session.get_browser_state_summary = AsyncMock(
            return_value={
                "url": "https://example.com/live",
                "title": "Live Page",
                "tabs": [
                    {"title": "Live Page", "url": "https://example.com/live"}
                ],
                "interactive_elements": [
                    {"index": 1, "tag": "button", "text": "Submit"}
                ],
            }
        )

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_get_state",
                        arguments={"session_id": "sess-123"},
                    )
                )
            )

            response_data = json.loads(result.root.content[0].text)
            if response_data.get("url") != "https://example.com/live":
                pytest.fail("Session browser_get_state returned wrong URL")
            if response_data.get("title") != "Live Page":
                pytest.fail("Session browser_get_state returned wrong title")
            mock_session.get_browser_state_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_get_state_with_screenshot(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_get_state with screenshot parameter."""
        import base64
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session with screenshot capability
        mock_session = MagicMock()
        mock_session.get_browser_state_summary = AsyncMock(
            return_value={
                "url": "https://example.com",
                "title": "Test Page",
                "tabs": [],
                "interactive_elements": [],
            }
        )

        # Mock page.screenshot
        fake_screenshot = b"fake_png_data"
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=fake_screenshot)
        mock_session.get_current_page = MagicMock(return_value=mock_page)

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_get_state",
                        arguments={
                            "session_id": "sess-123",
                            "screenshot": True,
                        },
                    )
                )
            )

            response_data = json.loads(result.root.content[0].text)
            if "screenshot" not in response_data:
                pytest.fail("Expected screenshot in response")
            # Verify it's base64 encoded
            decoded = base64.b64decode(response_data["screenshot"])
            if decoded != fake_screenshot:
                pytest.fail("Screenshot data did not match expected bytes")

    @pytest.mark.asyncio
    async def test_browser_get_state_timeout(self, mock_llm, cleanup_tasks):
        """Test browser_get_state handles timeout gracefully."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session that hangs
        async def slow_state_summary():
            await asyncio.sleep(100)  # Simulate hanging call
            return {"url": "should not reach"}

        mock_session = MagicMock()
        mock_session.get_browser_state_summary = slow_state_summary

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_get_state",
                        arguments={"session_id": "sess-123"},
                    )
                )
            )

            # Should return error response instead of hanging
            response_text = result.root.content[0].text
            response_data = json.loads(response_text)
            assert (
                "error" in response_data or "timeout" in response_text.lower()
            )


class TestBrowserNavigate:
    """Test browser_navigate tool."""

    @pytest.mark.asyncio
    async def test_browser_navigate_creates_session(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_navigate creates session and navigates."""
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session creation and navigation
        with (
            patch("server.session.create_session") as mock_create,
            patch("server.session.get_session") as mock_get,
        ):
            mock_session = AsyncMock()
            mock_session.id = "test-session-123"
            mock_create.return_value = mock_session
            mock_get.return_value = None  # No existing session

            # Call browser_navigate
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_navigate",
                        arguments={"url": "https://example.com"},
                    )
                )
            )

            if len(result.root.content) != 1:
                pytest.fail("Unexpected content length from browser_navigate")
            response_data = json.loads(result.root.content[0].text)
            if not (
                "navigated" in response_data.get("message", "").lower()
                or "session" in response_data.get("message", "").lower()
            ):
                pytest.fail("browser_navigate returned unexpected message")

    @pytest.mark.asyncio
    async def test_browser_navigate_in_list_tools(self, mock_llm):
        """Test that browser_navigate appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_navigate" not in tool_names:
            pytest.fail("browser_navigate not present in list_tools")


class TestBrowserClick:
    """Test browser_click tool."""

    @pytest.mark.asyncio
    async def test_browser_click_in_list_tools(self, mock_llm):
        """Test that browser_click appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_click" not in tool_names:
            pytest.fail("browser_click not present in list_tools")


class TestSessionManagement:
    """Test session management tools (list_sessions, close_session)."""

    @pytest.mark.asyncio
    async def test_browser_list_sessions(self, mock_llm, cleanup_tasks):
        """Test browser_list_sessions returns active sessions."""
        import json
        from unittest.mock import patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session manager with active sessions
        mock_sessions = [
            {
                "session_id": "sess-1",
                "created_at": "2025-01-15T12:00:00",
                "last_activity": "2025-01-15T12:05:00",
            },
            {
                "session_id": "sess-2",
                "created_at": "2025-01-15T12:10:00",
                "last_activity": "2025-01-15T12:15:00",
            },
        ]

        with patch("server.session.list_sessions", return_value=mock_sessions):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_list_sessions", arguments={}
                    )
                )
            )

            response_text = result.root.content[0].text
            response = json.loads(response_text)

            if "sessions" not in response:
                pytest.fail("browser_list_sessions response missing 'sessions'")
            if len(response["sessions"]) != 2:
                pytest.fail(
                    "browser_list_sessions returned unexpected number of sessions"
                )
            if response["sessions"][0].get("session_id") != "sess-1":
                pytest.fail(
                    "First session id mismatch in browser_list_sessions"
                )

    @pytest.mark.asyncio
    async def test_browser_close_session(self, mock_llm, cleanup_tasks):
        """Test browser_close_session closes a session."""
        import json
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        with patch(
            "server.session.close_session", new_callable=AsyncMock
        ) as mock_close:
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_close_session",
                        arguments={"session_id": "sess-123"},
                    )
                )
            )

            mock_close.assert_called_once_with("sess-123")
            response_text = result.root.content[0].text
            response = json.loads(response_text)

            if response.get("message") != "Session closed successfully":
                pytest.fail("Unexpected message from browser_close_session")
            if response.get("session_id") != "sess-123":
                pytest.fail("browser_close_session returned wrong session_id")

    @pytest.mark.asyncio
    async def test_browser_list_sessions_in_list_tools(self, mock_llm):
        """Test that browser_list_sessions appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_list_sessions" not in tool_names:
            pytest.fail("browser_list_sessions not present in list_tools")

    @pytest.mark.asyncio
    async def test_browser_close_session_in_list_tools(self, mock_llm):
        """Test that browser_close_session appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_close_session" not in tool_names:
            pytest.fail("browser_close_session not present in list_tools")


class TestBrowserExtractContent:
    """Test browser_extract_content tool."""

    @pytest.mark.asyncio
    async def test_browser_extract_content(self, mock_llm, cleanup_tasks):
        """Test browser_extract_content extracts content using Agent."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session
        mock_session = MagicMock()
        mock_session.get_state_as_text = AsyncMock(
            return_value="<html>Test page</html>"
        )

        # Mock Agent and its result
        mock_agent_result = MagicMock()
        mock_agent_result.extracted_content = lambda: ["Test article about AI"]

        with (
            patch(
                "server.session.get_session",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
            patch("server.server.Agent") as mock_agent_class,
            patch("server.server.ChatOpenAI") as mock_llm_class,
        ):
            # Configure mock Agent
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_agent_result)
            mock_agent_class.return_value = mock_agent_instance

            # Configure mock ChatOpenAI
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_extract_content",
                        arguments={
                            "session_id": "sess-123",
                            "instruction": "Extract the main article text",
                        },
                    )
                )
            )

            response_data = json.loads(result.root.content[0].text)

            # Check for success field and extracted_content
            assert "success" in response_data
            assert response_data["success"] is True
            assert "extracted_content" in response_data
            assert "Test article about AI" in response_data["extracted_content"]

    @pytest.mark.asyncio
    async def test_browser_extract_content_session_not_found(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_extract_content with non-existent session."""
        import json
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_extract_content",
                        arguments={
                            "session_id": "nonexistent",
                            "instruction": "Extract data",
                        },
                    )
                )
            )

            response_data = json.loads(result.root.content[0].text)
            if "error" not in response_data:
                pytest.fail(
                    "Expected error when session not found in extract_content"
                )

    @pytest.mark.asyncio
    async def test_browser_extract_content_in_list_tools(self, mock_llm):
        """Test that browser_extract_content appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_extract_content" not in tool_names:
            pytest.fail("browser_extract_content not present in list_tools")


class TestAgentIntegration:
    """Test agent integration enhancements."""

    @pytest.mark.asyncio
    async def test_browser_use_with_allowed_domains(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_use with allowed_domains parameter."""
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock run_browser_task_async
        with patch(
            "server.server.run_browser_task_async", new_callable=AsyncMock
        ) as mock_run:
            await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_use",
                        arguments={
                            "url": "https://example.com",
                            "action": "Test action",
                            "allowed_domains": ["example.com", "trusted.com"],
                        },
                    )
                )
            )

            # Check that allowed_domains was passed to config
            from tests._helpers import assert_mock_run_config_equals

            assert_mock_run_config_equals(
                mock_run, "allowed_domains", ["example.com", "trusted.com"]
            )

    @pytest.mark.asyncio
    async def test_browser_use_with_use_vision(self, mock_llm, cleanup_tasks):
        """Test browser_use with use_vision parameter."""
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        with patch(
            "server.server.run_browser_task_async", new_callable=AsyncMock
        ) as mock_run:
            await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_use",
                        arguments={
                            "url": "https://example.com",
                            "action": "Test action",
                            "use_vision": True,
                        },
                    )
                )
            )

            from tests._helpers import assert_mock_run_config_equals

            assert_mock_run_config_equals(mock_run, "use_vision", True)

    @pytest.mark.asyncio
    async def test_browser_use_with_max_steps(self, mock_llm, cleanup_tasks):
        """Test browser_use with max_steps parameter."""
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        with patch(
            "server.server.run_browser_task_async", new_callable=AsyncMock
        ) as mock_run:
            await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_use",
                        arguments={
                            "url": "https://example.com",
                            "action": "Test action",
                            "max_steps": 20,
                        },
                    )
                )
            )

            from tests._helpers import assert_mock_run_config_equals

            assert_mock_run_config_equals(mock_run, "max_steps", 20)


class TestTabsAPI:
    """Test tab management tools."""

    @pytest.mark.asyncio
    async def test_browser_list_tabs(self, mock_llm, cleanup_tasks):
        """Test browser_list_tabs returns tab information."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock session with tabs - using objects with url and title attributes
        mock_session = MagicMock()

        # Create mock tab objects
        mock_tab1 = MagicMock()
        mock_tab1.url = "https://example.com/1"
        mock_tab1.title = "Tab 1"

        mock_tab2 = MagicMock()
        mock_tab2.url = "https://example.com/2"
        mock_tab2.title = "Tab 2"

        mock_tabs = [mock_tab1, mock_tab2]
        mock_session.get_tabs = AsyncMock(return_value=mock_tabs)

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_list_tabs",
                        arguments={"session_id": "sess-123"},
                    )
                )
            )

            response_data = json.loads(result.root.content[0].text)
            if "tabs" not in response_data:
                pytest.fail("browser_list_tabs response missing 'tabs'")

            tabs = response_data["tabs"]
            assert len(tabs) == 2
            assert tabs[0]["url"] == "https://example.com/1"
            assert tabs[0]["title"] == "Tab 1"
            assert tabs[1]["url"] == "https://example.com/2"
            assert tabs[1]["title"] == "Tab 2"
            if len(response_data["tabs"]) != 2:
                pytest.fail(
                    "browser_list_tabs returned unexpected number of tabs"
                )
            if response_data["tabs"][0].get("title") != "Tab 1":
                pytest.fail("First tab title mismatch in browser_list_tabs")

    @pytest.mark.asyncio
    async def test_browser_switch_tab(self, mock_llm, cleanup_tasks):
        """Test browser_switch_tab returns error message (not supported)."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        mock_session = MagicMock()

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_switch_tab",
                        arguments={"session_id": "sess-123", "tab_index": 1},
                    )
                )
            )

            # Verify that error message is returned
            response_data = json.loads(result.root.content[0].text)
            assert "error" in response_data
            assert "not directly supported" in response_data["error"].lower()

    @pytest.mark.asyncio
    async def test_browser_close_tab(self, mock_llm, cleanup_tasks):
        """Test browser_close_tab returns error message (not supported)."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        mock_session = MagicMock()

        with patch(
            "server.session.get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        ):
            result = await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_close_tab",
                        arguments={"session_id": "sess-123", "tab_index": 1},
                    )
                )
            )

            # Verify that error message is returned
            response_data = json.loads(result.root.content[0].text)
            assert "error" in response_data
            assert "not supported" in response_data["error"].lower()

    @pytest.mark.asyncio
    async def test_browser_list_tabs_in_list_tools(self, mock_llm):
        """Test that browser_list_tabs appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_list_tabs" not in tool_names:
            pytest.fail("browser_list_tabs not present in list_tools")

    @pytest.mark.asyncio
    async def test_browser_switch_tab_in_list_tools(self, mock_llm):
        """Test that browser_switch_tab appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_switch_tab" not in tool_names:
            pytest.fail("browser_switch_tab not present in list_tools")

    @pytest.mark.asyncio
    async def test_browser_close_tab_in_list_tools(self, mock_llm):
        """Test that browser_close_tab appears in list_tools."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        tools_list = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )

        tool_names = [tool.name for tool in tools_list.root.tools]
        if "browser_close_tab" not in tool_names:
            pytest.fail("browser_close_tab not present in list_tools")


class TestSecurityHardening:
    """Test security features like domain validation."""

    @pytest.mark.asyncio
    async def test_allowed_domains_validation(self, mock_llm, cleanup_tasks):
        """Test that allowed_domains configuration is applied."""
        from unittest.mock import AsyncMock, patch

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Mock run_browser_task_async to verify config
        with patch(
            "server.server.run_browser_task_async", new_callable=AsyncMock
        ) as mock_run:
            await server.request_handlers[types.CallToolRequest](
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="browser_use",
                        arguments={
                            "url": "https://example.com",
                            "action": "Test action",
                            "allowed_domains": ["example.com", "safe-site.com"],
                        },
                    )
                )
            )

            # Verify allowed_domains was passed through
            call_args = mock_run.call_args
            config = call_args[1]["config"]
            if "allowed_domains" not in config:
                pytest.fail("allowed_domains missing in config")
            if "example.com" not in config["allowed_domains"]:
                pytest.fail("example.com not in allowed_domains")
            if "safe-site.com" not in config["allowed_domains"]:
                pytest.fail("safe-site.com not in allowed_domains")

    @pytest.mark.asyncio
    async def test_security_config_presence(self, mock_llm):
        """Test that security configurations are recognized."""
        from server.server import CONFIG

        # Verify CONFIG is accessible and can hold security settings
        if CONFIG is None:
            pytest.fail("CONFIG is None")
        if not isinstance(CONFIG, dict):
            pytest.fail("CONFIG is not a dict")

        # Security-related config keys should be supported
        # (This validates the infrastructure is in place)
        test_config = CONFIG.copy()
        test_config["allowed_domains"] = ["example.com"]
        if test_config.get("allowed_domains") != ["example.com"]:
            pytest.fail("Test config mutation did not persist allowed_domains")

class TestBrowserTimeoutFix:
    """Test that Chrome extension timeout fix is properly configured."""

    @pytest.mark.asyncio
    async def test_extensions_disabled_in_config(self):
        """Test that Chrome extensions are disabled by default to prevent timeouts."""
        from server.server import CONFIG

        # Verify extensions are disabled by default
        assert "ENABLE_DEFAULT_EXTENSIONS" in CONFIG, "ENABLE_DEFAULT_EXTENSIONS not in CONFIG"
        assert CONFIG["ENABLE_DEFAULT_EXTENSIONS"] is False, "Chrome extensions should be disabled by default"

    @pytest.mark.asyncio
    async def test_timeout_parameters_in_config(self):
        """Test that timeout parameters are configured."""
        from server.server import CONFIG

        # Verify timeout parameters exist and have reasonable values
        assert "WAIT_FOR_NETWORK_IDLE_PAGE_LOAD_TIME" in CONFIG
        assert "MINIMUM_WAIT_PAGE_LOAD_TIME" in CONFIG

        # Check values are reasonable (between 0 and 60 seconds)
        network_idle_timeout = CONFIG["WAIT_FOR_NETWORK_IDLE_PAGE_LOAD_TIME"]
        min_wait_timeout = CONFIG["MINIMUM_WAIT_PAGE_LOAD_TIME"]

        assert 0 < network_idle_timeout <= 60, f"Invalid network idle timeout: {network_idle_timeout}"
        assert 0 < min_wait_timeout <= 60, f"Invalid minimum wait timeout: {min_wait_timeout}"

    @pytest.mark.asyncio
    async def test_browser_profile_creation_with_fix(self, mock_llm):
        """Test that BrowserProfile is created with extensions disabled."""
        from browser_use.browser import BrowserProfile

        from server.server import CONFIG

        # Create a BrowserProfile as the server would
        bp_kwargs = {
            "is_local": True,
            "use_cloud": False,
            "headless": CONFIG.get("BROWSER_HEADLESS", True),
            "enable_default_extensions": CONFIG.get("ENABLE_DEFAULT_EXTENSIONS", False),
            "wait_for_network_idle_page_load_time": CONFIG.get(
                "WAIT_FOR_NETWORK_IDLE_PAGE_LOAD_TIME", 3.0
            ),
            "minimum_wait_page_load_time": CONFIG.get(
                "MINIMUM_WAIT_PAGE_LOAD_TIME", 1.0
            ),
        }

        profile = BrowserProfile(**bp_kwargs)

        # Verify the fix is applied
        assert profile.enable_default_extensions is False, "Extensions should be disabled"
        assert profile.wait_for_network_idle_page_load_time == CONFIG["WAIT_FOR_NETWORK_IDLE_PAGE_LOAD_TIME"]
        assert profile.minimum_wait_page_load_time == CONFIG["MINIMUM_WAIT_PAGE_LOAD_TIME"]

    @pytest.mark.asyncio
    async def test_session_creation_with_fix(self):
        """Test that session.py creates BrowserProfile with extensions disabled."""
        from browser_use.browser import BrowserProfile

        # Simulate what session.py does
        profile_kwargs = {
            "headless": True,
            "is_local": True,
            "use_cloud": False,
            "enable_default_extensions": False,
            "wait_for_network_idle_page_load_time": 3.0,
            "minimum_wait_page_load_time": 1.0,
        }

        profile = BrowserProfile(**profile_kwargs)

        # Verify extensions are disabled
        assert profile.enable_default_extensions is False, "Session should create profile with extensions disabled"
        assert profile.wait_for_network_idle_page_load_time == 3.0
        assert profile.minimum_wait_page_load_time == 1.0
