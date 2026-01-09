"""Tests for MCP tool validation errors and edge cases."""

import pytest

from server.server import create_mcp_server, task_store


class TestValidationErrors:
    """Test MCP tool validation and error handling."""

    @pytest.mark.asyncio
    async def test_browser_navigate_missing_url(self, mock_llm, cleanup_tasks):
        """Test browser_navigate returns error when url is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_navigate", arguments={}
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        # MCP validation errors can be plain text or JSON
        assert "url" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_click_missing_session_id(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_click returns error when session_id is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_click", arguments={"element_index": 0}
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert "session" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_click_missing_index(self, mock_llm, cleanup_tasks):
        """Test browser_click returns error when element_index is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_click", arguments={"session_id": "test"}
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert (
            "index" in response_text.lower()
            or "element" in response_text.lower()
        )
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_extract_missing_session_id(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_extract_content returns error when session_id is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_extract_content",
                    arguments={"instruction": "Extract text"},
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert "session" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_extract_missing_instruction(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_extract_content returns error when instruction is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_extract_content",
                    arguments={"session_id": "test"},
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert "instruction" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_close_session_missing_id(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_close_session returns error when session_id is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_close_session", arguments={}
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert "session" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_get_task_status_missing_id(self, mock_llm, cleanup_tasks):
        """Test get_task_status returns error when task_id is missing."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="get_task_status", arguments={}
                )
            )
        )

        response_text = result.root.content[0].text  # type: ignore[attr-defined,union-attr]
        assert "task" in response_text.lower()
        assert result.root.isError or "error" in response_text.lower()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_browser_get_state_missing_both_ids(
        self, mock_llm, cleanup_tasks
    ):
        """Test browser_get_state without task_id or session_id."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="browser_get_state", arguments={}
                )
            )
        )

        # browser_get_state without IDs might return empty state or error
        # Just verify it doesn't crash
        assert len(result.root.content) > 0  # type: ignore[attr-defined]


class TestTaskCleanup:
    """Test task lifecycle management."""

    @pytest.mark.asyncio
    async def test_task_store_operations(self, mock_config, cleanup_tasks):
        """Test basic task store operations."""
        from datetime import datetime

        # Add task to store
        task_id = "test-task-ops"
        task_store[task_id] = {
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }

        # Verify it's in store
        assert task_id in task_store

        # Update status
        task_store[task_id]["status"] = "completed"
        assert task_store[task_id]["status"] == "completed"

        # Remove task
        del task_store[task_id]
        assert task_id not in task_store


class TestResourceProvider:
    """Test MCP resource provider functionality."""

    @pytest.mark.asyncio
    async def test_list_resources_works(self, mock_llm, cleanup_tasks):
        """Test that list_resources returns successfully."""
        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        result = await server.request_handlers[types.ListResourcesRequest](
            types.ListResourcesRequest()
        )

        # Should return resources list
        assert hasattr(result.root, "resources")
        assert isinstance(result.root.resources, list)

    @pytest.mark.asyncio
    async def test_read_resource_with_valid_uri(self, mock_llm, cleanup_tasks):
        """Test reading a resource with valid URI."""
        # Add test task
        task_id = "test-resource-read"
        task_store[task_id] = {
            "status": "completed",
            "result": "test result",
        }

        server = create_mcp_server(llm=mock_llm)
        from mcp import types

        # Try to read the browser_tasks resource
        result = await server.request_handlers[types.ReadResourceRequest](
            types.ReadResourceRequest(
                params=types.ReadResourceRequestParams(
                    uri="resource://browser_tasks"  # type: ignore[arg-type]
                )
            )
        )

        # Should return content
        assert len(result.root.contents) > 0  # type: ignore[attr-defined]
        assert result.root.contents[0].text is not None  # type: ignore[attr-defined,union-attr]
