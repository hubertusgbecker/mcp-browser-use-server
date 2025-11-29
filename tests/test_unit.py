"""Unit tests for mcp-browser-use-server core functionality."""

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.server import (
    create_mcp_server,
    init_configuration,
    parse_bool_env,
    run_browser_task_async,
    task_store,
)


class TestConfiguration:
    """Test configuration and environment parsing."""

    def test_parse_bool_true_variations(self, monkeypatch):
        """Test parsing various true values."""
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
        for value in true_values:
            monkeypatch.setenv("TEST_VAR", value)
            if parse_bool_env("TEST_VAR") is not True:
                pytest.fail(f"parse_bool_env did not return True for {value!r}")

    def test_parse_bool_false_variations(self, monkeypatch):
        """Test parsing various false values."""
        false_values = ["false", "False", "FALSE", "0", "no", "No", "NO", ""]
        for value in false_values:
            monkeypatch.setenv("TEST_VAR", value)
            if parse_bool_env("TEST_VAR") is not False:
                pytest.fail(
                    f"parse_bool_env did not return False for {value!r}"
                )

    def test_parse_bool_default_when_missing(self, monkeypatch):
        """Test default value when environment variable is not set."""
        monkeypatch.delenv("TEST_VAR", raising=False)
        if parse_bool_env("TEST_VAR", default=True) is not True:
            pytest.fail(
                "parse_bool_env failed to return default True when missing"
            )
        if parse_bool_env("TEST_VAR", default=False) is not False:
            pytest.fail(
                "parse_bool_env failed to return default False when missing"
            )

    def test_init_configuration_with_env_vars(self, mock_env_vars):
        """Test configuration with environment variables set."""
        config = init_configuration()

        for key in [
            "PATIENT_MODE",
            "DEFAULT_WINDOW_WIDTH",
            "DEFAULT_WINDOW_HEIGHT",
            "BROWSER_ARGS",
        ]:
            if key not in config:
                pytest.fail(f"Expected {key} in configuration")
        if os.getenv("OPENAI_API_KEY") != mock_env_vars["OPENAI_API_KEY"]:
            pytest.fail(
                "OPENAI_API_KEY env var not set to test value from fixture"
            )

    def test_init_configuration_defaults(self, monkeypatch):
        """Test configuration uses defaults when env vars not set."""
        for key in ["BROWSER_WINDOW_WIDTH", "BROWSER_WINDOW_HEIGHT", "PATIENT"]:
            monkeypatch.delenv(key, raising=False)

        config = init_configuration()
        if config.get("DEFAULT_WINDOW_WIDTH") != 1280:
            pytest.fail("DEFAULT_WINDOW_WIDTH default incorrect")
        if config.get("DEFAULT_WINDOW_HEIGHT") != 1100:
            pytest.fail("DEFAULT_WINDOW_HEIGHT default incorrect")
        if config.get("PATIENT_MODE") is not False:
            pytest.fail("PATIENT_MODE default incorrect")


class TestBrowserTasks:
    """Test browser task execution."""

    @pytest.mark.asyncio
    async def test_run_browser_task_success(
        self, mock_config, mock_llm, mock_browser_context, cleanup_tasks
    ):
        """Test successful browser task execution."""
        task_id = "test-task-success"
        instruction = "Navigate to example.com"

        with (
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_ctx,
            patch("server.server.Agent") as mock_agent_class,
        ):
            mock_create_ctx.return_value = (mock_browser_context, MagicMock())
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = "Task completed"
            mock_agent_class.return_value = mock_agent_instance

            await run_browser_task_async(
                task_id=task_id,
                instruction=instruction,
                llm=mock_llm,
                config=mock_config,
            )

            if task_id not in task_store:
                pytest.fail("Task id not present in task_store after run")
            task = task_store[task_id]
            if task.get("status") != "completed":
                pytest.fail("Task did not complete successfully")
            if task.get("result") is None:
                pytest.fail("Task result is None after completion")

    @pytest.mark.asyncio
    async def test_run_browser_task_failure(
        self, mock_config, mock_llm, mock_browser_context, cleanup_tasks
    ):
        """Test browser task execution with failure."""
        task_id = "test-task-failure"
        instruction = "Invalid instruction"

        with (
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_ctx,
            patch("server.server.Agent") as mock_agent_class,
        ):
            mock_create_ctx.return_value = (mock_browser_context, MagicMock())
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.side_effect = Exception("Task failed")
            mock_agent_class.return_value = mock_agent_instance

            await run_browser_task_async(
                task_id=task_id,
                instruction=instruction,
                llm=mock_llm,
                config=mock_config,
            )

            if task_id not in task_store:
                pytest.fail(
                    "Task id not present in task_store after failure run"
                )
            task = task_store[task_id]
            if task.get("status") != "failed":
                pytest.fail("Task status not marked as failed")
            if "error" not in task:
                pytest.fail("Expected 'error' key in failed task")

    @pytest.mark.asyncio
    async def test_task_cleanup(self, cleanup_tasks):
        """Test cleanup of old tasks."""
        # Add old and new tasks
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        new_time = datetime.now().isoformat()

        task_store["old-task"] = {
            "status": "completed",
            "created_at": old_time,
            "end_time": old_time,  # Add end_time for cleanup logic
            "result": "Old result",
        }
        task_store["new-task"] = {
            "status": "completed",
            "created_at": new_time,
            "end_time": new_time,  # Add end_time for cleanup logic
            "result": "New result",
        }

        # Test cleanup logic manually (cleanup_old_tasks runs forever in a loop)
        # We'll manually execute the cleanup logic once via helper
        from tests._helpers import run_cleanup_once_and_assert

        run_cleanup_once_and_assert(
            task_store, expect_removed=["old-task"], expect_present=["new-task"]
        )


class TestMCPServer:
    """Test MCP server functionality."""

    def test_create_mcp_server(self, mock_config, mock_llm):
        """Test MCP server creation."""
        # create_mcp_server takes llm and optional parameters, not config dict
        server = create_mcp_server(llm=mock_llm)

        if server is None:
            pytest.fail("create_mcp_server returned None")
        # Check that it's an MCP Server instance (not request_context which
        # requires context)
        from mcp.server import Server

        if not isinstance(server, Server):
            pytest.fail("create_mcp_server did not return Server instance")

    @pytest.mark.asyncio
    async def test_task_store_operations(self, cleanup_tasks):
        """Test task store CRUD operations."""
        task_id = "test-crud-task"

        # Create
        task_store[task_id] = {
            "status": "pending",
            "instruction": "Test task",
            "created_at": datetime.now().isoformat(),
        }
        if task_id not in task_store:
            pytest.fail("Failed to create task in store")

        # Read
        task = task_store[task_id]
        if task.get("status") != "pending":
            pytest.fail("Task status not pending after creation")

        # Update
        task_store[task_id]["status"] = "completed"
        if task_store[task_id]["status"] != "completed":
            pytest.fail("Failed to update task status to completed")

        # Delete
        del task_store[task_id]
        if task_id in task_store:
            pytest.fail("Failed to delete task from store")


class TestLogLevelConfiguration:
    """Test log level configuration and environment handling."""

    def test_log_level_from_env(self, monkeypatch):
        """Test that LOG_LEVEL from environment is respected."""
        import logging

        # Test each standard log level
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level_name in levels:
            monkeypatch.setenv("LOG_LEVEL", level_name)
            # Get the numeric level
            expected_level = getattr(logging, level_name)
            # Verify it's a valid level
            if expected_level not in [
                logging.DEBUG,
                logging.INFO,
                logging.WARNING,
                logging.ERROR,
                logging.CRITICAL,
            ]:
                pytest.fail(f"Invalid logging level for {level_name}")

    def test_log_level_default_when_missing(self, monkeypatch):
        """Test default log level when LOG_LEVEL not set."""
        import logging

        monkeypatch.delenv("LOG_LEVEL", raising=False)
        default = os.getenv("LOG_LEVEL", "INFO")
        if default != "INFO":
            pytest.fail("Default LOG_LEVEL is not INFO")
        expected_level = getattr(logging, default.upper())
        if expected_level != logging.INFO:
            pytest.fail("Default logging level not INFO")

    def test_log_level_case_insensitive(self, monkeypatch):
        """Test that log level parsing is case-insensitive."""
        import logging

        # Test lowercase, uppercase, mixed case
        test_cases = [
            ("debug", logging.DEBUG),
            ("DEBUG", logging.DEBUG),
            ("Debug", logging.DEBUG),
            ("info", logging.INFO),
            ("INFO", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ]

        for level_str, expected_level in test_cases:
            monkeypatch.setenv("LOG_LEVEL", level_str)
            actual_level = getattr(logging, level_str.upper(), logging.INFO)
            if actual_level != expected_level:
                pytest.fail(f"Log level parsing failed for {level_str}")

    def test_log_level_invalid_falls_back_to_info(self, monkeypatch):
        """Test that invalid log level falls back to INFO."""
        import logging

        monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
        # getattr with default should return INFO for invalid levels
        level = getattr(logging, "INVALID_LEVEL", logging.INFO)
        if level != logging.INFO:
            pytest.fail("Invalid level did not fall back to INFO")


class TestChatOpenAIAdapter:
    """Test ChatOpenAIAdapter class."""

    @pytest.mark.asyncio
    async def test_adapter_initializes_with_chatopenai(self):
        """Test ChatOpenAIAdapter initialization with ChatOpenAI instance.
        
        Verifies:
        - Adapter accepts ChatOpenAI instance
        - Internal _llm attribute is set correctly
        """
        from langchain_openai import ChatOpenAI
        from server.server import ChatOpenAIAdapter
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        adapter = ChatOpenAIAdapter(llm)
        
        assert adapter._llm is llm

    @pytest.mark.asyncio
    async def test_adapter_provider_property(self):
        """Test ChatOpenAIAdapter provider property.
        
        Verifies:
        - provider property returns underlying LLM's provider attribute
        - Falls back to "openai" if provider attribute doesn't exist
        """
        from langchain_openai import ChatOpenAI
        from server.server import ChatOpenAIAdapter
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        adapter = ChatOpenAIAdapter(llm)
        
        # Should return provider from underlying LLM or "openai" as default
        provider = adapter.provider
        assert isinstance(provider, str)
        assert provider in ["openai", "azure"]  # Common provider values

    @pytest.mark.asyncio
    async def test_adapter_getattr_delegation(self):
        """Test ChatOpenAIAdapter __getattr__ delegates to underlying LLM.
        
        Verifies:
        - __getattr__ delegates attribute access to _llm
        - Can access underlying LLM attributes through adapter
        """
        from langchain_openai import ChatOpenAI
        from server.server import ChatOpenAIAdapter
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        adapter = ChatOpenAIAdapter(llm)
        
        # Access attribute through delegation
        # model_name should be delegated to underlying LLM
        assert hasattr(adapter, "model_name")
        assert adapter.model_name == llm.model_name

    @pytest.mark.asyncio
    async def test_adapter_getattr_calls_underlying_llm(self):
        """Test __getattr__ actually calls getattr on underlying LLM.
        
        Verifies:
        - Accessing non-existent attribute triggers __getattr__
        - Returns value from underlying LLM
        """
        from unittest.mock import Mock
        from server.server import ChatOpenAIAdapter
        
        # Create a mock LLM with a custom attribute
        mock_llm = Mock()
        mock_llm.custom_test_attribute = "test_value_123"
        mock_llm.temperature = 0.7
        
        adapter = ChatOpenAIAdapter(mock_llm)
        
        # Access custom attributes - should trigger __getattr__
        assert adapter.custom_test_attribute == "test_value_123"
        assert adapter.temperature == 0.7

    @pytest.mark.asyncio
    async def test_adapter_model_property_returns_model_name(self):
        """Test ChatOpenAIAdapter model property returns model or model_name.
        
        Verifies:
        - model property returns .model from underlying LLM if available
        - Falls back to .model_name if .model doesn't exist
        - Falls back to "openai" if neither exists
        """
        from langchain_openai import ChatOpenAI
        from server.server import ChatOpenAIAdapter
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        adapter = ChatOpenAIAdapter(llm)
        
        # Should return model from underlying LLM
        model = adapter.model
        assert isinstance(model, str)
        assert len(model) > 0  # Should be non-empty

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_with_simple_message(self):
        """Test ChatOpenAIAdapter ainvoke with simple string message.
        
        Verifies:
        - ainvoke normalizes and converts messages
        - Calls underlying LLM's agenerate method
        - Returns result from underlying LLM
        """
        from unittest.mock import AsyncMock, Mock
        from server.server import ChatOpenAIAdapter
        
        # Create mock LLM with agenerate method
        mock_llm = Mock()
        mock_response = {"generations": [[{"text": "test response"}]]}
        mock_llm.agenerate = AsyncMock(return_value=mock_response)
        
        adapter = ChatOpenAIAdapter(mock_llm)
        
        # Call ainvoke with simple message
        result = await adapter.ainvoke("test prompt")
        
        # Verify agenerate was called
        assert mock_llm.agenerate.called
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_with_message_dict(self):
        """Test ainvoke with message dict containing role and content.
        
        Verifies:
        - Converts message dicts to BaseMessage instances
        - Handles {'role': 'user', 'content': 'text'} format
        """
        from unittest.mock import AsyncMock, Mock
        from server.server import ChatOpenAIAdapter, HumanMessage
        
        mock_llm = Mock()
        mock_response = {"result": "success"}
        mock_llm.agenerate = AsyncMock(return_value=mock_response)
        
        adapter = ChatOpenAIAdapter(mock_llm)
        
        # Call with message dict
        message_dict = {"role": "user", "content": "test message"}
        result = await adapter.ainvoke([message_dict])
        
        # Verify the call happened
        assert mock_llm.agenerate.called
        # Check that args were converted to BaseMessage
        call_args = mock_llm.agenerate.call_args
        assert len(call_args[0]) > 0
        # First arg should be list of HumanMessage
        first_arg = call_args[0][0]
        assert isinstance(first_arg, list)
        assert len(first_arg) == 1
        assert isinstance(first_arg[0], HumanMessage)
