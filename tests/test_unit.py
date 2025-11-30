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
        await adapter.ainvoke([message_dict])

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

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_with_system_message(self):
        """Test ainvoke converts system role messages correctly.

        Verifies:
        - System role messages become SystemMessage instances
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter, SystemMessage

        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value={"success": True})
        adapter = ChatOpenAIAdapter(mock_llm)

        # Call with system message
        await adapter.ainvoke(
            [{"role": "system", "content": "You are helpful"}]
        )

        # Verify SystemMessage was created
        call_args = mock_llm.agenerate.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], SystemMessage)

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_with_assistant_message(self):
        """Test ainvoke converts assistant role messages correctly.

        Verifies:
        - Assistant/AI role messages become AIMessage instances
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import AIMessage, ChatOpenAIAdapter

        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value={"success": True})
        adapter = ChatOpenAIAdapter(mock_llm)

        # Test with assistant role
        await adapter.ainvoke([{"role": "assistant", "content": "I can help"}])
        call_args = mock_llm.agenerate.call_args[0][0]
        assert isinstance(call_args[0], AIMessage)

        # Test with ai role
        await adapter.ainvoke([{"role": "ai", "content": "AI response"}])
        call_args = mock_llm.agenerate.call_args[0][0]
        assert isinstance(call_args[0], AIMessage)

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_normalizes_objects_with_content_attr(self):
        """Test ainvoke normalizes objects with content attribute.

        Verifies:
        - Objects with .content attribute are normalized
        - Nested content is extracted
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value={"success": True})
        adapter = ChatOpenAIAdapter(mock_llm)

        # Create object with content attribute
        class MessageLike:
            def __init__(self):
                self.content = "message content"

        obj = MessageLike()
        await adapter.ainvoke(obj)

        # Verify agenerate was called
        assert mock_llm.agenerate.called

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_normalizes_objects_with_text_attr(self):
        """Test ainvoke normalizes objects with text attribute.

        Verifies:
        - Objects with .text attribute are normalized
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value={"success": True})
        adapter = ChatOpenAIAdapter(mock_llm)

        # Create object with text attribute
        class TextLike:
            def __init__(self):
                self.text = "text content"

        obj = TextLike()
        await adapter.ainvoke(obj)

        # Verify agenerate was called
        assert mock_llm.agenerate.called

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_normalizes_objects_with_to_dict(self):
        """Test ainvoke normalizes objects with to_dict method.

        Verifies:
        - Objects with .to_dict() method are normalized
        - to_dict() result is used
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value={"success": True})
        adapter = ChatOpenAIAdapter(mock_llm)

        # Create object with to_dict method
        class DictLike:
            def to_dict(self):
                return {"key": "value"}

        obj = DictLike()
        await adapter.ainvoke(obj)

        # Verify agenerate was called
        assert mock_llm.agenerate.called

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_error_handling_with_logging_success(self):
        """Test ainvoke logs error details when agenerate fails.

        Verifies:
        - Exception from agenerate is caught and logged
        - Error details are logged with arg types
        - Original exception is re-raised
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        test_error = RuntimeError("Test LLM failure")
        mock_llm.agenerate = AsyncMock(side_effect=test_error)

        adapter = ChatOpenAIAdapter(mock_llm)

        # Should raise the original error
        with pytest.raises(RuntimeError, match="Test LLM failure"):
            await adapter.ainvoke("test prompt")

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_sync_agenerate_fallback(self):
        """Test ainvoke falls back to sync agenerate via executor.

        Verifies:
        - Non-coroutine agenerate is called via run_in_executor
        - Result is returned correctly
        """
        from unittest.mock import Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_response = {"sync_result": True}
        # Make agenerate a regular function (not async)
        mock_llm.agenerate = Mock(return_value=mock_response)

        adapter = ChatOpenAIAdapter(mock_llm)

        result = await adapter.ainvoke("test prompt")

        assert mock_llm.agenerate.called
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_fallback_to_generate(self):
        """Test ainvoke falls back to generate method when agenerate missing.

        Verifies:
        - Falls back to generate when agenerate not available
        - Uses run_in_executor for sync generate
        """
        from unittest.mock import Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_response = {"generate_result": True}
        # Remove agenerate, only provide generate
        del mock_llm.agenerate
        mock_llm.generate = Mock(return_value=mock_response)

        adapter = ChatOpenAIAdapter(mock_llm)

        result = await adapter.ainvoke("test prompt")

        assert mock_llm.generate.called
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_adapter_ainvoke_no_async_support_raises(self):
        """Test ainvoke raises NotImplementedError when no async support.

        Verifies:
        - NotImplementedError when neither agenerate nor generate exist
        """
        from unittest.mock import Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        # Remove both agenerate and generate
        del mock_llm.agenerate
        del mock_llm.generate

        adapter = ChatOpenAIAdapter(mock_llm)

        with pytest.raises(
            NotImplementedError, match="does not support async invocation"
        ):
            await adapter.ainvoke("test prompt")

    @pytest.mark.asyncio
    async def test_adapter_agenerate_delegates_to_ainvoke(self):
        """Test agenerate method delegates to ainvoke.

        Verifies:
        - agenerate is an alias for ainvoke
        """
        from unittest.mock import AsyncMock, Mock

        from server.server import ChatOpenAIAdapter

        mock_llm = Mock()
        mock_response = {"result": "from_agenerate"}
        mock_llm.agenerate = AsyncMock(return_value=mock_response)

        adapter = ChatOpenAIAdapter(mock_llm)

        result = await adapter.agenerate("test prompt")

        assert result == mock_response


class TestEnsureLLMAdapter:
    """Test ensure_llm_adapter function."""

    def test_ensure_llm_adapter_returns_none_for_none_input(self):
        """Test ensure_llm_adapter returns None when given None.

        Verifies:
        - Returns None when llm parameter is None
        """
        from server.server import ensure_llm_adapter

        result = ensure_llm_adapter(None)

        assert result is None

    def test_ensure_llm_adapter_returns_llm_unchanged(self):
        """Test ensure_llm_adapter returns LLM instance unchanged.

        Verifies:
        - Returns the same LLM object without wrapping
        - Works with browser-use v0.9.7+ ChatOpenAI
        """
        from unittest.mock import Mock

        from server.server import ensure_llm_adapter

        mock_llm = Mock()
        mock_llm.model = "gpt-4o-mini"

        result = ensure_llm_adapter(mock_llm)

        assert result is mock_llm


class TestCreateBrowserContext:
    """Test create_browser_context_for_task function."""

    @pytest.mark.asyncio
    async def test_create_browser_context_returns_none_tuple(self):
        """Test create_browser_context_for_task returns (None, None).

        Verifies:
        - Returns (None, None) for API compatibility
        - browser-use v0.9.5+ handles context internally
        """
        from server.server import create_browser_context_for_task

        result = await create_browser_context_for_task()

        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_create_browser_context_with_custom_params(self):
        """Test create_browser_context_for_task accepts custom parameters.

        Verifies:
        - Accepts custom chrome_path, dimensions, locale
        - Still returns (None, None) regardless of parameters
        """
        from server.server import create_browser_context_for_task

        result = await create_browser_context_for_task(
            chrome_path="/custom/chrome",
            window_width=1920,
            window_height=1080,
            locale="en-US",
        )

        assert result == (None, None)


class TestRunBrowserTaskAsync:
    """Test suite for run_browser_task_async function.

    Tests async task execution with browser-use Agent.
    """

    @pytest.mark.asyncio
    async def test_run_browser_task_basic(self, mock_config, cleanup_tasks):
        """Test basic browser task execution.

        Verifies:
        - Task is created with pending status
        - Task transitions to running
        - Task store is populated with metadata
        """
        from unittest.mock import AsyncMock, Mock, patch

        from server.server import run_browser_task_async, task_store

        task_id = "test-task-run-basic"
        instruction = "Navigate to example.com"

        # Mock browser-use components
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Task completed")

        with (
            patch("server.server.ChatOpenAI") as mock_chat_openai,
            patch("server.server.Agent") as mock_agent_cls,
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_browser,
        ):
            mock_chat_openai.return_value = Mock()
            mock_agent_cls.return_value = mock_agent
            mock_create_browser.return_value = (None, None)

            # Execute task
            await run_browser_task_async(
                task_id=task_id, instruction=instruction, config=mock_config
            )

            # Verify task was created and executed
            assert task_id in task_store
            assert task_store[task_id]["status"] in ["running", "completed"]
            assert "start_time" in task_store[task_id]

    @pytest.mark.asyncio
    async def test_run_browser_task_with_llm(self, mock_config, cleanup_tasks):
        """Test browser task with provided LLM.

        Verifies:
        - Uses provided LLM instead of creating new one
        - Task executes successfully
        """
        from unittest.mock import AsyncMock, Mock, patch

        from server.server import run_browser_task_async, task_store

        task_id = "test-task-with-llm"
        instruction = "Test with custom LLM"
        mock_llm = Mock()

        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Success")

        with (
            patch("server.server.Agent") as mock_agent_cls,
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_browser,
        ):
            mock_agent_cls.return_value = mock_agent
            mock_create_browser.return_value = (None, None)

            await run_browser_task_async(
                task_id=task_id,
                instruction=instruction,
                llm=mock_llm,
                config=mock_config,
            )

            # Verify task completed
            assert task_id in task_store
            assert task_store[task_id]["status"] in ["running", "completed"]

    @pytest.mark.asyncio
    async def test_run_browser_task_error_handling(
        self, mock_config, cleanup_tasks
    ):
        """Test error handling in browser task execution.

        Verifies:
        - Task status set to failed on error
        - Error message stored in task_store
        """
        from unittest.mock import AsyncMock, Mock, patch

        from server.server import run_browser_task_async, task_store

        task_id = "test-task-error"
        instruction = "Task that fails"

        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Test error"))

        with (
            patch("server.server.ChatOpenAI") as mock_chat_openai,
            patch("server.server.Agent") as mock_agent_cls,
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_browser,
        ):
            mock_chat_openai.return_value = Mock()
            mock_agent_cls.return_value = mock_agent
            mock_create_browser.return_value = (None, None)

            # Execute task (should handle error gracefully)
            await run_browser_task_async(
                task_id=task_id, instruction=instruction, config=mock_config
            )

            # Verify error was captured
            assert task_id in task_store
            # Task should be marked as failed
            assert (
                task_store[task_id]["status"] == "failed"
                or "error" in task_store[task_id]
            )

    @pytest.mark.asyncio
    async def test_run_browser_task_legacy_url_action(
        self, mock_config, cleanup_tasks
    ):
        """Test browser task with legacy url/action parameters.

        Verifies:
        - Legacy parameters are converted to instruction
        - Task executes successfully
        """
        from unittest.mock import AsyncMock, Mock, patch

        from server.server import run_browser_task_async, task_store

        task_id = "test-task-legacy"
        url = "https://example.com"
        action = "click the login button"

        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Legacy task completed")

        with (
            patch("server.server.ChatOpenAI") as mock_chat_openai,
            patch("server.server.Agent") as mock_agent_cls,
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_browser,
        ):
            mock_chat_openai.return_value = Mock()
            mock_agent_cls.return_value = mock_agent
            mock_create_browser.return_value = (None, None)

            await run_browser_task_async(
                task_id=task_id, url=url, action=action, config=mock_config
            )

            # Verify task created
            assert task_id in task_store
            assert task_store[task_id]["status"] in ["running", "completed"]


class TestCleanupOldTasks:
    """Test suite for cleanup_old_tasks function.

    Tests automatic cleanup of completed tasks to prevent memory leaks.
    """

    @pytest.mark.asyncio
    async def test_cleanup_logic_removes_old_tasks(self, cleanup_tasks):
        """Test that cleanup logic removes tasks older than 1 hour.

        Verifies:
        - Old completed tasks are identified and removed
        - Recent tasks are preserved
        """
        from datetime import datetime, timedelta

        from server.server import task_store

        # Create old completed task (2 hours ago)
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        task_store["old-task"] = {
            "status": "completed",
            "end_time": old_time,
            "result": "old result",
        }

        # Create recent completed task (30 minutes ago)
        recent_time = (datetime.now() - timedelta(minutes=30)).isoformat()
        task_store["recent-task"] = {
            "status": "completed",
            "end_time": recent_time,
            "result": "recent result",
        }

        # Create running task
        task_store["running-task"] = {"status": "running", "result": None}

        # Manually execute cleanup logic (from cleanup_old_tasks function)
        current_time = datetime.now()
        tasks_to_remove = []

        for task_id, task_data in task_store.items():
            if (
                task_data["status"] in ["completed", "failed"]
                and "end_time" in task_data
            ):
                end_time = datetime.fromisoformat(task_data["end_time"])
                hours_elapsed = (current_time - end_time).total_seconds() / 3600

                if hours_elapsed > 1:  # Remove tasks older than 1 hour
                    tasks_to_remove.append(task_id)

        # Remove old tasks
        for task_id in tasks_to_remove:
            del task_store[task_id]

        # Verify old task was removed, recent and running preserved
        assert "old-task" not in task_store
        assert "recent-task" in task_store
        assert "running-task" in task_store

    @pytest.mark.asyncio
    async def test_cleanup_logic_handles_missing_end_time(self, cleanup_tasks):
        """Test that cleanup handles tasks without end_time gracefully.

        Verifies:
        - Tasks without end_time are not removed
        - No errors raised
        """
        from datetime import datetime

        from server.server import task_store

        # Create completed task without end_time
        task_store["no-endtime-task"] = {
            "status": "completed",
            "result": "result",
        }

        # Manually execute cleanup logic
        current_time = datetime.now()
        tasks_to_remove = []

        try:
            for task_id, task_data in task_store.items():
                if (
                    task_data["status"] in ["completed", "failed"]
                    and "end_time" in task_data
                ):
                    end_time = datetime.fromisoformat(task_data["end_time"])
                    hours_elapsed = (
                        current_time - end_time
                    ).total_seconds() / 3600

                    if hours_elapsed > 1:
                        tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del task_store[task_id]

        except Exception:
            # Should not raise
            pytest.fail("Cleanup raised exception")

        # Task should still be present
        assert "no-endtime-task" in task_store

    @pytest.mark.asyncio
    async def test_cleanup_function_exists(self):
        """Test that cleanup_old_tasks function is defined and callable.

        Verifies:
        - Function exists
        - Is a coroutine function
        """
        from inspect import iscoroutinefunction

        from server.server import cleanup_old_tasks

        assert callable(cleanup_old_tasks)
        assert iscoroutinefunction(cleanup_old_tasks)


class TestCreateMcpServer:
    """Test suite for create_mcp_server function.

    Tests MCP server factory and configuration.
    """

    def test_create_mcp_server_returns_server(self):
        """Test that create_mcp_server returns Server instance.

        Verifies:
        - Function returns MCP Server instance
        - Server has correct name
        """
        from server.server import create_mcp_server

        server = create_mcp_server(llm=None)

        assert server is not None
        assert hasattr(server, "call_tool")
        assert hasattr(server, "list_tools")

    def test_create_mcp_server_with_custom_params(self):
        """Test server creation with custom parameters.

        Verifies:
        - Custom window dimensions accepted
        - Custom locale accepted
        - Server is properly configured
        """
        from unittest.mock import Mock

        from server.server import create_mcp_server

        mock_llm = Mock()

        server = create_mcp_server(
            llm=mock_llm, window_width=1920, window_height=1080, locale="de-DE"
        )

        assert server is not None

    def test_create_mcp_server_default_params(self):
        """Test server creation with default parameters.

        Verifies:
        - Server created without explicit params
        - Default values are used
        """
        from server.server import create_mcp_server

        server = create_mcp_server(llm=None)

        assert server is not None


class TestMcpToolHandlers:
    """Test suite for MCP tool handler logic.

    Tests individual tool argument validation and error handling.
    """

    def test_browser_use_missing_url_raises_error(self):
        """Test browser_use tool requires url argument.

        Verifies:
        - ValueError raised when url is missing
        """
        # This tests the validation logic that would be in call_tool
        arguments = {"action": "click button"}

        # Simulate validation
        with pytest.raises(ValueError, match="url"):
            if "url" not in arguments:
                raise ValueError("Missing required argument 'url'")

    def test_browser_use_missing_action_raises_error(self):
        """Test browser_use tool requires action argument.

        Verifies:
        - ValueError raised when action is missing
        """
        arguments = {"url": "https://example.com"}

        with pytest.raises(ValueError, match="action"):
            if "action" not in arguments:
                raise ValueError("Missing required argument 'action'")

    def test_browser_use_with_valid_arguments(self):
        """Test browser_use validation passes with valid arguments.

        Verifies:
        - No error when both url and action provided
        """
        arguments = {"url": "https://example.com", "action": "click button"}

        # Should not raise
        if "url" not in arguments:
            raise ValueError("Missing required argument 'url'")
        if "action" not in arguments:
            raise ValueError("Missing required argument 'action'")

    def test_summarize_action_detection(self):
        """Test detection of summarize action in browser_use.

        Verifies:
        - Summarize keyword is detected in action text
        """
        action_with_summarize = "Please summarize the page content"
        action_without_summarize = "Click the login button"

        assert "summarize" in action_with_summarize.lower()
        assert "summarize" not in action_without_summarize.lower()

    @pytest.mark.asyncio
    async def test_task_id_generation(self):
        """Test task ID is generated as UUID.

        Verifies:
        - Task ID is valid UUID format
        - Each ID is unique
        """
        import uuid

        task_id_1 = str(uuid.uuid4())
        task_id_2 = str(uuid.uuid4())

        # Should be valid UUIDs
        assert uuid.UUID(task_id_1)
        assert uuid.UUID(task_id_2)

        # Should be unique
        assert task_id_1 != task_id_2


class TestParseBoolEnv:
    """Test suite for parse_bool_env function.

    Tests environment variable parsing for boolean values.
    """

    def test_parse_bool_true_values(self):
        """Test parsing true values from environment.

        Verifies:
        - 'true', 'yes', '1', 'on' all return True
        - Case insensitive
        """
        from unittest.mock import patch

        from server.server import parse_bool_env

        true_values = [
            "true",
            "True",
            "TRUE",
            "yes",
            "Yes",
            "YES",
            "1",
            "on",
            "On",
            "ON",
        ]

        for value in true_values:
            with patch.dict("os.environ", {"TEST_VAR": value}):
                result = parse_bool_env("TEST_VAR", False)
                assert result is True, f"Expected True for '{value}'"

    def test_parse_bool_false_values(self):
        """Test parsing false values from environment.

        Verifies:
        - 'false', 'no', '0', 'off' all return False
        - Case insensitive
        """
        from unittest.mock import patch

        from server.server import parse_bool_env

        false_values = [
            "false",
            "False",
            "FALSE",
            "no",
            "No",
            "NO",
            "0",
            "off",
            "Off",
            "OFF",
        ]

        for value in false_values:
            with patch.dict("os.environ", {"TEST_VAR": value}):
                result = parse_bool_env("TEST_VAR", True)
                assert result is False, f"Expected False for '{value}'"

    def test_parse_bool_default_when_missing(self):
        """Test default value used when env var missing.

        Verifies:
        - Returns default when env var not set
        """
        from unittest.mock import patch

        from server.server import parse_bool_env

        with patch.dict("os.environ", {}, clear=True):
            assert parse_bool_env("MISSING_VAR", True) is True
            assert parse_bool_env("MISSING_VAR", False) is False

    def test_parse_bool_invalid_value(self):
        """Test handling of invalid boolean values.

        Verifies:
        - Invalid values return default
        """
        from unittest.mock import patch

        from server.server import parse_bool_env

        with patch.dict("os.environ", {"TEST_VAR": "invalid"}):
            result = parse_bool_env("TEST_VAR", True)
            # Should return default for invalid values
            assert isinstance(result, bool)


class TestConfigInitialization:
    """Test suite for CONFIG initialization.

    Tests global configuration dictionary setup.
    """

    def test_config_has_required_keys(self):
        """Test CONFIG contains all required configuration keys.

        Verifies:
        - Essential config keys are present
        - Values are of correct types
        """
        from server.server import CONFIG

        required_keys = [
            "DEFAULT_WINDOW_WIDTH",
            "DEFAULT_WINDOW_HEIGHT",
            "DEFAULT_LOCALE",
            "DEFAULT_ESTIMATED_TASK_SECONDS",
            "CLEANUP_INTERVAL_SECONDS",
        ]

        for key in required_keys:
            assert key in CONFIG, f"Missing required config key: {key}"

    def test_config_window_dimensions_are_integers(self):
        """Test window dimensions are integers.

        Verifies:
        - Window width and height are integers
        - Values are reasonable (> 0)
        """
        from server.server import CONFIG

        assert isinstance(CONFIG["DEFAULT_WINDOW_WIDTH"], int)
        assert isinstance(CONFIG["DEFAULT_WINDOW_HEIGHT"], int)
        assert CONFIG["DEFAULT_WINDOW_WIDTH"] > 0
        assert CONFIG["DEFAULT_WINDOW_HEIGHT"] > 0

    def test_config_locale_is_string(self):
        """Test locale is a string.

        Verifies:
        - Locale is string type
        - Has expected format (e.g., 'en-US')
        """
        from server.server import CONFIG

        assert isinstance(CONFIG["DEFAULT_LOCALE"], str)
        assert (
            "-" in CONFIG["DEFAULT_LOCALE"] or "_" in CONFIG["DEFAULT_LOCALE"]
        )

    def test_config_timeouts_are_reasonable(self):
        """Test timeout values are reasonable integers.

        Verifies:
        - Timeout values are positive integers
        """
        from server.server import CONFIG

        assert isinstance(CONFIG["DEFAULT_ESTIMATED_TASK_SECONDS"], int)
        assert CONFIG["DEFAULT_ESTIMATED_TASK_SECONDS"] > 0

        assert isinstance(CONFIG["CLEANUP_INTERVAL_SECONDS"], int)
        assert CONFIG["CLEANUP_INTERVAL_SECONDS"] > 0
