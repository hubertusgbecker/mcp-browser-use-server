"""Tests for browser task execution functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.server import (
    run_browser_task_async,
    task_store,
)


class TestBrowserTaskExecution:
    """Test browser task execution workflows."""

    @pytest.mark.asyncio
    async def test_run_browser_task_basic_workflow(self, mock_llm, cleanup_tasks):
        """Test basic browser task execution workflow.

        Covers:
        - Task initialization in task_store
        - Status transitions: pending → running → completed
        - Result storage
        """
        task_id = "test-basic-workflow"
        instruction = "Navigate to example.com"

        # Mock browser context creation
        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()

        # Mock Agent execution
        mock_agent_result = MagicMock()
        mock_agent_result.final_result = MagicMock(extracted_content="Test result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)
                mock_agent_instance = AsyncMock()
                mock_agent_instance.run.return_value = mock_agent_result
                mock_agent_cls.return_value = mock_agent_instance

                # Execute task
                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Verify task was created and completed
                assert task_id in task_store
                assert task_store[task_id]["status"] == "completed"
                assert task_store[task_id]["result"] is not None
                assert "start_time" in task_store[task_id]
                assert "end_time" in task_store[task_id]

    @pytest.mark.asyncio
    async def test_step_callback_execution(self, mock_llm, cleanup_tasks):
        """Test step callback during task execution.

        Covers lines 614-655: step_callback function
        """
        task_id = "test-step-callback"
        instruction = "Multi-step task"

        # Mock browser and agent
        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()

        mock_agent_result = MagicMock()
        mock_agent_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                # Capture step_callback when Agent is instantiated
                captured_step_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_step_callback
                    captured_step_callback = kwargs.get("on_step")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_agent_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                # Execute task
                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Verify step_callback was passed
                assert captured_step_callback is not None

                # Test step_callback with dict format (lines 617-620)
                mock_agent_output = {"step": 1}
                captured_step_callback(mock_agent_output)

                assert task_store[task_id]["progress"]["current_step"] == 1
                assert len(task_store[task_id]["progress"]["steps"]) > 0

    @pytest.mark.asyncio
    async def test_step_callback_with_args(self, mock_llm, cleanup_tasks):
        """Test step_callback with args format.

        Covers lines 621-623: multiple args format
        """
        task_id = "test-step-args"
        instruction = "Test"

        # Setup mocks
        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_step_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_step_callback
                    captured_step_callback = kwargs.get("on_step")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Call with args format: (?, agent_output, step_number)
                mock_agent_output = MagicMock()
                captured_step_callback(None, mock_agent_output, 2)

                assert task_store[task_id]["progress"]["current_step"] == 2

    @pytest.mark.asyncio
    async def test_step_callback_with_kwargs(self, mock_llm, cleanup_tasks):
        """Test step_callback with kwargs format.

        Covers lines 624-629: kwargs format
        """
        task_id = "test-step-kwargs"
        instruction = "Test"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_step_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_step_callback
                    captured_step_callback = kwargs.get("on_step")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Call with kwargs format
                captured_step_callback(step_number=3, agent_output=MagicMock())

                assert task_store[task_id]["progress"]["current_step"] == 3

    @pytest.mark.asyncio
    async def test_step_callback_without_step_number(self, mock_llm, cleanup_tasks):
        """Test step_callback when step number is missing.

        Covers lines 631-636: warning when no step number
        """
        task_id = "test-no-step"
        instruction = "Test"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_step_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_step_callback
                    captured_step_callback = kwargs.get("on_step")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Call without step number - should log warning and return early
                initial_step = task_store[task_id]["progress"]["current_step"]
                captured_step_callback(agent_output=MagicMock())

                # Step should not have been updated
                assert task_store[task_id]["progress"]["current_step"] == initial_step

    @pytest.mark.asyncio
    async def test_step_callback_with_agent_output_attributes(self, mock_llm, cleanup_tasks):
        """Test step_callback accessing agent_output attributes.

        Covers lines 645-653: accessing current_state.next_goal
        """
        task_id = "test-agent-attrs"
        instruction = "Test"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_step_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_step_callback
                    captured_step_callback = kwargs.get("on_step")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Create agent_output with current_state.next_goal
                mock_agent_output = MagicMock()
                mock_agent_output.current_state = MagicMock()
                mock_agent_output.current_state.next_goal = "Test goal"

                captured_step_callback({"step": 1}, agent_output=mock_agent_output)

                # Verify goal was captured in step info
                steps = task_store[task_id]["progress"]["steps"]
                assert any("goal" in step and step["goal"] == "Test goal" for step in steps)

    @pytest.mark.asyncio
    async def test_done_callback_execution(self, mock_llm, cleanup_tasks):
        """Test done_callback during task completion.

        Covers lines 658-677: done_callback function
        """
        task_id = "test-done-callback"
        instruction = "Test"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_done_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_done_callback
                    captured_done_callback = kwargs.get("on_done")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Verify done_callback was passed
                assert captured_done_callback is not None

                # Test with history.history attribute
                mock_history = MagicMock()
                mock_history.history = [1, 2, 3]  # 3 steps

                await captured_done_callback(mock_history)

                # Verify completion step was added
                steps = task_store[task_id]["progress"]["steps"]
                assert any(step.get("status") == "completed" for step in steps)

    @pytest.mark.asyncio
    async def test_done_callback_with_list_history(self, mock_llm, cleanup_tasks):
        """Test done_callback with list/tuple history.

        Covers lines 662-665: isinstance check for list/tuple
        """
        task_id = "test-done-list"
        instruction = "Test"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()
        mock_result = MagicMock()
        mock_result.final_result = MagicMock(extracted_content="Result")

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                captured_done_callback = None

                def agent_init(*args, **kwargs):
                    nonlocal captured_done_callback
                    captured_done_callback = kwargs.get("on_done")
                    mock_instance = AsyncMock()
                    mock_instance.run.return_value = mock_result
                    return mock_instance

                mock_agent_cls.side_effect = agent_init

                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Test with list
                await captured_done_callback([1, 2])

                # Should complete successfully
                steps = task_store[task_id]["progress"]["steps"]
                assert any(step.get("status") == "completed" for step in steps)

    @pytest.mark.asyncio
    async def test_browser_task_with_error(self, mock_llm, cleanup_tasks):
        """Test browser task execution with error.

        Covers error handling path
        """
        task_id = "test-error"
        instruction = "Test error"

        mock_context = AsyncMock()
        mock_context.page = MagicMock()
        mock_browser = AsyncMock()

        with patch("server.server.create_browser_context_for_task") as mock_create:
            with patch("server.server.Agent") as mock_agent_cls:
                mock_create.return_value = (mock_context, mock_browser)

                # Make Agent.run() raise an error
                mock_agent_instance = AsyncMock()
                mock_agent_instance.run.side_effect = RuntimeError("Test error")
                mock_agent_cls.return_value = mock_agent_instance

                # Execute task
                await run_browser_task_async(
                    task_id=task_id,
                    instruction=instruction,
                    llm=mock_llm
                )

                # Verify task failed
                assert task_id in task_store
                assert task_store[task_id]["status"] == "failed"
                assert "error" in task_store[task_id]
                assert "Test error" in str(task_store[task_id]["error"])
