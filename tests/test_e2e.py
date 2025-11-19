"""End-to-end tests for mcp-browser-use-server with real browsers and API calls."""

import os

import pytest


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("RUN_E2E_TESTS") != "true",
    reason="E2E tests require RUN_E2E_TESTS=true environment variable",
)
class TestRealBrowserAutomation:
    """End-to-end tests with actual browser automation.

    These tests require:
    - RUN_E2E_TESTS=true environment variable
    - Valid OPENAI_API_KEY
    - Playwright browsers installed
    """

    @pytest.mark.asyncio
    async def test_real_browser_navigation(self):
        """Test real browser navigation to a webpage."""
        from server.server import run_browser_task_async, task_store
        from tests._helpers import (
            build_e2e_llm_and_config,
            wait_for_task_completion,
        )

        llm, config = build_e2e_llm_and_config()
        if llm is None:
            pytest.skip("No valid OpenAI API key configured for E2E tests")

        task_id = "e2e-navigation-test"
        instruction = (
            "Navigate to https://example.com and verify the page loaded"
        )

        # Run actual task and wait for completion
        await run_browser_task_async(
            task_id=task_id, instruction=instruction, llm=llm, config=config
        )

        task = await wait_for_task_completion(
            task_id, task_store, timeout_sec=120
        )
        if task is None:
            pytest.fail("E2E task did not complete or was not found")
        if task.get("status") not in ["completed", "failed"]:
            pytest.fail("E2E task has unexpected status")

        # Clean up
        if task_id in task_store:
            del task_store[task_id]

    @pytest.mark.asyncio
    async def test_real_browser_content_extraction(self):
        """Test extracting content from a real webpage."""
        from server.server import run_browser_task_async, task_store
        from tests._helpers import (
            build_e2e_llm_and_config,
            wait_for_task_completion,
        )

        llm, config = build_e2e_llm_and_config()
        if llm is None:
            pytest.skip("No valid OpenAI API key configured for E2E tests")

        task_id = "e2e-extraction-test"
        instruction = (
            "Go to https://example.com and extract the main heading text"
        )

        await run_browser_task_async(
            task_id=task_id, instruction=instruction, llm=llm, config=config
        )

        task = await wait_for_task_completion(
            task_id, task_store, timeout_sec=120
        )
        if task is None:
            pytest.fail("E2E extraction task did not complete or was not found")
        if task.get("status") not in ["completed", "failed"]:
            pytest.fail("E2E extraction task has unexpected status")

        # If completed successfully, check for result
        if task.get("status") == "completed":
            result = task.get("result")
            if result is None:
                pytest.fail(
                    "E2E extraction task completed but returned no result"
                )

        # Clean up
        if task_id in task_store:
            del task_store[task_id]
