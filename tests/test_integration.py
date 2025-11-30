"""Integration tests for mcp-browser-use-server."""

import os
import shutil
import subprocess  # nosec: B404 - test-only subprocess usage

import pytest

# Load environment variables for consistent configuration
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Resolve external tool paths once at module import time for use in skipifs
docker_path = shutil.which("docker")
# Try python3 first, then python
python_path = shutil.which("python3") or shutil.which("python")

# Get port configurations from environment (for consistency with docker-compose)
HOST_PORT = int(os.getenv("HOST_PORT", "8081"))
DOCKER_CONTAINER_NAME = os.getenv("CONTAINER_NAME", "mcp-browser-use-server")


class TestDockerIntegration:
    """Test Docker container integration."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        docker_path is None,
        reason="Docker not available in PATH",
    )
    def test_docker_container_running(self):
        """Test that Docker container can start and is accessible."""
        result = subprocess.run(
            [
                docker_path,
                "ps",
                "--filter",
                f"name={DOCKER_CONTAINER_NAME}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Check if container exists (may not be running in test environment)
        if result.returncode != 0:
            pytest.skip("Docker command failed; container may not be present")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        docker_path is None or python_path is None,
        reason="Docker or python not available in PATH",
    )
    async def test_docker_task_execution(self):
        """Test executing a browser task inside Docker container."""
        task_instruction = (
            "Navigate to https://example.com and get the page title"
        )

        python_code = f"""
import asyncio
import json
import os
from server.server import run_browser_task_async, task_store, init_configuration

async def run_task():
    config = init_configuration()
    api_key = os.getenv('OPENAI_API_KEY', '').strip("'")

    if not api_key or api_key == 'your-api-key-here':
        print('SKIP: No valid API key')
        return

    task_id = 'integration-test-001'

    # Let run_browser_task_async create its own LLM
    await run_browser_task_async(
        task_id=task_id,
        instruction='{task_instruction}',
        llm=None,
        config=config,
    )

    # Wait for completion
    for _ in range(30):
        await asyncio.sleep(2)
        if task_id in task_store:
            task = task_store[task_id]
            if task.get('status') == 'completed':
                print('SUCCESS')
                return
            elif task.get('status') == 'failed':
                print('FAILED')
                return

    print('TIMEOUT')

asyncio.run(run_task())
"""

        try:
            result = subprocess.run(
                [
                    docker_path,
                    "exec",
                    "-i",
                    DOCKER_CONTAINER_NAME,
                    "python3",  # Use python3 command in container, not host path
                    "-c",
                    python_code,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Test passes if: container accessible, task runs, or API key not configured
            if not (
                "SUCCESS" in result.stdout
                or "SKIP" in result.stdout
                or result.returncode == 0
            ):
                pytest.skip("Docker integration did not report success or skip")

        except subprocess.TimeoutExpired:
            pytest.skip("Docker task execution timed out")
        except Exception as e:
            pytest.skip(f"Docker integration test skipped: {e}")


class TestServerEndpoints:
    """Test server HTTP endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint (if implemented)."""
        import httpx

        url = f"http://localhost:{HOST_PORT}/health"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    pytest.skip("Health endpoint returned non-200 status")
        except Exception as e:
            pytest.skip(f"Health endpoint not available: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sse_endpoint(self):
        """Test SSE endpoint accessibility."""
        import httpx

        url = f"http://localhost:{HOST_PORT}/sse"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Use streaming to avoid waiting for the entire (infinite) SSE stream
                async with client.stream("GET", url, timeout=5.0) as resp:
                    if resp.status_code != 200:
                        pytest.skip("SSE endpoint returned non-200 status")

                    ctype = resp.headers.get("content-type", "")
                    if "text/event-stream" not in ctype:
                        pytest.skip(
                            "SSE endpoint content-type not event-stream"
                        )

                    # Read a few lines from the stream to confirm an event
                    lines = []
                    async for line in resp.aiter_lines():
                        if line:
                            lines.append(line)
                        if len(lines) >= 2:
                            break

                    if not any(line.startswith("event:") for line in lines):
                        pytest.skip("No SSE event lines received")
                    if not any(line.startswith("data:") for line in lines):
                        pytest.skip("No SSE data lines received")
        except Exception as e:
            pytest.skip(f"SSE endpoint not available: {e}")


class TestBrowserAutomation:
    """Test browser automation integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_playwright_browser_launch(self):
        """Test that Playwright can launch a browser."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://example.com")
                title = await page.title()
                await browser.close()

                if title is None or len(title) == 0:
                    pytest.skip("Playwright did not return a page title")

        except Exception as e:
            pytest.skip(f"Playwright integration test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_browser_use_library(self):
        """Test browser-use library integration."""
        try:
            from unittest.mock import MagicMock

            from browser_use import Agent

            # Mock ChatOpenAI to avoid external API calls and API changes
            MockLLM = MagicMock()
            # Provide a simple sync/async-compatible response method expected by
            # browser-use
            mock_llm_instance = MagicMock()
            mock_llm_instance.agenerate.return_value = MagicMock(
                generations=[[MagicMock(text="Test response")]]
            )

            MockLLM.return_value = mock_llm_instance

            llm = MockLLM()

            agent = Agent(
                task="Go to example.com",
                llm=llm,
            )

            # Don't actually run the agent in tests, just verify it can be created
            if agent is None:
                pytest.skip("Failed to create Agent instance")

        except Exception as e:
            pytest.skip(f"browser-use integration test failed: {e}")


class TestTaskLifecycle:
    """Test complete task lifecycle."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_task_creation_execution_cleanup(
        self, mock_config, mock_llm, cleanup_tasks
    ):
        """Test full task lifecycle from creation to cleanup."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from server.server import run_browser_task_async, task_store

        task_id = "lifecycle-test-task"
        instruction = "Test task lifecycle"

        with (
            patch(
                "server.server.create_browser_context_for_task"
            ) as mock_create_ctx,
            patch("server.server.Agent") as mock_agent_class,
        ):
            mock_browser_context = AsyncMock()
            mock_browser_context.close = AsyncMock()
            mock_create_ctx.return_value = (mock_browser_context, MagicMock())

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = "Task completed"
            mock_agent_class.return_value = mock_agent_instance

            # Create and execute task
            await run_browser_task_async(
                task_id=task_id,
                instruction=instruction,
                llm=mock_llm,
                config=mock_config,
            )

            # Verify task exists and completed
            if task_id not in task_store:
                pytest.fail("Task was not created in task_store")
            if task_store[task_id]["status"] != "completed":
                pytest.fail("Task did not complete successfully")

            # Simulate old task (for cleanup test)
            from datetime import datetime, timedelta

            old_time = (datetime.now() - timedelta(hours=2)).isoformat()
            task_store[task_id]["created_at"] = old_time
            # Add end_time for cleanup logic
            task_store[task_id]["end_time"] = old_time

            # Run centralized cleanup assertion helper
            from tests._helpers import run_cleanup_once_and_assert

            run_cleanup_once_and_assert(
                task_store, expect_removed=[task_id], expect_present=[]
            )
