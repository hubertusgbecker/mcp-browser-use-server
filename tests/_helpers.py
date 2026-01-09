"""Test helpers shared across test files to reduce duplicated code.

Provide small utilities used by multiple tests so jscpd doesn't flag
copied code in multiple test files.
"""

import asyncio

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route


def create_starlette_app_with_sse() -> Starlette:
    """Create a minimal Starlette app with a health endpoint and SSE mount.

    Returns:
        Starlette: configured Starlette app ready for testing.
    """

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as _streams:
            # no-op: used by tests to ensure the mount and path exist
            # Reference _streams to satisfy linters (it's intentionally unused)
            _ = _streams
            await asyncio.sleep(0)

    async def _health(request):
        return PlainTextResponse("ok")

    app = Starlette(
        debug=True,
        routes=[
            Route("/health", endpoint=_health),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    return app


def build_e2e_llm_and_config():
    """Return (llm, config) for E2E tests if API key present, else None.

    This centralises the logic to obtain the OpenAI API key and construct
    a ChatOpenAI instance using SecretStr to satisfy type checks.
    """

    import os

    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from server.server import init_configuration

    api_key = os.getenv("OPENAI_API_KEY", "").strip("'\"")
    if not api_key or api_key == "your-api-key-here":
        return None, None

    config = init_configuration()
    model_name = os.getenv("LLM_MODEL", "gpt-5-mini")

    llm = ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        temperature=0,  # type: ignore[call-arg]
    )
    return llm, config


async def wait_for_task_completion(
    task_id: str, task_store: dict, timeout_sec: int = 120
):
    """Wait until task appears in task_store with completed/failed status.

    Returns the task dict if found, otherwise None.
    """

    import asyncio

    waited = 0
    interval = 2
    while waited < timeout_sec:
        await asyncio.sleep(interval)
        waited += interval
        if task_id in task_store:
            task = task_store[task_id]
            if task.get("status") in ["completed", "failed"]:
                return task

    return None


def run_cleanup_once_and_assert(
    task_store: dict, expect_removed: list, expect_present: list
) -> None:
    """Run a single pass of cleanup logic and assert expected removals/presence.

    Args:
        task_store: The shared task store dictionary.
        expect_removed: List of task_ids expected to be removed.
        expect_present: List of task_ids expected to remain present.
    """

    from datetime import datetime

    current_time = datetime.now()
    tasks_to_remove = []

    for tid, task_data in list(task_store.items()):
        if (
            task_data.get("status") in ["completed", "failed"]
            and "end_time" in task_data
        ):
            end_time = datetime.fromisoformat(task_data["end_time"])
            hours_elapsed = (current_time - end_time).total_seconds() / 3600

            if hours_elapsed > 1:  # Remove tasks older than 1 hour
                tasks_to_remove.append(tid)

    for tid in tasks_to_remove:
        task_store.pop(tid, None)

    for rid in expect_removed:
        if rid in task_store:
            import pytest

            pytest.fail(f"{rid} was not removed by cleanup logic")

    for pid in expect_present:
        if pid not in task_store:
            import pytest

            pytest.fail(f"{pid} was unexpectedly removed by cleanup logic")


def run_cli_with_fake_run(cli_mod, args: list, monkeypatch) -> None:
    """Run the CLI with a fake async `run_browser_task_async` and assert success.

    Args:
        cli_mod: The CLI module object (contains `cli`).
        args: Argument list to pass to the CLI runner.
        monkeypatch: pytest monkeypatch fixture.
    """

    async def fake_run(task_id, instruction, llm, config):
        return None

    with monkeypatch.context() as m:
        m.setattr("server.server.run_browser_task_async", fake_run)
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, args)
        if result.exit_code != 0:
            import pytest

            pytest.fail(f"CLI failed: {result.output}")


def assert_config_key_equals(call_args, key: str, expected):
    """Assert that a config key is present in mock call args and equals expected."""

    config = call_args[1]["config"]
    import pytest

    if key not in config:
        pytest.fail(f"{key} not present in config passed to run")
    if config[key] != expected:
        pytest.fail(
            f"{key} mismatch in passed config: expected {expected}, got {config[key]}"
        )


def assert_mock_run_config_equals(mock_run, key: str, expected):
    """Convenience wrapper to assert config key equals expected from a mock run."""

    call_args = mock_run.call_args
    assert_config_key_equals(call_args, key, expected)
