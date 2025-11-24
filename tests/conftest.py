"""Pytest configuration and fixtures for mcp-browser-use-server tests."""

# Standard library
import os
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock

# Third-party
import pytest

# Optional third-party with fallback: dotenv may not be installed in minimal test envs
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - fallback when dotenv isn't available in minimal test env

    def load_dotenv(*args, **kwargs):
        """Fallback no-op load_dotenv implementation when python-dotenv is not installed.

        Some CI or lightweight test environments might not install `python-dotenv`.
        Providing a no-op keeps tests portable and avoids an ImportError while
        preserving expected behavior for projects that don't rely on .env files.
        """
        return None


# Load environment variables
load_dotenv()

# Add server directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Integration server process handle (shared across tests)
_integration_server_proc = None


def _is_port_open(host: str, port: int) -> bool:
    """Check if TCP port is open on host."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def integration_server(request):
    """Start the local server when an integration test runs.

    This fixture is autouse so integration tests don't need to declare it.
    It starts the server once and leaves it running for the test session.
    If the server cannot be started the test is skipped.
    """
    global _integration_server_proc

    if request.node.get_closest_marker("integration"):
        if _integration_server_proc is None:
            # Start the server using the project's uv wrapper
            logfile = open("server.log", "a", encoding="utf-8")
            try:
                # Resolve absolute path to `uv` to avoid starting processes
                # with a partial executable path (Bandit B607).
                uv_path = shutil.which("uv")
                if not uv_path:
                    # If uv isn't available, skip integration tests early.
                    pytest.skip(
                        "`uv` not available in PATH; skipping integration tests"
                    )

                # Intentionally start the local test server for integration tests.
                # Use absolute executable path to satisfy Bandit checks.
                _integration_server_proc = subprocess.Popen(  # nosec: B603
                    [uv_path, "run", "server", "--port", "8081"],
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                )
            finally:
                try:
                    logfile.close()
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Failed to close logfile: %s", exc
                    )

            # Wait for server to accept connections
            start = time.time()
            timeout = 15
            while time.time() - start < timeout:
                if _is_port_open("127.0.0.1", 8081):
                    break
                time.sleep(0.2)
            else:
                pytest.skip("Failed to start integration server on port 8081")

    yield


def pytest_sessionfinish(session, exitstatus):
    """Stop the integration server started for tests at session end."""
    global _integration_server_proc
    if _integration_server_proc:
        try:
            _integration_server_proc.terminate()
            _integration_server_proc.wait(timeout=5)
        except Exception as exc:
            try:
                _integration_server_proc.kill()
            except Exception as exc_kill:
                # Log at debug level for test teardown; keep behavior tolerant
                import logging

                logging.getLogger(__name__).debug(
                    "Failed to kill integration server: %s; kill error: %s",
                    exc,
                    exc_kill,
                )
        _integration_server_proc = None


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "OPENAI_API_KEY": "placeholder",
        "CHROME_PATH": "/usr/bin/chromium",
        "PATIENT": "false",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "OPENAI_API_KEY": "placeholder",
        "CHROME_PATH": "/usr/bin/chromium",
        "PATIENT": False,
        "LOG_LEVEL": "DEBUG",
        "DEFAULT_WINDOW_WIDTH": 1280,
        "DEFAULT_WINDOW_HEIGHT": 1100,
        "PATIENT_MODE": False,
    }


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM."""
    mock = AsyncMock()
    mock.agenerate.return_value = MagicMock(
        generations=[[MagicMock(text="Test response")]]
    )
    return mock


@pytest.fixture
async def mock_browser_context():
    """Mock BrowserContext instance."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    mock.page = AsyncMock()
    return mock


@pytest.fixture
async def cleanup_tasks():
    """Cleanup task store after tests."""
    yield
    # Import task_store from server module
    try:
        from server.server import task_store

        task_store.clear()
    except ImportError:
        pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


# Pytest collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
