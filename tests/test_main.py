"""Tests for server/__main__.py module."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMainModule:
    """Test __main__ module entry point."""

    def test_main_module_structure(self):
        """Test that __main__ module has correct structure."""
        # Read the __main__ module content without executing it
        main_path = Path(__file__).parent.parent / "server" / "__main__.py"
        content = main_path.read_text()

        # Verify it has the required imports and structure
        assert "import sys" in content
        assert "from server import main" in content
        assert "sys.exit(main())" in content

    def test_main_module_can_be_executed(self):
        """Test that __main__ module can be executed as python -m server."""
        # Execute the module as subprocess to avoid sys.exit in test process
        result = subprocess.run(
            [sys.executable, "-m", "server", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Should show help message and exit cleanly
        assert "Usage" in result.stdout or "Usage" in result.stderr
        # Exit code 0 for --help is expected
        assert result.returncode == 0

    def test_main_module_file_exists(self):
        """Test that __main__.py exists in server package."""
        main_path = Path(__file__).parent.parent / "server" / "__main__.py"
        assert main_path.exists()
        assert main_path.is_file()

    def test_main_module_is_importable(self):
        """Test that server package can be imported without execution side effects."""
        # We import the server package (not __main__) to verify structure
        import server

        assert hasattr(server, "main")
        assert callable(server.main)
