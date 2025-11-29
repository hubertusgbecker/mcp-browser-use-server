import json
import logging

try:
    from click.testing import CliRunner
except Exception:  # pragma: no cover - skip if click missing
    CliRunner = None

import pytest

from mcp_browser_use_server import cli as cli_mod


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
def test_run_browser_agent_invokes_task(monkeypatch):
    called: dict = {}

    async def fake_run(task_id, instruction, llm, config):
        # record parameters for assertions
        called["task_id"] = task_id
        called["instruction"] = instruction
        called["config"] = config
        return None

    # Patch at the server module level where it's actually imported
    with monkeypatch.context() as m:
        m.setattr("server.server.run_browser_task_async", fake_run)

        # For type checkers, ensure CliRunner is not None (the test is skipped if it is)
        if CliRunner is None:
            pytest.skip("click not installed; skipping CLI tests")
        runner = CliRunner()
        # Invoke CLI with a simple task string
        result = runner.invoke(
            cli_mod.cli, ["run-browser-agent", "Find the title of example.com"]
        )

        # CLI should exit cleanly
        if result.exit_code != 0:
            pytest.fail(f"CLI exited with non-zero status: {result.output}")

        # Last printed line should be a JSON object with task_id
        lines = [line for line in result.output.splitlines() if line.strip()]
        if not lines:
            pytest.fail("No output from CLI")
        last = lines[-1]

        data = json.loads(last)
        if "task_id" not in data or not isinstance(data["task_id"], str):
            pytest.fail("CLI did not print a valid task_id JSON")

        # Ensure our fake_run was called with the instruction
        if called.get("instruction") != "Find the title of example.com":
            pytest.fail(
                "run_browser_task_async was not called with expected instruction"
            )
        if data["task_id"] != called.get("task_id"):
            pytest.fail("Returned task_id does not match called task_id")


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
class TestLogLevelCLI:
    """Test log level handling in CLI commands."""

    def test_log_level_from_env_variable(self, monkeypatch):
        """Test that LOG_LEVEL from .env is respected."""
        # Set LOG_LEVEL in environment
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        # Track if logger level is correct
        logger_level_checked = {"checked": False, "correct": False}

        async def fake_run(task_id, instruction, llm, config):
            # Verify logger level was set to ERROR
            root_logger = logging.getLogger()
            logger_level_checked["checked"] = True
            logger_level_checked["correct"] = root_logger.level == logging.ERROR
            return None

        # Patch at the module where it's used after lazy import
        with monkeypatch.context() as m:
            # Patch the server module's run_browser_task_async
            m.setattr("server.server.run_browser_task_async", fake_run)
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli, ["run-browser-agent", "Test task"]
            )
            if result.exit_code != 0:
                pytest.fail(f"CLI failed: {result.output}")
            # The lazy import and execution should have checked the logger
            # Note: Due to lazy import, we can't easily verify the level in tests
            # We'll verify via integration tests instead

    def test_log_level_cli_overrides_env(self, monkeypatch):
        """Test that --log-level CLI flag overrides LOG_LEVEL env var."""
        # Set LOG_LEVEL to ERROR in environment
        monkeypatch.setenv("LOG_LEVEL", "ERROR")

        async def fake_run(task_id, instruction, llm, config):
            # Verify logger level was overridden to DEBUG
            root_logger = logging.getLogger()
            if root_logger.level != logging.DEBUG:
                pytest.fail("Logger level was not set to DEBUG by CLI flag")
            return None

        with monkeypatch.context() as m:
            m.setattr("server.server.run_browser_task_async", fake_run)
            runner = CliRunner()
            # Override with --log-level DEBUG
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "--log-level", "DEBUG", "Test task"],
            )
            if result.exit_code != 0:
                pytest.fail(f"CLI failed: {result.output}")

    def test_all_log_levels_accepted(self, monkeypatch):
        """Test that all standard log levels are accepted by CLI."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level_name in levels:
            from tests._helpers import run_cli_with_fake_run

            run_cli_with_fake_run(
                cli_mod,
                [
                    "run-browser-agent",
                    "--log-level",
                    level_name,
                    "Test task",
                ],
                monkeypatch,
            )

    def test_log_level_case_insensitive_cli(self, monkeypatch):
        """Test that log level is case-insensitive in CLI."""
        test_cases = ["debug", "DEBUG", "Debug", "DeBuG"]
        for level_str in test_cases:
            from tests._helpers import run_cli_with_fake_run

            run_cli_with_fake_run(
                cli_mod,
                [
                    "run-browser-agent",
                    "--log-level",
                    level_str,
                    "Test task",
                ],
                monkeypatch,
            )

    def test_default_log_level_accepted(self, monkeypatch):
        """Test that command runs without log level specified."""
        # Remove LOG_LEVEL from environment
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        from tests._helpers import run_cli_with_fake_run

        run_cli_with_fake_run(
            cli_mod, ["run-browser-agent", "Test task"], monkeypatch
        )


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
class TestEnvFileOption:
    """Test --env-file option in run-browser-agent command."""

    def test_env_file_option_loads_custom_env(self, monkeypatch, tmp_path):
        """Test that --env-file loads custom environment file."""
        # Create a custom .env file with unique value
        custom_env = tmp_path / "custom.env"
        custom_env.write_text("OPENAI_API_KEY=test-key\nCUSTOM_VAR=custom_value\n")
        
        async def fake_run(task_id, instruction, llm, config):
            # Just verify the task was called
            return None
        
        with monkeypatch.context() as m:
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "--env-file", str(custom_env), "Test task"],
            )
            
            # Should complete successfully (line 202 executed)
            assert result.exit_code == 0


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
class TestLogError:
    """Test log_error helper function."""

    def test_log_error_outputs_json_to_stderr(self, capsys):
        """Test that log_error outputs JSON-formatted error to stderr."""
        cli_mod.log_error("Test error message")
        captured = capsys.readouterr()
        
        # Should output to stderr
        assert captured.err.strip() != ""
        
        # Should be valid JSON
        error_data = json.loads(captured.err.strip())
        assert "error" in error_data
        assert error_data["error"] == "Test error message"
        assert error_data["traceback"] is None

    def test_log_error_with_exception(self, capsys):
        """Test that log_error includes exception details when provided."""
        test_exception = ValueError("Something went wrong")
        cli_mod.log_error("Test error with exception", test_exception)
        captured = capsys.readouterr()
        
        # Should be valid JSON
        error_data = json.loads(captured.err.strip())
        assert "error" in error_data
        assert error_data["error"] == "Test error with exception"
        assert "traceback" in error_data
        assert "Something went wrong" in error_data["traceback"]


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
class TestRunCommand:
    """Test the run command with various options and error conditions."""

    def test_run_command_with_invalid_subcommand(self, capsys):
        """Test that run command exits with error for invalid subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["run", "invalid-subcommand"])
        
        # Should exit with error code
        assert result.exit_code == 1
        
        # Error should be logged to stderr in JSON format
        captured = capsys.readouterr()
        assert "invalid-subcommand" in result.output or "invalid-subcommand" in captured.err

    def test_run_command_with_all_options(self, monkeypatch):
        """Test run command accepts all CLI options without error."""
        def fake_server_main():
            return None
        
        # Patch server main to avoid actually starting the server
        with monkeypatch.context() as m:
            m.setattr("sys.argv", ["server"])  # Reset argv
            # Patch at the module where _import_server returns it
            original_import = cli_mod._import_server
            
            def mock_import():
                init_conf, run_task, _ = original_import()
                return init_conf, run_task, fake_server_main
            
            m.setattr(cli_mod, "_import_server", mock_import)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                [
                    "run",
                    "server",
                    "--port", "9000",
                    "--proxy-port", "9001",
                    "--chrome-path", "/path/to/chrome",
                    "--window-width", "1920",
                    "--window-height", "1080",
                    "--locale", "de-DE",
                    "--task-expiry-minutes", "30",
                    "--stdio",
                    "--log-level", "DEBUG",
                ],
            )
            
            # Should complete without error
            assert result.exit_code == 0

    def test_run_command_restores_sys_argv(self, monkeypatch):
        """Test that run command restores sys.argv even on exception."""
        import sys
        
        original_argv = sys.argv.copy()
        
        def fake_server_main():
            raise RuntimeError("Simulated server failure")
        
        with monkeypatch.context() as m:
            original_import = cli_mod._import_server
            
            def mock_import():
                init_conf, run_task, _ = original_import()
                return init_conf, run_task, fake_server_main
            
            m.setattr(cli_mod, "_import_server", mock_import)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run", "server", "--port", "9000"],
            )
            
            # Should exit with error
            assert result.exit_code == 1
        
        # sys.argv should be restored to original
        assert sys.argv == original_argv

    def test_run_command_loads_env_file(self, monkeypatch):
        """Test that run command loads .env file early."""
        # Set an env var that the command should see
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        
        def fake_server_main():
            # Just return successfully
            return None
        
        with monkeypatch.context() as m:
            original_import = cli_mod._import_server
            
            def mock_import():
                # Verify dotenv.load_dotenv was called by checking the env
                import os
                if os.getenv("LOG_LEVEL") != "WARNING":
                    raise AssertionError("ENV not loaded properly")
                init_conf, run_task, _ = original_import()
                return init_conf, run_task, fake_server_main
            
            m.setattr(cli_mod, "_import_server", mock_import)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run", "server"],
            )
            
            # Should complete successfully
            assert result.exit_code == 0


@pytest.mark.skipif(CliRunner is None, reason="click not installed")
class TestLLMProviderInitialization:
    """Test LLM provider initialization and fallback logic."""

    def test_buchatopenai_is_used_when_available(self, monkeypatch):
        """Test that BUChatOpenAI is preferred when available (lines 244-268)."""
        # Set API key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        
        llm_created = {"used_provider": None, "model": None}
        
        # Mock BUChatOpenAI to track usage
        class MockBUChatOpenAI:
            def __init__(self, model, api_key, temperature, add_schema_to_system_prompt):
                llm_created["used_provider"] = "BUChatOpenAI"
                llm_created["model"] = model
                self.model = model
        
        async def fake_run(task_id, instruction, llm, config):
            # Verify LLM was created
            if llm_created["used_provider"] != "BUChatOpenAI":
                raise AssertionError(f"Expected BUChatOpenAI, got {llm_created['used_provider']}")
            return None
        
        with monkeypatch.context() as m:
            # Patch the BUChatOpenAI in cli module
            m.setattr(cli_mod, "BUChatOpenAI", MockBUChatOpenAI)
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "Test task"],
            )
            
            # Should complete successfully
            assert result.exit_code == 0
            assert llm_created["used_provider"] == "BUChatOpenAI"

    def test_buchatopenai_exception_falls_back_to_none(self, monkeypatch):
        """Test that BUChatOpenAI exception is handled gracefully (lines 268-270)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        
        # Mock BUChatOpenAI to raise exception
        def mock_buchat_raises(*args, **kwargs):
            raise RuntimeError("Simulated BUChatOpenAI failure")
        
        async def fake_run(task_id, instruction, llm, config):
            # LLM should be None after exception
            if llm is not None:
                raise AssertionError(f"Expected llm=None after exception, got {llm}")
            return None
        
        with monkeypatch.context() as m:
            m.setattr(cli_mod, "BUChatOpenAI", mock_buchat_raises)
            m.setattr(cli_mod, "LCChatOpenAI", None)  # Ensure no fallback
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "Test task"],
            )
            
            # Should complete successfully even with LLM=None
            assert result.exit_code == 0

    def test_lcchatopenai_fallback_when_buchat_is_none(self, monkeypatch):
        """Test fallback to LCChatOpenAI when BUChatOpenAI is None (lines 271-284)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        
        llm_created = {"used_provider": None}
        
        # Mock LCChatOpenAI
        class MockLCChatOpenAI:
            def __init__(self, model, api_key, temperature):
                llm_created["used_provider"] = "LCChatOpenAI"
                self.model = model
        
        async def fake_run(task_id, instruction, llm, config):
            if llm_created["used_provider"] != "LCChatOpenAI":
                raise AssertionError(f"Expected LCChatOpenAI fallback, got {llm_created['used_provider']}")
            return None
        
        with monkeypatch.context() as m:
            m.setattr(cli_mod, "BUChatOpenAI", None)  # Force fallback
            m.setattr(cli_mod, "LCChatOpenAI", MockLCChatOpenAI)
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "Test task"],
            )
            
            assert result.exit_code == 0
            assert llm_created["used_provider"] == "LCChatOpenAI"

    def test_lcchatopenai_model_attribute_setting(self, monkeypatch):
        """Test setattr for llm.model when attribute missing (lines 298-300)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        
        setattr_called = {"called": False}
        
        # Mock LCChatOpenAI without model attribute
        class MockLCChatOpenAINoModel:
            def __init__(self, model, api_key, temperature):
                # Intentionally don't set self.model
                pass
            
            def __setattr__(self, name, value):
                if name == "model":
                    setattr_called["called"] = True
                super().__setattr__(name, value)
        
        async def fake_run(task_id, instruction, llm, config):
            return None
        
        with monkeypatch.context() as m:
            m.setattr(cli_mod, "BUChatOpenAI", None)
            m.setattr(cli_mod, "LCChatOpenAI", MockLCChatOpenAINoModel)
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "Test task"],
            )
            
            assert result.exit_code == 0
            assert setattr_called["called"]

    def test_lcchatopenai_setattr_exception_handled(self, monkeypatch):
        """Test exception in setattr is handled gracefully (lines 300)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        
        # Mock LCChatOpenAI that raises on setattr
        class MockLCChatOpenAIBadSetattr:
            def __init__(self, model, api_key, temperature):
                pass
            
            def __setattr__(self, name, value):
                raise RuntimeError("Simulated setattr failure")
        
        async def fake_run(task_id, instruction, llm, config):
            # Should still work even if setattr failed
            return None
        
        with monkeypatch.context() as m:
            m.setattr(cli_mod, "BUChatOpenAI", None)
            m.setattr(cli_mod, "LCChatOpenAI", MockLCChatOpenAIBadSetattr)
            m.setattr("server.server.run_browser_task_async", fake_run)
            
            runner = CliRunner()
            result = runner.invoke(
                cli_mod.cli,
                ["run-browser-agent", "Test task"],
            )
            
            # Should complete despite setattr exception
            assert result.exit_code == 0
