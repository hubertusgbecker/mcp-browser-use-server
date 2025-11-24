"""Command line interface for mcp-browser-use-server.

This module provides a command-line interface for starting the MCP Browser Use Server
and running a simple CLI command to execute a browser agent task for testing.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any, Optional, cast

import click
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from pythonjsonlogger import jsonlogger


def _import_server():
    """Lazily import server functions to avoid early logging configuration."""
    from mcp_browser_use_server.server import (
        init_configuration,
        run_browser_task_async,
    )
    from mcp_browser_use_server.server import main as server_main

    return init_configuration, run_browser_task_async, server_main


# LLM providers: prefer browser_use's own ChatOpenAI for compatibility
BUChatOpenAI: Any = None
LCChatOpenAI: Any = None
try:
    from browser_use.llm.openai.chat import ChatOpenAI as _BUChatOpenAI

    BUChatOpenAI = _BUChatOpenAI
except Exception:
    BUChatOpenAI = None

try:
    from langchain_openai import ChatOpenAI as _LCChatOpenAI

    LCChatOpenAI = _LCChatOpenAI
except Exception:
    LCChatOpenAI = None

# Configure logging for CLI
logger = logging.getLogger()
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler(sys.stderr)
formatter = jsonlogger.JsonFormatter(
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"name":"%(name)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_error(message: str, error: Optional[Exception] = None):
    """Log error in JSON format to stderr"""
    error_data = {"error": message, "traceback": str(error) if error else None}
    print(json.dumps(error_data), file=sys.stderr)


@click.group()
def cli():
    """MCP Browser Use Server command line interface."""


@cli.command()
@click.argument("subcommand")
@click.option("--port", default=8081, help="Port to listen on for SSE")
@click.option(
    "--proxy-port",
    default=None,
    type=int,
    help="Port for the proxy to listen on (when using stdio mode)",
)
@click.option("--chrome-path", default=None, help="Path to Chrome executable")
@click.option("--window-width", default=1280, help="Browser window width")
@click.option("--window-height", default=1100, help="Browser window height")
@click.option("--locale", default="en-US", help="Browser locale")
@click.option(
    "--task-expiry-minutes",
    default=60,
    help="Minutes after which tasks are considered expired",
)
@click.option(
    "--stdio",
    is_flag=True,
    default=False,
    help="Enable stdio mode with mcp-proxy",
)
@click.option(
    "--log-level",
    default=None,
    help="Logging level for server (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
def run(
    subcommand,
    port,
    proxy_port,
    chrome_path,
    window_width,
    window_height,
    locale,
    task_expiry_minutes,
    stdio,
    log_level,
):
    """Run the browser-use MCP server.

    SUBCOMMAND: should be 'server'
    """
    if subcommand != "server":
        log_error(
            f"Unknown subcommand: {subcommand}. Only 'server' is supported."
        )
        sys.exit(1)

    try:
        # Load .env early to respect LOG_LEVEL setting
        load_dotenv(override=False)

        # We need to construct the command line arguments to pass to the
        # server's Click command
        old_argv = sys.argv.copy()

        # Build a new argument list for the server command
        new_argv = [
            "server",  # Program name
            "--port",
            str(port),
        ]

        if chrome_path:
            new_argv.extend(["--chrome-path", chrome_path])

        if proxy_port is not None:
            new_argv.extend(["--proxy-port", str(proxy_port)])

        new_argv.extend(["--window-width", str(window_width)])
        new_argv.extend(["--window-height", str(window_height)])
        new_argv.extend(["--locale", locale])
        new_argv.extend(["--task-expiry-minutes", str(task_expiry_minutes)])

        if stdio:
            new_argv.append("--stdio")

        # Determine effective log level: CLI flag > LOG_LEVEL env var > INFO default
        effective_log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        new_argv.extend(["--log-level", effective_log_level])

        # Replace sys.argv temporarily
        sys.argv = new_argv

        # Run the server's command directly
        try:
            # Import server main lazily here so CLI logging config takes effect
            _, _, server_main = _import_server()
            return server_main()
        finally:
            # Restore original sys.argv
            sys.argv = old_argv

    except Exception as e:
        log_error("Error starting server", e)
        sys.exit(1)


@cli.command("run-browser-agent")
@click.option(
    "--env-file",
    "env_file",
    "-e",
    default=None,
    help="Path to a .env file to load configurations from.",
)
@click.option(
    "--log-level",
    "-l",
    default=None,
    help="Override the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.argument("task", required=True)
def run_browser_agent(
    env_file: Optional[str], log_level: Optional[str], task: str
):
    """Runs a browser agent task from the CLI.

    TASK: The instruction for the browser agent (quoted string).
    """
    try:
        # Load .env early to get LOG_LEVEL and other settings
        load_dotenv(override=False)  # Load default .env first

        # Load custom env file if provided (overrides default .env)
        if env_file:
            load_dotenv(dotenv_path=env_file, override=True)

        # Determine effective log level: CLI flag > LOG_LEVEL env var > INFO default
        effective_log_level = log_level or os.getenv("LOG_LEVEL") or "INFO"
        lvl = getattr(logging, effective_log_level.upper(), logging.INFO)
        logger.setLevel(lvl)

        # Also apply to commonly noisy loggers so CLI flag takes effect
        noisy = [
            "browser_use",
            "playwright",
            "mcp",
            "uvicorn",
            "Agent",
            "tools",
            "BrowserSession",
        ]
        for name in noisy:
            try:
                logging.getLogger(name).setLevel(lvl)
            except Exception:
                # Best-effort; do not fail CLI if a logger name is invalid
                pass

        # Create a unique task id
        task_id = str(uuid.uuid4())

        # Build config from environment
        init_configuration, run_browser_task_async, _ = _import_server()
        # Re-apply logging level in case server import changed handlers/levels
        logger.setLevel(lvl)
        for name in [
            "browser_use",
            "playwright",
            "mcp",
            "uvicorn",
            "Agent",
            "tools",
            "BrowserSession",
        ]:
            try:
                logging.getLogger(name).setLevel(lvl)
            except Exception:
                pass

        config = init_configuration()

        # Build an LLM from OPENAI_API_KEY if available
        llm = None
        api_key = os.getenv("OPENAI_API_KEY")
        # Choose model from env or default
        model_name = os.getenv("LLM_MODEL", "gpt-5-mini")
        if api_key:
            # Prefer browser_use's ChatOpenAI which accepts the message types
            # browser_use uses
            if BUChatOpenAI is not None:
                try:
                    # Use add_schema_to_system_prompt so browser-use embeds structured
                    # output schemas into the system prompt instead of using the
                    # response_format parameter which some models don't support.
                    llm = BUChatOpenAI(
                        model=model_name,
                        api_key=str(api_key),
                        temperature=0,
                        add_schema_to_system_prompt=True,
                    )
                except Exception:
                    llm = None
            elif LCChatOpenAI is not None:
                try:
                    llm = LCChatOpenAI(
                        model=model_name,
                        api_key=str(api_key),
                        temperature=0,
                    )
                    # Ensure `.model` exists for downstream compatibility
                    if not hasattr(llm, "model"):
                        try:
                            setattr(llm, "model", model_name)
                        except Exception as exc:
                            logger.debug("Failed to set llm.model: %s", exc)
                except Exception:
                    llm = None

        # Run the browser task synchronously via asyncio
        # Cast `llm` to BaseLanguageModel when calling the typed API
        llm_for_call = cast(Optional[BaseLanguageModel], llm)
        asyncio.run(
            run_browser_task_async(
                task_id, instruction=task, llm=llm_for_call, config=config
            )
        )

        # Print task id for tracking
        print(json.dumps({"task_id": task_id}))

    except Exception as e:
        log_error("CLI run_browser_agent command failed", e)
        sys.exit(1)


if __name__ == "__main__":
    cli()
