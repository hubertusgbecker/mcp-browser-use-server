"""
Browser Use MCP Server

This module implements an MCP (Model-Control-Protocol) server for browser
automation using the browser_use library. It provides functionality to
interact with a browser instance via an async task queue, allowing for
long-running browser tasks to be executed asynchronously while providing
status updates and results.

The server supports Server-Sent Events (SSE) for web-based interfaces.
"""

# Standard library imports
import asyncio
import html
import json
import logging
import os
import re
import shutil
import sys

# Set up SSE transport
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

# Third-party imports
import click
import mcp.types as types
import requests
import uvicorn

# Browser-use library imports
from browser_use import Agent
from browser_use.browser import BrowserProfile
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel

if TYPE_CHECKING:
    # Provide minimal type stubs for static checking to avoid importing
    # the real `langchain_core.schema` during type checking in CI where
    # stubs may not be installed. These match the minimal API used below.
    class BaseMessage:  # type: ignore
        def __init__(
            self, content: str
        ) -> None:  # pragma: no cover - typing only
            self.content: str = content

    class SystemMessage(BaseMessage):  # type: ignore
        pass

    class HumanMessage(BaseMessage):  # type: ignore
        pass

    class AIMessage(BaseMessage):  # type: ignore
        pass
else:
    try:
        # Import message types for LangChain v0-style messaging at runtime.
        from langchain_core.schema import (
            AIMessage,  # type: ignore[import]
            BaseMessage,  # type: ignore[import]
            HumanMessage,  # type: ignore[import]
            SystemMessage,  # type: ignore[import]
        )
    except Exception:
        # Fallback names if package layout differs; define minimal stand-ins
        class BaseMessage:  # type: ignore
            def __init__(self, content: str):
                self.content = content

        class SystemMessage(BaseMessage):
            pass

        class HumanMessage(BaseMessage):
            pass

        class AIMessage(BaseMessage):
            pass


# LLM provider
from types import SimpleNamespace

from langchain_openai import ChatOpenAI

# Keep a minimal provider attribute to maintain compatibility with code that
# inspects `llm.provider` during initialization. Avoid adding method shims
# globally — we use an adapter for async compatibility instead.
if not hasattr(ChatOpenAI, "provider"):
    setattr(ChatOpenAI, "provider", "openai")


from typing import cast

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from pythonjsonlogger import jsonlogger
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route


class ChatOpenAIAdapter:
    """Typed adapter for ChatOpenAI implementing the minimal async methods
    expected by browser-use Agent and other consumers.

    This class wraps an existing `ChatOpenAI` instance and exposes async
    `agenerate`/`ainvoke` methods, and preserves other attributes via
    delegation. It subclasses `BaseLanguageModel` so it is compatible with
    type checks that expect language model instances.
    """

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    @property
    def provider(self) -> str:
        return getattr(self._llm, "provider", "openai")

    def __getattr__(self, name: str):
        return getattr(self._llm, name)

    @property
    def model(self) -> str:
        """Expose a `.model` attribute for compatibility with
        browser-use expectations.
        """
        return getattr(
            self._llm, "model", getattr(self._llm, "model_name", "openai")
        )

    async def ainvoke(self, *args, **kwargs) -> Any:
        def _normalize(obj: Any) -> Any:
            """Normalize message-like objects to primitives or dicts.

            This attempts to convert common message objects (for example
            browser-use SystemMessage or other BaseMessage wrappers) into
            simple strings or {'role', 'content'} dicts which OpenAI and
            langchain expect. It is intentionally defensive to handle
            multiple shapes.
            """

            # Primitive types are returned as-is
            if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
                return obj

            # Lists/tuples: normalize elements
            if isinstance(obj, (list, tuple)):
                return [_normalize(i) for i in obj]

            # Dicts: normalize values
            if isinstance(obj, dict):
                return {k: _normalize(v) for k, v in obj.items()}

            # Common message shapes
            if hasattr(obj, "role") and hasattr(obj, "content"):
                return {
                    "role": getattr(obj, "role"),
                    "content": _normalize(getattr(obj, "content")),
                }

            if hasattr(obj, "content"):
                return _normalize(getattr(obj, "content"))

            if hasattr(obj, "text"):
                return getattr(obj, "text")

            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                try:
                    return _normalize(obj.to_dict())
                except Exception:
                    pass

            # Fallback: string representation
            try:
                return str(obj)
            except Exception:
                return obj

        def _normalize_args_kwargs(
            args_in: tuple[Any, ...], kwargs_in: dict[str, Any]
        ) -> tuple[list[Any], dict[str, Any]]:
            new_args: list[Any] = []
            for a in args_in:
                new_args.append(_normalize(a))

            new_kwargs: dict[str, Any] = {}
            for k, v in kwargs_in.items():
                new_kwargs[k] = _normalize(v)

            return new_args, new_kwargs

        def _convert_to_base_messages(obj: Any) -> Any:
            """Convert normalized message dicts/lists into BaseMessage instances

            If obj is a list of {'role','content'} dicts, convert to the
            corresponding LangChain message objects. Otherwise return obj.
            """

            # Handle list of messages
            if (
                isinstance(obj, list)
                and obj
                and all(
                    isinstance(i, dict) and "role" in i and "content" in i
                    for i in obj
                )
            ):
                res: list[BaseMessage] = []
                for item in obj:
                    role = (item.get("role") or "user").lower()
                    content = item.get("content")
                    # Ensure content is a string for message constructors
                    content_str = str(content) if content is not None else ""
                    if role == "system":
                        res.append(SystemMessage(content_str))
                    elif role in ("assistant", "ai"):
                        res.append(AIMessage(content_str))
                    else:
                        res.append(HumanMessage(content_str))
                return res

            # Single dict representing a message
            if isinstance(obj, dict) and "role" in obj and "content" in obj:
                role = (obj.get("role") or "user").lower()
                content = obj.get("content")
                content_str = str(content) if content is not None else ""
                if role == "system":
                    return SystemMessage(content_str)
                elif role in ("assistant", "ai"):
                    return AIMessage(content_str)
                else:
                    return HumanMessage(content_str)

            return obj

        if hasattr(self._llm, "agenerate") and callable(
            getattr(self._llm, "agenerate")
        ):
            maybe = getattr(self._llm, "agenerate")
            n_args, n_kwargs = _normalize_args_kwargs(args, kwargs)
            # Convert message-shaped structures into BaseMessage instances
            try:
                n_args = [_convert_to_base_messages(a) for a in n_args]
                for k, v in list(n_kwargs.items()):
                    n_kwargs[k] = _convert_to_base_messages(v)
            except Exception:
                # Conversion best-effort; continue if conversion fails
                pass

            # Additional best-effort conversions: if a top-level arg is a dict,
            # convert {'messages': [...]} into a list of BaseMessage, or
            # otherwise stringify dicts to avoid passing raw dicts to LLM.
            def _best_effort(obj: Any) -> Any:
                if isinstance(obj, dict):
                    if "messages" in obj and isinstance(obj["messages"], list):
                        return _convert_to_base_messages(obj["messages"])
                    if "role" in obj and "content" in obj:
                        return _convert_to_base_messages(obj)
                    try:
                        return json.dumps(obj)
                    except Exception:
                        return str(obj)
                if isinstance(obj, list):
                    return [_best_effort(i) for i in obj]
                return obj

            try:
                n_args = [_best_effort(a) for a in n_args]
                for k, v in list(n_kwargs.items()):
                    n_kwargs[k] = _best_effort(v)
            except Exception:
                pass
            # Debug: log normalized arg types to help diagnose invalid types
            try:

                def _summarize(x: Any) -> Any:
                    if isinstance(x, list):
                        return [type(i).__name__ for i in x[:3]] + (
                            ["..."] if len(x) > 3 else []
                        )
                    return type(x).__name__

                logger.debug(
                    "ChatOpenAIAdapter normalized args types: %s",
                    [_summarize(a) for a in n_args],
                )
                logger.debug(
                    "ChatOpenAIAdapter normalized kwargs types: %s",
                    {k: _summarize(v) for k, v in n_kwargs.items()},
                )
            except Exception:
                pass
            if asyncio.iscoroutinefunction(maybe):
                try:
                    logger.debug(
                        "Calling underlying agenerate with args=%s kwargs=%s",
                        repr(n_args),
                        repr(n_kwargs),
                    )
                    return await maybe(*n_args, **n_kwargs)
                except Exception as e:
                    try:
                        logger.error(
                            (
                                "agenerate call failed (%s). Arg types: %s; "
                                "Kwarg types: %s; Arg reprs: %s; "
                                "Kwarg reprs: %s"
                            ),
                            type(e).__name__,
                            [type(a).__name__ for a in n_args],
                            {k: type(v).__name__ for k, v in n_kwargs.items()},
                            [repr(a)[:1000] for a in n_args],
                            {k: repr(v)[:1000] for k, v in n_kwargs.items()},
                        )
                    except Exception:
                        logger.error(
                            "agenerate call failed and logging of args "
                            "also failed: %s",
                            str(e),
                        )
                    raise
            loop = asyncio.get_event_loop()
            try:
                logger.debug(
                    "Calling underlying agenerate (sync) with args=%s kwargs=%s",
                    repr(n_args),
                    repr(n_kwargs),
                )
                return await loop.run_in_executor(
                    None, lambda: maybe(*n_args, **n_kwargs)
                )
            except Exception as e:
                try:
                    logger.error(
                        (
                            "agenerate(sync) call failed (%s). Arg types: %s; "
                            "Kwarg types: %s; Arg reprs: %s; "
                            "Kwarg reprs: %s"
                        ),
                        type(e).__name__,
                        [type(a).__name__ for a in n_args],
                        {k: type(v).__name__ for k, v in n_kwargs.items()},
                        [repr(a)[:1000] for a in n_args],
                        {k: repr(v)[:1000] for k, v in n_kwargs.items()},
                    )
                except Exception:
                    logger.error(
                        "agenerate(sync) call failed and logging of args "
                        "also failed: %s",
                        str(e),
                    )
                raise

        if hasattr(self._llm, "generate") and callable(
            getattr(self._llm, "generate")
        ):
            loop = asyncio.get_event_loop()
            # Normalize and convert for the sync generate call as well
            n_args, n_kwargs = _normalize_args_kwargs(args, kwargs)
            try:
                n_args = [_convert_to_base_messages(a) for a in n_args]
                for k, v in list(n_kwargs.items()):
                    n_kwargs[k] = _convert_to_base_messages(v)
            except Exception:
                pass
            try:
                logger.debug(
                    "Calling underlying generate with args=%s kwargs=%s",
                    repr(n_args),
                    repr(n_kwargs),
                )
                return await loop.run_in_executor(
                    None,
                    lambda: getattr(self._llm, "generate")(*n_args, **n_kwargs),
                )
            except Exception as e:
                try:
                    logger.error(
                        (
                            "generate call failed (%s). Arg types: %s; "
                            "Kwarg types: %s; Arg reprs: %s; "
                            "Kwarg reprs: %s"
                        ),
                        type(e).__name__,
                        [type(a).__name__ for a in n_args],
                        {k: type(v).__name__ for k, v in n_kwargs.items()},
                        [repr(a)[:1000] for a in n_args],
                        {k: repr(v)[:1000] for k, v in n_kwargs.items()},
                    )
                except Exception:
                    logger.error(
                        "generate call failed and logging of args "
                        "also failed: %s",
                        str(e),
                    )
                raise

        raise NotImplementedError(
            "Underlying LLM does not support async invocation"
        )

    async def agenerate(self, *args, **kwargs) -> Any:
        return await self.ainvoke(*args, **kwargs)


def ensure_llm_adapter(llm: Optional[Any]) -> Optional[Any]:
    """Wrap known LLM implementations with adapter where needed.

    Currently wraps `ChatOpenAI` instances to provide compatibility.
    """
    if llm is None:
        return None

    if isinstance(llm, ChatOpenAI):
        # Use a minimal direct OpenAI-backed adapter that accepts dict messages
        # from browser-use and converts them to OpenAI REST format
        class MinimalOpenAIAdapter:
            def __init__(
                self, model_name: str, api_key: str, base_url: str = ""
            ):
                self.model = model_name
                self.model_name = model_name  # browser-use expects this
                self.api_key = api_key
                self.base_url = (
                    base_url if base_url else "https://api.openai.com/v1"
                )
                # browser-use expects these attributes
                self.provider = "openai"

            async def ainvoke(self, *args, **kwargs):
                """Accept messages and invoke OpenAI API - required by
                browser-use token_cost_service.
                """
                # Handle different call patterns from browser-use
                messages = args[0] if args else kwargs.get("messages", [])

                # Normalize messages to list of dicts format
                if not isinstance(messages, list):
                    messages = [{"role": "user", "content": str(messages)}]
                else:
                    # Convert LangChain message objects to dict format
                    normalized = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            normalized.append(msg)
                        elif hasattr(msg, "type") and hasattr(msg, "content"):
                            # LangChain message object
                            role = (
                                msg.type
                                if msg.type in ["system", "user", "assistant"]
                                else "user"
                            )
                            normalized.append(
                                {"role": role, "content": msg.content}
                            )
                        else:
                            # Fallback: convert to string
                            normalized.append(
                                {"role": "user", "content": str(msg)}
                            )
                    messages = normalized

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 500),
                    "temperature": kwargs.get("temperature", 0.7),
                }

                url = f"{self.base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                resp = requests.post(
                    url, headers=headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                body = resp.json()

                text = body["choices"][0]["message"]["content"]

                # Return a simple namespace with content attribute
                return SimpleNamespace(content=text)

            async def agenerate(self, messages, **kwargs):
                """Accept messages as list of dicts with role/content."""
                # messages should be a list of dicts like
                # [{"role":"system","content":"..."}]
                if not isinstance(messages, list):
                    # Fallback: wrap in a list
                    messages = [{"role": "user", "content": str(messages)}]

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 500),
                    "temperature": kwargs.get("temperature", 0.7),
                }

                url = f"{self.base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                resp = requests.post(
                    url, headers=headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                body = resp.json()

                text = body["choices"][0]["message"]["content"]

                # Return an object with `generations` like LangChain's agenerate
                gen = SimpleNamespace(text=text)
                return SimpleNamespace(generations=[[gen]])

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAI adapter")

        base_url = CONFIG.get("OPENAI_REVERSE_PROXY", "")
        model_name = getattr(
            llm, "model", os.environ.get("LLM_MODEL", "gpt-4o-mini")
        )

        return MinimalOpenAIAdapter(model_name, api_key, base_url)

    return llm


# Configure logging
logger = logging.getLogger()
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler(sys.stderr)
formatter = jsonlogger.JsonFormatter(
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"name":"%(name)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
# Do not set root logger level at import time; allow `main()` to control levels
logger.setLevel(logging.NOTSET)

# Ensure uvicorn also logs to stderr in JSON format

# Configure uvicorn logger handlers but do not override levels here
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []
uvicorn_logger.addHandler(handler)

# Ensure all other loggers use the same format
logging.getLogger("browser_use").addHandler(handler)
logging.getLogger("playwright").addHandler(handler)
logging.getLogger("mcp").addHandler(handler)

# Load environment variables
load_dotenv()


def parse_bool_env(env_var: str, default: bool = False) -> bool:
    """
    Parse a boolean environment variable.

    Args:
        env_var: The environment variable name
        default: Default value if not set

    Returns:
        Boolean value of the environment variable
    """
    value = os.environ.get(env_var)
    if value is None:
        return default

    # Consider various representations of boolean values
    return value.lower() in ("true", "yes", "1", "y", "on")


def init_configuration() -> Dict[str, Any]:
    """
    Initialize configuration from environment variables with defaults.

    Returns:
        Dictionary containing all configuration parameters
    """
    config = {
        # Browser window settings
        "DEFAULT_WINDOW_WIDTH": int(
            os.environ.get("BROWSER_WINDOW_WIDTH", 1280)
        ),
        "DEFAULT_WINDOW_HEIGHT": int(
            os.environ.get("BROWSER_WINDOW_HEIGHT", 1100)
        ),
        # Browser config settings
        "DEFAULT_LOCALE": os.environ.get("BROWSER_LOCALE", "en-US"),
        "DEFAULT_USER_AGENT": os.environ.get(
            "BROWSER_USER_AGENT",
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.102 Safari/537.36"
            ),
        ),
        # Task settings
        "DEFAULT_TASK_EXPIRY_MINUTES": int(
            os.environ.get("TASK_EXPIRY_MINUTES", 60)
        ),
        "DEFAULT_ESTIMATED_TASK_SECONDS": int(
            os.environ.get("ESTIMATED_TASK_SECONDS", 60)
        ),
        "CLEANUP_INTERVAL_SECONDS": int(
            os.environ.get("CLEANUP_INTERVAL_SECONDS", 3600)
        ),  # 1 hour
        "MAX_AGENT_STEPS": int(os.environ.get("MAX_AGENT_STEPS", 10)),
        # Browser arguments
        "BROWSER_ARGS": [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",  # Use random port to avoid conflicts
        ],
        # Run browsers in headless mode by default. Can be overridden via env var.
        "BROWSER_HEADLESS": parse_bool_env("BROWSER_HEADLESS", True),
        # Patient mode - if true, functions wait for task completion before returning
        "PATIENT_MODE": parse_bool_env("PATIENT", False),
        # OpenAI reverse proxy URL (empty string means use default OpenAI endpoint)
        "OPENAI_REVERSE_PROXY": os.environ.get("OPENAI_REVERSE_PROXY", ""),
    }

    return config


# Initialize configuration
CONFIG = init_configuration()

# Task storage for async operations
task_store: Dict[str, Dict[str, Any]] = {}


async def create_browser_context_for_task(
    chrome_path: Optional[str] = None,
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> Tuple[Any, Any]:
    """
    Placeholder for browser context creation (browser-use 0.9.5 handles this
    internally).

    This function maintains API compatibility but browser-use Agent now
    handles browser/context creation internally.

    Args:
        chrome_path: Path to Chrome executable (unused in 0.9.5+)
        window_width: Browser window width (unused in 0.9.5+)
        window_height: Browser window height (unused in 0.9.5+)
        locale: Browser locale (unused in 0.9.5+)

    Returns:
        A tuple of (None, None) for API compatibility

    Note:
        browser-use 0.9.5+ uses BrowserSession and BrowserProfile internally.
        Context creation is handled by the Agent class.
    """
    # In browser-use 0.9.5+, Agent handles browser session creation
    # Return None values for backward compatibility with existing code
    return None, None


async def run_browser_task_async(
    task_id: str,
    instruction: Optional[str] = None,
    llm: Optional[BaseLanguageModel] = None,
    config: Optional[Dict[str, Any]] = None,
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
    # legacy parameters from older handlers
    url: Optional[str] = None,
    action: Optional[str] = None,
) -> None:
    """Run a browser task asynchronously and store the result.

    This implementation initializes a task entry, runs a browser-use Agent,
    collects results, and updates `task_store`. It is intentionally
    defensive to tolerate mock objects used in tests.
    """
    browser = None
    context = None

    # Initialize a minimal task entry
    task_store[task_id] = {
        "status": "pending",
        "steps": [],
    }

    task_store[task_id]["message"] = (
        f"Browser task started. Please wait for "
        f"{CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} seconds. Then check "
        "the result using browser_get_result or the resource URI. "
        "Always wait exactly 5 seconds between status checks."
    )

    task_store[task_id]["estimated_time"] = (
        f"{CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} seconds"
    )

    task_store[task_id]["resource_uri"] = f"resource://browser_task/{task_id}"

    task_store[task_id]["sleep_command"] = "sleep 5"
    task_store[task_id]["instruction"] = (
        "Use the terminal command 'sleep 5' to wait 5 seconds between "
        "status checks. IMPORTANT: Always use exactly 5 seconds, no more "
        "and no less."
    )

    task_store[task_id]["result"] = None
    task_store[task_id]["error"] = None
    task_store[task_id]["progress"] = {
        "current_step": 0,
        "total_steps": 0,
        "steps": [],
    }

    # Helper to normalize step callbacks used by browser-use
    def step_callback(*args, **kwargs):
        agent_output = None
        step_number = None

        if len(args) == 1 and isinstance(args[0], dict):
            data = args[0]
            step_number = data.get("step") or data.get("step_number")
            agent_output = data
        elif len(args) >= 3:
            agent_output = args[1]
            step_number = args[2]
        elif "step_number" in kwargs or "step" in kwargs:
            step_number = kwargs.get("step_number") or kwargs.get("step")
            agent_output = kwargs.get("agent_output")
        else:
            agent_output = kwargs.get("agent_output") or kwargs.get("output")
            step_number = kwargs.get("step") or kwargs.get("step_number")

        if step_number is None:
            logger.warning(
                "Task %s: step_callback called without step number",
                task_id,
            )
            return

        task_store[task_id]["progress"]["current_step"] = step_number
        task_store[task_id]["progress"]["total_steps"] = max(
            task_store[task_id]["progress"]["total_steps"], step_number
        )

        step_info = {"step": step_number, "time": datetime.now().isoformat()}

        try:
            if agent_output and hasattr(agent_output, "current_state"):
                if hasattr(agent_output.current_state, "next_goal"):
                    step_info["goal"] = agent_output.current_state.next_goal
        except Exception:
            logger.debug(
                "Ignored error accessing agent_output attributes", exc_info=True
            )

        task_store[task_id]["progress"]["steps"].append(step_info)
        logger.info("Task %s: Step %s completed", task_id, step_number)

    async def done_callback(history: Any) -> None:
        try:
            if hasattr(history, "history"):
                n_steps = len(history.history)
            elif isinstance(history, (list, tuple)):
                n_steps = len(history)
            else:
                n_steps = 1

            logger.info("Task %s: Completed with %s steps", task_id, n_steps)

            current_step = task_store[task_id]["progress"]["current_step"] + 1
            task_store[task_id]["progress"]["steps"].append(
                {
                    "step": current_step,
                    "time": datetime.now().isoformat(),
                    "status": "completed",
                }
            )
        except Exception as e:
            logger.error("Error in done_callback for task %s: %s", task_id, e)

    try:
        task_store[task_id]["status"] = "running"
        task_store[task_id]["start_time"] = datetime.now().isoformat()

        # Create browser/context if the library needs it
        chrome_path = None
        if config and isinstance(config, dict):
            chrome_path = config.get("CHROME_PATH")
            window_width = config.get("WINDOW_WIDTH", window_width)
            window_height = config.get("WINDOW_HEIGHT", window_height)

        if not chrome_path:
            chrome_path = os.environ.get("CHROME_PATH")

        created = await create_browser_context_for_task(
            chrome_path=chrome_path,
            window_width=window_width,
            window_height=window_height,
            locale=locale,
        )

        # Accept either (browser, context) or (context, browser)
        if isinstance(created, tuple) and len(created) == 2:
            a, b = created
            if hasattr(a, "page") and not hasattr(a, "new_context"):
                context, browser = a, b
            elif hasattr(b, "page") and not hasattr(b, "new_context"):
                context, browser = b, a
            else:
                browser, context = created
        else:
            browser = None
            context = created

        # Determine task text
        if instruction:
            task_text = instruction
        elif url or action:
            task_text = f"First, navigate to {url}. Then, {action}"
        else:
            cfg_url = None
            cfg_action = None
            if config and isinstance(config, dict):
                cfg_url = config.get("url") or config.get("URL")
                cfg_action = config.get("action") or config.get("ACTION")

            if cfg_url or cfg_action:
                task_text = f"First, navigate to {cfg_url}. Then, {cfg_action}"
            else:
                task_text = "Perform the requested browser task"

        # If no LLM was provided, instantiate a default ChatOpenAI using
        # environment variables so we can wrap it with our adapter.
        if llm is None:
            try:
                model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    # ChatOpenAI expects `model` and `openai_api_key` may not be
                    # accepted in all langchain versions. Use modern `model=` and
                    # rely on environment for API key when appropriate.
                    llm = ChatOpenAI(model=model_name)
                    os.environ["OPENAI_API_KEY"] = api_key
                else:
                    llm = ChatOpenAI(model=model_name)
            except Exception as e:
                logger.warning(
                    "Failed to create default ChatOpenAI: %s", str(e)
                )

        llm_for_agent = ensure_llm_adapter(llm)

        # Build a browser profile for local runs when appropriate
        browser_profile_for_agent = None
        try:
            use_cloud_key = os.getenv("BROWSER_USE_API_KEY")
            if not use_cloud_key:
                bp_kwargs: dict[str, Any] = {
                    "is_local": True,
                    "use_cloud": False,
                    "headless": CONFIG.get("BROWSER_HEADLESS", True),
                }
                if chrome_path:
                    bp_kwargs["executable_path"] = chrome_path

                browser_profile_for_agent = BrowserProfile(**bp_kwargs)
        except Exception:
            browser_profile_for_agent = None

        agent: Any = Agent(
            task=task_text,
            llm=cast(Any, llm_for_agent),
            browser_profile=browser_profile_for_agent,
            register_new_step_callback=step_callback,
            register_done_callback=done_callback,
        )

        agent_result = await agent.run(max_steps=CONFIG["MAX_AGENT_STEPS"])

        # Summarize results
        final_result = "No final result available"
        is_successful = False
        has_errors = False
        errors: list = []
        urls_visited: list[str] = []
        action_names: list = []
        extracted_content: list = []
        steps_taken = 0

        if agent_result:
            if hasattr(agent_result, "all_results"):
                results = agent_result.all_results
                if results:
                    last_result = results[-1]
                    if (
                        hasattr(last_result, "extracted_content")
                        and last_result.extracted_content
                    ):
                        final_result = str(last_result.extracted_content)

                    is_successful = any(
                        hasattr(r, "is_done") and r.is_done for r in results
                    )

                    errors = [
                        str(r.error)
                        for r in results
                        if hasattr(r, "error") and r.error
                    ]
                    has_errors = len(errors) > 0

                    extracted_content = [
                        str(r.extracted_content)
                        for r in results
                        if hasattr(r, "extracted_content")
                        and r.extracted_content
                    ]

                    steps_taken = len(results)

            if hasattr(agent_result, "all_model_outputs"):
                action_names = [
                    (
                        list(output.keys())[0]
                        if isinstance(output, dict) and output
                        else None
                    )
                    for output in agent_result.all_model_outputs
                ]
                action_names = [a for a in action_names if a]

        response_data = {
            "final_result": final_result,
            "success": is_successful,
            "has_errors": has_errors,
            "errors": errors,
            "urls_visited": urls_visited,
            "actions_performed": action_names,
            "extracted_content": extracted_content,
            "steps_taken": steps_taken,
        }

        task_store[task_id]["status"] = "completed"
        task_store[task_id]["end_time"] = datetime.now().isoformat()
        task_store[task_id]["result"] = response_data

    except Exception as e:
        logger.error("Error in async browser task: %s", str(e))
        tb = traceback.format_exc()

        task_store[task_id]["status"] = "failed"
        task_store[task_id]["end_time"] = datetime.now().isoformat()
        task_store[task_id]["error"] = str(e)
        task_store[task_id]["traceback"] = tb

    finally:
        try:

            async def _maybe_await_close(obj):
                if not obj:
                    return
                if hasattr(obj, "close") and callable(getattr(obj, "close")):
                    res = obj.close()
                    if asyncio.iscoroutine(res):
                        await res

            await _maybe_await_close(context)
            await _maybe_await_close(browser)
            logger.info("Browser resources for task %s cleaned up", task_id)
        except Exception as e:
            logger.error(
                "Error cleaning up browser resources for task %s: %s",
                task_id,
                str(e),
            )


async def cleanup_old_tasks() -> None:
    """
    Periodically clean up old completed tasks to prevent memory leaks.

    This function runs continuously in the background, removing tasks that have been
    completed or failed for more than 1 hour to conserve memory.
    """
    while True:
        try:
            # Sleep first to avoid cleaning up tasks too early
            await asyncio.sleep(CONFIG["CLEANUP_INTERVAL_SECONDS"])

            current_time = datetime.now()
            tasks_to_remove = []

            # Find completed tasks older than 1 hour
            for task_id, task_data in task_store.items():
                if (
                    task_data["status"] in ["completed", "failed"]
                    and "end_time" in task_data
                ):
                    end_time = datetime.fromisoformat(task_data["end_time"])
                    hours_elapsed = (
                        current_time - end_time
                    ).total_seconds() / 3600

                    if hours_elapsed > 1:  # Remove tasks older than 1 hour
                        tasks_to_remove.append(task_id)

            # Remove old tasks
            for task_id in tasks_to_remove:
                del task_store[task_id]

            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

        except Exception as e:
            logger.error(f"Error in task cleanup: {str(e)}")


def create_mcp_server(
    llm: Optional[Any],
    task_expiry_minutes: int = CONFIG["DEFAULT_TASK_EXPIRY_MINUTES"],
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> Server:
    """
    Create and configure an MCP server for browser interaction.

    Args:
        llm: The language model to use for browser agent
        task_expiry_minutes: Minutes after which tasks are considered expired
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale

    Returns:
        Configured MCP server instance
    """
    # Create MCP server instance
    app = Server("browser_use")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[
        Union[types.TextContent, types.ImageContent, types.EmbeddedResource]
    ]:
        """
        Handle tool calls from the MCP client.

        Args:
            name: The name of the tool to call
            arguments: The arguments to pass to the tool

            Returns:
            A list of content objects to return to the client.
            When PATIENT_MODE is enabled, the browser_use tool waits for
            the task to complete and returns the full result immediately
            instead of just the task ID.

        Raises:
            ValueError: If required arguments are missing
        """
        # Handle browser_use tool
        if name == "browser_use":
            # Check required arguments
            if "url" not in arguments:
                raise ValueError("Missing required argument 'url'")
            if "action" not in arguments:
                raise ValueError("Missing required argument 'action'")

            # If the action requests a summary, use a deterministic direct
            # summarization fallback that fetches the page and calls the
            # OpenAI Chat Completions API. This avoids instability from the
            # browser-use agent's LLM interface mismatches and ensures a
            # reliable response for summarization requests.
            action_text = str(arguments.get("action", ""))
            url_text = str(arguments.get("url", ""))
            if "summarize" in action_text.lower():
                try:
                    # Fetch page
                    resp = requests.get(url_text, timeout=15)
                    resp.raise_for_status()
                    html_text = resp.text
                    # Naive HTML -> text by stripping tags
                    text_only = re.sub(
                        r"<script[\s\S]*?</script>",
                        "",
                        html_text,
                        flags=re.IGNORECASE,
                    )
                    text_only = re.sub(
                        r"<style[\s\S]*?</style>",
                        "",
                        text_only,
                        flags=re.IGNORECASE,
                    )
                    text_only = re.sub(r"<[^>]+>", " ", text_only)
                    text_only = html.unescape(text_only)
                    # Truncate to a reasonable token window
                    excerpt = text_only.strip()[:32000]

                    # Build prompt
                    max_chars = 300
                    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
                    openai_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_key:
                        raise RuntimeError(
                            "OPENAI_API_KEY not set for direct summarizer"
                        )

                    prompt = (
                        f"Summarize the following web page content in at most "
                        f"{max_chars} characters. Output only the summary text "
                        "with no extra commentary.\n\nContent:\n"
                    ) + excerpt

                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a concise summarizer.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.2,
                    }

                    r = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=30,
                    )
                    r.raise_for_status()
                    content = r.json()
                    summary = ""
                    try:
                        summary = content["choices"][0]["message"][
                            "content"
                        ].strip()
                    except Exception:
                        summary = json.dumps(content)[:1000]

                    # Ensure length constraint
                    if len(summary) > max_chars:
                        summary = summary[: max_chars - 1]

                    result_obj = {
                        "final_result": summary,
                        "success": True,
                        "has_errors": False,
                        "errors": [],
                        "urls_visited": [url_text],
                        "actions_performed": ["summarize"],
                        "extracted_content": [summary],
                        "steps_taken": 1,
                    }

                    return [
                        types.TextContent(
                            type="text", text=json.dumps(result_obj, indent=2)
                        )
                    ]
                except Exception as e:
                    logger.error("Direct summarizer failed: %s", str(e))
                    # Fall through to normal async task path

            # Generate a task ID
            task_id = str(uuid.uuid4())

            # Extract optional agent configuration parameters
            allowed_domains = arguments.get("allowed_domains")
            use_vision = arguments.get("use_vision", False)
            max_steps = arguments.get("max_steps")

            # Build enhanced config for agent
            agent_config = CONFIG.copy()
            if allowed_domains is not None:
                agent_config["allowed_domains"] = allowed_domains
            if use_vision:
                agent_config["use_vision"] = use_vision
            if max_steps is not None:
                agent_config["max_steps"] = max_steps

            # Initialize task in store
            task_store[task_id] = {
                "id": task_id,
                "status": "pending",
                "url": arguments["url"],
                "action": arguments["action"],
                "created_at": datetime.now().isoformat(),
            }

            # Start task in background
            _task = asyncio.create_task(
                run_browser_task_async(
                    task_id=task_id,
                    url=arguments["url"],
                    action=arguments["action"],
                    llm=llm,
                    window_width=window_width,
                    window_height=window_height,
                    locale=locale,
                    config=agent_config,
                )
            )

            # If PATIENT is set, wait for the task to complete
            if CONFIG["PATIENT_MODE"]:
                try:
                    await _task
                    # Return the completed task result instead of just the ID
                    task_data = task_store[task_id]
                    if task_data["status"] == "failed":
                        logger.error(
                            (
                                f"Task {task_id} failed: "
                                f"{task_data.get('error', 'Unknown error')}"
                            )
                        )
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(task_data, indent=2),
                        )
                    ]
                except Exception as e:
                    logger.error(f"Error in patient mode execution: {str(e)}")
                    traceback_str = traceback.format_exc()
                    # Update task store with error
                    task_store[task_id]["status"] = "failed"
                    task_store[task_id]["error"] = str(e)
                    task_store[task_id]["traceback"] = traceback_str
                    task_store[task_id]["end_time"] = datetime.now().isoformat()
                    # Return error information
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(task_store[task_id], indent=2),
                        )
                    ]

            # Return task ID immediately with explicit sleep instruction
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "pending",
                            "message": (
                                f"Browser task started. Please wait for "
                                f"{CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} "
                                "seconds. Then check the result using "
                                "browser_get_result or the resource URI. "
                                "Always wait exactly 5 seconds between "
                                "status checks."
                            ),
                            "estimated_time": (
                                f"{CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} "
                                "seconds"
                            ),
                            "resource_uri": (
                                f"resource://browser_task/{task_id}"
                            ),
                            "sleep_command": "sleep 5",
                            "instruction": (
                                "Use the terminal command 'sleep 5' to wait 5 "
                                "seconds between status checks. IMPORTANT: "
                                "Always use exactly 5 seconds, no more and "
                                "no less."
                            ),
                        },
                        indent=2,
                    ),
                )
            ]

        # Handle browser_get_result tool
        elif name == "browser_get_result":
            # Get result of async task
            if "task_id" not in arguments:
                raise ValueError("Missing required argument 'task_id'")

            task_id = arguments["task_id"]

            if task_id not in task_store:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Task not found", "task_id": task_id},
                            indent=2,
                        ),
                    )
                ]

            # Get the current task data
            task_data = task_store[task_id].copy()

            # If task is still running, add simple guidance
            if task_data["status"] == "running":
                # Add a simple next check suggestion
                progress = task_data.get("progress", {})
                current_step = progress.get("current_step", 0)

                if current_step > 0:
                    # Simple message based on current step
                    task_data["message"] = (
                        f"Task is running (step {current_step}). "
                        "Wait 5 seconds before checking again."
                    )
                    task_data["sleep_command"] = "sleep 5"
                    task_data["instruction"] = (
                        "Use the terminal command 'sleep 5' to wait 5 seconds "
                        "before checking again. IMPORTANT: Always use exactly "
                        "5 seconds, no more and no less."
                    )
                else:
                    task_data["message"] = (
                        "Task is starting. Wait 5 seconds before checking "
                        "again."
                    )
                    task_data["sleep_command"] = "sleep 5"
                    task_data["instruction"] = (
                        "Use the terminal command 'sleep 5' to wait 5 seconds "
                        "before checking again. IMPORTANT: Always use exactly "
                        "5 seconds, no more and no less."
                    )

            # Return current task status and result if available
            return [
                types.TextContent(
                    type="text", text=json.dumps(task_data, indent=2)
                )
            ]

        # Handle browser_get_state tool
        elif name == "browser_get_state":
            import base64

            from server import session as session_manager

            # Support both task_id (legacy) and session_id (new)
            task_id_arg = cast(Optional[str], arguments.get("task_id"))
            session_id_arg = cast(Optional[str], arguments.get("session_id"))
            include_screenshot = cast(bool, arguments.get("screenshot", False))

            # If session_id provided, get live state from session
            if session_id_arg:
                browser_session = await session_manager.get_session(
                    session_id_arg
                )
                if not browser_session:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": f"Session {session_id_arg} not found"
                                },
                                indent=2,
                            ),
                        )
                    ]

                # Get live browser state
                state_response = (
                    await browser_session.get_browser_state_summary()
                )

                # Add screenshot if requested
                if include_screenshot:
                    try:
                        page = browser_session.get_current_page()
                        screenshot_bytes = await page.screenshot()
                        state_response["screenshot"] = base64.b64encode(
                            screenshot_bytes
                        ).decode("utf-8")
                    except Exception as e:
                        state_response["screenshot_error"] = str(e)

                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(state_response, indent=2),
                    )
                ]

            # Legacy: get state from task_store
            if not task_id_arg:
                raise ValueError("Either task_id or session_id required")

            # At this point task_id_arg is guaranteed to be a string; cast for typing
            # Get state from task store (inline cast for typing)
            if cast(str, task_id_arg) not in task_store:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Task not found",
                                "task_id": cast(str, task_id_arg),
                            },
                            indent=2,
                        ),
                    )
                ]

            task_data = task_store[cast(str, task_id_arg)].copy()

            # Build state response
            state_response = {
                "url": task_data.get("url", ""),
                "title": task_data.get("title", ""),
                "tabs": task_data.get("tabs", []),
                "interactive_elements": task_data.get(
                    "interactive_elements", []
                ),
            }

            return [
                types.TextContent(
                    type="text", text=json.dumps(state_response, indent=2)
                )
            ]

        # Handle browser_navigate tool
        elif name == "browser_navigate":
            from server import session as session_manager

            if "url" not in arguments:
                raise ValueError("Missing required argument 'url'")

            url = arguments["url"]
            new_tab = arguments.get("new_tab", False)
            session_id = arguments.get("session_id", "default")

            # Get or create session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                browser_session = await session_manager.create_session(
                    session_id=session_id,
                    headless=CONFIG.get("BROWSER_HEADLESS", True),
                    chrome_path=os.environ.get("CHROME_PATH"),
                )

            # Navigate using event bus
            from browser_use.browser.events import NavigateToUrlEvent

            event = browser_session.event_bus.dispatch(
                NavigateToUrlEvent(url=url, new_tab=new_tab)
            )
            await event

            message = (
                f"Opened new tab with URL: {url}"
                if new_tab
                else f"Navigated to: {url}"
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "message": message,
                            "url": url,
                            "session_id": session_id,
                        },
                        indent=2,
                    ),
                )
            ]

        # Handle browser_click tool
        elif name == "browser_click":
            from server import session as session_manager

            if "index" not in arguments:
                raise ValueError("Missing required argument 'index'")

            index = arguments["index"]
            new_tab = arguments.get("new_tab", False)
            session_id = arguments.get("session_id", "default")

            # Get session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": f"No active session {session_id}. Use browser_navigate first."
                            },
                            indent=2,
                        ),
                    )
                ]

            # Get element by index
            element = await browser_session.get_dom_element_by_index(index)
            if not element:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Element with index {index} not found"},
                            indent=2,
                        ),
                    )
                ]

            # Click the element
            from browser_use.browser.events import ClickElementEvent

            event = browser_session.event_bus.dispatch(
                ClickElementEvent(node=element)
            )
            await event

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "message": f"Clicked element {index}",
                            "session_id": session_id,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "browser_list_sessions":
            from server import session as session_manager

            # Get all active sessions
            sessions = session_manager.list_sessions()

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"sessions": sessions},
                        indent=2,
                    ),
                )
            ]

        elif name == "browser_close_session":
            from server import session as session_manager

            if "session_id" not in arguments:
                raise ValueError("Missing required argument 'session_id'")

            session_id = arguments["session_id"]

            # Close the session
            await session_manager.close_session(session_id)

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "message": "Session closed successfully",
                            "session_id": session_id,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "browser_extract_content":
            from server import session as session_manager

            if "session_id" not in arguments:
                raise ValueError("Missing required argument 'session_id'")
            if "instruction" not in arguments:
                raise ValueError("Missing required argument 'instruction'")

            session_id = arguments["session_id"]
            instruction = arguments["instruction"]

            # Get session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Session {session_id} not found"},
                            indent=2,
                        ),
                    )
                ]

            # Extract content using session's extract_content method
            try:
                extracted = await browser_session.extract_content(instruction)
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(extracted, indent=2),
                    )
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Extraction failed: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        elif name == "browser_list_tabs":
            from server import session as session_manager

            if "session_id" not in arguments:
                raise ValueError("Missing required argument 'session_id'")

            session_id = arguments["session_id"]

            # Get session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Session {session_id} not found"},
                            indent=2,
                        ),
                    )
                ]

            # List tabs
            tabs = await browser_session.list_tabs()
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"tabs": tabs}, indent=2),
                )
            ]

        elif name == "browser_switch_tab":
            from server import session as session_manager

            if "session_id" not in arguments:
                raise ValueError("Missing required argument 'session_id'")
            if "tab_index" not in arguments:
                raise ValueError("Missing required argument 'tab_index'")

            session_id = arguments["session_id"]
            tab_index = arguments["tab_index"]

            # Get session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Session {session_id} not found"},
                            indent=2,
                        ),
                    )
                ]

            # Switch tab
            await browser_session.switch_tab(tab_index)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "message": f"Switched to tab {tab_index}",
                            "session_id": session_id,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "browser_close_tab":
            from server import session as session_manager

            if "session_id" not in arguments:
                raise ValueError("Missing required argument 'session_id'")
            if "tab_index" not in arguments:
                raise ValueError("Missing required argument 'tab_index'")

            session_id = arguments["session_id"]
            tab_index = arguments["tab_index"]

            # Get session
            browser_session = await session_manager.get_session(session_id)
            if not browser_session:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Session {session_id} not found"},
                            indent=2,
                        ),
                    )
                ]

            # Close tab
            await browser_session.close_tab(tab_index)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "message": f"Closed tab {tab_index}",
                            "session_id": session_id,
                        },
                        indent=2,
                    ),
                )
            ]

        else:
            raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """
        List the available tools for the MCP client.

        Returns different tool descriptions based on the PATIENT_MODE
        configuration. When PATIENT_MODE is enabled, the browser_use tool
        description indicates it returns complete results directly. When
        disabled, it indicates async operation.

        Returns:
            A list of tool definitions appropriate for the current configuration
        """
        patient_mode = CONFIG["PATIENT_MODE"]

        if patient_mode:
            return [
                types.Tool(
                    name="browser_use",
                    description=(
                        "Performs a browser action and returns the complete "
                        "result directly (patient mode active)"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["url", "action"],
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to",
                            },
                            "action": {
                                "type": "string",
                                "description": "Action to perform in the browser",
                            },
                            "allowed_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of allowed domains for navigation "
                                    "(security)"
                                ),
                            },
                            "use_vision": {
                                "type": "boolean",
                                "description": (
                                    "Use vision/screenshot analysis for actions"
                                ),
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of agent steps to execute"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_get_result",
                    description=(
                        "Gets the result of an asynchronous browser task. "
                        "(Not needed in patient mode as browser_use returns "
                        "complete results directly)"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["task_id"],
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "ID of the task to get results for",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_get_state",
                    description=(
                        "Get the current state of the browser including URL, "
                        "title, tabs, and interactive elements. Supports both task_id "
                        "(legacy) and session_id (live state with optional screenshot)"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": (
                                    "ID of the task to get state for (legacy)"
                                ),
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "ID of the session to get live state for"
                                ),
                            },
                            "screenshot": {
                                "type": "boolean",
                                "description": (
                                    "Include base64-encoded screenshot "
                                    "(only with session_id)"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_navigate",
                    description=(
                        "Navigate to a URL in a browser session. Creates a new "
                        "session if session_id is not provided."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to",
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "Browser session ID (creates new if "
                                    "not provided)"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_click",
                    description=(
                        "Click an interactive element by its index from the "
                        "browser state"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "element_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "element_index": {
                                "type": "integer",
                                "description": (
                                    "Index of element to click from "
                                    "browser state"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_list_sessions",
                    description="List all active browser sessions",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                types.Tool(
                    name="browser_close_session",
                    description="Close an active browser session",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID to close",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_extract_content",
                    description=(
                        "Extract specific content from the current page using "
                        "natural language instruction"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "instruction"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "instruction": {
                                "type": "string",
                                "description": (
                                    "Natural language instruction for what to "
                                    "extract"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_list_tabs",
                    description="List all tabs in a browser session",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_switch_tab",
                    description="Switch to a specific tab by index",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "tab_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "tab_index": {
                                "type": "integer",
                                "description": "Index of the tab to switch to",
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_close_tab",
                    description="Close a specific tab by index",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "tab_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "tab_index": {
                                "type": "integer",
                                "description": "Index of the tab to close",
                            },
                        },
                    },
                ),
            ]
        else:
            return [
                types.Tool(
                    name="browser_use",
                    description=(
                        "Performs a browser action and returns a task ID "
                        "for async execution"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["url", "action"],
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to",
                            },
                            "action": {
                                "type": "string",
                                "description": "Action to perform in the browser",
                            },
                            "allowed_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of allowed domains for navigation "
                                    "(security)"
                                ),
                            },
                            "use_vision": {
                                "type": "boolean",
                                "description": (
                                    "Use vision/screenshot analysis for actions"
                                ),
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of agent steps to execute"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_get_result",
                    description="Gets the result of an asynchronous browser task",
                    inputSchema={
                        "type": "object",
                        "required": ["task_id"],
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "ID of the task to get results for",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_get_state",
                    description=(
                        "Get the current state of the browser including URL, "
                        "title, tabs, and interactive elements. Supports both task_id "
                        "(legacy) and session_id (live state with optional screenshot)"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": (
                                    "ID of the task to get state for (legacy)"
                                ),
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "ID of the session to get live state for"
                                ),
                            },
                            "screenshot": {
                                "type": "boolean",
                                "description": (
                                    "Include base64-encoded screenshot "
                                    "(only with session_id)"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_navigate",
                    description=(
                        "Navigate to a URL in a browser session. Creates a new "
                        "session if session_id is not provided."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to",
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "Browser session ID (creates new if "
                                    "not provided)"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_click",
                    description=(
                        "Click an interactive element by its index from the "
                        "browser state"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "element_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "element_index": {
                                "type": "integer",
                                "description": (
                                    "Index of element to click from "
                                    "browser state"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_list_sessions",
                    description="List all active browser sessions",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                types.Tool(
                    name="browser_close_session",
                    description="Close an active browser session",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID to close",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_extract_content",
                    description=(
                        "Extract specific content from the current page using "
                        "natural language instruction"
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "instruction"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "instruction": {
                                "type": "string",
                                "description": (
                                    "Natural language instruction for what to "
                                    "extract"
                                ),
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_list_tabs",
                    description="List all tabs in a browser session",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_switch_tab",
                    description="Switch to a specific tab by index",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "tab_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "tab_index": {
                                "type": "integer",
                                "description": "Index of the tab to switch to",
                            },
                        },
                    },
                ),
                types.Tool(
                    name="browser_close_tab",
                    description="Close a specific tab by index",
                    inputSchema={
                        "type": "object",
                        "required": ["session_id", "tab_index"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Browser session ID",
                            },
                            "tab_index": {
                                "type": "integer",
                                "description": "Index of the tab to close",
                            },
                        },
                    },
                ),
            ]

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        """
        List the available resources for the MCP client.

        Returns:
            A list of resource definitions
        """
        # List all completed tasks as resources
        resources = []
        for task_id, task_data in task_store.items():
            if task_data["status"] in ["completed", "failed"]:
                # Construct a plain dict and cast to types.Resource to avoid
                # strict constructor signature checks during static typing.
                resources.append(
                    cast(
                        types.Resource,
                        {
                            "uri": f"resource://browser_task/{task_id}",
                            "title": f"Browser Task Result: {task_id[:8]}",
                            "description": (
                                f"Result of browser task for URL: "
                                f"{task_data.get('url', 'unknown')}"
                            ),
                        },
                    )
                )
        return resources

    @app.read_resource()
    async def read_resource(uri: str) -> list[types.ResourceContents]:
        """
        Read a resource for the MCP client.

        Args:
            uri: The URI of the resource to read

        Returns:
            The contents of the resource
        """
        # Extract task ID from URI
        if not uri.startswith("resource://browser_task/"):
            return [
                cast(
                    types.ResourceContents,
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Invalid resource URI: {uri}"}, indent=2
                        ),
                    },
                )
            ]

        task_id = uri.replace("resource://browser_task/", "")
        if task_id not in task_store:
            return [
                cast(
                    types.ResourceContents,
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Task not found: {task_id}"}, indent=2
                        ),
                    },
                )
            ]

        # Return task data
        return [
            cast(
                types.ResourceContents,
                {
                    "type": "text",
                    "text": json.dumps(task_store[task_id], indent=2),
                },
            )
        ]

    # Add cleanup_old_tasks function to app for later scheduling
    app.cleanup_old_tasks = cleanup_old_tasks  # type: ignore[attr-defined]

    return app


@click.command()
@click.option("--port", default=8081, help="Port to listen on for SSE")
@click.option(
    "--proxy-port",
    default=None,
    type=int,
    help="Port for the proxy to listen on. If specified, enables proxy mode.",
)
@click.option("--chrome-path", default=None, help="Path to Chrome executable")
@click.option(
    "--window-width",
    default=CONFIG["DEFAULT_WINDOW_WIDTH"],
    help="Browser window width",
)
@click.option(
    "--window-height",
    default=CONFIG["DEFAULT_WINDOW_HEIGHT"],
    help="Browser window height",
)
@click.option(
    "--locale", default=CONFIG["DEFAULT_LOCALE"], help="Browser locale"
)
@click.option(
    "--task-expiry-minutes",
    default=CONFIG["DEFAULT_TASK_EXPIRY_MINUTES"],
    help="Minutes after which tasks are considered expired",
)
@click.option(
    "--stdio",
    is_flag=True,
    default=False,
    help="Enable stdio mode. If specified, enables proxy mode.",
)
@click.option(
    "--headless/--no-headless",
    default=None,
    help="Run browsers in headless mode (default: from BROWSER_HEADLESS env)",
)
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level for server (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
def main(
    port: int,
    proxy_port: Optional[int],
    chrome_path: str,
    window_width: int,
    window_height: int,
    locale: str,
    task_expiry_minutes: int,
    stdio: bool,
    headless: Optional[bool],
    log_level: str,
) -> int:
    """
    Run the browser-use MCP server.

    This function initializes the MCP server and runs it with the SSE transport.
    Each browser task will create its own isolated browser context.

    The server can run in two modes:
    1. Direct SSE mode (default): Just runs the SSE server
     2. Proxy mode (enabled by --stdio or --proxy-port):
         Runs both SSE server and mcp-proxy

    Args:
        port: Port to listen on for SSE
        proxy_port: Port for the proxy to listen on. If specified, enables proxy mode.
        chrome_path: Path to Chrome executable
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale
        task_expiry_minutes: Minutes after which tasks are considered expired
        stdio: Enable stdio mode. If specified, enables proxy mode.

    Returns:
        Exit code (0 for success)
    """
    # Store Chrome path in environment variable if provided
    if chrome_path:
        os.environ["CHROME_PATH"] = chrome_path
        logger.info(f"Using Chrome path: {chrome_path}")
    else:
        logger.info(
            "No Chrome path specified, letting Playwright use its default browser"
        )

    # Initialize LLM and wrap with adapter for compatibility
    # Allow overriding the default model via environment variable `LLM_MODEL`.
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_raw = ChatOpenAI(model=model_name, temperature=0.0)
    llm = ensure_llm_adapter(llm_raw)

    # Create MCP server
    # If headless is explicitly provided via CLI, override CONFIG
    if headless is not None:
        CONFIG["BROWSER_HEADLESS"] = bool(headless)

    app = create_mcp_server(
        llm=llm,
        task_expiry_minutes=task_expiry_minutes,
        window_width=window_width,
        window_height=window_height,
        locale=locale,
    )

    sse = SseServerTransport("/messages/")

    # Create the Starlette app for SSE
    async def handle_sse(request):
        """Handle SSE connections from clients."""
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        except Exception as e:
            # Log the error and return an explicit HTTP 500 response instead
            # of re-raising. Re-raising here could cause the endpoint to end
            # without returning a Response object which results in Starlette
            # attempting to call a NoneType as an ASGI application. Returning
            # a PlainTextResponse on error guarantees a valid ASGI response
            # is always returned.
            logger.error(f"Error in handle_sse: {str(e)}", exc_info=True)
            return PlainTextResponse(
                f"Internal server error in SSE handler: {str(e)}",
                status_code=500,
            )

        # Starlette requires an ASGI-compatible response to be returned from the
        # endpoint handler. When the SSE connection ends the context manager
        # exits; returning an empty PlainTextResponse ensures the framework has
        # a proper response object and avoids a 'NoneType' callable error.
        return PlainTextResponse("")

    async def _health(request):
        """Simple health endpoint for Docker and load balancers."""
        return PlainTextResponse("ok")

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/health", endpoint=_health),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    # Add startup event
    @starlette_app.on_event("startup")
    async def startup_event():
        """Initialize the server on startup."""
        logger.info("Starting MCP server...")

        # Sanity checks for critical configuration
        if port <= 0 or port > 65535:
            logger.error(f"Invalid port number: {port}")
            raise ValueError(f"Invalid port number: {port}")

        if window_width <= 0 or window_height <= 0:
            logger.error(
                f"Invalid window dimensions: {window_width}x{window_height}"
            )
            raise ValueError(
                f"Invalid window dimensions: {window_width}x{window_height}"
            )

        if task_expiry_minutes <= 0:
            logger.error(f"Invalid task expiry minutes: {task_expiry_minutes}")
            raise ValueError(
                f"Invalid task expiry minutes: {task_expiry_minutes}"
            )

        # Start background task cleanup
        asyncio.create_task(app.cleanup_old_tasks())

        # Ensure a compatible `uvx` executable is available for subprocess
        # invocations performed by browser-use when installing or launching
        # Playwright browsers. Some runtime environments expect `uvx` to be
        # present; if not, create a lightweight wrapper that forwards to the
        # project's `uv` binary and prepend it to PATH so subprocess_exec can
        # find it without requiring root privileges.
        try:
            uvx_path = shutil.which("uvx")
            if not uvx_path:
                uv_path = shutil.which("uv")
                if uv_path:
                    wrapper_path = "/tmp/uvx"
                    with open(wrapper_path, "w", encoding="utf-8") as wf:
                        wf.write(f'#!/bin/sh\nexec "{uv_path}" "$@"\n')
                    os.chmod(wrapper_path, 0o755)
                    os.environ["PATH"] = f"/tmp:{os.environ.get('PATH', '')}"
                    logger.info(
                        "Created uvx wrapper at %s pointing to %s",
                        wrapper_path,
                        uv_path,
                    )
                else:
                    logger.debug(
                        (
                            "Neither 'uvx' nor 'uv' found in PATH; "
                            "skipping uvx wrapper creation"
                        )
                    )
        except Exception:
            logger.exception("Failed to create uvx wrapper")
        logger.info("Task cleanup process scheduled")

    # Function to run uvicorn in a separate thread
    def run_uvicorn():
        # Configure uvicorn to use JSON logging and respect chosen level
        # Support all standard Python logging levels including CRITICAL
        chosen_level = getattr(logging, log_level.upper(), logging.INFO)

        # Set root and common logger levels
        logging.getLogger().setLevel(chosen_level)
        logging.getLogger("uvicorn").setLevel(chosen_level)
        logging.getLogger("uvicorn.error").setLevel(chosen_level)
        logging.getLogger("uvicorn.access").setLevel(chosen_level)
        # Also set levels for noisy browser-use and related loggers
        for logger_name in [
            "browser_use",
            "playwright",
            "mcp",
            "Agent",
            "tools",
            "BrowserSession",
        ]:
            try:
                logging.getLogger(logger_name).setLevel(chosen_level)
            except Exception:
                pass

        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": (
                        '{"time":"%(asctime)s","level":"%(levelname)s",'
                        '"name":"%(name)s","message":"%(message)s"}'
                    ),
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": log_level.upper()},
                "uvicorn": {
                    "handlers": ["default"],
                    "level": log_level.upper(),
                },
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": log_level.upper(),
                },
                "uvicorn.access": {
                    "handlers": ["default"],
                    "level": log_level.upper(),
                },
            },
        }

        uvicorn.run(
            starlette_app,
            host="0.0.0.0",  # nosec
            port=port,
            log_config=log_config,
            log_level=log_level.lower(),
        )

    # If proxy mode is enabled, run both the SSE server and mcp-proxy
    if stdio:
        import subprocess  # nosec

        # Start the SSE server in a separate thread
        sse_thread = threading.Thread(target=run_uvicorn)
        sse_thread.daemon = True
        sse_thread.start()

        # Give the SSE server a moment to start
        time.sleep(1)

        proxy_cmd = [
            "mcp-proxy",
            f"http://localhost:{port}/sse",
            "--sse-port",
            str(proxy_port),
            "--allow-origin",
            "*",
        ]

        logger.info(f"Running proxy command: {' '.join(proxy_cmd)}")
        logger.info(
            f"SSE server running on port {port}, proxy running on port {proxy_port}"
        )

        try:
            # Using trusted command arguments from CLI parameters
            with subprocess.Popen(proxy_cmd) as proxy_process:  # nosec
                proxy_process.wait()
        except Exception as e:
            logger.error(f"Error starting mcp-proxy: {str(e)}")
            logger.error(f"Command was: {' '.join(proxy_cmd)}")
            return 1
    else:
        logger.info(f"Running in direct SSE mode on port {port}")
        run_uvicorn()

    return 0


if __name__ == "__main__":
    main()
