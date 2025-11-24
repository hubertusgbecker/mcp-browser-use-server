# GitHub Copilot Instructions for mcp-browser-use-server

> Comprehensive guide for GitHub Copilot to provide context-aware suggestions for mcp-browser-use-server

## Table of Contents

1. [Project Context](#project-context)
2. [Critical Development Rules](#critical-development-rules)
3. [Code Patterns](#code-patterns)
4. [MCP-Specific Patterns](#mcp-specific-patterns)
5. [Session Management](#session-management)
6. [Testing Patterns](#testing-patterns)
7. [Common Tasks](#common-tasks)
8. [Configuration & Environment](#configuration--environment)
9. [Error Handling & Logging](#error-handling--logging)
10. [Performance & Optimization](#performance--optimization)
11. [Security Guidelines](#security-guidelines)
12. [Troubleshooting Guide](#troubleshooting-guide)
13. [Quick Reference](#quick-reference)

---

## Project Context

You are working on **mcp-browser-use-server**, a production-ready MCP (Model Context Protocol) server for browser automation. This project enables AI agents to control web browsers through the browser-use library.

**Key Facts:**
- **Language:** Python 3.11+
- **Package Manager:** uv (Astral) - **MANDATORY**
- **Protocol:** MCP (Model Context Protocol)
- **Browser Engine:** Playwright + browser-use
- **LLM Integration:** OpenAI + LangChain
- **Transports:** SSE (web) and stdio (local)
- **Test Coverage Target:** >95%
- **License:** MIT


---

## Critical Development Rules

**These rules are MANDATORY and MUST be followed in all suggestions:**

### 1. Package Manager: uv Only (Non-Negotiable)

```bash
# ✅ CORRECT - Always use uv
uv sync                    # Install dependencies
uv pip install package     # Add package
uv run script.py          # Run scripts
uv build                  # Build package
uv tool install pkg       # Install as tool

# ❌ WRONG - Never suggest these
pip install package
python -m pip install
conda install
```

**Why:** For consistency and reproducibility across developer machines and CI.

**CI Integration:**
```yaml
# Always use in GitHub Actions
- uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    python-version: "3.11"
- run: uv sync
- run: uv run pytest tests/
```

### 2. Type Safety First (Required)

**All functions MUST have:**
- Complete type hints on parameters and return values
- Pydantic v2 models for data validation and schemas
- Proper Optional/Union types for nullable values

```python
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# ✅ CORRECT
class TaskConfig(BaseModel):
    """Task configuration schema."""
    instruction: str = Field(..., description="Task instruction")
    task_id: Optional[str] = Field(None, description="Optional task ID")
    timeout: int = Field(120, ge=1, description="Timeout in seconds")

async def execute_task(
    config: TaskConfig,
    llm: BaseLanguageModel,
) -> Dict[str, Any]:
    """Execute browser task with config."""
    ...

# ❌ WRONG - No type hints
def execute_task(config, llm):
    ...
```

**Verification:**
```bash
uv run mypy src/ server/  # Must pass with no errors
```

### 3. Async by Default (Required)

**All I/O operations MUST be async:**

```python
# ✅ CORRECT
async def fetch_data(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# ✅ CORRECT - Test marked as async
@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data("https://example.com")
    assert result is not None

# ❌ WRONG - Blocking I/O in async context
async def fetch_data(url: str):
    response = requests.get(url)  # Blocking!
    return response.json()
```

### 4. Comprehensive Testing (>95% Coverage)

**Every feature addition MUST include:**

1. **Unit tests** - Test individual functions in isolation
2. **Integration tests** - Test component interactions
3. **Fixtures** - Use from `tests/conftest.py`
4. **Coverage** - Verify with `./run_tests.sh coverage`

```python
# ✅ CORRECT - Complete test
@pytest.mark.asyncio
async def test_create_session(mock_config, cleanup_tasks):
    """Test session creation with proper cleanup.
    
    Verifies:
    - Session is created with unique ID
    - Session is tracked in sessions dict
    - Browser context is initialized
    """
    session_id = str(uuid.uuid4())
    session = await create_session(session_id, mock_config)
    
    assert session is not None
    assert session.session_id == session_id
    assert session_id in sessions
    
    # Cleanup
    await session.close()
```

**Test Commands:**
```bash
./run_tests.sh fast        # Quick tests (pre-commit)
./run_tests.sh all         # Full suite
./run_tests.sh coverage    # With coverage report
uv run pytest tests/test_server.py::test_name -vv  # Specific test
```

### 5. Documentation Standards (Google Style)

**All public functions MUST have:**

```python
async def process_task(
    task_id: str,
    instruction: str,
    config: Dict[str, Any],
    timeout: int = 120,
) -> Dict[str, Any]:
    """Process a browser automation task.
    
    Executes the given instruction using browser-use agent and
    returns the result with status information.
    
    Args:
        task_id: Unique identifier for the task
        instruction: Natural language instruction for the agent
        config: Configuration dictionary with API keys and settings
        timeout: Maximum execution time in seconds (default: 120)
    
    Returns:
        Dictionary containing:
        - status: "completed", "failed", or "timeout"
        - result: Task execution result
        - duration: Execution time in seconds
        - error: Error message if failed (optional)
    
    Raises:
        ValueError: If task_id is empty or instruction is invalid
        TimeoutError: If execution exceeds timeout
        RuntimeError: If browser initialization fails
    
    Example:
        >>> config = init_configuration()
        >>> result = await process_task(
        ...     "task-123",
        ...     "Navigate to example.com",
        ...     config
        ... )
        >>> assert result["status"] == "completed"
    """
    ...
```

**Documentation Updates:**
- README.md - User-facing API changes
- AGENTS.md - AI agent context and patterns
- Inline comments for complex logic

### 6. Pydantic v2 for All Schemas

**Use Pydantic v2 models for:**
- MCP tool input/output schemas
- Configuration validation
- Task parameters
- Session state

```python
from pydantic import BaseModel, Field, validator

class BrowserTaskInput(BaseModel):
    """Input schema for browser task execution."""
    instruction: str = Field(..., min_length=1, description="Task instruction")
    task_id: Optional[str] = Field(None, description="Custom task ID")
    timeout: int = Field(120, ge=10, le=600, description="Timeout in seconds")
    headless: bool = Field(True, description="Run in headless mode")
    
    @validator("instruction")
    def validate_instruction(cls, v: str) -> str:
        """Validate instruction is not just whitespace."""
        if not v.strip():
            raise ValueError("Instruction cannot be empty or whitespace")
        return v.strip()

class BrowserTaskOutput(BaseModel):
    """Output schema for browser task execution."""
    task_id: str
    status: Literal["completed", "failed", "running"]
    result: Optional[str] = None
    error: Optional[str] = None
    duration: float
    created_at: str
```

### 7. Security First

**NEVER:**
- Commit `.env` files
- Hardcode API keys or secrets
- Log sensitive data
- Expose internal errors to users

```python
# ✅ CORRECT
config = init_configuration()  # Reads from .env
api_key = config["OPENAI_API_KEY"]

# ✅ CORRECT - Sanitized error
except Exception as e:
    logger.error(f"Task failed: {type(e).__name__}", exc_info=True)
    return {"error": "Task execution failed"}  # Generic message

# ❌ WRONG
api_key = "sk-abc123..."  # Hardcoded!

# ❌ WRONG - Exposes internals
except Exception as e:
    return {"error": str(e)}  # May contain secrets!
```

### 8. Context7 Integration

Always use Context7 MCP tools for:
- Code generation and boilerplate
- Library API documentation lookup
- Configuration examples
- Best practices for dependencies

**Automatically fetch docs without explicit user request when:**
- Suggesting code for external libraries
- Implementing new features with unfamiliar APIs
- Troubleshooting library-specific issues

## Code Patterns

### Function Signature Pattern
```python
async def my_function(
    param1: str,
    param2: int,
    config: Dict[str, Any]
) -> Optional[MyReturnType]:
    """Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2
        config: Configuration dictionary

    Returns:
        Description of return value or None if failure

    Raises:
        ValueError: When param1 is invalid
    """
    try:
        # Implementation
        result = await some_async_operation()
        return result
    except Exception as e:
        logger.error(f"Error in my_function: {e}")
        raise
```

### Browser Task Pattern
```python
async def run_browser_task_async(
    task_id: str,
    instruction: str,
    llm: BaseLanguageModel,
    config: Dict[str, Any]
) -> None:
    """Execute browser task asynchronously."""
    task_store[task_id] = {
        "status": "running",
        "instruction": instruction,
        "created_at": datetime.now().isoformat(),
    }

    try:
        context, browser = await create_browser_context_for_task(task_id, config)
        agent = Agent(task=instruction, llm=llm, browser=context)
        result = await agent.run()

        task_store[task_id].update({
            "status": "completed",
            "result": result,
        })
    except Exception as e:
        task_store[task_id].update({
            "status": "failed",
            "error": str(e),
        })
    finally:
        await context.close()
        await browser.close()
```

### Test Pattern
```python
class TestMyFeature:
    """Test suite for my feature."""

    @pytest.mark.asyncio
    async def test_success_case(self, mock_config, cleanup_tasks):
        """Test successful execution."""
        result = await my_function("param", mock_config)
        assert result is not None
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config, cleanup_tasks):
        """Test error handling."""
        with pytest.raises(ValueError):
            await my_function("invalid", mock_config)
```

---

## MCP-Specific Patterns

### Tool Registration Pattern

**When adding new MCP tools, follow this exact structure:**

```python
# 1. Define tool in list_tools()
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        # ... existing tools ...
        types.Tool(
            name="my_new_tool",
            description="Clear, concise description of what the tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Description of param1"
                    },
                    "param2": {
                        "type": "integer",
                        "description": "Description of param2",
                        "default": 10
                    }
                },
                "required": ["param1"]
            }
        )
    ]

# 2. Implement handler in call_tool()
@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent | types.ImageContent]:
    if name == "my_new_tool":
        # Extract and validate parameters
        param1 = arguments.get("param1")
        if not param1:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "param1 is required"}, indent=2)
            )]
        
        param2 = arguments.get("param2", 10)
        
        try:
            # Execute tool logic
            result = await my_tool_implementation(param1, param2)
            
            # Return success response
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            logger.error(f"Error in my_new_tool: {e}", exc_info=True)
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    # ... other tools ...
```

### Resource Provider Pattern

**For serving data via MCP resources:**

```python
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available MCP resources."""
    resources = [
        types.Resource(
            uri="task://list",
            name="All Tasks",
            description="List of all browser tasks",
            mimeType="application/json"
        )
    ]
    
    # Dynamic resources (e.g., per-task)
    for task_id, task_data in task_store.items():
        resources.append(types.Resource(
            uri=f"task://{task_id}",
            name=f"Task {task_id}",
            description=f"Status: {task_data['status']}",
            mimeType="application/json"
        ))
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource by URI."""
    if uri == "task://list":
        return json.dumps(list(task_store.keys()), indent=2)
    
    if uri.startswith("task://"):
        task_id = uri.replace("task://", "")
        if task_id in task_store:
            return json.dumps(task_store[task_id], indent=2)
        raise ValueError(f"Task {task_id} not found")
    
    raise ValueError(f"Unknown resource URI: {uri}")
```

### MCP Response Format

**Always return properly formatted MCP responses:**

```python
# ✅ CORRECT - Structured JSON response
return [types.TextContent(
    type="text",
    text=json.dumps({
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }, indent=2)
)]

# ✅ CORRECT - Image response (screenshots)
return [types.ImageContent(
    type="image",
    data=base64_encoded_image,
    mimeType="image/png"
)]

# ✅ CORRECT - Multiple content items
return [
    types.TextContent(type="text", text="Task completed"),
    types.ImageContent(type="image", data=screenshot_data, mimeType="image/png")
]

# ❌ WRONG - Raw string
return "Task completed"  # Not MCP-compliant!

# ❌ WRONG - Unformatted dict
return {"status": "success"}  # Must wrap in TextContent!
```

---

## Session Management

### Session Lifecycle Pattern

**Sessions are long-lived browser contexts for multi-step interactions:**

```python
from server.session import BrowserSession

# Create session
async def create_session(
    session_id: str,
    config: Dict[str, Any]
) -> BrowserSession:
    """Create and initialize a new browser session.
    
    Args:
        session_id: Unique session identifier
        config: Configuration with browser settings
    
    Returns:
        Initialized BrowserSession instance
    """
    session = BrowserSession(session_id, config)
    await session.initialize()  # Setup browser context
    sessions[session_id] = session
    logger.info(f"Created session {session_id}")
    return session

# Use session
async def use_session(session_id: str) -> Dict[str, Any]:
    """Perform operations on existing session."""
    if session_id not in sessions:
        raise ValueError(f"Session {session_id} not found")
    
    session = sessions[session_id]
    
    # Navigate
    await session.navigate("https://example.com")
    
    # Get current state
    state = await session.get_state(screenshot=True)
    
    # Interact with elements
    await session.click_element(0)  # Click first element
    
    # Extract content with LLM
    content = await session.extract_content(
        "Extract all product names and prices"
    )
    
    return {
        "url": state["url"],
        "content": content
    }

# Cleanup session
async def cleanup_session(session_id: str) -> None:
    """Close and remove session."""
    if session_id in sessions:
        session = sessions[session_id]
        await session.close()
        del sessions[session_id]
        logger.info(f"Cleaned up session {session_id}")
```

### Session State Serialization

**Always serialize session state for MCP responses:**

```python
async def get_session_state(
    session_id: str,
    screenshot: bool = False
) -> Dict[str, Any]:
    """Get serializable session state.
    
    Returns:
        Dictionary with:
        - session_id: Session identifier
        - url: Current URL
        - title: Page title
        - elements: List of interactive elements
        - tabs: List of open tabs
        - screenshot: Base64 PNG if requested
    """
    session = sessions[session_id]
    state = await session.get_state(screenshot=screenshot)
    
    return {
        "session_id": session_id,
        "url": state["url"],
        "title": state["title"],
        "elements": [
            {
                "index": i,
                "tag": elem["tag"],
                "text": elem["text"][:100],  # Truncate
                "attributes": elem["attributes"]
            }
            for i, elem in enumerate(state["elements"])
        ],
        "tabs": state.get("tabs", []),
        "screenshot": state.get("screenshot") if screenshot else None
    }
```

### Tab Management Pattern

```python
async def manage_tabs(
    session_id: str,
    action: str,
    tab_index: Optional[int] = None
) -> Dict[str, Any]:
    """Manage browser tabs in session.
    
    Args:
        session_id: Session identifier
        action: "list", "switch", "close", or "new"
        tab_index: Tab index for switch/close actions
    """
    session = sessions[session_id]
    
    if action == "list":
        tabs = await session.list_tabs()
        return {"tabs": tabs}
    
    elif action == "switch":
        if tab_index is None:
            raise ValueError("tab_index required for switch")
        await session.switch_tab(tab_index)
        return {"message": f"Switched to tab {tab_index}"}
    
    elif action == "close":
        if tab_index is None:
            raise ValueError("tab_index required for close")
        await session.close_tab(tab_index)
        return {"message": f"Closed tab {tab_index}"}
    
    elif action == "new":
        await session.new_tab()
        return {"message": "Created new tab"}
    
    else:
        raise ValueError(f"Unknown action: {action}")
```

---

## Testing Patterns

### Async Test Template

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

class TestMyFeature:
    """Test suite for my feature.
    
    Fixtures used:
    - mock_config: Pre-configured test configuration
    - cleanup_tasks: Auto-cleanup task store
    - mock_llm: Mocked LangChain LLM
    - mock_browser_context: Mocked browser context
    """

    @pytest.mark.asyncio
    async def test_success_case(self, mock_config, cleanup_tasks):
        """Test successful execution.
        
        Verifies:
        - Function returns expected result
        - Task is created and tracked
        - Cleanup is performed
        """
        # Arrange
        task_id = "test-task-123"
        instruction = "Test instruction"
        
        # Act
        result = await my_function(task_id, instruction, mock_config)
        
        # Assert
        assert result is not None
        assert result["status"] == "completed"
        assert task_id in task_store
        assert task_store[task_id]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config, cleanup_tasks):
        """Test error handling.
        
        Verifies:
        - Exception is raised correctly
        - Error is logged
        - Cleanup occurs even on failure
        """
        with pytest.raises(ValueError, match="Invalid instruction"):
            await my_function("test-id", "", mock_config)

    @pytest.mark.asyncio
    async def test_with_mock_browser(self, mock_browser_context, mock_config):
        """Test with mocked browser.
        
        Verifies:
        - Browser context is used correctly
        - No actual browser is launched
        """
        with patch("server.server.create_browser_context_for_task") as mock_create:
            mock_create.return_value = (mock_browser_context, AsyncMock())
            
            result = await run_browser_task_async("test-id", "task", mock_config)
            
            assert mock_create.called
            assert result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_with_real_browser(self, mock_config):
        """Integration test with real browser.
        
        Requires:
        - Server running on port 8081
        - Playwright browsers installed
        """
        # This test uses real browser - slower but comprehensive
        result = await full_workflow_test(mock_config)
        assert result["status"] == "completed"
```

### Fixture Usage

**Available fixtures from `tests/conftest.py`:**

```python
# Mock environment variables
def test_with_env(mock_env_vars):
    assert os.getenv("OPENAI_API_KEY") == "placeholder"

# Mock configuration
async def test_with_config(mock_config):
    assert mock_config["LOG_LEVEL"] == "DEBUG"

# Mock LLM
async def test_with_llm(mock_llm):
    response = await mock_llm.agenerate(["test"])
    assert response is not None

# Auto-cleanup tasks
async def test_with_cleanup(cleanup_tasks):
    task_store["test-id"] = {"status": "completed"}
    # Automatically cleaned up after test

# Integration server (auto-started)
@pytest.mark.integration
async def test_with_server(integration_server):
    # Server is running on port 8081
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8081/sse")
        assert response.status_code == 200
```

---

## Project Structure Quick Reference

```
mcp-browser-use-server/
├── server/
│   ├── server.py              # ⭐ Core MCP server implementation
│   │   ├── create_mcp_server()      # MCP server factory
│   │   ├── list_tools()             # Tool registration
│   │   ├── call_tool()              # Tool execution
│   │   ├── list_resources()         # Resource listing
│   │   ├── read_resource()          # Resource reading
│   │   └── main()                   # Entry point
│   └── session.py             # ⭐ Session manager
│       ├── BrowserSession           # Session class
│       ├── navigate()               # Navigation
│       ├── click_element()          # Interaction
│       ├── get_state()              # State retrieval
│       └── extract_content()        # LLM extraction
├── src/mcp_browser_use_server/
│   ├── cli.py                 # CLI interface
│   └── server.py              # Re-exports
├── tests/
│   ├── conftest.py            # ⭐ Test fixtures (mock_config, cleanup_tasks, etc.)
│   ├── test_unit.py           # Fast unit tests
│   ├── test_server.py         # Server tests
│   ├── test_browser_tools.py  # Browser tool tests
│   ├── test_integration.py    # Integration tests
│   └── test_e2e.py           # E2E tests
├── pyproject.toml             # ⭐ Dependencies & metadata
├── docker-compose.yaml        # Container orchestration
├── .env                       # ⚠️  Secrets (gitignored)
└── README.md                  # User documentation

⭐ = Most frequently edited
⚠️  = Security sensitive
```

### Common Tasks

### Adding a New Feature
1. Write tests first (TDD approach)
2. Implement feature with type hints
3. Add docstrings
4. Run tests: `./run_tests.sh fast` (uses uv under the hood)
5. Check types: `uv run mypy src/ server/`
6. Format code: `uv run ruff format .`
7. Update documentation if needed

### Adding a New MCP Tool
1. Add to `create_mcp_server()` in `server/server.py`
2. Register in `list_tools()` handler
3. Implement in `call_tool()` handler
4. Add tests in `tests/test_mcp_server.py`
5. Document in README.md API Reference section
6. If the tool manages live browser state, register or reuse session helpers in `server/session.py` and update `server/server.py` handlers accordingly

### Fixing a Bug
1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify test passes
4. Run full test suite: `./run_tests.sh all`
5. Check for regressions

### Adding Dependencies
```bash
# Edit pyproject.toml [project.dependencies]
# Then sync
uv sync

# Or install directly via the project's wrapper
uv pip install package-name
```

## Import Organization

```python
# Standard library
import asyncio
import json
import logging
from typing import Any, Dict, Optional

# Third-party
import pytest
from langchain_openai import ChatOpenAI
from mcp import types
from mcp.server import Server

# Local imports
from server.server import (
    create_mcp_server,
    init_configuration,
    run_browser_task_async,
    task_store,
)
```

---

## Configuration & Environment

### Environment Variable Pattern

```python
from dotenv import load_dotenv
import os
from typing import Dict, Any

def init_configuration() -> Dict[str, Any]:
    """Load and validate configuration from environment.
    
    Returns:
        Configuration dictionary with all required settings
    
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()
    
    # Required settings
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    config = {
        # Required
        "OPENAI_API_KEY": openai_key,
        
        # Optional with defaults
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "BROWSER_HEADLESS": parse_bool_env("BROWSER_HEADLESS", True),
        "PATIENT": parse_bool_env("PATIENT", False),
        
        # Browser settings
        "DEFAULT_WINDOW_WIDTH": int(os.getenv("DEFAULT_WINDOW_WIDTH", "1280")),
        "DEFAULT_WINDOW_HEIGHT": int(os.getenv("DEFAULT_WINDOW_HEIGHT", "1100")),
        
        # Optional paths
        "CHROME_PATH": os.getenv("CHROME_PATH"),
    }
    
    return config

def parse_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variable.
    
    Accepts: true/false, yes/no, 1/0, on/off (case-insensitive)
    """
    value = os.getenv(key)
    if value is None:
        return default
    
    return value.lower() in ("true", "yes", "1", "on")
```

### .env File Management

```bash
# Create .env file
cat > .env << EOF
# Required: OpenAI API key for LLM
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Model selection
LLM_MODEL=gpt-4o-mini

# Optional: Browser mode
BROWSER_HEADLESS=true

# Optional: Logging level
LOG_LEVEL=INFO
EOF

# Verify .env is gitignored
grep -q "^\.env$" .gitignore && echo "✓ .env is gitignored"
```

**⚠️ Security Note:** Never commit `.env` files. If a secret is detected in the working tree, remove it immediately and rotate the secret.

---

## Error Handling & Logging

### Error Handling Pattern

```python
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def safe_operation(
    param: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Template for error-safe operations.
    
    Args:
        param: Operation parameter
        config: Configuration dictionary
        
    Returns:
        Result dict with status and data
        
    Raises:
        ValueError: If param is invalid
    """
    try:
        # Validate inputs
        if not param or not param.strip():
            raise ValueError("param cannot be empty or whitespace")
        
        # Perform operation
        result = await risky_operation(param, config)
        
        # Return success
        return {
            "status": "success",
            "data": result
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "status": "error",
            "error": f"Validation failed: {str(e)}"
        }
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        return {
            "status": "error",
            "error": "Operation timed out"
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "Operation failed"  # Generic message
        }
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

# ✅ CORRECT - Appropriate log levels
logger.debug("Detailed debugging: browser state = %s", state)
logger.info("Task %s started", task_id)
logger.warning("Task %s approaching timeout", task_id)
logger.error("Task %s failed: %s", task_id, error, exc_info=True)

# ✅ CORRECT - Lazy formatting
logger.info("Processing %d tasks", len(tasks))  # Not evaluated if INFO disabled

# ✅ CORRECT - Structured logging
logger.info(
    "Task completed",
    extra={
        "task_id": task_id,
        "duration": duration,
        "status": "completed"
    }
)

# ❌ WRONG - String concatenation
logger.info("Task " + task_id + " started")  # Always evaluated

# ❌ WRONG - Exposing sensitive data
logger.info(f"API key: {api_key}")  # Never log secrets!

# ❌ WRONG - Wrong log level
logger.error("Task started")  # Should be INFO
```

---

## Security Guidelines

### Secrets Management

```python
# ✅ CORRECT - Load from environment
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ CORRECT - Use in code without logging
llm = ChatOpenAI(api_key=api_key)

# ✅ CORRECT - Sanitized error messages
try:
    result = await llm.agenerate(prompts)
except Exception as e:
    logger.error("LLM call failed", exc_info=True)
    return {"error": "LLM call failed"}  # No details exposed

# ❌ WRONG - Hardcoded secret
api_key = "sk-abc123..."  # NEVER DO THIS!

# ❌ WRONG - Secret in logs
logger.info(f"Using key: {api_key}")

# ❌ WRONG - Secret in error message
return {"error": f"API call failed with key {api_key}"}
```

### Input Validation

```python
from pydantic import BaseModel, Field, validator

class TaskInput(BaseModel):
    """Validated task input."""
    instruction: str = Field(..., min_length=1, max_length=10000)
    task_id: Optional[str] = Field(None, regex=r"^[a-zA-Z0-9_-]+$")
    
    @validator("instruction")
    def validate_instruction(cls, v: str) -> str:
        """Validate instruction is not just whitespace."""
        if not v.strip():
            raise ValueError("Instruction cannot be empty")
        
        # Sanitize potentially dangerous content
        dangerous_patterns = ["<script>", "javascript:", "eval("]
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Instruction contains dangerous pattern: {pattern}")
        
        return v.strip()

# Use validated input
async def execute_task(input: TaskInput, config: Dict[str, Any]):
    """Execute task with validated input."""
    # input.instruction is already validated
    result = await run_browser_task_async(
        input.task_id or str(uuid.uuid4()),
        input.instruction,
        config
    )
    return result
```

### Docker Security

```yaml
# docker-compose.yaml security settings
services:
  mcp-browser-use-server:
    security_opt:
      - no-new-privileges:true
    read_only: false  # Need write for /tmp
    tmpfs:
      - /tmp
      - /var/tmp
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if binding to port <1024
```

---

## Docker Commands

```bash
# Development
docker-compose --profile dev up

# Production
docker-compose up -d

# Rebuild
docker-compose up -d --build

# View logs
docker-compose logs -f mcp-browser-use-server

# Stop
docker-compose down
```

## Testing Commands

```bash
# Fast tests (no slow tests)
./run_tests.sh fast

# Unit tests only
./run_tests.sh unit

# With coverage
./run_tests.sh coverage

# Specific test file
uv run pytest tests/test_config.py -v

# Specific test
uv run pytest tests/test_config.py::TestParseBoolean::test_parse_bool_true_variations -v
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
./run_tests.sh fast

# Commit with descriptive message
git commit -m "feat: Add new browser automation feature"

# Push and create PR
git push origin feature/my-feature
```

## Commit Message Convention

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or updates
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Code Review Checklist

Before suggesting or approving code:
- [ ] Type hints on all functions
- [ ] Docstrings on public APIs
- [ ] Tests added/updated
- [ ] No hardcoded credentials or secrets
- [ ] Error handling implemented
- [ ] Async/await used for I/O
- [ ] Follows existing code patterns
- [ ] Documentation updated if needed

## Performance Considerations

- Use async/await for I/O operations
- Clean up browser contexts after tasks
- Implement task cleanup for old completed tasks
- Set resource limits in Docker
- Use connection pooling where applicable
- Avoid blocking operations in async functions

## Security Guidelines

- Never commit `.env` files
- Validate user inputs in browser tasks
- Use Docker secrets for credentials
- Sanitize error messages (no sensitive data)
- Set proper file permissions
- Use HTTPS in production

---

## Performance & Optimization

### Browser Resource Management

```python
# ✅ CORRECT - Proper cleanup
async def execute_task(task_id: str, config: Dict[str, Any]):
    """Execute task with proper resource cleanup."""
    context = None
    browser = None
    try:
        context, browser = await create_browser_context_for_task(task_id, config)
        # Execute task
        result = await agent.run()
        return result
    finally:
        # Always cleanup, even on error
        if context:
            await context.close()
        if browser:
            await browser.close()

# ✅ CORRECT - Use context manager
async def execute_with_context_manager(task_id: str, config: Dict[str, Any]):
    """Execute task using context manager for automatic cleanup."""
    async with create_browser_context(task_id, config) as (context, browser):
        result = await agent.run()
        return result
    # Automatic cleanup
```

### Session Reuse Pattern

```python
# ✅ CORRECT - Reuse sessions for multiple operations
async def multi_step_workflow(urls: List[str]) -> List[Dict[str, Any]]:
    """Process multiple URLs efficiently using session reuse."""
    session_id = str(uuid.uuid4())
    session = await create_session(session_id, config)
    
    results = []
    try:
        for url in urls:
            await session.navigate(url)
            content = await session.extract_content("Extract main content")
            results.append({"url": url, "content": content})
    finally:
        await cleanup_session(session_id)
    
    return results

# ❌ WRONG - Creating new browser for each URL
async def inefficient_workflow(urls: List[str]) -> List[Dict[str, Any]]:
    results = []
    for url in urls:
        # Creates and destroys browser each time - SLOW!
        result = await run_browser_task_async("task", f"Go to {url}", config)
        results.append(result)
    return results
```

### Async Concurrency

```python
import asyncio

# ✅ CORRECT - Parallel execution
async def process_multiple_tasks(tasks: List[str]) -> List[Dict[str, Any]]:
    """Process multiple tasks concurrently."""
    async def process_one(task: str) -> Dict[str, Any]:
        return await run_browser_task_async(str(uuid.uuid4()), task, config)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*[process_one(task) for task in tasks])
    return results

# ✅ CORRECT - With error handling
async def process_with_error_handling(tasks: List[str]) -> List[Dict[str, Any]]:
    """Process tasks concurrently with individual error handling."""
    async def safe_process(task: str) -> Dict[str, Any]:
        try:
            return await run_browser_task_async(str(uuid.uuid4()), task, config)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    results = await asyncio.gather(*[safe_process(task) for task in tasks])
    return results
```

### Memory Management

```python
# Task cleanup pattern
async def cleanup_old_tasks(max_age_hours: int = 24) -> int:
    """Clean up old completed tasks to prevent memory leaks.
    
    Args:
        max_age_hours: Maximum age of tasks to keep
    
    Returns:
        Number of tasks removed
    """
    from datetime import datetime, timedelta
    
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    removed = 0
    
    for task_id, task_data in list(task_store.items()):
        if task_data["status"] in ["completed", "failed"]:
            created_at = datetime.fromisoformat(task_data["created_at"])
            if created_at < cutoff:
                del task_store[task_id]
                removed += 1
    
    logger.info(f"Cleaned up {removed} old tasks")
    return removed

# Schedule periodic cleanup
import asyncio

async def start_cleanup_task():
    """Start background task cleanup."""
    while True:
        await asyncio.sleep(3600)  # Every hour
        await cleanup_old_tasks(max_age_hours=24)
```

### Performance Metrics

**Typical Performance:**
- Task startup: <2 seconds
- Simple navigation: 3-5 seconds
- Complex automation: 10-30 seconds
- Memory per browser: 200-500 MB
- Concurrent tasks: 5-10 (depends on resources)

**Optimization Tips:**
1. Use headless mode in production: `BROWSER_HEADLESS=true`
2. Reuse sessions for related operations
3. Set appropriate timeouts
4. Use Docker resource limits
5. Clean up old tasks regularly

---

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue: `OPENAI_API_KEY not set`

```bash
# Check if .env exists
cat .env

# Verify key is loaded
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"

# Set key
echo "OPENAI_API_KEY=your-key" >> .env
```

#### Issue: Chrome/Chromium not found

```bash
# Install Playwright browsers
uv pip install playwright
uv run playwright install chromium --with-deps

# Or specify custom Chrome path
echo "CHROME_PATH=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" >> .env

# Find Chrome on macOS
mdfind -name "Google Chrome.app" | head -1
ls -la "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

#### Issue: Port already in use

```bash
# Find process using port
lsof -ti:8081

# Kill process
lsof -ti:8081 | xargs kill -9

# Or use different port
uv run server --port 8081
```

#### Issue: Tests hanging or failing

```bash
# Ensure dependencies are installed
uv sync --all-extras

# Install Playwright browsers
uv run playwright install chromium

# Run with verbose output
uv run pytest tests/ -vv

# Run specific test
uv run pytest tests/test_server.py::test_name -vv

# Check server port availability
lsof -ti:8081 | xargs kill -9
```

#### Issue: Memory leak / high memory usage

```python
# Check task store size
print(f"Tasks in store: {len(task_store)}")

# Check sessions
print(f"Active sessions: {len(sessions)}")

# Manual cleanup
await cleanup_old_tasks(max_age_hours=1)

# Close orphaned sessions
for session_id in list(sessions.keys()):
    await cleanup_session(session_id)
```

#### Issue: Task hangs indefinitely

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uv run server --port 8081

# Check task status
curl http://localhost:8081/task/{task_id}

# Use headed mode to see browser
BROWSER_HEADLESS=false uv run server --port 8081
```

### Debug Mode

```bash
# Enable comprehensive logging
LOG_LEVEL=DEBUG uv run server --port 8081

# Or in .env
echo "LOG_LEVEL=DEBUG" >> .env
```

**Debug output includes:**
- MCP protocol messages
- Browser automation steps
- LLM requests/responses
- Task state transitions
- Session lifecycle events

---

## Common Pitfalls to Avoid

### ❌ **Critical Errors**

1. **Using pip instead of uv**
   ```bash
   # ❌ WRONG
   pip install package
   
   # ✅ CORRECT
   uv pip install package
   ```

2. **Missing type hints**
   ```python
   # ❌ WRONG
   def process(data):
       return data
   
   # ✅ CORRECT
   def process(data: Dict[str, Any]) -> Dict[str, Any]:
       return data
   ```

3. **Blocking in async context**
   ```python
   # ❌ WRONG
   async def fetch_data():
       response = requests.get(url)  # Blocking!
       return response.json()
   
   # ✅ CORRECT
   async def fetch_data():
       async with httpx.AsyncClient() as client:
           response = await client.get(url)
           return response.json()
   ```

4. **Not cleaning up resources**
   ```python
   # ❌ WRONG
   async def run_task():
       context, browser = await create_browser_context()
       result = await agent.run()
       return result  # Leak!
   
   # ✅ CORRECT
   async def run_task():
       context, browser = await create_browser_context()
       try:
           result = await agent.run()
           return result
       finally:
           await context.close()
           await browser.close()
   ```

5. **Hardcoding secrets**
   ```python
   # ❌ WRONG
   api_key = "sk-abc123..."
   
   # ✅ CORRECT
   config = init_configuration()
   api_key = config["OPENAI_API_KEY"]
   ```

6. **Skipping tests**
   ```bash
   # ❌ WRONG
   git commit -m "feat: Add feature"  # No tests!
   
   # ✅ CORRECT
   ./run_tests.sh fast
   uv run mypy src/ server/
   git commit -m "feat: Add feature with tests"
   ```

7. **Exposing internal errors**
   ```python
   # ❌ WRONG
   except Exception as e:
       return {"error": str(e)}  # May contain secrets!
   
   # ✅ CORRECT
   except Exception as e:
       logger.error(f"Operation failed: {e}", exc_info=True)
       return {"error": "Operation failed"}  # Generic message
   ```

---

## Quick Reference

### Development Commands

```bash
# Setup & Installation
uv venv --python 3.11              # Create virtual environment
source .venv/bin/activate          # Activate (macOS/Linux)
uv sync                            # Install all dependencies
uv pip install playwright          # Add Playwright
uv run playwright install chromium # Install browser

# Running Server
uv run server --port 8081          # SSE mode
uv run server --stdio              # stdio mode
LOG_LEVEL=DEBUG uv run server      # Debug mode
BROWSER_HEADLESS=false uv run server  # Visible browser

# Testing
./run_tests.sh fast                # Quick tests (pre-commit)
./run_tests.sh all                 # Full test suite
./run_tests.sh unit                # Unit tests only
./run_tests.sh integration         # Integration tests
./run_tests.sh coverage            # With coverage report
uv run pytest tests/ -v            # Verbose output
uv run pytest tests/test_server.py::test_name -vv  # Specific test

# Code Quality
uv run ruff format .               # Format code
uv run ruff check .                # Lint code
uv run mypy src/ server/           # Type check

# Building & Installing
uv build                           # Build package
uv tool install dist/*.whl         # Install as tool
uv tool uninstall mcp-browser-use-server  # Uninstall tool

# Docker
docker-compose up -d               # Start services
docker-compose --profile dev up    # Development mode
docker-compose logs -f             # Follow logs
docker-compose down                # Stop services
docker-compose up -d --build       # Rebuild and start
```

### Common File Paths

```bash
# Core Implementation
server/server.py                   # MCP server
server/session.py                  # Session manager
src/mcp_browser_use_server/cli.py  # CLI interface

# Configuration
.env                               # Secrets (gitignored)
pyproject.toml                     # Dependencies
docker-compose.yaml                # Container config

# Testing
tests/conftest.py                  # Test fixtures
tests/test_unit.py                 # Unit tests
tests/test_integration.py          # Integration tests

# Documentation
README.md                          # User guide
AGENTS.md                          # AI agent context
.github/copilot-instructions.md    # This file
```

### Environment Variables Quick Reference

```bash
# Required
OPENAI_API_KEY=sk-...              # OpenAI API key

# Model Selection
LLM_MODEL=gpt-4o-mini              # LLM model to use

# Browser Settings
BROWSER_HEADLESS=true              # Headless mode
CHROME_PATH=/path/to/chrome        # Custom Chrome path

# Server Settings
LOG_LEVEL=INFO                     # DEBUG, INFO, WARN, ERROR
PATIENT=false                      # Wait for completion

# Performance
DEFAULT_WINDOW_WIDTH=1280
DEFAULT_WINDOW_HEIGHT=1100
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
./run_tests.sh fast
uv run mypy src/ server/

# Commit with conventional message
git commit -m "feat: Add new browser automation feature"
git commit -m "fix: Resolve session cleanup issue"
git commit -m "docs: Update API reference"
git commit -m "test: Add session management tests"

# Push and create PR
git push origin feature/my-feature
```

### Docker Commands

```bash
# Development with hot reload
docker-compose --profile dev up

# Production deployment
docker-compose up -d

# Rebuild containers
docker-compose up -d --build

# View logs
docker-compose logs -f mcp-browser-use-server

# Access container shell
docker-compose exec mcp-browser-use-server /bin/bash

# Stop and remove
docker-compose down

# Stop and remove with volumes
docker-compose down -v
```

### Troubleshooting Quick Fixes

```bash
# Port already in use
lsof -ti:8081 | xargs kill -9

# Missing Playwright browsers
uv run playwright install chromium --with-deps

# Clear task store (if stuck)
# Restart server - tasks stored in memory

# Check logs
tail -f server.log

# Test minimal setup
uv run python -c "from server.server import init_configuration; print(init_configuration())"
```

### Code Review Checklist

Before suggesting or approving code:
- [ ] Type hints on all functions (verify with `uv run mypy`)
- [ ] Docstrings on public APIs (Google style)
- [ ] Tests added/updated (run `./run_tests.sh fast`)
- [ ] No hardcoded credentials or secrets
- [ ] Error handling implemented
- [ ] Async/await used for I/O operations
- [ ] Follows existing code patterns
- [ ] Documentation updated if needed
- [ ] Coverage maintained >95%

### Performance Checklist

- [ ] Use headless mode in production (`BROWSER_HEADLESS=true`)
- [ ] Reuse sessions for related operations
- [ ] Clean up browser contexts after tasks
- [ ] Set appropriate timeouts
- [ ] Use Docker resource limits
- [ ] Implement task cleanup for old completed tasks
- [ ] Avoid blocking operations in async functions

### Security Checklist

- [ ] Never commit `.env` files (verify with `git status`)
- [ ] Validate user inputs in browser tasks (use Pydantic)
- [ ] Use Docker secrets for credentials in production
- [ ] Sanitize error messages (no sensitive data)
- [ ] Set proper file permissions
- [ ] Use HTTPS in production
- [ ] Rotate secrets if exposed
- [ ] Use `secrets` for token generation (not `random`)

---

## When Suggesting Code

**Guidelines for GitHub Copilot:**

1. **Check existing patterns** in the codebase first
   - Look at `server/server.py` for MCP patterns
   - Check `tests/conftest.py` for test fixtures
   - Review `server/session.py` for session management

2. **Use consistent style** with existing code
   - Follow type hints patterns
   - Match docstring format (Google style)
   - Use existing error handling patterns

3. **Include type hints** and docstrings
   - All parameters must have types
   - All returns must have types
   - Docstrings required for public functions

4. **Suggest tests** alongside implementation
   - Unit test for each function
   - Integration test for workflows
   - Use fixtures from `conftest.py`

5. **Consider error cases** and edge cases
   - What if parameter is None?
   - What if network fails?
   - What if timeout occurs?

6. **Think about async** implications
   - Is this I/O bound? → Use async
   - Are there multiple operations? → Consider asyncio.gather
   - Does cleanup happen? → Use try/finally or context manager

7. **Document assumptions** in comments
   - Why this approach over alternatives?
   - What are the limitations?
   - What needs attention in future?

8. **Follow uv conventions** for Python operations
   - Never suggest `pip` commands
   - Always use `uv` wrappers
   - Check `pyproject.toml` for dependencies

### Helpful Commands Cheat Sheet

```bash
# Development
uv sync                          # Install dependencies
uv run server --port 8081        # Run server
LOG_LEVEL=DEBUG uv run server    # Run with debug logging

# Testing
./run_tests.sh fast              # Quick test run
./run_tests.sh coverage          # With coverage report
uv run pytest tests/ -v          # Verbose tests

# Code Quality
uv run ruff check .              # Lint code
uv run ruff format .             # Format code
uv run mypy src/ server/         # Type check

# Building
uv build                         # Build package
uv tool install dist/*.whl       # Install built package

# Docker
docker-compose up -d             # Start services
docker-compose logs -f           # Follow logs
docker-compose down              # Stop services
```

---

## Additional Resources

- **AGENTS.md** - Comprehensive AI agent context and patterns
- **README.md** - User-facing documentation and API reference
- **pyproject.toml** - Project configuration and dependencies
- **tests/conftest.py** - Available test fixtures
- **Docker Compose** - Container orchestration configuration

---

*These instructions help GitHub Copilot provide context-aware suggestions for the mcp-browser-use-server project.*
