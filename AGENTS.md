# AGENTS.md - AI Agent Context for MCP Browser Use Server

> Comprehensive developer guide for AI agents and contributors working with mcp-browser-use-server

## Project Overview

**Project Name**: mcp-browser-use-server  
**Owner**: Dr. Hubertus Becker  
**Repository**: <https://github.com/hubertusgbecker/mcp-browser-use-server>  
**License**: MIT  
**Python Version**: 3.11+  
**Package Manager**: uv (Astral) - **REQUIRED FOR ALL PYTHON OPERATIONS**

---

## Table of Contents

1. [Development Rules](#development-rules)
2. [Purpose & Core Technologies](#purpose)
3. [Quickstart Installation](#quickstart-installation)
4. [Architecture](#architecture)
5. [MCP Tools Reference](#mcp-tools-reference)
6. [Configuration Guide](#configuration-guide)
7. [Best Practices & Prompting](#best-practices--prompting)
8. [Testing & Quality](#testing--quality)
9. [Project Structure](#project-structure)
10. [Common Patterns](#common-patterns)
11. [Troubleshooting](#troubleshooting)

---

## Development Rules

**Critical Guidelines - Follow These Always:**

### 1. Package Manager: uv Only

```bash
# ALWAYS use uv - never use pip directly
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

**Never run:**
- ❌ `pip install ...`
- ❌ `python -m pip install ...`
- ❌ `pip freeze > requirements.txt`

**Always use:**
- ✅ `uv sync` (install dependencies)
- ✅ `uv pip install package-name` (add package)
- ✅ `uv run script.py` (run scripts)
- ✅ `uv build` (build package)

### 2. Type-Safe Coding (Required)

```python
# All functions MUST have type hints
async def my_function(
    param1: str,
    param2: int,
    config: Dict[str, Any]
) -> Optional[MyReturnType]:
    """Brief description.
    
    Args:
        param1: Description
        param2: Description  
        config: Configuration dictionary
        
    Returns:
        Description or None if failure
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

**Requirements:**
- Use Pydantic v2 models for all schemas, configs, and data validation
- All public functions need type hints
- Run `uv run mypy src/ server/` before commits
- Use `Optional[T]`, `Union[T1, T2]`, `List[T]`, `Dict[K, V]`

### 3. Async by Default

```python
# Prefer async for I/O operations
async def process_task(task_id: str) -> Dict[str, Any]:
    async with browser_context() as context:
        result = await agent.run()
        return result

# Mark tests with @pytest.mark.asyncio
@pytest.mark.asyncio
async def test_my_feature():
    result = await my_async_function()
    assert result is not None
```

### 4. Pre-commit & Code Quality

```bash
# ALWAYS run before making PRs
uv run ruff format .
uv run ruff check .
uv run mypy src/ server/
./run_tests.sh fast
```

### 5. Documentation Standards

- Google-style docstrings for all public functions
- Include Args, Returns, Raises sections
- Update README.md and AGENTS.md for significant changes
- Use descriptive variable and function names

### 6. Testing Requirements

- Target: >95% test coverage
- Add tests when adding features
- Run `./run_tests.sh fast` before commits
- Use fixtures from `tests/conftest.py`

### 7. Security & Secrets

- **NEVER commit `.env` files**
- Use environment variables for all secrets
- Validate user inputs in browser tasks
- Log errors without exposing sensitive data

### 8. Model & LLM Usage

- Default to OpenAI models specified in config
- Support model selection via `LLM_MODEL` env var
- Allow users to bring their own API keys
- Don't hardcode model names in examples

---

---

## Purpose

This project provides a **production-ready MCP (Model Context Protocol) server** that enables AI agents to control web browsers through the browser-use library. It bridges AI assistants with browser automation capabilities, supporting both Server-Sent Events (SSE) and stdio transports.

**Key Capabilities:**
- 🌐 Full browser control for AI agents via MCP protocol
- 🔄 Dual transport support (SSE for web, stdio for local)
- ⚡ Async task execution with status tracking
- 🔁 Persistent browser sessions with live inspection
- 📺 VNC streaming for real-time visualization
- 🐳 Production-ready Docker deployment
- 🧪 Comprehensive test suite (95%+ coverage)

---

## Core Technologies

### Primary Stack
- **Python 3.11+**: Core implementation language
- **MCP (Model Context Protocol)**: AI agent communication protocol
- **browser-use**: Browser automation library (wraps Playwright)
- **Playwright**: Browser control engine via CDP
- **LangChain + OpenAI**: LLM integration for agent intelligence
- **FastAPI/Starlette**: Web framework for SSE transport
- **uvicorn**: ASGI server for SSE mode

### Development Tools
- **uv**: Fast Python package manager (**REQUIRED**)
- **pytest**: Testing framework with async support
- **ruff**: Lightning-fast linter and formatter
- **mypy**: Static type checker
- **Docker + docker-compose**: Containerization
- **Context7 MCP Tool**: AI code generation assistance

---

## Quickstart Installation

### Prerequisites

**Required:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Playwright browsers (Chromium)

**Optional:**
- Docker + docker-compose (for containerized deployment)
- mcp-proxy (for stdio mode)

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/hubertusgbecker/mcp-browser-use-server.git
cd mcp-browser-use-server

# Create virtual environment with uv
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install Playwright browsers
uv pip install playwright
uv run playwright install --with-deps --no-shell chromium
```

### 2. Configure Environment

Create `.env` file in project root:

```bash
# Required: OpenAI API key for LLM
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Model selection (default: gpt-4o-mini)
LLM_MODEL=gpt-4o-mini

# Optional: Browser mode (true=headless, false=visible)
BROWSER_HEADLESS=true

# Optional: Logging level
LOG_LEVEL=INFO

# Optional: Custom Chrome/Chromium path
# CHROME_PATH=/path/to/chrome
```

> **Security Note**: Never commit `.env` files to version control!

### 3. Run the Server

**SSE Mode (Web-based):**
```bash
# Run from source
uv run server --port 8081

# Or if installed as package
mcp-browser-use-server run server --port 8081
```

**stdio Mode (Local AI assistants):**
```bash
# Build and install as global tool
uv build
uv tool uninstall mcp-browser-use-server 2>/dev/null || true
uv tool install dist/mcp_browser_use_server-*.whl

# Run with stdio transport
mcp-browser-use-server run server --port 8081 --stdio --proxy-port 9000
```

**Docker Mode:**
```bash
# Using docker-compose (recommended)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services  
docker-compose down
```

### 4. Verify Installation

Test the server is running:

```bash
# For SSE mode
curl http://localhost:8081/sse

# Check server logs
tail -f server.log
```

---

---

## Architecture

### System Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Server Layer                            │
│  ┌─────────────┐                    ┌──────────────┐           │
│  │  SSE Mode   │                    │  stdio Mode  │           │
│  │  (HTTP/SSE) │                    │  (stdin/out) │           │
│  └──────┬──────┘                    └──────┬───────┘           │
│         │                                  │                    │
│         └───────────┬──────────────────────┘                    │
│                     │                                           │
│             ┌───────▼────────┐                                 │
│             │  MCP Protocol  │                                 │
│             │    Handler     │                                 │
│             │  (list/call)   │                                 │
│             └───────┬────────┘                                 │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   Task & Session Management                      │
│  ┌──────────────────────┐      ┌────────────────────────┐     │
│  │   Task Queue (async) │      │  Session Manager       │     │
│  │   - Create tasks     │      │  - Live sessions       │     │
│  │   - Track status     │      │  - Tab management      │     │
│  │   - Manage lifecycle │      │  - Content extraction  │     │
│  └──────────┬───────────┘      └──────────┬─────────────┘     │
└─────────────┼─────────────────────────────┼───────────────────┘
              │                             │
┌─────────────▼─────────────────────────────▼───────────────────┐
│              Browser Automation Layer                          │
│  ┌────────────┐    ┌─────────────┐    ┌──────────────┐      │
│  │  Browser   │───▶│   Context   │───▶│    Agent     │      │
│  │  Instance  │    │   (Page)    │    │  (browser-   │      │
│  │            │    │             │    │    use)      │      │
│  └────────────┘    └─────────────┘    └──────┬───────┘      │
│                                               │               │
│                                        ┌──────▼───────┐      │
│                                        │  Playwright  │      │
│                                        │   Browser    │      │
│                                        │     (CDP)    │      │
│                                        └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. MCP Server (`server/server.py`)
- **Protocol Handler**: Processes MCP tool calls and resource requests
- **Transport Support**: SSE (HTTP) and stdio (local)
- **Tool Registry**: Exposes browser automation tools to AI agents
- **Resource Provider**: Serves task results and browser state

**Key Functions:**
- `create_mcp_server()`: MCP server factory
- `init_configuration()`: Config initialization from env
- `main()`: Entry point for both transports

#### 2. Task Management
- **Async Task Queue** (`task_store`): Thread-safe task tracking
- **Lifecycle States**: pending → running → completed/failed
- **Automatic Cleanup**: Removes old completed tasks
- **Status API**: Real-time task status queries

**Functions:**
- `run_browser_task_async()`: Execute browser tasks
- `cleanup_old_tasks()`: Resource management

#### 3. Session Manager (`server/session.py`)
- **Persistent Sessions**: Long-lived browser sessions across MCP calls
- **Live Inspection**: Get current DOM state, screenshot, tabs
- **Tab Control**: Create, switch, close tabs
- **LLM Extraction**: AI-assisted content extraction from pages

**Key Features:**
- Session lifecycle management
- Element interaction by index
- Multi-tab support
- State serialization for MCP responses

#### 4. Browser Context Management
- **Isolated Contexts**: Each task gets its own browser context
- **Playwright Integration**: Low-level browser control
- **Configuration**: Headless mode, window size, extensions
- **Resource Cleanup**: Automatic context/browser disposal

**Functions:**
- `create_browser_context_for_task()`: Setup isolated browser
- Browser lifecycle hooks

#### 5. Agent System
- **browser-use Agent**: AI-powered browser automation
- **LLM Integration**: OpenAI/LangChain for decision-making
- **Step Execution**: Multi-step tasks with callbacks
- **Error Handling**: Retry logic and failure recovery

---

## CI Notes

**GitHub Actions Integration:**
- All Python workflows use `astral-sh/setup-uv@v5` action
- Install `uv` on runners with caching support
- Run commands via `uv pip` / `uv run` (never system `pip`)
- Test suite: `./run_tests.sh fast` in CI
- Lint/format: `uv run ruff format .` and `uv run ruff check .`

**Example CI snippet:**
```yaml
- name: Set up Python and uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    python-version: "3.11"

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: ./run_tests.sh fast
```

---

## MCP Tools Reference

The server exposes the following MCP tools for AI agents:

### Task Management Tools

#### `run_browser_task`
Execute a browser automation task asynchronously.

```json
{
  "instruction": "Navigate to example.com and click the login button",
  "task_id": "optional-custom-id"
}
```

**Parameters:**
- `instruction` (string, required): Natural language task description
- `task_id` (string, optional): Custom task ID for tracking

**Returns:** Task ID for status tracking

**Example Use Cases:**
- "Navigate to https://news.ycombinator.com and get top 3 posts"
- "Search Google for 'MCP protocol' and summarize first result"
- "Fill out contact form at example.com with test data"

#### `get_task_status`
Check status of a running or completed task.

```json
{
  "task_id": "task-uuid-here"
}
```

**Returns:**
```json
{
  "status": "completed|running|failed",
  "result": "Task output",
  "error": "Error message if failed",
  "created_at": "ISO timestamp"
}
```

#### `cancel_task`
Cancel a running browser task.

```json
{
  "task_id": "task-uuid-here"
}
```

#### `list_all_tasks`
Get all tasks with their current status.

**Returns:** Array of task objects with ID, status, timestamps

### Browser Session Tools

#### `browser_get_state`
Get current browser state for a task or live session.

```json
{
  "task_id": "optional-task-id",
  "session_id": "optional-session-id",
  "screenshot": true
}
```

**Returns:**
- Current URL
- Page title
- Interactive elements list
- Screenshot (base64 PNG) if requested

#### `browser_navigate`
Navigate an existing or new live session to a URL.

```json
{
  "url": "https://example.com",
  "session_id": "optional-session-id"
}
```

**Returns:** Updated session state with new page info

#### `browser_click`
Click an element in a live session by its index.

```json
{
  "session_id": "session-uuid",
  "element_index": 0
}
```

**Note:** Use `browser_get_state` first to see element indices

#### `browser_extract_content`
Use LLM-assisted extraction to get structured content from page.

```json
{
  "session_id": "session-uuid",
  "instruction": "Extract all product names and prices"
}
```

**Returns:** Extracted text or structured result based on instruction

### Session Management

#### `browser_list_sessions`
List all active browser sessions.

**Returns:** Array of session objects with IDs, URLs, creation times

#### `browser_close_session`
Close and cleanup a live browser session.

```json
{
  "session_id": "session-uuid"
}
```

### Tab Management

#### `browser_list_tabs`
List all open tabs in a session.

```json
{
  "session_id": "session-uuid"
}
```

#### `browser_switch_tab`
Switch to a different tab by index.

```json
{
  "session_id": "session-uuid",
  "tab_index": 0
}
```

#### `browser_close_tab`
Close a specific tab.

```json
{
  "session_id": "session-uuid",
  "tab_index": 0
}
```

---

## Configuration Guide

### Environment Variables

```bash
# === Required ===
OPENAI_API_KEY=sk-...             # OpenAI API key for LLM

# === Model Selection ===
LLM_MODEL=gpt-4o-mini             # Model to use (default: gpt-4o-mini)
                                  # Options: gpt-4o, gpt-4-turbo, gpt-3.5-turbo

# === Browser Settings ===
BROWSER_HEADLESS=true             # Run browser in headless mode
CHROME_PATH=/path/to/chrome       # Custom Chrome/Chromium path

# === Server Settings ===
LOG_LEVEL=INFO                    # Logging level: DEBUG, INFO, WARN, ERROR
PATIENT=false                     # Wait for task completion (sync mode)

# === Performance ===
DEFAULT_WINDOW_WIDTH=1280         # Browser window width
DEFAULT_WINDOW_HEIGHT=1100        # Browser window height
```

### Client Configuration

#### SSE Mode (Web-based clients)

```json
{
  "mcpServers": {
    "mcp-browser-use-server": {
      "url": "http://localhost:8081/sse"
    }
  }
}
```

#### stdio Mode (Local AI assistants)

```json
{
  "mcpServers": {
    "mcp-browser-use-server": {
      "command": "mcp-browser-use-server",
      "args": [
        "run",
        "server",
        "--port",
        "8081",
        "--stdio",
        "--proxy-port",
        "9000"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### Configuration Paths by Client

| Client           | Configuration File Path                                           |
|------------------|-------------------------------------------------------------------|
| **Cursor**       | `./.cursor/mcp.json`                                              |
| **Windsurf**     | `~/.codeium/windsurf/mcp_config.json`                             |
| **Claude (Mac)** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude (Win)** | `%APPDATA%\Claude\claude_desktop_config.json`                     |

---

## Best Practices & Prompting

### Effective Task Instructions

#### ✅ **Be Specific (Recommended)**

```python
task = """
1. Navigate to https://quotes.toscrape.com/
2. Extract the first 3 quotes with their authors
3. Save results to quotes.json
4. Return the formatted JSON
"""
```

**Why this works:**
- Clear step-by-step instructions
- Specifies exact actions and data
- Defines output format

#### ❌ **Too Vague**

```python
task = "Get some quotes from the internet"
```

**Problems:**
- Unclear source
- No format specified
- Ambiguous quantity

### Prompting Patterns

#### 1. **Navigate and Extract**

```python
instruction = """
1. Go to https://news.ycombinator.com
2. Find the top-ranked post title and URL
3. Return as JSON: {"title": "...", "url": "..."}
"""
```

#### 2. **Form Interaction**

```python
instruction = """
1. Navigate to example.com/contact
2. Fill in name: "Test User"
3. Fill in email: "test@example.com"
4. Fill in message: "Testing MCP server"
5. Click Submit button
6. Verify success message appears
"""
```

#### 3. **Search and Aggregate**

```python
instruction = """
1. Search Google for "Model Context Protocol"
2. Get titles and URLs of first 5 results
3. Return as numbered list
"""
```

#### 4. **Error Recovery**

```python
instruction = """
1. Try to navigate to target-site.com
2. If blocked by captcha or error:
   - Use DuckDuckGo to search for site content
   - Extract relevant information from search results
3. Return extracted data or error explanation
"""
```

### Session vs Task Mode

**Use Tasks when:**
- One-time automation needed
- No follow-up interactions required
- Want automatic cleanup

**Use Sessions when:**
- Multi-step interaction across MCP calls
- Need to inspect intermediate state
- Want to reuse browser context
- Building conversational workflows

**Example Session Workflow:**
```python
# Step 1: Create session and navigate
call_tool("browser_navigate", {"url": "https://example.com"})

# Step 2: Get current state
state = call_tool("browser_get_state", {"session_id": "...", "screenshot": true})

# Step 3: Click element based on state
call_tool("browser_click", {"session_id": "...", "element_index": 2})

# Step 4: Extract content
result = call_tool("browser_extract_content", {
    "session_id": "...",
    "instruction": "Get product prices"
})

# Step 5: Cleanup
call_tool("browser_close_session", {"session_id": "..."})
```

---

## Testing & Quality

### Running Tests

```bash
# Fast test suite (skip slow tests) - used in CI
./run_tests.sh fast

# All tests
./run_tests.sh all

# Unit tests only
./run_tests.sh unit

# Integration tests
./run_tests.sh integration

# With coverage report
./run_tests.sh coverage

# Specific test file
uv run pytest tests/test_server.py -v

# Specific test function
uv run pytest tests/test_server.py::TestConfig::test_parse_bool -v
```

### Test Categories

**Unit Tests** (`test_unit.py`)
- Fast, isolated component tests
- Mock external dependencies
- Test individual functions

**Integration Tests** (`test_integration.py`)
- Test component interactions
- Real browser automation
- Requires server running

**E2E Tests** (`test_e2e.py`)
- Full system tests
- Real AI agent workflows
- Requires `RUN_E2E_TESTS=true`

**Performance Tests** (`test_performance.py`)
- Load testing
- Latency measurements
- Resource usage

### Quality Checks

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/ server/

# Run all checks (recommended before PR)
uv run ruff format . && \
uv run ruff check . && \
uv run mypy src/ server/ && \
./run_tests.sh fast
```

### Writing Tests

**Test Template:**
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_my_feature(mock_config, cleanup_tasks):
    """Test description following Google style.
    
    Tests should:
    - Be descriptive
    - Test one thing
    - Use fixtures from conftest.py
    - Clean up resources
    """
    # Arrange
    task_id = "test-task-id"
    
    # Act
    result = await my_async_function(task_id, mock_config)
    
    # Assert
    assert result is not None
    assert result["status"] == "completed"
```

**Using Fixtures:**
- `mock_config`: Pre-configured test config
- `mock_llm`: Mock LangChain LLM
- `cleanup_tasks`: Auto-cleanup task store
- `mock_browser_context`: Mock browser context

---

---

## Project Structure

```
mcp-browser-use-server/
├── src/
│   └── mcp_browser_use_server/
│       ├── __init__.py          # Package initialization
│       ├── cli.py               # Command-line interface
│       └── server.py            # Re-exports from server/
├── server/
│   ├── __init__.py
│   ├── __main__.py
│   ├── server.py                # Core server implementation
│   │   ├── parse_bool_env()            # ENV parsing
│   │   ├── init_configuration()        # Config initialization
│   │   ├── create_browser_context_for_task()  # Browser setup
│   │   ├── run_browser_task_async()    # Task execution
│   │   ├── cleanup_old_tasks()         # Resource cleanup
│   │   ├── create_mcp_server()         # MCP server factory
│   │   └── main()                      # Entry point
│   └── session.py               # Session manager implementation
│       ├── BrowserSession class        # Persistent session handling
│       ├── navigate()                  # Session navigation
│       ├── click_element()             # Element interaction
│       ├── get_state()                 # State serialization
│       └── extract_content()           # LLM extraction
├── tests/
│   ├── conftest.py              # Pytest fixtures and configuration
│   ├── test_unit.py             # Fast unit tests
│   ├── test_server.py           # Server functionality tests
│   ├── test_browser_tools.py   # Browser tool tests
│   ├── test_integration.py     # Integration tests
│   ├── test_e2e.py             # End-to-end tests
│   └── test_cli.py             # CLI interface tests
├── docker-compose.yaml          # Docker orchestration
├── Dockerfile                   # Container image
├── pyproject.toml              # Project metadata and dependencies
├── pytest.ini                  # Test configuration
├── run_tests.sh                # Test runner script
├── README.md                   # User documentation
├── AGENTS.md                   # This file - AI agent context
└── .github/
    ├── copilot-instructions.md  # GitHub Copilot context
    └── workflows/
        └── ci-lint-test.yml     # CI pipeline

```

### Key Files Deep Dive

#### `server/server.py` (Core Implementation)

**Main Components:**
- `task_store: Dict[str, Any]` - In-memory task tracking
- `sessions: Dict[str, BrowserSession]` - Active session registry
- `create_mcp_server()` - Factory for MCP Server instance
- Tool handlers: `list_tools()`, `call_tool()`
- Resource handlers: `list_resources()`, `read_resource()`

**Tool Handlers:**
Each MCP tool has a handler block in `call_tool()`:
```python
if tool_name == "run_browser_task":
    # Extract params, validate, execute task
    # Return result via MCP types.CallToolResult
```

#### `server/session.py` (Session Manager)

**BrowserSession Class:**
```python
class BrowserSession:
    """Persistent browser session for multi-step interactions."""
    
    def __init__(self, session_id: str, config: Dict[str, Any])
    async def navigate(self, url: str) -> Dict[str, Any]
    async def click_element(self, index: int) -> Dict[str, Any]
    async def get_state(self, screenshot: bool = False) -> Dict[str, Any]
    async def extract_content(self, instruction: str) -> str
    async def close(self) -> None
```

**Key Patterns:**
- Async context managers for browser lifecycle
- State serialization for MCP responses
- Error handling with descriptive messages
- Resource cleanup on session close

#### `tests/conftest.py` (Test Fixtures)

**Available Fixtures:**
- `mock_env_vars`: Environment variable mocking
- `mock_config`: Pre-configured test config
- `mock_llm`: Mock LangChain LLM
- `mock_browser_context`: Mock browser context
- `cleanup_tasks`: Auto-cleanup task store
- `integration_server`: Auto-start server for integration tests

---

## Common Patterns

### Adding a New MCP Tool

**Step 1: Define Tool in `list_tools()`**

```python
mcp.types.Tool(
    name="my_new_tool",
    description="Clear description of what the tool does",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            },
            "param2": {
                "type": "integer",
                "description": "Another parameter"
            }
        },
        "required": ["param1"]
    }
)
```

**Step 2: Implement Handler in `call_tool()`**

```python
if tool_name == "my_new_tool":
    # Extract parameters with validation
    param1 = arguments.get("param1")
    if not param1:
        return [mcp.types.TextContent(
            type="text",
            text="Error: param1 is required"
        )]
    
    param2 = arguments.get("param2", 10)  # Optional with default
    
    try:
        # Implement tool logic
        result = await my_tool_implementation(param1, param2)
        
        return [mcp.types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        logger.error(f"Error in my_new_tool: {e}")
        return [mcp.types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]
```

**Step 3: Add Tests**

```python
@pytest.mark.asyncio
async def test_my_new_tool(mock_config, cleanup_tasks):
    """Test my_new_tool functionality."""
    # Test implementation
    pass
```

**Step 4: Document in README.md**

Add tool to API Reference section with usage examples.

### Working with Browser Sessions

**Creating a Session:**
```python
from server.session import BrowserSession

async def create_session(session_id: str, config: Dict[str, Any]):
    session = BrowserSession(session_id, config)
    sessions[session_id] = session
    await session.initialize()  # Setup browser
    return session
```

**Using a Session:**
```python
async def use_session(session_id: str):
    if session_id not in sessions:
        raise ValueError(f"Session {session_id} not found")
    
    session = sessions[session_id]
    
    # Navigate
    await session.navigate("https://example.com")
    
    # Get state
    state = await session.get_state(screenshot=True)
    
    # Interact
    await session.click_element(0)
    
    # Extract
    content = await session.extract_content("Get all links")
    
    return content
```

**Cleanup:**
```python
async def cleanup_session(session_id: str):
    if session_id in sessions:
        session = sessions[session_id]
        await session.close()
        del sessions[session_id]
```

### Error Handling Pattern

```python
async def safe_operation(param: str) -> Dict[str, Any]:
    """Template for error-safe operations.
    
    Args:
        param: Operation parameter
        
    Returns:
        Result dict with status and data
        
    Raises:
        ValueError: If param is invalid
    """
    try:
        # Validate inputs
        if not param:
            raise ValueError("param cannot be empty")
        
        # Perform operation
        result = await risky_operation(param)
        
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
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Operation failed: {str(e)}"
        }
```

### Configuration Loading

```python
from dotenv import load_dotenv
import os

def init_configuration() -> Dict[str, Any]:
    """Load configuration from environment."""
    load_dotenv()
    
    config = {
        # Required
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        
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
    
    # Validate required config
    if not config["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return config
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: `OPENAI_API_KEY not set`

**Symptom:** Server fails to start with KeyError or ValueError

**Solution:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Or export in shell
export OPENAI_API_KEY=your-key-here

# Verify it's loaded
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

#### Issue: `Chrome/Chromium not found`

**Symptom:** Playwright fails to launch browser

**Solution:**
```bash
# Install Playwright browsers
uv pip install playwright
uv run playwright install chromium --with-deps

# Or specify custom Chrome path in .env
echo "CHROME_PATH=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" >> .env
```

**Find Chrome path on macOS:**
```bash
# Using mdfind
mdfind -name "Google Chrome.app" | head -1

# Direct check
ls -la "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

#### Issue: Port already in use

**Symptom:** `Address already in use` error

**Solution:**
```bash
# Find process using port 8081
lsof -ti:8081

# Kill the process
lsof -ti:8081 | xargs kill -9

# Or use different port
uv run server --port 8082
```

#### Issue: Tests failing

**Symptom:** Test suite fails or hangs

**Solution:**
```bash
# Ensure all dependencies installed
uv sync --all-extras

# Install Playwright browsers for tests
uv run playwright install chromium

# Run with verbose output
uv run pytest tests/ -vv

# Run specific failing test
uv run pytest tests/test_server.py::test_name -vv

# Check if server port is available
lsof -ti:8081 | xargs kill -9
```

#### Issue: Task hangs indefinitely

**Symptom:** Browser task never completes

**Diagnostics:**
```bash
# Enable debug logging
LOG_LEVEL=DEBUG uv run server --port 8081

# Check task status
curl http://localhost:8081/task/{task_id}
```

**Common Causes:**
1. **Infinite loop in agent** - Check agent logic
2. **Page never loads** - Network timeout or blocked by firewall
3. **Element not found** - Selector or wait condition issue

**Solutions:**
- Add timeout to tasks
- Use `PATIENT=false` for async mode
- Check browser console in headed mode (`BROWSER_HEADLESS=false`)

#### Issue: Memory leak / high memory usage

**Symptom:** Server memory grows over time

**Solution:**
```bash
# Implement task cleanup
# Tasks auto-cleanup after completion
# Check cleanup_old_tasks() is running

# Verify cleanup in logs
tail -f server.log | grep cleanup

# Manual cleanup
curl -X POST http://localhost:8081/cleanup
```

**Prevention:**
- Close browser sessions after use
- Set task cleanup interval
- Use Docker with memory limits

#### Issue: Session not found

**Symptom:** `Session {id} not found` error

**Cause:** Session expired or never created

**Solution:**
```bash
# List active sessions
call_tool("browser_list_sessions", {})

# Create new session
call_tool("browser_navigate", {"url": "https://example.com"})
# This auto-creates session if session_id not provided

# Always store session_id from responses
```

#### Issue: Docker container fails to start

**Symptom:** Container exits immediately

**Diagnostics:**
```bash
# Check logs
docker-compose logs -f mcp-browser-use-server

# Check environment
docker-compose config

# Verify .env file
cat .env
```

**Common Issues:**
- Missing `OPENAI_API_KEY` in docker-compose.yaml
- Port 8081 already bound on host
- Insufficient Docker resources

**Solution:**
```bash
# Add API key to docker-compose.yaml environment section
# Or use .env file
docker-compose --env-file .env up -d

# Change port mapping
# In docker-compose.yaml: "8082:8081"
```

### Debug Mode

Enable comprehensive logging for troubleshooting:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or via environment
LOG_LEVEL=DEBUG uv run server --port 8081
```

**Debug Output Includes:**
- MCP protocol messages
- Browser automation steps
- LLM requests/responses
- Task state transitions
- Session lifecycle events

### Getting Help

1. **Check Logs:**
   ```bash
   tail -f server.log
   # Or with Docker
   docker-compose logs -f
   ```

2. **Search Issues:**
   - [GitHub Issues](https://github.com/hubertusgbecker/mcp-browser-use-server/issues)
   - Check closed issues for solved problems

3. **Create Issue:**
   - Include logs (with secrets redacted)
   - Describe steps to reproduce
   - Mention OS, Python version, uv version

4. **Test Minimal Example:**
   ```python
   # Create minimal reproduction
   import asyncio
   from server.server import init_configuration, create_browser_context_for_task
   
   async def test():
       config = init_configuration()
       context, browser = await create_browser_context_for_task("test", config)
       print("Success!")
       await context.close()
       await browser.close()
   
   asyncio.run(test())
   ```

---

## Performance Optimization

### Best Practices

1. **Use Headless Mode in Production:**
   ```bash
   BROWSER_HEADLESS=true
   ```

2. **Limit Concurrent Tasks:**
   - Configure task queue size
   - Implement rate limiting
   - Use Docker resource limits

3. **Session Reuse:**
   ```python
   # Reuse sessions for multiple operations
   session_id = create_session()
   
   # Multiple operations on same session
   await navigate(session_id, url1)
   await extract_content(session_id, "data1")
   
   await navigate(session_id, url2)
   await extract_content(session_id, "data2")
   
   # Close when done
   await close_session(session_id)
   ```

4. **Optimize LLM Calls:**
   ```bash
   # Use faster model for extraction
   LLM_MODEL=gpt-3.5-turbo  # Faster, cheaper
   
   # Or configure per-task
   # Use gpt-4o for complex tasks only
   ```

5. **Docker Resource Limits:**
   ```yaml
   # In docker-compose.yaml
   services:
     mcp-browser-use-server:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 2G
           reservations:
             cpus: '1.0'
             memory: 1G
   ```

### Performance Metrics

**Typical Performance:**
- Task startup: <2 seconds
- Simple navigation: 3-5 seconds
- Complex automation: 10-30 seconds
- Memory per browser: 200-500 MB
- Concurrent tasks: 5-10 (depends on resources)

---

## Context7 Integration

**Always use Context7 MCP tool for:**
- Code generation and boilerplate
- API documentation lookup
- Configuration examples
- Troubleshooting guides

**Usage Pattern:**
```bash
# Automatically resolve library docs
# When generating code, Context7 fetches latest docs
# No need to explicitly ask - it's integrated
```

**Benefits:**
- Up-to-date API references
- Accurate code examples
- Library-specific best practices

---

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.