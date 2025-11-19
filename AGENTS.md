# AGENT.md - AI Agent Context for MCP Browser Use Server

## Project Overview

**Project Name**: mcp-browser-use-server
**Owner**: Dr. Hubertus Becker
**Repository**: <https://github.com/hubertusgbecker/mcp-browser-use-server>
**License**: MIT
**Python Version**: 3.11+
**Package Manager**: uv (Astral)

## Purpose

This project provides a production-ready MCP (Model Context Protocol) server that enables AI agents to control web browsers through the browser-use library. It bridges AI assistants with browser automation capabilities, supporting both Server-Sent Events (SSE) and stdio transports.

## Core Technologies

### Primary Stack
- **Python 3.11+**: Core implementation language
- **MCP (Model Context Protocol)**: AI agent communication protocol
- **browser-use**: Browser automation library
- **Playwright**: Browser control engine
- **LangChain + OpenAI**: LLM integration
- **FastAPI/Starlette**: Web framework for SSE
- **uvicorn**: ASGI server

### Development Tools
- **uv**: Fast Python package manager (use for ALL Python operations)
- **pytest**: Testing framework
- **ruff**: Linting and formatting
- **black**: Code formatting
- **mypy**: Type checking
- **Docker**: Containerization
- **Context7 MCP Tool**: AI code generation and documentation assistance

## Architecture

### Component Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server Layer                        │
│  ┌─────────────┐              ┌──────────────┐             │
│  │  SSE Mode   │              │  stdio Mode  │             │
│  └──────┬──────┘              └──────┬───────┘             │
│         │                            │                      │
│         └────────────┬───────────────┘                      │
│                      │                                      │
│              ┌───────▼────────┐                            │
│              │  MCP Protocol  │                            │
│              │    Handler     │                            │
│              └───────┬────────┘                            │
└──────────────────────┼──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Task Management                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Task Queue (async)                              │      │
│  │  - Create tasks                                  │      │
│  │  - Track status                                  │      │
│  │  - Manage lifecycle                              │      │
│  └──────────────────┬───────────────────────────────┘      │
└─────────────────────┼────────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│              Browser Automation Layer                      │
│  ┌────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │  Browser   │───▶│   Context   │───▶│    Agent     │  │
│  │  Manager   │    │   Manager   │    │  (browser-   │  │
│  │            │    │             │    │    use)      │  │
│  └────────────┘    └─────────────┘    └──────┬───────┘  │
│                                               │           │
│                                        ┌──────▼───────┐  │
│                                        │  Playwright  │  │
│                                        │   Browser    │  │
│                                        └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```text

### Key Components
\
## CI Notes

- **CI: `uv` required**: All Python-related CI workflows rely on Astral's `uv` wrapper to manage virtual environments and run tooling. Workflows use `astral-sh/setup-uv@v5` to ensure `uv` is available on runners. When adding new GitHub Actions that run Python commands, use `astral-sh/setup-uv@v5` and prefer `uv pip` / `uv run` instead of system `pip`/`python` calls.


1. **MCP Server** (`server/server.py`)
   - Handles MCP protocol communication
   - Supports SSE and stdio transports
   - Exposes tools and resources

2. **Task Management**
   - Async task queue (`task_store`)
   - Task lifecycle: pending → running → completed/failed
   - Automatic cleanup of old tasks

3. **Browser Context Management**
   - Creates isolated browser contexts per task
   - Configures Playwright browser instances
   - Handles browser lifecycle

4. **Agent System**
   - Uses browser-use Agent for AI-powered automation
   - Integrates with OpenAI LLM
   - Step-by-step task execution with callbacks

5. **Session Manager**
    - `server/session.py` manages persistent BrowserSession instances used by several MCP tools
    - Sessions support navigation, clicks, tab management and LLM-assisted content extraction

## Project Structure

```
mcp-browser-use-server/
├── src/
│   └── mcp_browser_use_server/
│       ├── **init**.py          # Package initialization
│       ├── cli.py               # Command-line interface
│       └── server.py            # Re-exports from server/
├── server/
│   ├── **init**.py
│   ├── **main**.py
│   └── server.py                # Core server implementation
│       ├── parse_bool_env()            # ENV parsing
│       ├── init_configuration()        # Config initialization
│       ├── create_browser_context_for_task()  # Browser setup
│       ├── run_browser_task_async()    # Task execution
│       ├── cleanup_old_tasks()         # Resource cleanup
│       ├── create_mcp_server()         # MCP server factory
│       └── main()                      # Entry point
├── tests/
│   ├── conftest.py              # Pytest fixtures and configuration
│   ├── test_config.py           # Configuration tests
│   ├── test_browser_tasks.py   # Browser task execution tests
│   ├── test_mcp_server.py      # MCP server tests
│   ├── test_integration.py     # Integration tests
│   ├── test_e2e.py             # End-to-end tests
│   └── test_performance.py     # Performance and load tests
├── docker-compose.yaml          # Docker orchestration
├── Dockerfile                   # Container image
├── pyproject.toml              # Project metadata and dependencies
├── pytest.ini                  # Test configuration
├── run_tests.sh                # Test runner script
├── README.md                   # User documentation
├── AGENTS.md                   # This file - AI context
└── .github/
    └── copilot-instructions.md  # GitHub Copilot context
```

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.