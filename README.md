# mcp-browser-use-server

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-hubertusgbecker-blue)](https://github.com/hubertusgbecker/mcp-browser-use-server)
[![PyPI version](https://badge.fury.io/py/mcp-browser-use-server.svg)](https://badge.fury.io/py/mcp-browser-use-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Security: KICS+Trivy](https://img.shields.io/badge/security-KICS%2BTrivy-green)](./.github/workflows/kics-security-scan.yml)

**An MCP server that enables AI agents to control web browsers using [browser-use](https://github.com/browser-use/browser-use).**

</div>

## Overview

The MCP Browser Use Server provides a robust, production-ready interface for AI agents to interact with web browsers through the Model Context Protocol (MCP). It supports both Server-Sent Events (SSE) and stdio transports, enabling seamless integration with various AI assistants and tools.

### Key Features

- 🌐 **Browser Automation**: Full browser control through AI agents
- 🔄 **Dual Transport**: SSE and stdio protocol support
- 📺 **VNC Streaming**: Real-time browser visualization
- ⚡ **Async Tasks**: Non-blocking browser operations
- 🐳 **Docker Support**: Containerized deployment with docker-compose
 - 🔁 **Persistent Sessions**: Long-lived browser sessions with live inspection, tab control and content extraction (see `server/session.py`)
- 🧪 **Comprehensive Tests**: Industry-ready test suite with 95%+ coverage
- 🔒 **Production Ready**: Robust error handling and resource management

## Quick Start

### Prerequisites

Ensure you have the following installed:

- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **[Playwright](https://playwright.dev/)** - Browser automation library
- **[mcp-proxy](https://github.com/sparfenyuk/mcp-proxy)** - Required for stdio mode

```bash
# uv (Astral) is required for Python workflows.
# For CI (GitHub Actions) prefer the `astral-sh/setup-uv@v5` action which
# installs and configures `uv` on the runner with caching support. Example:

```yaml
- name: Set up Python and uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    python-version: "3.11"
```

# Install mcp-proxy globally
uv tool install mcp-proxy

# Update shell to recognize new tools
uv tool update-shell
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Required: OpenAI API key for LLM
# Recommended: set `LLM_MODEL` environment variable to your preferred model
# Default: `gpt-5-mini` (you can override with models like `gpt-4-turbo` or `gpt-4o` if available)
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-5-mini

# Optional: Custom Chrome/Chromium path
CHROME_PATH=/path/to/chrome

# Optional: Patient mode (wait for task completion)
PATIENT=false

# Optional: Logging level
LOG_LEVEL=INFO

# Control headless browser mode used by the server (true = headless, false = visible)
# Default: true
BROWSER_HEADLESS=true
```

> **Security Note**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/hubertusgbecker/mcp-browser-use-server.git
cd mcp-browser-use-server

# Install dependencies with uv
uv sync

# Install Playwright browsers
uv pip install playwright
uv run playwright install --with-deps --no-shell chromium
```

### As a Package (Production)

```bash
# Install from PyPI (when published)
uv pip install mcp-browser-use-server

# Or install from built wheel
uv build
uv tool install dist/mcp_browser_use_server-*.whl
```

## Usage

### SSE Mode (Recommended for Web Interfaces)

Server-Sent Events mode is ideal for web-based integrations and remote access.

```bash
# Run from source
uv run server --port 8081

# Or if installed as tool
mcp-browser-use-server run server --port 8081
```

The server will be available at `http://localhost:8081/sse`

### stdio Mode (For Local AI Assistants)

Standard I/O mode is perfect for local AI assistants like Claude Desktop, Cursor, or Windsurf.

```bash
# Build and install as global tool
uv build
uv tool uninstall mcp-browser-use-server 2>/dev/null || true
uv tool install dist/mcp_browser_use_server-*.whl

# Run with stdio transport
mcp-browser-use-server run server --port 8081 --stdio --proxy-port 9000
```

Note: this package also exposes a CLI alias `mcp-browser-use-cli` that points to the same entrypoint. You can use either `mcp-browser-use-server` or `mcp-browser-use-cli` after installation.


### Docker Deployment

Docker provides an isolated, reproducible environment with VNC support for visualization.

```bash
# Using docker-compose (recommended)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Or use Docker directly
docker build -t mcp-browser-use-server .
docker run --rm -p 8081:8081 -p 5900:5900 \
  -e OPENAI_API_KEY=your-key \
  mcp-browser-use-server
```

#### Docker Compose Profiles

The `docker-compose.yaml` supports multiple profiles:

```bash
# Development mode with hot reload
docker-compose --profile dev up

# With stdio proxy
docker-compose --profile stdio up

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up
```

## Client Configuration

### SSE Mode Configuration

For web-based clients and remote connections:

```json
{
  "mcpServers": {
    "mcp-browser-use-server": {
      "url": "http://localhost:8081/sse"
    }
  }
}
```

### stdio Mode Configuration

For local AI assistants:

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

### Configuration Paths by Client

| Client           | Configuration File Path                                           |
|------------------|-------------------------------------------------------------------|
| **Cursor**       | `./.cursor/mcp.json`                                              |
| **Windsurf**     | `~/.codeium/windsurf/mcp_config.json`                             |
| **Claude (Mac)** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude (Win)** | `%APPDATA%\Claude\claude_desktop_config.json`                     |

## Testing

The project includes a comprehensive test suite for ensuring reliability and correctness.

### Running Tests

```bash
# Run all tests
./run_tests.sh all

# Run only unit tests
./run_tests.sh unit

# Run integration tests
./run_tests.sh integration

# Run with coverage report
./run_tests.sh coverage

# Run fast tests only (skip slow tests)
./run_tests.sh fast

# Or use pytest directly
uv run pytest tests/
uv run pytest tests/ -v --cov=src --cov=server

CI
--
The repository includes a GitHub Actions workflow that runs ruff and the fast test
suite on pushes and pull requests to `main`. The workflow file is at
`.github/workflows/ci-lint-test.yml`.

Run the same checks locally before pushing:

```bash
# Format and lint with ruff (via the project's uv wrapper)
uv run ruff format .
uv run ruff check .

# Run the fast test suite (the same target used by CI)
./run_tests.sh fast
```
```

### Test Categories

- **Unit Tests**: Fast, isolated tests for individual functions
- **Integration Tests**: Test component interactions
- **E2E Tests**: Full system tests (require `RUN_E2E_TESTS=true`)
- **Performance Tests**: Load and performance validation

### CI/CD Integration

Tests are designed to run in CI/CD pipelines. See `.github/workflows/` for GitHub Actions examples.

## VNC Browser Visualization

Watch browser automation in real-time using VNC:

```bash
# Start server with VNC (Docker)
docker-compose up -d

# Connect using noVNC (browser-based)
git clone https://github.com/novnc/noVNC
cd noVNC
./utils/novnc_proxy --vnc localhost:5900

# Or use any VNC client
# Default password: browser-use
```

Then open <http://localhost:6080/vnc.html> in your browser to watch the automation.

## Development

### Local Development Workflow

```bash
# Install development dependencies
uv sync --all-extras

# Run linters and formatters
uv run ruff check .
uv run ruff format .
uv run black .
uv run isort .

# Type checking
uv run mypy src/ server/

# Run tests during development
uv run pytest tests/ -v

# Build package
uv build
```

### Project Structure

```
mcp-browser-use-server/
├── src/
│   └── mcp_browser_use_server/  # Package source
│       ├── __init__.py
│       ├── cli.py               # CLI interface
│       └── server.py            # Server re-exports
├── server/
│   ├── __init__.py
│   ├── __main__.py
│   └── server.py                # Core server implementation
├── tests/                       # Comprehensive test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_config.py           # Configuration tests
│   ├── test_browser_tasks.py   # Browser task tests
│   ├── test_mcp_server.py      # MCP server tests
│   ├── test_integration.py     # Integration tests
│   ├── test_e2e.py             # End-to-end tests
│   └── test_performance.py     # Performance tests
├── docker-compose.yaml          # Docker orchestration
├── Dockerfile                   # Container definition
├── pyproject.toml              # Project configuration
├── pytest.ini                  # Test configuration
└── README.md                   # This file
```

### Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`./run_tests.sh all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality Standards

- **Test Coverage**: Maintain >95% coverage
- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public APIs
- **Linting**: Code must pass ruff and mypy checks
- **Formatting**: Use black and isort for consistent formatting

## Example Usage

### Basic Browser Task

Ask your AI assistant:

```text
Navigate to https://news.ycombinator.com and return the top ranked article's title and URL
```

### Advanced Examples

```text
# Search and extract
Go to Google, search for "MCP protocol", and summarize the first 3 results

# Form interaction
Navigate to the contact form at example.com/contact, fill in the fields with test data, and submit

# Data extraction
Visit GitHub's trending page and list the top 5 trending repositories today

# Screenshot capture
Navigate to example.com and take a screenshot of the homepage
```

## API Reference

### Available MCP Tools

The server exposes the following tools via MCP:

- **`run_browser_task`**: Execute a browser automation task
  - Parameters: `instruction` (string), `task_id` (optional)
  - Returns: Task ID for async tracking

- **`get_task_status`**: Check the status of a running task
  - Parameters: `task_id` (string)
  - Returns: Task status, progress, and results

- **`cancel_task`**: Cancel a running browser task
  - Parameters: `task_id` (string)
  - Returns: Cancellation confirmation

- **`list_all_tasks`**: List all tasks
  - Returns: Array of all tasks with their statuses

New session and browser tools

- **`browser_get_state`**: Get current browser state for a task or live session
  - Parameters: `task_id` (optional) or `session_id` (optional), `screenshot` (bool)
  - Returns: JSON state summary; `screenshot=true` returns base64 PNG

- **`browser_navigate`**: Navigate an existing or new live session to a URL
  - Parameters: `url` (string), optional `session_id`
  - Returns: Confirmation and updated session state

- **`browser_click`**: Click an element in a live session (by index)
  - Parameters: `session_id`, `element_index`
  - Returns: Action result

- **`browser_extract_content`**: Use session extraction (LLM-assisted) to extract or summarize content
  - Parameters: `session_id`, `instruction` (string)
  - Returns: Extracted text or structured result

- **Session management**: `browser_list_sessions`, `browser_close_session` to list and close long-lived sessions

- **Tabs API**: `browser_list_tabs`, `browser_switch_tab`, `browser_close_tab` for tab control

These tools are implemented in `server/server.py` and rely on the session manager `server/session.py`.

### Available MCP Resources

- **Task Results**: Access completed task results via `task://{task_id}`
- **Task History**: View task execution history

## Troubleshooting

### Common Issues

**Issue**: `OPENAI_API_KEY not set`
```bash
# Solution: Set the API key in .env or environment
export OPENAI_API_KEY=your-key-here
```

**Issue**: `Chrome/Chromium not found`
```bash
# Solution: Install Chromium via Playwright
uv run playwright install chromium --with-deps

# Or specify custom path in .env
CHROME_PATH=/usr/bin/chromium-browser
```

**Issue**: Tests failing
```bash
# Solution: Ensure all dependencies are installed
uv sync --all-extras
uv run playwright install chromium

# Run tests with verbose output
uv run pytest tests/ -vv
```

**Issue**: Port already in use
```bash
# Solution: Use a different port
uv run server --port 8001

# Or find and kill the process using the port
lsof -ti:8081 | xargs kill -9
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set log level in .env
LOG_LEVEL=DEBUG

# Or via environment variable
LOG_LEVEL=DEBUG uv run server --port 8081
```

## Architecture

The server follows a modular architecture:

1. **MCP Server**: Handles protocol communication (SSE/stdio)
2. **Browser Manager**: Manages browser instances and contexts
3. **Task Queue**: Async task execution and tracking
4. **Agent System**: AI-powered browser automation via browser-use
5. **Resource Manager**: Memory and cleanup management

See [AGENTS.md](./AGENTS.md) for detailed architecture documentation.

## Performance

- **Concurrent Tasks**: Supports multiple simultaneous browser tasks
- **Memory Management**: Automatic cleanup of old tasks and browser contexts
- **Resource Limits**: Configurable via Docker compose
- **Async Operations**: Non-blocking task execution

Typical performance metrics:
- Task startup: <2 seconds
- Simple navigation: 3-5 seconds
- Complex automation: 10-30 seconds
- Memory per browser: 200-500 MB

## Security

### Best Practices

- **Never commit** `.env` files or API keys to version control
- Use **Docker secrets** for production deployments
- **Limit** exposed ports in production
- **Set resource limits** via Docker to prevent DoS
- **Validate** all user inputs in browser tasks
- Use **HTTPS** for SSE mode in production

### Environment Isolation

Docker provides strong isolation. For additional security:

```yaml
# In docker-compose.yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

Copyright (c) 2025 Dr. Hubertus Becker

## Acknowledgments

- Built on [browser-use](https://github.com/browser-use/browser-use) for browser automation
- Uses [MCP](https://github.com/modelcontextprotocol) for AI agent communication
- Powered by [Playwright](https://playwright.dev/) for browser control
- Package management via [uv](https://github.com/astral-sh/uv)

## Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/hubertusgbecker/mcp-browser-use-server/issues)
- **Discussions**: [Join the conversation](https://github.com/hubertusgbecker/mcp-browser-use-server/discussions)
- **Email**: For private inquiries, contact via GitHub

## Changelog

### Version 0.9.5 (2025-11-11)

- Complete rebranding and optimization
- Comprehensive test suite added
- Docker Compose support
- Enhanced documentation
- Production-ready deployment configuration

---

<div align="center">

[GitHub](https://github.com/hubertusgbecker) • [Repository](https://github.com/hubertusgbecker/mcp-browser-use-server)

</div>
