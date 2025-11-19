# GitHub Copilot Instructions for mcp-browser-use-server

## Project Context

You are working on **mcp-browser-use-server**, a production-ready MCP (Model Context Protocol) server for browser automation  This project enables AI agents to control web browsers through the browser-use library.


## Key Principles

### 1. Use uv for Everything Python
- **Always use `uv`** for package management, not pip or conda
- Commands: `uv sync`, `uv run`, `uv build`, `uv tool install`
- Never suggest `pip install` - use `uv pip install` or `uv sync`

Note: For consistency and reproducibility across developer machines and CI, never run `python -m pip` or `pip` directly in documentation, scripts or CI. Always prefer `uv` wrappers above.

### CI Note: setup-uv in workflows
When writing or updating GitHub Actions that run Python commands, prefer using the `astral-sh/setup-uv@v5` action to install and configure `uv` on the runner. After the action, use `uv venv` and `uv pip` / `uv run` for packaging and test commands instead of system `pip`/`python` calls.

### 2. Type Safety First
- All functions MUST have type hints
- Use `Optional[T]`, `Union[T1, T2]`, `List[T]`, `Dict[K, V]`
- Run `mypy` to verify: `uv run mypy src/ server/`

### 3. Async by Default
- Use `async def` for I/O operations
- Use `await` for async calls
- Mark tests with `@pytest.mark.asyncio`

### 4. Comprehensive Testing
- Target: >95% test coverage
- Categories: unit, integration, e2e, performance
- Run tests: `./run_tests.sh fast` or `uv run pytest tests/`
- Always add tests when adding features

### 5. Documentation Standards
- Google-style docstrings for all public functions
- Include Args, Returns, Raises sections
- Update README.md and AGENT.md for significant changes

### 6. Context7-Aware Suggestions
Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

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

## Project Structure Quick Reference

```
Key Files:
-- server/server.py          # Core server implementation
-- server/session.py         # Persistent session manager (live sessions, tabs, extraction)
-- src/mcp_browser_use_server/cli.py  # CLI interface
- tests/conftest.py         # Test fixtures
- pyproject.toml           # Dependencies & metadata
- docker-compose.yaml      # Container orchestration
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

## Configuration Management

Always use the environment variable pattern:
```python
# Read from .env or environment
config = init_configuration()

# Access config values
api_key = config["OPENAI_API_KEY"]
chrome_path = config.get("CHROME_PATH")  # Optional values
patient_mode = config["PATIENT"]

### Note on `.env`

The project keeps runtime secrets and local configuration in a `.env` file which is excluded from
version control by the repository `.gitignore`. If a secret is detected in the working tree (for
example by Mega-Linter), there is no need to perform a git history rewrite when the file was never
committed. Instead: remove or rotate the secret locally and update any affected CI/secret stores.

```

## Error Handling

```python
try:
    result = await risky_operation()
except SpecificError as e:
    logger.error(f"Specific error occurred: {e}", exc_info=True)
    # Handle specific error
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Handle generic error
    raise RuntimeError(f"Operation failed: {e}") from e
```

## Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages", exc_info=True)  # Include traceback
```

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

## Common Pitfalls to Avoid

1. **Don't use pip** - Always use `uv`
2. **Don't forget type hints** - mypy should pass
3. **Don't skip tests** - Maintain coverage
4. **Don't block in async** - Use async/await properly
5. **Don't hardcode paths** - Use config or env vars
6. **Don't ignore errors** - Log and handle properly
7. **Don't commit secrets** - Use .env (gitignored)

## Quick Reference Links

- [AGENTS.md](../AGENTS.md) - Comprehensive AI context
- [README.md](../README.md) - User documentation
- [pyproject.toml](../pyproject.toml) - Project configuration
- [tests/conftest.py](../tests/conftest.py) - Test fixtures

## When Suggesting Code

1. **Check existing patterns** in the codebase first
2. **Use consistent style** with existing code
3. **Include type hints** and docstrings
4. **Suggest tests** alongside implementation
5. **Consider error cases** and edge cases
6. **Think about async** implications
7. **Document assumptions** in comments
8. **Follow uv conventions** for Python operations

## Helpful Commands Cheat Sheet (uv-only)

```bash
# Development
uv sync                          # Install dependencies
uv run server --port 8081        # Run server
LOG_LEVEL=DEBUG uv run server    # Run with debug logging

# Testing
./run_tests.sh fast              # Quick test run (uses uv)
./run_tests.sh coverage          # With coverage report
uv run pytest tests/ -v          # Verbose tests via uv

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

*These instructions help GitHub Copilot provide context-aware suggestions for the mcp-browser-use-server project.*
