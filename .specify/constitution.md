# Project Constitution: MCP Browser Use Server

## Project Identity

**Name**: MCP Browser Use Server  
**Purpose**: Production-ready Model Context Protocol (MCP) server for browser automation using browser-use library  
**Core Value**: Reliable, secure, and scalable browser automation for AI agents

## Technical Principles

### 1. Reliability First
- All code must have comprehensive test coverage (unit, integration, e2e)
- Docker health checks must accurately reflect service state
- Configuration must be consistent across all deployment methods
- Failures must be graceful with clear error messages

### 2. Security & Safety
- No hardcoded credentials or API keys
- All secrets managed via environment variables or Docker secrets
- Container runs as non-root user (appuser)
- Security scanning via Mega-Linter and dependency audits
- No unnecessary capabilities in containers

### 3. Developer Experience
- Clear, comprehensive documentation
- Simple setup process (Docker Compose for quick start)
- Consistent port configuration via HOST_PORT
- JSON-formatted structured logging
- Hot reload support for development

### 4. Production Readiness
- Health checks for all services
- Graceful shutdown handling
- Resource cleanup (old tasks, browser contexts)
- VNC support for debugging browser sessions
- Prometheus metrics and Grafana dashboards

### 5. Code Quality
- Type hints for all Python code
- Comprehensive docstrings
- Linting with ruff, mypy, bandit
- Pre-commit hooks for consistency
- No code duplication (checked via jscpd)

## Development Workflow

### Branching Strategy
- `main`: Production-ready code, all tests passing
- Feature branches: `feature/<description>`
- Hotfix branches: `hotfix/<description>`

### Testing Requirements
- All PRs must pass: unit, integration, e2e, and docker tests
- Test coverage must not decrease
- All linters must pass (Mega-Linter)
- Security scans must pass (Bandit, KICS)

### Commit Standards
- Use conventional commits: `fix:`, `feat:`, `docs:`, `chore:`, etc.
- Include test results in commit messages for major changes
- Reference issues/tickets when applicable

## Architecture Principles

### 1. Separation of Concerns
- Server logic (`server/server.py`) handles MCP protocol and browser orchestration
- CLI (`src/mcp_browser_use_server/cli.py`) provides command-line interface
- Docker configuration separate from application code

### 2. Configuration Management
- Environment variables for all configurable values
- Sensible defaults for all settings
- `.env.example` as documentation and template
- HOST_PORT controls both external and internal ports

### 3. Browser Automation
- Use browser-use library's native ChatOpenAI for LLM integration
- Support both headless and headed modes
- VNC for visual debugging
- Proper cleanup of browser contexts and sessions

### 4. API Design
- RESTful endpoints for HTTP mode
- Server-Sent Events (SSE) for real-time updates
- MCP protocol compliance for tool integration
- Clear error responses with actionable messages

## Quality Gates

### Before Merge
- [ ] All tests passing (unit, integration, e2e, docker)
- [ ] Linters passing (ruff, mypy, bandit, hadolint, etc.)
- [ ] Security scans clean
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

### Before Release
- [ ] Version bumped in `pyproject.toml`
- [ ] Docker image builds successfully
- [ ] Integration with Magg aggregator verified
- [ ] README reflects current state
- [ ] Migration guide (if breaking changes)

## Dependencies

### Core Dependencies
- Python 3.13+
- browser-use library for automation
- FastAPI/Starlette for HTTP server
- MCP SDK for protocol implementation
- Playwright for browser control

### Development Dependencies
- pytest for testing
- ruff for linting
- mypy for type checking
- uv for package management

### Infrastructure
- Docker for containerization
- Docker Compose for orchestration
- Prometheus for metrics
- Grafana for visualization

## Constraints

### Performance
- Health check response < 1 second
- Task cleanup runs every hour
- Browser sessions timeout after inactivity
- Maximum agent steps configurable (default: 10)

### Compatibility
- Support macOS, Linux, and Docker environments
- Python 3.13+ required
- Compatible with Magg MCP aggregator
- Works with GitHub Copilot, Claude, and other AI assistants

## Success Metrics

- Test coverage > 80%
- All security scans passing
- Docker health checks reliable
- Zero hardcoded secrets
- Documentation completeness
- Community adoption (GitHub stars, forks)

## Evolution

This constitution is a living document. Updates require:
1. Discussion in GitHub issue
2. Consensus from maintainers
3. Update to this file
4. Communication to contributors

---

**Last Updated**: 2025-11-29  
**Version**: 1.0.0
