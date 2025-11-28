# Project Constitution: MCP Browser Use Server

## Project Identity

**Name**: MCP Browser Use Server  
**Purpose**: Production-ready Model Context Protocol (MCP) server for browser automation using browser-use library  
**Core Value**: Reliable, secure, and scalable browser automation for AI agents

## Technical Principles

### 0. Test-Driven Development (TDD) - PRIMARY PRINCIPLE

**TDD is the foundation of all development in this project. No exceptions.**

#### TDD Workflow (Mandatory)
1. **Write the test FIRST** - Before any implementation code
2. **Run the test** - Verify it fails (red)
3. **Write minimal code** - Just enough to make the test pass (green)
4. **Run the test** - Verify it passes
5. **Refactor** - Improve code while keeping tests green
6. **Repeat** - For every feature, component, function, or unit

#### Granularity Requirements
- **Tiny steps**: Each test should verify ONE specific behavior
- **Fine-grained**: Test individual functions/methods, not entire features at once
- **Iterative**: Build features incrementally, test by test
- **Component-level**: Test each component in isolation before integration
- **Integration last**: Only after all unit tests pass, test integration

#### Never Do
- ❌ Big bang implementations
- ❌ Writing multiple features before testing
- ❌ Skipping tests "to save time"
- ❌ Writing implementation before tests
- ❌ Testing after the fact

#### Always Do
- ✅ Write test first, every time
- ✅ One test at a time
- ✅ Smallest possible increment
- ✅ Run tests after every change
- ✅ Keep all tests passing (green)

### 1. Reliability Through Testing
- **100% test coverage goal** - Every line of code must be tested
- **Tests before code** - No implementation without a failing test first
- **Continuous testing** - Run tests after every small change
- **Test pyramid**: Many unit tests → fewer integration tests → few e2e tests
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

### TDD-First Development Process

**Every feature, component, or unit follows this exact process:**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/<description>
   ```

2. **For Each Unit/Component/Feature:**
   
   a. **Write Test First**
      - Create test file: `tests/test_<component>.py`
      - Write ONE test for ONE behavior
      - Test should fail (RED)
   
   b. **Run Test - Verify Failure**
      ```bash
      pytest tests/test_<component>.py -v
      ```
   
   c. **Write Minimal Implementation**
      - Write just enough code to pass the test
      - No extra features or "nice-to-haves"
   
   d. **Run Test - Verify Success**
      ```bash
      pytest tests/test_<component>.py -v
      ```
   
   e. **Refactor (Optional)**
      - Improve code quality
      - Keep tests green
   
   f. **Commit Small Change**
      ```bash
      git add tests/ src/
      git commit -m "test: add test for <behavior>"
      git commit -m "feat: implement <behavior>"
      ```
   
   g. **Repeat** - Next test, next behavior

3. **Integration Testing**
   - Only after all unit tests pass
   - Test component interactions
   - Follow same TDD process

4. **E2E Testing**
   - Only after integration tests pass
   - Test complete user workflows
   - Follow same TDD process

### Branching Strategy
- `main`: Production-ready code, **ALL TESTS PASSING**
- Feature branches: `feature/<description>` - TDD workflow required
- Hotfix branches: `hotfix/<description>` - TDD workflow required

### Testing Requirements (TDD Enforced)
- **Tests written BEFORE implementation** - Verified in PR review
- All PRs must pass: unit, integration, e2e, and docker tests
- **Test coverage must be 100% for new code** - No exceptions
- Test coverage must never decrease overall
- All linters must pass (Mega-Linter)
- Security scans must pass (Bandit, KICS)
- **Each commit should include both test and implementation** - Paired commits

### Commit Standards (TDD Pattern)

**Preferred Pattern - Paired Commits:**
```bash
# First: Add failing test
git commit -m "test: add test for user authentication"

# Second: Make test pass
git commit -m "feat: implement user authentication"
```

**Commit Message Format:**
- Use conventional commits: `test:`, `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`
- **Always commit test before implementation**
- Include test results in commit messages
- Reference issues/tickets when applicable

**Example Commit Sequence:**
```
test: add test for login validation
feat: implement login validation
test: add test for password hashing
feat: implement password hashing
refactor: extract validation logic
test: add integration test for auth flow
feat: wire up auth components
```

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

### TDD Verification (Required for Every PR)
- [ ] **Tests written before implementation** - Verified in git history
- [ ] **Each feature has corresponding tests** - 1:1 mapping
- [ ] **Test coverage 100% for new code** - No untested lines
- [ ] **All tests passing** - Green across the board
- [ ] **Test commits precede implementation commits** - Proper TDD order
- [ ] **Small, incremental commits** - No big bang changes

### Before Merge
- [ ] All TDD verification items above ✅
- [ ] All tests passing (unit, integration, e2e, docker)
- [ ] Test coverage ≥ 80% overall, 100% for new code
- [ ] Linters passing (ruff, mypy, bandit, hadolint, etc.)
- [ ] Security scans clean
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
- [ ] Commit history shows TDD pattern (test → implementation)

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
