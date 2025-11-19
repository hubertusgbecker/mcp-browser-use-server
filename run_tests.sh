#!/usr/bin/env bash
# run_tests.sh - Comprehensive test runner for mcp-browser-use-server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MCP Browser Use Server Test Suite ===${NC}"

# Check if uv is installed
	if ! command -v uv &>/dev/null; then
		echo -e "${RED}Error: uv is not installed${NC}"
		echo "Please see https://astral.sh/uv for installation instructions."
		echo "CI recommendation: use the 'astral-sh/setup-uv@v5' GitHub Action to provide uv on runners"
		exit 1
	fi

# Ensure pytest is available inside the uv-managed Python environment.
ensure_pytest_installed() {
	if ! uv run python -c "import pytest" >/dev/null 2>&1; then
		echo -e "${YELLOW}pytest not found in uv environment; installing test extras...${NC}"
		# Install the test extras from the project (uses uv pip)
		uv pip install ".[test]" || {
			echo -e "${RED}Failed to install test extras. Try running 'uv sync' manually.${NC}"
			exit 1
		}
	fi
}


# Parse command line arguments
TEST_TYPE="${1:-all}"
COVERAGE="${2:-true}"

echo -e "${YELLOW}Test type: $TEST_TYPE${NC}"

# Run different test suites based on argument
case $TEST_TYPE in
unit)
	echo -e "${GREEN}Running unit tests...${NC}"
	ensure_pytest_installed
	uv run python -m pytest -m unit tests/
	;;
integration)
	echo -e "${GREEN}Running integration tests...${NC}"
	ensure_pytest_installed
	uv run python -m pytest -m integration tests/
	;;
e2e)
	echo -e "${GREEN}Running end-to-end tests...${NC}"
	export RUN_E2E_TESTS=true
	ensure_pytest_installed
	uv run python -m pytest -m e2e tests/
	;;
fast)
	echo -e "${GREEN}Running fast tests (excluding slow)...${NC}"
	ensure_pytest_installed
	uv run python -m pytest -m "not slow" tests/
	;;
slow)
	echo -e "${GREEN}Running slow tests...${NC}"
	ensure_pytest_installed
	uv run python -m pytest -m slow tests/
	;;
coverage)
	echo -e "${GREEN}Running tests with detailed coverage...${NC}"
	ensure_pytest_installed
	uv run python -m pytest --cov=src --cov=server --cov-report=html --cov-report=term-missing tests/
	echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
	;;
all)
	echo -e "${GREEN}Running all tests...${NC}"
	ensure_pytest_installed
	uv run python -m pytest tests/
	# Optionally run Mega-Linter after tests when RUN_LINTER is set
	if [ "${RUN_LINTER:-false}" = "true" ]; then
		echo -e "${YELLOW}RUN_LINTER=true: running Mega-Linter...${NC}"
		if command -v npx >/dev/null 2>&1; then
			npx --yes mega-linter-runner --install || true
			npx --yes mega-linter-runner --run || true
		else
			echo -e "${RED}npx is required to run Mega-Linter locally. Install Node.js/npm.${NC}"
		fi
	fi
	;;
lint)
	echo -e "${GREEN}Running Mega-Linter locally...${NC}"
	if command -v npx >/dev/null 2>&1; then
		echo -e "${YELLOW}Installing Mega-Linter runner (if needed)...${NC}"
		npx --yes mega-linter-runner --install || true
		echo -e "${YELLOW}Running Mega-Linter... This may take some time.${NC}"
		npx --yes mega-linter-runner --run || {
			echo -e "${RED}Mega-Linter run failed${NC}"
			exit 1
		}
	else
		echo -e "${RED}npx is required to run Mega-Linter locally. Install Node.js/npm.${NC}"
		exit 1
	fi
	;;
*)
	echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
	echo "Usage: $0 [unit|integration|e2e|fast|slow|coverage|all]"
	exit 1
	;;
esac

# Check exit code
if [ $? -eq 0 ]; then
	echo -e "${GREEN}✓ Tests passed!${NC}"
else
	echo -e "${RED}✗ Tests failed!${NC}"
	exit 1
fi
