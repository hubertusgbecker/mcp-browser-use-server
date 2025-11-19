SHELL := /bin/bash
.PHONY: lint lint-docker install-runner

# Run Mega-Linter using npx when available, otherwise fall back to Docker
lint: install-runner
	@echo "Running Mega-Linter via npx..."
	npx --yes mega-linter-runner --run

install-runner:
	@if command -v npx >/dev/null 2>&1; then \
	  echo "Installing Mega-Linter runner (if needed)..."; \
	  npx --yes mega-linter-runner --install || true; \
	else \
	  echo "npx not found; use 'make lint-docker' to run via Docker"; \
	fi

# Use Docker fallback when node/npx isn't available or for CI isolation
lint-docker:
	@echo "Running Mega-Linter via Docker (fallback)..."
	docker run --rm -e RUN_LOCAL=true -e MEGALINTER_REPO_CONFIG=.mega-linter.yml \
	  -v "$(PWD)":/tmp/lint -w /tmp/lint nvuillam/mega-linter:latest
