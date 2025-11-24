FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed

# Install build dependencies and clean up in the same layer
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y clang git && \
    rm -rf /var/lib/apt/lists/*

# Install Chromium and dependencies for browser automation
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        chromium \
        chromium-driver \
        fonts-liberation \
        libnss3 \
        libatk-bridge2.0-0 \
        libgtk-3-0 \
        libxss1 \
        libasound2 \
        libgbm1 \
        libu2f-udev \
        libvulkan1 \
        xdg-utils \
        --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/chromium /usr/local/bin/chromium || true

# Set environment variable for Chromium path
ENV CHROME_PATH=/usr/bin/chromium

# Install Python before the project for caching
RUN uv python install 3.13

WORKDIR /app
# Copy lock and pyproject first to leverage Docker layer caching without BuildKit
COPY uv.lock pyproject.toml /app/
# Create a cache directory for uv to use inside the image
RUN mkdir -p /root/.cache/uv
# Install dependencies according to the lock file. This mirrors the intent of
# the BuildKit `--mount=type=cache` + `--mount=type=bind` usage but is
# compatible with legacy Docker builders (e.g. Synology DSM). Note: copying
# the full source later avoids invalidating this layer when application code
# changes.
RUN uv sync --frozen --no-install-project --no-dev
COPY . /app
# Ensure the uv cache directory exists for subsequent operations
RUN mkdir -p /root/.cache/uv && uv sync --frozen --no-dev

FROM debian:bookworm-slim AS runtime

# VNC password will be read from Docker secrets or fallback to default
# Create a fallback default password file
RUN mkdir -p /run/secrets && \
    echo "browser-use" > /run/secrets/vnc_password_default

# Install required packages including Node, Chromium dependencies, VNC and fonts.
# Use a robust, fail-fast script so missing packages are noticed during build
# and global npm installs are run with --unsafe-perm to allow root installs.
RUN set -eux; \
    apt-get update; \
    apt-get install --no-install-recommends -y ca-certificates curl gnupg dirmngr lsb-release apt-transport-https; \
    # Install Node 20.x via NodeSource (reliable across Debian releases)
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get install --no-install-recommends -y nodejs; \
    # Install desktop, VNC server/tools, fonts, curl and shellcheck in one step
    apt-get install --no-install-recommends -y \
        xfce4 \
        xfce4-terminal \
        dbus-x11 \
        tigervnc-standalone-server \
        tigervnc-tools \
        shellcheck \
        fonts-freefont-ttf \
        fonts-ipafont-gothic \
        fonts-wqy-zenhei \
        fonts-thai-tlwg \
        fonts-kacst \
        fonts-symbola \
        fonts-noto-color-emoji; \
    # Clean up apt caches to reduce image size
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*
RUN set -eux; \
    # Install global node packages required by the runtime in a separate step
    # Each package is explicitly pinned by version to make installs reproducible
    npm i -g --unsafe-perm proxy-login-automator@1.2.0 mega-linter-runner@9.1.0; \
    # Ensure no leftover caches
    rm -rf /var/cache/npm /var/lib/apt/lists/* || true


# Copy only necessary files from builder
COPY --from=builder /python /python
COPY --from=builder /app /app
# Copy uv binary from builder to /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
# Set proper permissions
RUN chmod -R 755 /python /app && chmod +x /usr/local/bin/uv

ENV ANONYMIZED_TELEMETRY=false \
    PATH="/usr/local/bin:/app/.venv/bin:$PATH" \
    DISPLAY=:0 \
    CHROME_BIN=/usr/bin/chromium \
    CHROMIUM_FLAGS="--no-sandbox --headless --disable-gpu --disable-software-rasterizer --disable-dev-shm-usage"

# Combine VNC setup commands to reduce layers. Create VNC files in the
# non-root user's home directory so the runtime user can manage them
RUN mkdir -p /home/appuser/.vnc && \
    printf '#!/bin/sh\nunset SESSION_MANAGER\nunset DBUS_SESSION_BUS_ADDRESS\nstartxfce4' > /home/appuser/.vnc/xstartup && \
    chmod +x /home/appuser/.vnc/xstartup

# Use a maintained entrypoint script for container startup instead of
# generating an inline boot script. The script handles VNC password setup
# and optional services at runtime and then execs the container command.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Install hadolint and actionlint binaries (download latest releases)
RUN set -eux; \
    HADOLINT_VERSION="2.12.0"; \
    ACTIONLINT_VERSION="1.7.7"; \
    # hadolint (static binary)
    curl -sSL -o /usr/local/bin/hadolint "https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-$(uname -s)-$(uname -m)" && \
    chmod +x /usr/local/bin/hadolint || true; \
    # actionlint (static binary) - release provides linux/amd64
    curl -sSL -o /usr/local/bin/actionlint "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint-linux-amd64" && \
    chmod +x /usr/local/bin/actionlint || true; \
    # Verify installation (non-fatal)
    if command -v hadolint >/dev/null 2>&1; then hadolint --version || true; fi; \
    if command -v actionlint >/dev/null 2>&1; then actionlint --version || true; fi

# Install Playwright's system dependencies as root before creating appuser
RUN /app/.venv/bin/playwright install-deps chromium

# Create appuser before installing Playwright so browsers are owned by appuser
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin appuser && \
    mkdir -p /home/appuser && chown -R appuser:appuser /app /home/appuser /run/secrets

# Install Playwright browsers as appuser (without --with-deps since we installed deps as root)
USER appuser
RUN /app/.venv/bin/playwright install chromium

EXPOSE 8081

# Use the repository entrypoint which is safer and easier to maintain
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Simple healthcheck to satisfy scanners (uses curl to check server availability)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD curl -f http://127.0.0.1:8081/health || exit 1

# Default command: run the server in the foreground so containers stay up
# when no explicit command is provided. Users can still override this by
# passing a different command to `docker run`.
CMD ["/app/.venv/bin/mcp-browser-use-server", "run", "server", "--port", "8081", "--log-level", "INFO"]
