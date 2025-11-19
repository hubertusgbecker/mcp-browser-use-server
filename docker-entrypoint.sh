#!/bin/bash
set -euo pipefail

# Resolve home directory for runtime user (fallback to /home/appuser)
VNC_HOME=${HOME:-/home/appuser}

mkdir -p "$VNC_HOME/.vnc" || true
chown -R $(id -u):$(id -g) "$VNC_HOME/.vnc" || true

# Use Docker secret for VNC password if available, else fallback to default
if command -v vncpasswd >/dev/null 2>&1; then
  if [ -f "/run/secrets/vnc_password" ]; then
    cat /run/secrets/vnc_password | vncpasswd -f > "$VNC_HOME/.vnc/passwd" || true
  elif [ -f "/run/secrets/vnc_password_default" ]; then
    cat /run/secrets/vnc_password_default | vncpasswd -f > "$VNC_HOME/.vnc/passwd" || true
  fi
  chmod 600 "$VNC_HOME/.vnc/passwd" || true
else
  echo "vncpasswd not found; skipping VNC password setup" >&2
fi

# Start VNC server if available
if command -v vncserver >/dev/null 2>&1; then
  vncserver -depth 24 -geometry 1920x1080 -localhost no -rfbauth "$VNC_HOME/.vnc/passwd" :0 || true
else
  echo "vncserver not found; skipping VNC server startup" >&2
fi

# Start proxy automator if available
if command -v proxy-login-automator >/dev/null 2>&1; then
  # Start in background and redirect output; capture exit code separately to
  # avoid using `||` after a background job which can cause a shell syntax
  # error in some environments.
  proxy-login-automator &>/dev/null &
  _pid=$!
  if ! kill -0 "$_pid" >/dev/null 2>&1; then
    echo "warning: failed to start proxy-login-automator" >&2
  fi
else
  echo "proxy-login-automator not found; skipping" >&2
fi

# Exec the provided command (e.g. python server)
exec "$@"
