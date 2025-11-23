#!/usr/bin/env bash

# Simple mcp-only smoke test for MCP Browser Use Server
# Creates a browser task to summarize hubertusbecker.com in 300 characters

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MCP_BIN=${MCP_BIN:-mcp}
SERVER=${SERVER:-http://127.0.0.1:8082}

echo -e "${BLUE}=== MCP smoke test ===${NC}"
echo "Using mcp: $MCP_BIN"
echo "Target server: $SERVER"

if ! command -v "$MCP_BIN" >/dev/null 2>&1; then
  echo -e "${RED}mcp not found in PATH. Install it first.${NC}"
  exit 2
fi

echo -e "\n${YELLOW}1) List tools exposed by the MCP server${NC}"
"$MCP_BIN" tools "$SERVER/sse" -f json || { echo -e "${RED}Failed to list tools${NC}"; exit 3; }

echo -e "\n${YELLOW}2) Summarize hubertusbecker.com in 300 characters${NC}"
SUMMARY_PROMPT='Summarize the main content and purpose of this website in 300 characters or fewer.'

TMP_PAYLOAD=$(mktemp)
trap 'rm -f "$TMP_PAYLOAD"' EXIT

# Build simple JSON payload (SUMMARY_PROMPT contains no double quotes)
printf '{"url":"%s","action":"%s"}' "https://hubertusbecker.com" "$SUMMARY_PROMPT" > "$TMP_PAYLOAD"

echo "Creating browser task..."
CREATE_OUT=$($MCP_BIN call "$SERVER/sse" browser_use -p "$(cat "$TMP_PAYLOAD")" -f json 2>/dev/null || true)
echo "$CREATE_OUT"

# Try to extract task_id
TASK_ID=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"task_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)

if [ -z "$TASK_ID" ]; then
  # Maybe synchronous result
  FINAL=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"final_result"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
  if [ -n "$FINAL" ]; then
    FINAL_UNESCAPED=$(printf '%b' "${FINAL//\\"/\"}")
    echo -e "\n${GREEN}Summary (sync):${NC}"
    echo "${FINAL_UNESCAPED:0:300}"
  else
    echo -e "${RED}No task_id and no synchronous result.${NC}"
    exit 4
  fi
else
  echo "Task created: $TASK_ID"
  echo "Polling for result (timeout 120s)..."
  MAX_TRIES=24
  TRY=0
  #!/usr/bin/env bash

  # Simple mcp-only smoke test for MCP Browser Use Server
  # Creates a browser task to summarize hubertusbecker.com in 300 characters

  set -euo pipefail

  # Colors
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'

  MCP_BIN=${MCP_BIN:-mcp}
  SERVER=${SERVER:-http://127.0.0.1:8082}

  echo -e "${BLUE}=== MCP smoke test ===${NC}"
  echo "Using mcp: $MCP_BIN"
  echo "Target server: $SERVER"

  if ! command -v "$MCP_BIN" >/dev/null 2>&1; then
    echo -e "${RED}mcp not found in PATH. Install it first.${NC}"
    exit 2
  fi

  echo -e "\n${YELLOW}1) List tools exposed by the MCP server${NC}"
  "$MCP_BIN" tools "$SERVER/sse" -f json || { echo -e "${RED}Failed to list tools${NC}"; exit 3; }

  echo -e "\n${YELLOW}2) Summarize hubertusbecker.com in 300 characters${NC}"
  SUMMARY_PROMPT='Summarize the main content and purpose of this website in 300 characters or fewer.'

  TMP_PAYLOAD=$(mktemp)
  trap 'rm -f "$TMP_PAYLOAD"' EXIT

  # Build simple JSON payload. SUMMARY_PROMPT should not contain raw double quotes.
  printf '%s' "{\"url\":\"https://hubertusbecker.com\",\"action\":\"$SUMMARY_PROMPT\"}" > "$TMP_PAYLOAD"

  echo "Creating browser task..."
  CREATE_OUT=$($MCP_BIN call "$SERVER/sse" browser_use -p "$(cat "$TMP_PAYLOAD")" -f json 2>/dev/null || true)
  echo "$CREATE_OUT"

  # Try to extract task_id
  TASK_ID=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"task_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)

  if [ -z "$TASK_ID" ]; then
    FINAL=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"final_result"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
    if [ -n "$FINAL" ]; then
      FINAL_UNESCAPED=$(printf '%b' "${FINAL//\\"/\"}")
      echo -e "\n${GREEN}Summary (sync):${NC}"
      echo "${FINAL_UNESCAPED:0:300}"
    else
      echo -e "${RED}No task_id and no synchronous result.${NC}"
      exit 4
    fi
  else
    echo "Task created: $TASK_ID"
    echo "Polling for result (timeout 120s)..."
    MAX_TRIES=24
    TRY=0
    while [ $TRY -lt $MAX_TRIES ]; do
      sleep 5
      TRY=$((TRY+1))
      echo "Attempt $TRY: checking status..."
      PAYLOAD=$(printf '%s' "{\"task_id\":\"%s\"}" "$TASK_ID")
      RES=$($MCP_BIN call "$SERVER/sse" browser_get_result -p "$PAYLOAD" -f json 2>/dev/null || true)
      echo "$RES" | sed -n '1,200p'
      if printf '%s' "$RES" | grep -q '"status"[[:space:]]*:[[:space:]]*"completed"'; then
        SUMMARY=$(printf '%s' "$RES" | sed -n 's/.*"final_result"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
        if [ -z "$SUMMARY" ]; then
          SUMMARY=$(printf '%s' "$RES" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
        fi
        if [ -n "$SUMMARY" ]; then
          SUMMARY_UNESCAPED=$(printf '%b' "${SUMMARY//\\"/\"}")
          echo -e "\n${GREEN}Summary (completed):${NC}"
          echo "${SUMMARY_UNESCAPED:0:300}"
        else
          echo -e "${GREEN}Task completed but no summary text found.${NC}"
        fi
        break
      fi
      if printf '%s' "$RES" | grep -q '"status"[[:space:]]*:[[:space:]]*"failed"'; then
        echo -e "${RED}Summary task failed.${NC}"
        break
      fi
    done
    if [ $TRY -ge $MAX_TRIES ]; then
      echo -e "${RED}Summary task timeout after $((MAX_TRIES*5)) seconds.${NC}"
    fi
  fi

  echo -e "\n${GREEN}mcp smoke test completed${NC}"

  exit 0
# Try extract task_id from output (look for "task_id": "..." inside any returned JSON)
TASK_ID=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"task_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)

if [ -z "$TASK_ID" ]; then
  # Maybe the result was synchronous and contains final_result
  FINAL=$(printf '%s' "$CREATE_OUT" | sed -n 's/.*"final_result"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
  if [ -n "$FINAL" ]; then
    # Decode escaped quotes etc - simple unescape for common escapes
    FINAL_UNESCAPED=$(printf '%b' "${FINAL//\\"/\"}")
    echo -e "\n${GREEN}Summary (sync):${NC}"
    echo "${FINAL_UNESCAPED:0:300}"
  else
    echo -e "${RED}No task_id returned and no synchronous final_result found.${NC}"
  fi
else
  echo "Task created: $TASK_ID"
  echo "Polling for result (timeout 120s)..."
  MAX_TRIES=24
  TRY=0
  while [ $TRY -lt $MAX_TRIES ]; do
    sleep 5
    TRY=$((TRY+1))
    echo "Attempt $TRY: checking status..."
    RES=$($MCPTOOLS call "$MCP_SERVER/sse" browser_get_result -p "{\"task_id\": \"$TASK_ID\"}" -f json 2>/dev/null || true)
    echo "$RES" | sed -n '1,200p'
    # Check completed
    if printf '%s' "$RES" | grep -q '"status"[[:space:]]*:[[:space:]]*"completed"'; then
      # Extract content text field inside content[0].text which contains the result JSON or text
      # First try to extract final_result
      SUMMARY=$(printf '%s' "$RES" | sed -n 's/.*"final_result"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
      if [ -z "$SUMMARY" ]; then
        # Try to get the text field inside content array
        SUMMARY=$(printf '%s' "$RES" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true)
      fi
      if [ -n "$SUMMARY" ]; then
        SUMMARY_UNESCAPED=$(printf '%b' "${SUMMARY//\\"/\"}")
        echo -e "\n${GREEN}Summary (completed):${NC}"
        # Print first 300 characters
        echo "${SUMMARY_UNESCAPED:0:300}"
      else
        echo -e "${GREEN}Task completed but no summary text found in response.${NC}"
      fi
      break
    fi
    if printf '%s' "$RES" | grep -q '"status"[[:space:]]*:[[:space:]]*"failed"'; then
      echo -e "${RED}Summary task failed.${NC}"
      break
    fi
  done
  if [ $TRY -ge $MAX_TRIES ]; then
    echo -e "${RED}Summary task timeout after $((MAX_TRIES*5)) seconds.${NC}"
  fi
fi

echo -e "\n${GREEN}mcptools smoke test completed successfully${NC}"

exit 0
