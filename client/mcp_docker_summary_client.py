#!/usr/bin/env python3
"""MCP summary client.

This script verifies a running Dockerized MCP server by:
- checking the server TCP port is open
- connecting over SSE to the MCP server
- listing available tools
- running a small browser task and polling for completion
- creating a short-lived session, navigating and retrieving state

Usage:
    uv run python client/mcp_summary_client.py

Environment:
    MCP_SERVER_URL: Optional full URL to the server SSE endpoint (default: http://127.0.0.1:8081/sse)
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, Optional

import mcp
from mcp.client.sse import sse_client


def safe_load_json(text: str) -> Optional[Any]:
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


async def tcp_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        fut = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


def parse_host_port_from_url(url: str) -> tuple[str, int]:
    # naive parse for http://host:port/path
    m = re.match(r"https?://([^:/]+)(?::(\d+))?", url)
    if not m:
        return "127.0.0.1", 8081
    host = m.group(1)
    port = int(m.group(2)) if m.group(2) else 80
    return host, port


async def run_task_test(session: mcp.ClientSession, tools: list) -> None:
    task_tool_names = ["run_browser_task", "browser_use", "browser_task"]
    task_tool = None
    for t in tools:
        if t.name in task_tool_names:
            task_tool = t.name
            break

    if not task_tool:
        print("No task tool found (run_browser_task/browser_use). Skipping task test.")
        return

    # browser_use requires url and action parameters
    url = "https://quotes.toscrape.com/"
    action = "Extract the first 3 quotes with their authors and return as JSON array"

    print(f"Calling task tool: {task_tool}")
    print(f"  URL: {url}")
    print(f"  Action: {action}")
    
    # Call with correct parameters based on tool name
    if task_tool == "browser_use":
        resp = await session.call_tool(task_tool, {"url": url, "action": action})
    else:
        # Fallback for other tool names
        resp = await session.call_tool(task_tool, {"instruction": f"Go to {url} and {action}"})

    text = ""
    if resp and getattr(resp, "content", None):
        text = getattr(resp.content[0], "text", "") or ""

    print(f"Raw response from {task_tool}:")
    print(text[:500] if len(text) > 500 else text)  # Print first 500 chars
    print()

    data = safe_load_json(text) if text else None

    # If the tool returns final result immediately
    if data and data.get("final_result"):
        print("Task returned immediate final_result")
        print(json.dumps(data, indent=2))
        return

    task_id = None
    if data:
        task_id = data.get("task_id") or data.get("id")

    if not task_id:
        # Try to extract task_id from plain text
        m = re.search(r"task[_-]?id[:=]\s*([a-zA-Z0-9_-]+)", text or "", re.I)
        if m:
            task_id = m.group(1)

    if not task_id:
        print("Could not determine task_id from call_tool response; raw response:")
        print(text)
        return

    print(f"Polling for task_id={task_id}")

    # determine poll tool
    poll_tool = None
    for name in ("browser_get_result", "get_task_status", "task_status"):
        if any(t.name == name for t in tools):
            poll_tool = name
            break

    if not poll_tool:
        print("No poll tool available; cannot continue polling")
        return

    for i in range(60):
        await asyncio.sleep(2 if i < 10 else 5)
        try:
            result_resp = await session.call_tool(poll_tool, {"task_id": task_id})
        except Exception as e:
            print(f"Poll call failed: {e}")
            continue

        result_text = ""
        if result_resp and getattr(result_resp, "content", None):
            result_text = getattr(result_resp.content[0], "text", "") or ""

        if not result_text.strip():
            print(f"[{i}] Empty poll response; continuing")
            continue

        result_data = safe_load_json(result_text)
        if not result_data:
            print(f"[{i}] Non-JSON poll response:\n{result_text}")
            continue

        status = result_data.get("status") or result_data.get("state")
        print(f"[{i}] status={status}")
        if status in ("completed", "failed"):
            print("Final task result:")
            print(json.dumps(result_data, indent=2))
            return

    print("Task polling timed out")


async def run_session_test(session: mcp.ClientSession, tools: list) -> None:
    # session tools expected: browser_navigate, browser_get_state, browser_close_session
    if not any(t.name == "browser_navigate" for t in tools):
        print("No session tools available; skipping session test")
        return

    print("Creating short session via browser_navigate")
    nav_resp = await session.call_tool("browser_navigate", {"url": "https://example.com"})
    nav_text = ""
    if nav_resp and getattr(nav_resp, "content", None):
        nav_text = getattr(nav_resp.content[0], "text", "") or ""

    nav_data = safe_load_json(nav_text) or {}
    session_id = nav_data.get("session_id") or nav_data.get("id")

    if not session_id:
        # Some implementations return the state directly; print and skip close
        print("browser_navigate did not return session_id; response:")
        print(nav_text)
        return

    print(f"Created session: {session_id}")

    if any(t.name == "browser_get_state" for t in tools):
        state_resp = await session.call_tool("browser_get_state", {"session_id": session_id, "screenshot": False})
        state_text = ""
        if state_resp and getattr(state_resp, "content", None):
            state_text = getattr(state_resp.content[0], "text", "") or ""
        
        # Try to parse as JSON, if not just print raw
        state_data = safe_load_json(state_text)
        if state_data:
            print("Session state (JSON):")
            print(json.dumps(state_data, indent=2))
        else:
            print("Session state (raw):")
            print(state_text[:500] if len(state_text) > 500 else state_text)

    if any(t.name == "browser_close_session" for t in tools):
        close_resp = await session.call_tool("browser_close_session", {"session_id": session_id})
        print("Closed session response:")
        if close_resp and getattr(close_resp, "content", None):
            print(getattr(close_resp.content[0], "text", ""))


async def main() -> int:
    url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8081/sse")
    host, port = parse_host_port_from_url(url)

    print(f"Checking TCP port {host}:{port}...")
    if not await tcp_port_open(host, port):
        print(f"Unable to reach {host}:{port}. Is the Docker container running and exposing the port?")
        return 2

    print("Connecting to MCP server over SSE...")
    try:
        async with sse_client(url) as (read_stream, write_stream):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                print("Available tools:", [t.name for t in tools])

                # Run task test
                await run_task_test(session, tools)

                # Run session test
                await run_session_test(session, tools)

    except Exception as e:
        print(f"Error communicating with MCP server: {e}")
        return 3

    print("MCP container smoke tests completed")
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
