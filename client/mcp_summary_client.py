#!/usr/bin/env python3
"""MCP client to call the `browser_use` tool and poll for a result.

This script connects to the local MCP server over SSE, requests the
available tools, calls `browser_use` for a given URL and action, and
polls `browser_get_result` until the task completes.

Usage:
    uv run python client/mcp_summary_client.py
"""

import asyncio
import json
import re
from typing import Any

import mcp
from mcp.client.sse import sse_client


def safe_load_json(text: str) -> Any:
    """Attempt to parse JSON, return None on failure.

    The server may sometimes return empty or non-JSON text. This helper
    centralizes a forgiving parse attempt.
    """
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        # Try to extract a JSON fragment if the response contains one
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


async def main() -> None:
    url = "http://127.0.0.1:8082/sse"
    async with sse_client(url) as (read_stream, write_stream):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("Tools:", [t.name for t in tools])

            # Call browser_use
            resp = await session.call_tool(
                "browser_use",
                {
                    "url": "https://hubertusbecker.com",
                    "action": "summarize in 300 characters",
                },
            )
            print("CallTool result:", resp)

            # Extract task_id from response. Some tool implementations may
            # return the final result immediately instead of a task id.
            text = ""
            if resp and getattr(resp, "content", None):
                text = getattr(resp.content[0], "text", "") or ""

            data = safe_load_json(text)
            if data is None:
                # If the response contains a final_result string (plain text),
                # print it and exit.
                if text and "final_result" in text:
                    print("Immediate final result (raw):")
                    print(text)
                    return
                print("Unable to parse task_id from response")
                return

            task_id = data.get("task_id")
            if not task_id:
                # If there is a final result present directly, show it.
                if data.get("final_result"):
                    print("Immediate final result:", json.dumps(data, indent=2))
                    return
                print("No task_id returned by call_tool response")
                return

            # Poll until task completes
            for _ in range(60):
                await asyncio.sleep(5)
                result_resp = await session.call_tool(
                    "browser_get_result", {"task_id": task_id}
                )
                result_text = ""
                if result_resp and getattr(result_resp, "content", None):
                    result_text = (
                        getattr(result_resp.content[0], "text", "") or ""
                    )

                if not result_text.strip():
                    print("Empty result; continuing to poll")
                    continue

                result_data = safe_load_json(result_text)
                if result_data is None:
                    print("Failed to parse result JSON; raw:")
                    print(result_text)
                    continue

                status = result_data.get("status")
                print(f"Polled status: {status}")
                if status in ("completed", "failed"):
                    print("Final result:", json.dumps(result_data, indent=2))
                    break


if __name__ == "__main__":
    asyncio.run(main())
