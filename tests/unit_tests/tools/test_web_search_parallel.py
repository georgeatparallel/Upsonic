"""Exercise provider selection through Upsonic and its actual MCP HTTP client."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from upsonic import __version__
from upsonic.tools import ToolManager, WebSearch, aWebSearch


@pytest.fixture
def mcp_wire(monkeypatch):
    requests = []
    state = {"payload": {"results": [
        {"title": "First", "url": "https://example.com/first", "excerpts": ["One", "Two"]},
        {"title": "Second", "url": "https://example.com/second", "excerpts": ["Three"]},
    ]}}

    async def serve(request):
        requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == state.get("block_method"):
            state["entered"].set()
            await asyncio.Event().wait()
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "description": "Search",
                       "inputSchema": {"type": "object", "properties": {
                           "objective": {"type": "string"},
                           "search_queries": {"type": "array", "items": {"type": "string"}},
                       }, "required": ["objective", "search_queries"]}}]}
        elif method == "tools/call":
            if state.get("http_error"):
                return httpx.Response(503)
            if state.get("rpc_error"):
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"],
                                                "error": {"code": -32603, "message": "Failure"}})
            result = {"content": [{"type": "text", "text": json.dumps(state["payload"])}],
                      "isError": state.get("tool_error", False)}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    original = httpx.AsyncClient.__init__

    def initialize(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(serve)
        original(client, *args, **kwargs)
        state.setdefault("clients", []).append(client)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", initialize)
    return requests, state


@pytest.mark.asyncio
async def test_native_selection_results_and_wire_identity(mcp_wire):
    requests, _ = mcp_wire
    manager = ToolManager()
    manager.register_tools(tools=[WebSearch])
    result = await manager.execute_tool(tool_name="WebSearch", args={
        "query": "official docs", "provider": "parallel", "max_results": 1,
    })
    assert result.success
    assert "First" in str(result.content) and "https://example.com/first" in str(result.content)
    assert "One\nTwo" in result.content["func"]
    assert "Second" not in str(result.content)
    calls = [json.loads(r.content) for r in requests if r.method == "POST"]
    assert any(c.get("method") == "tools/list" for c in calls)
    call = next(c for c in calls if c.get("method") == "tools/call")
    assert call["params"]["arguments"] == {"objective": "official docs", "search_queries": ["official docs"]}
    assert all(r.headers["user-agent"] == f"upsonic/{__version__}" for r in requests)
    assert all("authorization" not in r.headers and "x-api-key" not in r.headers for r in requests)
    assert all(str(r.url) == "https://search.parallel.ai/mcp" for r in requests)


def test_default_stays_duckduckgo(mcp_wire):
    requests, _ = mcp_wire
    client = MagicMock()
    client.text.return_value = [{"title": "Incumbent", "href": "https://example.org", "body": "Original"}]
    with patch("upsonic.tools.builtin_tools._DDGS_AVAILABLE", True), patch("upsonic.tools.builtin_tools.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value = client
        assert "Incumbent" in WebSearch("query")
        assert "Incumbent" in WebSearch("query", provider="duckduckgo")
    assert not requests


@pytest.mark.asyncio
async def test_async_empty_success_and_reconnection(mcp_wire):
    requests, state = mcp_wire
    state["payload"] = {"results": [], "warnings": ["Adjusted query"]}
    for _ in range(2):
        result = await aWebSearch("query", provider="parallel")
        assert result.startswith("Web search results for:") and "Adjusted query" in result
    assert sum(json.loads(r.content).get("method") == "initialize" for r in requests if r.method == "POST") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool_error", "rpc_error", "http_error", "malformed", "invalid_result"])
async def test_failures_stay_failures(mcp_wire, failure):
    _, state = mcp_wire
    if failure == "malformed":
        state["payload"] = {"unrelated": []}
    elif failure == "invalid_result":
        state["payload"] = {"results": [{"title": "No source"}]}
    else:
        state[failure] = True
    if failure == "http_error":
        # The maintained MCP SDK cancels its session on a broken HTTP stream.
        # Cancellation must propagate, rather than becoming empty search success.
        with pytest.raises(asyncio.CancelledError):
            await aWebSearch("query", provider="parallel")
    else:
        assert (await aWebSearch("query", provider="parallel")).startswith("Error performing web search:")


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"provider": "unknown"}, {"provider": "parallel", "query": " "},
    {"provider": "parallel", "max_results": 0}, {"provider": "parallel", "max_results": True},
])
async def test_invalid_input_never_dispatches(mcp_wire, args):
    requests, _ = mcp_wire
    with pytest.raises(ValueError):
        await aWebSearch(**{"query": "query", **args})
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
async def test_cancellation_closes_client_during_setup_and_call(mcp_wire, method):
    _, state = mcp_wire
    state.update(block_method=method, entered=asyncio.Event())
    task = asyncio.create_task(aWebSearch("query", provider="parallel"))
    await asyncio.wait_for(state["entered"].wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(client.is_closed for client in state["clients"])
