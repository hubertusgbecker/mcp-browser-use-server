import pytest
from starlette.testclient import TestClient

from server.server import PlainTextResponse, Route, Starlette
from tests._helpers import create_starlette_app_with_sse


def test_health_endpoint_minimal():
    """Minimal health endpoint test using Starlette TestClient.

    This test is intentionally self-contained and does not import heavy
    runtime dependencies so it can run quickly in CI.
    """

    async def _health(request):
        return PlainTextResponse("ok")

    starlette_app = Starlette(routes=[Route("/health", endpoint=_health)])

    # Use TestClient to call the health endpoint
    with TestClient(starlette_app) as client:
        res = client.get("/health")
        if res.status_code != 200:
            pytest.fail(f"Expected status 200, got {res.status_code}")
        if res.text != "ok":
            pytest.fail(f"Expected body 'ok', got {res.text!r}")


@pytest.mark.asyncio
async def test_health_endpoint_is_ok(mock_llm):
    """Health endpoint should return plain 'ok'."""
    app = create_starlette_app_with_sse()

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("/health")
        if resp.status_code != 200:
            pytest.fail(f"Expected status 200, got {resp.status_code}")
        if resp.text != "ok":
            pytest.fail(f"Expected body 'ok', got {resp.text!r}")


def test_sse_route_and_messages_mount_registered(mock_llm):
    """Verify that /sse route and /messages/ mount are registered on the app."""
    app = create_starlette_app_with_sse()

    # Check that /sse route exists
    paths = {getattr(route, 'path', '') for route in app.routes}
    if "/sse" not in paths:
        pytest.fail("/sse route not registered on the app")
    # Check that the messages mount exists
    if not any(
        getattr(route, "path", "").startswith("/messages")
        for route in app.routes
    ):
        pytest.fail("/messages mount not registered on the app")
