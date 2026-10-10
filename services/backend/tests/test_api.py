"""Users only reach their own threads and runs."""

import httpx
import pytest

from assistant.api.identity import VISITOR_COOKIE, Identity
from assistant.api.main import app, lifespan
from assistant.config import settings
from tests.conftest import GRAPH, TEST_USER, SlowFake, cleanup, fake_graph

pytestmark = pytest.mark.db

ALICE = {settings.identity_header: TEST_USER}
BOB = {settings.identity_header: "pytest-bob"}


@pytest.fixture
async def client():
    async with lifespan(app):
        # Threads on a graph no worker has, so no model ever runs for them.
        app.state.graphs[GRAPH] = fake_graph(
            SlowFake(words=["x"]), app.state.checkpointer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1"
        ) as c:
            yield c
        await cleanup(app.state.pool, app.state.checkpointer)
        async with app.state.pool.connection() as conn:
            await conn.execute("DELETE FROM threads WHERE user_id = 'pytest-bob'")


async def test_other_users_get_nothing(client):
    tid = (await client.post("/threads", json={"graph": GRAPH}, headers=ALICE)).json()[
        "thread_id"
    ]
    run = (
        await client.post(
            f"/threads/{tid}/runs", json={"content": "ahoj"}, headers=ALICE
        )
    ).json()["run_id"]

    assert tid not in (await client.get("/threads", headers=BOB)).text
    assert (await client.get(f"/threads/{tid}", headers=BOB)).status_code == 404
    assert (
        await client.post(f"/threads/{tid}/runs", json={"content": "x"}, headers=BOB)
    ).status_code == 404
    assert (
        await client.patch(f"/threads/{tid}", json={"title": "pwned"}, headers=BOB)
    ).status_code == 404
    assert (await client.get(f"/runs/{run}/stream", headers=BOB)).status_code == 404
    assert (await client.post(f"/runs/{run}/cancel", headers=BOB)).status_code == 409
    assert (await client.delete(f"/threads/{tid}", headers=BOB)).status_code == 404
    # No header means a new visitor, who isn't Alice either.
    assert tid not in (await client.get("/threads")).text

    mine = (await client.get(f"/threads/{tid}", headers=ALICE)).json()
    assert mine["title"] != "pwned"


async def test_too_long_message_is_refused(client):
    tid = (await client.post("/threads", json={"graph": GRAPH}, headers=ALICE)).json()[
        "thread_id"
    ]
    r = await client.post(
        f"/threads/{tid}/runs",
        json={"content": "x" * (settings.max_input_chars + 1)},
        headers=ALICE,
    )
    assert r.status_code == 413


@pytest.fixture
async def visitors():
    """Two browsers with their own cookie jars, and no trusted header."""
    async with lifespan(app):
        app.state.graphs[GRAPH] = fake_graph(
            SlowFake(words=["x"]), app.state.checkpointer
        )
        app.state.identity = Identity(header=None, secure_cookie=False)
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as a,
            httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as b,
        ):
            yield a, b


async def test_visitors_only_see_their_own_threads(visitors):
    a, b = visitors
    r = await a.post("/threads", json={"graph": GRAPH})
    tid = r.json()["thread_id"]
    assert VISITOR_COOKIE in r.cookies
    try:
        # The same browser keeps its threads; another one gets its own id.
        assert tid in (await a.get("/threads")).text
        assert tid not in (await b.get("/threads")).text
        assert (await b.get(f"/threads/{tid}")).status_code == 404
        assert a.cookies[VISITOR_COOKIE] != b.cookies[VISITOR_COOKIE]
    finally:
        await a.delete(f"/threads/{tid}")


async def test_header_is_ignored_unless_configured(visitors):
    a, b = visitors
    forged = {"Cf-Access-Authenticated-User-Email": TEST_USER}
    tid = (await a.post("/threads", json={"graph": GRAPH}, headers=forged)).json()[
        "thread_id"
    ]
    try:
        assert tid not in (await b.get("/threads", headers=forged)).text
    finally:
        await a.delete(f"/threads/{tid}")


async def test_invalid_visitor_cookie_gets_a_new_id(visitors):
    a, _ = visitors
    a.cookies.set(VISITOR_COOKIE, "admin")
    r = await a.get("/threads")
    assert r.json() == []
    assert r.cookies[VISITOR_COOKIE] != "admin"
