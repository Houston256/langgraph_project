"""Shared fixtures. Unit tests need nothing; `db` tests need the compose postgres.

uv run pytest                 # all; db tests skip without postgres
docker compose up -d postgres # for the db tests
"""

import asyncio
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

FIXTURES = Path(__file__).parent / "fixtures"

# Before assistant.config is imported: real .env values win, dummies fill the gaps,
# so the tests run without a .env and without the private prompt.
load_dotenv(find_dotenv())
for key, value in {
    "OPENAI_API_KEY": "test",
    "MODEL_NAME": "openai/gpt-6-luna",
    "QDRANT_URL_GRPC": "localhost:6334",
    "QDRANT_API_KEY": "test",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_GRPC_PORT": "6334",
    "POSTGRES_URL": "postgresql://assistant:assistant@localhost:5432/assistant",
}.items():
    os.environ.setdefault(key, value)
os.environ["SYSTEM_PROMPT_FILE"] = str(FIXTURES / "prompt.md")

import psycopg
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langgraph.graph import END, START, MessagesState, StateGraph

from assistant.config import settings
from assistant.db import Listener, open_db
from assistant.graphs.registry import GraphSpec
from assistant.runs import store
from assistant.runs.events import PgEventLog
from assistant.usage import Pricing
from assistant.worker import queue
from assistant.worker.execution import RunExecutor
from assistant.worker.queue import Worker

TEST_USER = "pytest"
GRAPH = "pytest"


def postgres_reachable() -> bool:
    try:
        psycopg.connect(settings.postgres_url, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


def pytest_collection_modifyitems(config, items):
    if postgres_reachable():
        return
    skip = pytest.mark.skip(
        reason="postgres not reachable (docker compose up -d postgres)"
    )
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


class SlowFake(BaseChatModel):
    """Streams `words` with a delay; `fail_once` raises a provider error on the first call."""

    words: list[str]
    delay: float = 0.02
    first_delay: float = 0.0
    fail_once: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "slow-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage("".join(self.words)))]
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        if self.fail_once is not None:
            error, self.fail_once = self.fail_once, None
            raise error
        await asyncio.sleep(self.first_delay)
        for word in self.words:
            await asyncio.sleep(self.delay)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                await run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk


def fake_graph(llm: BaseChatModel, checkpointer) -> GraphSpec:
    async def chatbot(state: MessagesState, config):
        return {"messages": [await llm.ainvoke(state["messages"], config=config)]}

    g = StateGraph(MessagesState)
    g.add_node("chatbot", chatbot)
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", END)
    return GraphSpec(g.compile(checkpointer=checkpointer), "chatbot")


async def no_title(pool, run, callbacks) -> None:
    pass


class Env:
    """Pool, event log and fake graph; `start_worker()` runs a worker in the background."""

    def __init__(self, pool, checkpointer, listener, llm):
        self.pool = pool
        self.checkpointer = checkpointer
        self.listener = listener
        self.llm = llm
        self.events = PgEventLog(pool, listener)
        # A graph name no real worker has, so running compose workers leave these runs alone.
        self.graphs = {GRAPH: fake_graph(llm, checkpointer)}
        self.workers: list[tuple[Worker, asyncio.Task]] = []

    def start_worker(self, lease: float | None = None) -> Worker:
        executor = RunExecutor(self.pool, self.events, self.graphs, Pricing(), no_title)
        worker = Worker(self.pool, self.events, self.listener, executor)
        if lease is not None:
            worker.lease = lease
        self.workers.append((worker, asyncio.create_task(worker.run())))
        return worker

    async def thread(self) -> dict:
        return await store.create_thread(self.pool, TEST_USER, GRAPH)

    async def collect(
        self, run_id, after: int = 0, idle: float = 15
    ) -> list[tuple[int, dict]]:
        out = []
        async for item in self.events.read(run_id, after, idle):
            assert item is not None, "stream went idle"
            out.append(item)
        return out

    async def status(self, run_id) -> str | None:
        return await store.run_status(self.pool, run_id)

    async def wait_status(self, run_id, want: set[str], timeout: float = 10) -> str:
        for _ in range(int(timeout / 0.05)):
            if (s := await self.status(run_id)) in want:
                return s
            await asyncio.sleep(0.05)
        raise AssertionError(
            f"{run_id} never reached {want}, is {await self.status(run_id)}"
        )


@pytest.fixture(autouse=True)
def shop(monkeypatch):
    monkeypatch.setattr(settings, "allowed_link_domains", ["myshop.cz", "myshop.sk"])


WORDS = [f"w{i} " for i in range(40)]
FULL = "".join(WORDS).strip()


@pytest.fixture
async def env(monkeypatch):
    # No network: limits off, so no OpenRouter price lookup; fast lease polling.
    monkeypatch.setattr(settings, "limit_global_usd_per_month", None)
    monkeypatch.setattr(queue, "IDLE_POLL_SECONDS", 0.2)
    listener = Listener(["runs", "run_cancel", "run_events"])
    async with open_db() as (pool, checkpointer):
        listen = asyncio.create_task(listener.run())
        await asyncio.sleep(0.2)
        e = Env(pool, checkpointer, listener, SlowFake(words=WORDS))
        try:
            yield e
        finally:
            for _, task in e.workers:
                task.cancel()
            await asyncio.gather(*(t for _, t in e.workers), return_exceptions=True)
            listen.cancel()
            await cleanup(pool, checkpointer)


async def cleanup(pool, checkpointer) -> None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT thread_id FROM threads WHERE user_id = %s", (TEST_USER,)
        )
        for row in await cur.fetchall():
            await checkpointer.adelete_thread(str(row["thread_id"]))
        await conn.execute("DELETE FROM threads WHERE user_id = %s", (TEST_USER,))
        await conn.execute("DELETE FROM usage WHERE user_id = %s", (TEST_USER,))
