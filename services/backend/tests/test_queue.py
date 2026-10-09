"""The run queue end to end with a fake model: streaming, cancel, interrupt, crash, drain, retry."""

import asyncio

import httpx
import openai
import pytest

from assistant.runs import store
from assistant.runs.messages import to_ui_messages
from assistant.runs.store import LeaseLost
from assistant.usage import limit_exceeded
from assistant.utils.streaming import create_config
from tests.conftest import FULL, GRAPH, TEST_USER

pytestmark = pytest.mark.db


def text_of(events, after_reset: bool = True) -> str:
    if after_reset:
        resets = [i for i, (_, e) in enumerate(events) if e["type"] == "reset"]
        events = events[resets[-1] :] if resets else events
    return "".join(e["delta"] for _, e in events if e["type"] == "text-delta").strip()


async def thread_texts(env, thread_id) -> list[tuple[str, str]]:
    state = await env.graphs[GRAPH].graph.aget_state(create_config(str(thread_id)))
    return [
        (m["role"], " ".join(p.get("text", "") for p in m["parts"]).strip())
        for m in to_ui_messages(state.values.get("messages", []))
    ]


async def test_stream_merges_deltas_and_resumes_from_any_seq(env):
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hi")
    events = await env.collect(run)
    resumed = await env.collect(run, after=events[1][0])
    assert [e for _, e in events][2:] == [e for _, e in resumed]
    assert text_of(events) == FULL
    # 40 tokens in fewer events: deltas are merged.
    assert sum(e["type"] == "text-delta" for _, e in events) < 40
    assert events[-1][1] == {"type": "run-end", "status": "done", "error": None}
    assert await thread_texts(env, t["thread_id"]) == [
        ("user", "hi"),
        ("assistant", FULL),
    ]


async def test_cancel_keeps_the_partial_answer(env):
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "stop me")
    await env.wait_status(run, {"running"})
    await asyncio.sleep(0.4)
    assert await store.cancel_run(env.pool, run, TEST_USER)
    events = await env.collect(run)
    assert events[-1][1]["status"] == "cancelled"
    partial = text_of(events)
    assert 0 < len(partial) < len(FULL)
    assert await thread_texts(env, t["thread_id"]) == [
        ("user", "stop me"),
        ("assistant", partial),
    ]


async def test_new_message_interrupts_and_runs_after(env):
    env.start_worker()
    t = await env.thread()
    first = await store.create_run(env.pool, t["thread_id"], "first")
    await env.wait_status(first, {"running"})
    await asyncio.sleep(0.3)
    second = await store.create_run(env.pool, t["thread_id"], "second")
    assert (await env.collect(first))[-1][1]["status"] == "cancelled"
    assert (await env.collect(second))[-1][1]["status"] == "done"
    roles = await thread_texts(env, t["thread_id"])
    assert [r for r, _ in roles] == ["user", "assistant", "user", "assistant"]
    assert roles[2:] == [("user", "second"), ("assistant", FULL)]


async def test_cancelled_pending_run_keeps_its_message(env):
    env.start_worker()
    t = await env.thread()
    a = await store.create_run(env.pool, t["thread_id"], "a")
    await env.wait_status(a, {"running"})
    b = await store.create_run(env.pool, t["thread_id"], "b")
    c = await store.create_run(
        env.pool, t["thread_id"], "c"
    )  # cancels b while still pending
    for run, want in ((a, "cancelled"), (b, "cancelled"), (c, "done")):
        assert (await env.collect(run))[-1][1]["status"] == want
    users = [
        text for role, text in await thread_texts(env, t["thread_id"]) if role == "user"
    ]
    assert users == ["a", "b", "c"]


async def test_crashed_run_is_taken_over_without_duplicates(env):
    crashed = env.start_worker(lease=1.0)
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "crash")
    await env.wait_status(run, {"running"})
    await asyncio.sleep(0.3)
    # A hard kill: tasks stop without finishing or handing back the run.
    env.workers[0][1].cancel()
    for task in list(crashed.tasks):
        task.cancel()
    await asyncio.sleep(0.1)
    assert await env.status(run) == "running"
    env.start_worker(lease=1.0)
    events = await env.collect(run)
    assert events[-1][1]["status"] == "done"
    assert any(e["type"] == "reset" for _, e in events)
    assert text_of(events) == FULL
    assert await thread_texts(env, t["thread_id"]) == [
        ("user", "crash"),
        ("assistant", FULL),
    ]


async def test_shutdown_drains_or_hands_back(env):
    worker = env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "drain")
    await env.wait_status(run, {"running"})
    env.workers[0][1].cancel()
    await worker.shutdown(5)
    assert await env.status(run) == "done"

    worker = env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hand back")
    await env.wait_status(run, {"running"})
    env.workers[-1][1].cancel()
    await worker.shutdown(0.1)
    async with env.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT status, locked_until FROM runs WHERE run_id = %s", (run,)
        )
        row = await cur.fetchone()
    assert row["status"] == "running" and row["locked_until"] is None
    env.start_worker()
    assert (await env.collect(run))[-1][1]["status"] == "done"


async def test_provider_402_is_retried(env):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    env.llm.fail_once = openai.APIStatusError(
        "credit hold",
        response=httpx.Response(402, request=request, headers={"retry-after": "1"}),
        body=None,
    )
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "retry me")
    events = await env.collect(run)
    kinds = [e["type"] for _, e in events]
    assert events[-1][1]["status"] == "done"
    assert "status" in kinds and "reset" in kinds
    assert text_of(events) == FULL


async def test_monthly_cap_refuses_new_runs(env, monkeypatch):
    from assistant.config import settings

    monkeypatch.setattr(settings, "limit_global_usd_per_month", 20)
    async with env.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT coalesce(sum(cost_usd), 0) AS s FROM usage WHERE created_at >= date_trunc('month', now())"
        )
        spent = float((await cur.fetchone())["s"])
        await conn.execute(
            "INSERT INTO usage (user_id, model, input_tokens, cached_tokens, output_tokens, cost_usd)"
            " VALUES (%s, 'test', 0, 0, 0, %s)",
            (TEST_USER, 20 - spent + 1),
        )
    reason = await limit_exceeded(env.pool, TEST_USER)
    assert reason and "monthly" in reason


async def test_hourly_run_limit_survives_thread_delete(env, monkeypatch):
    from uuid import uuid4

    from assistant.config import settings

    monkeypatch.setattr(settings, "limit_user_runs_per_hour", 2)
    async with env.pool.connection() as conn:
        for _ in range(2):
            await conn.execute(
                "INSERT INTO usage (run_id, user_id, model, input_tokens, cached_tokens, output_tokens)"
                " VALUES (%s, %s, 'test', 0, 0, 0)",
                (uuid4(), TEST_USER),
            )
    t = await env.thread()
    await store.delete_thread(env.pool, t["thread_id"], TEST_USER)
    reason = await limit_exceeded(env.pool, TEST_USER)
    assert reason and "hour" in reason


async def test_worker_keeps_claiming_after_a_db_error(env, monkeypatch):
    claim = store.claim_run
    calls = 0

    async def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("connection lost")
        return await claim(*args, **kwargs)

    monkeypatch.setattr(store, "claim_run", flaky)
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hi")
    assert (await env.collect(run))[-1][1]["status"] == "done"
    assert calls > 1


async def test_cleanup_deletes_events_of_finished_runs(env):
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hi")
    await env.collect(run)
    assert await store.trim_events(env.pool, 0) > 0
    assert await env.events.last_seq(run) == 0


async def test_deleting_an_idle_thread_deletes_it_at_once(env):
    t = await env.thread()
    assert await store.delete_thread(env.pool, t["thread_id"], TEST_USER)
    assert await store.purge_thread(env.pool, t["thread_id"], env.checkpointer)


async def test_deleting_a_thread_stops_its_run_then_deletes_it(env):
    env.start_worker()
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hi")
    await env.wait_status(run, {"running"})
    assert await store.delete_thread(env.pool, t["thread_id"], TEST_USER)
    assert await store.get_thread(env.pool, t["thread_id"], TEST_USER) is None
    # The worker deletes the thread once the run has stopped.
    for _ in range(200):
        if await env.status(run) is None:
            break
        await asyncio.sleep(0.05)
    assert await env.status(run) is None
    assert await thread_texts(env, t["thread_id"]) == []


async def test_an_old_attempt_cannot_write_events(env):
    t = await env.thread()
    run = await store.create_run(env.pool, t["thread_id"], "hi")
    first = await store.claim_run(env.pool, 0.1, [GRAPH])
    await asyncio.sleep(0.2)
    second = await store.claim_run(env.pool, 30, [GRAPH])
    assert (first["attempt"], second["attempt"]) == (1, 2)
    with pytest.raises(LeaseLost):
        await env.events.append(run, 1, [(1, {"type": "status", "text": "old"})])
    await env.events.append(run, 2, [(1, {"type": "status", "text": "new"})])


async def test_two_workers_run_every_run_exactly_once(env):
    workers = [env.start_worker(), env.start_worker()]
    claimed = {id(w): 0 for w in workers}
    for w in workers:

        async def counting(run, w=w, execute=w.execute):
            claimed[id(w)] += 1
            await execute(run)

        w.execute = counting
    threads = [await env.thread() for _ in range(24)]
    runs = [await store.create_run(env.pool, t["thread_id"], "hi") for t in threads]
    results = await asyncio.gather(*(env.collect(r) for r in runs))

    assert all(events[-1][1]["status"] == "done" for events in results)
    assert all(text_of(events, after_reset=False) == FULL for events in results)
    async with env.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT max(attempt) AS a FROM runs WHERE run_id = ANY(%s)", (runs,)
        )
        assert (await cur.fetchone())["a"] == 1
    # 24 runs, 16 slots per worker: both had to take some.
    assert all(claimed.values()) and sum(claimed.values()) == 24
