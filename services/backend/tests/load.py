"""Load test of api, workers and postgres with a fake model: no model calls, no cost.

uv run python -m tests.load [--users 25,50,100,200] [--workers 2] [--seconds 45] [--think 30]

Starts the api and the workers as their own processes, then simulated users chat:
each sends a message, reads the stream to the end, waits about --think seconds, repeats.
Needs the compose postgres. Prints time to first text and run time per step.
"""
# ruff: noqa: I001

import argparse
import asyncio
import json
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

# conftest sets the env before assistant.config loads.
from tests.conftest import SlowFake, fake_graph, no_title

from assistant.api import main as api
from assistant.config import settings
from assistant.db import Listener, open_db
from assistant.runs.events import PgEventLog
from assistant.usage import Pricing
from assistant.worker.execution import RunExecutor
from assistant.worker.queue import Worker

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}/api/v1"
USER_PREFIX = "load-"
# Its own graph name, so pytest workers and compose workers leave these runs alone.
GRAPH = "load"


def model() -> SlowFake:
    """About like the real model: 1.5 s to the first token, 3.3 s per answer."""
    return SlowFake(words=[f"w{i} " for i in range(60)], delay=0.03, first_delay=1.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with api.lifespan(app):
        app.state.graphs[GRAPH] = fake_graph(model(), app.state.checkpointer)
        yield


app = api.app
app.router.lifespan_context = lifespan


async def run_worker() -> None:
    listener = Listener(["runs", "run_cancel"])
    async with open_db(max_size=8) as (pool, checkpointer):
        graphs = {GRAPH: fake_graph(model(), checkpointer)}
        events = PgEventLog(pool, listener)
        executor = RunExecutor(pool, events, graphs, Pricing(), no_title)
        worker = Worker(pool, events, listener, executor)
        tasks = [
            asyncio.create_task(listener.run()),
            asyncio.create_task(worker.run()),
        ]
        stop = asyncio.Event()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
        await stop.wait()
        for t in tasks:
            t.cancel()


class Stats:
    def __init__(self) -> None:
        self.first: list[float] = []
        self.total: list[float] = []
        self.errors: list[str] = []


async def chat(
    http: httpx.AsyncClient, user: str, stop: float, think: float, stats: Stats
):
    headers = {settings.identity_header: user}
    await asyncio.sleep(random.uniform(0, think))
    try:
        r = await http.post("/threads", json={"graph": GRAPH}, headers=headers)
        r.raise_for_status()
        thread = r.json()
    except Exception as e:  # noqa: BLE001  counted, this user gives up
        stats.errors.append(f"thread: {type(e).__name__}")
        return
    while time.monotonic() < stop:
        start = time.monotonic()
        first = None
        try:
            r = await http.post(
                f"/threads/{thread['thread_id']}/runs",
                json={"content": "hi"},
                headers=headers,
            )
            r.raise_for_status()
            run_id = r.json()["run_id"]
            async with http.stream(
                "GET", f"/runs/{run_id}/stream", headers=headers
            ) as s:
                async for line in s.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "text-delta" and first is None:
                        first = time.monotonic() - start
                    if event["type"] == "run-end":
                        if event["status"] != "done":
                            stats.errors.append(event["status"])
                        break
            if first is not None:
                stats.first.append(first)
            stats.total.append(time.monotonic() - start)
        except Exception as e:  # noqa: BLE001  counted, the user keeps going
            stats.errors.append(type(e).__name__)
        await asyncio.sleep(random.uniform(0.5 * think, 1.5 * think))


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return (
        values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")
    )


def cpu(pids: list[int]) -> str:
    out = subprocess.run(
        ["ps", "-o", "%cpu=", "-p", ",".join(map(str, pids))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return "/".join(f"{float(c):.0f}" for c in out)


async def drive(
    users: list[int], seconds: float, think: float, pids: list[int]
) -> None:
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(base_url=BASE, timeout=60, limits=limits) as http:
        print(
            f"{'users':>5} {'runs/s':>6} {'first p50':>9} {'first p95':>9} "
            f"{'run p50':>7} {'run p95':>7} {'errors':>6}  cpu % api/workers"
        )
        for n in users:
            stats = Stats()
            stop = time.monotonic() + seconds
            jobs = [
                asyncio.create_task(
                    chat(http, f"{USER_PREFIX}{n}-{i}", stop, think, stats)
                )
                for i in range(n)
            ]
            await asyncio.sleep(seconds * 0.8)
            load = cpu(pids)
            await asyncio.gather(*jobs)
            print(
                f"{n:>5} {len(stats.total) / seconds:>6.1f} "
                f"{pct(stats.first, 0.5):>8.2f}s {pct(stats.first, 0.95):>8.2f}s "
                f"{pct(stats.total, 0.5):>6.2f}s {pct(stats.total, 0.95):>6.2f}s "
                f"{len(stats.errors):>6}  {load}"
            )
            if stats.errors:
                print("       errors:", dict(Counter(stats.errors)))


async def cleanup() -> None:
    async with open_db(max_size=2) as (pool, checkpointer), pool.connection() as conn:
        cur = await conn.execute(
            "SELECT thread_id FROM threads WHERE user_id LIKE %s", (USER_PREFIX + "%",)
        )
        for row in await cur.fetchall():
            await checkpointer.adelete_thread(str(row["thread_id"]))
        await conn.execute(
            "DELETE FROM threads WHERE user_id LIKE %s", (USER_PREFIX + "%",)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", default="25,50,100,200")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--think", type=float, default=30)
    parser.add_argument("--role", choices=["drive", "worker"], default="drive")
    args = parser.parse_args()
    if args.role == "worker":
        asyncio.run(run_worker())
        return

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "tests.load:app",
                "--port",
                str(PORT),
                "--log-level",
                "warning",
            ]
        )
    ]
    procs += [
        subprocess.Popen([sys.executable, "-m", "tests.load", "--role", "worker"])
        for _ in range(args.workers)
    ]
    try:
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/")
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        users = [int(n) for n in args.users.split(",")]
        asyncio.run(drive(users, args.seconds, args.think, [p.pid for p in procs]))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
