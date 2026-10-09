"""python -m assistant.worker"""

import asyncio
import logging
import signal

from assistant.config import settings
from assistant.db import Listener, open_db
from assistant.graphs.registry import open_graphs
from assistant.runs.events import PgEventLog
from assistant.usage import Pricing
from assistant.worker.cleanup import trim_events_forever
from assistant.worker.execution import RunExecutor
from assistant.worker.queue import Worker
from assistant.worker.titles import make_title

log = logging.getLogger("assistant.worker")

# Runs hold a connection only per query, so a few serve all of them.
DB_POOL_SIZE = 8


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    listener = Listener(["runs", "run_cancel"])
    async with (
        open_db(max_size=DB_POOL_SIZE) as (pool, checkpointer),
        open_graphs(checkpointer) as graphs,
    ):
        # Readers live in the api; the worker's log only writes.
        events = PgEventLog(pool, listener)
        executor = RunExecutor(pool, events, graphs, Pricing(), make_title)
        worker = Worker(pool, events, listener, executor)
        tasks = [
            asyncio.create_task(listener.run()),
            loop := asyncio.create_task(worker.run()),
            cleanup := asyncio.create_task(trim_events_forever(pool)),
        ]

        stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        log.info("worker up, concurrency %s", settings.worker_concurrency)
        await stop.wait()

        log.info("shutting down")
        loop.cancel()
        cleanup.cancel()
        await worker.shutdown(settings.shutdown_grace_seconds)
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
