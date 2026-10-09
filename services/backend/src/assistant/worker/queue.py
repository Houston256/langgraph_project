"""Claims runs from Postgres, keeps their leases, and decides how each one ends."""

import asyncio
import logging
from typing import Any
from uuid import UUID

from assistant.config import settings
from assistant.db import Listener, Pool
from assistant.runs import store
from assistant.runs.events import Emitter, EventLog
from assistant.runs.store import LeaseLost, RunStatus
from assistant.worker.execution import RunExecutor
from assistant.worker.retry import retry_delay

log = logging.getLogger(__name__)

# How long to wait for a notification before polling for runs with expired leases.
IDLE_POLL_SECONDS = 5.0


class Worker:
    def __init__(
        self, pool: Pool, events: EventLog, listener: Listener, executor: RunExecutor
    ):
        self.pool = pool
        self.events = events
        self.listener = listener
        self.executor = executor
        self.lease = settings.run_lease_seconds
        self.flush_interval = settings.flush_interval_ms / 1000
        self.slots = asyncio.Semaphore(settings.worker_concurrency)
        self.tasks: dict[asyncio.Task, tuple[UUID, int]] = {}

    async def run(self) -> None:
        while True:
            await self.slots.acquire()
            with self.listener.subscribe("runs") as woke:
                try:
                    run = await store.claim_run(
                        self.pool, self.lease, list(self.executor.graphs)
                    )
                except Exception:
                    log.exception("claiming a run failed")
                    run = None
                if run is None:
                    self.slots.release()
                    try:
                        await asyncio.wait_for(woke.wait(), IDLE_POLL_SECONDS)
                    except TimeoutError:
                        pass
                    continue
            task = asyncio.create_task(self.execute(run))
            self.tasks[task] = (run["run_id"], run["attempt"])
            task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task) -> None:
        self.tasks.pop(task, None)
        self.slots.release()

    async def shutdown(self, grace_seconds: float) -> None:
        """Lets local runs finish for up to grace_seconds, then hands the rest back to the queue."""
        if self.tasks:
            log.info("waiting up to %ss for %s run(s)", grace_seconds, len(self.tasks))
            await asyncio.wait(set(self.tasks), timeout=grace_seconds)
        held = dict(self.tasks)
        for task in held:
            task.cancel()
        await asyncio.gather(*held, return_exceptions=True)
        for run_id, attempt in held.values():
            await store.release_run(self.pool, run_id, attempt)

    async def execute(self, run: dict[str, Any]) -> None:
        run_id: UUID = run["run_id"]
        attempt: int = run["attempt"]
        log.info("run %s attempt %s (%s)", run_id, attempt, run["status"])
        emitter = Emitter(
            self.events,
            run_id,
            attempt,
            await self.events.last_seq(run_id),
            self.flush_interval,
        )

        async def end(status: RunStatus, error: str | None = None) -> None:
            # Status first: a client that sees run-end reloads the thread.
            if await store.finish_run(self.pool, run_id, attempt, status, error):
                await emitter.emit(
                    {"type": "run-end", "status": status, "error": error}
                )
                await emitter.flush()
                # Deletes the thread if it was deleted while this run was open.
                checkpointer = self.executor.graphs[run["graph"]].graph.checkpointer
                await store.purge_thread(self.pool, run["thread_id"], checkpointer)

        try:
            async with emitter:
                if attempt > settings.run_max_attempts:
                    await end(RunStatus.ERROR, "too many attempts")
                elif missing := await self.executor.unpriced_models():
                    # Limits can't be enforced on calls whose cost is unknown.
                    await end(
                        RunStatus.ERROR, f"no price known for {', '.join(missing)}"
                    )
                elif run["status"] == RunStatus.CANCELLING:
                    await self.executor.finalize_cancel(run)
                    await end(RunStatus.CANCELLED)
                else:
                    outcome = await self.supervise(run, emitter)
                    if outcome == RunStatus.CANCELLED:
                        await emitter.flush()
                        await self.executor.finalize_cancel(run)
                    await end(outcome)
        except LeaseLost:
            log.warning("run %s lost its lease", run_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            delay = retry_delay(e, attempt)
            if delay is not None and attempt < settings.run_max_attempts:
                log.warning(
                    "run %s attempt %s: %s, retrying in %.0fs",
                    run_id,
                    attempt,
                    e,
                    delay,
                )
                await emitter.emit(
                    {"type": "status", "text": "The model is busy, trying again…"}
                )
                await store.retry_run(self.pool, run_id, attempt, delay)
                return
            log.exception("run %s failed", run_id)
            await end(RunStatus.ERROR, f"{type(e).__name__}: {e}")

    async def supervise(self, run: dict[str, Any], emitter: Emitter) -> RunStatus:
        """Runs the graph while renewing the lease. Returns DONE or CANCELLED."""
        run_id, attempt = run["run_id"], run["attempt"]
        with self.listener.subscribe("run_cancel", str(run_id)) as cancel:
            work = asyncio.create_task(self.executor.stream(run, emitter))
            try:
                while True:
                    waiter = asyncio.create_task(cancel.wait())
                    await asyncio.wait(
                        {work, waiter},
                        timeout=self.lease / 3,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    waiter.cancel()
                    if work.done():
                        work.result()
                        return RunStatus.DONE
                    status = await store.heartbeat(
                        self.pool, run_id, attempt, self.lease
                    )
                    if status is None:
                        work.cancel()
                        await asyncio.gather(work, return_exceptions=True)
                        # Another worker owns the run now; buffered events would clash with its seqs.
                        emitter.drop()
                        raise LeaseLost
                    if status == RunStatus.CANCELLING:
                        return RunStatus.CANCELLED
                    cancel.clear()
            finally:
                if not work.done():
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
