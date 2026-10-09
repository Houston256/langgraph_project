from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg.types.json import Jsonb

from assistant.db import Pool, notify


class RunStatus(StrEnum):
    """Values of runs.status: pending -> running -> done | error, or -> cancelling -> cancelled."""

    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


TERMINAL = {RunStatus.DONE, RunStatus.CANCELLED, RunStatus.ERROR}


class LeaseLost(Exception):
    """The run's lease went to another attempt."""


# Newest open run of the thread, for the running indicator.
_ACTIVE_RUN = """
    (SELECT r.run_id FROM runs r
     WHERE r.thread_id = t.thread_id AND r.status IN ('pending', 'running', 'cancelling')
     ORDER BY r.created_at DESC LIMIT 1) AS active_run_id
"""


async def create_thread(pool: Pool, user_id: str, graph: str) -> dict[str, Any]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO threads (thread_id, user_id, graph) VALUES (%s, %s, %s) "
            "RETURNING thread_id, graph, title, created_at, updated_at, NULL::uuid AS active_run_id",
            (uuid4(), user_id, graph),
        )
        return await cur.fetchone()  # type: ignore[return-value]


async def list_threads(pool: Pool, user_id: str) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT thread_id, graph, title, created_at, updated_at, {_ACTIVE_RUN} "
            "FROM threads t WHERE user_id = %s AND deleted_at IS NULL ORDER BY updated_at DESC",
            (user_id,),
        )
        return await cur.fetchall()


async def get_thread(
    pool: Pool, thread_id: UUID, user_id: str
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT thread_id, graph, title, created_at, updated_at, {_ACTIVE_RUN} "
            "FROM threads t WHERE thread_id = %s AND user_id = %s AND deleted_at IS NULL",
            (thread_id, user_id),
        )
        return await cur.fetchone()


async def rename_thread(pool: Pool, thread_id: UUID, user_id: str, title: str) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE threads SET title = %s, updated_at = now() "
            "WHERE thread_id = %s AND user_id = %s AND deleted_at IS NULL",
            (title, thread_id, user_id),
        )
        return cur.rowcount == 1


async def delete_thread(pool: Pool, thread_id: UUID, user_id: str) -> bool:
    """Hides the thread and stops its runs; purge_thread then deletes it."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "UPDATE threads SET deleted_at = now() "
            "WHERE thread_id = %s AND user_id = %s AND deleted_at IS NULL",
            (thread_id, user_id),
        )
        if cur.rowcount != 1:
            return False
        cur = await conn.execute(
            "UPDATE runs SET status = 'cancelling' "
            "WHERE thread_id = %s AND status IN ('pending', 'running') RETURNING run_id",
            (thread_id,),
        )
        for row in await cur.fetchall():
            await notify(conn, "run_cancel", str(row["run_id"]))
        # A cancelled pending run still needs a worker to finalize it.
        await notify(conn, "runs")
        return True


async def purge_thread(
    pool: Pool, thread_id: UUID, checkpointer: BaseCheckpointSaver
) -> bool:
    """Deletes a deleted thread with its runs, events and checkpoints, once no run is open."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "SELECT 1 FROM threads WHERE thread_id = %s AND deleted_at IS NOT NULL "
            "FOR UPDATE",
            (thread_id,),
        )
        if await cur.fetchone() is None:
            return False
        cur = await conn.execute(
            "SELECT 1 FROM runs WHERE thread_id = %s "
            "AND status IN ('pending', 'running', 'cancelling') LIMIT 1",
            (thread_id,),
        )
        if await cur.fetchone() is not None:
            return False
        # Checkpoints first: if this fails, the thread row is still there to retry from.
        await checkpointer.adelete_thread(str(thread_id))
        await conn.execute("DELETE FROM threads WHERE thread_id = %s", (thread_id,))
        return True


async def create_run(pool: Pool, thread_id: UUID, content: str) -> UUID | None:
    """Cancels the thread's active run, if any, and queues a new one. None if the thread is gone."""
    run_id = uuid4()
    async with pool.connection() as conn, conn.transaction():
        # Serializes run creation per thread.
        cur = await conn.execute(
            "SELECT 1 FROM threads WHERE thread_id = %s AND deleted_at IS NULL FOR UPDATE",
            (thread_id,),
        )
        if await cur.fetchone() is None:
            return None
        cur = await conn.execute(
            "UPDATE runs SET status = 'cancelling' "
            "WHERE thread_id = %s AND status IN ('pending', 'running') RETURNING run_id",
            (thread_id,),
        )
        for row in await cur.fetchall():
            await notify(conn, "run_cancel", str(row["run_id"]))
        await conn.execute(
            "INSERT INTO runs (run_id, thread_id, input) VALUES (%s, %s, %s)",
            (run_id, thread_id, Jsonb({"content": content})),
        )
        await conn.execute(
            "UPDATE threads SET updated_at = now() WHERE thread_id = %s", (thread_id,)
        )
        await notify(conn, "runs")
    return run_id


async def get_run(pool: Pool, run_id: UUID, user_id: str) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT r.run_id, r.thread_id, r.status, r.error, r.created_at, r.finished_at "
            "FROM runs r JOIN threads t USING (thread_id) "
            "WHERE r.run_id = %s AND t.user_id = %s AND t.deleted_at IS NULL",
            (run_id, user_id),
        )
        return await cur.fetchone()


async def run_status(pool: Pool, run_id: UUID) -> RunStatus | None:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT status FROM runs WHERE run_id = %s", (run_id,))
        row = await cur.fetchone()
        return RunStatus(row["status"]) if row else None


async def cancel_run(pool: Pool, run_id: UUID, user_id: str) -> bool:
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "UPDATE runs r SET status = 'cancelling' FROM threads t "
            "WHERE r.thread_id = t.thread_id AND t.user_id = %s AND t.deleted_at IS NULL "
            "AND r.run_id = %s AND r.status IN ('pending', 'running') RETURNING r.run_id",
            (user_id, run_id),
        )
        if await cur.fetchone() is None:
            return False
        await notify(conn, "run_cancel", str(run_id))
        # A cancelled pending run still needs a worker to finalize it.
        await notify(conn, "runs")
        return True


# --- worker side ---


async def claim_run(
    pool: Pool, lease_seconds: float, graphs: list[str]
) -> dict[str, Any] | None:
    """Takes the oldest claimable run: pending, or open with an expired lease.

    A run waits while an older run of the same thread is still open. Only runs of
    threads whose graph this worker has are claimed.
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            UPDATE runs SET
                status = CASE WHEN status = 'pending' THEN 'running' ELSE status END,
                attempt = attempt + 1,
                locked_until = now() + make_interval(secs => %(lease)s)
            WHERE run_id = (
                SELECT r.run_id FROM runs r
                WHERE (
                    r.status = 'pending'
                    OR (r.status IN ('running', 'cancelling')
                        AND (r.locked_until IS NULL OR r.locked_until < now()))
                )
                AND NOT EXISTS (
                    SELECT 1 FROM runs o
                    WHERE o.thread_id = r.thread_id AND o.created_at < r.created_at
                    AND o.status IN ('running', 'cancelling')
                )
                AND (SELECT graph FROM threads t WHERE t.thread_id = r.thread_id) = ANY(%(graphs)s)
                ORDER BY r.created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING run_id, thread_id, input, status, attempt,
                (SELECT graph FROM threads t WHERE t.thread_id = runs.thread_id) AS graph,
                (SELECT user_id FROM threads t WHERE t.thread_id = runs.thread_id) AS user_id,
                (SELECT title FROM threads t WHERE t.thread_id = runs.thread_id) AS title
            """,
            {"lease": lease_seconds, "graphs": graphs},
        )
        return await cur.fetchone()


async def heartbeat(
    pool: Pool, run_id: UUID, attempt: int, lease_seconds: float
) -> RunStatus | None:
    """Extends the lease. Returns the run status, or None if the lease was lost."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE runs SET locked_until = now() + make_interval(secs => %s) "
            "WHERE run_id = %s AND attempt = %s AND status IN ('running', 'cancelling') "
            "RETURNING status",
            (lease_seconds, run_id, attempt),
        )
        row = await cur.fetchone()
        return RunStatus(row["status"]) if row else None


async def finish_run(
    pool: Pool,
    run_id: UUID,
    attempt: int,
    status: RunStatus,
    error: str | None = None,
) -> bool:
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "UPDATE runs SET status = %s, error = %s, finished_at = now(), locked_until = NULL "
            "WHERE run_id = %s AND attempt = %s AND status IN ('running', 'cancelling') "
            "RETURNING thread_id",
            (status, error, run_id, attempt),
        )
        row = await cur.fetchone()
        if row is None:
            return False
        await conn.execute(
            "UPDATE threads SET updated_at = now() WHERE thread_id = %s",
            (row["thread_id"],),
        )
        # The thread's next run can start now.
        await notify(conn, "runs")
        return True


async def retry_run(
    pool: Pool, run_id: UUID, attempt: int, delay_seconds: float
) -> None:
    """Hands the run back; any worker may claim it again after the delay."""
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE runs SET locked_until = now() + make_interval(secs => %s) "
            "WHERE run_id = %s AND attempt = %s",
            (delay_seconds, run_id, attempt),
        )


async def set_title(pool: Pool, thread_id: UUID, title: str) -> None:
    """Only fills an empty title, so a rename always wins."""
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE threads SET title = %s WHERE thread_id = %s AND title IS NULL",
            (title, thread_id),
        )


async def trim_events(pool: Pool, older_than_hours: float) -> int:
    """Deletes the events of runs that ended long enough ago."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM run_events e USING runs r WHERE e.run_id = r.run_id "
            "AND r.finished_at < now() - make_interval(secs => %s)",
            (older_than_hours * 3600,),
        )
        return cur.rowcount


async def release_run(pool: Pool, run_id: UUID, attempt: int) -> None:
    """Gives the run back on shutdown so another worker picks it up at once."""
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE runs SET locked_until = NULL WHERE run_id = %s AND attempt = %s",
            (run_id, attempt),
        )
        await notify(conn, "runs")
