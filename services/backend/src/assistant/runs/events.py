import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, Self
from uuid import UUID

from psycopg.types.json import Jsonb

from assistant.db import Listener, Pool, notify
from assistant.runs.store import LeaseLost

Event = dict[str, Any]

READ_PAGE = 500


def is_end(event: Event) -> bool:
    return event.get("type") == "run-end"


class EventLog(Protocol):
    async def append(
        self, run_id: UUID, attempt: int, events: list[tuple[int, Event]]
    ) -> None:
        """Raises LeaseLost if the run has moved on to another attempt."""
        ...

    def read(
        self, run_id: UUID, after: int, idle_seconds: float
    ) -> AsyncIterator[tuple[int, Event] | None]:
        """Events with seq > after, then new ones as they come, until run-end.

        Yields None after idle_seconds without events.
        """
        ...

    async def last_seq(self, run_id: UUID) -> int: ...

    async def all(self, run_id: UUID) -> list[Event]: ...

    async def trim(self, run_id: UUID) -> None: ...


class PgEventLog:
    def __init__(self, pool: Pool, listener: Listener):
        self.pool = pool
        self.listener = listener

    async def append(
        self, run_id: UUID, attempt: int, events: list[tuple[int, Event]]
    ) -> None:
        if not events:
            return
        async with self.pool.connection() as conn, conn.transaction():
            # The row lock makes a new claim of the run wait until this write is done.
            cur = await conn.execute(
                "SELECT 1 FROM runs WHERE run_id = %s AND attempt = %s FOR SHARE",
                (run_id, attempt),
            )
            if await cur.fetchone() is None:
                raise LeaseLost
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO run_events (run_id, seq, data) VALUES (%s, %s, %s)",
                    [(run_id, seq, Jsonb(data)) for seq, data in events],
                )
            await notify(conn, "run_events", str(run_id))

    async def read(
        self, run_id: UUID, after: int, idle_seconds: float
    ) -> AsyncIterator[tuple[int, Event] | None]:
        while True:
            # Subscribe before reading so a write between the read and the wait is not missed.
            with self.listener.subscribe("run_events", str(run_id)) as woke:
                async with self.pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT seq, data FROM run_events WHERE run_id = %s AND seq > %s "
                        "ORDER BY seq LIMIT %s",
                        (run_id, after, READ_PAGE),
                    )
                    rows = await cur.fetchall()
                for row in rows:
                    after = row["seq"]
                    yield after, row["data"]
                    if is_end(row["data"]):
                        return
                if len(rows) == READ_PAGE:
                    continue
                try:
                    await asyncio.wait_for(woke.wait(), idle_seconds)
                except TimeoutError:
                    yield None

    async def last_seq(self, run_id: UUID) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT coalesce(max(seq), 0) AS seq FROM run_events WHERE run_id = %s",
                (run_id,),
            )
            row = await cur.fetchone()
            return row["seq"]  # type: ignore[index]

    async def all(self, run_id: UUID) -> list[Event]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT data FROM run_events WHERE run_id = %s ORDER BY seq", (run_id,)
            )
            return [row["data"] for row in await cur.fetchall()]

    async def trim(self, run_id: UUID) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM run_events WHERE run_id = %s", (run_id,))


class Emitter:
    """Writes one run's events, merging text deltas for flush_interval seconds.

    flush_interval=0 writes every event at once.
    """

    def __init__(
        self,
        log: EventLog,
        run_id: UUID,
        attempt: int,
        last_seq: int,
        flush_interval: float,
    ):
        self.log = log
        self.run_id = run_id
        self.attempt = attempt
        self.seq = last_seq
        self.flush_interval = flush_interval
        self._pending: list[tuple[int, Event]] = []
        self._text: Event | None = None
        self._text_since = 0.0
        self._lock = asyncio.Lock()
        self._ticker: asyncio.Task | None = None

    async def __aenter__(self) -> Self:
        if self.flush_interval > 0:
            self._ticker = asyncio.create_task(self._tick())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ticker:
            self._ticker.cancel()
        # Text still buffered when the run is interrupted is written too.
        await self.flush()

    async def emit(self, event: Event) -> None:
        async with self._lock:
            if event["type"] == "text-delta" and self.flush_interval > 0:
                if self._text and self._text["message_id"] == event["message_id"]:
                    self._text["delta"] += event["delta"]
                else:
                    self._stage_text()
                    self._text = dict(event)
                    self._text_since = time.monotonic()
                if time.monotonic() - self._text_since < self.flush_interval:
                    return
                self._stage_text()
            else:
                self._stage_text()
                self._stage(event)
            await self._write()

    def drop(self) -> None:
        """Discards buffered events."""
        self._pending = []
        self._text = None

    async def flush(self) -> None:
        async with self._lock:
            self._stage_text()
            await self._write()

    def _stage(self, event: Event) -> None:
        self.seq += 1
        self._pending.append((self.seq, event))

    def _stage_text(self) -> None:
        if self._text:
            self._stage(self._text)
            self._text = None

    async def _write(self) -> None:
        if self._pending:
            events, self._pending = self._pending, []
            await self.log.append(self.run_id, self.attempt, events)

    async def _tick(self) -> None:
        # Writes buffered text when the model pauses between deltas.
        while True:
            await asyncio.sleep(self.flush_interval)
            if (
                self._text
                and time.monotonic() - self._text_since >= self.flush_interval
            ):
                await self.flush()
