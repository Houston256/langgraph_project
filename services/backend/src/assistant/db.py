import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from assistant.config import settings
from assistant.migrations import migrate

log = logging.getLogger(__name__)

Pool = AsyncConnectionPool[AsyncConnection[DictRow]]

# The checkpointer runs its own transactions and reads rows as dicts.
CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

# Any constant works; it only has to be the same in every process.
SCHEMA_LOCK_ID = 7_431_001


@asynccontextmanager
async def open_db(max_size: int = 10) -> AsyncIterator[tuple[Pool, AsyncPostgresSaver]]:
    """Pool and checkpointer, with migrations and checkpointer tables applied."""
    pool = Pool(
        conninfo=settings.postgres_url,
        min_size=1,
        max_size=max_size,
        open=False,
        kwargs=CONNECTION_KWARGS,
        # Drops connections that died, e.g. when postgres restarted.
        check=Pool.check_connection,
    )
    async with pool:
        checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        # api and worker start together; the lock keeps their setup from racing.
        async with pool.connection() as conn:
            await conn.execute("SELECT pg_advisory_lock(%s)", (SCHEMA_LOCK_ID,))
            try:
                await migrate(conn)
                await checkpointer.setup()
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (SCHEMA_LOCK_ID,))
        yield pool, checkpointer


async def notify(conn: AsyncConnection, channel: str, payload: str = "") -> None:
    await conn.execute("SELECT pg_notify(%s, %s)", (channel, payload))


class Listener:
    """One LISTEN connection per process that wakes local waiters."""

    def __init__(self, channels: list[str]):
        self.channels = channels
        self._waiters: dict[tuple[str, str | None], set[asyncio.Event]] = {}

    @contextmanager
    def subscribe(
        self, channel: str, payload: str | None = None
    ) -> Iterator[asyncio.Event]:
        """Event set on a matching notification. payload=None matches any payload."""
        event = asyncio.Event()
        key = (channel, payload)
        self._waiters.setdefault(key, set()).add(event)
        try:
            yield event
        finally:
            self._waiters[key].discard(event)
            if not self._waiters[key]:
                del self._waiters[key]

    def _wake(self, channel: str, payload: str | None) -> None:
        for event in self._waiters.get((channel, payload), ()):
            event.set()

    def _wake_all(self) -> None:
        for events in self._waiters.values():
            for event in events:
                event.set()

    async def run(self) -> None:
        while True:
            try:
                async with await AsyncConnection.connect(
                    settings.postgres_url, autocommit=True
                ) as conn:
                    for channel in self.channels:
                        await conn.execute(f'LISTEN "{channel}"')
                    # Notifications sent while disconnected are lost, so waiters re-check the db.
                    self._wake_all()
                    async for n in conn.notifies():
                        self._wake(n.channel, n.payload)
                        self._wake(n.channel, None)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("listener connection lost, reconnecting")
                self._wake_all()
                await asyncio.sleep(1)
