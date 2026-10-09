import asyncio
import logging

from assistant.config import settings
from assistant.db import Pool
from assistant.runs import store

log = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 3600


async def trim_events_forever(pool: Pool) -> None:
    """Deletes old run events every hour. Idempotent, so several workers doing it is fine."""
    while True:
        try:
            if n := await store.trim_events(pool, settings.event_retention_hours):
                log.info("deleted %s old run events", n)
        except Exception:
            log.exception("event cleanup failed")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
