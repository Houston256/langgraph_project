"""Applies the NNN_*.sql files of this package in order, each once, recorded in schema_migrations."""

import logging
import re
from importlib.resources import files
from importlib.resources.abc import Traversable

from psycopg import AsyncConnection

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"\d{3}_.*\.sql")
PACKAGE = files(__name__)


def migration_files(directory: Traversable = PACKAGE) -> list[Traversable]:
    return sorted(
        (f for f in directory.iterdir() if NAME_RE.fullmatch(f.name)),
        key=lambda f: f.name,
    )


async def migrate(conn: AsyncConnection, directory: Traversable = PACKAGE) -> list[str]:
    """Runs the files not applied yet; returns their names. The caller holds the lock."""
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    cur = await conn.execute("SELECT name FROM schema_migrations")
    done = {row["name"] for row in await cur.fetchall()}
    applied = []
    for path in migration_files(directory):
        if path.name in done:
            continue
        async with conn.transaction():
            # prepare=False: a file holds several statements, which prepared statements can't.
            await conn.execute(path.read_text(encoding="utf-8"), prepare=False)  # type: ignore[arg-type]
            await conn.execute(
                "INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,)
            )
        log.info("applied migration %s", path.name)
        applied.append(path.name)
    return applied
