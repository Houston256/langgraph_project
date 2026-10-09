import psycopg
import pytest
from psycopg.rows import dict_row

from assistant.config import settings
from assistant.migrations import migrate, migration_files

pytestmark = pytest.mark.db


async def test_applies_new_files_once_in_order(tmp_path):
    (tmp_path / "001_create.sql").write_text(
        "CREATE TABLE t (x int);\nINSERT INTO t VALUES (1);"
    )
    (tmp_path / "002_more.sql").write_text(
        "INSERT INTO t VALUES (2); INSERT INTO t VALUES (3);"
    )
    (tmp_path / "notes.sql").write_text("this is not a migration")
    async with await psycopg.AsyncConnection.connect(
        settings.postgres_url, autocommit=True, row_factory=dict_row
    ) as conn:
        await conn.execute("CREATE SCHEMA pytest_migrations")
        try:
            await conn.execute("SET search_path TO pytest_migrations")
            assert await migrate(conn, tmp_path) == ["001_create.sql", "002_more.sql"]
            assert await migrate(conn, tmp_path) == []
            cur = await conn.execute("SELECT array_agg(x ORDER BY x) AS xs FROM t")
            assert (await cur.fetchone())["xs"] == [1, 2, 3]
        finally:
            await conn.execute("DROP SCHEMA pytest_migrations CASCADE")


async def test_repo_migrations_are_applied():
    names = [f.name for f in migration_files()]
    async with await psycopg.AsyncConnection.connect(
        settings.postgres_url, autocommit=True, row_factory=dict_row
    ) as conn:
        await migrate(conn)
        cur = await conn.execute("SELECT name FROM schema_migrations ORDER BY name")
        assert [r["name"] for r in await cur.fetchall()] == names
