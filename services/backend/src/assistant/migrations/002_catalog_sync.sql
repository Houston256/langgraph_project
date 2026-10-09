-- Last product feed sync; one row.

CREATE TABLE IF NOT EXISTS catalog_sync (
    id            int PRIMARY KEY,
    last_modified text,
    synced_at     timestamptz NOT NULL,
    stats         jsonb
);
