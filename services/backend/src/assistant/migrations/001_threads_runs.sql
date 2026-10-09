-- Threads, the run queue and each run's event stream.

CREATE TABLE IF NOT EXISTS threads (
    thread_id  uuid PRIMARY KEY,
    user_id    text NOT NULL,
    graph      text NOT NULL,
    title      text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS threads_user_idx ON threads (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    run_id       uuid PRIMARY KEY,
    thread_id    uuid NOT NULL REFERENCES threads ON DELETE CASCADE,
    input        jsonb NOT NULL,
    status       text NOT NULL DEFAULT 'pending',
    attempt      int NOT NULL DEFAULT 0,
    locked_until timestamptz,
    error        text,
    created_at   timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at  timestamptz
);
-- At most one waiting or running run per thread; cancelling runs don't count.
CREATE UNIQUE INDEX IF NOT EXISTS runs_one_active_idx
    ON runs (thread_id) WHERE status IN ('pending', 'running');
CREATE INDEX IF NOT EXISTS runs_open_idx
    ON runs (created_at) WHERE status IN ('pending', 'running', 'cancelling');

CREATE TABLE IF NOT EXISTS run_events (
    run_id uuid NOT NULL REFERENCES runs ON DELETE CASCADE,
    seq    bigint NOT NULL,
    data   jsonb NOT NULL,
    PRIMARY KEY (run_id, seq)
);
