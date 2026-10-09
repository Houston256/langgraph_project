-- A deleted thread with a run still open: hidden, and removed once the run stops.

ALTER TABLE threads ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
