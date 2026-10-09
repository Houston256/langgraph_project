-- One row per model call, for the spending limits.

CREATE TABLE IF NOT EXISTS usage (
    id            bigserial PRIMARY KEY,
    run_id        uuid,
    user_id       text NOT NULL,
    model         text NOT NULL,
    input_tokens  int NOT NULL,
    cached_tokens int NOT NULL,
    output_tokens int NOT NULL,
    -- NULL when the model had no known price
    cost_usd      numeric,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_user_idx ON usage (user_id, created_at);
CREATE INDEX IF NOT EXISTS usage_time_idx ON usage (created_at);
