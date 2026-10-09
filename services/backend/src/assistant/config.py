from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )
    # openai-compatible api; set the base url to use OpenRouter
    openai_api_key: SecretStr
    openai_base_url: str | None = None
    model_name: str
    summary_model_name: str = "gpt-5-nano"
    embedding_model: str = "text-embedding-3-large"
    # qdrant
    qdrant_url_grpc: str
    qdrant_api_key: SecretStr
    qdrant_host: str
    qdrant_port: int
    qdrant_grpc_port: int
    # postgres
    postgres_url: str
    # runs
    # runs at once per worker; they mostly wait on the model, so this can be high
    worker_concurrency: int = 64
    run_lease_seconds: float = 30
    run_max_attempts: int = 5
    # 0 writes every text delta as its own event
    flush_interval_ms: int = 50
    # on SIGTERM, time running runs get to finish; keep below compose stop_grace_period
    shutdown_grace_seconds: float = 100
    max_input_chars: int = 4000
    # per model call, reasoning included
    max_output_tokens: int = 1500
    # answers may only link to these domains and their subdomains; other links lose their URL
    allowed_link_domains: list[str]
    # relative to the working directory: services/backend locally, /app in docker
    system_prompt_file: str = "prompts/shopping-assistant.md"
    # spending limits, unset = off; cost from OpenRouter prices
    limit_global_usd_per_month: float | None = 20
    limit_user_usd_per_week: float | None = None
    limit_user_runs_per_hour: int | None = None
    # provider errors worth retrying (rate limits, credit holds, 5xx) are retried this often
    run_retry_max_seconds: float = 60
    # finished runs' events are deleted after this
    event_retention_hours: float = 24
    # product feed sync
    catalog_feed_url: str
    catalog_sync_time: str = "03:33"
    catalog_sync_tz: str = "UTC"
    # identity: trusted header set by the proxy in front, else one fixed user
    identity_header: str = "Cf-Access-Authenticated-User-Email"
    default_user_id: str = "local"


settings = Settings()  # type: ignore
