from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    # langfuse
    langfuse_public_key: str
    langfuse_secret_key: SecretStr
    langfuse_host: str
    # openai
    openai_api_key: SecretStr
    model_name: str
    # qdrant
    qdrant_url_grpc: str
    qdrant_api_key: SecretStr
    qdrant_host: str
    qdrant_port: int
    qdrant_grpc_port: int


settings = Settings()  # type: ignore
