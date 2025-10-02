from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    langfuse_public_key: str
    langfuse_secret_key: SecretStr
    langfuse_host: str
    openai_api_key: SecretStr
    model_name: str
    qdrant_url_grpc: str
    qdrant_api_key: SecretStr


settings = Settings() # type: ignore