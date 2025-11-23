from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    sentinelhub_client_id: str
    sentinelhub_client_secret: str

    class Config:
        env_file = ".env"

settings = Settings()
