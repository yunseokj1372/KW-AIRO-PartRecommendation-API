import pydantic_settings

class Settings(pydantic_settings.BaseSettings):
    MODELS_PATH: str
    POD_JSONS_PATH: str
    INSTRUCTIONS_PATH: str
    SEAL_PATH: str
    SECRET_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()