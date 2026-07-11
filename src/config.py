import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_DOT_ENV_PATH: Path = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_DOT_ENV_PATH)


class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_DIR: Path
    FAQ_PATH: Path
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1024

    def __init__(self) -> None:
        load_dotenv(dotenv_path=_DOT_ENV_PATH, override=True)
        api_key: str | None = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"GROQ_API_KEY is not set. "
                f"Create a .env file at {_DOT_ENV_PATH} with GROQ_API_KEY=gsk_your_key"
            )
        self.GROQ_API_KEY = api_key
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", self.GROQ_MODEL)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.FAISS_INDEX_DIR = _PROJECT_ROOT / "data" / "faiss_index"
        self.FAQ_PATH = _PROJECT_ROOT / "data" / "gigacorp_faq.txt"

        raw_chunk_size = os.getenv("CHUNK_SIZE")
        if raw_chunk_size is not None:
            self.CHUNK_SIZE = int(raw_chunk_size)

        raw_chunk_overlap = os.getenv("CHUNK_OVERLAP")
        if raw_chunk_overlap is not None:
            self.CHUNK_OVERLAP = int(raw_chunk_overlap)

        raw_temperature = os.getenv("TEMPERATURE")
        if raw_temperature is not None:
            self.TEMPERATURE = float(raw_temperature)

        raw_max_tokens = os.getenv("MAX_TOKENS")
        if raw_max_tokens is not None:
            self.MAX_TOKENS = int(raw_max_tokens)


settings = Settings()
