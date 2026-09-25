import os
from dataclasses import dataclass
from pathlib import Path


def _csv(name: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.getenv(name, "").split(",") if v.strip())


@dataclass(frozen=True)
class ServerSettings:
    data_dir: Path = Path("data")
    # postgresql://user:pass@host:5432/db. Unset means SQLite at data_dir/assistant.db.
    database_url: str | None = None
    # Empty means no authentication (local use). Set APP_API_KEYS for any shared deployment.
    api_keys: tuple[str, ...] = ()
    cors_origins: tuple[str, ...] = ()

    max_upload_bytes: int = 20 * 1024 * 1024
    max_cached_sessions: int = 50
    session_ttl_days: int = 7

    chat_per_minute: int = 10
    uploads_per_minute: int = 20
    sessions_per_minute: int = 10
    session_token_budget: int = 300_000

    log_format: str = "text"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "assistant.db"

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            database_url=os.getenv("DATABASE_URL") or None,
            api_keys=_csv("APP_API_KEYS"),
            cors_origins=_csv("CORS_ORIGINS"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
            session_ttl_days=int(os.getenv("SESSION_TTL_DAYS", "7")),
            chat_per_minute=int(os.getenv("CHAT_PER_MINUTE", "10")),
            session_token_budget=int(os.getenv("SESSION_TOKEN_BUDGET", "300000")),
            log_format=os.getenv("LOG_FORMAT", "text"),
        )
