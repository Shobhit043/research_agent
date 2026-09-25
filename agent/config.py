import os
from dataclasses import dataclass


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    max_retries: int = 3
    max_output_tokens: int = 2048

    max_tool_iterations: int = 5
    tool_timeout: float = 30.0
    # Groq's free tier caps a single request at 8K tokens, so history must stay well under it.
    history_token_budget: int = 3500

    max_web_results: int = 3
    web_page_chars: int = 1500
    fetch_page_chars: int = 4000
    request_timeout: float = 8.0

    chunk_size: int = 1000
    chunk_overlap: int = 150
    doc_search_k: int = 4

    use_embeddings: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Empty means every tool. Each bound tool costs ~150 prompt tokens per model call.
    enabled_tools: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("AGENT_MODEL", cls.model),
            use_embeddings=env_bool("USE_EMBEDDINGS", cls.use_embeddings),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            enabled_tools=tuple(t.strip() for t in os.getenv("ENABLED_TOOLS", "").split(",") if t.strip()),
        )
