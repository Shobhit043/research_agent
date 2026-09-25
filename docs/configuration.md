# Configuration

All settings are environment variables, usually set in `.env` (copy `.env.example`). Only
`GROQ_API_KEY` is required.

## Model

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — (required) | Groq API key. Without it, `main.py` exits with a message. |
| `AGENT_MODEL` | `openai/gpt-oss-20b` | Groq model used for routing and answering. |

## Retrieval and tools

| Variable | Default | Description |
|---|---|---|
| `USE_EMBEDDINGS` | `true` | Hybrid BM25 + embedding search. `false` uses BM25 only (no model download). |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model. Changing it only affects newly uploaded documents. |
| `FASTEMBED_CACHE_PATH` | fastembed's default | Where the embedding model is stored. The Docker image sets `/opt/models`. |
| `ENABLED_TOOLS` | all | Comma-separated subset of `search_documents, read_document, web_search, fetch_url, wikipedia_search, arxiv_search, get_time, place_info`. A typo fails at startup. Each bound tool costs about 150 prompt tokens per model call. |
| `HTTP_USER_AGENT` | `ResearchAssistant/0.4 (self-hosted research tool)` | Sent by the web tools. Include a contact URL or email: Wikipedia and arXiv throttle anonymous clients. |

## Server

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Listen address. Docker sets `0.0.0.0`. The server warns if it listens publicly without `APP_API_KEYS`. |
| `PORT` | `8000` | Listen port. |
| `APP_API_KEYS` | none (auth off) | Comma-separated keys. When set, every endpoint except `/api/health` needs `Authorization: Bearer <key>`, and each session is private to the key that created it. |
| `CORS_ORIGINS` | none | Comma-separated origins allowed to call the API from other sites. Not needed for the bundled UI. |
| `LOG_FORMAT` | `text` | `json` for one JSON object per line (Loki, CloudWatch, Datadog). |

Command-line flags for `main.py`: `--host`, `--port`, `--no-browser` (don't open a tab), and
`-v`/`--verbose` (log routing decisions, tool calls and timings).

## Storage

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | none (SQLite) | `postgresql://user:password@host:5432/db` switches to PostgreSQL. |
| `DATA_DIR` | `data` | SQLite location (`DATA_DIR/assistant.db`). Unused with Postgres. |
| `SESSION_TTL_DAYS` | `7` | Sessions idle for longer are deleted on startup. |
| `POSTGRES_PASSWORD` | `assistant` | Used only by `docker-compose.yml` for its database service. Change it. |

## Limits

| Variable | Default | Description |
|---|---|---|
| `CHAT_PER_MINUTE` | `10` | Messages per session per minute. Uploads are limited to 20 per session per minute, and new sessions to 10 per IP per minute. |
| `SESSION_TOKEN_BUDGET` | `300000` | Total model tokens a session may use before chat returns `429`. |
| `MAX_UPLOAD_MB` | `20` | Maximum upload size. The UI checks it too. |

## Tracing (optional)

LangChain sends traces to [LangSmith](https://smith.langchain.com) automatically when these are
set. Each run is tagged with its trace ID and session ID.

| Variable | Example |
|---|---|
| `LANGSMITH_TRACING` | `true` |
| `LANGSMITH_API_KEY` | your key |
| `LANGSMITH_PROJECT` | `research-assistant` |

## Tuning in code

Less common knobs live in [`agent/config.py`](../agent/config.py) `Settings`:

| Setting | Default | Notes |
|---|---|---|
| `max_tool_iterations` | 5 | Tool rounds per answer before a final answer is forced |
| `tool_timeout` | 30 s | Per tool call |
| `history_token_budget` | 3,500 | History is trimmed to this, to stay under Groq's 8K-per-request limit |
| `max_output_tokens` | 2,048 | Cap on the model's answer length |
| `chunk_size` / `chunk_overlap` | 1,000 / 150 characters | Document chunking |
| `doc_search_k` | 4 | Passages returned per document search |
| `max_web_results` | 3 | Pages read per web search |
