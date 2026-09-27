# Development

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
python main.py -v              # verbose: logs routing, tool calls and timings
```

## Project layout

```
agent/                     the research agent; no web dependencies
  research_agent.py        LangGraph StateGraph: analyze, direct, research, tools, finalize nodes
  prompts.py               system and routing prompts
  schemas.py               QueryAnalysis, TurnResult, AgentEvent and related models
  citations.py             citation verification and source collection
  guardrails.py            untrusted-content fencing, injection detection
  config.py                Settings (agent-level configuration)
  documents/
    loader.py              PDF/TXT/MD extraction, cleaning, page-bounded chunking
    store.py               hybrid BM25 + dense index with reciprocal rank fusion
    embeddings.py          fastembed wrapper
    ocr.py                 RapidOCR engine for scanned pages and images
  tools/
    __init__.py            tool registry, ENABLED_TOOLS, prompt guidance
    http.py                shared HTTP: SSRF guard, redirect checks, retries, TTL cache
    document_tools.py      search_documents, read_document
    web_tools.py           web_search, fetch_url
    knowledge_tools.py     wikipedia_search, arxiv_search
    place_tools.py         get_time, place_info
web/
  server.py                FastAPI app: routes, streaming, middleware
  sessions.py              LRU cache of live agents, rebuilt from storage
  storage.py               SQLite and PostgreSQL (+ pgvector) backends
  security.py              API-key auth, rate limiting, security headers
  observability.py         request IDs, JSON logging, Prometheus metrics
  config.py                ServerSettings
  static/                  single-page UI: index.html, app.js, styles.css
eval/                      corpus, dataset, metrics, LLM judge, runner, results
tests/                     offline test suite
main.py                    entry point
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, pyproject.toml
```

## Tests

```bash
python -m pytest                     # 155 tests, offline, about 15 s
ruff check .                         # lint

# Also run the API and pgvector tests against PostgreSQL:
docker run -d --name pg -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=assistant_test -p 5432:5432 pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://test:test@localhost:5432/assistant_test python -m pytest
```

Nothing touches the network or needs an API key. The fakes are in [`tests/fakes.py`](../tests/fakes.py):
- `ScriptedLLM` replays scripted replies and streams them the way Groq does, including tool-call
  chunks, usage metadata and errors raised mid-script.
- `KeywordEmbedder` is a deterministic embedder in which synonyms share a vector.
- The HTTP tests swap in a fake session and DNS resolver.
- An autouse `fake_ocr` fixture (in `conftest.py`) replaces the OCR engine, so unit tests never
  load OCR models. One test marked `real_ocr` runs the real engine on a generated scan (~4 s).

| File | Covers |
|---|---|
| `test_agent.py` | Graph topology, routing, tool loop, parallel calls, duplicates, timeouts, budget exhaustion, gpt-oss error recovery, streaming order, commit-on-completion and rollback on disconnect, history trimming, injection flags |
| `test_tools.py` | SSRF blocking (including redirects), fetch, Wikipedia, arXiv, time and place, document reading, the registry |
| `test_documents.py` | Loaders and error cases, OCR (scanned pages, page cap, images, real OCR), chunking, BM25, hybrid search, vector persistence |
| `test_citations.py` | Citation verification, bracket normalisation, source collection |
| `test_api.py` | Every endpoint, on SQLite and on PostgreSQL: uploads (including OCR'd images), streaming, auth and ownership, rate limits, token budget, persistence across restart, migration, health, metrics; with pgvector, dense search from the database, vector backfill, and the dimension-mismatch fallback |
| `test_eval.py` | Evaluation metrics, report rendering, dataset integrity |

CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs lint and the full suite
against a Postgres service container, then builds the Docker image and smoke-tests `/api/health`.

## Adding a tool

1. **Write a builder** in `agent/tools/`, returning a LangChain `@tool`. Its docstring is what the
   model reads, so say when to use the tool.

   ```python
   def build_weather_tool(settings: Settings) -> BaseTool:
       @tool
       def weather(location: str) -> str:
           """Current weather for a place. Use for questions about weather right now.

           Args:
               location: A place name, e.g. "Oslo".
           """
           ...
           return f"[X1] Weather in {name}\nURL: {source_url}\n{summary}"
       return weather
   ```

2. **Fetch over HTTP only through `tools/http.py`** (`safe_get`, `cache`), so the SSRF guard,
   size cap, retries and cache apply.
3. **Return citable output.** Documents use a `[file p.N]` header line. Web-like sources use
   `[<prefix><n>] <title>` followed by `URL: <url>`. Using a new prefix letter? Add it to the
   regexes in `agent/citations.py` and `CITATION` in `web/static/app.js`, and to `prefixOf` in
   `webCitationLinks` so badges link.
4. **Register it** in `agent/tools/__init__.py`: add a line to `GUIDANCE` (what the prompt says
   about it) and to `builders`. If it only makes sense with uploads, add it to `DOCUMENT_TOOLS`.
5. **Label its progress step** in the UI's `TOOL_LABELS`, and, if its main argument isn't
   `query`, `url`, `name` or `location`, in `TOOL_SUBJECT`.
6. **Test it offline** in `tests/test_tools.py` with the `web` fixture's canned responses.

Keep the toolset small: every bound tool adds about 150 tokens to each model call, and Groq's free
tier allows about 8K per request.

## Conventions

- **Errors inside tools become tool output** (`"Error: …"`) so the model can recover. The agent
  catches exceptions and timeouts, so tools don't need their own broad `try` blocks.
- **Never call `requests` directly** from tools; use `safe_get`.
- **Keep the agent independent of the web layer.** `agent/` must not import `web/`.
- **Keep comments rare.** Explain *why* when it isn't obvious (a workaround, an invariant),
  never *what*.
- Formatting and lint rules are in `pyproject.toml` (`ruff`, line length 120).
