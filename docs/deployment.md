# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env        # set GROQ_API_KEY, APP_API_KEYS, POSTGRES_PASSWORD, HTTP_USER_AGENT
docker compose up -d --build
curl localhost:8000/api/health
# {"status": "ok", "database": "postgres", "auth_required": true, "hybrid_search": true}
```

This starts two services:

| Service | Image | Notes |
|---|---|---|
| `assistant` | built from `Dockerfile` | Waits for the database to be healthy; JSON logs; restarts automatically |
| `db` | `pgvector/pgvector:pg16` | PostgreSQL 16 with pgvector; data in the `postgres-data` volume; `pg_isready` healthcheck |

About the image:
- It runs as a non-root user (uid 10001).
- The embedding and OCR models are baked in, so containers start without downloading them.
- Its healthcheck calls `/api/health`, which fails if the database is down.
- It's about 1.5 GB: OpenCV and the OCR models (~600 MB) plus the ONNX runtime and embedding model.
  To slim it, remove `rapidocr_onnxruntime` and `pypdfium2` from `requirements.txt` and the OCR
  line from the `Dockerfile`; uploads then work for text-based files only.

To use an existing Postgres, drop the `db` service and set `DATABASE_URL`. The schema is created
and migrated automatically on startup, and it's safe for several instances to start at once.
For server-side vector search, the database needs the [pgvector](https://github.com/pgvector/pgvector)
extension (available on most managed services: AWS RDS, Cloud SQL, Azure, Supabase, Neon) and a
user allowed to `CREATE EXTENSION vector`, or an administrator who has already created it. Without
it, the app works the same with vectors in memory; `/api/health` reports which mode is active.

> **Quoted values in `.env`:** Docker Compose strips quotes around values, but
> `docker run --env-file` doesn't, so `GROQ_API_KEY="gsk_…"` becomes an invalid key. Use Compose,
> or write values without quotes.

## Without Docker

```bash
pip install -r requirements.txt
export DATABASE_URL=postgresql://assistant:secret@db-host:5432/assistant   # omit for SQLite
python main.py --host 0.0.0.0 --port 8000 --no-browser
```

Put it behind a reverse proxy (nginx, Caddy, a cloud load balancer) for TLS. Proxies must not
buffer the streaming endpoint; the server sends `X-Accel-Buffering: no` for nginx. The server
only trusts forwarding headers from `127.0.0.1` by default; if your proxy runs elsewhere, set
uvicorn's `FORWARDED_ALLOW_IPS` so per-IP rate limits see real client addresses.

## Security checklist

Before exposing the server beyond your machine:

- [ ] **Set `APP_API_KEYS`** to long random values. Without it, anyone who can reach the server
      can spend your Groq quota; the server prints a warning.
- [ ] **Change `POSTGRES_PASSWORD`** from the Compose default, and don't publish port 5432.
- [ ] **Serve over HTTPS.** API keys travel in request headers.
- [ ] **Set `HTTP_USER_AGENT`** with a contact address, as Wikipedia and arXiv ask.
- [ ] **Review limits:** `CHAT_PER_MINUTE`, `SESSION_TOKEN_BUDGET`, `MAX_UPLOAD_MB`, `SESSION_TTL_DAYS`.
- [ ] **Keep `.env` out of version control** (it's in `.gitignore`).

Built in and needing no configuration:
- content security policy and security headers;
- upload validation (type, size, path stripping);
- SSRF protection on every outbound fetch;
- prompt-injection fencing;
- sanitised rendering of answers.

## Scaling

- **Several app instances** can share one PostgreSQL database. Each keeps its own in-memory
  cache of live sessions and rebuilds a session from the database when needed.
- The per-session lock and rate limits live in each process. For strict enforcement across
  instances, route each session to one instance (sticky sessions), or move limits to a shared store.
- With pgvector, vector search runs in the database and restored sessions hold no vectors in
  memory. Chunk text and the BM25 index are still in memory per active session.
- The binding constraint is usually Groq's rate limit, not the server.

## Monitoring

| Signal | Where |
|---|---|
| Liveness and readiness | `GET /api/health` (`503` when the database is down) |
| Metrics | `GET /api/metrics`, Prometheus format (see the [API reference](api.md#metrics)) |
| Logs | stdout. `LOG_FORMAT=json` gives one object per line with `request_id`. `-v` adds routing and tool-call detail. |
| Traces | LangSmith, via the `LANGSMITH_*` variables ([configuration](configuration.md#tracing-optional)) |

Useful alerts: `assistant_turn_errors_total` rising (model API trouble or rate limits),
`assistant_injection_flags_total` above zero (someone probing), and turn latency p95.

## Backups

- **PostgreSQL:** `docker compose exec db pg_dump -U assistant assistant > backup.sql`
- **SQLite:** copy `data/assistant.db` while the server is stopped, or use `sqlite3 data/assistant.db ".backup backup.db"`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GROQ_API_KEY is not set` | Create `.env` from `.env.example`. |
| `Invalid API Key` in Docker | Quotes in `.env` with `docker run --env-file`; see above. |
| `Groq rate limit reached` / `429` | The free tier's per-minute (8K tokens) or daily (200K tokens) limit. Wait, reduce `ENABLED_TOOLS`, or upgrade the Groq tier. |
| `Request too large` (413 from Groq) | A turn exceeded ~8K tokens. Lower `history_token_budget` or `max_web_results`, or disable tools. |
| Uploads take a long time the first time | The embedding model (~67 MB) downloads on first use. The Docker image includes it. |
| "OCR found no readable text either" | The scan is too faint, rotated or low-resolution. Rescan at 300 DPI or higher. |
| Scanned pages missing from answers | Only `OCR_MAX_PAGES` pages (30) are OCR'd per document; raise it, or split the file. |
| `/api/health` shows `"vector_search": "memory"` on Postgres | pgvector isn't installed or the user can't create it; the log says which. Use the `pgvector/pgvector` image or install the extension. |
| Wikipedia or arXiv time out | They throttle anonymous clients. Set `HTTP_USER_AGENT` with contact details. |
| Health returns `503` | The database is unreachable. Check `DATABASE_URL` and that Postgres is running. |
| The browser keeps asking for a key | The server has `APP_API_KEYS` set; enter one of those keys. |
