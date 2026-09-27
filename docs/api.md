# API reference

The bundled UI uses this API, and so can any other client. All endpoints are under `/api`, and
request and response bodies are JSON unless stated otherwise.

**Authentication.** If `APP_API_KEYS` is set, send `Authorization: Bearer <key>` on every request
except `GET /api/health`. Sessions are private to the key that created them; another key gets
`404`, so session IDs can't be probed.

**Tracing.** Every response has an `X-Request-ID` header. Send your own (letters, digits, `.`,
`_`, `-`, up to 64 characters) to correlate with server logs.

## Endpoints

| Method and path | Purpose | Success |
|---|---|---|
| `GET /api/health` | Liveness and database check | `200`, or `503` if the database is unreachable |
| `GET /api/metrics` | Prometheus metrics | `200` text |
| `POST /api/sessions` | Create a session | `201` |
| `GET /api/sessions/{id}` | Documents, past turns and token usage | `200` |
| `GET /api/sessions/{id}/documents` | List indexed documents | `200` |
| `POST /api/sessions/{id}/documents` | Upload a file (multipart field `file`): PDF, TXT, MD, PNG, JPG, WebP or TIFF | `201` |
| `DELETE /api/sessions/{id}/documents/{name}` | Remove a document | `200` |
| `POST /api/sessions/{id}/chat/stream` | Ask a question; Server-Sent Events | `200` stream |
| `POST /api/sessions/{id}/chat` | Ask a question; one JSON response | `200` |
| `POST /api/sessions/{id}/reset` | Clear the conversation (documents are kept) | `204` |

## Examples

```bash
# Create a session
SID=$(curl -s -X POST localhost:8000/api/sessions | jq -r .session_id)

# Upload a document (PDF, TXT or MD)
curl -s -F "file=@report.pdf" localhost:8000/api/sessions/$SID/documents
# {"documents": [{"name": "report.pdf", "chunks": 12}]}

# Ask, streaming progress
curl -N -H "Content-Type: application/json" \
  -d '{"message": "What are the main risks?", "attachments": ["report.pdf"]}' \
  localhost:8000/api/sessions/$SID/chat/stream

# Ask, one JSON response
curl -s -H "Content-Type: application/json" -d '{"message": "What time is it in Tokyo?"}' \
  localhost:8000/api/sessions/$SID/chat | jq .answer
```

## Request bodies

**Chat** (both chat endpoints):

| Field | Type | Notes |
|---|---|---|
| `message` | string, 1–4,000 characters | Required |
| `attachments` | list of up to 10 file names | Optional. Names of documents already uploaded to the session; the agent is told which files the message refers to. Unknown names are ignored. They're saved with the turn and returned in `GET /api/sessions/{id}`. |

## Streaming events

`POST /chat/stream` returns `text/event-stream`. Each event looks like this:

```
event: tool_start
data: {"id": "fc_1", "name": "search_documents", "args": {"query": "main risks"}}
```

| Event | Data | Meaning |
|---|---|---|
| `analysis` | `{"needs_retrieval": bool, "sub_questions": [str]}` | The routing decision |
| `tool_start` | `{"id", "name", "args"}` | A tool call began |
| `tool_end` | `{"id", "name", "duration_ms", "flagged"}` | It finished; `flagged` means possible prompt injection |
| `token` | `{"text": str}` | Streamed answer text |
| `reset` | `{}` | Discard the text streamed so far (the model switched to calling tools) |
| `done` | a `TurnResult` | The final, citation-checked answer. Always the last event on success. |
| `error` | `{"status": int, "message": str}` | The turn failed. `429` means rate-limited; `502` means a model API error. |

The streamed text is a preview. Render the `answer` from `done`, which has had fabricated
citations removed. If the client disconnects, the turn is rolled back and nothing is saved.

## TurnResult

Returned by `POST /chat` and in the `done` event:

```json
{
  "answer": "Revenue grew 23% to $612M [report.pdf p.1].",
  "analysis": {"needs_retrieval": true, "sub_questions": ["What was the revenue growth?"]},
  "tool_calls": [{"name": "search_documents", "args": {"query": "revenue growth"},
                  "output": "[report.pdf p.1]\n…", "duration_ms": 12, "flagged": false}],
  "sources": [{"kind": "document", "label": "report.pdf p.1", "url": null, "ref": null},
              {"kind": "web", "label": "Tokyo (Wikipedia)", "url": "https://…", "ref": "W1"}],
  "removed_citations": ["[report.pdf p.9]"],
  "warnings": [],
  "usage": {"input_tokens": 1674, "output_tokens": 409, "llm_calls": 3},
  "latency_ms": 1890,
  "trace_id": "42b2f449a88b4fc9"
}
```

- `tool_calls[].output` is truncated to 600 characters.
- `sources[].ref` is the citation tag without brackets (`2`, `W1`, `A1`, `U1`, `P1`). Web tags
  restart at 1 on each call of the same tool.

## Other responses

| Endpoint | Body |
|---|---|
| `POST /sessions` | `{"session_id": str, "token_budget": int}` |
| `GET /sessions/{id}` | `{"session_id", "documents": [{"name", "chunks"}], "turns": [{"question", "attachments", "result": TurnResult}], "tokens_used", "token_budget"}` |
| Upload, list, delete documents | `{"documents": [{"name": str, "chunks": int, "ocr": bool}]}`; `ocr` is true when some text came from OCR |
| `GET /health` | `{"status": "ok" \| "degraded", "database": "postgres" \| "sqlite", "auth_required": bool, "hybrid_search": bool, "vector_search": "pgvector" \| "memory"}` |

## Errors

Errors are JSON `{"detail": "…"}` with these status codes:

| Status | When |
|---|---|
| `400` | Uploaded file has no name |
| `401` | Missing or wrong API key |
| `404` | Unknown session, a session owned by another key, or an unknown document |
| `409` | The session is still answering the previous message |
| `413` | Upload is over `MAX_UPLOAD_MB` |
| `415` | Unsupported file type (supported: `.pdf`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`) |
| `422` | Invalid body, or a file with no readable text (corrupt, empty, not UTF-8, or a scan where OCR found nothing) |
| `429` | Rate limit (with a `Retry-After` header), session token budget used up, or Groq rate limit |
| `502` | The model API failed |
| `503` | Health check only: the database is unreachable |

## Metrics

`GET /api/metrics` uses the Prometheus text format:

| Metric | Labels |
|---|---|
| `assistant_http_requests_total` | `method`, `route`, `status` |
| `assistant_turns_total` | `route` (`research` or `direct`) |
| `assistant_turn_latency_seconds` (summary) | — |
| `assistant_llm_tokens_total` | `direction` (`input` or `output`) |
| `assistant_tool_calls_total` | `tool` |
| `assistant_turn_errors_total` | `kind` |
| `assistant_turns_cancelled_total` | — |
| `assistant_removed_citations_total` | — |
| `assistant_injection_flags_total` | — |
| `assistant_uploads_total` | `type` |
| `assistant_uptime_seconds` | — |

Counters are per process and reset on restart.
