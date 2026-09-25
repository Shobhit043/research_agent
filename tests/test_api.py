import json
import os

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

import web.server as server
from agent.schemas import QueryAnalysis
from fakes import KeywordEmbedder, make_agent, tool_call
from web.config import ServerSettings
from web.security import RateLimiter

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def reset_postgres(url: str) -> None:
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, turns, sessions CASCADE")


class Harness:
    """Builds apps over one database, so a 'restart' is just a new app on the same DB."""

    def __init__(self, tmp_path, database_url=None):
        self.tmp_path = tmp_path
        self.database_url = database_url
        self.replies: list = []
        self.agents: list = []
        self.apps: list = []

    def factory(self, session_id):
        agent, _ = make_agent(self.replies, with_docs=False)
        agent.store._embedder = KeywordEmbedder()
        agent.session_id = session_id
        self.agents.append(agent)
        return agent

    def client(self, **overrides) -> TestClient:
        settings = ServerSettings(data_dir=self.tmp_path, database_url=self.database_url, **overrides)
        app = server.create_app(agent_factory=self.factory, settings=settings)
        self.apps.append(app)
        return TestClient(app)

    def close(self):
        for app in self.apps:
            app.state.storage.close()


# Every API test runs on SQLite, and on PostgreSQL too when TEST_DATABASE_URL is set.
@pytest.fixture(params=[
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")),
])
def harness(request, tmp_path):
    database_url = None
    if request.param == "postgres":
        reset_postgres(TEST_DATABASE_URL)
        database_url = TEST_DATABASE_URL
    harness = Harness(tmp_path, database_url)
    yield harness
    harness.close()


@pytest.fixture
def client(harness):
    return harness.client()


@pytest.fixture
def session(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def upload(client, session, name, content, mime="application/pdf"):
    return client.post(f"/api/sessions/{session}/documents", files={"file": (name, content, mime)})


def api_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ----- UI, health, headers -----

def test_serves_ui_with_security_headers_and_request_id(client):
    response = client.get("/", headers={"X-Request-ID": "trace-123"})

    assert response.status_code == 200
    assert "Research Assistant" in response.text
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Request-ID"] == "trace-123"


def test_malformed_request_id_is_replaced(client):
    response = client.get("/api/health", headers={"X-Request-ID": "bad id\r\nx"})

    assert response.headers["X-Request-ID"] != "bad id\r\nx"


def test_health_reports_database_backend(client, harness):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["database"] == ("postgres" if harness.database_url else "sqlite")


def test_health_fails_when_database_is_down(client, monkeypatch):
    monkeypatch.setattr(client.app.state.storage, "ping", lambda: False)

    response = client.get("/api/health")

    assert response.status_code == 503 and response.json()["status"] == "degraded"


# ----- uploads -----

def test_upload_indexes_file_and_strips_path_from_name(client, session, make_pdf):
    pdf = make_pdf("x.pdf", ["Revenue grew 18 percent"]).read_bytes()

    response = upload(client, session, "../../secret/Report 2025.pdf", pdf)

    assert response.status_code == 201
    assert response.json()["documents"] == [{"name": "Report 2025.pdf", "chunks": 1}]
    assert client.get(f"/api/sessions/{session}/documents").json() == response.json()


@pytest.mark.parametrize(
    "name, content, status",
    [("malware.exe", b"MZ", 415), ("broken.pdf", b"not a pdf", 422), ("empty.txt", b"   ", 422)],
)
def test_upload_rejects_bad_files(client, session, name, content, status):
    response = upload(client, session, name, content)

    assert response.status_code == status
    assert response.json()["detail"]


def test_upload_enforces_size_limit(harness):
    client = harness.client(max_upload_bytes=10)
    session = client.post("/api/sessions").json()["session_id"]

    assert upload(client, session, "big.txt", b"x" * 50, "text/plain").status_code == 413


def test_delete_document(client, session):
    upload(client, session, "notes.txt", b"alpha beta", "text/plain")

    assert client.delete(f"/api/sessions/{session}/documents/notes.txt").json() == {"documents": []}
    assert client.delete(f"/api/sessions/{session}/documents/notes.txt").status_code == 404


# ----- chat -----

def test_chat_returns_answer_with_trace_sources_and_usage(client, session, harness):
    upload(client, session, "notes.md", b"Groq serves open models fast.", "text/markdown")
    harness.replies += [
        tool_call("search_documents", {"query": "groq"}, "c1"),
        "Groq is fast [notes.md]. Also [notes.md p.9].",
    ]

    response = client.post(f"/api/sessions/{session}/chat", json={"message": "what is groq?"})

    assert response.status_code == 200
    turn = response.json()
    assert turn["answer"] == "Groq is fast [notes.md]. Also."
    assert turn["removed_citations"] == ["[notes.md p.9]"]
    assert turn["tool_calls"][0]["name"] == "search_documents"
    assert "Groq serves open models" in turn["tool_calls"][0]["output"]
    assert turn["sources"] == [{"kind": "document", "label": "notes.md", "url": None, "ref": None}]
    assert turn["usage"]["llm_calls"] == 3 and turn["trace_id"]


def test_chat_stream_sends_progress_then_done(client, session, harness):
    upload(client, session, "notes.md", b"Groq serves open models fast.", "text/markdown")
    harness.replies += [tool_call("search_documents", {"query": "groq"}, "c1"), "Groq is fast."]

    response = client.post(f"/api/sessions/{session}/chat/stream", json={"message": "what is groq?"})

    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "analysis" and kinds[-1] == "done"
    assert {"tool_start", "tool_end", "token"} <= set(kinds)
    assert events[-1][1]["answer"] == "Groq is fast."


def test_stream_reports_model_errors_as_events(client, session, harness):
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append(api_error(groq.RateLimitError, 429))

    events = parse_sse(client.post(f"/api/sessions/{session}/chat/stream", json={"message": "hi"}).text)

    kind, data = events[-1]
    assert kind == "error" and data["status"] == 429 and "rate limit" in data["message"]
    assert harness.agents[0].messages == []


def test_chat_validation_and_unknown_session(client, session):
    assert client.post(f"/api/sessions/{session}/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/sessions/nope/chat", json={"message": "hi"}).status_code == 404


def test_concurrent_chat_on_same_session_is_rejected(client, session):
    lock = client.app.state.sessions.get(session, "anonymous").lock
    lock._locked = True  # simulate a turn in progress without an event loop
    try:
        assert client.post(f"/api/sessions/{session}/chat", json={"message": "hi"}).status_code == 409
    finally:
        lock._locked = False


@pytest.mark.parametrize(
    "error, status",
    [(api_error(groq.RateLimitError, 429), 429), (api_error(groq.InternalServerError, 500), 502)],
)
def test_model_errors_map_to_http_status(client, session, harness, error, status):
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append(error)

    response = client.post(f"/api/sessions/{session}/chat", json={"message": "hi"})

    assert response.status_code == status
    assert harness.agents[0].messages == []


def test_reset_clears_history(client, session, harness):
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append("hello")
    client.post(f"/api/sessions/{session}/chat", json={"message": "hi"})

    assert client.post(f"/api/sessions/{session}/reset").status_code == 204
    assert harness.agents[0].messages == []
    assert client.get(f"/api/sessions/{session}").json()["turns"] == []


# ----- persistence -----

def test_session_survives_restart_with_history_documents_and_vectors(harness, make_pdf):
    first = harness.client()
    session = first.post("/api/sessions").json()["session_id"]
    upload(first, session, "report.pdf", make_pdf("r.pdf", ["Quarterly sales climbed"]).read_bytes())
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append("Hello!")
    first.post(f"/api/sessions/{session}/chat", json={"message": "hi"})

    restarted = harness.client()
    info = restarted.get(f"/api/sessions/{session}").json()

    assert info["documents"] == [{"name": "report.pdf", "chunks": 1}]
    assert [t["question"] for t in info["turns"]] == ["hi"]
    assert info["turns"][0]["result"]["answer"] == "Hello!"
    assert info["tokens_used"] > 0
    restored = harness.agents[-1]
    assert [m.text for m in restored.messages] == ["hi", "Hello!"]
    assert restored.store.hybrid, "stored vectors should be reused"
    assert restored.store.search("turnover", k=1), "semantic search works after restore"


# ----- auth, ownership, limits -----

def test_api_key_required_when_configured(harness):
    client = harness.client(api_keys=("secret-key",))

    assert client.post("/api/sessions").status_code == 401
    assert client.post("/api/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/sessions", headers={"Authorization": "Bearer secret-key"}).status_code == 201
    assert client.get("/api/health").status_code == 200, "health stays public for load balancers"


def test_sessions_are_private_to_their_key(harness):
    client = harness.client(api_keys=("alice-key", "bob-key"))
    alice = {"Authorization": "Bearer alice-key"}
    bob = {"Authorization": "Bearer bob-key"}
    session = client.post("/api/sessions", headers=alice).json()["session_id"]

    assert client.get(f"/api/sessions/{session}", headers=alice).status_code == 200
    assert client.get(f"/api/sessions/{session}", headers=bob).status_code == 404


def test_chat_rate_limit(harness):
    client = harness.client(chat_per_minute=2)
    session = client.post("/api/sessions").json()["session_id"]
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies += ["a", "b", "c"]

    codes = [client.post(f"/api/sessions/{session}/chat", json={"message": "hi"}).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


def test_token_budget_blocks_further_chat(harness):
    client = harness.client(session_token_budget=50)
    session = client.post("/api/sessions").json()["session_id"]
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies += ["first", "second"]

    assert client.post(f"/api/sessions/{session}/chat", json={"message": "hi"}).status_code == 200
    blocked = client.post(f"/api/sessions/{session}/chat", json={"message": "hi"})
    assert blocked.status_code == 429 and "token budget" in blocked.json()["detail"]


def test_rate_limiter_window_and_retry_after():
    limiter = RateLimiter(limit=2, window=60)
    limiter.check("k")
    limiter.check("k")

    with pytest.raises(server.HTTPException) as exc:
        limiter.check("k")
    assert exc.value.status_code == 429 and int(exc.value.headers["Retry-After"]) > 0
    limiter.check("other-key")


def test_metrics_endpoint_counts_turns_and_tokens(client, session, harness):
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append("hello")
    client.post(f"/api/sessions/{session}/chat", json={"message": "hi"})

    text = client.get("/api/metrics").text

    assert 'assistant_turns_total{route="direct"} 1' in text
    assert 'assistant_llm_tokens_total{direction="input"} 200' in text
    assert "assistant_turn_latency_seconds_count 1" in text
    assert 'assistant_http_requests_total{method="POST",route="/api/sessions",status="201"}' in text


def test_attachments_are_passed_to_agent_and_saved_with_the_turn(client, session, harness):
    upload(client, session, "notes.md", b"Groq serves open models fast.", "text/markdown")
    harness.agents[0].llm.analysis = QueryAnalysis(needs_retrieval=False)
    harness.replies.append("Summary.")

    response = client.post(f"/api/sessions/{session}/chat", json={
        "message": "Summarise this", "attachments": ["notes.md", "deleted.pdf", "notes.md"],
    })

    assert response.status_code == 200
    human = harness.agents[0].messages[0].text
    assert "[Attached files: notes.md]" in human, "unknown names dropped, duplicates collapsed"
    [turn] = client.get(f"/api/sessions/{session}").json()["turns"]
    assert turn["question"] == "Summarise this"
    assert turn["attachments"] == ["notes.md"]


def test_database_from_before_attachments_is_migrated(tmp_path):
    import sqlite3

    from web.storage import SQLiteStorage

    db = tmp_path / "assistant.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at REAL NOT NULL,"
            " updated_at REAL NOT NULL, messages TEXT NOT NULL DEFAULT '[]', tokens_used INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
            " question TEXT NOT NULL, result TEXT NOT NULL, created_at REAL NOT NULL);"
            "INSERT INTO sessions VALUES ('s1', 'anonymous', 0, 0, '[]', 0);"
            "INSERT INTO turns (session_id, question, result, created_at) VALUES ('s1', 'old', '{}', 0);"
        )

    storage = SQLiteStorage(db)

    assert storage.list_turns("s1") == [{"question": "old", "result": {}, "attachments": []}]
