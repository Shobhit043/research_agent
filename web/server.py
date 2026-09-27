import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import groq
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import MutableHeaders

from agent.config import Settings
from agent.documents import SUPPORTED_EXTENSIONS, DocumentLoadError, DocumentStore, FastEmbedEmbedder
from agent.research_agent import ResearchAgent
from agent.schemas import TurnResult
from web.config import ServerSettings
from web.observability import Metrics, request_id_var
from web.security import SECURITY_HEADERS, RateLimiter, authenticate, client_ip
from web.sessions import Session, SessionManager
from web.storage import Storage, create_storage

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Names of documents attached to this message in the UI (already uploaded to the session).
    attachments: list[str] = Field(default_factory=list, max_length=10)


class DocumentInfo(BaseModel):
    name: str
    chunks: int
    ocr: bool = False  # some or all of the text came from OCR


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class SessionInfo(BaseModel):
    session_id: str
    documents: list[DocumentInfo]
    turns: list[dict]
    tokens_used: int
    token_budget: int


AgentFactory = Callable[[str, Storage], ResearchAgent]


def default_agent_factory(settings: Settings) -> tuple[AgentFactory, FastEmbedEmbedder | None]:
    """One shared embedding model for all sessions; it's the only heavy object."""
    embedder = FastEmbedEmbedder(settings.embedding_model) if settings.use_embeddings else None

    def factory(session_id: str, storage: Storage) -> ResearchAgent:
        vector_search = None
        if embedder is not None and storage.vector_search_enabled:
            def vector_search(query, k):
                return storage.vector_search(session_id, query, k)
        store = DocumentStore(embedder=embedder, vector_search=vector_search)
        return ResearchAgent(settings=settings, store=store, session_id=session_id)

    return factory, embedder


class ObservabilityMiddleware:
    """Request ids, security headers and request metrics.

    Pure ASGI (not BaseHTTPMiddleware) so streaming responses and client-disconnect
    detection pass straight through, and the request id stays set while a stream runs.
    """

    def __init__(self, app, metrics: Metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        status = 500

        async def send_with_headers(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            route = getattr(scope.get("route"), "path", "static")
            self.metrics.inc("assistant_http_requests_total", method=scope["method"], route=route, status=str(status))
            request_id_var.reset(token)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _safe_filename(raw: str | None) -> str:
    # Keep only the final path component: browsers can send "C:\\fakepath\\x.pdf" or "../x.pdf".
    name = Path((raw or "").replace("\\", "/")).name.strip()
    if not name:
        raise HTTPException(400, "Uploaded file has no name.")
    return name[:120]


async def _save_upload(upload: UploadFile, suffix: str, limit: int) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while block := await upload.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, f"File exceeds the {limit // (1024 * 1024)} MB limit.")
                out.write(block)
    except BaseException:
        os.unlink(path)
        raise
    return path


def _model_error(exc: groq.APIError) -> tuple[int, str]:
    if isinstance(exc, groq.RateLimitError):
        return 429, "Groq rate limit reached. Wait a few seconds and try again."
    return 502, f"Model API error: {exc}"


def create_app(
    agent_factory: AgentFactory | None = None,
    settings: ServerSettings | None = None,
) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    agent_settings = Settings.from_env()
    embedder = None
    if agent_factory is None:
        agent_factory, embedder = default_agent_factory(agent_settings)

    storage = create_storage(
        settings.database_url, settings.db_path,
        embedding_dim=agent_settings.embedding_dim, use_pgvector=settings.use_pgvector,
    )
    purged = storage.purge_idle(settings.session_ttl_days)
    if purged:
        logger.info("Purged %d sessions idle for over %d days", purged, settings.session_ttl_days)
    sessions = SessionManager(agent_factory, storage, settings.max_cached_sessions)
    metrics = Metrics()
    chat_limiter = RateLimiter(settings.chat_per_minute)
    upload_limiter = RateLimiter(settings.uploads_per_minute)
    session_limiter = RateLimiter(settings.sessions_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if embedder is not None:
            # Load the embedding model in the background so the first upload isn't slow.
            threading.Thread(target=_warm_up, args=(embedder,), daemon=True).start()
        yield
        storage.close()

    app = FastAPI(title="Research Assistant", version="0.3.0", lifespan=lifespan)
    app.state.sessions = sessions
    app.state.storage = storage
    app.state.metrics = metrics
    app.add_middleware(ObservabilityMiddleware, metrics=metrics)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    def owner(request: Request) -> str:
        return authenticate(request, settings.api_keys)

    def documents_of(session: Session) -> list[DocumentInfo]:
        ocr = session.agent.store.ocr_sources()
        return [
            DocumentInfo(name=n, chunks=c, ocr=n in ocr) for n, c in session.agent.store.chunk_counts().items()
        ]

    async def begin_turn(session_id: str, who: str) -> Session:
        session = sessions.get(session_id, who)
        chat_limiter.check(session_id)
        if storage.tokens_used(session_id) >= settings.session_token_budget:
            raise HTTPException(429, "This session has used its token budget. Start a new session.")
        if session.lock.locked():
            raise HTTPException(409, "Still working on the previous message.")
        await session.lock.acquire()
        return session

    def prepare_question(session: Session, request: ChatRequest) -> tuple[str, str, list[str]]:
        """The text to store and show, the text the agent sees, and the attachments that exist."""
        shown = request.message.strip()
        indexed = set(session.agent.store.sources)
        attached = [name for name in dict.fromkeys(request.attachments) if name in indexed]
        if not attached:
            return shown, shown, []
        # Tells the router and model which files "this document" refers to.
        return shown, f"{shown}\n\n[Attached files: {', '.join(attached)}]", attached

    def record_turn(session: Session, question: str, result: TurnResult, attachments: list[str]) -> None:
        storage.save_turn(session.id, question, result.model_dump(), session.agent.messages,
                          result.usage.total_tokens, attachments)
        metrics.inc("assistant_turns_total", route="research" if result.tool_calls else "direct")
        metrics.inc("assistant_llm_tokens_total", result.usage.input_tokens, direction="input")
        metrics.inc("assistant_llm_tokens_total", result.usage.output_tokens, direction="output")
        metrics.inc("assistant_removed_citations_total", len(result.removed_citations))
        metrics.inc("assistant_injection_flags_total", sum(c.flagged for c in result.tool_calls))
        for call in result.tool_calls:
            metrics.inc("assistant_tool_calls_total", tool=call.name)
        metrics.observe_turn(result.latency_ms / 1000)
        logger.info("Turn %s finished in %d ms, %d tokens", result.trace_id, result.latency_ms,
                    result.usage.total_tokens)

    # ----- routes -----

    @app.get("/api/health")
    def health() -> JSONResponse:
        # Load balancers and Docker healthchecks use this, so a dead database must fail it.
        database_ok = storage.ping()
        return JSONResponse(
            {
                "status": "ok" if database_ok else "degraded",
                "database": storage.backend if database_ok else f"{storage.backend} unreachable",
                "auth_required": bool(settings.api_keys),
                "hybrid_search": embedder is not None,
                "vector_search": "pgvector" if storage.vector_search_enabled else "memory",
            },
            status_code=200 if database_ok else 503,
        )

    @app.get("/api/metrics", response_class=PlainTextResponse)
    def metrics_endpoint(who: str = Depends(owner)) -> str:
        return metrics.render()

    @app.post("/api/sessions", status_code=201)
    def create_session(request: Request, who: str = Depends(owner)) -> dict:
        session_limiter.check(client_ip(request))
        return {"session_id": sessions.create(who).id, "token_budget": settings.session_token_budget}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, who: str = Depends(owner)) -> SessionInfo:
        session = sessions.get(session_id, who)
        return SessionInfo(
            session_id=session.id,
            documents=documents_of(session),
            turns=storage.list_turns(session.id),
            tokens_used=storage.tokens_used(session.id),
            token_budget=settings.session_token_budget,
        )

    @app.get("/api/sessions/{session_id}/documents")
    def list_documents(session_id: str, who: str = Depends(owner)) -> DocumentList:
        return DocumentList(documents=documents_of(sessions.get(session_id, who)))

    @app.post("/api/sessions/{session_id}/documents", status_code=201)
    async def upload_document(session_id: str, file: UploadFile, who: str = Depends(owner)) -> DocumentList:
        session = sessions.get(session_id, who)
        upload_limiter.check(session_id)
        name = _safe_filename(file.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(415, f"Unsupported file type '{suffix or name}'. Supported: {supported}.")

        path = await _save_upload(file, suffix, settings.max_upload_bytes)
        try:
            async with session.lock:
                # Parsing and embedding are CPU-bound; keep them off the event loop.
                await run_in_threadpool(session.agent.add_document, path, name)
                chunks, vectors = session.agent.store.export(name)
                storage.replace_document(session.id, name, chunks, vectors)
        except DocumentLoadError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            os.unlink(path)
        metrics.inc("assistant_uploads_total", type=suffix.lstrip("."))
        return DocumentList(documents=documents_of(session))

    @app.delete("/api/sessions/{session_id}/documents/{name}")
    async def delete_document(session_id: str, name: str, who: str = Depends(owner)) -> DocumentList:
        session = sessions.get(session_id, who)
        async with session.lock:
            if not session.agent.remove_document(name):
                raise HTTPException(404, f"No document named '{name}'.")
            storage.delete_document(session.id, name)
        return DocumentList(documents=documents_of(session))

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, request: ChatRequest, who: str = Depends(owner)) -> TurnResult:
        session = await begin_turn(session_id, who)
        shown, question, attached = prepare_question(session, request)
        try:
            async for event in session.agent.astream(question):
                if event.type == "done":
                    record_turn(session, shown, event.result, attached)
                    return event.result
        except groq.APIError as exc:
            metrics.inc("assistant_turn_errors_total", kind=type(exc).__name__)
            status, message = _model_error(exc)
            raise HTTPException(status, message) from exc
        finally:
            session.lock.release()
        raise HTTPException(500, "Agent finished without an answer.")

    @app.post("/api/sessions/{session_id}/chat/stream")
    async def chat_stream(session_id: str, request: ChatRequest, who: str = Depends(owner)) -> StreamingResponse:
        session = await begin_turn(session_id, who)
        shown, question, attached = prepare_question(session, request)

        async def events() -> AsyncIterator[str]:
            try:
                async for event in session.agent.astream(question):
                    if event.type == "done":
                        record_turn(session, shown, event.result, attached)
                        yield _sse("done", event.result.model_dump())
                    else:
                        yield _sse(event.type, event.data)
            except groq.APIError as exc:
                metrics.inc("assistant_turn_errors_total", kind=type(exc).__name__)
                status, message = _model_error(exc)
                yield _sse("error", {"status": status, "message": message})
            except asyncio.CancelledError:
                # Client disconnected; the agent has already rolled back the partial turn.
                metrics.inc("assistant_turns_cancelled_total")
                raise
            except Exception:
                logger.exception("Turn failed")
                metrics.inc("assistant_turn_errors_total", kind="internal")
                yield _sse("error", {"status": 500, "message": "Something went wrong answering that."})
            finally:
                session.lock.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/reset", status_code=204)
    async def reset(session_id: str, who: str = Depends(owner)) -> None:
        session = sessions.get(session_id, who)
        async with session.lock:
            session.agent.reset()
            storage.reset_session(session.id)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _warm_up(embedder: FastEmbedEmbedder) -> None:
    started = time.perf_counter()
    try:
        embedder.embed_query("warm up")
        logger.info("Embedding model ready in %.1fs", time.perf_counter() - started)
    except Exception as exc:
        logger.warning("Embedding model unavailable (%s); uploads will use keyword search only", exc)
