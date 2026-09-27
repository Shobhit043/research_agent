import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import HTTPException

from agent.research_agent import ResearchAgent
from web.storage import Storage

logger = logging.getLogger(__name__)

NOT_FOUND = "Session not found. It may have expired; start a new one."


@dataclass
class Session:
    id: str
    owner: str
    agent: ResearchAgent
    # Serialises work on one agent: its history and index aren't safe for concurrent turns.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionManager:
    """Live agents cached in memory (LRU); SQLite is the source of truth.

    An evicted or pre-restart session is rebuilt from storage on its next request.
    """

    def __init__(
        self,
        agent_factory: Callable[[str, Storage], ResearchAgent],
        storage: Storage,
        max_cached: int = 50,
    ):
        self._factory = agent_factory
        self._storage = storage
        self._max = max_cached
        self._cache: OrderedDict[str, Session] = OrderedDict()

    def create(self, owner: str) -> Session:
        session_id = uuid.uuid4().hex
        self._storage.create_session(session_id, owner)
        session = Session(session_id, owner, self._factory(session_id, self._storage))
        self._remember(session)
        return session

    def get(self, session_id: str, owner: str) -> Session:
        session = self._cache.get(session_id)
        if session is None:
            session = self._restore(session_id)
        # 404 rather than 403, so session ids belonging to other users aren't confirmed to exist.
        if session.owner != owner:
            raise HTTPException(404, NOT_FOUND)
        self._cache.move_to_end(session_id)
        return session

    def forget(self, session_id: str) -> None:
        self._cache.pop(session_id, None)

    def _restore(self, session_id: str) -> Session:
        owner = self._storage.session_owner(session_id)
        if owner is None:
            raise HTTPException(404, NOT_FOUND)
        agent = self._factory(session_id, self._storage)
        agent.messages = self._storage.load_messages(session_id)
        chunks, vectors = self._storage.load_chunks(session_id)
        if chunks:
            # With pgvector the vectors are queried in the database, so don't re-embed here.
            agent.store.add(chunks, vectors, embed=not self._storage.vector_search_enabled)
        logger.info("Restored session %s (%d messages, %d chunks)", session_id, len(agent.messages), len(chunks))
        session = Session(session_id, owner, agent)
        self._remember(session)
        return session

    def _remember(self, session: Session) -> None:
        self._cache[session.id] = session
        # Evict idle sessions only; one mid-turn keeps its agent until the turn finishes.
        for session_id in list(self._cache):
            if len(self._cache) <= self._max:
                break
            if not self._cache[session_id].lock.locked():
                del self._cache[session_id]
