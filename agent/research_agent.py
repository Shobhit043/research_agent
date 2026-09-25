import asyncio
import datetime
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    message_chunk_to_message,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from agent.citations import collect_sources, evidence_pages, verify_citations
from agent.config import Settings
from agent.documents import DocumentStore, ingest_document
from agent.guardrails import scan_for_injection, wrap_untrusted
from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    BUDGET_EXHAUSTED,
    NO_TOOLS,
    QUERY_ANALYSIS_PROMPT,
)
from agent.schemas import AgentEvent, QueryAnalysis, ToolCallRecord, TurnResult, Usage
from agent.tools import DOCUMENT_TOOLS, build_tools, tool_guidance

logger = logging.getLogger(__name__)

_COMPACT_CHARS = 300
_TRACE_CHARS = 600
_FINAL_ANSWER_NUDGE = (
    "Stop searching. Using only the tool results above, answer my question now in plain text. "
    "If the results don't contain the answer, say that the information isn't available."
)
_BUDGET_FALLBACK = (
    "I hit the research step limit before finishing. Try a narrower question, "
    "or ask me to summarise what I found so far."
)


def _is_tool_use_failed(exc: groq.APIError) -> bool:
    # Before streaming starts Groq returns a 400 with code tool_use_failed; once a stream is
    # under way the same failure arrives as a bare APIError: "... but model called a tool".
    message = str(exc)
    return "tool_use_failed" in message or "model called a tool" in message


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@dataclass
class _TurnState:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started: float = field(default_factory=time.perf_counter)
    usage: Usage = field(default_factory=Usage)
    durations: dict[str, int] = field(default_factory=dict)
    flagged: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


class ResearchAgent:
    """Tool-calling research assistant: analyse the query, gather evidence, answer with citations.

    `astream` is the core: an async generator of progress events ending in `done`.
    `arun`, `run` and `ask` are conveniences on top of it.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        store: DocumentStore | None = None,
        session_id: str | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.llm = llm or ChatGroq(
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            max_tokens=self.settings.max_output_tokens,
        )
        self.store = store if store is not None else DocumentStore()
        self.session_id = session_id
        self.messages: list[BaseMessage] = []

        self._tools = build_tools(self.store, self.settings)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", AGENT_SYSTEM_PROMPT), MessagesPlaceholder("history")]
        )
        self._analyzer = ChatPromptTemplate.from_messages(
            [("system", QUERY_ANALYSIS_PROMPT), ("human", "{question}")]
        # json_schema (native structured output) rather than the default forced tool call:
        # gpt-oss often answers short inputs like "thanks" in prose, which Groq rejects.
        ) | self.llm.with_structured_output(QueryAnalysis, method="json_schema", include_raw=True)

    # ----- documents and history -----

    def add_document(self, path: str | Path, name: str | None = None) -> int:
        chunks = ingest_document(path, self.settings.chunk_size, self.settings.chunk_overlap, name=name)
        self.store.add(chunks)
        logger.info("Indexed %s: %d chunks (hybrid=%s)", name or Path(path).name, len(chunks), self.store.hybrid)
        return len(chunks)

    def remove_document(self, name: str) -> bool:
        return self.store.remove(name)

    def reset(self) -> None:
        self.messages.clear()

    # ----- entry points -----

    def ask(self, question: str) -> str:
        return self.run(question).answer

    def run(self, question: str) -> TurnResult:
        return asyncio.run(self.arun(question))

    async def arun(self, question: str) -> TurnResult:
        async for event in self.astream(question):
            if event.type == "done":
                return event.result
        raise RuntimeError("agent stream ended without a result")

    async def astream(self, question: str) -> AsyncIterator[AgentEvent]:
        turn = _TurnState()
        checkpoint = len(self.messages)
        try:
            self.messages.append(HumanMessage(question))
            analysis = await self._analyze(question, turn)
            logger.info("Query analysis [%s]:\n%s", turn.trace_id, analysis.render())
            yield AgentEvent(type="analysis", data=analysis.model_dump())

            variables = {
                "documents": self._documents_label(),
                "analysis": analysis.render(),
                "today": datetime.date.today().isoformat(),
            }
            if analysis.needs_retrieval:
                async for event in self._research(variables, turn):
                    yield event
            else:
                try:
                    async for event in self._reply({**variables, "tool_guidance": NO_TOOLS}, turn):
                        yield event
                except groq.APIError as exc:
                    if not _is_tool_use_failed(exc):
                        raise
                    # The model tried to call a tool, so the router was wrong: research instead.
                    logger.info("Model requested a tool on the direct path; switching to research")
                    yield AgentEvent(type="reset")
                    async for event in self._research(variables, turn):
                        yield event
        except BaseException:
            # Failed, cancelled, or abandoned mid-stream (the client disconnected): drop the
            # partial turn so history never holds a tool call without its result.
            del self.messages[checkpoint:]
            raise
        yield AgentEvent(type="done", result=self._finish_turn(checkpoint, analysis, turn))

    # ----- turn steps -----

    async def _analyze(self, question: str, turn: _TurnState) -> QueryAnalysis:
        fallback = QueryAnalysis(needs_retrieval=True, sub_questions=[question])
        try:
            output = await self._analyzer.ainvoke(
                {
                    "question": question,
                    "documents": self._documents_label(),
                    "recent": self._recent_conversation(),
                },
                config=self._run_config(turn, "query_analysis"),
            )
        except Exception as exc:
            # Routing is an optimisation; if it fails, researching is the safe default.
            logger.warning("Query analysis failed (%s); defaulting to retrieval", exc)
            return fallback
        raw = output.get("raw")
        turn.usage.add(getattr(raw, "usage_metadata", None))
        return output.get("parsed") or fallback

    async def _research(self, variables: dict, turn: _TurnState) -> AsyncIterator[AgentEvent]:
        has_documents = bool(self.store.sources)
        tools = [t for name, t in self._tools.items() if has_documents or name not in DOCUMENT_TOOLS]
        guidance = tool_guidance(tools)
        tools_by_name = {t.name: t for t in tools}
        chain = self._prompt | self.llm.bind_tools(tools)
        seen: set[tuple[str, str]] = set()

        for _ in range(self.settings.max_tool_iterations):
            response = None
            try:
                async for item in self._call_model(chain, {**variables, "tool_guidance": guidance}, turn):
                    if isinstance(item, AIMessage):
                        response = item
                    else:
                        yield item
            except groq.APIError as exc:
                if not _is_tool_use_failed(exc):
                    raise
                # e.g. gpt-oss calling its built-in browser tool; retry the round (it counts
                # against the budget, so this can't loop forever).
                logger.warning("Malformed tool call rejected by Groq; retrying the round: %s", exc)
                yield AgentEvent(type="reset")
                continue
            self.messages.append(response)
            if not response.tool_calls:
                return

            for call in response.tool_calls:
                yield AgentEvent(type="tool_start", data={"id": call["id"], "name": call["name"], "args": call["args"]})
            # Independent tool calls run concurrently; gather keeps results in call order.
            results = await asyncio.gather(
                *(self._run_tool_call(call, tools_by_name, seen, turn) for call in response.tool_calls)
            )
            for call, message in zip(response.tool_calls, results, strict=True):
                self.messages.append(message)
                yield AgentEvent(type="tool_end", data={
                    "id": call["id"],
                    "name": call["name"],
                    "duration_ms": turn.durations.get(call["id"], 0),
                    "flagged": call["id"] in turn.flagged,
                })

        logger.warning("Tool budget of %d rounds exhausted; forcing a final answer",
                       self.settings.max_tool_iterations)
        final_variables = {**variables, "tool_guidance": BUDGET_EXHAUSTED}
        # gpt-oss sometimes still reaches for a tool here. A user-turn nudge is a stronger signal
        # than the system prompt; it goes to this one call only and is never stored in history.
        for nudge in (None, HumanMessage(_FINAL_ANSWER_NUDGE)):
            try:
                async for event in self._reply(final_variables, turn, nudge):
                    yield event
                return
            except groq.APIError as exc:
                if not _is_tool_use_failed(exc):
                    raise
                yield AgentEvent(type="reset")
        self.messages.append(AIMessage(_BUDGET_FALLBACK))

    async def _reply(
        self, variables: dict, turn: _TurnState, nudge: BaseMessage | None = None
    ) -> AsyncIterator[AgentEvent]:
        # No tools bound, and not tool_choice="none": Groq answers that with a 400
        # (tool_use_failed) whenever the model still emits a tool call.
        response = None
        async for item in self._call_model(self._prompt | self.llm, variables, turn, nudge):
            if isinstance(item, AIMessage):
                response = item
            else:
                yield item
        self.messages.append(response)

    async def _call_model(
        self, chain, variables: dict, turn: _TurnState, nudge: BaseMessage | None = None
    ) -> AsyncIterator[AgentEvent | AIMessage]:
        """Stream one model call: yields token events, then the complete AIMessage last."""
        aggregate = None
        streamed = False
        history = self._history() + ([nudge] if nudge else [])
        async for chunk in chain.astream(
            {**variables, "history": history},
            config=self._run_config(turn, "agent_step"),
        ):
            aggregate = chunk if aggregate is None else aggregate + chunk
            if chunk.text:
                streamed = True
                yield AgentEvent(type="token", data={"text": chunk.text})

        message = message_chunk_to_message(aggregate) if aggregate is not None else AIMessage("")
        turn.usage.add(message.usage_metadata)
        if message.tool_calls and streamed:
            # Text streamed before the model decided to call tools isn't the answer.
            yield AgentEvent(type="reset")
        yield message

    async def _run_tool_call(
        self, call: dict, tools: dict[str, BaseTool], seen: set[tuple[str, str]], turn: _TurnState
    ) -> ToolMessage:
        # Every tool call must get a ToolMessage reply, or the next API request is rejected.
        name, args = call["name"], call["args"]
        key = (name, json.dumps(args, sort_keys=True, default=str))
        started = time.perf_counter()

        if name not in tools:
            raw = f"Error: unknown tool '{name}'. Available tools: {', '.join(tools)}."
        elif key in seen:
            raw = "Skipped: this exact call already ran this turn. Use its earlier result."
        else:
            seen.add(key)
            logger.info("Tool call [%s]: %s(%s)", turn.trace_id, name, args)
            try:
                result = await asyncio.wait_for(
                    tools[name].ainvoke(args, config=self._run_config(turn, name)),
                    timeout=self.settings.tool_timeout,
                )
                raw = str(result)
            except TimeoutError:
                raw = f"Error: {name} timed out after {self.settings.tool_timeout:.0f}s."
            except Exception as exc:
                # Report failures back to the model so it can retry or answer without them.
                logger.warning("Tool %s failed: %s", name, exc)
                raw = f"Error: {name} failed: {exc}"

        turn.durations[call["id"]] = _elapsed_ms(started)
        injection = scan_for_injection(raw)
        if injection:
            turn.flagged.add(call["id"])
            turn.warnings.append(
                f"Possible prompt injection in {name} result ({injection[0]!r}); treated as data."
            )
            logger.warning("Possible prompt injection in %s result: %s", name, injection)
        return ToolMessage(
            content=wrap_untrusted(name, raw, bool(injection)),
            artifact=raw,
            name=name,
            tool_call_id=call["id"],
        )

    def _finish_turn(self, turn_start: int, analysis: QueryAnalysis, turn: _TurnState) -> TurnResult:
        messages = self.messages[turn_start:]
        results = {
            m.tool_call_id: (m.artifact if isinstance(m.artifact, str) else m.text)
            for m in messages
            if isinstance(m, ToolMessage)
        }
        outputs = list(results.values())

        answer = self.messages[-1]
        text, removed = verify_citations(answer.text, evidence_pages(outputs))
        if removed:
            logger.warning("Removed citations not backed by retrieved passages: %s", removed)
        # Store the cleaned text so a fabricated citation isn't repeated in later turns.
        self.messages[-1] = answer.model_copy(update={"content": text})

        calls = [
            ToolCallRecord(
                name=call["name"],
                args=call["args"],
                output=results.get(call["id"], "")[:_TRACE_CHARS],
                duration_ms=turn.durations.get(call["id"], 0),
                flagged=call["id"] in turn.flagged,
            )
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        return TurnResult(
            answer=text,
            analysis=analysis,
            tool_calls=calls,
            sources=collect_sources(outputs),
            removed_citations=removed,
            warnings=turn.warnings,
            usage=turn.usage,
            latency_ms=_elapsed_ms(turn.started),
            trace_id=turn.trace_id,
        )

    # ----- helpers -----

    def _run_config(self, turn: _TurnState, run_name: str) -> dict:
        # Shows up in LangSmith when tracing is enabled; harmless otherwise.
        return {
            "run_name": run_name,
            "metadata": {"trace_id": turn.trace_id, "session_id": self.session_id},
            "tags": ["research-assistant"],
        }

    def _history(self) -> list[BaseMessage]:
        trimmed = trim_messages(
            self.messages,
            max_tokens=self.settings.history_token_budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        if trimmed:
            return trimmed

        # The current turn alone is over budget: shrink its oldest tool results until it fits.
        turn = list(self.messages[self._last_human_index():])
        for index, message in enumerate(turn):
            if count_tokens_approximately(turn) <= self.settings.history_token_budget:
                break
            if isinstance(message, ToolMessage) and len(message.content) > _COMPACT_CHARS:
                turn[index] = message.model_copy(update={
                    "content": message.content[:_COMPACT_CHARS] + "\n...[truncated to fit context]"
                })
        return turn

    def _last_human_index(self) -> int:
        for index in range(len(self.messages) - 1, -1, -1):
            if isinstance(self.messages[index], HumanMessage):
                return index
        return 0

    def _recent_conversation(self, turns: int = 4, max_chars: int = 400) -> str:
        previous = [
            m for m in self.messages[:-1]
            if isinstance(m, (HumanMessage, AIMessage)) and m.text
        ][-turns:]
        if not previous:
            return "(none)"
        return "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text[:max_chars]}"
            for m in previous
        )

    def _documents_label(self) -> str:
        return ", ".join(self.store.sources) or "none"
