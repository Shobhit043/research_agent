"""The research agent, orchestrated as a LangGraph state machine.

    START → analyze ─┬─ direct ──────────────────────────┬─→ END
                     │     └─ (model wanted a tool) ─┐   │
                     └─────────────────────────────→ research ⇄ tools
                                                     └─ (budget spent) → finalize → END

Each node is one step: routing, a model call, or a batch of tool calls. Nodes stream progress
through LangGraph's custom stream writer, and `ResearchAgent.astream` turns those into
`AgentEvent`s. The graph works on a copy of the conversation; history is committed only when
a turn completes, so a failed, cancelled or abandoned turn leaves no trace.
"""

import asyncio
import datetime
import json
import logging
import operator
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, TypedDict

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
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from agent.citations import collect_sources, evidence_pages, verify_citations
from agent.config import Settings
from agent.documents import DocumentStore, ingest_document
from agent.guardrails import scan_for_injection, wrap_untrusted
from agent.prompts import AGENT_SYSTEM_PROMPT, BUDGET_EXHAUSTED, NO_TOOLS, QUERY_ANALYSIS_PROMPT
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

Route = Literal["direct", "research", "tools", "finalize", "end"]
Writer = Callable[[dict], None]


def _is_tool_use_failed(exc: groq.APIError) -> bool:
    # Before streaming starts Groq returns a 400 with code tool_use_failed; once a stream is
    # under way the same failure arrives as a bare APIError: "... but model called a tool".
    message = str(exc)
    return "tool_use_failed" in message or "model called a tool" in message


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@dataclass
class TurnContext:
    """Per-turn bookkeeping shared by the nodes (LangGraph runtime context, not graph state)."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started: float = field(default_factory=time.perf_counter)
    usage: Usage = field(default_factory=Usage)
    durations: dict[str, int] = field(default_factory=dict)
    flagged: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


class TurnState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # prior history + this turn
    question: str
    analysis: QueryAnalysis
    variables: dict  # prompt variables: documents, analysis, today
    rounds: int  # research rounds used, including rejected ones
    seen: Annotated[list[str], operator.add]  # tool calls already run this turn (for dedup)
    next: Route  # set by each node; read by the conditional edges


class ResearchAgent:
    """Tool-calling research assistant: analyse the query, gather evidence, answer with citations.

    `astream` is the core: an async generator of progress events ending in `done`.
    `arun`, `run` and `ask` are conveniences on top of it. `graph` is the compiled LangGraph.
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
        self.graph = self._build_graph()

    # ----- documents and history -----

    def add_document(self, path: str | Path, name: str | None = None) -> int:
        chunks = ingest_document(
            path, self.settings.chunk_size, self.settings.chunk_overlap, name=name,
            ocr=self.settings.ocr_enabled, ocr_max_pages=self.settings.ocr_max_pages,
        )
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
        context = TurnContext()
        history_length = len(self.messages)
        initial: TurnState = {
            "messages": [*self.messages, HumanMessage(question)],
            "question": question,
            "rounds": 0,
            "seen": [],
        }
        config = {
            # Every research round is two graph steps (model, tools); leave room for retries.
            "recursion_limit": 4 * self.settings.max_tool_iterations + 10,
            **self._run_config(context, "research_turn"),
        }
        final: TurnState | None = None
        async for mode, payload in self.graph.astream(
            initial, config=config, context=context, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                yield AgentEvent(**payload)
            else:
                final = payload

        # Commit only now: if the graph raised or the consumer stopped iterating (the browser
        # disconnected), we never get here and history is untouched.
        self.messages = list(final["messages"])
        yield AgentEvent(type="done", result=self._finish_turn(history_length, final["analysis"], context))

    # ----- graph -----

    def _build_graph(self):
        graph = StateGraph(TurnState, context_schema=TurnContext)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("direct", self._direct_node)
        graph.add_node("research", self._research_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("finalize", self._finalize_node)

        def route(state: TurnState) -> str:
            return state["next"]

        graph.add_edge(START, "analyze")
        graph.add_conditional_edges("analyze", route, {"direct": "direct", "research": "research"})
        graph.add_conditional_edges("direct", route, {"end": END, "research": "research"})
        graph.add_conditional_edges(
            "research", route, {"tools": "tools", "research": "research", "finalize": "finalize", "end": END}
        )
        graph.add_edge("tools", "research")
        graph.add_edge("finalize", END)
        return graph.compile(name="research_agent")

    async def _analyze_node(self, state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        """Route the question: research with tools, or answer directly."""
        context = runtime.context
        analysis = await self._analyze(state["question"], state["messages"][:-1], context)
        logger.info("Query analysis [%s]:\n%s", context.trace_id, analysis.render())
        get_stream_writer()({"type": "analysis", "data": analysis.model_dump()})
        variables = {
            "documents": self._documents_label(),
            "analysis": analysis.render(),
            "today": datetime.date.today().isoformat(),
        }
        return {
            "analysis": analysis,
            "variables": variables,
            "next": "research" if analysis.needs_retrieval else "direct",
        }

    async def _direct_node(self, state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        """Answer without tools (small talk, general knowledge)."""
        writer = get_stream_writer()
        try:
            message = await self._call_model(
                self._prompt | self.llm, {**state["variables"], "tool_guidance": NO_TOOLS},
                state["messages"], runtime.context, writer,
            )
        except groq.APIError as exc:
            if not _is_tool_use_failed(exc):
                raise
            # The model tried to call a tool, so the router was wrong: research instead.
            logger.info("Model requested a tool on the direct path; switching to research")
            writer({"type": "reset"})
            return {"next": "research"}
        return {"messages": [message], "next": "end"}

    async def _research_node(self, state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        """One model call with tools bound: it either requests tools or answers."""
        rounds = state.get("rounds", 0)
        if rounds >= self.settings.max_tool_iterations:
            logger.warning("Tool budget of %d rounds exhausted; forcing a final answer", rounds)
            return {"next": "finalize"}

        tools = self._bound_tools()
        writer = get_stream_writer()
        try:
            message = await self._call_model(
                self._prompt | self.llm.bind_tools(tools),
                {**state["variables"], "tool_guidance": tool_guidance(tools)},
                state["messages"], runtime.context, writer,
            )
        except groq.APIError as exc:
            if not _is_tool_use_failed(exc):
                raise
            # e.g. gpt-oss calling its built-in browser tool; retry the round (it counts
            # against the budget, so this can't loop forever).
            logger.warning("Malformed tool call rejected by Groq; retrying the round: %s", exc)
            writer({"type": "reset"})
            return {"rounds": rounds + 1, "next": "research"}
        return {"messages": [message], "rounds": rounds + 1, "next": "tools" if message.tool_calls else "end"}

    async def _tools_node(self, state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        """Run every tool call from the last model message, concurrently."""
        context = runtime.context
        writer = get_stream_writer()
        calls = state["messages"][-1].tool_calls
        tools_by_name = {t.name: t for t in self._bound_tools()}
        seen = set(state.get("seen", []))
        before = set(seen)

        for call in calls:
            writer({"type": "tool_start", "data": {"id": call["id"], "name": call["name"], "args": call["args"]}})
        # gather keeps results in call order; every call gets exactly one ToolMessage reply.
        results = await asyncio.gather(
            *(self._run_tool_call(call, tools_by_name, seen, context) for call in calls)
        )
        for call in calls:
            writer({"type": "tool_end", "data": {
                "id": call["id"],
                "name": call["name"],
                "duration_ms": context.durations.get(call["id"], 0),
                "flagged": call["id"] in context.flagged,
            }})
        return {"messages": list(results), "seen": sorted(seen - before)}

    async def _finalize_node(self, state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        """The tool budget is spent: answer from what was gathered, without tools."""
        writer = get_stream_writer()
        variables = {**state["variables"], "tool_guidance": BUDGET_EXHAUSTED}
        # gpt-oss sometimes still reaches for a tool here. A user-turn nudge is a stronger signal
        # than the system prompt; it goes to this one call only and is never stored in history.
        for nudge in (None, HumanMessage(_FINAL_ANSWER_NUDGE)):
            try:
                message = await self._call_model(
                    self._prompt | self.llm, variables, state["messages"], runtime.context, writer, nudge
                )
                return {"messages": [message]}
            except groq.APIError as exc:
                if not _is_tool_use_failed(exc):
                    raise
                writer({"type": "reset"})
        return {"messages": [AIMessage(_BUDGET_FALLBACK)]}

    # ----- steps used by the nodes -----

    def _bound_tools(self) -> list[BaseTool]:
        has_documents = bool(self.store.sources)
        return [t for name, t in self._tools.items() if has_documents or name not in DOCUMENT_TOOLS]

    async def _analyze(self, question: str, previous: list[BaseMessage], context: TurnContext) -> QueryAnalysis:
        fallback = QueryAnalysis(needs_retrieval=True, sub_questions=[question])
        try:
            output = await self._analyzer.ainvoke(
                {
                    "question": question,
                    "documents": self._documents_label(),
                    "recent": self._recent_conversation(previous),
                },
                config=self._run_config(context, "query_analysis"),
            )
        except Exception as exc:
            # Routing is an optimisation; if it fails, researching is the safe default.
            logger.warning("Query analysis failed (%s); defaulting to retrieval", exc)
            return fallback
        raw = output.get("raw")
        context.usage.add(getattr(raw, "usage_metadata", None))
        return output.get("parsed") or fallback

    async def _call_model(
        self, chain, variables: dict, messages: list[BaseMessage], context: TurnContext, writer: Writer,
        nudge: BaseMessage | None = None,
    ) -> AIMessage:
        """Stream one model call, emitting token events, and return the complete message.

        No tools bound means the model *can't* call one; `tool_choice="none"` would instead get
        a 400 (tool_use_failed) from Groq whenever gpt-oss emits a tool call anyway.
        """
        aggregate = None
        streamed = False
        history = self._history(messages) + ([nudge] if nudge else [])
        async for chunk in chain.astream(
            {**variables, "history": history}, config=self._run_config(context, "agent_step")
        ):
            aggregate = chunk if aggregate is None else aggregate + chunk
            if chunk.text:
                streamed = True
                writer({"type": "token", "data": {"text": chunk.text}})

        message = message_chunk_to_message(aggregate) if aggregate is not None else AIMessage("")
        context.usage.add(message.usage_metadata)
        if message.tool_calls and streamed:
            # Text streamed before the model decided to call tools isn't the answer.
            writer({"type": "reset"})
        return message

    async def _run_tool_call(
        self, call: dict, tools: dict[str, BaseTool], seen: set[str], context: TurnContext
    ) -> ToolMessage:
        # Every tool call must get a ToolMessage reply, or the next API request is rejected.
        name, args = call["name"], call["args"]
        key = json.dumps([name, args], sort_keys=True, default=str)
        started = time.perf_counter()

        if name not in tools:
            raw = f"Error: unknown tool '{name}'. Available tools: {', '.join(tools)}."
        elif key in seen:
            raw = "Skipped: this exact call already ran this turn. Use its earlier result."
        else:
            seen.add(key)
            logger.info("Tool call [%s]: %s(%s)", context.trace_id, name, args)
            try:
                result = await asyncio.wait_for(
                    tools[name].ainvoke(args, config=self._run_config(context, name)),
                    timeout=self.settings.tool_timeout,
                )
                raw = str(result)
            except TimeoutError:
                raw = f"Error: {name} timed out after {self.settings.tool_timeout:.0f}s."
            except Exception as exc:
                # Report failures back to the model so it can retry or answer without them.
                logger.warning("Tool %s failed: %s", name, exc)
                raw = f"Error: {name} failed: {exc}"

        context.durations[call["id"]] = _elapsed_ms(started)
        injection = scan_for_injection(raw)
        if injection:
            context.flagged.add(call["id"])
            context.warnings.append(
                f"Possible prompt injection in {name} result ({injection[0]!r}); treated as data."
            )
            logger.warning("Possible prompt injection in %s result: %s", name, injection)
        return ToolMessage(
            content=wrap_untrusted(name, raw, bool(injection)),
            artifact=raw,
            name=name,
            tool_call_id=call["id"],
        )

    def _finish_turn(self, turn_start: int, analysis: QueryAnalysis, context: TurnContext) -> TurnResult:
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
                duration_ms=context.durations.get(call["id"], 0),
                flagged=call["id"] in context.flagged,
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
            warnings=context.warnings,
            usage=context.usage,
            latency_ms=_elapsed_ms(context.started),
            trace_id=context.trace_id,
        )

    # ----- helpers -----

    def _run_config(self, context: TurnContext, run_name: str) -> dict:
        # Shows up in LangSmith when tracing is enabled; harmless otherwise.
        return {
            "run_name": run_name,
            "metadata": {"trace_id": context.trace_id, "session_id": self.session_id},
            "tags": ["research-assistant"],
        }

    def _history(self, messages: list[BaseMessage] | None = None) -> list[BaseMessage]:
        messages = self.messages if messages is None else messages
        trimmed = trim_messages(
            messages,
            max_tokens=self.settings.history_token_budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        if trimmed:
            return trimmed

        # The current turn alone is over budget: shrink its oldest tool results until it fits.
        turn = list(messages[_last_human_index(messages):])
        for index, message in enumerate(turn):
            if count_tokens_approximately(turn) <= self.settings.history_token_budget:
                break
            if isinstance(message, ToolMessage) and len(message.content) > _COMPACT_CHARS:
                turn[index] = message.model_copy(update={
                    "content": message.content[:_COMPACT_CHARS] + "\n...[truncated to fit context]"
                })
        return turn

    @staticmethod
    def _recent_conversation(previous: list[BaseMessage], turns: int = 4, max_chars: int = 400) -> str:
        recent = [m for m in previous if isinstance(m, (HumanMessage, AIMessage)) and m.text][-turns:]
        if not recent:
            return "(none)"
        return "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text[:max_chars]}"
            for m in recent
        )

    def _documents_label(self) -> str:
        return ", ".join(self.store.sources) or "none"


def _last_human_index(messages: list[BaseMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return 0
