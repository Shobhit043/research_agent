from typing import Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Routing decision for a user question."""

    needs_retrieval: bool = Field(
        description=(
            "True if answering requires searching the web or the user's uploaded "
            "documents; false for small talk, simple arithmetic, rephrasing earlier "
            "answers, or stable general knowledge."
        )
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description=(
            "The question decomposed into self-contained sub-questions. "
            "A single item if the question is already atomic."
        ),
    )

    def render(self) -> str:
        lines = [f"needs_retrieval: {self.needs_retrieval}"]
        if self.sub_questions:
            lines.append("sub-questions:")
            lines.extend(f"- {q}" for q in self.sub_questions)
        return "\n".join(lines)


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    output: str
    duration_ms: int = 0
    flagged: bool = False


class Source(BaseModel):
    kind: Literal["document", "web"]
    label: str
    url: str | None = None
    ref: str | None = None  # citation tag without brackets, e.g. "2", "W1", "A1"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, usage_metadata: dict | None) -> None:
        self.llm_calls += 1
        if usage_metadata:
            self.input_tokens += usage_metadata.get("input_tokens", 0)
            self.output_tokens += usage_metadata.get("output_tokens", 0)


class TurnResult(BaseModel):
    """One answered question plus the trace of how it was researched."""

    answer: str
    analysis: QueryAnalysis
    tool_calls: list[ToolCallRecord]
    sources: list[Source]
    removed_citations: list[str]
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0
    trace_id: str = ""


class AgentEvent(BaseModel):
    """Progress event streamed while a turn runs. The last one is always `done`."""

    type: Literal["analysis", "tool_start", "tool_end", "token", "reset", "done"]
    data: dict = Field(default_factory=dict)
    result: TurnResult | None = None
