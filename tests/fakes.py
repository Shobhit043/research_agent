import json
import re
from typing import ClassVar

import groq
import httpx
import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from agent.config import Settings
from agent.documents import DocumentStore
from agent.research_agent import ResearchAgent
from agent.schemas import QueryAnalysis

USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


class Script:
    """Iterator of scripted replies; exception items are raised, like a failing API call."""

    def __init__(self, items: list):
        # Keep the caller's list (not a copy) so replies can be added after the agent exists.
        self.items = items

    def __iter__(self):
        return self

    def __next__(self):
        if not self.items:
            raise StopIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ScriptedLLM(GenericFakeChatModel):
    """Replays scripted AI messages, streaming them the way Groq does, and records prompts."""

    analysis: QueryAnalysis = QueryAnalysis(needs_retrieval=True)
    bind_calls: list[dict] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append({"tools": [t.name for t in tools], **kwargs})
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: {
            "raw": AIMessage("", usage_metadata=USAGE),
            "parsed": self.analysis,
        })

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages[0].text)
        item = next(self.messages)
        message = AIMessage(content=item) if isinstance(item, str) else item

        for token in re.split(r"(\s)", message.text) if message.text else []:
            if token:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=USAGE))


class KeywordEmbedder:
    """Deterministic fake: one dimension per concept, so synonyms share a vector."""

    CONCEPTS: ClassVar[list[set[str]]] = [
        {"revenue", "sales", "turnover", "income"},
        {"risk", "threat", "danger"},
        {"staff", "employees", "headcount"},
    ]

    def _vector(self, text: str) -> np.ndarray:
        words = set(re.findall(r"[a-z]+", text.lower()))
        vector = np.array([len(words & concept) for concept in self.CONCEPTS] + [0.01], np.float32)
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


def tool_use_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.BadRequestError(
        "Error code: 400 - tool_use_failed",
        response=httpx.Response(400, request=request),
        body={"error": {"code": "tool_use_failed"}},
    )


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def make_agent(replies, analysis=None, settings=None, with_docs=True):
    llm = ScriptedLLM(messages=Script(replies))
    if analysis:
        llm.analysis = analysis
    store = DocumentStore()
    if with_docs:
        store.add([Document(page_content="Groq serves open models fast",
                            metadata={"source": "notes.pdf", "page": 2})])
    return ResearchAgent(llm=llm, settings=settings or Settings(), store=store), llm
