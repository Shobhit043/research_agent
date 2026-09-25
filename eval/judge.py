"""LLM-as-judge metrics, following the RAGAS approach: faithfulness splits the answer into
claims and checks each against the retrieved context; correctness compares to a reference.

The judge is a different (larger) model than the agent to reduce self-preference bias.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_MAX_CONTEXT_CHARS = 6000


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim] = Field(
        description="Atomic factual claims made by the answer, each checked against the context."
    )


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str = Field(description="One sentence.")


FAITHFULNESS_PROMPT = """You check whether an answer is supported by retrieved context.

Split the ANSWER into atomic factual claims. Ignore greetings, citations, and statements
that information is unavailable. For each claim, decide whether the CONTEXT supports it.
Simple arithmetic on figures stated in the context counts as supported.

CONTEXT:
{context}

ANSWER:
{answer}"""

CORRECTNESS_PROMPT = """You grade an answer against a reference answer.

- correct: contains the key facts of the reference and nothing that contradicts it.
- partial: some key facts are right but others are missing or wrong.
- incorrect: wrong, or missing the key facts.
If the reference says the information is unavailable, the answer is correct only if it
says it could not find that information, and incorrect if it invents an answer.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""

_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judge:
    def __init__(self, llm: BaseChatModel):
        self._faithfulness = ChatPromptTemplate.from_messages(
            [("human", FAITHFULNESS_PROMPT)]
        ) | llm.with_structured_output(FaithfulnessVerdict, method="json_schema")
        self._correctness = ChatPromptTemplate.from_messages(
            [("human", CORRECTNESS_PROMPT)]
        ) | llm.with_structured_output(CorrectnessVerdict, method="json_schema")

    def faithfulness(self, answer: str, context: str) -> tuple[float | None, list[dict]]:
        """Share of claims supported by the context; None when there's no context or no claims."""
        if not context.strip():
            return None, []
        verdict = self._faithfulness.invoke({"answer": answer, "context": context[:_MAX_CONTEXT_CHARS]})
        claims = [c.model_dump() for c in verdict.claims]
        if not claims:
            return None, []
        return sum(c["supported"] for c in claims) / len(claims), claims

    def correctness(self, question: str, reference: str, answer: str) -> tuple[float, str]:
        verdict = self._correctness.invoke({"question": question, "reference": reference, "answer": answer})
        return _SCORES[verdict.verdict], verdict.reason
