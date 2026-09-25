"""Deterministic metrics computed from a TurnResult. LLM-judged metrics live in judge.py."""

import re

from agent.schemas import TurnResult

_DOC_CITATION = re.compile(r"\[([^\[\]\n]+?\.(?:pdf|txt|md))(?: p\.(\d+))?\]", re.IGNORECASE)


def cited_labels(answer: str) -> list[str]:
    return [f"{name} p.{page}" if page else name for name, page in _DOC_CITATION.findall(answer)]


def retrieval_recall(result: TurnResult, gold: list[str]) -> float | None:
    """Share of gold sources that the tools actually returned. None if there is no gold."""
    if not gold:
        return None
    retrieved = {s.label.lower() for s in result.sources if s.kind == "document"}
    return sum(g.lower() in retrieved for g in gold) / len(gold)


def citation_precision(result: TurnResult) -> float | None:
    """Share of the model's document citations that survived verification."""
    kept = len(cited_labels(result.answer))
    total = kept + len(result.removed_citations)
    return kept / total if total else None


def cites_gold(result: TurnResult, gold: list[str]) -> bool | None:
    if not gold:
        return None
    cited = {label.lower() for label in cited_labels(result.answer)}
    return any(g.lower() in cited for g in gold)


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
