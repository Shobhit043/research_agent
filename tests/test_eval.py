import json
from pathlib import Path

from agent.schemas import QueryAnalysis, Source, TurnResult
from eval import metrics
from eval.run_eval import load_cases, render_report, summarise

CORPUS = Path(__file__).parent.parent / "eval" / "corpus"


def turn(answer, sources=(), removed=()):
    return TurnResult(
        answer=answer,
        analysis=QueryAnalysis(needs_retrieval=True),
        tool_calls=[],
        sources=[Source(kind="document", label=s) for s in sources],
        removed_citations=list(removed),
    )


def test_retrieval_recall_counts_gold_sources_returned():
    result = turn("x", sources=["a.pdf p.1", "b.md"])

    assert metrics.retrieval_recall(result, ["a.pdf p.1", "a.pdf p.4"]) == 0.5
    assert metrics.retrieval_recall(result, []) is None


def test_citation_precision_uses_removed_citations():
    result = turn("Fact [a.pdf p.1] and [b.md].", removed=["[a.pdf p.9]"])

    assert metrics.citation_precision(result) == 2 / 3
    assert metrics.citation_precision(turn("no citations")) is None


def test_cites_gold():
    assert metrics.cites_gold(turn("x [a.pdf p.1]"), ["a.pdf p.1"]) is True
    assert metrics.cites_gold(turn("x [a.pdf p.2]"), ["a.pdf p.1"]) is False


def test_dataset_is_well_formed_and_gold_sources_exist():
    cases = load_cases(None, None)
    files = {p.name for p in CORPUS.iterdir()} | {"northwind_annual_report_2025.pdf"}

    assert len({c["id"] for c in cases}) == len(cases) >= 15
    for case in cases:
        assert {"question", "reference", "category", "gold_sources", "expect_retrieval"} <= case.keys()
        for gold in case["gold_sources"]:
            assert gold.split(" p.")[0] in files, gold


def test_summary_and_report_render():
    rows = [
        {"id": "a", "category": "fact", "question": "q", "answer": "A | b", "correctness": 1.0,
         "correctness_reason": "", "faithfulness": 1.0, "unsupported_claims": [], "retrieval_recall": 1.0,
         "cites_gold": True, "citation_precision": 1.0, "route_correct": True, "latency_ms": 1000, "tokens": 900},
        {"id": "b", "category": "direct", "question": "q", "answer": "B", "correctness": 0.5,
         "correctness_reason": "missed a fact", "faithfulness": None, "unsupported_claims": ["x"],
         "retrieval_recall": None, "cites_gold": None, "citation_precision": None, "route_correct": False,
         "latency_ms": 3000, "tokens": 100},
    ]

    summary = summarise(rows)
    report = render_report("test", {"model": "m", "judge_model": "j", "hybrid": True}, summary, rows)

    assert summary["overall"]["correctness"] == 0.75
    assert summary["overall"]["routing_accuracy"] == 0.5
    assert "| **overall** | 2 | 75% |" in report
    assert "A \\| b" in report and "missed a fact" in report
    json.dumps(summary)
