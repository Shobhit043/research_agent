"""Evaluate the agent on eval/dataset.jsonl against the fictional corpus in eval/corpus.

    python -m eval.run_eval                     # hybrid retrieval, all cases
    python -m eval.run_eval --no-embeddings     # BM25 only, for comparison
    python -m eval.run_eval --resume            # continue after a rate-limit interruption
    python -m eval.run_eval --categories paraphrase --label bm25-paraphrase --no-embeddings

Results go to eval/results/<label>.jsonl (one row per case) and <label>.md (report).
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import groq
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq

from agent.config import Settings
from agent.documents import DocumentStore, FastEmbedEmbedder, ingest_document
from agent.research_agent import ResearchAgent
from eval import metrics
from eval.build_corpus import CORPUS
from eval.build_corpus import main as build_corpus
from eval.judge import Judge

HERE = Path(__file__).parent
T = TypeVar("T")


def load_cases(categories: set[str] | None, ids: set[str] | None) -> list[dict]:
    lines = (HERE / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    return [
        c for c in cases
        if (not categories or c["category"] in categories) and (not ids or c["id"] in ids)
    ]


def build_store(settings: Settings, use_embeddings: bool) -> DocumentStore:
    if not (CORPUS / "northwind_annual_report_2025.pdf").exists():
        build_corpus()
    store = DocumentStore(FastEmbedEmbedder(settings.embedding_model) if use_embeddings else None)
    for path in sorted(CORPUS.iterdir()):
        store.add(ingest_document(path, settings.chunk_size, settings.chunk_overlap))
    print(f"Indexed {len(store)} chunks from {len(store.sources)} files (hybrid={store.hybrid})")
    return store


async def with_retries(make_call: Callable[[], Awaitable[T]], attempts: int = 5) -> T:
    """Groq's free tier rate-limits per minute; wait it out rather than failing the run."""
    for attempt in range(attempts - 1):
        try:
            return await make_call()
        except groq.RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s")
            await asyncio.sleep(wait)
    return await make_call()


async def run_case(case: dict, store: DocumentStore, settings: Settings, judge: Judge) -> dict:
    agent = ResearchAgent(settings=settings, store=store)
    result = await with_retries(lambda: agent.arun(case["question"]))
    context = "\n\n".join(
        m.artifact if isinstance(m.artifact, str) else m.text
        for m in agent.messages
        if isinstance(m, ToolMessage)
    )
    faithfulness, claims = await with_retries(
        lambda: asyncio.to_thread(judge.faithfulness, result.answer, context)
    )
    correctness, reason = await with_retries(
        lambda: asyncio.to_thread(judge.correctness, case["question"], case["reference"], result.answer)
    )
    gold = case["gold_sources"]
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": result.answer,
        "correctness": correctness,
        "correctness_reason": reason,
        "faithfulness": faithfulness,
        "unsupported_claims": [c["claim"] for c in claims if not c["supported"]],
        "retrieval_recall": metrics.retrieval_recall(result, gold),
        "cites_gold": metrics.cites_gold(result, gold),
        "citation_precision": metrics.citation_precision(result),
        "route_correct": result.analysis.needs_retrieval == case["expect_retrieval"],
        "removed_citations": result.removed_citations,
        "tool_calls": [f"{c.name}({json.dumps(c.args)})" for c in result.tool_calls],
        "latency_ms": result.latency_ms,
        "tokens": result.usage.total_tokens,
    }


def summarise(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in rows)

    def stats(group: list[dict]) -> dict:
        return {
            "n": len(group),
            "correctness": metrics.mean([r["correctness"] for r in group]),
            "faithfulness": metrics.mean([r["faithfulness"] for r in group]),
            "retrieval_recall": metrics.mean([r["retrieval_recall"] for r in group]),
            "citation_precision": metrics.mean([r["citation_precision"] for r in group]),
            "cites_gold": metrics.mean([None if r["cites_gold"] is None else float(r["cites_gold"]) for r in group]),
            "routing_accuracy": metrics.mean([float(r["route_correct"]) for r in group]),
        }

    categories = sorted({r["category"] for r in rows})
    return {
        "overall": stats(rows),
        "by_category": {c: stats([r for r in rows if r["category"] == c]) for c in categories},
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        "avg_tokens": metrics.mean([r["tokens"] for r in rows]),
    }


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def render_report(label: str, settings: dict, summary: dict, rows: list[dict]) -> str:
    columns = [
        "correctness", "faithfulness", "retrieval_recall", "cites_gold", "citation_precision", "routing_accuracy",
    ]
    header = "| Group | n | Correct | Faithful | Retrieval recall | Cites gold | Citation precision | Routing |"
    lines = [
        f"# Evaluation: {label}",
        "",
        f"Agent `{settings['model']}` · judge `{settings['judge_model']}` · "
        f"hybrid retrieval: {settings['hybrid']} · {len(rows)} cases",
        "",
        header,
        "|---|---|---|---|---|---|---|---|",
    ]
    groups = [("**overall**", summary["overall"]), *summary["by_category"].items()]
    for name, group in groups:
        lines.append(f"| {name} | {group['n']} | " + " | ".join(_pct(group[c]) for c in columns) + " |")
    lines += [
        "",
        f"Latency p50 {summary['latency_p50_ms'] / 1000:.1f} s · p95 {summary['latency_p95_ms'] / 1000:.1f} s · "
        f"avg {summary['avg_tokens']:.0f} tokens per question",
        "",
        "## Cases",
        "",
        "| Case | Correct | Faithful | Recall | Answer |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        answer = r["answer"].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {_pct(r['correctness'])} | {_pct(r['faithfulness'])} | "
            f"{_pct(r['retrieval_recall'])} | {answer[:160]}{'…' if len(answer) > 160 else ''} |"
        )
    issues = [r for r in rows if r["correctness"] < 1 or r["unsupported_claims"]]
    if issues:
        lines += [
            "", "## Issues", "",
            "Incorrect answers, and correct answers containing a claim the context doesn't support.", "",
        ]
        for r in issues:
            lines.append(f"- **{r['id']}**: {r['correctness_reason']}")
            for claim in r["unsupported_claims"]:
                lines.append(f"  - unsupported claim: {claim}")
    return "\n".join(lines) + "\n"


async def run_all(
    cases: list[dict], done: dict[str, dict], store: DocumentStore, settings: Settings,
    judge: Judge, results_path: Path,
) -> None:
    for number, case in enumerate(cases, start=1):
        if case["id"] in done:
            continue
        print(f"[{number}/{len(cases)}] {case['id']}: {case['question']}")
        row = await run_case(case, store, settings, judge)
        print(f"    correct={row['correctness']} faithful={row['faithfulness']} "
              f"recall={row['retrieval_recall']} {row['latency_ms']} ms")
        done[case["id"]] = row
        with results_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", help="results file name (default: hybrid or bm25)")
    parser.add_argument("--no-embeddings", action="store_true", help="BM25 only")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--categories", help="comma-separated subset, e.g. fact,paraphrase")
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--resume", action="store_true", help="skip cases already in the results file")
    args = parser.parse_args()

    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)

    settings = Settings.from_env()
    hybrid = not args.no_embeddings
    subset = bool(args.categories or args.ids)
    # A partial run must never overwrite the full report, so subsets get their own label.
    label = args.label or ("hybrid" if hybrid else "bm25") + ("-subset" if subset else "")
    results_path = HERE / "results" / f"{label}.jsonl"
    results_path.parent.mkdir(exist_ok=True)

    cases = load_cases(
        set(args.categories.split(",")) if args.categories else None,
        set(args.ids.split(",")) if args.ids else None,
    )
    done: dict[str, dict] = {}
    if args.resume and results_path.exists():
        rows = map(json.loads, results_path.read_text(encoding="utf-8").splitlines())
        done = {row["id"]: row for row in rows}
    elif results_path.exists():
        # Keep the previous run until this one produces results; an interrupted run
        # (rate limits, Ctrl+C) shouldn't destroy the last good one.
        results_path.replace(results_path.with_suffix(".prev.jsonl"))

    store = build_store(settings, hybrid)
    judge = Judge(ChatGroq(model=args.judge_model, temperature=0, max_retries=3))

    # One event loop for the whole run: the Groq async client can't be reused across loops.
    asyncio.run(run_all(cases, done, store, settings, judge, results_path))

    rows = [done[c["id"]] for c in cases if c["id"] in done]
    summary = summarise(rows)
    report = render_report(label, {"model": settings.model, "judge_model": args.judge_model, "hybrid": hybrid},
                           summary, rows)
    (HERE / "results" / f"{label}.md").write_text(report, encoding="utf-8")
    print("\n" + report.split("## Cases")[0])


if __name__ == "__main__":
    main()
