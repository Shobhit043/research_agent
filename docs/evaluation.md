# Evaluation

The evaluation harness in [`eval/`](../eval/) runs the real agent against questions with known
answers and scores what it produces. It serves as a regression baseline: run it before and after
changing prompts, retrieval or models.

## Method

**Corpus.** Three fictional documents, so the model can't answer from memory:
[an annual report PDF](../eval/corpus/northwind_annual_report_2025.pdf) (4 pages), a
[battery whitepaper](../eval/corpus/helios_battery_whitepaper.md) and an
[HR policy](../eval/corpus/remote_work_policy.txt).

**Questions.** [17 cases](../eval/dataset.jsonl), each with a reference answer, gold source pages
and the expected routing:

| Category | n | Tests |
|---|---|---|
| fact | 6 | Direct lookups |
| paraphrase | 4 | Questions worded differently from the source ("workforce" vs "headcount") |
| multi-hop | 3 | Combining facts or computing from them |
| unanswerable | 2 | The information isn't in the documents; the correct answer says so |
| direct | 2 | Small talk and arithmetic that need no retrieval |

**Metrics.**

| Metric | How it's measured |
|---|---|
| Correctness | An LLM judge compares the answer with the reference: correct 1, partial 0.5, incorrect 0 |
| Faithfulness | The judge splits the answer into claims and checks each against the retrieved text (the RAGAS approach) |
| Retrieval recall | Share of gold source pages the tools actually returned |
| Cites gold | Whether the answer cites a gold source |
| Citation precision | Share of the model's document citations that passed verification |
| Routing accuracy | Whether the research/direct decision matched the expected one |

The judge is `openai/gpt-oss-120b`, a larger model than the agent, to reduce self-preference.
Retrieval recall and citation precision are computed in code, not by the judge.

## Results

Run on 2026-09-24 (`gpt-oss-20b` agent, `gpt-oss-120b` judge):

| Retrieval | Correct | Faithful | Retrieval recall | Cites gold | Citation precision | Routing |
|---|---|---|---|---|---|---|
| **Hybrid (BM25 + embeddings)** | **100%** | **94%** | **100%** | **92%** | 100% | 100% |
| BM25 only | 88% | 83% | 92% | 85% | 100% | 100% |

Per-case reports: [hybrid](../eval/results/hybrid.md), [BM25](../eval/results/bm25.md).

What the numbers show:
- **Hybrid retrieval fixes what keyword search misses.** BM25 failed "How much did Northwind spend
  on R&D?" because "R&D" shares no terms with "research and development spending".
- **Citation verification held in both modes.** No fabricated document citation reached an answer.
- **Faithfulness is the remaining risk.** Both lost points in the hybrid run are answers that added
  general knowledge the documents don't state. For example, "80% capacity is typically where
  degradation becomes noticeable" was attached to a correct figure.
- **The harness found real bugs.** The BM25 run exposed a generic fallback answer when the tool
  budget ran out, and an unrecognised Groq streaming error. Both are fixed and covered by tests.
  These numbers predate those fixes.

Caveats: 17 questions is a small set, the corpus is short and synthetic, and LLM judges have their
own error rate. Latencies in the reports are inflated, because the agent and the judge share one
free-tier rate limit; unthrottled answers took about 2 s. The dataset predates the web, time and
place tools, which it doesn't exercise.

## Running it

```bash
python -m eval.run_eval                          # hybrid retrieval, all cases
python -m eval.run_eval --no-embeddings          # BM25-only baseline
python -m eval.run_eval --resume                 # continue after an interruption
python -m eval.run_eval --categories paraphrase  # a subset (written as <label>-subset.*)
python -m eval.run_eval --judge-model openai/gpt-oss-120b --label my-experiment
```

Results go to `eval/results/<label>.jsonl` (one row per case, written as it goes) and
`<label>.md` (the report). A new full run moves the previous `.jsonl` to `.prev.jsonl` rather
than deleting it, and subset runs never overwrite the full report.

A full run of 17 cases uses roughly 100K Groq tokens, including the judge. On the free tier
(200K tokens per day) it may stop on rate limits; rerun with `--resume`.

## Adding cases

Add a line to [`eval/dataset.jsonl`](../eval/dataset.jsonl):

```json
{"id": "nw-plant", "category": "fact", "question": "Where will Northwind's new plant be?",
 "reference": "Monterrey, Mexico, opening in Q3 2026.",
 "gold_sources": ["northwind_annual_report_2025.pdf p.3"], "expect_retrieval": true}
```

`tests/test_eval.py` checks that every case is well formed and that its gold sources exist in the
corpus. To change the PDF, edit `eval/build_corpus.py` and run `python -m eval.build_corpus`.
