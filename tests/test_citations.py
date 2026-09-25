from agent.citations import collect_sources, evidence_pages, verify_citations

TOOL_OUTPUT = "[acme.pdf p.2]\nThe main risk is supplier concentration.\n\n[acme.pdf p.1]\nRevenue grew."


def test_evidence_pages_reads_search_documents_headers():
    assert evidence_pages([TOOL_OUTPUT]) == {("acme.pdf", 2), ("acme.pdf", 1)}


def test_backed_citations_are_kept_and_brackets_normalised():
    text, removed = verify_citations("Risk is suppliers【acme.pdf p.2】.", evidence_pages([TOOL_OUTPUT]))

    assert text == "Risk is suppliers[acme.pdf p.2]."
    assert removed == []


def test_fabricated_citations_are_removed():
    answer = "Use nicotine gum [acme.pdf p.12]. Revenue grew [acme.pdf p.1]."

    text, removed = verify_citations(answer, evidence_pages([TOOL_OUTPUT]))

    assert text == "Use nicotine gum. Revenue grew [acme.pdf p.1]."
    assert removed == ["[acme.pdf p.12]"]


def test_all_document_citations_removed_when_nothing_was_retrieved():
    text, removed = verify_citations("Fact [paper.pdf p.3].", set())

    assert text == "Fact."
    assert removed


def test_text_file_citations_have_no_page():
    evidence = evidence_pages(["[notes.md]\nGroq is fast."])

    text, removed = verify_citations("Fast [notes.md]. Slow [notes.md p.2].", evidence)

    assert text == "Fast [notes.md]. Slow."
    assert removed == ["[notes.md p.2]"]


def test_markdown_links_are_not_treated_as_citations():
    answer = "See [the readme.md](https://example.com/readme.md)."

    assert verify_citations(answer, set()) == (answer, [])


def test_collect_sources_lists_documents_and_web_pages_once():
    web = "[1] Groq docs\nURL: https://groq.com/docs\nbody\n\n[2] \nURL: https://x.io/a\nbody"

    sources = collect_sources([TOOL_OUTPUT, web, web])

    assert [(s.kind, s.label, s.url) for s in sources] == [
        ("document", "acme.pdf p.2", None),
        ("document", "acme.pdf p.1", None),
        ("web", "Groq docs", "https://groq.com/docs"),
        ("web", "https://x.io/a", "https://x.io/a"),
    ]


def test_web_citations_are_normalised_not_removed():
    text, removed = verify_citations("Spain won【2†L4-L9】.", set())

    assert text == "Spain won[2]."
    assert removed == []


def test_stray_gpt_oss_markers_are_removed():
    text, _ = verify_citations("It is 9 am【get_time】. Born 1912【W1†L3】.", set())

    assert text == "It is 9 am. Born 1912[W1]."
