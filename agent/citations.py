import re

from agent.schemas import Source

_EXT = r"\.(?:pdf|txt|md)"
# [file.pdf p.3] or [notes.md], plus gpt-oss's habit of using 【】. Not markdown link text.
_DOC_CITATION = re.compile(
    rf"[\[【]\s*([^\[\]【】\n]+?{_EXT})(?:\s+p\.\s*(\d+))?\s*[\]】](?!\()", re.IGNORECASE
)
_WEB_CITATION = re.compile(r"【([WAUP]?\d+)[^】]*】")
# Anything else in 【】 is a gpt-oss artefact (e.g. 【get_time】), not a real source.
_STRAY_MARKER = re.compile(r"\s?【[^】\n]*】")
# Headers printed by the tools above each passage / result.
_EVIDENCE_HEADER = re.compile(rf"^\[(.+?{_EXT})(?: p\.(\d+))?\]$", re.MULTILINE | re.IGNORECASE)
# [3] from web_search, [W1] Wikipedia, [A1] arXiv, [U1] fetch_url, [P1] place_info.
_WEB_RESULT = re.compile(r"^\[([WAUP]?\d+)\] (.*)\nURL: (\S+)", re.MULTILINE)

Page = tuple[str, int | None]
_MAX_LABEL = 120


def _page(value: str | None) -> int | None:
    return int(value) if value else None


def evidence_pages(tool_outputs: list[str]) -> set[Page]:
    """(file, page) pairs that tools actually returned; page is None for text files."""
    return {
        (name.lower(), _page(page))
        for text in tool_outputs
        for name, page in _EVIDENCE_HEADER.findall(text)
    }


def verify_citations(answer: str, evidence: set[Page]) -> tuple[str, list[str]]:
    """Normalise citation brackets and drop document citations with no supporting evidence.

    Returns the cleaned answer and the citations that were removed.
    """
    removed: list[str] = []

    def check(match: re.Match) -> str:
        name, page = match.group(1).strip(), _page(match.group(2))
        if (name.lower(), page) in evidence:
            return f"[{name} p.{page}]" if page is not None else f"[{name}]"
        removed.append(match.group(0))
        return ""

    cleaned = _DOC_CITATION.sub(check, answer)
    cleaned = _WEB_CITATION.sub(r"[\1]", cleaned)
    cleaned = _STRAY_MARKER.sub("", cleaned)
    if removed:
        cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return cleaned, removed


def collect_sources(tool_outputs: list[str]) -> list[Source]:
    """Every document passage and web page the tools returned this turn, deduplicated."""
    sources: dict[str, Source] = {}
    for text in tool_outputs:
        for name, page in _EVIDENCE_HEADER.findall(text):
            label = f"{name} p.{page}" if page else name
            sources.setdefault(label, Source(kind="document", label=label))
        for ref, title, url in _WEB_RESULT.findall(text):
            label = (title.strip() or url)[:_MAX_LABEL]
            sources.setdefault(url, Source(kind="web", label=label, url=url, ref=ref))
    return list(sources.values())
