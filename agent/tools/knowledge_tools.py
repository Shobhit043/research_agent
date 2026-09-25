"""Keyless, reliable sources: Wikipedia for encyclopedic facts, arXiv for research papers."""

import json
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.tools.http import cache, safe_get

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_BYTES = 1_000_000


def _get(url: str, settings: Settings) -> bytes:
    cached = cache.get(("api", url))
    if cached is None:
        cached = safe_get(url, settings.request_timeout, _MAX_BYTES).body
        cache.set(("api", url), cached)
    return cached


def build_wikipedia_tool(settings: Settings) -> BaseTool:
    @tool
    def wikipedia_search(query: str) -> str:
        """
        Search Wikipedia and return the introductions of the best-matching articles.
        Use this for well-established facts: definitions, history, people, places,
        organisations, science. Prefer web_search for news or fast-changing topics.

        Args:
            query: A topic or short search phrase.
        """
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search", "gsrsearch": query, "gsrlimit": "3",
            "prop": "extracts|info", "exintro": "1", "explaintext": "1", "exlimit": "3",
            "inprop": "url", "redirects": "1",
        }
        data = json.loads(_get(f"{WIKIPEDIA_API}?{urlencode(params)}", settings))
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
        if not pages:
            return "No Wikipedia articles found."
        return "\n\n".join(
            f"[W{number}] {page['title']} (Wikipedia)\nURL: {page.get('fullurl', '')}\n"
            f"{page.get('extract', '').strip()[: settings.web_page_chars]}"
            for number, page in enumerate(pages, start=1)
        )

    return wikipedia_search


def build_arxiv_tool(settings: Settings) -> BaseTool:
    @tool
    def arxiv_search(query: str, max_results: int = 3) -> str:
        """
        Search arXiv for academic papers and return titles, authors, dates and abstracts.
        Use this for research literature in physics, maths, computer science, AI/ML,
        statistics, quantitative biology and finance.

        Args:
            query: Keywords describing the research topic.
            max_results: Number of papers (1-5).
        """
        count = max(1, min(max_results, 5))
        params = {"search_query": f"all:{query}", "max_results": str(count), "sortBy": "relevance"}
        root = ET.fromstring(_get(f"{ARXIV_API}?{urlencode(params)}", settings))
        entries = root.findall("atom:entry", _ATOM)
        if not entries:
            return "No arXiv papers found."
        sections = []
        for number, entry in enumerate(entries, start=1):
            def text(tag: str, node=entry) -> str:
                found = node.find(f"atom:{tag}", _ATOM)
                return " ".join((found.text or "").split()) if found is not None else ""

            authors = [text("name", a) for a in entry.findall("atom:author", _ATOM)]
            author_text = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            sections.append(
                f"[A{number}] {text('title')} ({author_text}, {text('published')[:10]})\n"
                f"URL: {text('id')}\n{text('summary')[:1000]}"
            )
        return "\n\n".join(sections)

    return arxiv_search
