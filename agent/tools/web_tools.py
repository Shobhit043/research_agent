import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.tools.http import BlockedURLError, cache, safe_get

_MAX_DOWNLOAD_BYTES = 2_000_000
_BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]


def html_to_text(raw: bytes) -> tuple[str, str]:
    """(title, readable text). Parses bytes so BeautifulSoup honours the page's own charset;
    requests assumes ISO-8859-1 when the header omits one."""
    soup = BeautifulSoup(raw, features="lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return title, soup.get_text(separator="\n", strip=True)


def fetch_page(url: str, settings: Settings, max_chars: int) -> tuple[str, str, str]:
    """(final url, title, text) of an HTML or plain-text page."""
    cached = cache.get(("page", url, max_chars))
    if cached:
        return cached
    fetched = safe_get(url, settings.request_timeout, _MAX_DOWNLOAD_BYTES)
    if "html" in fetched.content_type:
        title, text = html_to_text(fetched.body)
    elif fetched.content_type.startswith("text/"):
        title, text = "", fetched.body.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"unsupported content type '{fetched.content_type or 'unknown'}'")
    result = (fetched.url, title, text[:max_chars])
    cache.set(("page", url, max_chars), result)
    return result


def build_web_search_tool(settings: Settings) -> BaseTool:
    @tool
    def web_search(query: str, max_results: int = 2) -> str:
        """
        Search the internet for information that is current, external, or not
        reliably available from the model's internal knowledge.

        Use this whenever the user asks to search, look up, verify, or research
        something online, or when the answer depends on recent or changing facts.
        Never simulate or assume search results; always call this tool instead.

        Args:
            query: A clear, specific search query describing what to find.
            max_results: Number of pages to read (capped server-side).
        """
        count = max(1, min(max_results, settings.max_web_results))
        results = cache.get(("search", query, count))
        if results is None:
            with DDGS() as ddgs:
                results = ddgs.text(query=query, max_results=count) or []
            cache.set(("search", query, count), results)
        if not results:
            return "No web results found."

        sections = []
        for number, result in enumerate(results, start=1):
            url = result.get("href", "")
            try:
                url, _, body = fetch_page(url, settings, settings.web_page_chars)
            except (requests.RequestException, ValueError) as exc:
                # The search snippet is still useful evidence when the full page is unreachable.
                body = f"{result.get('body', '')}\n(full page unavailable: {exc})"
            sections.append(f"[{number}] {result.get('title', '')}\nURL: {url}\n{body}")
        return "\n\n".join(sections)

    return web_search


def build_fetch_url_tool(settings: Settings) -> BaseTool:
    @tool
    def fetch_url(url: str) -> str:
        """
        Read the text of one specific web page. Use this when the user gives a URL,
        or to read a page from search results in more depth than the search snippet.

        Args:
            url: The full http(s) URL of the page to read.
        """
        try:
            final_url, title, text = fetch_page(url, settings, settings.fetch_page_chars)
        except BlockedURLError as exc:
            return f"Refused to fetch {url}: {exc}."
        if not text.strip():
            return f"{url} returned no readable text."
        return f"[U1] {title or final_url}\nURL: {final_url}\n{text}"

    return fetch_url
