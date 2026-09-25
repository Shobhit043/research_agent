import json
import socket

import pytest
from langchain_core.documents import Document

from agent.config import Settings
from agent.documents import DocumentStore, chunk_documents
from agent.tools import ALL_TOOLS, build_tools, tool_guidance
from agent.tools import http as tool_http
from agent.tools.document_tools import build_read_document_tool, merge_chunks
from agent.tools.http import BlockedURLError, TTLCache, check_url, safe_get
from agent.tools.knowledge_tools import build_arxiv_tool, build_wikipedia_tool
from agent.tools.place_tools import build_get_time_tool, build_place_info_tool
from agent.tools.web_tools import build_fetch_url_tool

# ----- fakes -----

PUBLIC = {"example.com": "93.184.216.34", "en.wikipedia.org": "185.15.59.224", "export.arxiv.org": "151.101.3.42",
          "geocoding-api.open-meteo.com": "46.4.105.116"}
PRIVATE = {"localhost": "127.0.0.1", "intranet": "10.0.0.5", "metadata": "169.254.169.254", "v6local": "::1"}


def fake_resolve(host, port, type=None):
    address = {**PUBLIC, **PRIVATE}.get(host)
    if address is None:
        raise socket.gaierror("unknown host")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (address, port))]


class FakeResponse:
    def __init__(self, status=200, body=b"", content_type="text/html", location=None):
        self.status_code = status
        self.body = body
        self.headers = {"Content-Type": content_type, **({"location": location} if location else {})}
        self.is_redirect = location is not None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise tool_http.requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        return self.routes[url]


@pytest.fixture
def web(monkeypatch):
    """Route HTTP to canned responses and DNS to the fake resolver; clear the tool cache."""
    session = FakeSession({})
    monkeypatch.setattr(tool_http, "_session", session)
    monkeypatch.setattr(tool_http.socket, "getaddrinfo", fake_resolve)
    monkeypatch.setattr(tool_http, "cache", TTLCache())
    for module in ("agent.tools.web_tools", "agent.tools.knowledge_tools"):
        monkeypatch.setattr(f"{module}.cache", tool_http.cache)
    return session


# ----- SSRF protection -----

@pytest.mark.parametrize("url", [
    "http://localhost:8000/admin",
    "http://intranet/secrets",
    "http://metadata/latest/meta-data/",
    "http://v6local/",
    "file:///etc/passwd",
    "ftp://example.com/file",
    "http://no-such-host.invalid/",
])
def test_blocks_non_public_and_non_http_urls(url):
    with pytest.raises(BlockedURLError):
        check_url(url, resolve=fake_resolve)


def test_allows_public_https():
    check_url("https://example.com/page", resolve=fake_resolve)


def test_redirect_to_internal_address_is_blocked(web):
    web.routes["https://example.com/go"] = FakeResponse(302, location="http://metadata/latest/")

    with pytest.raises(BlockedURLError, match="non-public"):
        safe_get("https://example.com/go", timeout=5, max_bytes=1000)
    assert web.requested == ["https://example.com/go"], "the internal URL must never be requested"


def test_redirect_to_public_address_is_followed_and_body_capped(web):
    web.routes["https://example.com/a"] = FakeResponse(301, location="/b")
    web.routes["https://example.com/b"] = FakeResponse(body=b"x" * 5000)

    fetched = safe_get("https://example.com/a", timeout=5, max_bytes=100)

    assert fetched.url == "https://example.com/b"
    assert len(fetched.body) == 100


def test_fetch_url_tool_extracts_title_and_text(web):
    html = (b"<html><head><title>Hello</title><script>evil()</script></head>"
            b"<body><nav>menu</nav><p>Body text</p></body></html>")
    web.routes["https://example.com/"] = FakeResponse(body=html)

    output = build_fetch_url_tool(Settings()).invoke({"url": "https://example.com/"})

    assert output.startswith("[U1] Hello\nURL: https://example.com/")
    assert "Body text" in output and "evil" not in output and "menu" not in output


def test_fetch_url_tool_refuses_internal_urls(web):
    output = build_fetch_url_tool(Settings()).invoke({"url": "http://localhost:8000/api/metrics"})

    assert output.startswith("Refused to fetch")


def test_ttl_cache_expires_and_evicts():
    cache = TTLCache(ttl=60, max_items=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert cache.get("a") is None and cache.get("c") == 3
    cache.ttl = -1
    assert cache.get("c") is None


# ----- knowledge sources -----

def test_wikipedia_tool_formats_ranked_articles(web):
    payload = {"query": {"pages": [
        {"index": 2, "title": "Second", "fullurl": "https://en.wikipedia.org/wiki/Second", "extract": "Two."},
        {"index": 1, "title": "Alan Turing", "fullurl": "https://en.wikipedia.org/wiki/Alan_Turing",
         "extract": "Alan Turing was a mathematician."},
    ]}}
    tool = build_wikipedia_tool(Settings())
    web.routes = type("AnyUrl", (dict,), {"__getitem__": lambda self, url: FakeResponse(
        body=json.dumps(payload).encode(), content_type="application/json")})()

    output = tool.invoke({"query": "Alan Turing"})

    assert output.index("[W1] Alan Turing") < output.index("[W2] Second")
    assert "URL: https://en.wikipedia.org/wiki/Alan_Turing" in output


def test_arxiv_tool_parses_atom_feed(web):
    feed = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/1706.03762v7</id><published>2017-06-12T17:57:34Z</published>
      <title>Attention Is All
        You Need</title><summary>The dominant sequence transduction models...</summary>
      <author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author>
      <author><name>Niki Parmar</name></author><author><name>Jakob Uszkoreit</name></author>
    </entry></feed>"""
    web.routes = type("AnyUrl", (dict,), {"__getitem__": lambda self, url: FakeResponse(
        body=feed, content_type="application/atom+xml")})()

    output = build_arxiv_tool(Settings()).invoke({"query": "transformers"})

    assert output.startswith(
        "[A1] Attention Is All You Need (Ashish Vaswani, Noam Shazeer, Niki Parmar et al., 2017-06-12)"
    )
    assert "URL: http://arxiv.org/abs/1706.03762v7" in output


# ----- time and place -----

GEO = {"results": [
    {"name": "Paris", "latitude": 48.85341, "longitude": 2.3488, "elevation": 42.0, "country": "France",
     "admin1": "Île-de-France", "timezone": "Europe/Paris", "population": 2138551},
    {"name": "Paris", "latitude": 33.66094, "longitude": -95.55551, "elevation": 177.0, "country": "United States",
     "admin1": "Texas", "timezone": "America/Chicago", "population": 24171},
]}


def serve_json(web, payload):
    web.routes = type("AnyUrl", (dict,), {"__getitem__": lambda self, url: FakeResponse(
        body=json.dumps(payload).encode(), content_type="application/json")})()


def test_get_time_accepts_an_iana_timezone_without_network(web):
    output = build_get_time_tool(Settings()).invoke({"location": "Asia/Kolkata"})

    assert output.startswith("Current time in Asia/Kolkata:")
    assert "UTC+05:30" in output
    assert web.requested == [], "timezone names must not trigger a geocoding call"


def test_get_time_resolves_a_place_to_its_timezone(web):
    serve_json(web, GEO)

    output = build_get_time_tool(Settings()).invoke({"location": "Paris"})

    assert output.startswith("Current time in Paris, Île-de-France, France:")
    assert "Europe/Paris" in output


def test_get_time_reports_unknown_places(web):
    serve_json(web, {})

    assert "Could not find" in build_get_time_tool(Settings()).invoke({"location": "Atlantis"})


def test_place_info_lists_matches_with_map_links(web):
    serve_json(web, GEO)

    output = build_place_info_tool(Settings()).invoke({"query": "Paris"})

    assert output.startswith("[P1] Paris, Île-de-France, France\nURL: https://www.openstreetmap.org/")
    assert "Population: 2,138,551" in output and "Timezone: Europe/Paris" in output
    assert "[P2] Paris, Texas, United States" in output


def test_place_info_country_hint_reranks_matches(web):
    serve_json(web, GEO)

    output = build_place_info_tool(Settings()).invoke({"query": "Paris, Texas"})

    assert output.startswith("[P1] Paris, Texas, United States")
    assert "America/Chicago" in output


# ----- documents -----

def test_merge_chunks_removes_overlap():
    text = "".join(f"sentence {i}. " for i in range(200))
    chunks = chunk_documents([Document(page_content=text, metadata={"source": "a.pdf", "page": 1})], 300, 60)

    assert len(chunks) > 3
    assert merge_chunks(chunks) == text.strip()


def test_read_document_by_page_and_whole():
    store = DocumentStore()
    store.add(chunk_documents([
        Document(page_content="Page one text", metadata={"source": "r.pdf", "page": 1}),
        Document(page_content="Page two text", metadata={"source": "r.pdf", "page": 2}),
    ], 1000, 100))
    tool = build_read_document_tool(store)

    assert tool.invoke({"name": "r.pdf", "page": 2}) == "[r.pdf p.2]\nPage two text"
    whole = tool.invoke({"name": "r.pdf"})
    assert "[r.pdf p.1]\nPage one text" in whole and "[r.pdf p.2]" in whole
    assert "no page 9" in tool.invoke({"name": "r.pdf", "page": 9})
    assert "Available: r.pdf" in tool.invoke({"name": "missing.pdf"})


# ----- registry -----

def test_registry_builds_all_tools_by_default():
    assert list(build_tools(DocumentStore(), Settings())) == list(ALL_TOOLS)


def test_registry_respects_enabled_tools_and_rejects_typos():
    tools = build_tools(DocumentStore(), Settings(enabled_tools=("get_time", "web_search")))
    assert set(tools) == {"get_time", "web_search"}

    with pytest.raises(ValueError, match="get_tme"):
        build_tools(DocumentStore(), Settings(enabled_tools=("get_tme",)))


def test_guidance_mentions_only_bound_tools():
    tools = build_tools(DocumentStore(), Settings(enabled_tools=("place_info",)))

    guidance = tool_guidance(list(tools.values()))

    assert "place_info" in guidance and "web_search" not in guidance
