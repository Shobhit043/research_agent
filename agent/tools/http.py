"""Hardened HTTP for tools: SSRF protection, redirect validation, size caps, retries, caching.

Tools fetch URLs chosen by the model, which may have been steered by a web page or an
uploaded file. Without a guard, a prompt injection could make the server fetch
http://169.254.169.254/ (cloud credentials) or services on the internal network.
"""

import ipaddress
import os
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Wikimedia and arXiv ask API clients to identify themselves with contact details and
# throttle generic agents harder, so deployments should set HTTP_USER_AGENT.
USER_AGENT = os.getenv("HTTP_USER_AGENT", "ResearchAssistant/0.4 (self-hosted research tool)")
MAX_REDIRECTS = 5


class BlockedURLError(ValueError):
    """The URL points somewhere tools must not reach."""


@dataclass(frozen=True)
class Fetched:
    url: str
    content_type: str
    body: bytes


def _build_session() -> requests.Session:
    session = requests.Session()
    # Short, bounded backoff only: honouring Retry-After could stall a user's turn past the tool timeout.
    retry = Retry(
        total=2, backoff_factor=0.5, status_forcelist=(429, 502, 503, 504), allowed_methods=("GET",),
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = USER_AGENT
    return session


_session = _build_session()


def check_url(url: str, resolve=None) -> None:
    """Allow only http(s) URLs whose host resolves exclusively to public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(f"only http and https URLs are allowed, not '{parsed.scheme or 'none'}'")
    host = parsed.hostname
    if not host:
        raise BlockedURLError("URL has no host")
    resolve = resolve or socket.getaddrinfo
    try:
        infos = resolve(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise BlockedURLError(f"could not resolve {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global or address.is_multicast:
            raise BlockedURLError(f"{host} resolves to a non-public address ({address})")


def safe_get(url: str, timeout: float, max_bytes: int, resolve=None) -> Fetched:
    """GET with every redirect hop re-validated, so a public URL can't bounce to an internal one."""
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url, resolve)
        response = _session.get(url, timeout=timeout, stream=True, allow_redirects=False)
        try:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            body = bytearray()
            for block in response.iter_content(chunk_size=65536):
                body += block
                if len(body) >= max_bytes:
                    break
            return Fetched(url=url, content_type=response.headers.get("Content-Type", ""), body=bytes(body[:max_bytes]))
        finally:
            response.close()
    raise BlockedURLError(f"more than {MAX_REDIRECTS} redirects")


class TTLCache:
    """Small thread-safe LRU cache with expiry; saves quota on repeated searches and fetches."""

    def __init__(self, ttl: float = 600, max_items: int = 256):
        self.ttl = ttl
        self.max_items = max_items
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._items.get(key)
            if entry is None or time.monotonic() - entry[0] > self.ttl:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return entry[1]

    def set(self, key, value) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


cache = TTLCache()
