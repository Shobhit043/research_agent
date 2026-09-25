import contextvars
import json
import logging
import threading
import time
from collections import Counter

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregators (Loki, CloudWatch, Datadog...)."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(verbose: bool, fmt: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter() if fmt == "json"
        else logging.Formatter("  [%(levelname)s] %(request_id)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    # Third-party INFO logs (HTTP requests) drown out the agent trace.
    for noisy in ("httpx", "groq", "primp", "ddgs", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Metrics:
    """In-process counters exposed in Prometheus text format at /api/metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Counter = Counter()
        self._latency_sum = 0.0
        self._latency_count = 0
        self.started = time.time()

    def inc(self, name: str, amount: float = 1, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += amount

    def observe_turn(self, seconds: float) -> None:
        with self._lock:
            self._latency_sum += seconds
            self._latency_count += 1

    def render(self) -> str:
        lines = [
            "# TYPE assistant_uptime_seconds gauge",
            f"assistant_uptime_seconds {time.time() - self.started:.0f}",
        ]
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                label_text = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_text}}} {value:g}" if label_text else f"{name} {value:g}")
            lines += [
                "# TYPE assistant_turn_latency_seconds summary",
                f"assistant_turn_latency_seconds_sum {self._latency_sum:.3f}",
                f"assistant_turn_latency_seconds_count {self._latency_count}",
            ]
        return "\n".join(lines) + "\n"
