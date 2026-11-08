"""Structured JSON logging with a trace id carried in a contextvar (no extra dependencies).

    log = get_logger("consumer")
    with trace(trace_id):
        log.info("processed", extra={"fields": {"facts": 3}})
-> {"ts": "...", "level": "INFO", "logger": "memorybench.consumer", "msg": "processed", "trace_id": "...", "facts": 3}
"""
import contextlib
import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id.get()


@contextlib.contextmanager
def trace(trace_id: str | None):
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if (tid := _trace_id.get()) is not None:
            out["trace_id"] = tid
        out.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def get_logger(name: str, stream=None) -> logging.Logger:
    log = logging.getLogger(f"memorybench.{name}")
    if not log.handlers:
        h = logging.StreamHandler(stream or sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log
