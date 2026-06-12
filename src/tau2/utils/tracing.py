"""
Lightweight span tracing for agent runs.

Every LLM call made during a simulation is logged as a "span" row to a SQLite
database. Spans are linked into a trace via `trace_id` (= simulation id) and
`parent_span_id`.

Currently only LLM-call spans are emitted. Tool-call spans can be added later
via `log_tool_span`: when an LLM-call span requests tool calls, its span_id is
stashed in `current_parent_span_id` so the corresponding tool-call spans can
pick it up as their parent.
"""

import json
import sqlite3
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional

from tau2.utils.utils import DATA_DIR

TRACE_DB_PATH = DATA_DIR / "traces" / "spans.db"

# The simulation (trace) currently running on this thread.
current_trace_id: ContextVar[Optional[str]] = ContextVar(
    "current_trace_id", default=None
)

# The span id that the *next* tool-call span(s) should report as their parent.
# Set by `log_llm_span` whenever the LLM response includes tool calls.
current_parent_span_id: ContextVar[Optional[str]] = ContextVar(
    "current_parent_span_id", default=None
)

_db_lock = threading.Lock()
_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT,
    parent_span_id TEXT,
    span_type TEXT NOT NULL,
    name TEXT,
    tools_called TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    duration_seconds REAL,
    error TEXT,
    timestamp TEXT NOT NULL
)
"""


def _get_connection() -> sqlite3.Connection:
    """Get a connection local to this thread, creating the DB/table if needed."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(TRACE_DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        with _db_lock:
            conn.execute(_SCHEMA)
            conn.commit()
        _local.conn = conn
    return conn


def start_trace(trace_id: str) -> None:
    """Set the trace id for the current thread and clear any stale parent span."""
    current_trace_id.set(trace_id)
    current_parent_span_id.set(None)


def end_trace() -> None:
    """Clear trace context for the current thread."""
    current_trace_id.set(None)
    current_parent_span_id.set(None)


def log_llm_span(
    *,
    name: str,
    tools_called: Optional[list[str]] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> str:
    """Log one LLM-call span and return its span_id.

    LLM-call spans are always top-level (parent_span_id=None). If the call
    requested tool calls, this span's id is stashed in
    `current_parent_span_id` so future tool-call spans can reference it as
    their parent.
    """
    span_id = str(uuid.uuid4())
    trace_id = current_trace_id.get()
    tools_called = tools_called or []

    conn = _get_connection()
    with _db_lock:
        conn.execute(
            "INSERT INTO spans "
            "(span_id, trace_id, parent_span_id, span_type, name, tools_called, "
            "prompt_tokens, completion_tokens, duration_seconds, error, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span_id,
                trace_id,
                None,
                "llm_call",
                name,
                json.dumps(tools_called),
                prompt_tokens,
                completion_tokens,
                duration_seconds,
                error,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    current_parent_span_id.set(span_id if tools_called else None)
    return span_id


def log_tool_span(
    *,
    name: str,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> str:
    """Log one tool-execution span and return its span_id.

    Its parent is whichever LLM-call span most recently requested tool calls
    (tracked via `current_parent_span_id`).
    """
    span_id = str(uuid.uuid4())
    trace_id = current_trace_id.get()
    parent_span_id = current_parent_span_id.get()

    conn = _get_connection()
    with _db_lock:
        conn.execute(
            "INSERT INTO spans "
            "(span_id, trace_id, parent_span_id, span_type, name, tools_called, "
            "prompt_tokens, completion_tokens, duration_seconds, error, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span_id,
                trace_id,
                parent_span_id,
                "tool_call",
                name,
                None,
                None,
                None,
                duration_seconds,
                error,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    return span_id
