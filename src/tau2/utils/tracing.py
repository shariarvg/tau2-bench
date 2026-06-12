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

import functools
import inspect
import json
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional

from tau2.utils.utils import DATA_DIR

DEFAULT_TRACE_DB_PATH = DATA_DIR / "traces" / "spans.db"

# call_names whose response content should be passed to the monitor agents.
MONITORED_CALL_NAMES = {"agent_response"}

# Whether span logging is active, and where the SQLite database lives. Set
# via `configure_tracing` (the `tau2 run` CLI wires this to --trace-db /
# --trace-db-path). Disabled by default: no DB file is created unless a run
# opts in.
_tracing_enabled = False
_trace_db_path: Path = DEFAULT_TRACE_DB_PATH


def configure_tracing(enabled: bool, db_path: Optional[Path] = None) -> None:
    """Enable/disable span logging and optionally set the SQLite database path.

    Call this once before running any simulations.
    """
    global _tracing_enabled, _trace_db_path
    _tracing_enabled = enabled
    if db_path is not None:
        _trace_db_path = Path(db_path)

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
    annotation TEXT,
    timestamp TEXT NOT NULL
)
"""


def _get_connection() -> Optional[sqlite3.Connection]:
    """Get a connection local to this thread, creating the DB/table if needed.

    Returns None if tracing is disabled (see `configure_tracing`).
    """
    if not _tracing_enabled:
        return None
    conn = getattr(_local, "conn", None)
    if conn is None:
        _trace_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(_trace_db_path, timeout=30)
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


def _get_monitor_annotations(
    content: str, history: Optional[str] = None
) -> Optional[dict]:
    """Run each registered MonitorAgent against an agent response.

    Returns a dict keyed by monitor name (e.g. "uncertainty", "drift"), each
    mapping to {"score": int, "rationale": str}, or None if no monitor
    produced a result. Best-effort only: a failing monitor is skipped rather
    than breaking the main simulation.
    """
    from tau2.utils.monitors import MONITORS

    annotations = {}
    for monitor in MONITORS:
        result = monitor.annotate(content, history=history)
        if result is not None:
            annotations[monitor.key] = result
    return annotations or None


def log_llm_span(
    *,
    name: str,
    tools_called: Optional[list[str]] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
    content: Optional[str] = None,
    history: Optional[str] = None,
) -> Optional[str]:
    """Log one LLM-call span and return its span_id, or None if tracing is disabled.

    LLM-call spans are always top-level (parent_span_id=None). If the call
    requested tool calls, this span's id is stashed in
    `current_parent_span_id` so future tool-call spans can reference it as
    their parent.

    If `name` is in `MONITORED_CALL_NAMES` and `content` is provided, each
    registered MonitorAgent (see `tau2.utils.monitors`) scores the response
    for a known failure mode, and the combined JSON result is stored in the
    `annotation` column, instead of storing the full response content.
    """
    conn = _get_connection()
    if conn is None:
        return None

    span_id = str(uuid.uuid4())
    trace_id = current_trace_id.get()
    tools_called = tools_called or []

    annotation = None
    if name in MONITORED_CALL_NAMES and content:
        annotation = _get_monitor_annotations(content, history=history)

    with _db_lock:
        conn.execute(
            "INSERT INTO spans "
            "(span_id, trace_id, parent_span_id, span_type, name, tools_called, "
            "prompt_tokens, completion_tokens, duration_seconds, error, annotation, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                json.dumps(annotation) if annotation is not None else None,
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
) -> Optional[str]:
    """Log one tool-execution span and return its span_id, or None if tracing is disabled.

    Its parent is whichever LLM-call span most recently requested tool calls
    (tracked via `current_parent_span_id`).
    """
    conn = _get_connection()
    if conn is None:
        return None

    span_id = str(uuid.uuid4())
    trace_id = current_trace_id.get()
    parent_span_id = current_parent_span_id.get()

    with _db_lock:
        conn.execute(
            "INSERT INTO spans "
            "(span_id, trace_id, parent_span_id, span_type, name, tools_called, "
            "prompt_tokens, completion_tokens, duration_seconds, error, annotation, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                None,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    return span_id


def trace_llm_call(func):
    """Decorator that logs an LLM-call span for each call to `func`.

    Intended for `tau2.utils.llm_utils.generate`. Reads `model`, `messages`,
    and `call_name` from the call's bound arguments, and `tools_called`,
    token usage, content, and duration from the returned message. On
    exception, logs an error span with no message-derived fields.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        model = bound.arguments.get("model")
        messages = bound.arguments.get("messages") or []
        call_name = bound.arguments.get("call_name")

        start_time = time.perf_counter()
        try:
            message = func(*args, **kwargs)
        except Exception as e:
            log_llm_span(
                name=call_name or model,
                duration_seconds=time.perf_counter() - start_time,
                error=str(e),
            )
            raise

        usage = message.usage
        tool_calls = message.tool_calls
        history = "\n".join(
            f"{m.role}: {m.content}" for m in messages if getattr(m, "content", None)
        )
        log_llm_span(
            name=call_name or model,
            tools_called=[tc.name for tc in tool_calls] if tool_calls else None,
            prompt_tokens=usage.get("prompt_tokens") if usage else None,
            completion_tokens=usage.get("completion_tokens") if usage else None,
            duration_seconds=message.generation_time_seconds,
            content=message.content,
            history=history or None,
        )
        return message

    return wrapper
