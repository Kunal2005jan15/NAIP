"""Live web grounding for the API: a safe wrapper around src/rag_grounding_31.py.

Rules from docs/security.md section 8, enforced here:
- Retrieved text is untrusted. It is returned as plain text with source links, never HTML.
- Length is capped and markup is stripped.
- The response is labelled as external sources, not model output.
- Nothing here touches features, predictions or the database.
- A missing key, a failed call or an exhausted budget returns available=false, never an error.

The search queries are built from the district and crop looked up in the database plus a fixed
alert type, so no client text ever reaches the search provider.

Each live call costs three Tavily searches (schemes, MSP, IMD), and the free plan allows about
1000 a month. Results are therefore cached, and live calls are capped per day.

Settings (environment variables):
  NAIP_GROUNDING_TTL_SECONDS   cache lifetime, default 21600 (6 hours)
  NAIP_GROUNDING_MAX_PER_DAY   live calls per rolling 24 hours per process, default 10
"""

import html
import importlib.util
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ALERT_TYPES = ("drought", "flood", "general")
RAG_MODULE = Path(__file__).resolve().parents[2] / "src" / "rag_grounding_31.py"
NOTICE = (
    "External web sources retrieved live, not model output. They are context only: they "
    "do not feed the predictions, and their accuracy is not checked by this system."
)
DAY_SECONDS = 24 * 3600
MAX_RESULTS_PER_SECTION = 3
CATEGORY_LIMIT = 80
QUERY_LIMIT = 200
TITLE_LIMIT = 160
SNIPPET_LIMIT = 280
URL_LIMIT = 500

_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")
_URL = re.compile(r"^https?://[^\s<>\"']+$", re.IGNORECASE)

_lock = threading.Lock()
_cache: dict = {}
_live_calls: deque = deque()
_backend: dict = {}


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def ttl_seconds() -> int:
    return _int_env("NAIP_GROUNDING_TTL_SECONDS", 6 * 3600)


def daily_budget() -> int:
    return _int_env("NAIP_GROUNDING_MAX_PER_DAY", 10)


def reset_state() -> None:
    """Clear the cache, the call counter and the loaded backend (used by tests)."""
    with _lock:
        _cache.clear()
        _live_calls.clear()
        _backend.clear()


# ---------------------------------------------------------------- sanitising
def clean_text(value, limit: int) -> str:
    """Plain text only: decode entities first so escaped tags are caught, then drop all markup."""
    text = html.unescape(str(value or ""))
    text = _TAG.sub(" ", text).replace("<", " ").replace(">", " ")
    return _SPACE.sub(" ", text).strip()[:limit]


def clean_url(value) -> str | None:
    url = str(value or "").strip()
    return url if len(url) <= URL_LIMIT and _URL.match(url) else None


def shape_sections(raw_sections) -> list:
    sections = []
    for section in raw_sections or []:
        results = []
        for item in section.get("results") or []:
            if len(results) >= MAX_RESULTS_PER_SECTION:
                break
            url = clean_url(item.get("url"))
            if url is None:  # a result that cannot be cited is dropped
                continue
            results.append(
                {
                    "title": clean_text(item.get("title"), TITLE_LIMIT),
                    "snippet": clean_text(item.get("content"), SNIPPET_LIMIT),
                    "url": url,
                }
            )
        sections.append(
            {
                "category": clean_text(section.get("category"), CATEGORY_LIMIT),
                "query": clean_text(section.get("query"), QUERY_LIMIT),
                "results": results,
                # the provider's error text is not passed on, only the fact that it failed
                "failed": bool(section.get("error")),
            }
        )
    return sections


# ---------------------------------------------------------------- backend
def _load_backend():
    try:
        spec = importlib.util.spec_from_file_location("naip_rag_grounding", RAG_MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_live_grounding
    except Exception:  # missing file, broken import: grounding is optional
        return None


def get_backend():
    if "fn" not in _backend:
        _backend["fn"] = _load_backend()
    return _backend["fn"]


# ---------------------------------------------------------------- public
def _stamp(now: float) -> str:
    return datetime.fromtimestamp(now, tz=timezone.utc).isoformat()


def _unavailable(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "cached": False,
        "retrieved_at": None,
        "notice": NOTICE,
        "sections": [],
    }


def _calls_last_day(now: float) -> int:
    while _live_calls and now - _live_calls[0] >= DAY_SECONDS:
        _live_calls.popleft()
    return len(_live_calls)


def _refund(now: float) -> None:
    try:
        _live_calls.remove(now)
    except ValueError:
        pass


def get_context(state: str, district: str, crop: str, alert_type: str, now: float | None = None):
    """Cited web context for one district. Never raises."""
    now = time.time() if now is None else now
    key = (state, district, crop, alert_type)
    with _lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < ttl_seconds():
            return {**hit[1], "cached": True}
        if _calls_last_day(now) >= daily_budget():
            return _unavailable("rate_limited")
        # reserve the slot before the slow call so concurrent requests cannot overshoot
        _live_calls.append(now)

    backend = get_backend()
    raw = None
    if backend is not None:
        try:
            raw = backend(state, district, crop, alert_type)
        except Exception:  # the provider or the network failed; degrade, do not error
            raw = None

    if not raw or not raw.get("available"):
        with _lock:
            _refund(now)  # nothing was spent, so the attempt does not count against the budget
        return _unavailable("not_configured" if backend is not None and raw else "unavailable")

    payload = {
        "available": True,
        "reason": None,
        "cached": False,
        "retrieved_at": _stamp(now),
        "notice": NOTICE,
        "sections": shape_sections(raw.get("sections")),
    }
    if any(section["results"] for section in payload["sections"]):
        with _lock:  # an all-empty or all-failed answer is not worth keeping for hours
            _cache[key] = (now, payload)
    return payload
