"""Tests for api.app.grounding: sanitising, caching, budget and graceful degradation."""

import pytest

from api.app import grounding

NOW = 1_000_000.0


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("NAIP_GROUNDING_TTL_SECONDS", raising=False)
    monkeypatch.delenv("NAIP_GROUNDING_MAX_PER_DAY", raising=False)
    grounding.reset_state()
    yield
    grounding.reset_state()


def good_backend(calls):
    def backend(state, district, crop, alert_type):
        calls.append((state, district, crop, alert_type))
        return {
            "available": True,
            "sections": [
                {
                    "category": "Government Relief Schemes",
                    "query": "Haryana drought relief",
                    "results": [
                        {
                            "title": "Relief <b>scheme</b>",
                            "content": "Details",
                            "url": "https://x.in/a",
                        }
                    ],
                }
            ],
        }

    return backend


def use_backend(monkeypatch, backend):
    monkeypatch.setattr(grounding, "get_backend", lambda: backend)


def test_clean_text_strips_markup_including_escaped_tags():
    dirty = "Hi <script>alert(1)</script> &lt;img src=x onerror=y&gt;  there"
    out = grounding.clean_text(dirty, 200)
    assert "<" not in out and ">" not in out
    assert out.startswith("Hi") and out.endswith("there")


def test_clean_text_caps_length_and_handles_none():
    assert len(grounding.clean_text("a" * 1000, 50)) == 50
    assert grounding.clean_text(None, 50) == ""


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "data:text/html,x", "ftp://x.in", "", None, "https://a b.in"]
)
def test_clean_url_rejects_non_http_links(url):
    assert grounding.clean_url(url) is None


def test_clean_url_accepts_https():
    assert grounding.clean_url("https://imd.gov.in/forecast") == "https://imd.gov.in/forecast"


def test_shape_sections_drops_uncitable_results_caps_count_and_hides_error_text():
    raw = [
        {
            "category": "Weather Outlook (IMD)",
            "query": "q",
            "results": [
                {"title": "no link", "content": "x", "url": "javascript:1"},
                *[{"title": f"t{i}", "content": "c", "url": f"https://a.in/{i}"} for i in range(5)],
            ],
            "error": "secret provider message",
        }
    ]
    section = grounding.shape_sections(raw)[0]
    assert len(section["results"]) == 3  # the uncitable one is skipped, then the cap applies
    assert all(r["url"].startswith("https://") for r in section["results"])
    assert section["failed"] is True
    assert "secret" not in str(section)


def test_get_context_returns_sanitised_labelled_payload(monkeypatch):
    calls = []
    use_backend(monkeypatch, good_backend(calls))
    out = grounding.get_context("Haryana", "Sirsa", "Wheat", "drought", now=NOW)
    assert out["available"] is True and out["cached"] is False
    assert "not model output" in out["notice"]
    assert out["sections"][0]["results"][0]["title"] == "Relief scheme"
    assert calls == [("Haryana", "Sirsa", "Wheat", "drought")]


def test_second_identical_request_is_served_from_cache(monkeypatch):
    calls = []
    use_backend(monkeypatch, good_backend(calls))
    grounding.get_context("Haryana", "Sirsa", "Wheat", "drought", now=NOW)
    again = grounding.get_context("Haryana", "Sirsa", "Wheat", "drought", now=NOW + 60)
    assert again["cached"] is True
    assert len(calls) == 1


def test_cache_expires_after_the_ttl(monkeypatch):
    calls = []
    use_backend(monkeypatch, good_backend(calls))
    grounding.get_context("Haryana", "Sirsa", "Wheat", "drought", now=NOW)
    grounding.get_context("Haryana", "Sirsa", "Wheat", "drought", now=NOW + 7 * 3600)
    assert len(calls) == 2


def test_daily_budget_caps_live_calls_and_recovers_after_a_day(monkeypatch):
    monkeypatch.setenv("NAIP_GROUNDING_MAX_PER_DAY", "2")
    calls = []
    use_backend(monkeypatch, good_backend(calls))
    for i, name in enumerate(["A", "B"]):
        assert grounding.get_context("Haryana", name, "Wheat", "general", now=NOW + i)["available"]
    blocked = grounding.get_context("Haryana", "C", "Wheat", "general", now=NOW + 10)
    assert blocked["available"] is False and blocked["reason"] == "rate_limited"
    assert len(calls) == 2
    later = grounding.get_context("Haryana", "C", "Wheat", "general", now=NOW + 25 * 3600)
    assert later["available"] is True


def test_missing_key_degrades_without_spending_budget(monkeypatch):
    monkeypatch.setenv("NAIP_GROUNDING_MAX_PER_DAY", "1")
    use_backend(monkeypatch, lambda *a: {"available": False, "reason": "No Tavily API key"})
    first = grounding.get_context("Haryana", "Sirsa", "Wheat", "general", now=NOW)
    assert first["available"] is False and first["reason"] == "not_configured"
    assert "Tavily" not in str(first)
    use_backend(monkeypatch, good_backend([]))
    assert grounding.get_context("Haryana", "Sirsa", "Wheat", "general", now=NOW + 1)["available"]


def test_backend_exception_and_missing_backend_never_raise(monkeypatch):
    def boom(*args):
        raise RuntimeError("network down")

    use_backend(monkeypatch, boom)
    assert grounding.get_context("H", "S", "Wheat", "general", now=NOW)["reason"] == "unavailable"
    use_backend(monkeypatch, None)
    assert grounding.get_context("H", "S", "Wheat", "general", now=NOW)["available"] is False


def test_answer_with_no_results_is_not_cached(monkeypatch):
    calls = []

    def empty(state, district, crop, alert_type):
        calls.append(1)
        return {"available": True, "sections": [{"category": "x", "results": [], "error": "e"}]}

    use_backend(monkeypatch, empty)
    grounding.get_context("H", "S", "Wheat", "general", now=NOW)
    grounding.get_context("H", "S", "Wheat", "general", now=NOW + 1)
    assert len(calls) == 2
