"""Tests for /watch, /kharif-risk, /districts/.../kharif and /districts/.../context."""

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.app import grounding, repository
from api.app.deps import get_conn
from api.app.main import create_app
from api.app.routers import v1

RUN = {
    "run_id": uuid.UUID("4ea1963f-c6ac-4b6a-b8a9-761d64e4e53a"),
    "status": "ok",
    "gate_passed": True,
    "model_version": "xgb_tuned_846a40d1",
    "weather_data_as_of": date.today() - timedelta(days=1),
    "git_sha": "abc123",
    "started_at": datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc),
    "finished_at": datetime(2026, 10, 8, 9, 1, tzinfo=timezone.utc),
}
EXTRAS = {"prediction_basis": "basis", "baseline_yield_year": 2019, "empirical_coverage": {}}
DISTRICT = {
    "district_id": 7,
    "slug": "haryana/sirsa",
    "state": "Haryana",
    "district": "Sirsa",
    "lat": 29.5,
    "lon": 75.0,
}
WATCH_ROW = {
    "state": "Haryana",
    "district": "Sirsa",
    "slug": "haryana/sirsa",
    "crop": "Wheat",
    "alert_level": "urgent",
    "change_note": "Predicted yield shifted -730 kg/ha since last check (worsening).",
    "yield_change": -730.04,
    "rai_change": -1.234,
    "severity_change": 0.5,
    "pred_yield": 4100.55,
}
KHARIF_ROW = {
    "state": "Haryana",
    "district": "Ambala",
    "slug": "haryana/ambala",
    "partial_rainfall_mm": 80.0,
    "days_elapsed": 23,
    "hist_mean": 160.0,
    "hist_std": 40.0,
    "n_years": 22,
    "zscore": -2.0,
    "risk_level": "high",
    "explanation": "Rainfall so far is well below the same window in past years.",
}
PRED_ROW = {"drought_flag": 1.0, "flood_flag": 0.0}


@pytest.fixture
def client(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_conn] = lambda: object()
    monkeypatch.setattr(repository, "latest_run", lambda conn: RUN)
    monkeypatch.setattr(repository, "run_extras", lambda conn, run_id, mv: EXTRAS)
    monkeypatch.setattr(repository, "list_watch", lambda conn, *a: ([WATCH_ROW], 1))
    monkeypatch.setattr(repository, "watch_counts", lambda conn, run_id: {"urgent": 1})
    monkeypatch.setattr(repository, "list_kharif", lambda conn, *a: ([KHARIF_ROW], 1))
    monkeypatch.setattr(repository, "kharif_counts", lambda conn, run_id: {"high": 1})
    monkeypatch.setattr(repository, "get_kharif", lambda conn, run_id, slug: KHARIF_ROW)
    monkeypatch.setattr(repository, "get_district", lambda conn, slug: dict(DISTRICT))
    monkeypatch.setattr(repository, "get_prediction", lambda conn, run_id, slug, crop: PRED_ROW)
    v1._META_CACHE.clear()
    grounding.reset_state()
    return TestClient(app)


# ---------------------------------------------------------------- /watch
def test_watch_envelope_counts_and_shaped_rows(client):
    response = client.get("/api/v1/watch")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data", "page", "meta"}
    assert body["meta"]["counts_by_level"] == {"urgent": 1}
    row = body["data"][0]
    assert row["alert_level"] == "urgent"
    assert row["yield_change_kg_ha"] == -730.0
    assert row["rai_change"] == -1.23
    assert response.headers["etag"].startswith('W/"')


def test_watch_filters_are_normalised_before_reaching_the_repository(client, monkeypatch):
    seen = {}

    def spy(conn, run_id, level, crop, state_slug, limit, offset):
        seen.update(level=level, crop=crop, state=state_slug, limit=limit, offset=offset)
        return [], 0

    monkeypatch.setattr(repository, "list_watch", spy)
    client.get("/api/v1/watch?alert_level=URGENT&crop=wheat&state=Uttar%20Pradesh&limit=5")
    assert seen == {
        "level": "urgent",
        "crop": "Wheat",
        "state": "uttar-pradesh",
        "limit": 5,
        "offset": 0,
    }


def test_watch_unknown_level_is_a_400_listing_the_allowed_values(client):
    response = client.get("/api/v1/watch?alert_level=panic")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INVALID_ALERT_LEVEL"
    assert "urgent" in error["details"]["allowed"]


def test_watch_without_a_promoted_run_is_a_409(client, monkeypatch):
    monkeypatch.setattr(repository, "latest_run", lambda conn: None)
    assert client.get("/api/v1/watch").status_code == 409


# ---------------------------------------------------------------- /kharif-risk
def test_kharif_list_has_rainfall_vs_normal_and_counts(client, monkeypatch):
    in_season = {**RUN, "weather_data_as_of": date(2026, 9, 1)}
    monkeypatch.setattr(repository, "latest_run", lambda conn: in_season)
    body = client.get("/api/v1/kharif-risk?risk_level=high").json()
    assert body["meta"]["counts_by_level"] == {"high": 1}
    row = body["data"][0]
    assert row["risk_level"] == "high"
    assert row["rainfall_vs_normal_pct"] == -50.0
    assert "notice" not in body["meta"]


def test_kharif_out_of_season_adds_a_notice(client, monkeypatch):
    winter = {**RUN, "weather_data_as_of": date(2026, 2, 1)}
    monkeypatch.setattr(repository, "latest_run", lambda conn: winter)
    body = client.get("/api/v1/kharif-risk").json()
    assert "June to October" in body["meta"]["notice"]


def test_kharif_unknown_level_is_a_400(client):
    response = client.get("/api/v1/kharif-risk?risk_level=extreme")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_RISK_LEVEL"


def test_district_kharif_detail(client):
    response = client.get("/api/v1/districts/haryana/sirsa/kharif")
    assert response.status_code == 200
    assert response.json()["data"]["zscore"] == -2.0


def test_district_kharif_unknown_district_and_missing_row(client, monkeypatch):
    monkeypatch.setattr(repository, "get_kharif", lambda conn, run_id, slug: None)
    missing = client.get("/api/v1/districts/haryana/sirsa/kharif")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NO_KHARIF_RISK"
    monkeypatch.setattr(repository, "get_district", lambda conn, slug: None)
    unknown = client.get("/api/v1/districts/haryana/atlantis/kharif")
    assert unknown.json()["error"]["code"] == "DISTRICT_NOT_FOUND"


# ---------------------------------------------------------------- /context (RAG)
def fake_context(record):
    def get_context(state, district, crop, alert_type, now=None):
        record.append((state, district, crop, alert_type))
        return {
            "available": True,
            "reason": None,
            "cached": False,
            "retrieved_at": "2026-10-27T00:00:00+00:00",
            "notice": grounding.NOTICE,
            "sections": [],
        }

    return get_context


def test_context_uses_database_names_and_derives_alert_type_from_flags(client, monkeypatch):
    calls = []
    monkeypatch.setattr(grounding, "get_context", fake_context(calls))
    response = client.get("/api/v1/districts/HARYANA/Sirsa/context")
    assert response.status_code == 200
    assert calls == [("Haryana", "Sirsa", "Wheat", "drought")]
    data = response.json()["data"]
    assert data["alert_type"] == "drought" and data["crop"] == "Wheat"
    assert data["district"]["slug"] == "haryana/sirsa"
    assert "not model output" in data["notice"]
    assert response.headers["cache-control"] == "private, max-age=900"


def test_context_explicit_alert_type_wins_and_unknown_is_a_400(client, monkeypatch):
    calls = []
    monkeypatch.setattr(grounding, "get_context", fake_context(calls))
    client.get("/api/v1/districts/haryana/sirsa/context?alert_type=flood&crop=rice")
    assert calls == [("Haryana", "Sirsa", "Rice", "flood")]
    bad = client.get("/api/v1/districts/haryana/sirsa/context?alert_type=<script>")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "INVALID_ALERT_TYPE"


def test_context_unknown_district_never_reaches_the_search_provider(client, monkeypatch):
    calls = []
    monkeypatch.setattr(grounding, "get_context", fake_context(calls))
    monkeypatch.setattr(repository, "get_district", lambda conn, slug: None)
    response = client.get("/api/v1/districts/haryana/atlantis/context")
    assert response.status_code == 404
    assert calls == []


def test_context_degrades_to_available_false_not_an_error(client, monkeypatch):
    monkeypatch.setattr(grounding, "get_backend", lambda: None)
    response = client.get("/api/v1/districts/haryana/sirsa/context")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["available"] is False and data["sections"] == []


def test_context_without_a_promoted_run_falls_back_to_general(client, monkeypatch):
    calls = []
    monkeypatch.setattr(grounding, "get_context", fake_context(calls))
    monkeypatch.setattr(repository, "latest_run", lambda conn: None)
    client.get("/api/v1/districts/haryana/sirsa/context")
    assert calls[0][3] == "general"


# ---------------------------------------------------------------- repository SQL
class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.calls.append((" ".join(sql.split()), params))
        self.rows = self.conn.responses.pop(0)

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def cursor(self, row_factory=None):
        return FakeCursor(self)


def test_list_watch_filters_and_ranks_urgent_first():
    conn = FakeConn([{"n": 3}], [WATCH_ROW])
    repository.list_watch(conn, "run-1", "urgent", "Wheat", "haryana", 10, 0)
    count_sql, count_params = conn.calls[0]
    list_sql, list_params = conn.calls[1]
    assert "w.alert_level = %(level)s" in count_sql and "d.slug LIKE %(like)s" in count_sql
    assert count_params == {
        "run_id": "run-1",
        "level": "urgent",
        "crop": "Wheat",
        "like": "haryana/%",
    }
    assert "WHEN 'urgent' THEN 0" in list_sql
    assert "abs(coalesce(w.yield_change, 0)) DESC" in list_sql
    assert list_params["limit"] == 10


def test_list_watch_without_filters_has_only_the_run_condition():
    conn = FakeConn([{"n": 0}], [])
    repository.list_watch(conn, "run-1", None, None, None, 50, 0)
    sql, params = conn.calls[0]
    assert "alert_level = %(level)s" not in sql and "LIKE" not in sql
    assert params == {"run_id": "run-1"}


def test_list_kharif_ranks_high_risk_and_lowest_zscore_first():
    conn = FakeConn([{"n": 1}], [KHARIF_ROW])
    repository.list_kharif(conn, "run-1", "high", None, 50, 0)
    list_sql, _ = conn.calls[1]
    assert "WHEN 'high' THEN 0" in list_sql
    assert "k.zscore ASC NULLS LAST" in list_sql


def test_counts_return_a_level_to_number_mapping():
    conn = FakeConn([{"alert_level": "urgent", "n": 2}, {"alert_level": "stable", "n": 100}])
    assert repository.watch_counts(conn, "run-1") == {"urgent": 2, "stable": 100}
    conn = FakeConn([{"risk_level": "low", "n": 4}])
    assert repository.kharif_counts(conn, "run-1") == {"low": 4}


def test_shape_kharif_handles_missing_history_without_dividing_by_zero():
    no_history = repository.shape_kharif({**KHARIF_ROW, "hist_mean": None, "zscore": None})
    assert no_history["rainfall_vs_normal_pct"] is None and no_history["zscore"] is None
    zero_mean = repository.shape_kharif({**KHARIF_ROW, "hist_mean": 0.0})
    assert zero_mean["rainfall_vs_normal_pct"] is None


def test_shape_watch_rounds_and_tolerates_nulls():
    out = repository.shape_watch({**WATCH_ROW, "yield_change": None, "pred_yield": None})
    assert out["yield_change_kg_ha"] is None and out["pred_yield_kg_ha"] is None
    assert out["alert_level"] == "urgent"
