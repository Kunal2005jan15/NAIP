"""Route-level tests: the real FastAPI app with the repository layer replaced by canned data."""

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.app import repository
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
EXTRAS = {
    "prediction_basis": "current_2026_weather_with_2019_yield_baseline",
    "baseline_yield_year": 2019,
    "empirical_coverage": {"Wheat": {"n": 236, "coverage": 0.788}},
}
PRED_ROW = {
    "state": "Haryana",
    "district": "Sirsa",
    "slug": "haryana/sirsa",
    "crop": "Wheat",
    "season": "Rabi",
    "pred_yield": 5173.259,
    "pred_lower": 3742.2183,
    "pred_upper": 5026.95,
    "baseline_yield_year": 2019,
    "prediction_basis": EXTRAS["prediction_basis"],
    "rainfall_anomaly_index": 0.4167,
    "drought_flag": 0.0,
    "flood_flag": 0.0,
    "baseline_yield": 4900.0,
}


@pytest.fixture
def client(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_conn] = lambda: object()
    monkeypatch.setattr(repository, "latest_run", lambda conn: RUN)
    monkeypatch.setattr(repository, "run_extras", lambda conn, run_id, mv: EXTRAS)
    monkeypatch.setattr(repository, "list_predictions", lambda conn, *a: ([PRED_ROW], 1))
    monkeypatch.setattr(repository, "get_district", lambda conn, slug: None)
    v1._META_CACHE.clear()
    return TestClient(app)


def test_predictions_envelope_headers_and_contains_point(client):
    response = client.get("/api/v1/predictions?crop=wheat")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data", "page", "meta"}
    assert body["page"] == {"limit": 50, "offset": 0, "total": 1}
    assert body["meta"]["interval"]["nominal_quantiles"] == [0.02, 0.98]
    assert body["meta"]["baseline_yield_year"] == 2019
    assert body["data"][0]["contains_point"] is False
    assert response.headers["etag"].startswith('W/"')
    assert "s-maxage=300" in response.headers["cache-control"]


def test_etag_revalidation_returns_304(client):
    first = client.get("/api/v1/predictions")
    second = client.get("/api/v1/predictions", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304


def test_unknown_sort_is_a_400_with_the_error_envelope(client):
    response = client.get("/api/v1/predictions?sort=features")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SORT"


def test_out_of_range_limit_is_a_422_envelope(client):
    response = client.get("/api/v1/predictions?limit=0")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_district_is_a_404_envelope(client):
    response = client.get("/api/v1/districts/haryana/atlantis/prediction")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DISTRICT_NOT_FOUND"


def test_no_promoted_run_is_a_409_but_health_still_answers(client, monkeypatch):
    monkeypatch.setattr(repository, "latest_run", lambda conn: None)
    assert client.get("/api/v1/predictions").status_code == 409
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["data"]["status"] == "no_promoted_run"


def test_rice_returns_an_empty_list_with_a_notice(client, monkeypatch):
    monkeypatch.setattr(repository, "list_predictions", lambda conn, *a: ([], 0))
    body = client.get("/api/v1/predictions?crop=Rice").json()
    assert body["data"] == []
    assert "Wheat only" in body["meta"]["notice"]


def test_health_reports_fresh_data(client):
    data = client.get("/api/v1/health").json()["data"]
    assert data["status"] == "ok"
    assert data["data_age_days"] == 1
    assert data["latest_run"]["gate_passed"] is True


def test_unknown_route_uses_the_error_envelope(client):
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
