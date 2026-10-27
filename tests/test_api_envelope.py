"""Tests for the framework-free API helpers (no web server, no database)."""

import uuid
from datetime import date, datetime, timezone

import pytest

from api.app.envelope import (
    build_meta,
    etag_for,
    etag_matches,
    normalise_crop,
    ok,
    page_info,
    resolve_sort,
    slugify,
)
from api.app.errors import ApiError

RUN = {
    "run_id": uuid.UUID("4ea1963f-c6ac-4b6a-b8a9-761d64e4e53a"),
    "weather_data_as_of": date(2026, 10, 5),
    "model_version": "xgb_tuned_846a40d1",
    "gate_passed": True,
    "finished_at": datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc),
}
ALLOWED = {"pred_yield": "p.pred_yield", "district": "d.name"}


def test_meta_is_json_safe_and_carries_interval_honesty():
    extras = {
        "prediction_basis": "current_2026_weather_with_2019_yield_baseline",
        "baseline_yield_year": 2019,
        "empirical_coverage": {"Wheat": {"n": 236, "coverage": 0.788}},
    }
    meta = build_meta(RUN, extras)
    assert meta["run_id"] == "4ea1963f-c6ac-4b6a-b8a9-761d64e4e53a"
    assert meta["weather_data_as_of"] == "2026-10-05"
    assert meta["generated_at"].startswith("2026-10-08T09:30")
    assert meta["baseline_yield_year"] == 2019
    assert meta["interval"]["nominal_quantiles"] == [0.02, 0.98]
    assert meta["interval"]["empirical_coverage"]["Wheat"]["coverage"] == 0.788


def test_ok_and_page_shapes():
    assert ok([1], meta={"a": 1}, page=page_info(50, 0, 118)) == {
        "data": [1],
        "page": {"limit": 50, "offset": 0, "total": 118},
        "meta": {"a": 1},
    }
    assert ok({"x": 1}) == {"data": {"x": 1}}


def test_sort_only_accepts_allow_listed_names():
    assert resolve_sort("pred_yield", "desc", ALLOWED, "district") == ("p.pred_yield", "DESC")
    assert resolve_sort(None, "asc", ALLOWED, "district") == ("d.name", "ASC")
    for bad in ["pred_yield; DROP TABLE runs", "features", "p.pred_yield"]:
        with pytest.raises(ApiError) as err:
            resolve_sort(bad, "asc", ALLOWED, "district")
        assert err.value.status == 400
        assert err.value.code == "INVALID_SORT"


def test_order_is_validated():
    with pytest.raises(ApiError) as err:
        resolve_sort("pred_yield", "sideways", ALLOWED, "district")
    assert err.value.code == "INVALID_ORDER"


def test_crop_normalisation():
    assert normalise_crop("wheat") == "Wheat"
    assert normalise_crop(" RICE ") == "Rice"
    assert normalise_crop(None) is None
    with pytest.raises(ApiError) as err:
        normalise_crop("maize")
    assert err.value.code == "INVALID_CROP"


def test_etag_depends_on_run_and_query():
    a = etag_for("run-1", "/api/v1/predictions?crop=Wheat")
    assert a == etag_for("run-1", "/api/v1/predictions?crop=Wheat")
    assert a != etag_for("run-2", "/api/v1/predictions?crop=Wheat")
    assert a != etag_for("run-1", "/api/v1/predictions?crop=Rice")
    assert etag_matches(a, a)
    assert etag_matches(f'W/"zzz", {a}', a)
    assert etag_matches("*", a)
    assert not etag_matches(None, a)
    assert not etag_matches('W/"other"', a)


def test_slugify():
    assert slugify("Uttar Pradesh") == "uttar-pradesh"
    assert slugify("Fatehgarh Sahib") == "fatehgarh-sahib"


def test_error_body_shape():
    body = ApiError(404, "DISTRICT_NOT_FOUND", "No district.", {"slug": "x/y"}).body()
    expected = {"code": "DISTRICT_NOT_FOUND", "message": "No district.", "details": {"slug": "x/y"}}
    assert body == {"error": expected}
