"""Tests for the pure builders in pipeline.db.write_run (no database)."""

import uuid

import pandas as pd
import pytest

from pipeline.db.write_run import (
    aggregate_checks,
    build_kharif_rows,
    build_prediction_rows,
    build_watch_rows,
    require_all_matched,
)

RUN = uuid.uuid4()


def test_prediction_rows_carry_interval_and_null_not_nan():
    df = pd.DataFrame(
        {
            "district_id": [1],
            "crop": ["Wheat"],
            "season": ["Rabi"],
            "current_pred_yield": [4100.5],
            "baseline_yield_year": [2019],
            "prediction_basis": [None],
            "f1": [float("nan")],
            "f2": [2.0],
            "shap_f1": [0.1],
            "shap_f2": [-0.2],
        }
    )
    rows = build_prediction_rows(RUN, df, ["f1", "f2"], [3900.0], [4300.0])
    run_id, district_id, crop, season, pred, lo, up, year, basis, feats, shap = rows[0]
    assert (run_id, district_id, crop, season) == (RUN, 1, "Wheat", "Rabi")
    assert (pred, lo, up, year, basis) == (4100.5, 3900.0, 4300.0, 2019, None)
    assert feats.obj == {"f1": None, "f2": 2.0}
    assert shap.obj == {"f1": 0.1, "f2": -0.2}


def test_watch_rows_handle_missing_changes():
    df = pd.DataFrame(
        {
            "district_id": [1],
            "crop": ["Wheat"],
            "alert_level": ["Stable"],
            "change_note": [float("nan")],
            "yield_change_since_last": [float("nan")],
            "rai_change_since_last": [0.5],
            "severity_change_since_last": [float("nan")],
        }
    )
    prev = uuid.uuid4()
    row = build_watch_rows(RUN, prev, df)[0]
    assert row == (RUN, 1, "Wheat", "Stable", None, None, 0.5, None, prev)


def test_kharif_rows_map_columns():
    df = pd.DataFrame(
        {
            "district_id": [3],
            "actual_partial_rainfall_mm": [735.2],
            "days_elapsed": [120],
            "window_hist_mean": [1067.4],
            "window_hist_std": [170.0],
            "n_years": [6],
            "kharif_risk_zscore": [-1.95],
            "kharif_risk_level": ["high"],
            "risk_explanation": ["Below normal"],
        }
    )
    row = build_kharif_rows(RUN, df)[0]
    assert row == (RUN, 3, 735.2, 120, 1067.4, 170.0, 6, -1.95, "high", "Below normal")


def test_aggregate_checks_status_rules():
    results = [
        {"check": "a", "status": "pass", "detail": "fine"},
        {"check": "a", "status": "warn", "detail": "heads up"},
        {"check": "b", "status": "overridden", "detail": "big", "override_ref": "x (until 10-20)"},
        {"check": "b", "status": "pass", "detail": "other"},
        {"check": "c", "status": "fail", "detail": "broken"},
        {"check": "c", "status": "overridden", "detail": "also", "override_ref": "y"},
    ]
    rows = {name: (status, detail, ref) for name, status, detail, ref in aggregate_checks(results)}
    assert rows["a"] == ("pass", "heads up", None)
    assert rows["b"][0] == "overridden"
    assert rows["b"][2] == "x (until 10-20)"
    assert rows["c"][0] == "fail"


def test_unmatched_or_duplicate_districts_are_refused():
    ok = pd.DataFrame({"district_id": [1, 2], "crop": ["Wheat", "Wheat"]})
    assert len(require_all_matched(ok, 2, "t")) == 2
    with pytest.raises(RuntimeError):
        require_all_matched(ok, 3, "t")
    dup = pd.DataFrame({"district_id": [1, 1], "crop": ["Wheat", "Wheat"]})
    with pytest.raises(RuntimeError):
        require_all_matched(dup, 2, "t")
