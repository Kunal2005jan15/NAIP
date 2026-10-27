"""Tests for SQL building and row shaping in api.app.repository, using a fake connection."""

from api.app import repository


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


def prediction_row(**overrides):
    row = {
        "state": "Haryana",
        "district": "Sirsa",
        "slug": "haryana/sirsa",
        "crop": "Wheat",
        "season": "Rabi",
        "pred_yield": 5173.259,
        "pred_lower": 3742.2183,
        "pred_upper": 5026.95,
        "baseline_yield_year": 2019,
        "prediction_basis": "current_2026_weather_with_2019_yield_baseline",
        "rainfall_anomaly_index": 0.4167,
        "drought_flag": 0.0,
        "flood_flag": 1.0,
        "baseline_yield": 4900.0,
    }
    row.update(overrides)
    return row


def test_contains_point_false_when_point_is_outside_the_band():
    out = repository.shape_prediction(prediction_row())
    assert out["contains_point"] is False
    assert out["pred_yield_kg_ha"] == 5173.3
    assert out["change_vs_baseline_kg_ha"] == 273.3
    assert out["drought_flag"] is False
    assert out["flood_flag"] is True
    assert out["rainfall_anomaly_index"] == 0.42


def test_contains_point_true_and_missing_values_become_none():
    inside = repository.shape_prediction(prediction_row(pred_lower=4000.0, pred_upper=5500.0))
    assert inside["contains_point"] is True
    unknown = repository.shape_prediction(
        prediction_row(pred_lower=None, pred_upper=None, baseline_yield=None, drought_flag=None)
    )
    assert unknown["contains_point"] is None
    assert unknown["change_vs_baseline_kg_ha"] is None
    assert unknown["drought_flag"] is None
    assert unknown["pred_lower"] is None


def test_detail_adds_baseline_and_interval_width():
    out = repository.shape_prediction(prediction_row(pred_lower=4000.0, pred_upper=5500.0), True)
    assert out["baseline_yield_kg_ha"] == 4900.0
    assert out["baseline_yield_year"] == 2019
    assert out["interval_width_kg_ha"] == 1500.0
    assert out["prediction_basis"].startswith("current_2026")


def test_list_predictions_filters_and_uses_given_sort_only():
    conn = FakeConn([{"n": 2}], [prediction_row()])
    rows, total = repository.list_predictions(
        conn, "run-1", "Wheat", "punjab", "p.pred_yield", "DESC", 10, 20
    )
    assert total == 2 and len(rows) == 1
    count_sql, count_params = conn.calls[0]
    list_sql, list_params = conn.calls[1]
    assert "p.crop = %(crop)s" in count_sql and "d.slug LIKE %(like)s" in count_sql
    assert count_params == {"run_id": "run-1", "crop": "Wheat", "like": "punjab/%"}
    assert "ORDER BY p.pred_yield DESC, d.slug" in list_sql
    assert list_params["limit"] == 10 and list_params["offset"] == 20


def test_list_predictions_without_filters_has_only_the_run_condition():
    conn = FakeConn([{"n": 0}], [])
    repository.list_predictions(conn, "run-1", None, None, "d.name", "ASC", 50, 0)
    sql, params = conn.calls[0]
    assert "p.crop = %(crop)s" not in sql and "LIKE" not in sql
    assert params == {"run_id": "run-1"}


def test_list_districts_state_filter():
    conn = FakeConn([{"n": 21}], [])
    repository.list_districts(conn, "punjab", "state", "ASC", 50, 0)
    sql, params = conn.calls[0]
    assert "WHERE slug LIKE %(like)s" in sql
    assert params == {"like": "punjab/%"}


def test_run_extras_rounds_coverage_and_handles_empty_run():
    conn = FakeConn(
        [{"prediction_basis": "basis", "baseline_yield_year": 2019}],
        [{"crop": "Wheat", "n": 236, "coverage": 0.78813559}],
    )
    out = repository.run_extras(conn, "run-1", "xgb_tuned_x")
    assert out["baseline_yield_year"] == 2019
    assert out["empirical_coverage"] == {"Wheat": {"n": 236, "coverage": 0.788}}
    empty = repository.run_extras(FakeConn([], []), "run-1", "xgb_tuned_x")
    assert empty["prediction_basis"] is None
    assert empty["empirical_coverage"] == {}
