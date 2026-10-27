"""All SQL for the read API. Runs as the SELECT-only naip_reader role.

Sort expressions are interpolated into SQL, so they must only ever come from the
allow-lists below via envelope.resolve_sort - never from client input directly.
"""

from psycopg.rows import dict_row

DISTRICT_SORTS = {"state": "state", "district": "name"}
# Vocabularies written by src/28_district_watch.py and src/32_kharif_early_warning.py,
# listed from most to least urgent. They validate filters and drive the SQL ranking below.
WATCH_LEVELS = ("urgent", "watch", "new", "elevated_stable", "minor_change", "stable", "baseline")
KHARIF_LEVELS = ("high", "moderate", "low", "insufficient_data")
_WATCH_RANK = (
    "CASE w.alert_level WHEN 'urgent' THEN 0 WHEN 'watch' THEN 1 WHEN 'new' THEN 2 "
    "WHEN 'elevated_stable' THEN 3 WHEN 'minor_change' THEN 4 WHEN 'stable' THEN 5 ELSE 6 END"
)
_KHARIF_RANK = (
    "CASE k.risk_level WHEN 'high' THEN 0 WHEN 'moderate' THEN 1 WHEN 'low' THEN 2 ELSE 3 END"
)
PREDICTION_SORTS = {
    "pred_yield": "p.pred_yield",
    "pred_lower": "p.pred_lower",
    "pred_upper": "p.pred_upper",
    "state": "d.state",
    "district": "d.name",
    "change_vs_baseline": "(p.pred_yield - f.yield_kg_ha)",
}


def _rows(conn, sql: str, params: dict | None = None) -> list:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params or {})
        return cur.fetchall()


def _one(conn, sql: str, params: dict | None = None):
    rows = _rows(conn, sql, params)
    return rows[0] if rows else None


# ---------- runs ----------
def latest_run(conn):
    return _one(
        conn,
        "SELECT run_id, status, gate_passed, model_version, weather_data_as_of, "
        "git_sha, started_at, finished_at FROM runs WHERE is_latest",
    )


def run_extras(conn, run_id, model_version: str) -> dict:
    basis = _one(
        conn,
        "SELECT prediction_basis, max(baseline_yield_year) AS baseline_yield_year "
        "FROM predictions WHERE run_id = %(run_id)s "
        "GROUP BY prediction_basis ORDER BY count(*) DESC LIMIT 1",
        {"run_id": run_id},
    )
    coverage = _rows(
        conn,
        "SELECT crop, count(*) AS n, avg((actual BETWEEN lower AND upper)::int) AS coverage "
        "FROM test_predictions WHERE model_version = %(mv)s GROUP BY crop ORDER BY crop",
        {"mv": model_version},
    )
    return {
        "prediction_basis": basis["prediction_basis"] if basis else None,
        "baseline_yield_year": basis["baseline_yield_year"] if basis else None,
        "empirical_coverage": {
            r["crop"]: {"n": r["n"], "coverage": round(float(r["coverage"]), 3)} for r in coverage
        },
    }


# ---------- districts ----------
def _district_where(state_slug: str | None) -> tuple[str, dict]:
    if state_slug:
        return "WHERE slug LIKE %(like)s", {"like": f"{state_slug}/%"}
    return "", {}


def list_districts(conn, state_slug, sort_expr, direction, limit, offset):
    where, params = _district_where(state_slug)
    total = _one(conn, f"SELECT count(*) AS n FROM districts {where}", params)["n"]
    rows = _rows(
        conn,
        f"SELECT slug, state, name AS district, lat, lon FROM districts {where} "
        f"ORDER BY {sort_expr} {direction}, slug LIMIT %(limit)s OFFSET %(offset)s",
        {**params, "limit": limit, "offset": offset},
    )
    return rows, total


def get_district(conn, slug: str):
    return _one(
        conn,
        "SELECT district_id, slug, state, name AS district, lat, lon "
        "FROM districts WHERE slug = %(slug)s",
        {"slug": slug},
    )


def district_coverage(conn, district_id: int) -> list:
    return _rows(
        conn,
        "SELECT crop, season, min(year) AS first_year, max(year) AS last_year, "
        "count(*) AS n_years FROM feature_history WHERE district_id = %(id)s "
        "GROUP BY crop, season ORDER BY crop, season",
        {"id": district_id},
    )


def live_crops(conn, run_id, district_id: int) -> list:
    rows = _rows(
        conn,
        "SELECT crop FROM predictions WHERE run_id = %(run_id)s "
        "AND district_id = %(id)s ORDER BY crop",
        {"run_id": run_id, "id": district_id},
    )
    return [r["crop"] for r in rows]


# ---------- predictions ----------
_PREDICTION_FROM = """
FROM predictions p
JOIN districts d USING (district_id)
LEFT JOIN feature_history f
  ON f.district_id = p.district_id AND f.crop = p.crop
 AND f.season = p.season AND f.year = p.baseline_yield_year
"""
_PREDICTION_COLUMNS = """
SELECT d.state, d.name AS district, d.slug, p.crop, p.season,
       p.pred_yield, p.pred_lower, p.pred_upper,
       p.baseline_yield_year, p.prediction_basis,
       (p.features->>'rainfall_anomaly_index')::float8 AS rainfall_anomaly_index,
       (p.features->>'drought_flag')::float8 AS drought_flag,
       (p.features->>'flood_flag')::float8 AS flood_flag,
       f.yield_kg_ha AS baseline_yield
"""


def list_predictions(conn, run_id, crop, state_slug, sort_expr, direction, limit, offset):
    conditions, params = ["p.run_id = %(run_id)s"], {"run_id": run_id}
    if crop:
        conditions.append("p.crop = %(crop)s")
        params["crop"] = crop
    if state_slug:
        conditions.append("d.slug LIKE %(like)s")
        params["like"] = f"{state_slug}/%"
    where = "WHERE " + " AND ".join(conditions)
    total = _one(conn, f"SELECT count(*) AS n {_PREDICTION_FROM} {where}", params)["n"]
    rows = _rows(
        conn,
        f"{_PREDICTION_COLUMNS} {_PREDICTION_FROM} {where} "
        f"ORDER BY {sort_expr} {direction}, d.slug LIMIT %(limit)s OFFSET %(offset)s",
        {**params, "limit": limit, "offset": offset},
    )
    return rows, total


def get_prediction(conn, run_id, slug: str, crop: str):
    return _one(
        conn,
        f"{_PREDICTION_COLUMNS} {_PREDICTION_FROM} "
        "WHERE p.run_id = %(run_id)s AND d.slug = %(slug)s AND p.crop = %(crop)s",
        {"run_id": run_id, "slug": slug, "crop": crop},
    )


def _round(value, digits):
    return None if value is None else round(float(value), digits)


def _flag(value):
    return None if value is None else bool(value)


def shape_prediction(row: dict, detail: bool = False) -> dict:
    point, lower, upper = row["pred_yield"], row["pred_lower"], row["pred_upper"]
    baseline = row["baseline_yield"]
    out = {
        "state": row["state"],
        "district": row["district"],
        "slug": row["slug"],
        "crop": row["crop"],
        "season": row["season"],
        "pred_yield_kg_ha": _round(point, 1),
        "pred_lower": _round(lower, 1),
        "pred_upper": _round(upper, 1),
        # the point and the band come from three separate models, so they can disagree
        "contains_point": None if lower is None or upper is None else lower <= point <= upper,
        "drought_flag": _flag(row["drought_flag"]),
        "flood_flag": _flag(row["flood_flag"]),
        "rainfall_anomaly_index": _round(row["rainfall_anomaly_index"], 2),
        "change_vs_baseline_kg_ha": None if baseline is None else _round(point - baseline, 1),
    }
    if detail:
        out["baseline_yield_kg_ha"] = _round(baseline, 1)
        out["baseline_yield_year"] = row["baseline_yield_year"]
        out["prediction_basis"] = row["prediction_basis"]
        out["interval_width_kg_ha"] = (
            None if lower is None or upper is None else _round(upper - lower, 1)
        )
    return out


# ---------- district watch ----------
_WATCH_FROM = """
FROM watch_events w
JOIN districts d USING (district_id)
LEFT JOIN predictions p
  ON p.run_id = w.run_id AND p.district_id = w.district_id AND p.crop = w.crop
"""
_WATCH_COLUMNS = """
SELECT d.state, d.name AS district, d.slug, w.crop, w.alert_level, w.change_note,
       w.yield_change, w.rai_change, w.severity_change, p.pred_yield
"""


def list_watch(conn, run_id, level, crop, state_slug, limit, offset):
    conditions, params = ["w.run_id = %(run_id)s"], {"run_id": run_id}
    if level:
        conditions.append("w.alert_level = %(level)s")
        params["level"] = level
    if crop:
        conditions.append("w.crop = %(crop)s")
        params["crop"] = crop
    if state_slug:
        conditions.append("d.slug LIKE %(like)s")
        params["like"] = f"{state_slug}/%"
    where = "WHERE " + " AND ".join(conditions)
    total = _one(conn, f"SELECT count(*) AS n {_WATCH_FROM} {where}", params)["n"]
    rows = _rows(
        conn,
        f"{_WATCH_COLUMNS} {_WATCH_FROM} {where} "
        f"ORDER BY {_WATCH_RANK}, abs(coalesce(w.yield_change, 0)) DESC, d.slug, w.crop "
        "LIMIT %(limit)s OFFSET %(offset)s",
        {**params, "limit": limit, "offset": offset},
    )
    return rows, total


def watch_counts(conn, run_id) -> dict:
    rows = _rows(
        conn,
        "SELECT alert_level, count(*) AS n FROM watch_events "
        "WHERE run_id = %(run_id)s GROUP BY alert_level",
        {"run_id": run_id},
    )
    return {r["alert_level"]: r["n"] for r in rows}


def shape_watch(row: dict) -> dict:
    return {
        "state": row["state"],
        "district": row["district"],
        "slug": row["slug"],
        "crop": row["crop"],
        "alert_level": row["alert_level"],
        "change_note": row["change_note"],
        "pred_yield_kg_ha": _round(row["pred_yield"], 1),
        "yield_change_kg_ha": _round(row["yield_change"], 1),
        "rai_change": _round(row["rai_change"], 2),
        "severity_change": _round(row["severity_change"], 2),
    }


# ---------- kharif early warning ----------
_KHARIF_FROM = "FROM kharif_risk k JOIN districts d USING (district_id)"
_KHARIF_COLUMNS = """
SELECT d.state, d.name AS district, d.slug, k.partial_rainfall_mm, k.days_elapsed,
       k.hist_mean, k.hist_std, k.n_years, k.zscore, k.risk_level, k.explanation
"""


def list_kharif(conn, run_id, level, state_slug, limit, offset):
    conditions, params = ["k.run_id = %(run_id)s"], {"run_id": run_id}
    if level:
        conditions.append("k.risk_level = %(level)s")
        params["level"] = level
    if state_slug:
        conditions.append("d.slug LIKE %(like)s")
        params["like"] = f"{state_slug}/%"
    where = "WHERE " + " AND ".join(conditions)
    total = _one(conn, f"SELECT count(*) AS n {_KHARIF_FROM} {where}", params)["n"]
    rows = _rows(
        conn,
        f"{_KHARIF_COLUMNS} {_KHARIF_FROM} {where} "
        f"ORDER BY {_KHARIF_RANK}, k.zscore ASC NULLS LAST, d.slug "
        "LIMIT %(limit)s OFFSET %(offset)s",
        {**params, "limit": limit, "offset": offset},
    )
    return rows, total


def get_kharif(conn, run_id, slug: str):
    return _one(
        conn,
        f"{_KHARIF_COLUMNS} {_KHARIF_FROM} WHERE k.run_id = %(run_id)s AND d.slug = %(slug)s",
        {"run_id": run_id, "slug": slug},
    )


def kharif_counts(conn, run_id) -> dict:
    rows = _rows(
        conn,
        "SELECT risk_level, count(*) AS n FROM kharif_risk "
        "WHERE run_id = %(run_id)s GROUP BY risk_level",
        {"run_id": run_id},
    )
    return {r["risk_level"]: r["n"] for r in rows}


def shape_kharif(row: dict) -> dict:
    partial, mean = row["partial_rainfall_mm"], row["hist_mean"]
    # same-calendar-window comparison: the baseline is the historical rainfall for the
    # same days of the season, never a full-season total (see README bug #3)
    vs_normal = None
    if partial is not None and mean is not None and mean > 0:
        vs_normal = _round((partial - mean) / mean * 100, 1)
    return {
        "state": row["state"],
        "district": row["district"],
        "slug": row["slug"],
        "risk_level": row["risk_level"],
        "zscore": _round(row["zscore"], 2),
        "partial_rainfall_mm": _round(partial, 1),
        "days_elapsed": row["days_elapsed"],
        "hist_mean_mm": _round(mean, 1),
        "hist_std_mm": _round(row["hist_std"], 1),
        "n_years": row["n_years"],
        "rainfall_vs_normal_pct": vs_normal,
        "explanation": row["explanation"],
    }
