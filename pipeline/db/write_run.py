"""Write one refresh run into Postgres and promote it if the sanity gate passed.

Usage: python -m pipeline.db.write_run [writer|owner]

Run it after the refresh (src/00_run_full_refresh.py, which ends with the
sanity gate in src/33). It reads the files the refresh produced, computes the
80% prediction interval with the serving quantile models, and follows
docs/database.md section 8:

  1. insert a runs row with status 'running'
  2. write predictions, watch_events, kharif_risk and sanity_checks under it
  3. record the gate result
  4. in one transaction: promote (is_latest) if the gate passed, else mark 'flagged'

Any failure after step 1 marks the run 'failed' and leaves the previously
promoted run untouched.
"""

import json
import os
import pickle
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
from psycopg.types.json import Jsonb

from pipeline.db.connection import connect
from pipeline.db.seed_reference import (
    MODELS_DIR,
    ROOT,
    SERVING_MODELS,
    attach_district_id,
    find_file,
    sha256_file,
)

LIVE_FILES = {
    "predictions": ROOT / "data" / "processed" / "current_predictions_full.csv",
    "watch": ROOT / "outputs" / "metrics" / "district_watch_feed.csv",
    "kharif": ROOT / "data" / "processed" / "kharif_2026_risk_flags.csv",
    "sanity": ROOT / "data" / "processed" / "sanity_report.json",
}
CROP_SEASON = {"Wheat": "Rabi", "Rice": "Kharif"}

SQL_RUN = """
INSERT INTO runs
  (run_id, started_at, status, model_version, git_sha, weather_data_as_of)
VALUES (%s, now(), 'running', %s, %s, %s)
"""
SQL_PREDICTION = """
INSERT INTO predictions
  (run_id, district_id, crop, season, pred_yield, pred_lower, pred_upper,
   baseline_yield_year, prediction_basis, features, shap)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
SQL_WATCH = """
INSERT INTO watch_events
  (run_id, district_id, crop, alert_level, change_note, yield_change,
   rai_change, severity_change, prev_run_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
SQL_KHARIF = """
INSERT INTO kharif_risk
  (run_id, district_id, partial_rainfall_mm, days_elapsed, hist_mean, hist_std,
   n_years, zscore, risk_level, explanation)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
SQL_SANITY = """
INSERT INTO sanity_checks (run_id, check_name, status, detail, override_ref)
VALUES (%s, %s, %s, %s, %s)
"""


# ---------- small conversion helpers ----------
def _num(value):
    return None if pd.isna(value) else float(value)


def _int(value):
    return None if pd.isna(value) else int(value)


def _text(value):
    return None if pd.isna(value) else str(value)


def _records(df: pd.DataFrame) -> list:
    """List of dicts with NaN written as None (a JSONB column cannot hold NaN)."""
    return json.loads(df.to_json(orient="records", double_precision=15))


# ---------- pure builders (unit-tested, no database) ----------
def build_prediction_rows(run_id, df, features, lower, upper) -> list:
    feats = _records(df[features])
    shap_cols = [f"shap_{f}" for f in features]
    shaps = _records(df[shap_cols].rename(columns=lambda c: c.removeprefix("shap_")))
    rows = []
    for r, lo, up, f, s in zip(df.itertuples(), lower, upper, feats, shaps, strict=True):
        season = _text(getattr(r, "season", None)) or CROP_SEASON[r.crop]
        rows.append(
            (
                run_id,
                int(r.district_id),
                r.crop,
                season,
                float(r.current_pred_yield),
                _num(lo),
                _num(up),
                _int(r.baseline_yield_year),
                _text(r.prediction_basis),
                Jsonb(f),
                Jsonb(s),
            )
        )
    return rows


def build_watch_rows(run_id, prev_run_id, df) -> list:
    return [
        (
            run_id,
            int(r.district_id),
            r.crop,
            r.alert_level,
            _text(r.change_note),
            _num(r.yield_change_since_last),
            _num(r.rai_change_since_last),
            _num(r.severity_change_since_last),
            prev_run_id,
        )
        for r in df.itertuples()
    ]


def build_kharif_rows(run_id, df) -> list:
    return [
        (
            run_id,
            int(r.district_id),
            _num(r.actual_partial_rainfall_mm),
            _int(r.days_elapsed),
            _num(r.window_hist_mean),
            _num(r.window_hist_std),
            _int(r.n_years),
            _num(r.kharif_risk_zscore),
            _text(r.kharif_risk_level),
            _text(r.risk_explanation),
        )
        for r in df.itertuples()
    ]


def aggregate_checks(results: list) -> list:
    """Collapse the per-line results from src/33 into one row per named check.

    Status is 'fail' if any line failed, else 'overridden' if any line was
    overridden, else 'pass'. Warnings that are not overrides keep status 'pass'
    but stay visible in the detail text.
    """
    order, grouped = [], {}
    for item in results:
        if item["check"] not in grouped:
            grouped[item["check"]] = []
            order.append(item["check"])
        grouped[item["check"]].append(item)
    rows = []
    for name in order:
        items = grouped[name]
        statuses = {i["status"] for i in items}
        if "fail" in statuses:
            status = "fail"
        elif "overridden" in statuses:
            status = "overridden"
        else:
            status = "pass"
        notable = [i["detail"] for i in items if i["status"] != "pass"]
        detail = " | ".join(notable if notable else [i["detail"] for i in items])
        refs = [i["override_ref"] for i in items if i.get("override_ref")]
        rows.append((name, status, detail[:2000], "; ".join(refs) or None))
    return rows


def require_all_matched(df: pd.DataFrame, expected: int, source: str) -> pd.DataFrame:
    if len(df) != expected:
        raise RuntimeError(f"{source}: only {len(df)} of {expected} rows matched a known district")
    if df.duplicated(["district_id", "crop"] if "crop" in df.columns else ["district_id"]).any():
        raise RuntimeError(f"{source}: duplicate district rows")
    return df


def require_fresh(report_path: Path, inputs: list) -> None:
    newest_input = max(p.stat().st_mtime for p in inputs)
    if report_path.stat().st_mtime < newest_input:
        raise RuntimeError(
            "sanity_report.json is older than the prediction files. "
            "Re-run: python src/33_pipeline_sanity_checks.py"
        )


# ---------- environment-dependent helpers ----------
def git_sha():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_verified_models(conn):
    """Return (model_version, lower_model, upper_model) after checking file hashes.

    The hashes recorded by the seed script must match the files on disk, so a
    swapped or retrained model is refused instead of silently used.
    """
    row = conn.execute(
        "SELECT model_version, metrics FROM model_versions WHERE is_serving"
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "No serving model in the database. Run: python -m pipeline.db.seed_reference"
        )
    version, metrics = row
    expected = (metrics or {}).get("artifacts", {})
    for name in SERVING_MODELS:
        actual = sha256_file(MODELS_DIR / name)
        if actual != expected.get(name):
            raise RuntimeError(
                f"models/{name} does not match the hash recorded for {version}. "
                "If you retrained, re-run: python -m pipeline.db.seed_reference"
            )
    lower_model = _load_pickle(MODELS_DIR / "xgb_lower.pkl")
    upper_model = _load_pickle(MODELS_DIR / "xgb_upper.pkl")
    return version, lower_model, upper_model


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "writer"
    for path in LIVE_FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the refresh first (run_refresh.bat).")
    require_fresh(LIVE_FILES["sanity"], list(LIVE_FILES.values())[:3])

    text = find_file("feature_list_v2.txt").read_text(encoding="utf-8")
    features = [line.strip() for line in text.splitlines() if line.strip()]
    report = json.loads(LIVE_FILES["sanity"].read_text(encoding="utf-8"))
    preds_raw = pd.read_csv(LIVE_FILES["predictions"])
    watch_raw = pd.read_csv(LIVE_FILES["watch"])
    kharif_raw = pd.read_csv(LIVE_FILES["kharif"])

    with connect(role) as conn:
        district_rows = conn.execute("SELECT state, name, district_id FROM districts").fetchall()
        ids = {(state, name): district_id for state, name, district_id in district_rows}
        version, lower_model, upper_model = load_verified_models(conn)

        preds = require_all_matched(
            attach_district_id(preds_raw, ids, "current_predictions_full.csv"),
            len(preds_raw),
            "predictions",
        )
        watch = require_all_matched(
            attach_district_id(watch_raw, ids, "district_watch_feed.csv"),
            len(watch_raw),
            "watch feed",
        )
        kharif = require_all_matched(
            attach_district_id(kharif_raw, ids, "kharif_2026_risk_flags.csv"),
            len(kharif_raw),
            "kharif risk",
        )
        x = preds[features].astype(float)
        lower, upper = lower_model.predict(x), upper_model.predict(x)
        as_of = pd.to_datetime(preds["weather_data_as_of"]).max().date()

        run_id = uuid.uuid4()
        conn.execute(SQL_RUN, (run_id, version, git_sha(), as_of))
        conn.commit()  # the 'running' row exists even if what follows fails
        try:
            previous = conn.execute("SELECT run_id FROM runs WHERE is_latest").fetchone()
            prev_run_id = previous[0] if previous else None
            with conn.cursor() as cur:
                prediction_rows = build_prediction_rows(run_id, preds, features, lower, upper)
                cur.executemany(SQL_PREDICTION, prediction_rows)
                cur.executemany(SQL_WATCH, build_watch_rows(run_id, prev_run_id, watch))
                cur.executemany(SQL_KHARIF, build_kharif_rows(run_id, kharif))
                cur.executemany(
                    SQL_SANITY, [(run_id, *row) for row in aggregate_checks(report["results"])]
                )
            gate_passed = bool(report["passed"])
            if gate_passed:
                conn.execute("UPDATE runs SET is_latest = FALSE WHERE is_latest")
                conn.execute(
                    "UPDATE runs SET is_latest = TRUE, status = 'ok', gate_passed = TRUE, "
                    "gate_report = %s, finished_at = now() WHERE run_id = %s",
                    (Jsonb(report), run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET status = 'flagged', gate_passed = FALSE, "
                    "gate_report = %s, finished_at = now() WHERE run_id = %s",
                    (Jsonb(report), run_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = now() WHERE run_id = %s",
                (run_id,),
            )
            conn.commit()
            raise

    print(f"run_id:        {run_id}")
    print(f"model:         {version}")
    print(f"weather as of: {as_of}")
    print(f"predictions:   {len(preds)}   watch_events: {len(watch)}   kharif_risk: {len(kharif)}")
    print(f"sanity gate:   {'PASSED' if gate_passed else 'FAILED'} ({report['failures']} failures)")
    print("result:        " + ("promoted to latest" if gate_passed else "FLAGGED (not promoted)"))


if __name__ == "__main__":
    main()
