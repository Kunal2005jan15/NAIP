# =============================================================
# NAIP REST API
# =============================================================
# Serves the same live outputs shown in the Streamlit dashboard
# (District Watch, Kharif early-warning, current predictions)
# over HTTP, so any MIS/ERP system can consume them directly.
#
# This reads the pipeline's OUTPUT FILES (CSVs written by scripts
# 27/28/32) rather than re-running the model per request, matching
# how the dashboard itself works — the API and dashboard are two
# views onto the same refresh pipeline, never two sources of truth.
#
# Run: uvicorn api.naip_api:app --reload --port 8000
# Docs: http://localhost:8000/docs
# =============================================================

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(
    title="NAIP API",
    description="District-level Wheat & Rice yield decision-support for 118 districts "
                 "across UP, Punjab, and Haryana. Reads the same refresh-pipeline outputs "
                 "as the Streamlit dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Data loading — read once at startup, refreshed on-demand via a
# tiny in-memory cache keyed by file mtime, so a running API server
# picks up a new daily refresh without needing a restart.
# ---------------------------------------------------------------

_cache = {}


def _load_csv(path: Path) -> pd.DataFrame:
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Required data file missing: {path.name}. "
                                                      f"Run the refresh pipeline (src/00_run_full_refresh.py) first.")
    df = pd.read_csv(path)
    _cache[path] = (mtime, df)
    return df


def predictions_df() -> pd.DataFrame:
    return _load_csv(METRICS / "current_predictions_2026.csv")


def watch_df() -> pd.DataFrame:
    return _load_csv(METRICS / "district_watch_feed.csv")


def kharif_df() -> pd.DataFrame:
    return _load_csv(PROCESSED / "kharif_2026_risk_flags.csv")


def districts_df() -> pd.DataFrame:
    # NOTE (2026-07-07 fix): data/processed/district_list.csv is the RAW 133-district
    # source list before modeling filters were applied (see README data table: "133 -> 118
    # districts"). Using it here silently returned 133 districts instead of the 118 the
    # model actually covers. The current_predictions feed reflects the real 118 modeled
    # districts, so that's the source of truth for this endpoint.
    df = _load_csv(METRICS / "current_predictions_2026.csv")
    return df[["state", "district"]].drop_duplicates().reset_index(drop=True)


def feature_list() -> list:
    path = PROCESSED / "feature_list_v2.txt"
    if not path.exists():
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    """System status, 40-feature schema check, data freshness."""
    features = feature_list()
    schema_ok = len(features) == 40

    freshness = {}
    for name, path in [
        ("current_predictions", METRICS / "current_predictions_2026.csv"),
        ("district_watch", METRICS / "district_watch_feed.csv"),
        ("kharif_risk", PROCESSED / "kharif_2026_risk_flags.csv"),
    ]:
        if path.exists():
            df = _load_csv(path)
            as_of_col = next((c for c in ("weather_data_as_of", "data_as_of") if c in df.columns), None)
            freshness[name] = {
                "rows": len(df),
                "weather_data_as_of": str(df[as_of_col].iloc[0]) if as_of_col else None,
            }
        else:
            freshness[name] = {"rows": 0, "weather_data_as_of": None, "status": "MISSING"}

    return {
        "status": "ok" if schema_ok else "degraded",
        "feature_schema": {"expected": 40, "found": len(features), "ok": schema_ok},
        "data_freshness": freshness,
    }


# ---------------------------------------------------------------
# GET /districts
# ---------------------------------------------------------------
@app.get("/districts", tags=["Reference"])
def get_districts():
    """All monitored districts."""
    df = districts_df()
    return df.to_dict(orient="records")


# ---------------------------------------------------------------
# GET /predictions
# ---------------------------------------------------------------
@app.get("/predictions", tags=["Predictions"])
def get_predictions(
    state: Optional[str] = Query(None, description="Filter by state, e.g. 'Punjab'"),
    drought_only: bool = Query(False, description="Only return districts with drought_flag=1"),
    sort_by: Optional[str] = Query(None, description="Column to sort by, e.g. 'current_pred_yield'"),
    descending: bool = Query(False),
):
    """All 118 districts, filterable by state / drought-only / sort order."""
    df = predictions_df().copy()
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if drought_only:
        df = df[df["drought_flag"] == 1]
    if sort_by:
        if sort_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"Unknown sort_by column '{sort_by}'. "
                                                          f"Available: {list(df.columns)}")
        df = df.sort_values(sort_by, ascending=not descending)
    return {"count": len(df), "results": df.to_dict(orient="records")}


# ---------------------------------------------------------------
# GET /predict/{district}
# ---------------------------------------------------------------
@app.get("/predict/{district}", tags=["Predictions"])
def predict_district(district: str):
    """Full prediction for one district: yield, interval, top SHAP factors, drought/flood flags."""
    preds = predictions_df()
    match = preds[preds["district"].str.lower() == district.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found. "
                                                      f"See /districts for valid names.")
    row = match.iloc[0].to_dict()

    # Pull top SHAP factors for this district from the watch feed, if available
    top_shap = []
    watch = watch_df()
    wmatch = watch[watch["district"].str.lower() == district.lower()]
    if not wmatch.empty:
        wrow = wmatch.iloc[0]
        shap_cols = [c for c in watch.columns if c.startswith("shap_")]
        shap_vals = {c.replace("shap_", ""): wrow[c] for c in shap_cols if pd.notna(wrow[c])}
        top_shap = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_shap = [{"feature": f, "shap_value": round(float(v), 2)} for f, v in top_shap]

    return {
        "state": row.get("state"),
        "district": row.get("district"),
        "crop": row.get("crop"),
        "predicted_yield_kg_ha": row.get("current_pred_yield"),
        "change_vs_2019_kg_ha": row.get("change_vs_2019"),
        "drought_flag": bool(row.get("drought_flag")),
        "flood_flag": bool(row.get("flood_flag")),
        "rainfall_anomaly_index": row.get("rainfall_anomaly_index"),
        "weather_data_as_of": row.get("weather_data_as_of"),
        "top_shap_factors": top_shap,
    }


# ---------------------------------------------------------------
# GET /watch
# ---------------------------------------------------------------
@app.get("/watch", tags=["Alerts"])
def get_watch(
    alert_level: Optional[str] = Query(
        None, description="Filter by alert level, e.g. 'urgent', 'watch', 'minor_change', 'new', 'stable'"
    )
):
    """District Watch alert feed, filterable by alert level."""
    df = watch_df().copy()
    if alert_level:
        df = df[df["alert_level"].str.lower() == alert_level.lower()]
    keep_cols = [c for c in [
        "state", "district", "crop", "current_pred_yield", "alert_level", "change_note",
        "yield_change_since_last", "snapshot_timestamp",
    ] if c in df.columns]
    return {"count": len(df), "results": df[keep_cols].to_dict(orient="records")}


# ---------------------------------------------------------------
# GET /kharif-risk
# ---------------------------------------------------------------
@app.get("/kharif-risk", tags=["Alerts"])
def get_kharif_risk(
    risk_level: Optional[str] = Query(None, description="Filter by 'low', 'moderate', 'high', or 'insufficient_data'")
):
    """Kharif 2026 early-warning risk classification. This is a risk flag, not a yield prediction —
    see risk_explanation for the same-calendar-window comparison behind each district's classification."""
    df = kharif_df().copy()
    if risk_level:
        df = df[df["kharif_risk_level"].str.lower() == risk_level.lower()]
    return {"count": len(df), "results": df.to_dict(orient="records")}