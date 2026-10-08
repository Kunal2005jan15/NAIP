"""Load reference data into Postgres.

Usage: python -m pipeline.db.seed_reference [owner|writer]

Reads from data/reference/ if present, otherwise data/processed/ (and
outputs/metrics/ for test predictions). Idempotent: re-running updates the
same rows. Everything loads in one transaction, so a failure leaves the
database unchanged.
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg.types.json import Jsonb

from pipeline.db.connection import connect

ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIRS = [
    ROOT / "data" / "reference",
    ROOT / "data" / "processed",
    ROOT / "outputs" / "metrics",
]
MODELS_DIR = ROOT / "models"
SERVING_MODELS = ("xgb_tuned.pkl", "xgb_lower.pkl", "xgb_upper.pkl")

# Same district, different spelling between source files and the district list.
# Mohali is deliberately absent: it is not in the 118-district list, so it is skipped.
ALIASES = {
    ("Punjab", "Shahid Bhagat Singh Nagar"): "Nawanshahr",  # see src/09_fix_failed_geocodes.py
    ("Punjab", "Ferozepur"): "Firozepur",
    ("Punjab", "Mukatsar"): "Muktsar",
    ("Punjab", "Ropar"): "Rupnagar",
}

# (source column, season, metric) for each environment file.
ENV_SPECS = {
    "nasa_seasonal_ALL_DISTRICTS.csv": [
        ("nasa_rainfall_kharif", "Kharif", "rainfall_mm"),
        ("nasa_rainfall_rabi", "Rabi", "rainfall_mm"),
        ("nasa_temp_avg_kharif", "Kharif", "temp_avg_c"),
        ("nasa_temp_avg_rabi", "Rabi", "temp_avg_c"),
        ("nasa_temp_max_kharif", "Kharif", "temp_max_c"),
        ("nasa_temp_max_rabi", "Rabi", "temp_max_c"),
    ],
    "ndvi_seasonal.csv": [
        ("ndvi_kharif", "Kharif", "ndvi"),
        ("ndvi_rabi", "Rabi", "ndvi"),
    ],
    "surface_water_seasonal.csv": [
        ("water_pct_kharif", "Kharif", "water_pct"),
        ("water_pct_rabi", "Rabi", "water_pct"),
    ],
    "groundwater_district.csv": [
        ("gw_depth_premonsoon_m", "Annual", "gw_depth_pre_m"),
        ("gw_depth_postmonsoon_m", "Annual", "gw_depth_post_m"),
    ],
}

SQL_DISTRICT = """
INSERT INTO districts (district_id, slug, state, name, lat, lon)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (district_id) DO UPDATE SET
  slug = EXCLUDED.slug, state = EXCLUDED.state, name = EXCLUDED.name,
  lat = EXCLUDED.lat, lon = EXCLUDED.lon
"""
SQL_MODEL = """
INSERT INTO model_versions
  (model_version, trained_at, feature_list, metrics, artifact_sha256, is_serving)
VALUES (%s, %s, %s, %s, %s, TRUE)
ON CONFLICT (model_version) DO UPDATE SET
  feature_list = EXCLUDED.feature_list,
  metrics = EXCLUDED.metrics,
  artifact_sha256 = EXCLUDED.artifact_sha256,
  trained_at = COALESCE(model_versions.trained_at, EXCLUDED.trained_at)
"""
SQL_FEATURE_HISTORY = """
INSERT INTO feature_history
  (district_id, crop, season, year, yield_kg_ha, area_ha, features)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (district_id, crop, season, year) DO UPDATE SET
  yield_kg_ha = EXCLUDED.yield_kg_ha, area_ha = EXCLUDED.area_ha,
  features = EXCLUDED.features
"""
SQL_ENV = """
INSERT INTO env_seasonal (district_id, year, season, metric, value)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (district_id, year, season, metric) DO UPDATE SET value = EXCLUDED.value
"""
SQL_TEST_PRED = """
INSERT INTO test_predictions
  (model_version, district_id, crop, year, actual, predicted, lower, upper)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (model_version, district_id, crop, year) DO UPDATE SET
  actual = EXCLUDED.actual, predicted = EXCLUDED.predicted,
  lower = EXCLUDED.lower, upper = EXCLUDED.upper
"""


def find_file(name: str) -> Path:
    for directory in SEARCH_DIRS:
        path = directory / name
        if path.exists():
            return path
    searched = [str(d) for d in SEARCH_DIRS]
    raise FileNotFoundError(f"{name} not found in any of {searched}")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(find_file(name))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _num(value):
    """Python float, or None for NaN/missing (Postgres JSON cannot hold NaN)."""
    return None if pd.isna(value) else float(value)


def attach_district_id(df: pd.DataFrame, ids: dict, source: str) -> pd.DataFrame:
    """Add district_id, applying spelling aliases; drop and report unknown districts."""
    names = [ALIASES.get((s, d), d) for s, d in zip(df["state"], df["district"], strict=True)]
    found = [ids.get((s, n)) for s, n in zip(df["state"], names, strict=True)]
    df = pd.concat([df, pd.Series(found, index=df.index, name="district_id")], axis=1)
    unknown = df[df["district_id"].isna()]
    if len(unknown):
        pairs = sorted(set(zip(unknown["state"], unknown["district"], strict=True)))
        print(f"  skipped {len(unknown)} rows from {source}: unknown districts {pairs}")
    return df.dropna(subset=["district_id"]).astype({"district_id": int})


def long_frame(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """Turn wide columns into (district_id, year, season, metric, value) rows."""
    parts = []
    for column, season, metric in specs:
        part = df[["district_id", "year", column]].rename(columns={column: "value"})
        parts.append(part.assign(season=season, metric=metric))
    return pd.concat(parts, ignore_index=True).dropna(subset=["value"])


def build_feature_rows(model_df: pd.DataFrame, ids: dict, features: list) -> list:
    df = attach_district_id(model_df, ids, "model_ready_v2.csv")
    # to_json writes NaN as null, which is what a JSONB column needs
    feature_dicts = json.loads(df[features].to_json(orient="records", double_precision=15))
    rows = []
    for record, feats in zip(df.itertuples(), feature_dicts, strict=True):
        rows.append(
            (
                int(record.district_id),
                record.crop,
                record.season,
                int(record.year),
                _num(record.yield_kg_ha),
                _num(record.area_ha),
                Jsonb(feats),
            )
        )
    return rows


def build_env_rows(ids: dict) -> list:
    frames = []
    for filename, specs in ENV_SPECS.items():
        df = attach_district_id(read_csv(filename), ids, filename)
        frames.append(long_frame(df, specs))
    env = pd.concat(frames, ignore_index=True)
    env = env.drop_duplicates(["district_id", "year", "season", "metric"], keep="last")
    return [
        (int(r.district_id), int(r.year), r.season, r.metric, float(r.value))
        for r in env.itertuples()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_districts(conn) -> dict:
    df = read_csv("districts_for_nasa_fetch.csv")
    df["slug"] = df["state"].map(slugify) + "/" + df["district"].map(slugify)
    existing = dict(conn.execute("SELECT slug, district_id FROM districts").fetchall())
    next_id = max(existing.values(), default=0) + 1
    rows, ids = [], {}
    for r in df.sort_values(["state", "district"]).itertuples():
        district_id = existing.get(r.slug)
        if district_id is None:
            district_id, next_id = next_id, next_id + 1
        rows.append((district_id, r.slug, r.state, r.district, float(r.lat), float(r.lon)))
        ids[(r.state, r.district)] = district_id
    with conn.cursor() as cur:
        cur.executemany(SQL_DISTRICT, rows)
    print(f"districts: {len(rows)}")
    return ids


def seed_model_version(conn, features: list) -> str:
    paths = {name: MODELS_DIR / name for name in SERVING_MODELS}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Serving model missing: {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    tuned_hash = hashes["xgb_tuned.pkl"]
    # content-derived, so the same artifact gets the same version on every machine
    version = f"xgb_tuned_{tuned_hash[:8]}"
    trained_at = datetime.fromtimestamp(paths["xgb_tuned.pkl"].stat().st_mtime, tz=timezone.utc)
    metrics = {"artifacts": hashes, "loaded_by": "seed_reference"}
    conn.execute(SQL_MODEL, (version, trained_at, Jsonb(features), Jsonb(metrics), tuned_hash))
    conn.execute("UPDATE model_versions SET is_serving = (model_version = %s)", (version,))
    print(f"model_versions: {version} (serving)")
    return version


def seed_feature_history(conn, ids: dict, features: list) -> None:
    rows = build_feature_rows(read_csv("model_ready_v2.csv"), ids, features)
    with conn.cursor() as cur:
        cur.executemany(SQL_FEATURE_HISTORY, rows)
    print(f"feature_history: {len(rows)}")


def seed_env_seasonal(conn, ids: dict) -> None:
    rows = build_env_rows(ids)
    with conn.cursor() as cur:
        cur.executemany(SQL_ENV, rows)
    print(f"env_seasonal: {len(rows)}")


def seed_test_predictions(conn, ids: dict, version: str) -> None:
    name = "test_predictions_FINAL.csv"
    df = attach_district_id(read_csv(name), ids, name)
    rows = [
        (
            version,
            int(r.district_id),
            r.crop,
            int(r.year),
            _num(r.actual_yield),
            _num(r.pred_yield),
            _num(r.pred_lower),
            _num(r.pred_upper),
        )
        for r in df.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany(SQL_TEST_PRED, rows)
    print(f"test_predictions: {len(rows)}")


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "owner"
    text = find_file("feature_list_v2.txt").read_text(encoding="utf-8")
    features = [line.strip() for line in text.splitlines() if line.strip()]
    with connect(role) as conn:  # commits on success, rolls back on any error
        ids = seed_districts(conn)
        version = seed_model_version(conn, features)
        seed_feature_history(conn, ids, features)
        seed_env_seasonal(conn, ids)
        seed_test_predictions(conn, ids, version)
    print("Seed complete.")


if __name__ == "__main__":
    main()
