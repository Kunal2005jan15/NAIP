# =============================================================
# NAIP Project - Step 43: Demo preflight check
# =============================================================
# Run from the repo root before a demo or presentation:
#     python src/43_demo_preflight.py
# Reports, in one screen, whether the dashboard and the API can start:
#   - every file the dashboard reads is present and non-empty
#   - how old the live data is
#   - the serving models match the 40-feature schema
#   - optional pieces (Tavily key, database URL) and which Python packages are missing
# Exit code 0 = ready, 1 = something required is missing.
# =============================================================

import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# (path, what breaks without it, how to regenerate it)
REQUIRED = [
    ("data/processed/model_ready_v2.csv", "dashboard history + scenario tabs", "src/17_feature_engineering_v2.py"),
    ("data/processed/feature_list_v2.txt", "model schema", "src/17_feature_engineering_v2.py"),
    ("data/processed/current_predictions_full.csv", "Current Season Outlook tab", "src/27_generate_current_predictions.py"),
    ("data/processed/kharif_2026_risk_flags.csv", "Kharif Early Warning tab", "src/32_kharif_early_warning.py"),
    ("outputs/metrics/district_watch_feed.csv", "District Watch homepage", "src/28_district_watch.py"),
    ("outputs/metrics/test_predictions_FINAL.csv", "Historical Results tab", "src/20_finalize_tuned_model.py"),
    ("models/xgb_tuned.pkl", "all predictions", "src/20_finalize_tuned_model.py"),
    ("models/xgb_lower.pkl", "prediction interval (lower)", "src/20_finalize_tuned_model.py"),
    ("models/xgb_upper.pkl", "prediction interval (upper)", "src/20_finalize_tuned_model.py"),
]
PACKAGES = ["pandas", "numpy", "xgboost", "shap", "streamlit", "fastapi", "uvicorn", "tavily"]
STALE_DAYS = 7

problems = 0


def say(mark, text):
    print(f"  [{mark}] {text}")


print("FILES")
for path, needed_for, regen in REQUIRED:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        say("ok", path)
    else:
        problems += 1
        say("MISSING", f"{path}  -> needed for {needed_for}; regenerate with: python {regen}")

print("\nDATA FRESHNESS")
live = Path("outputs/metrics/current_predictions_2026.csv")
if live.exists():
    try:
        import pandas as pd

        as_of = pd.read_csv(live, usecols=["weather_data_as_of"])["weather_data_as_of"].iloc[0]
        age = (time.time() - pd.Timestamp(as_of).timestamp()) / 86400
        flag = "ok" if age <= STALE_DAYS else "STALE"
        say(flag, f"weather data as of {as_of} ({age:.0f} days old)")
        if age > STALE_DAYS:
            say("..", "refresh before the demo: python src/00_run_full_refresh.py")
    except Exception as exc:
        say("??", f"could not read data age: {exc}")
else:
    say("??", "outputs/metrics/current_predictions_2026.csv not found")

print("\nMODEL SCHEMA")
features_file = Path("data/processed/feature_list_v2.txt")
if features_file.exists() and importlib.util.find_spec("xgboost"):
    n_expected = len([ln for ln in features_file.read_text().splitlines() if ln.strip()])
    for name in ("xgb_tuned", "xgb_lower", "xgb_upper"):
        pkl = Path(f"models/{name}.pkl")
        if not pkl.exists():
            continue
        try:
            with open(pkl, "rb") as f:
                n_model = pickle.load(f).n_features_in_
            if n_model == n_expected:
                say("ok", f"{name}: {n_model} features")
            else:
                problems += 1
                say("MISMATCH", f"{name} has {n_model} features, schema has {n_expected}")
        except Exception as exc:
            problems += 1
            say("ERROR", f"{name} would not load: {exc}")
else:
    say("..", "skipped (needs xgboost and the feature list)")

print("\nPACKAGES")
missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
say("ok" if not missing else "MISSING", "all installed" if not missing else ", ".join(missing))
if missing:
    say("..", "install with: pip install -r requirements.txt")
    problems += 1 if any(m in missing for m in ("pandas", "xgboost", "streamlit")) else 0

print("\nOPTIONAL")
tavily = bool(os.environ.get("TAVILY_API_KEY")) or Path("data/secrets/tavily_key.txt").exists()
say("ok" if tavily else "..", "Tavily key found (live web context on)" if tavily else
    "no Tavily key: the live-context panel will show a plain 'unavailable' card, nothing breaks")
db = bool(os.environ.get("DATABASE_URL_READER"))
say("ok" if db else "..", "DATABASE_URL_READER set (/api/v1 can serve)" if db else
    "no DATABASE_URL_READER: use the legacy API (uvicorn api.naip_api:app) for the demo")

print()
if problems:
    print(f"NOT READY: {problems} problem(s) above.")
    sys.exit(1)
print("READY: dashboard (streamlit run dashboard/app.py) and API can start.")