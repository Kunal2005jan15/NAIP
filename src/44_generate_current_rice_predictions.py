# =============================================================
# ANNA - Step 44: Current-Season RICE (Kharif 2026) Predictions
# =============================================================
# Mirror of scripts 25 + 27, but for Rice. Kharif 2026 (Jun-Oct)
# has now completed, so Rice can get the same treatment Wheat got:
#   - REAL Kharif 2026 weather (NASA POWER) replaces the Kharif
#     weather features
#   - REAL Kharif NDVI (script 35b) replaces ndvi_kharif
#   - Lag-yield features stay at each district's last published
#     year (the honest, stated 1-2 year government data lag)
#   - Quantile intervals + SHAP are added by the dashboard / here
#
# Run order (from project root):
#   python src/22_fetch_current_weather.py
#   python src/26_fix_fill_values.py
#   python src/35b_fetch_ndvi_current_kharif.py
#   python src/44_generate_current_rice_predictions.py
#
# v3 - RAINFALL SCALE CORRECTION (2026-10-27):
#   Validation against IMD (scripts 45/46) showed the live NASA POWER Kharif
#   rainfall (2020+) runs far above the 1996-2019 training baseline in
#   Punjab/Haryana (+35..+65%), which made every district look wet and the
#   model inputs fall outside the range it was trained on. Fix: each district's
#   2026 Kharif rainfall is expressed relative to that district's OWN
#   2020-2025 live history (same data source, so the offset cancels) and then
#   mapped onto the training scale:
#       rain_2026_on_training_scale = baseline_mean * (live_2026 / live_mean_2020_25)
#   Done by scaling daily rainfall, so dry streak / water balance follow.
#
# Output (Wheat files are NOT touched):
#   data/processed/current_rice_predictions_full.csv
#   outputs/metrics/current_rice_predictions_2026.csv
# =============================================================

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.features.weather import season_block_features

KHARIF_MONTHS = [6, 7, 8, 9, 10]      # same window as training (script 15)
KHARIF_YEAR = 2026
MIN_DAYS_OBSERVED = 140               # of 153; below this the season is "partial"

print("Loading Kharif 2026 weather, history, model...")
w = pd.read_csv('data/raw/weather_current_2020_2026_CLEAN.csv')
w['date'] = pd.to_datetime(w['date'])
w['month'] = w['date'].dt.month
w['year'] = w['date'].dt.year

history = pd.read_csv('data/processed/model_ready_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]
with open('models/xgb_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

data_as_of = w['date'].max().strftime('%Y-%m-%d')
print(f"Weather available through {data_as_of}")

# ---------------------------------------------------------------
# 0. Put live rainfall on the training scale (see v3 note above)
# ---------------------------------------------------------------
base = pd.read_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv')
hist_mean_k = base.groupby(['state', 'district'])['nasa_rainfall_kharif'].mean()

k_hist = w[w['month'].isin(KHARIF_MONTHS) & w['year'].between(2020, 2025)]
live_years = (k_hist.groupby(['state', 'district', 'year'])['rainfall_mm']
              .agg(total='sum', days='count').reset_index())
live_years = live_years[live_years['days'] >= MIN_DAYS_OBSERVED]      # complete seasons only
LIVE_MEAN = live_years.groupby(['state', 'district'])['total'].mean()
n_years = live_years.groupby(['state', 'district'])['total'].count()
LIVE_MEAN = LIVE_MEAN[n_years >= 4]                                  # need >= 4 complete years

scale_df = pd.concat([hist_mean_k.rename('hist_mean'), LIVE_MEAN.rename('live_mean')], axis=1).dropna()
scale_df['rain_scale'] = scale_df['hist_mean'] / scale_df['live_mean']
RAIN_SCALE = scale_df['rain_scale'].to_dict()
LIVE_MEAN_D = scale_df['live_mean'].to_dict()
print(f"Rainfall scale factors for {len(scale_df)} districts "
      f"(min {scale_df['rain_scale'].min():.2f}, median {scale_df['rain_scale'].median():.2f}, "
      f"max {scale_df['rain_scale'].max():.2f})")

# Informational: is temperature on the same footing? (should be ~0 C apart)
if 'nasa_temp_avg_kharif' in base.columns:
    t_live = k_hist.groupby('state')['temp_avg_c'].mean()
    t_base = base.groupby('state')['nasa_temp_avg_kharif'].mean()
    tc = pd.DataFrame({'live_2020_25_C': t_live.round(1), 'baseline_C': t_base.round(1)})
    tc['diff_C'] = (tc['live_2020_25_C'] - tc['baseline_C']).round(1)
    print("Kharif mean temperature, live vs baseline:")
    print(tc.to_string())

# ---------------------------------------------------------------
# 1. Kharif 2026 season aggregates per district
# ---------------------------------------------------------------
rows = []
for (state, district), g in w.groupby(['state', 'district']):
    k = g[(g['year'] == KHARIF_YEAR) & (g['month'].isin(KHARIF_MONTHS))].copy()
    days_obs = int(k['rainfall_mm'].notna().sum())
    raw_total = k['rainfall_mm'].sum(min_count=1)
    sc = RAIN_SCALE.get((state, district), np.nan)
    live_mean = LIVE_MEAN_D.get((state, district), np.nan)
    k['rainfall_mm'] = k['rainfall_mm'] * sc          # live scale -> training scale
    row = {
        'state': state, 'district': district,
        'kharif_days_observed': days_obs,
        'live_rain_2026_raw': raw_total,
        'rain_scale': sc,
        'rain_departure_pct': (raw_total / live_mean - 1) * 100 if live_mean == live_mean else np.nan,
        'cur_rain_kharif': k['rainfall_mm'].sum(min_count=1),
        'cur_temp_avg_kharif': k['temp_avg_c'].mean(),
        'cur_temp_max_kharif': k['temp_max_c'].mean(),
        'cur_humidity_kharif': k['humidity_pct'].mean(),
    }
    if days_obs > 0 and sc == sc:
        row.update(season_block_features(k, 'kharif', prefix='cur_').to_dict())
    rows.append(row)
cur = pd.DataFrame(rows)

short = (cur['kharif_days_observed'] < MIN_DAYS_OBSERVED).sum()
print(f"Districts with < {MIN_DAYS_OBSERVED} observed Kharif days: {short}")
print(f"Observed days: min {cur['kharif_days_observed'].min()}, median {cur['kharif_days_observed'].median():.0f}")

# ---------------------------------------------------------------
# 2. Kharif rainfall anomaly vs each district's own history
# ---------------------------------------------------------------
base_stats = base.groupby(['state', 'district'])['nasa_rainfall_kharif'].agg(
    hist_mean='mean', hist_std='std').reset_index()
cur = cur.merge(base_stats, on=['state', 'district'], how='left')

# Second pass: state-name spellings can differ between files (e.g. 'Uttar Pradesh'
# vs 'UP'). Same district-only fallback script 25 uses.
miss = cur['hist_mean'].isna()
if miss.any():
    by_district = base.groupby('district')['nasa_rainfall_kharif'].agg(
        hist_mean_d='mean', hist_std_d='std').reset_index()
    cur = cur.merge(by_district, on='district', how='left')
    cur['hist_mean'] = cur['hist_mean'].fillna(cur['hist_mean_d'])
    cur['hist_std'] = cur['hist_std'].fillna(cur['hist_std_d'])
    cur = cur.drop(columns=['hist_mean_d', 'hist_std_d'])
    print(f"Baseline rainfall missing for {miss.sum()} districts; "
          f"{cur['hist_mean'].isna().sum()} still missing after district-only fallback")

cur['cur_rai'] = (cur['cur_rain_kharif'] - cur['hist_mean']) / cur['hist_std']

# NEVER let a district keep a stale (2019) drought/flood flag just because its live
# anomaly could not be computed - drop it instead and say so loudly.
n_bad = cur['cur_rai'].isna().sum()
if n_bad:
    print(f"[WARN] Dropping {n_bad} districts with no computable Kharif anomaly:")
    print(cur.loc[cur['cur_rai'].isna(), ['state', 'district']].head(15).to_string(index=False))
    if n_bad > 0.1 * len(cur):
        raise SystemExit("More than 10% of districts have no baseline - fix the state/district "
                         "names in nasa_seasonal_ALL_DISTRICTS.csv before continuing.")
    cur = cur.dropna(subset=['cur_rai']).copy()
cur['cur_drought'] = (cur['cur_rai'] < -1.0).astype(int)
cur['cur_flood'] = (cur['cur_rai'] > 1.0).astype(int)
print(f"Kharif 2026 drought-flagged: {cur['cur_drought'].sum()} | flood-flagged: {cur['cur_flood'].sum()}")

# ---------------------------------------------------------------
# 3. Build Rice prediction rows (latest known row + live overrides)
# ---------------------------------------------------------------
latest = (history.sort_values('year')
          .groupby(['state', 'district', 'crop']).tail(1).copy())
rice = latest[latest['crop'] == 'Rice'].copy()
print(f"Rice districts with a baseline row: {len(rice)}")

rice = rice.merge(cur, on=['state', 'district'], how='left')

ndvi_path = 'data/processed/current_ndvi_kharif_2026.csv'
if os.path.exists(ndvi_path):
    nd = pd.read_csv(ndvi_path)
    rice = rice.merge(nd, on=['state', 'district'], how='left')
    print(f"Live Kharif NDVI match rate: {rice['current_ndvi_kharif'].notna().mean():.1%}")
else:
    print(f"[WARN] {ndvi_path} not found - run src/35b_fetch_ndvi_current_kharif.py. "
          "ndvi_kharif will stay at its last known value.")
    rice['current_ndvi_kharif'] = np.nan

overrides = {
    'nasa_rainfall_kharif':   'cur_rain_kharif',
    'nasa_temp_avg_kharif':   'cur_temp_avg_kharif',
    'nasa_temp_max_kharif':   'cur_temp_max_kharif',
    'nasa_humidity_kharif':   'cur_humidity_kharif',
    'gdd_kharif':             'cur_gdd_kharif',
    'max_dry_streak_kharif':  'cur_max_dry_streak_kharif',
    'rainfall_cv_kharif':     'cur_rainfall_cv_kharif',
    'water_balance_kharif':   'cur_water_balance_kharif',
    'rainfall_anomaly_index': 'cur_rai',
    'drought_flag':           'cur_drought',
    'flood_flag':             'cur_flood',
}
for feat, src in overrides.items():
    ok = rice[src].notna()
    rice.loc[ok, feat] = rice.loc[ok, src]
if os.path.exists(ndvi_path):
    ok = rice['current_ndvi_kharif'].notna()
    rice.loc[ok, 'ndvi_kharif'] = rice.loc[ok, 'current_ndvi_kharif']

rice = rice.dropna(subset=['cur_rain_kharif']).copy()
print(f"Rice rows with live Kharif weather: {len(rice)}")

rice['baseline_yield_year'] = int(latest['year'].max())
rice['weather_data_as_of'] = data_as_of

# ---------------------------------------------------------------
# 4. Predict + SHAP
# ---------------------------------------------------------------
missing = [f for f in FEATURES if f not in rice.columns]
if missing:
    raise SystemExit(f"Missing model features: {missing}")

X = rice[FEATURES].astype(float)
rice['current_pred_yield'] = model.predict(X)
rice['change_vs_2019'] = rice['current_pred_yield'] - rice['yield_kg_ha']

explainer = shap.TreeExplainer(model)
sv = pd.DataFrame(explainer.shap_values(X), columns=[f'shap_{c}' for c in FEATURES])
rice = pd.concat([rice.reset_index(drop=True), sv], axis=1)

# ---------------------------------------------------------------
# 5. Sanity gate - a wildly off mean means a feature problem
# ---------------------------------------------------------------
hist_mean = history[(history['crop'] == 'Rice') & (history['year'] >= history['year'].max() - 4)]['yield_kg_ha'].mean()
pred_mean = rice['current_pred_yield'].mean()
dev = (pred_mean - hist_mean) / hist_mean * 100
print(f"\nMean predicted Rice yield: {pred_mean:,.0f} kg/ha")
print(f"Mean actual Rice yield, last 5 published years: {hist_mean:,.0f} kg/ha ({dev:+.1f}% difference)")
if abs(dev) > 25:
    print("[WARN] Predicted mean is >25% away from recent history - inspect features before presenting.")
print(f"Higher than baseline: {(rice['change_vs_2019'] > 0).sum()} | lower: {(rice['change_vs_2019'] < 0).sum()}")

# ---------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------
os.makedirs('outputs/metrics', exist_ok=True)
summary_cols = ['state', 'district', 'crop', 'lat', 'lon', 'yield_kg_ha',
                'current_pred_yield', 'change_vs_2019', 'drought_flag', 'flood_flag',
                'rainfall_anomaly_index', 'rain_departure_pct', 'baseline_yield_year',
                'weather_data_as_of', 'kharif_days_observed']
rice[[c for c in summary_cols if c in rice.columns]].to_csv(
    'outputs/metrics/current_rice_predictions_2026.csv', index=False)
rice.to_csv('data/processed/current_rice_predictions_full.csv', index=False)

print("\nSaved outputs/metrics/current_rice_predictions_2026.csv")
print("Saved data/processed/current_rice_predictions_full.csv")
print("Done.")