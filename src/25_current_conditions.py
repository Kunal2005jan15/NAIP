# =============================================================
# NAIP Project - Step 25: Current Conditions Prediction
# =============================================================
# This is the REAL "live prediction" capability:
#   - Uses ACTUAL recorded weather from 2026 (fetched via NASA
#     POWER, not user-adjusted sliders)
#   - Combines it with the most recent known yield baseline
#     (2019 — the latest year government yield stats cover)
#   - Produces a genuine forward-looking estimate, with the
#     data-lag limitation stated explicitly, not hidden
#
# This is DIFFERENT from the Scenario Explorer (manual what-if
# sliders on historical years) — this uses real current data.
# =============================================================

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features.weather import season_block_features

print("Loading current weather (2020-2026, fill-value corrected) and historical model data...")

current_weather = pd.read_csv('data/raw/weather_current_2020_2026_CLEAN.csv')
current_weather['date'] = pd.to_datetime(current_weather['date'])
current_weather['month'] = current_weather['date'].dt.month
current_weather['year'] = current_weather['date'].dt.year

history = pd.read_csv('data/processed/model_ready_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

# ---------------------------------------------------------------
# AGGREGATE 2026 WEATHER (Rabi season: Nov 2025 - Mar 2026)
# ---------------------------------------------------------------
# We are mid-2026 (per project date). The most recently COMPLETED
# season is Rabi 2025-26 (wheat) which ran Nov 2025 - Mar/Apr 2026.
# Kharif 2026 (rice, Jun-Oct) is only just starting - too early
# to have meaningful season-total rainfall yet. We document this.

print("\nAggregating most recent COMPLETED season (Rabi 2025-26) per district...")




def get_latest_seasonal_weather(df):
    results = []
    for (state, district), group in df.groupby(['state', 'district']):
        # Rabi 2025-26: Nov+Dec 2025, Jan+Feb+Mar 2026
        rabi_now = group[
            ((group['year'] == 2025) & (group['month'].isin([11, 12]))) |
            ((group['year'] == 2026) & (group['month'].isin([1, 2, 3])))
        ]
        # Kharif 2026 so far (Jun 2026 onward) - partial season, informational only
        kharif_partial = group[(group['year'] == 2026) & (group['month'] >= 6)]

        row = {
            'state': state,
            'district': district,
            'current_rabi_rainfall': rabi_now['rainfall_mm'].sum(min_count=1),
            'current_rabi_temp_avg': rabi_now['temp_avg_c'].mean(),
            'current_rabi_temp_max': rabi_now['temp_max_c'].mean(),
            'current_kharif_partial_rainfall': kharif_partial['rainfall_mm'].sum(min_count=1),
            'current_kharif_partial_days': kharif_partial['rainfall_mm'].notna().sum(),
            'data_as_of': df['date'].max().strftime('%Y-%m-%d'),
        }
        if len(rabi_now) > 0:
            row.update(season_block_features(rabi_now, 'rabi', prefix='current_').to_dict())
        else:
            row.update({'current_gdd_rabi': np.nan, 'current_max_dry_streak_rabi': np.nan,
                        'current_rainfall_cv_rabi': np.nan, 'current_water_balance_rabi': np.nan})
        results.append(row)
    return pd.DataFrame(results)

current_seasonal = get_latest_seasonal_weather(current_weather)
print(f"Current seasonal data: {current_seasonal.shape}")
print(current_seasonal.head())

# ---------------------------------------------------------------
# COMPUTE CURRENT RAINFALL ANOMALY (vs. historical training baseline)
# ---------------------------------------------------------------

train_baseline = pd.read_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv')
rabi_baseline = train_baseline.groupby('district')['nasa_rainfall_rabi'].agg(
    hist_mean='mean', hist_std='std'
).reset_index()

current_seasonal = current_seasonal.merge(rabi_baseline, on='district', how='left')
current_seasonal['current_rai'] = (
    (current_seasonal['current_rabi_rainfall'] - current_seasonal['hist_mean']) /
    current_seasonal['hist_std']
)
current_seasonal['current_drought_flag'] = (current_seasonal['current_rai'] < -1.0).astype(int)
current_seasonal['current_flood_flag'] = (current_seasonal['current_rai'] > 1.0).astype(int)

print(f"\nCurrent (2025-26 Rabi season) drought districts: {current_seasonal['current_drought_flag'].sum()}")
print(f"Current (2025-26 Rabi season) flood districts:   {current_seasonal['current_flood_flag'].sum()}")

current_seasonal.to_csv('data/processed/current_conditions_2026.csv', index=False)
print("\nSaved to data/processed/current_conditions_2026.csv")

# ---------------------------------------------------------------
# BUILD CURRENT-CONDITIONS PREDICTION ROWS
# ---------------------------------------------------------------
# For each district, take its most recent known feature row
# (2019 - last year with real yield data) and REPLACE the
# weather features with the REAL current 2026 values.
# Lag features (last known yields) stay as recorded - this is
# the honest, stated limitation.

print("\n" + "="*60)
print("Building current-conditions prediction inputs...")
print("="*60)

latest_known = history.sort_values('year').groupby(['state', 'district', 'crop']).tail(1).copy()
print(f"Latest known data rows (1 per district-crop): {len(latest_known)}")
print(f"Year of latest known yield data: {latest_known['year'].max()} (most recent published)")

merged = latest_known.merge(
    current_seasonal[['state', 'district', 'current_rabi_rainfall', 'current_rabi_temp_avg',
                       'current_rabi_temp_max', 'current_rai', 'current_drought_flag',
                       'current_flood_flag', 'data_as_of',
                       'current_gdd_rabi', 'current_max_dry_streak_rabi',
                       'current_rainfall_cv_rabi', 'current_water_balance_rabi']],
    on=['state', 'district'], how='left'
)

# NDVI (Tier 2) - optional, graceful fallback if script 35 hasn't
# been run yet. Without this merge, ndvi_rabi stays at whatever
# value was last known for the district (likely NaN for live rows) -
# XGBoost handles the missing value natively, but won't get the
# real live signal until script 35 is run.
import os
current_ndvi_path = 'data/processed/current_ndvi_2026.csv'
if os.path.exists(current_ndvi_path):
    current_ndvi = pd.read_csv(current_ndvi_path)
    merged = merged.merge(current_ndvi, on=['state', 'district'], how='left')
    print(f"Live NDVI merged. Match rate: {merged['current_ndvi_rabi'].notna().mean():.1%}")
else:
    print(f"[INFO] {current_ndvi_path} not found - proceeding without live NDVI.")
    print("Run src/35_fetch_ndvi_current.py first to include it.")
    merged['current_ndvi_rabi'] = np.nan

# For Wheat (Rabi crop), override rabi weather features with REAL current data
wheat_mask = merged['crop'] == 'Wheat'
merged.loc[wheat_mask, 'nasa_rainfall_rabi'] = merged.loc[wheat_mask, 'current_rabi_rainfall']
merged.loc[wheat_mask, 'nasa_temp_avg_rabi'] = merged.loc[wheat_mask, 'current_rabi_temp_avg']
merged.loc[wheat_mask, 'nasa_temp_max_rabi'] = merged.loc[wheat_mask, 'current_rabi_temp_max']
merged.loc[wheat_mask, 'rainfall_anomaly_index'] = merged.loc[wheat_mask, 'current_rai']
merged.loc[wheat_mask, 'drought_flag'] = merged.loc[wheat_mask, 'current_drought_flag']
merged.loc[wheat_mask, 'flood_flag'] = merged.loc[wheat_mask, 'current_flood_flag']
merged.loc[wheat_mask, 'gdd_rabi'] = merged.loc[wheat_mask, 'current_gdd_rabi']
merged.loc[wheat_mask, 'max_dry_streak_rabi'] = merged.loc[wheat_mask, 'current_max_dry_streak_rabi']
merged.loc[wheat_mask, 'rainfall_cv_rabi'] = merged.loc[wheat_mask, 'current_rainfall_cv_rabi']
merged.loc[wheat_mask, 'water_balance_rabi'] = merged.loc[wheat_mask, 'current_water_balance_rabi']
merged.loc[wheat_mask, 'ndvi_rabi'] = merged.loc[wheat_mask, 'current_ndvi_rabi']

merged['baseline_yield_year'] = latest_known['year'].max()
merged['weather_data_as_of'] = merged['data_as_of']
merged['prediction_basis'] = 'current_2026_weather_with_2019_yield_baseline'

merged.to_csv('data/processed/current_prediction_inputs.csv', index=False)
print(f"\nSaved {len(merged)} current-conditions prediction inputs")
print("to data/processed/current_prediction_inputs.csv")

print("\n" + "="*60)
print("HONEST LIMITATION STATEMENT (for dashboard + paper)")
print("="*60)
print(f"""
This prediction mode uses REAL current weather (NASA POWER,
through {current_weather['date'].max().strftime('%Y-%m-%d')}) combined with each district's
most recently PUBLISHED yield data (year {latest_known['year'].max()}).

Government district-level yield statistics are published with a
1-2 year lag; no actual 2020-2026 yield ground truth exists yet.
This means:
  - Current weather signal: GENUINELY LIVE
  - Yield trend signal (lag_yield_1, etc.): LAST KNOWN, not live
This is a real-world data constraint, not a limitation specific
to this system - government and most published systems share it.
""")

print("✓ Current-conditions prediction pipeline built.")