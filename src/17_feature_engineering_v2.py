# =============================================================
# NAIP Project - Step 17: Feature Engineering v2 (Full Weather)
# =============================================================
import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv('data/processed/master_dataset_v5_full_weather.csv')
print(f"Loaded: {df.shape}")

df = df.sort_values(['state', 'district', 'crop', 'year']).reset_index(drop=True)

# Lag features
df['lag_yield_1'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(1)
df['lag_yield_2'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(2)
df['lag_yield_3'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(3)

df['rolling_yield_3yr'] = df.groupby(['district', 'crop'])['yield_kg_ha'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=2).mean()
)

df['yield_trend'] = df['lag_yield_1'] - df['lag_yield_3']

state_crop_year_avg = df.groupby(['state', 'crop', 'year'])['yield_kg_ha'].transform('mean')
df['yield_gap_vs_state'] = df['lag_yield_1'] - state_crop_year_avg.shift(1)

BASE_YEAR = 1997
df['time_trend'] = df['year'] - BASE_YEAR

season_map = {'Rabi': 0, 'Kharif': 1}
df['season_encoded'] = df['season'].map(season_map)

df['crop_encoded'] = (df['crop'] == 'Rice').astype(int)

state_map = {'Punjab': 0, 'Haryana': 1, 'Uttar Pradesh': 2}
df['state_encoded'] = df['state'].map(state_map)

df['log_area'] = np.log1p(df['area_ha'])

# Fertilizer (still from Mendeley - keep as secondary feature, 64% coverage)
df['fertilizer_per_ha'] = df['fertilizer_tons'] / (df['area_ha'] + 1)
df['nitrogen_per_ha']   = df['nitrogen_tons']   / (df['area_ha'] + 1)
df['irrigation_pct'] = (
    (df['irrigated_area_1000ha'] * 1000) / (df['area_ha'] + 1) * 100
).clip(0, 100)

# UPDATED FEATURE SET - using NASA weather (100% coverage) as primary
FEATURES = [
    'state_encoded', 'crop_encoded', 'season_encoded', 'time_trend',
    'lag_yield_1', 'lag_yield_2', 'lag_yield_3',
    'rolling_yield_3yr', 'yield_trend', 'yield_gap_vs_state',
    'log_area',
    'fertilizer_per_ha', 'nitrogen_per_ha', 'irrigation_pct',
    # NASA weather - now 100% complete, replaces weak Mendeley climate
    'nasa_rainfall_kharif', 'nasa_rainfall_rabi', 'nasa_rainfall_annual',
    'nasa_temp_avg_kharif', 'nasa_temp_max_kharif',
    'nasa_temp_avg_rabi', 'nasa_temp_max_rabi',
    'nasa_humidity_kharif', 'nasa_solar_annual',
    'heat_stress_days', 'frost_risk_days',
    'rainfall_anomaly_index', 'drought_flag', 'flood_flag',
    # NEW (Tier 1 accuracy work, 2026-06-28): richer in-season
    # agronomic signal, computed from data already on disk -
    # aimed at giving the model genuine weather-driven explanatory
    # power instead of leaning almost entirely on lag/trend.
    'gdd_kharif', 'max_dry_streak_kharif', 'rainfall_cv_kharif', 'water_balance_kharif',
    'gdd_rabi', 'max_dry_streak_rabi', 'rainfall_cv_rabi', 'water_balance_rabi',
]

# NDVI (Tier 2, satellite vegetation index) - only added to the
# feature list if script 34/16 actually produced it, so this script
# still works for anyone who hasn't set up Earth Engine yet.
if 'ndvi_kharif' in df.columns and 'ndvi_rabi' in df.columns:
    FEATURES += ['ndvi_kharif', 'ndvi_rabi']
    print("NDVI features detected - added to feature list.")
else:
    print("NDVI features not found in master dataset - proceeding without them.")

if 'water_pct_kharif' in df.columns and 'water_pct_rabi' in df.columns:
    FEATURES += ['water_pct_kharif', 'water_pct_rabi']
    print("Surface water features detected - added to feature list.")
else:
    print("Surface water features not found in master dataset - proceeding without them.")

if 'gw_depth_premonsoon_m' in df.columns and 'gw_depth_postmonsoon_m' in df.columns:
    gw_completeness = df['gw_depth_premonsoon_m'].notna().mean()
    if gw_completeness >= 0.70:
        FEATURES += ['gw_depth_premonsoon_m', 'gw_depth_postmonsoon_m']
        print(f"Groundwater depth features detected ({gw_completeness:.1%} complete) - added to feature list.")
    else:
        print(f"[INFO] Groundwater depth features detected but only {gw_completeness:.1%} complete.")
        print("       Threshold for inclusion as model feature: 70%. NOT added to FEATURES.")
        print("       Data remains in master dataset for analysis (see src/38_groundwater_analysis.py).")
else:
    print("Groundwater depth features not found in master dataset - proceeding without them.")

TARGET = 'yield_kg_ha'

model_df = df.dropna(subset=['lag_yield_1']).copy()
print(f"\nRows after dropping missing lags: {len(model_df)}")

print(f"\nFEATURE COMPLETENESS:")
for feat in FEATURES:
    pct = model_df[feat].notna().mean()
    status = "OK" if pct > 0.9 else "WARN" if pct > 0.5 else "FAIL"
    print(f"  [{status}] {feat}: {pct:.1%}")

TRAIN_END = 2015
VALID_END = 2017

train = model_df[model_df['year'] <= TRAIN_END]
valid = model_df[(model_df['year'] > TRAIN_END) & (model_df['year'] <= VALID_END)]
test  = model_df[model_df['year'] > VALID_END]

print(f"\nTRAIN: {train['year'].min()}-{train['year'].max()} | {len(train)} rows")
print(f"VALID: {valid['year'].min()}-{valid['year'].max()} | {len(valid)} rows")
print(f"TEST:  {test['year'].min()}-{test['year'].max()}  | {len(test)} rows")

model_df.to_csv('data/processed/model_ready_v2.csv', index=False)
train.to_csv('data/processed/train_v2.csv', index=False)
valid.to_csv('data/processed/valid_v2.csv', index=False)
test.to_csv('data/processed/test_v2.csv', index=False)

print(f"\nSaved train_v2.csv, valid_v2.csv, test_v2.csv")

with open('data/processed/feature_list_v2.txt', 'w') as f:
    f.write('\n'.join(FEATURES))
print(f"Saved feature list to data/processed/feature_list_v2.txt")