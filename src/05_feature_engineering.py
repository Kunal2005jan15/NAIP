# =============================================================
# NAIP Project - Step 5: Feature Engineering
# =============================================================
# This is where raw data becomes model-ready features.
#
# Key features we build:
# 1. Lag features     - last year's yield (most predictive single feature)
# 2. Trend features   - 3-year rolling average yield
# 3. Yield gap        - district yield vs state average (efficiency measure)
# 4. Season encoding  - Rabi=0, Kharif=1
# 5. Time trend       - captures technology/policy improvement over years
#
# CRITICAL: All lag/rolling features computed with strict
# temporal boundaries — no data leakage.
# =============================================================

import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

print("Loading master dataset...")
df = pd.read_csv('data/processed/master_dataset.csv')
print(f"Loaded: {df.shape}")

# ---------------------------------------------------------------
# STEP 1: SORT — Critical for temporal features
# ---------------------------------------------------------------
# Always sort by district + crop + year before computing lags
# If you don't sort first, lag features will be wrong

df = df.sort_values(['state', 'district', 'crop', 'year']).reset_index(drop=True)

print(f"\nData sorted by state/district/crop/year")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# ---------------------------------------------------------------
# STEP 2: LAG FEATURES
# ---------------------------------------------------------------
# lag_yield_1 = yield from 1 year ago (same district, same crop)
# lag_yield_2 = yield from 2 years ago
# lag_yield_3 = yield from 3 years ago
#
# WHY: Last year's yield is the strongest predictor of this year's yield.
# It captures soil health, farmer adaptation, technology adoption.
#
# LEAKAGE PREVENTION: groupby(['district','crop']) ensures we
# only look backward within the same district-crop combination.

print("\nComputing lag features...")

df['lag_yield_1'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(1)
df['lag_yield_2'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(2)
df['lag_yield_3'] = df.groupby(['district', 'crop'])['yield_kg_ha'].shift(3)

# ---------------------------------------------------------------
# STEP 3: ROLLING AVERAGE (Trend)
# ---------------------------------------------------------------
# rolling_yield_3yr = average yield over past 3 years
# This smooths out single-year weather shocks
# min_periods=2 means we compute it even if only 2 years available

df['rolling_yield_3yr'] = df.groupby(['district', 'crop'])['yield_kg_ha'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=2).mean()
)

# ---------------------------------------------------------------
# STEP 4: YIELD TREND (Is this district improving?)
# ---------------------------------------------------------------
# yield_trend = lag_yield_1 - lag_yield_3
# Positive = yield improving, Negative = yield declining

df['yield_trend'] = df['lag_yield_1'] - df['lag_yield_3']

# ---------------------------------------------------------------
# STEP 5: YIELD GAP (Efficiency metric)
# ---------------------------------------------------------------
# How does this district compare to its state average?
# yield_gap > 0 = above average, < 0 = below average
# This is a key feature for policy targeting

state_crop_year_avg = df.groupby(['state', 'crop', 'year'])['yield_kg_ha'].transform('mean')
df['yield_gap_vs_state'] = df['lag_yield_1'] - state_crop_year_avg.shift(1)

# ---------------------------------------------------------------
# STEP 6: TIME TREND FEATURE
# ---------------------------------------------------------------
# Captures overall productivity improvement over time
# (technology adoption, better seeds, policy changes)
# We use year - base_year so the number stays manageable

BASE_YEAR = 1997
df['time_trend'] = df['year'] - BASE_YEAR

# ---------------------------------------------------------------
# STEP 7: SEASON ENCODING
# ---------------------------------------------------------------
# Convert season names to numbers for the model

print("\nSeason distribution:")
print(df['season'].value_counts())

# Map seasons to binary
season_map = {
    'Rabi':       0,
    'Kharif':     1,
    'Whole Year': 0,  # treat as Rabi (conservative)
    'Rabi ':      0,  # handle trailing space
    'Kharif ':    1,
}
df['season_encoded'] = df['season'].map(season_map)

# If still missing, infer from crop
# Wheat = Rabi (winter), Rice = Kharif (monsoon)
df.loc[df['season_encoded'].isna() & (df['crop'] == 'Wheat'), 'season_encoded'] = 0
df.loc[df['season_encoded'].isna() & (df['crop'] == 'Rice'),  'season_encoded'] = 1

# ---------------------------------------------------------------
# STEP 8: CROP AND STATE ENCODING
# ---------------------------------------------------------------

df['crop_encoded'] = (df['crop'] == 'Rice').astype(int)  # Wheat=0, Rice=1

state_map = {
    'Punjab':        0,
    'Haryana':       1,
    'Uttar Pradesh': 2
}
df['state_encoded'] = df['state'].map(state_map)

# ---------------------------------------------------------------
# STEP 9: AREA FEATURE (normalized)
# ---------------------------------------------------------------
# Log transform area — reduces skew from large districts

df['log_area'] = np.log1p(df['area_ha'])

# ---------------------------------------------------------------
# STEP 10: FERTILIZER FEATURES (from Mendeley)
# ---------------------------------------------------------------
# Convert fertilizer to per-hectare intensity
# This is more meaningful than total tons

df['fertilizer_per_ha'] = df['fertilizer_tons'] / (df['area_ha'] + 1)
df['nitrogen_per_ha']   = df['nitrogen_tons']   / (df['area_ha'] + 1)

# ---------------------------------------------------------------
# STEP 11: IRRIGATION COVERAGE
# ---------------------------------------------------------------
df['irrigation_pct'] = (
    (df['irrigated_area_1000ha'] * 1000) / (df['area_ha'] + 1) * 100
).clip(0, 100)  # cap at 100%

# ---------------------------------------------------------------
# STEP 12: DEFINE FINAL FEATURE SET
# ---------------------------------------------------------------
# These are our model features — ordered by expected importance

FEATURES = [
    # Identity features
    'state_encoded',
    'crop_encoded',
    'season_encoded',
    'time_trend',
    
    # Lag/trend features (most important)
    'lag_yield_1',
    'lag_yield_2', 
    'lag_yield_3',
    'rolling_yield_3yr',
    'yield_trend',
    'yield_gap_vs_state',
    
    # Area
    'log_area',
    
    # Fertilizer/irrigation (from Mendeley, ~65% coverage)
    'fertilizer_per_ha',
    'nitrogen_per_ha',
    'irrigation_pct',
    
    # Climate from Mendeley
    'rainfall_kharif_mm',
    'rainfall_rabi_mm',
    'temp_max_kharif',
    'temp_max_rabi',
    'evapotrans_kharif',
]

TARGET = 'yield_kg_ha'

# ---------------------------------------------------------------
# STEP 13: PREPARE MODEL-READY DATASET
# ---------------------------------------------------------------
# Drop rows where lag_yield_1 is missing (first year per district)
# These rows have no temporal history — can't use them for training

model_df = df.dropna(subset=['lag_yield_1']).copy()

print(f"\nRows before dropping missing lags: {len(df)}")
print(f"Rows after dropping missing lags:  {len(model_df)}")

# Check feature completeness
print(f"\nFEATURE COMPLETENESS:")
for feat in FEATURES:
    if feat in model_df.columns:
        pct = model_df[feat].notna().mean()
        status = "✓" if pct > 0.8 else "⚠" if pct > 0.5 else "✗"
        print(f"  {status} {feat}: {pct:.1%} complete")
    else:
        print(f"  ✗ {feat}: NOT FOUND")

# ---------------------------------------------------------------
# STEP 14: TIME-CORRECT TRAIN/TEST SPLIT
# ---------------------------------------------------------------
# TRAIN: 1997-2015 (use for model training)
# VALID: 2016-2017 (tune hyperparameters)
# TEST:  2018-2019 (final evaluation — touch only once)
#
# This is the methodological contribution that separates
# our work from papers using random 80/20 splits

TRAIN_END = 2015
VALID_END = 2017

train = model_df[model_df['year'] <= TRAIN_END]
valid = model_df[(model_df['year'] > TRAIN_END) & (model_df['year'] <= VALID_END)]
test  = model_df[model_df['year'] > VALID_END]

print(f"\nTIME-CORRECT SPLIT:")
print(f"  TRAIN: {train['year'].min()}-{train['year'].max()} | {len(train)} rows")
print(f"  VALID: {valid['year'].min()}-{valid['year'].max()} | {len(valid)} rows")
print(f"  TEST:  {test['year'].min()}-{test['year'].max()}  | {len(test)} rows")

print(f"\n  Train yield mean: {train[TARGET].mean():.0f} kg/ha")
print(f"  Valid yield mean: {valid[TARGET].mean():.0f} kg/ha")
print(f"  Test  yield mean: {test[TARGET].mean():.0f} kg/ha")

# ---------------------------------------------------------------
# STEP 15: SAVE
# ---------------------------------------------------------------

model_df.to_csv('data/processed/model_ready.csv', index=False)
train.to_csv('data/processed/train.csv', index=False)
valid.to_csv('data/processed/valid.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)

print(f"\nSaved:")
print(f"  data/processed/model_ready.csv  ({len(model_df)} rows)")
print(f"  data/processed/train.csv        ({len(train)} rows)")
print(f"  data/processed/valid.csv        ({len(valid)} rows)")
print(f"  data/processed/test.csv         ({len(test)} rows)")

print("\n✓ Feature engineering complete. Ready for model training.")