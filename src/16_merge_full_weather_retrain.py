# =============================================================
# NAIP Project - Step 16: Merge Full Weather + Rebuild Features
# =============================================================
# Replaces weak/incomplete Mendeley weather (64% coverage)
# with full NASA POWER weather (100% coverage, all 118 districts)
# =============================================================

import pandas as pd
import numpy as np

master = pd.read_csv('data/processed/master_dataset_FINAL.csv')
nasa = pd.read_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv')

print(f"Master rows: {len(master)}")
print(f"NASA seasonal rows: {len(nasa)}")

# Merge on district + state + year
merged = pd.merge(
    master,
    nasa[['state', 'district', 'year', 
          'nasa_rainfall_kharif', 'nasa_temp_avg_kharif', 'nasa_temp_max_kharif',
          'nasa_humidity_kharif', 'nasa_rainfall_rabi', 'nasa_temp_avg_rabi',
          'nasa_temp_max_rabi', 'nasa_rainfall_annual', 'nasa_solar_annual',
          'heat_stress_days', 'frost_risk_days', 'rainfall_anomaly_index',
          'drought_flag', 'flood_flag',
          'gdd_kharif', 'max_dry_streak_kharif', 'rainfall_cv_kharif', 'water_balance_kharif',
          'gdd_rabi', 'max_dry_streak_rabi', 'rainfall_cv_rabi', 'water_balance_rabi']],
    on=['state', 'district', 'year'],
    how='left'
)

print(f"\nMerged rows: {len(merged)}")
print(f"NASA weather match rate: {merged['nasa_rainfall_annual'].notna().mean():.1%}")
print(f"(Should be ~100% now vs 64% with Mendeley)")

# NDVI (Tier 2, Google Earth Engine MODIS) - optional second merge.
# Graceful fallback: if script 34 hasn't been run yet, proceed
# without NDVI rather than hard-failing the whole pipeline.
import os
ndvi_path = 'data/processed/ndvi_seasonal.csv'
if os.path.exists(ndvi_path):
    ndvi = pd.read_csv(ndvi_path)
    merged = pd.merge(
        merged,
        ndvi[['state', 'district', 'year', 'ndvi_kharif', 'ndvi_rabi']],
        on=['state', 'district', 'year'],
        how='left'
    )
    print(f"\nNDVI merged. Match rate: {merged['ndvi_kharif'].notna().mean():.1%}")
    print("(Years 1997-1999 will show as NaN - pre-MODIS, expected)")
else:
    print(f"\n[INFO] {ndvi_path} not found - proceeding WITHOUT NDVI features.")
    print("Run src/34_fetch_ndvi_historical.py first to include them.")

merged.to_csv('data/processed/master_dataset_v5_full_weather.csv', index=False)
print(f"\nSaved to data/processed/master_dataset_v5_full_weather.csv")
print(f"Shape: {merged.shape}")