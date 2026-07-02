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

# Surface water (Tier 2) - same graceful fallback pattern as NDVI.
sw_path = 'data/processed/surface_water_seasonal.csv'
if os.path.exists(sw_path):
    sw = pd.read_csv(sw_path)
    merged = pd.merge(
        merged,
        sw[['state', 'district', 'year', 'water_pct_kharif', 'water_pct_rabi']],
        on=['state', 'district', 'year'],
        how='left'
    )
    print(f"\nSurface water merged. Match rate: {merged['water_pct_kharif'].notna().mean():.1%}")
    print("(Coverage: 1997-2021 only - JRC dataset has no 2022+ data yet)")
else:
    print(f"\n[INFO] {sw_path} not found - proceeding WITHOUT surface water features.")
    print("Run src/36_fetch_surface_water.py first to include them.")

# Groundwater (Tier 2) - Kuruva et al. (2025), Nature Scientific Data
gw_path = 'data/processed/groundwater_district.csv'
if os.path.exists(gw_path):
    gw = pd.read_csv(gw_path)
    merged = pd.merge(
        merged,
        gw[['state', 'district', 'year',
            'gw_depth_premonsoon_m', 'gw_depth_postmonsoon_m',
            'n_wells_premonsoon']],
        on=['state', 'district', 'year'],
        how='left'
    )
    match = merged['gw_depth_premonsoon_m'].notna().mean()
    print(f"\nGroundwater merged. Match rate: {match:.1%}")
    print("(Coverage: 2000-2022, Kuruva et al. 2025, DOI:10.1038/s41597-025-05899-5)")
    print("(Years 1997-1999 will show as NaN - pre-dataset, expected)")
else:
    print(f"\n[INFO] {gw_path} not found - proceeding WITHOUT groundwater features.")
    print("Run src/37_load_groundwater.py first to include them.")

merged.to_csv('data/processed/master_dataset_v5_full_weather.csv', index=False)
print(f"\nSaved to data/processed/master_dataset_v5_full_weather.csv")
print(f"Shape: {merged.shape}")