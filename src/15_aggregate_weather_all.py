# =============================================================
# NAIP Project - Step 15: Aggregate Full Weather Data + RAI
# =============================================================
# Same logic as before, but now across all 118 districts
# instead of 3. This directly fixes the weak weather signal
# problem identified via SHAP analysis.
# =============================================================

import pandas as pd
import numpy as np

print("Loading full weather dataset (991,200 rows)...")
weather = pd.read_csv('data/raw/weather_all_districts.csv')
weather['date'] = pd.to_datetime(weather['date'])
weather['month'] = weather['date'].dt.month
weather['year']  = weather['date'].dt.year

print(f"Loaded: {weather.shape}")
print(f"Districts: {weather['district'].nunique()}")

# ---------------------------------------------------------------
# SEASONAL AGGREGATION (same logic as the 3-district version)
# ---------------------------------------------------------------

def get_seasonal_weather(df):
    results = []
    
    grouped = df.groupby(['state', 'district', 'year'])
    total = len(grouped)
    
    for i, ((state, district, year), group) in enumerate(grouped):
        kharif = group[group['month'].between(6, 10)]
        rabi   = group[group['month'].isin([11, 12, 1, 2, 3])]
        annual = group
        
        row = {
            'state':                 state,
            'district':              district,
            'year':                  year,
            'nasa_rainfall_kharif':  kharif['rainfall_mm'].sum(),
            'nasa_temp_avg_kharif':  kharif['temp_avg_c'].mean(),
            'nasa_temp_max_kharif':  kharif['temp_max_c'].mean(),
            'nasa_humidity_kharif':  kharif['humidity_pct'].mean(),
            'nasa_rainfall_rabi':    rabi['rainfall_mm'].sum(),
            'nasa_temp_avg_rabi':    rabi['temp_avg_c'].mean(),
            'nasa_temp_max_rabi':    rabi['temp_max_c'].mean(),
            'nasa_rainfall_annual':  annual['rainfall_mm'].sum(),
            'nasa_solar_annual':     annual['solar_radiation'].mean(),
            'heat_stress_days':      (annual['temp_max_c'] > 35).sum(),
            'frost_risk_days':       (annual['temp_min_c'] < 2).sum(),
        }
        results.append(row)
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{total} district-years...")
    
    return pd.DataFrame(results)

print("\nAggregating to seasonal features (this takes 1-2 minutes)...")
nasa_seasonal = get_seasonal_weather(weather)
print(f"\nSeasonal features shape: {nasa_seasonal.shape}")

# ---------------------------------------------------------------
# RAINFALL ANOMALY INDEX (RAI) — per district, using its OWN
# long-term mean and std, computed across all available years
# ---------------------------------------------------------------

print("\nComputing Rainfall Anomaly Index (RAI) per district...")

rainfall_stats = nasa_seasonal.groupby('district')['nasa_rainfall_annual'].agg(
    mean_rainfall='mean',
    std_rainfall='std'
).reset_index()

nasa_seasonal = nasa_seasonal.merge(rainfall_stats, on='district')

nasa_seasonal['rainfall_anomaly_index'] = (
    (nasa_seasonal['nasa_rainfall_annual'] - nasa_seasonal['mean_rainfall']) /
    nasa_seasonal['std_rainfall']
)

nasa_seasonal['drought_flag'] = (nasa_seasonal['rainfall_anomaly_index'] < -1.0).astype(int)
nasa_seasonal['flood_flag']   = (nasa_seasonal['rainfall_anomaly_index'] >  1.0).astype(int)

print(f"\nDrought district-years flagged: {nasa_seasonal['drought_flag'].sum()}")
print(f"Flood district-years flagged:   {nasa_seasonal['flood_flag'].sum()}")
print(f"Total district-years:           {len(nasa_seasonal)}")

# ---------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------

nasa_seasonal.to_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv', index=False)
print(f"\nSaved to data/processed/nasa_seasonal_ALL_DISTRICTS.csv")
print(f"Shape: {nasa_seasonal.shape}")
print(f"\nSample:")
print(nasa_seasonal[['state', 'district', 'year', 'nasa_rainfall_annual', 
                       'rainfall_anomaly_index', 'drought_flag', 'flood_flag']].head(10))