# =============================================================
# NAIP Project - Step 15: Aggregate Full Weather Data + RAI
# =============================================================
# Same logic as before, but now across all 118 districts
# instead of 3. This directly fixes the weak weather signal
# problem identified via SHAP analysis.
# =============================================================

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `pipeline` imports work

from pipeline.features.weather import season_block_features as _season_block_features

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
    """
    BUG FIX (2026-06-28): the crop-year labeling convention (from
    the raw "2001-02"-style Year column) uses the SOWING year - so
    year=2001 means a Rabi crop sown Nov 2001, harvested Mar/Apr 2002.
    The previous version of this function defined "Rabi for year=2001"
    as months [11,12,1,2,3] WITHIN calendar year 2001 only - which
    grabs Nov-Dec 2001 (correct) PLUS Jan-Mar 2001 (the tail of the
    PREVIOUS season, 2000-01) while MISSING Jan-Mar 2002 entirely -
    the actual grain-filling/harvest-period weather for the season
    being modeled, and one of the most important determinants of
    Indian wheat yield (terminal heat stress). The model was never
    seeing the right weather for the season it was predicting.

    FIX: assign each calendar day to the correct CROP YEAR before
    aggregating, instead of grouping by calendar year:
      - Kharif (Jun-Oct): unaffected, already within one calendar year
      - Rabi (Nov-Mar): Nov/Dec -> that calendar year; Jan/Feb/Mar ->
        PREVIOUS calendar year (i.e. the season that started the
        preceding November)
      - Annual: switched to the agricultural year (Jun Y - May Y+1),
        matching India's official agricultural-year convention and
        fully containing one Kharif + one Rabi season for crop-year Y,
        instead of a calendar year that incoherently straddles parts
        of two different crop seasons.

    EXTENSION (2026-06-28, Tier 1 accuracy work): added GDD, max
    dry-spell streak, rainfall concentration (CV), and a simplified
    water-balance feature per season - all from data already on disk,
    aimed at giving the model genuine in-season agronomic signal
    instead of leaning almost entirely on lag/trend features (SHAP
    showed weather at ~7% of importance before the window fix).
    """
    d = df.copy()
    d['kharif_year'] = np.where(d['month'].between(6, 10), d['year'], np.nan)
    d['rabi_year'] = np.where(
        d['month'].isin([11, 12]), d['year'],
        np.where(d['month'].isin([1, 2, 3]), d['year'] - 1, np.nan)
    )
    d['agri_year'] = np.where(d['month'] >= 6, d['year'], d['year'] - 1)

    kharif_agg = d.dropna(subset=['kharif_year']).groupby(
        ['state', 'district', 'kharif_year']
    ).agg(
        nasa_rainfall_kharif=('rainfall_mm', 'sum'),
        nasa_temp_avg_kharif=('temp_avg_c', 'mean'),
        nasa_temp_max_kharif=('temp_max_c', 'mean'),
        nasa_humidity_kharif=('humidity_pct', 'mean'),
    ).reset_index().rename(columns={'kharif_year': 'year'})
    kharif_agg['year'] = kharif_agg['year'].astype(int)

    print("  Computing richer Kharif agronomic features (GDD, dry streak, CV, water balance)...")
    kharif_extra = d.dropna(subset=['kharif_year']).groupby(
        ['state', 'district', 'kharif_year']
    ).apply(lambda b: _season_block_features(b, 'kharif')).reset_index().rename(columns={'kharif_year': 'year'})
    kharif_extra['year'] = kharif_extra['year'].astype(int)
    kharif_agg = kharif_agg.merge(kharif_extra, on=['state', 'district', 'year'], how='left')

    rabi_agg = d.dropna(subset=['rabi_year']).groupby(
        ['state', 'district', 'rabi_year']
    ).agg(
        nasa_rainfall_rabi=('rainfall_mm', 'sum'),
        nasa_temp_avg_rabi=('temp_avg_c', 'mean'),
        nasa_temp_max_rabi=('temp_max_c', 'mean'),
        rabi_days_observed=('rainfall_mm', 'size'),
    ).reset_index().rename(columns={'rabi_year': 'year'})
    rabi_agg['year'] = rabi_agg['year'].astype(int)
    # Full Nov-Mar window is 151/152 days. The most recent crop-year's
    # Rabi season needs Jan-Mar of the FOLLOWING calendar year, which
    # may not exist if the raw weather feed ends mid-season (true for
    # 2019, since this dataset ends 2019-12-31). Flag partial seasons
    # explicitly rather than silently averaging over fewer days.
    rabi_agg['rabi_season_partial'] = (rabi_agg['rabi_days_observed'] < 140).astype(int)

    print("  Computing richer Rabi agronomic features (GDD, dry streak, CV, water balance)...")
    rabi_extra = d.dropna(subset=['rabi_year']).groupby(
        ['state', 'district', 'rabi_year']
    ).apply(lambda b: _season_block_features(b, 'rabi')).reset_index().rename(columns={'rabi_year': 'year'})
    rabi_extra['year'] = rabi_extra['year'].astype(int)
    rabi_agg = rabi_agg.merge(rabi_extra, on=['state', 'district', 'year'], how='left')

    annual_agg = d.groupby(['state', 'district', 'agri_year']).agg(
        nasa_rainfall_annual=('rainfall_mm', 'sum'),
        nasa_solar_annual=('solar_radiation', 'mean'),
        heat_stress_days=('temp_max_c', lambda x: (x > 35).sum()),
        frost_risk_days=('temp_min_c', lambda x: (x < 2).sum()),
    ).reset_index().rename(columns={'agri_year': 'year'})
    annual_agg['year'] = annual_agg['year'].astype(int)

    merged = kharif_agg.merge(rabi_agg, on=['state', 'district', 'year'], how='outer')
    merged = merged.merge(annual_agg, on=['state', 'district', 'year'], how='outer')
    return merged

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