# =============================================================
# NAIP Project - Step 4: Clean and Merge All Data Sources
# =============================================================
# Sources:
#   1. Kaggle CSV   - yield + area + production (2001-2020)
#   2. Mendeley XLS - yield + climate + fertilizer + irrigation (1990-2015)
#   3. NASA POWER   - daily weather we already pulled (2010-2024)
#
# Output: data/processed/master_dataset.csv
# =============================================================

import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------------------
# TARGET STATES AND CROPS
# ---------------------------------------------------------------

TARGET_STATES = ['Uttar Pradesh', 'Punjab', 'Haryana']
TARGET_CROPS  = ['Wheat', 'Rice']

# ---------------------------------------------------------------
# BLOCK 1: LOAD AND CLEAN KAGGLE DATA
# ---------------------------------------------------------------
# This gives us: State, District, Crop, Year, Season, 
#                Area(ha), Production(tons), Yield(T/ha)

print("=" * 60)
print("BLOCK 1: Loading Kaggle crop production data...")
print("=" * 60)

kaggle = pd.read_csv('data/raw/India Agriculture Crop Production.csv')

print(f"Raw shape: {kaggle.shape}")
print(f"States available: {kaggle['State'].nunique()}")
print(f"Crops available:  {kaggle['Crop'].nunique()}")
print(f"Year range:       {kaggle['Year'].unique()[:5]} ...")

# Filter to our target states
kaggle_filtered = kaggle[
    kaggle['State'].isin(TARGET_STATES) &
    kaggle['Crop'].isin(TARGET_CROPS)
].copy()

print(f"\nAfter filtering to UP/Punjab/Haryana + Wheat/Rice:")
print(f"Shape: {kaggle_filtered.shape}")

# Clean the Year column - it's in format "2001-02", extract start year
# "2001-02" -> 2001
kaggle_filtered['year'] = kaggle_filtered['Year'].str.split('-').str[0].astype(int)

# Convert yield from Tonnes/ha to kg/ha (multiply by 1000)
# This matches ICRISAT units for consistency
kaggle_filtered['yield_kg_ha'] = kaggle_filtered['Yield'] * 1000

# Rename columns to our standard schema
kaggle_clean = kaggle_filtered.rename(columns={
    'State':      'state',
    'District':   'district',
    'Crop':       'crop',
    'Season':     'season',
    'Area':       'area_ha',
    'Production': 'production_tons',
})[['state', 'district', 'crop', 'year', 'season', 
    'area_ha', 'production_tons', 'yield_kg_ha']]

# Standardize district and state names to title case
kaggle_clean['district'] = kaggle_clean['district'].str.title().str.strip()
kaggle_clean['state']    = kaggle_clean['state'].str.title().str.strip()
kaggle_clean['crop']     = kaggle_clean['crop'].str.title().str.strip()

# Drop rows where yield is missing or zero
kaggle_clean = kaggle_clean[
    kaggle_clean['yield_kg_ha'].notna() & 
    (kaggle_clean['yield_kg_ha'] > 0)
]

print(f"\nKaggle clean shape: {kaggle_clean.shape}")
print(f"Districts: {kaggle_clean['district'].nunique()}")
print(f"Year range: {kaggle_clean['year'].min()} - {kaggle_clean['year'].max()}")
print(f"\nSample:")
print(kaggle_clean.head())

# ---------------------------------------------------------------
# BLOCK 2: LOAD AND CLEAN MENDELEY (ICRISAT) DATA
# ---------------------------------------------------------------
# This gives us: District, Year, Rice Yield, Fertilizer,
#                Irrigation, Monthly Temp, Monthly Rainfall

print("\n" + "=" * 60)
print("BLOCK 2: Loading Mendeley ICRISAT data...")
print("=" * 60)

mendeley = pd.read_excel(
    'data/raw/main merge (droped _merge==2) (560 dist 1990-2015).xls'
)

print(f"Raw shape: {mendeley.shape}")
print(f"States: {mendeley['State Name'].unique()}")

# Filter to our target states
TARGET_STATES_MENDELEY = ['Uttar Pradesh', 'Punjab', 'Haryana']

mendeley_filtered = mendeley[
    mendeley['State Name'].isin(TARGET_STATES_MENDELEY)
].copy()

print(f"After state filter: {mendeley_filtered.shape}")

# Select only the columns we need
# We'll pull rice yield, fertilizer, irrigation, and seasonal climate
mendeley_cols = {
    'Dist Name':                              'district',
    'State Name':                             'state',
    'Year':                                   'year',
    'RICE YIELD (Kg per ha)':                 'rice_yield_mendeley',
    'TOTAL FERTILISER CONSUMPTION (tons)':    'fertilizer_tons',
    'GROSS IRRIGATED AREA (1000 ha)':         'irrigated_area_1000ha',
    'NITROGEN CONSUMPTION (tons)':            'nitrogen_tons',
    'PHOSPHATE CONSUMPTION (tons)':           'phosphate_tons',
    'POTASH CONSUMPTION (tons)':              'potash_tons',
    # Seasonal rainfall (most important for yield)
    'Rainy JUN-SEP PERCIPITATION (Millimeters)':  'rainfall_kharif_mm',
    'Winter JAN-FEB PERCIPITATION (Millimeters)':  'rainfall_rabi_mm',
    # Seasonal temperature
    'Rainy JUN-SEP MAXIMUM TEMPERATURE (Centigrate)':  'temp_max_kharif',
    'Summer MAR-MAY MAXIMUM TEMPERATURE (Centigrate)': 'temp_max_summer',
    'Winter JAN-FEB MAXIMUM TEMPERATURE (Centigrate)': 'temp_max_rabi',
    # Evapotranspiration (water stress indicator)
    'Rainy JUN-SEP ACTUAL EVAPOTRANSPIRATION (Millimeters)': 'evapotrans_kharif',
}

mendeley_clean = mendeley_filtered[list(mendeley_cols.keys())].rename(
    columns=mendeley_cols
).copy()

# Standardize names
mendeley_clean['district'] = mendeley_clean['district'].str.title().str.strip()
mendeley_clean['state']    = mendeley_clean['state'].str.title().str.strip()

print(f"\nMendeley clean shape: {mendeley_clean.shape}")
print(f"Districts: {mendeley_clean['district'].nunique()}")
print(f"Year range: {mendeley_clean['year'].min()} - {mendeley_clean['year'].max()}")
print(f"\nSample:")
print(mendeley_clean[['district', 'state', 'year', 
                        'rice_yield_mendeley', 
                        'fertilizer_tons',
                        'rainfall_kharif_mm']].head())

# ---------------------------------------------------------------
# BLOCK 3: MERGE KAGGLE + MENDELEY
# ---------------------------------------------------------------
# Join on district + state + year
# Kaggle gives yield for both wheat and rice
# Mendeley adds fertilizer, irrigation, climate

print("\n" + "=" * 60)
print("BLOCK 3: Merging Kaggle + Mendeley...")
print("=" * 60)

master = pd.merge(
    kaggle_clean,
    mendeley_clean,
    on=['district', 'state', 'year'],
    how='left'  # keep all kaggle rows, add mendeley where available
)

print(f"Merged shape: {master.shape}")
print(f"Mendeley match rate: {master['fertilizer_tons'].notna().mean():.1%}")

# ---------------------------------------------------------------
# BLOCK 4: ADD NASA POWER WEATHER DATA
# ---------------------------------------------------------------
# Aggregate daily weather to seasonal features
# Rabi season: November(prev year) - March (wheat)
# Kharif season: June - October (rice)

print("\n" + "=" * 60)
print("BLOCK 4: Processing NASA POWER weather data...")
print("=" * 60)

weather = pd.read_csv('data/raw/weather_raw.csv')
weather['date'] = pd.to_datetime(weather['date'])
weather['month'] = weather['date'].dt.month
weather['year']  = weather['date'].dt.year

# We only have 3 districts from NASA right now
# We'll map them to our master dataset districts
print(f"NASA districts: {weather['district'].unique()}")

# Aggregate to seasonal features per district per year
def get_seasonal_weather(df):
    """
    For each district-year, compute:
    - Kharif season (Jun-Oct): total rainfall, avg temp
    - Rabi season (Mar-May + Nov-Dec + Jan-Feb): total rainfall, avg temp
    - Annual stats
    - Heat stress days (temp_max > 35°C)
    """
    results = []
    
    for (district, year), group in df.groupby(['district', 'year']):
        # Kharif: June(6) to October(10)
        kharif = group[group['month'].between(6, 10)]
        # Rabi: November(11), December(12), January(1), February(2), March(3)
        rabi   = group[group['month'].isin([11, 12, 1, 2, 3])]
        # Full year
        annual = group
        
        row = {
            'district':              district,
            'year':                  year,
            # Kharif weather (for rice)
            'nasa_rainfall_kharif':  kharif['rainfall_mm'].sum(),
            'nasa_temp_avg_kharif':  kharif['temp_avg_c'].mean(),
            'nasa_temp_max_kharif':  kharif['temp_max_c'].mean(),
            'nasa_humidity_kharif':  kharif['humidity_pct'].mean(),
            # Rabi weather (for wheat)
            'nasa_rainfall_rabi':    rabi['rainfall_mm'].sum(),
            'nasa_temp_avg_rabi':    rabi['temp_avg_c'].mean(),
            'nasa_temp_max_rabi':    rabi['temp_max_c'].mean(),
            # Annual
            'nasa_rainfall_annual':  annual['rainfall_mm'].sum(),
            'nasa_solar_annual':     annual['solar_radiation'].mean(),
            # Heat stress: days above 35°C (critical for wheat)
            'heat_stress_days':      (annual['temp_max_c'] > 35).sum(),
            # Frost risk: days below 2°C (damages crops)
            'frost_risk_days':       (annual['temp_min_c'] < 2).sum(),
        }
        results.append(row)
    
    return pd.DataFrame(results)

nasa_seasonal = get_seasonal_weather(weather)

print(f"NASA seasonal features shape: {nasa_seasonal.shape}")
print(f"\nSample NASA seasonal features:")
print(nasa_seasonal.head())

# Save NASA seasonal separately too
nasa_seasonal.to_csv('data/processed/nasa_seasonal.csv', index=False)

# ---------------------------------------------------------------
# BLOCK 5: COMPUTE RAINFALL ANOMALY INDEX (RAI)
# ---------------------------------------------------------------
# RAI = (rainfall - mean_rainfall) / std_rainfall
# This is your novel feature — flags abnormal rainfall years
# Positive RAI = wetter than normal (flood risk)
# Negative RAI = drier than normal (drought risk)
# Reference: Pandya & Gontia (2023)

print("\n" + "=" * 60)
print("BLOCK 5: Computing Rainfall Anomaly Index (RAI)...")
print("=" * 60)

# Compute long-term mean and std per district
rainfall_stats = nasa_seasonal.groupby('district')['nasa_rainfall_annual'].agg(
    mean_rainfall='mean',
    std_rainfall='std'
).reset_index()

nasa_seasonal = nasa_seasonal.merge(rainfall_stats, on='district')

nasa_seasonal['rainfall_anomaly_index'] = (
    (nasa_seasonal['nasa_rainfall_annual'] - nasa_seasonal['mean_rainfall']) / 
    nasa_seasonal['std_rainfall']
)

# Classify anomaly
nasa_seasonal['drought_flag'] = (nasa_seasonal['rainfall_anomaly_index'] < -1.0).astype(int)
nasa_seasonal['flood_flag']   = (nasa_seasonal['rainfall_anomaly_index'] >  1.0).astype(int)

print("RAI sample:")
print(nasa_seasonal[['district', 'year', 'nasa_rainfall_annual', 
                       'rainfall_anomaly_index', 
                       'drought_flag', 'flood_flag']].head(10))

print(f"\nDrought years flagged: {nasa_seasonal['drought_flag'].sum()}")
print(f"Flood years flagged:   {nasa_seasonal['flood_flag'].sum()}")

# ---------------------------------------------------------------
# BLOCK 6: SAVE EVERYTHING
# ---------------------------------------------------------------

print("\n" + "=" * 60)
print("BLOCK 6: Saving processed files...")
print("=" * 60)

# Save master (kaggle + mendeley merged)
master.to_csv('data/processed/master_dataset.csv', index=False)
print(f"master_dataset.csv saved: {master.shape}")

# Save mendeley standalone
mendeley_clean.to_csv('data/processed/mendeley_clean.csv', index=False)
print(f"mendeley_clean.csv saved: {mendeley_clean.shape}")

# Save NASA with RAI
nasa_seasonal.to_csv('data/processed/nasa_seasonal_with_rai.csv', index=False)
print(f"nasa_seasonal_with_rai.csv saved: {nasa_seasonal.shape}")

# ---------------------------------------------------------------
# BLOCK 7: DATA QUALITY REPORT
# ---------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

print(f"\nMASTER DATASET SUMMARY:")
print(f"  Total rows:       {len(master)}")
print(f"  Districts:        {master['district'].nunique()}")
print(f"  States:           {master['state'].unique()}")
print(f"  Crops:            {master['crop'].unique()}")
print(f"  Year range:       {master['year'].min()} - {master['year'].max()}")
print(f"  Yield range:      {master['yield_kg_ha'].min():.0f} - {master['yield_kg_ha'].max():.0f} kg/ha")

print(f"\nMISSING VALUES IN MASTER:")
missing = master.isnull().sum()
missing = missing[missing > 0]
print(missing)

print(f"\nYIELD STATS BY STATE:")
print(master.groupby(['state', 'crop'])['yield_kg_ha'].agg(['mean', 'min', 'max']).round(0))

print(f"\nYIELD STATS BY YEAR (Wheat, all districts):")
wheat = master[master['crop'] == 'Wheat']
print(wheat.groupby('year')['yield_kg_ha'].mean().round(0).to_string())