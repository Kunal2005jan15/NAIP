# =============================================================
# NAIP Project - Step 37: Groundwater Level Loader
# Source: Kuruva et al. (2025), Nature Scientific Data
# DOI: 10.1038/s41597-025-05899-5
# Data: https://doi.org/10.6084/m9.figshare.29293877.v3
# =============================================================
# File to download from figshare:
#   CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv
# Save to: data/raw/groundwater/
#
# This is a WIDE FORMAT file (one row per well, one column per
# time period: Jan-00, May-00, Aug-00, Nov-00 ... Nov-22).
# This script:
#   1. Filters to your 3 states (Haryana, Punjab, Uttar Pradesh)
#   2. Melts the wide format to long (district-year-season)
#   3. Aggregates across wells per district (median depth, 
#      n_wells_counted for data quality transparency)
#   4. Maps to crop-year convention matching the rest of NAIP:
#      - Rabi (Wheat): uses Premonsoon (May) reading of SAME year
#        as the crop-year label (May 2001 -> rabi_year 2001)
#      - Kharif (Rice): uses Postmonsoon (Nov) reading of SAME year
#        (Nov 2001 -> kharif_year 2001)
#   5. Saves data/processed/groundwater_district.csv
#
# CITATION: Kuruva, S.K., Suryawanshi, M.R., Shakya, A. et al.
# Quality controlled, reliable groundwater level data with 
# corresponding specific yield over India. Sci Data 12, 1609 (2025).
# DOI: 10.1038/s41597-025-05899-5
# =============================================================

import pandas as pd
import numpy as np
import os

RAW_PATH = 'data/raw/groundwater/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv'
TARGET_STATES = ['Haryana', 'Punjab', 'Uttar Pradesh']

if not os.path.exists(RAW_PATH):
    print(f"File not found: {RAW_PATH}")
    print()
    print("Download instructions:")
    print("  1. Go to: https://doi.org/10.6084/m9.figshare.29293877.v3")
    print("  2. Download: CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv")
    print("  3. Save to: data/raw/groundwater/")
    print()
    print("This is a peer-reviewed, open-access dataset (Kuruva et al., 2025,")
    print("Nature Scientific Data). No login required to download from figshare.")
    exit(0)

print(f"Loading {RAW_PATH}...")
raw = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {raw.shape}")

# Normalise state column name
state_col = next((c for c in raw.columns if c.lower() == 'state'), None)
dist_col  = next((c for c in raw.columns if c.lower() == 'district'), None)
if not state_col or not dist_col:
    print(f"Could not find State/District columns. Actual columns: {raw.columns.tolist()[:20]}")
    exit(1)

# Filter to NAIP states
df = raw[raw[state_col].isin(TARGET_STATES)].copy()
print(f"After state filter: {len(df)} wells across {df[dist_col].nunique()} districts")

# Identify seasonal GWL columns (format: Mon-YY e.g. Jan-00, May-01)
import re
gwl_cols = [c for c in df.columns if re.match(r'^(Jan|May|Aug|Nov)-\d{2}$', c)]
print(f"GWL time columns found: {len(gwl_cols)} ({gwl_cols[0]} to {gwl_cols[-1]})")

# Melt to long format
id_cols = [state_col, dist_col]
melted = df[id_cols + gwl_cols].melt(
    id_vars=id_cols, value_vars=gwl_cols,
    var_name='period', value_name='gwl_m_bgl'
)
melted = melted.dropna(subset=['gwl_m_bgl'])
melted['gwl_m_bgl'] = pd.to_numeric(melted['gwl_m_bgl'], errors='coerce')
melted = melted.dropna(subset=['gwl_m_bgl'])

# Parse period -> month, year
melted['month_str'] = melted['period'].str[:3]
melted['year_2d']   = melted['period'].str[4:].astype(int)
melted['year']      = melted['year_2d'].apply(lambda y: 2000 + y)

print(f"Long format rows: {len(melted)}")
print(f"Year range: {melted['year'].min()} - {melted['year'].max()}")

# Map to crop-year convention matching NAIP's train/serve pipeline:
#   Premonsoon = May reading -> rabi year (planted Nov prior, harvested Apr-May)
#   Postmonsoon = Nov reading -> kharif year (planted June, harvested Oct-Nov)
#   Jan/Aug excluded from the primary features (used only if May/Nov unavailable)
rabi_gw  = melted[melted['month_str'] == 'May'].copy()
kharif_gw = melted[melted['month_str'] == 'Nov'].copy()

def aggregate_district_year(sub, state_col, dist_col):
    """Median depth across wells in district (robust to outlier wells)
    and well count for data quality transparency in the paper."""
    agg = sub.groupby([state_col, dist_col, 'year']).agg(
        gw_depth_median=('gwl_m_bgl', 'median'),
        gw_depth_mean=('gwl_m_bgl', 'mean'),
        n_wells=('gwl_m_bgl', 'count'),
    ).reset_index()
    agg = agg.rename(columns={state_col: 'state', dist_col: 'district'})
    return agg

rabi_agg  = aggregate_district_year(rabi_gw,  state_col, dist_col)
kharif_agg = aggregate_district_year(kharif_gw, state_col, dist_col)

# Merge into one file: one row per district-year
combined = rabi_agg.rename(columns={
    'gw_depth_median': 'gw_depth_premonsoon_m',
    'gw_depth_mean':   'gw_depth_premonsoon_mean_m',
    'n_wells':         'n_wells_premonsoon',
}).merge(
    kharif_agg.rename(columns={
        'gw_depth_median': 'gw_depth_postmonsoon_m',
        'gw_depth_mean':   'gw_depth_postmonsoon_mean_m',
        'n_wells':         'n_wells_postmonsoon',
    }),
    on=['state', 'district', 'year'], how='outer'
)

# Sanity check: depths should be 0-100m bgl in this region
implausible = combined[
    (combined['gw_depth_premonsoon_m'] > 100) |
    (combined['gw_depth_postmonsoon_m'] > 100)
]
if len(implausible) > 0:
    print(f"\n[WARN] {len(implausible)} district-years have depth >100m bgl")
    print("Check for unit conversion issues in the source file.")

combined.to_csv('data/processed/groundwater_district.csv', index=False)

print(f"\nSaved data/processed/groundwater_district.csv: {combined.shape}")
print(f"Districts covered: {combined['district'].nunique()}")
print(f"Years covered: {sorted(combined['year'].unique())}")
print(f"\nSample:")
print(combined[combined['district']=='Ambala'].head(4).to_string(index=False))
print()
print("Citation: Kuruva et al. (2025), Nature Scientific Data,")
print("DOI: 10.1038/s41597-025-05899-5")