# =============================================================
# NAIP Project - Step 36: Surface Water Availability (Tier 2)
# =============================================================
# WHY: irrigation_pct sits at only 63% completeness and near-zero
# SHAP importance - a weak, mostly-missing proxy for water
# availability. Real surface water extent (tanks, reservoirs,
# rivers, seasonal ponds) per district is a much more direct,
# always-available signal, and is the kind of feature most
# yield-prediction papers in this space don't include.
#
# Source: JRC Global Surface Water (Pekel et al., Nature 2016),
# via Google Earth Engine - same auth/project as script 34's NDVI
# pull, no new account needed.
#
# HONEST LIMITATION: this dataset's YearlyHistory only covers
# 1984-2021 - there is NO 2026 "live" surface water layer yet
# (unlike NASA POWER weather or MODIS NDVI, which update in near
# real time). Surface water extent at district level changes far
# more slowly year-to-year than weather does (a reservoir doesn't
# appear/disappear the way rainfall does), so the most recent
# available year (2021) is used as a CURRENT BASELINE rather than
# faked as live data - this is stated explicitly, not hidden.
#
# OUTPUT: data/processed/surface_water_seasonal.csv with one row
# per district-year: water_pct_kharif, water_pct_rabi (% of the
# 5km buffer classified as seasonal+permanent water).
# =============================================================

import ee
import pandas as pd
import time

PROJECT_ID = 'certificate-automation-458414'

print("Authenticating with Earth Engine...")
ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print("Earth Engine initialized.")

coords = pd.read_csv('data/processed/district_coordinates_final.csv')
print(f"Districts loaded: {len(coords)}")

features = []
for _, row in coords.iterrows():
    geom = ee.Geometry.Point([row['lon'], row['lat']]).buffer(5000)
    features.append(ee.Feature(geom, {'district': row['district'], 'state': row['state']}))
district_fc = ee.FeatureCollection(features)

yearly_water = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')

GSW_START_YEAR = 1997  # matches the project's training data start
GSW_END_YEAR = 2021    # last year this dataset covers

def get_water_pct_for_year(year):
    """waterClass: 0=no data, 1=not water, 2=seasonal water,
    3=permanent water. We count seasonal+permanent as 'water present'
    and report it as a % of the buffer area."""
    img = yearly_water.filter(ee.Filter.eq('year', year)).first()
    water_mask = img.select('waterClass').gte(2)  # seasonal or permanent
    reduced = water_mask.reduceRegions(
        collection=district_fc,
        reducer=ee.Reducer.mean(),  # mean of a 0/1 mask = fraction water
        scale=30
    )
    return reduced.getInfo()

results = []
for year in range(GSW_START_YEAR, GSW_END_YEAR + 1):
    print(f"Year {year}...")
    info = get_water_pct_for_year(year)
    water_map = {
        f['properties']['district']: f['properties'].get('mean')
        for f in info['features']
    }
    for _, row in coords.iterrows():
        d = row['district']
        pct = water_map.get(d)
        results.append({
            'state': row['state'], 'district': d, 'year': year,
            'water_pct': (pct * 100) if pct is not None else None,
        })
    time.sleep(0.5)

water_df = pd.DataFrame(results)

# Kharif and Rabi use the SAME yearly value - water extent doesn't
# meaningfully change within a year at this dataset's annual
# resolution, unlike rainfall/temperature which genuinely differ
# by season. This is a real limitation of the dataset's granularity,
# stated explicitly rather than fabricating a seasonal split.
water_df['water_pct_kharif'] = water_df['water_pct']
water_df['water_pct_rabi'] = water_df['water_pct']
water_df = water_df.drop(columns=['water_pct'])

water_df.to_csv('data/processed/surface_water_seasonal.csv', index=False)

print(f"\nSaved data/processed/surface_water_seasonal.csv: {water_df.shape}")
print(f"Completeness: {water_df['water_pct_kharif'].notna().mean():.1%}")

# Build the "current" baseline file (script 25/27 expects a live-style
# input) using the MOST RECENT available year (2021) - explicitly
# labeled as a baseline, not live data.
current_baseline = water_df[water_df['year'] == GSW_END_YEAR][
    ['state', 'district', 'water_pct_kharif', 'water_pct_rabi']
].rename(columns={
    'water_pct_kharif': 'current_water_pct_kharif',
    'water_pct_rabi': 'current_water_pct_rabi',
})
current_baseline.to_csv('data/processed/current_surface_water_baseline.csv', index=False)
print(f"Saved data/processed/current_surface_water_baseline.csv "
      f"(baseline year: {GSW_END_YEAR}, NOT live - see script docstring)")

print("\n[NOTE] This dataset has no 2022-2026 coverage yet. The 'current'")
print("file uses 2021 as a slowly-changing baseline, not live data -")
print("state this explicitly in the paper if you cite this feature.")