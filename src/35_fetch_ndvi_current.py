# =============================================================
# NAIP Project - Step 35: Live NDVI Fetch (current Rabi season)
# =============================================================
# COMPANION to script 34 (historical NDVI backfill) and script 25
# (live current-conditions). Without this, training would have
# ndvi_kharif/ndvi_rabi but the live dashboard would NOT - the
# exact train/serve mismatch shape already found and fixed once
# for the Rabi weather window. This closes that gap for NDVI.
#
# Uses the SAME Rabi window convention as scripts 15/25: Nov+Dec
# of the prior calendar year, Jan-Mar of the current one.
# =============================================================

import ee
import pandas as pd

PROJECT_ID = 'certificate-automation-458414'

print("Authenticating with Earth Engine...")
ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print("Earth Engine initialized.")

coords = pd.read_csv('data/processed/district_coordinates_final.csv')

features = []
for _, row in coords.iterrows():
    geom = ee.Geometry.Point([row['lon'], row['lat']]).buffer(5000)
    features.append(ee.Feature(geom, {'district': row['district'], 'state': row['state']}))
district_fc = ee.FeatureCollection(features)

modis = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')

# Live Rabi 2025-26: Nov+Dec 2025 through Mar 2026 (or through
# today, if the season is still in progress - MODIS composites lag
# real-time by ~2-3 weeks for processing, so this will reflect
# whatever's actually available, same honest-limitation pattern as
# script 25's weather pull).
composite = modis.filterDate('2025-11-01', '2026-03-31').mean().multiply(0.0001)
reduced = composite.reduceRegions(collection=district_fc, reducer=ee.Reducer.mean(), scale=250).getInfo()

rows = []
for f in reduced['features']:
    rows.append({
        'state': f['properties']['state'],
        'district': f['properties']['district'],
        'current_ndvi_rabi': f['properties'].get('mean'),
    })

ndvi_current = pd.DataFrame(rows)
ndvi_current.to_csv('data/processed/current_ndvi_2026.csv', index=False)

print(f"\nSaved data/processed/current_ndvi_2026.csv: {ndvi_current.shape}")
print(f"Completeness: {ndvi_current['current_ndvi_rabi'].notna().mean():.1%}")
print("\nRun this each time you refresh the dashboard's current-season data,")
print("alongside script 25 - same as the live weather refresh cadence.")