# =============================================================
# ANNA - Step 35b: Live NDVI Fetch (current KHARIF season)
# =============================================================
# Companion to script 35 (Rabi). Same MODIS NDVI, same 5 km
# district buffer, same method - only the window changes to the
# Kharif convention used in training (script 34): Jun 1 - Oct 31.
#
# Output: data/processed/current_ndvi_kharif_2026.csv
#         columns: state, district, current_ndvi_kharif
#
# Note: MODIS composites lag real time by ~2-3 weeks, so the last
# part of October may not be included yet. Same honest-limitation
# pattern as the weather pull.
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

composite = modis.filterDate('2026-06-01', '2026-10-31').mean().multiply(0.0001)
reduced = composite.reduceRegions(
    collection=district_fc, reducer=ee.Reducer.mean(), scale=250
).getInfo()

rows = []
for f in reduced['features']:
    rows.append({
        'state': f['properties']['state'],
        'district': f['properties']['district'],
        'current_ndvi_kharif': f['properties'].get('mean'),
    })

ndvi = pd.DataFrame(rows)
ndvi.to_csv('data/processed/current_ndvi_kharif_2026.csv', index=False)

print(f"\nSaved data/processed/current_ndvi_kharif_2026.csv: {ndvi.shape}")
print(f"Completeness: {ndvi['current_ndvi_kharif'].notna().mean():.1%}")
print(f"NDVI range: {ndvi['current_ndvi_kharif'].min():.3f} to {ndvi['current_ndvi_kharif'].max():.3f}")