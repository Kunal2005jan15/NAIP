# =============================================================
# NAIP Project - Step 34: NDVI Historical Backfill (Tier 2)
# =============================================================
# WHY: every feature so far is a proxy for crop condition (rainfall,
# temperature, soil moisture proxy via water_balance). NDVI is a
# DIRECT satellite measurement of vegetation health/vigor - it sees
# the crop itself, not just the weather that might affect it. This
# is the single highest-value addition identified in the Tier
# 2 review.
#
# CAUTION - UNLIKE EVERY OTHER SCRIPT SO FAR IN THIS PROJECT:
# this one has NOT been run or validated against real data, because
# Google Earth Engine isn't reachable from the sandbox this was
# written in. Treat the first local run as a real test. Likely
# friction points if something goes wrong:
#   - "project not registered for Earth Engine" -> make sure the
#     GCP project used in ee.Initialize() has the Earth Engine API
#     enabled (console.cloud.google.com -> APIs & Services)
#   - Quota/rate limit errors -> this script already batches all
#     118 districts into ONE reduceRegions() call per season-year
#     (40 calls total for 2000-2019) rather than one call per
#     district-year (4700+ calls) - if you still hit limits, add
#     a time.sleep(1) between iterations.
#   - MODIS MOD13Q1 starts Feb 2000 - years 1997-1999 will have NO
#     NDVI data. This is a real, unavoidable gap; leave those years
#     as NaN, the model already handles missing lag features this
#     way (see script 17's completeness report).
#
# OUTPUT: data/processed/ndvi_seasonal.csv with one row per
# district-year: ndvi_kharif, ndvi_rabi (mean NDVI, true -1..1
# scale, NOT the raw *10000 MODIS scale).
# =============================================================

import ee
import pandas as pd
import time

PROJECT_ID = 'certificate-automation-458414'

print("Authenticating with Earth Engine (browser window will open on first run)...")
ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print("Earth Engine initialized.")

coords = pd.read_csv('data/processed/district_coordinates_final.csv')
print(f"Districts loaded: {len(coords)}")

# 5km buffer around each district centroid - an area average is far
# less noisy than a single 250m pixel, and more representative of
# district-level agricultural land than a point sample.
features = []
for _, row in coords.iterrows():
    geom = ee.Geometry.Point([row['lon'], row['lat']]).buffer(5000)
    features.append(ee.Feature(geom, {'district': row['district'], 'state': row['state']}))
district_fc = ee.FeatureCollection(features)

modis = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')

def get_season_ndvi(start_date, end_date):
    """One server-side mean composite for the season, reduced over
    ALL districts in a single call - this is what keeps this script
    to ~40 API round-trips instead of thousands."""
    composite = modis.filterDate(start_date, end_date).mean().multiply(0.0001)  # MODIS NDVI is scaled x10000
    reduced = composite.reduceRegions(collection=district_fc, reducer=ee.Reducer.mean(), scale=250)
    return reduced.getInfo()

MODIS_START_YEAR = 2000  # MOD13Q1 coverage begins Feb 2000
END_YEAR = 2019  # matches the historical training data's last year

results = []
for year in range(MODIS_START_YEAR, END_YEAR + 1):
    print(f"Year {year}...")

    # Kharif: Jun-Oct of this year (within one calendar year, no
    # cross-year issue - same convention as script 15)
    kharif_info = get_season_ndvi(f'{year}-06-01', f'{year}-10-31')
    kharif_map = {
        f['properties']['district']: f['properties'].get('mean')
        for f in kharif_info['features']
    }

    # Rabi: Nov(year) - Mar(year+1) - SAME crop-year convention as
    # the fixed script 15 (Nov/Dec belong to `year`, Jan-Mar belong
    # to the season that STARTED the previous November). Getting this
    # wrong here would reintroduce the exact bug already fixed once.
    rabi_info = get_season_ndvi(f'{year}-11-01', f'{year + 1}-03-31')
    rabi_map = {
        f['properties']['district']: f['properties'].get('mean')
        for f in rabi_info['features']
    }

    for _, row in coords.iterrows():
        d = row['district']
        results.append({
            'state': row['state'],
            'district': d,
            'year': year,
            'ndvi_kharif': kharif_map.get(d),
            'ndvi_rabi': rabi_map.get(d),
        })

    time.sleep(0.5)  # gentle on API quota during the historical backfill

ndvi_df = pd.DataFrame(results)
ndvi_df.to_csv('data/processed/ndvi_seasonal.csv', index=False)

print(f"\nSaved data/processed/ndvi_seasonal.csv: {ndvi_df.shape}")
print(f"ndvi_kharif completeness: {ndvi_df['ndvi_kharif'].notna().mean():.1%}")
print(f"ndvi_rabi completeness:   {ndvi_df['ndvi_rabi'].notna().mean():.1%}")
print("\nNote: years 1997-1999 will be entirely absent (pre-MODIS) -")
print("this is expected, not an error. They'll show as NaN after the")
print("merge in script 16, same as other features with partial coverage.")
print("\n[UNVALIDATED] This is the first script in the project not yet")
print("confirmed against real data - please report the actual output")
print("(especially completeness %% and any errors) before we wire it")
print("into the feature list.")