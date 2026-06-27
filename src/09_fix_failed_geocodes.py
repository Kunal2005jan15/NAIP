# =============================================================
# NAIP Project - Step 9: Fix Failed Geocodes + Merge Duplicate
# =============================================================
# Issues found:
# 1. Spelling variants Nominatim doesn't recognize
# 2. Nawanshahr / Shahid Bhagat Singh Nagar are the SAME district
#    (renamed in 2008) - need to merge in master_dataset too
# =============================================================

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geocoded = pd.read_csv('data/processed/district_coordinates.csv')

geolocator = Nominatim(user_agent="naip_research_project_kunal_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

# ---------------------------------------------------------------
# MANUAL FIXES — corrected names to search
# ---------------------------------------------------------------

FIXES = {
    'Charki Dadri':    'Charkhi Dadri, Haryana, India',
    'Firozepur':       'Ferozepur, Punjab, India',
    'Kushi Nagar':     'Kushinagar, Uttar Pradesh, India',
    'Sant Kabeer Nagar':'Sant Kabir Nagar, Uttar Pradesh, India',
}

print("Fixing spelling-variant failures...")
for district_name, query in FIXES.items():
    location = geocode(query)
    if location:
        mask = geocoded['district'] == district_name
        geocoded.loc[mask, 'lat'] = location.latitude
        geocoded.loc[mask, 'lon'] = location.longitude
        geocoded.loc[mask, 'geocode_status'] = 'success_manual_fix'
        print(f"  FIXED  {district_name:30s} -> {location.latitude:.3f}, {location.longitude:.3f}")
    else:
        print(f"  STILL FAILED: {district_name}")

# ---------------------------------------------------------------
# SPECIAL CASE: Nawanshahr = Shahid Bhagat Singh Nagar
# Same district, renamed in 2008. Use Nawanshahr's coordinates.
# ---------------------------------------------------------------

nawanshahr_coords = geocoded[geocoded['district'] == 'Nawanshahr']

if len(nawanshahr_coords) > 0:
    lat = nawanshahr_coords['lat'].values[0]
    lon = nawanshahr_coords['lon'].values[0]
    
    mask = geocoded['district'] == 'Shahid Bhagat Singh Nagar'
    geocoded.loc[mask, 'lat'] = lat
    geocoded.loc[mask, 'lon'] = lon
    geocoded.loc[mask, 'geocode_status'] = 'success_same_as_nawanshahr'
    print(f"\n  Shahid Bhagat Singh Nagar -> using Nawanshahr coords: {lat:.3f}, {lon:.3f}")

print(f"\nFinal success count: {(geocoded['lat'].notna()).sum()}/{len(geocoded)}")

geocoded.to_csv('data/processed/district_coordinates_final.csv', index=False)
print("Saved to data/processed/district_coordinates_final.csv")

# ---------------------------------------------------------------
# MERGE DUPLICATE DISTRICT IN MASTER DATASET
# ---------------------------------------------------------------
# Rename "Shahid Bhagat Singh Nagar" to "Nawanshahr" everywhere
# in the master dataset so they're treated as ONE district
# with a continuous time series

print("\n" + "="*60)
print("Merging duplicate district in master_dataset.csv...")
print("="*60)

master = pd.read_csv('data/processed/master_dataset.csv')

before_unique = master[master['state']=='Punjab']['district'].nunique()

count_renamed = (master['district'] == 'Shahid Bhagat Singh Nagar').sum()
master.loc[master['district'] == 'Shahid Bhagat Singh Nagar', 'district'] = 'Nawanshahr'

after_unique = master[master['state']=='Punjab']['district'].nunique()

print(f"  Renamed {count_renamed} rows from 'Shahid Bhagat Singh Nagar' to 'Nawanshahr'")
print(f"  Punjab unique districts: {before_unique} -> {after_unique}")

# Check for duplicate rows now (same district+crop+year appearing twice)
dupes = master.duplicated(subset=['state', 'district', 'crop', 'year'], keep=False)
print(f"  Duplicate district-crop-year rows after merge: {dupes.sum()}")

if dupes.sum() > 0:
    print("\n  Sample duplicates (need averaging):")
    print(master[dupes].sort_values(['district','crop','year']).head(10))
    
    # Average duplicates (e.g. if same district-year appears under both names)
    master = master.groupby(
        ['state', 'district', 'crop', 'year', 'season'], as_index=False
    ).mean(numeric_only=True)
    print(f"\n  After averaging duplicates: {len(master)} rows")

master.to_csv('data/processed/master_dataset_v2.csv', index=False)
print(f"\nSaved corrected master dataset to data/processed/master_dataset_v2.csv")