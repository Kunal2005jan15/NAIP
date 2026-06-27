# =============================================================
# NAIP Project - Step 8: Fix District List + Geocode
# =============================================================
# Fix 1: Remove Uttarakhand districts mislabeled as "Uttar Pradesh"
#        (confirmed via Wikipedia: UP Reorganisation Act 2000)
# Fix 2: Drop corrupted Punjab entry "S"
# Then: Geocode each district to lat/lon using Nominatim (OSM)
# =============================================================

import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------------------------------------------------------------
# STEP 1: LOAD AND CLEAN
# ---------------------------------------------------------------

districts = pd.read_csv('data/processed/district_list.csv')
print(f"Before cleaning: {len(districts)} districts")

UTTARAKHAND_DISTRICTS = [
    'Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun',
    'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh',
    'Rudra Prayag', 'Tehri Garhwal', 'Udam Singh Nagar', 'Uttar Kashi'
]

before = len(districts)
districts = districts[
    ~((districts['state'] == 'Uttar Pradesh') &
      (districts['district'].isin(UTTARAKHAND_DISTRICTS)))
]
print(f"Removed {before - len(districts)} Uttarakhand districts mislabeled as UP")

# Check the corrupted Punjab entry before dropping
master = pd.read_csv('data/processed/master_dataset.csv')
mystery = master[(master['state'] == 'Punjab') & (master['district'] == 'S')]
print(f"\nRows affected by corrupted 'S' entry: {len(mystery)}")
if len(mystery) > 0:
    print(mystery[['crop', 'year', 'yield_kg_ha']].head())

districts = districts[districts['district'] != 'S']
print(f"\nAfter cleaning: {len(districts)} districts")
print(districts['state'].value_counts())

districts.to_csv('data/processed/district_list_clean.csv', index=False)

# ---------------------------------------------------------------
# STEP 2: GEOCODE EACH DISTRICT
# ---------------------------------------------------------------
# We append ", India" and the state name to improve match accuracy
# RateLimiter ensures we don't hit Nominatim's free-tier rate limit
# (max ~1 request/second)

print("\n" + "="*60)
print("GEOCODING DISTRICTS (this takes a few minutes)...")
print("="*60)

geolocator = Nominatim(user_agent="naip_research_project_kunal")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

results = []

for idx, row in districts.iterrows():
    district = row['district']
    state = row['state']
    
    query = f"{district} district, {state}, India"
    
    try:
        location = geocode(query)
        if location:
            results.append({
                'state': state,
                'district': district,
                'lat': location.latitude,
                'lon': location.longitude,
                'geocode_status': 'success'
            })
            print(f"  OK  {state:15s} {district:25s} -> {location.latitude:.3f}, {location.longitude:.3f}")
        else:
            results.append({
                'state': state,
                'district': district,
                'lat': None,
                'lon': None,
                'geocode_status': 'failed'
            })
            print(f"  FAIL {state:15s} {district:25s} -> NOT FOUND")
    except Exception as e:
        results.append({
            'state': state,
            'district': district,
            'lat': None,
            'lon': None,
            'geocode_status': f'error: {e}'
        })
        print(f"  ERROR {state:15s} {district:25s} -> {e}")

geocoded = pd.DataFrame(results)

# ---------------------------------------------------------------
# STEP 3: REPORT AND SAVE
# ---------------------------------------------------------------

success_count = (geocoded['geocode_status'] == 'success').sum()
fail_count = len(geocoded) - success_count

print(f"\n{'='*60}")
print(f"GEOCODING COMPLETE")
print(f"{'='*60}")
print(f"  Success: {success_count}/{len(geocoded)}")
print(f"  Failed:  {fail_count}/{len(geocoded)}")

if fail_count > 0:
    print(f"\nFailed districts (will need manual lookup):")
    print(geocoded[geocoded['geocode_status'] != 'success'][['state', 'district']].to_string(index=False))

geocoded.to_csv('data/processed/district_coordinates.csv', index=False)
print(f"\nSaved to data/processed/district_coordinates.csv")