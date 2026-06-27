# =============================================================
# NAIP Project - Step 14: Fetch NASA POWER Weather for ALL Districts
# =============================================================
# Replaces the 3-district proof-of-concept with full coverage.
# 118 districts x 15 years of daily data
# NASA POWER allows reasonable request volume - we rate limit
# to be respectful (1 request per ~2 seconds)
# =============================================================

import requests
import pandas as pd
import time
import os

os.makedirs("data/raw", exist_ok=True)

districts = pd.read_csv('data/processed/districts_for_nasa_fetch.csv')
print(f"Fetching weather for {len(districts)} districts...")
print(f"Estimated time: ~{len(districts) * 3 / 60:.1f} minutes")

PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN,T2M,RH2M,ALLSKY_SFC_SW_DWN"
START_YEAR = "1997"
END_YEAR   = "2019"

def fetch_nasa_power(district_name, state, lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": PARAMETERS,
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      f"{START_YEAR}0101",
        "end":        f"{END_YEAR}1231",
        "format":     "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        
        data = response.json()
        daily_data = data["properties"]["parameter"]
        
        df = pd.DataFrame(daily_data)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        
        df["district"] = district_name
        df["state"]    = state
        df["lat"]      = lat
        df["lon"]      = lon
        
        df = df.rename(columns={
            "PRECTOTCORR":        "rainfall_mm",
            "T2M_MAX":            "temp_max_c",
            "T2M_MIN":            "temp_min_c",
            "T2M":                "temp_avg_c",
            "RH2M":               "humidity_pct",
            "ALLSKY_SFC_SW_DWN":  "solar_radiation"
        })
        return df, None
    except Exception as e:
        return None, str(e)

all_data = []
failed = []

for idx, row in districts.iterrows():
    district_name = row['district']
    state         = row['state']
    lat           = row['lat']
    lon           = row['lon']
    
    df, error = fetch_nasa_power(district_name, state, lat, lon)
    
    if df is not None:
        all_data.append(df)
        print(f"  [{idx+1}/{len(districts)}] OK   {state:15s} {district_name:25s} ({len(df)} days)")
    else:
        failed.append({'state': state, 'district': district_name, 'error': error})
        print(f"  [{idx+1}/{len(districts)}] FAIL {state:15s} {district_name:25s} -> {error}")
    
    # Be polite to NASA's servers
    time.sleep(1.5)
    
    # Save progress every 20 districts in case of interruption
    if (idx + 1) % 20 == 0:
        combined_so_far = pd.concat(all_data).reset_index()
        combined_so_far.to_csv("data/raw/weather_all_districts_PARTIAL.csv", index=False)
        print(f"      --- Progress saved ({idx+1} districts done) ---")

# Final save
combined = pd.concat(all_data).reset_index()
combined.to_csv("data/raw/weather_all_districts.csv", index=False)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")
print(f"Successful districts: {len(all_data)}/{len(districts)}")
print(f"Failed districts:     {len(failed)}")
print(f"Total weather rows:   {len(combined)}")

if failed:
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv("data/raw/weather_fetch_failures.csv", index=False)
    print(f"\nFailed districts saved to data/raw/weather_fetch_failures.csv:")
    print(failed_df)

print(f"\nSaved to data/raw/weather_all_districts.csv")