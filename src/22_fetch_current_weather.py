# =============================================================
# NAIP Project - Step 22: Fetch Current Weather (2020-2026)
# =============================================================
# Extends weather coverage past the yield data's 2019 cutoff.
# This lets us compute LIVE 2026 drought/flood (RAI) flags and
# combine them with the most recent known yield baseline (2019)
# for an honest "current conditions" prediction mode.
#
# IMPORTANT: This does NOT give us 2020-2026 ACTUAL YIELD data
# (that's a separate, harder problem - government yield stats
# publish 1-2 years behind). This only extends WEATHER, which
# NASA POWER does provide near-present.
# =============================================================

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)

districts = pd.read_csv('data/processed/districts_for_nasa_fetch.csv')
print(f"Fetching CURRENT weather (2020-2026) for {len(districts)} districts...")
print(f"Estimated time: ~{len(districts) * 2 / 60:.1f} minutes")

PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN,T2M,RH2M,ALLSKY_SFC_SW_DWN"
START_YEAR = "2020"

# BUG FIX (2026-07-07): this used to be a hardcoded end="20260623" - frozen at
# whatever date it happened to be written on. That meant re-running this script
# NEVER advanced past June 23, 2026, no matter what day it was actually run on -
# not a Task Scheduler problem, not a NASA POWER lag, just a literal frozen date
# baked into the request. Fixed to compute "today" dynamically, with a 3-day
# buffer since NASA POWER's near-real-time data typically lags a few days behind
# the actual current date (their own processing latency, not a bug on our side).
NASA_POWER_LAG_DAYS = 3
END_DATE = (datetime.now() - timedelta(days=NASA_POWER_LAG_DAYS)).strftime("%Y%m%d")
print(f"Fetching through {END_DATE} (today minus {NASA_POWER_LAG_DAYS}-day NASA POWER lag buffer)")

def fetch_nasa_power(district_name, state, lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": PARAMETERS,
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      f"{START_YEAR}0101",
        "end":        END_DATE,
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

    time.sleep(1.5)

    if (idx + 1) % 20 == 0:
        combined_so_far = pd.concat(all_data).reset_index()
        combined_so_far.to_csv("data/raw/weather_current_PARTIAL.csv", index=False)
        print(f"      --- Progress saved ({idx+1} districts done) ---")

combined = pd.concat(all_data).reset_index()
combined.to_csv("data/raw/weather_current_2020_2026.csv", index=False)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")
print(f"Successful districts: {len(all_data)}/{len(districts)}")
print(f"Failed districts:     {len(failed)}")
print(f"Total weather rows:   {len(combined)}")
print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

if failed:
    pd.DataFrame(failed).to_csv("data/raw/weather_current_failures.csv", index=False)
    print(f"\nFailed districts saved to data/raw/weather_current_failures.csv")

print(f"\nSaved to data/raw/weather_current_2020_2026.csv")