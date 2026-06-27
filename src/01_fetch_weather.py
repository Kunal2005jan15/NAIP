# =============================================================
# NAIP Project - Step 1: Fetch Weather Data from NASA POWER API
# =============================================================
# NASA POWER API gives us free daily weather data for any lat/lon
# We'll pull data for key districts in UP, Punjab, Haryana

import requests
import pandas as pd
import time
import os

# ---------------------------------------------------------------
# DISTRICT COORDINATES
# Each district needs a lat/lon center point
# We start small - 3 districts, one per state
# Later we'll scale to all districts
# ---------------------------------------------------------------

DISTRICTS = {
    "Ludhiana_Punjab":    {"lat": 30.9010, "lon": 75.8573},
    "Karnal_Haryana":     {"lat": 29.6857, "lon": 76.9905},
    "Lucknow_UP":         {"lat": 26.8467, "lon": 80.9462},
}

# ---------------------------------------------------------------
# WHAT WEATHER VARIABLES DO WE WANT?
# PRECTOTCORR = Rainfall (mm/day)
# T2M_MAX     = Max Temperature (°C)
# T2M_MIN     = Min Temperature (°C)
# T2M         = Average Temperature (°C)
# RH2M        = Relative Humidity (%)
# ALLSKY_SFC_SW_DWN = Solar Radiation (optional but useful)
# ---------------------------------------------------------------

PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN,T2M,RH2M,ALLSKY_SFC_SW_DWN"

START_YEAR = "2010"
END_YEAR   = "2024"

# ---------------------------------------------------------------
# NASA POWER API FUNCTION
# ---------------------------------------------------------------

def fetch_nasa_power(district_name, lat, lon):
    """
    Fetches daily weather data from NASA POWER API for a given location.
    Returns a cleaned pandas DataFrame.
    """
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": PARAMETERS,
        "community":  "AG",           # AG = Agriculture community
        "longitude":  lon,
        "latitude":   lat,
        "start":      f"{START_YEAR}0101",   # YYYYMMDD format
        "end":        f"{END_YEAR}1231",
        "format":     "JSON"
    }
    
    print(f"Fetching data for {district_name}...")
    
    response = requests.get(url, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code} for {district_name}")
        return None
    
    data = response.json()
    
    # NASA POWER returns data nested under this key
    daily_data = data["properties"]["parameter"]
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    
    # Add district info
    df["district"] = district_name
    df["lat"]      = lat
    df["lon"]      = lon
    
    # Rename columns to readable names
    df = df.rename(columns={
        "PRECTOTCORR":        "rainfall_mm",
        "T2M_MAX":            "temp_max_c",
        "T2M_MIN":            "temp_min_c",
        "T2M":                "temp_avg_c",
        "RH2M":               "humidity_pct",
        "ALLSKY_SFC_SW_DWN":  "solar_radiation"
    })
    
    print(f"  Got {len(df)} days of data for {district_name}")
    return df


# ---------------------------------------------------------------
# MAIN: Loop through all districts and save
# ---------------------------------------------------------------

all_data = []

for district_name, coords in DISTRICTS.items():
    df = fetch_nasa_power(district_name, coords["lat"], coords["lon"])
    
    if df is not None:
        all_data.append(df)
    
    # Be polite to NASA's servers - wait 2 seconds between requests
    time.sleep(2)

# Combine all districts
combined = pd.concat(all_data)
combined = combined.reset_index()

# Save to raw data folder
os.makedirs("data/raw", exist_ok=True)
combined.to_csv("data/raw/weather_raw.csv", index=False)

print("\n--- SUMMARY ---")
print(f"Total rows: {len(combined)}")
print(f"Districts:  {combined['district'].unique()}")
print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
print(f"\nSaved to data/raw/weather_raw.csv")
print("\nFirst 5 rows:")
print(combined.head())