# =============================================================
# NAIP Project - Step 26: Diagnose & Fix -999 Fill Value Bug
# =============================================================
# NASA POWER uses -999 as a "no data available" placeholder for
# very recent dates where satellite processing hasn't caught up.
# Our aggregation summed these placeholders as if they were real
# rainfall, producing impossible negative totals.
#
# Fix: replace -999 (and any value <= -900, to be safe against
# similar placeholder conventions) with NaN BEFORE aggregating,
# then aggregate only over genuinely valid days.
# =============================================================

import pandas as pd
import numpy as np

print("Loading current weather data...")
current_weather = pd.read_csv('data/raw/weather_current_2020_2026.csv')

# ---------------------------------------------------------------
# DIAGNOSE: confirm the -999 fill value is actually present
# ---------------------------------------------------------------

weather_cols = ['rainfall_mm', 'temp_max_c', 'temp_min_c', 'temp_avg_c', 'humidity_pct', 'solar_radiation']

print("\nChecking for NASA POWER fill values (-999) before fix:")
for col in weather_cols:
    n_fill = (current_weather[col] <= -900).sum()
    print(f"  {col}: {n_fill} fill-value rows out of {len(current_weather)}")

# Show exactly which dates are affected (expect: most recent days)
fill_rows = current_weather[current_weather['rainfall_mm'] <= -900]
if len(fill_rows) > 0:
    print(f"\nDate range of affected rows: {fill_rows['date'].min()} to {fill_rows['date'].max()}")
    print(f"(This should be the most recent few days, where NASA hasn't finished processing yet)")

# ---------------------------------------------------------------
# FIX: replace fill values with NaN
# ---------------------------------------------------------------

current_weather_clean = current_weather.copy()
for col in weather_cols:
    current_weather_clean.loc[current_weather_clean[col] <= -900, col] = np.nan

print(f"\nAfter fix - remaining valid rows per column:")
for col in weather_cols:
    print(f"  {col}: {current_weather_clean[col].notna().sum()} / {len(current_weather_clean)} valid")

current_weather_clean.to_csv('data/raw/weather_current_2020_2026_CLEAN.csv', index=False)
print(f"\nSaved cleaned file to data/raw/weather_current_2020_2026_CLEAN.csv")
print("✓ Fix validated. Re-run script 25 using this CLEAN file.")