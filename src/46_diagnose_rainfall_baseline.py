# =============================================================
# ANNA - Step 46: Diagnose the rainfall baseline mismatch
# =============================================================
# Script 45 showed ANNA's Kharif-2026 rainfall running +22..+31% ABOVE
# its baseline in every state, while IMD reports deficits. A uniform
# offset like that usually means the live weather file and the
# historical baseline file are NOT on the same footing (different
# source version / window / units), not that it actually rained more.
#
# This script lines the two sources up year by year.
# Run: python src/46_diagnose_rainfall_baseline.py
# =============================================================

import pandas as pd

cur = pd.read_csv('data/raw/weather_current_2020_2026_CLEAN.csv')
cur['date'] = pd.to_datetime(cur['date'])
cur['year'] = cur['date'].dt.year
cur['month'] = cur['date'].dt.month
hist = pd.read_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv')

print("=" * 78)
print("1. Basic integrity of the live weather file")
print("=" * 78)
dups = cur.duplicated(['state', 'district', 'date']).sum()
print(f"Duplicate district-date rows: {dups}   (must be 0)")
print(f"Date range: {cur['date'].min().date()} to {cur['date'].max().date()}")
print(f"Rainfall: min {cur['rainfall_mm'].min():.1f}, max {cur['rainfall_mm'].max():.1f}, "
      f"mean daily {cur['rainfall_mm'].mean():.2f} mm")

# ---- Kharif totals (Jun-Oct) from the LIVE file, by year ----
k = cur[cur['month'].between(6, 10)]
k_tot = (k.groupby(['state', 'district', 'year'])['rainfall_mm']
           .agg(total='sum', days='count').reset_index())
k_state_year = k_tot.pivot_table(index='state', columns='year', values='total', aggfunc='mean').round(0)

# ---- Rabi totals (Nov-Mar, crop-year convention) from the LIVE file ----
cur['rabi_year'] = None
cur.loc[cur['month'].isin([11, 12]), 'rabi_year'] = cur['year']
cur.loc[cur['month'].isin([1, 2, 3]), 'rabi_year'] = cur['year'] - 1
r = cur.dropna(subset=['rabi_year'])
r_tot = r.groupby(['state', 'district', 'rabi_year'])['rainfall_mm'].sum().reset_index()
r_state_year = r_tot.pivot_table(index='state', columns='rabi_year', values='rainfall_mm', aggfunc='mean').round(0)

print("\n" + "=" * 78)
print("2. LIVE file: mean Kharif rainfall (mm, Jun-Oct) by state and year")
print("=" * 78)
print(k_state_year.to_string())

print("\n" + "=" * 78)
print("3. HISTORICAL baseline file: Kharif rainfall by state (mm)")
print("=" * 78)
print(f"Years covered: {int(hist['year'].min())} to {int(hist['year'].max())}"
      if 'year' in hist.columns else "No 'year' column in baseline file")
if 'year' in hist.columns:
    hy = hist.pivot_table(index='state', columns='year', values='nasa_rainfall_kharif', aggfunc='mean').round(0)
    print("Last 6 years of the baseline file:")
    print(hy.iloc[:, -6:].to_string())
h_mean = hist.groupby('state')['nasa_rainfall_kharif'].mean().round(0)

print("\n" + "=" * 78)
print("4. Side by side: do the two sources agree where they overlap?")
print("=" * 78)
cmp = pd.DataFrame({'baseline_mean_all_years': h_mean})
yrs = [y for y in k_state_year.columns if 2020 <= y <= 2025]
cmp['live_mean_2020_2025'] = k_state_year[yrs].mean(axis=1).round(0)
cmp['live_2026'] = k_state_year[2026] if 2026 in k_state_year.columns else float('nan')
cmp['live_vs_baseline_%'] = ((cmp['live_mean_2020_2025'] / cmp['baseline_mean_all_years'] - 1) * 100).round(1)
cmp['2026_vs_own_2020_25_%'] = ((cmp['live_2026'] / cmp['live_mean_2020_2025'] - 1) * 100).round(1)
print(cmp.to_string())

print("\nReference (IMD, season rainfall departure 2026): Punjab -36%, Haryana -24%, UP roughly -2%")

print("\n" + "=" * 78)
print("5. Same check for RABI (the wheat flags use the same method)")
print("=" * 78)
print("LIVE file mean Rabi rainfall (mm) by crop-year:")
print(r_state_year.to_string())
if 'nasa_rainfall_rabi' in hist.columns:
    print("\nBaseline-file mean Rabi rainfall (mm) by state:")
    print(hist.groupby('state')['nasa_rainfall_rabi'].mean().round(0).to_string())

print("\n" + "=" * 78)
print("6. How to read this")
print("=" * 78)
print("""- If 'live_vs_baseline_%' is about +20..+30 for EVERY state even for 2020-2025,
  the two files are on different footing (a step change at 2019->2020). The 2026
  anomaly is then an artifact, and the fix is to measure 2026 against the LIVE file's
  own 2020-2025 history (the method script 32 already uses), not the old baseline.
- If '2026_vs_own_2020_25_%' is negative for Punjab/Haryana, the live data itself
  agrees with IMD and only the baseline was wrong.
- If the Rabi table shows the same step change, the wheat drought flags are biased too.""")