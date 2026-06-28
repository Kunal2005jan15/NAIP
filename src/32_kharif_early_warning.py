# =============================================================
# NAIP Project - Step 32 (FIXED, v2): Kharif 2026 Early-Warning
# Risk Flags
# =============================================================
# HONEST SCOPE: This does NOT predict Kharif/Rice yield (the
# season is still in progress - that would be overclaiming).
# Instead, it classifies each district's RISK EXPOSURE for the
# current Kharif season using:
#   1. Partial-season rainfall so far (real, from script 25/26)
#   2. That SAME calendar window (e.g. June 1-21) in each
#      district's own 2020-2025 history - apples-to-apples,
#      not paced/extrapolated across the full season
#   3. The official IMD 2026 seasonal forecast (90% of LPA, 60%
#      chance deficient - retrieved via RAG in script 31),
#      reported as separate supporting context
#
# Output: a RISK LEVEL (low/moderate/high/insufficient_data),
# not a yield number. This is a legitimate, honestly-scoped
# early-warning capability.
#
# v2 FIX: v1 paced the partial total linearly across the full
# 153-day season and compared it to each district's FULL-SEASON
# historical mean. That structurally underestimates early in the
# season (monsoon rainfall is back-loaded, not uniform) and
# produced an implausible 117/118 "high risk" result. v2 instead
# compares the actual partial total to the SAME calendar window
# in prior years. See inline comments at STEP 1 for the full
# diagnosis.
# =============================================================

import pandas as pd
import numpy as np

print("Building Kharif 2026 early-warning risk classification...")

current_weather = pd.read_csv('data/raw/weather_current_2020_2026_CLEAN.csv')
current_weather['date'] = pd.to_datetime(current_weather['date'])
current_weather['month'] = current_weather['date'].dt.month
current_weather['year'] = current_weather['date'].dt.year

train_history = pd.read_csv('data/processed/model_ready_v2.csv')

# ---------------------------------------------------------------
# STEP 1: Partial Kharif 2026 rainfall so far, per district
# ---------------------------------------------------------------
# IMPORTANT FIX (2026-06-28): the original version of this script
# linearly "paced" the partial total across the full 153-day season
# (partial / days_elapsed * 153) and then compared that paced number
# against each district's FULL-SEASON historical mean. That comparison
# is structurally biased: Kharif rainfall is NOT uniform across the
# season - it is heavily back-loaded after monsoon onset (typically
# late June onward in North India). Any early-June window, paced
# linearly, will look catastrophically low against a full-season
# average in literally ANY year, including normal ones. That bias is
# what produced an implausible "117/118 districts high risk" result -
# confirmed by checking June 2026 daily values directly (no fill-value
# corruption; rainfall genuinely is low pre-onset, as expected).
#
# THE FIX: compare this year's actual partial total against the SAME
# CALENDAR WINDOW in prior years (apples-to-apples), instead of pacing
# it across the full season and comparing to a full-season baseline.

KHARIF_WINDOW_END_MONTH = current_weather[
    (current_weather['year'] == 2026) & (current_weather['month'] >= 6) &
    (current_weather['rainfall_mm'].notna())
]['date'].max()
window_day_cutoff = KHARIF_WINDOW_END_MONTH.day if pd.notna(KHARIF_WINDOW_END_MONTH) else None

kharif_2026_so_far = current_weather[
    (current_weather['year'] == 2026) & (current_weather['month'] == 6) &
    (current_weather['date'].dt.day <= window_day_cutoff) & (current_weather['rainfall_mm'].notna())
]

partial_rainfall = kharif_2026_so_far.groupby(['state', 'district']).agg(
    actual_partial_rainfall_mm=('rainfall_mm', lambda x: x.sum(min_count=1)),
    days_elapsed=('rainfall_mm', lambda x: x.notna().sum())
).reset_index()

print(f"Partial Kharif 2026 data: {len(partial_rainfall)} districts, "
      f"{partial_rainfall['days_elapsed'].iloc[0]} days elapsed so far "
      f"(June 1-{window_day_cutoff})")

# ---------------------------------------------------------------
# STEP 2: SAME-CALENDAR-WINDOW historical rainfall (June 1 - same
# day-of-month) for each district, using 2020-2025 (whatever years
# are available in the current-weather feed - NOT the older
# 1997-2019 yield-modeling history, which has no daily resolution).
# ---------------------------------------------------------------

hist_window = current_weather[
    (current_weather['year'] < 2026) & (current_weather['year'] >= 2020) &
    (current_weather['month'] == 6) & (current_weather['date'].dt.day <= window_day_cutoff)
]

hist_window_by_year = hist_window.groupby(['year', 'state', 'district']).agg(
    window_total_mm=('rainfall_mm', lambda x: x.sum(min_count=1))
).reset_index()

hist_window_stats = hist_window_by_year.groupby('district')['window_total_mm'].agg(
    window_hist_mean='mean', window_hist_std='std', n_years='count'
).reset_index()

merged = partial_rainfall.merge(hist_window_stats, on='district', how='left')

# ---------------------------------------------------------------
# STEP 3: Note the OFFICIAL IMD seasonal forecast for context
# (90% of LPA nationally, 60% chance deficient - verified live via
# RAG, script 31). This is a FULL-SEASON national outlook, so it is
# reported alongside the risk flag as supporting context - it is
# NOT mathematically blended into the early-window z-score, because
# doing so previously (multiplying a paced estimate by 0.90) was
# part of what made the original bug harder to spot: it dressed up a
# broken pacing number with a real, citable figure. Kept separate
# and labeled, so each number's source stays honest and traceable.
# ---------------------------------------------------------------

IMD_2026_LPA_PCT = 90  # IMD official, verified 2026-05-29
IMD_DEFICIENT_PROB = 0.60

# ---------------------------------------------------------------
# STEP 4: RISK CLASSIFICATION (not a yield prediction)
# z-score of this year's actual same-window total vs. that
# district's own same-window history (2020-2025).
# ---------------------------------------------------------------

MIN_YEARS_REQUIRED = 3  # don't classify off a 1-2 year baseline

def classify_risk(row):
    if (pd.isna(row['window_hist_mean']) or pd.isna(row['window_hist_std'])
            or row['window_hist_std'] == 0 or row['n_years'] < MIN_YEARS_REQUIRED):
        return 'insufficient_data'

    z = (row['actual_partial_rainfall_mm'] - row['window_hist_mean']) / row['window_hist_std']

    if z < -1.2:
        return 'high'
    elif z < -0.5:
        return 'moderate'
    else:
        return 'low'

merged['kharif_risk_zscore'] = (
    (merged['actual_partial_rainfall_mm'] - merged['window_hist_mean']) / merged['window_hist_std']
)
merged['kharif_risk_level'] = merged.apply(classify_risk, axis=1)

merged['risk_explanation'] = merged.apply(lambda r: (
    f"{r['actual_partial_rainfall_mm']:.0f}mm so far this June (days 1-{window_day_cutoff}) vs. "
    f"{r['window_hist_mean']:.0f}mm average for this same period in {int(r['n_years'])} prior years. "
    f"IMD's national outlook for the full season is {IMD_2026_LPA_PCT}% of LPA "
    f"({int(IMD_DEFICIENT_PROB*100)}% chance deficient) - context, not blended into this number."
) if pd.notna(r['window_hist_mean']) and r['n_years'] >= MIN_YEARS_REQUIRED
    else "Insufficient same-window historical baseline (fewer than 3 prior years) for this district.", axis=1)

print(f"\nRisk distribution across {len(merged)} districts:")
print(merged['kharif_risk_level'].value_counts())

merged.to_csv('data/processed/kharif_2026_risk_flags.csv', index=False)
print("\nSaved to data/processed/kharif_2026_risk_flags.csv")
print("\nIMPORTANT: This is a RISK CLASSIFICATION based on this season's actual")
print("rainfall-so-far compared to the SAME calendar window in 2020-2025, plus")
print("the official IMD outlook reported as context. It is NOT a yield prediction.")
print("Full Kharif yield outlook will only be available once the season completes (~October 2026).")
print("\n[OK] Kharif early-warning module built.")