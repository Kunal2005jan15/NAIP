# Diagnostic for the 88%-high-risk sanity gate failure (2026-07-08).
# Run from the NAIP root: python diagnose_kharif.py
# Doesn't change any pipeline files - just prints numbers so we can tell
# apart "real severe monsoon delay" from "windowing bug" before trusting
# either explanation.

import pandas as pd

df = pd.read_csv('data/processed/kharif_2026_risk_flags.csv')

print("=" * 70)
print("1. Overall distribution")
print("=" * 70)
print(df['kharif_risk_level'].value_counts())
print()

print("=" * 70)
print("2. z-score distribution (should be a spread, not everything jammed left)")
print("=" * 70)
print(df['kharif_risk_zscore'].describe())
print()

print("=" * 70)
print("3. n_years per district (low counts here would explain instability)")
print("=" * 70)
print(df['n_years'].value_counts().sort_index())
print()

print("=" * 70)
print("4. Sample of 8 districts - actual vs historical mean/std/z, by hand")
print("=" * 70)
cols = ['district', 'actual_partial_rainfall_mm', 'window_hist_mean',
        'window_hist_std', 'n_years', 'kharif_risk_zscore', 'kharif_risk_level']
present_cols = [c for c in cols if c in df.columns]
print(df[present_cols].sample(8, random_state=1).to_string(index=False))
print()

print("=" * 70)
print("5. Compare a FIXED small set of districts against last week's numbers")
print("=" * 70)
print("(Ambala/Panchkula/Hoshiarpur were flagged 'high' back on Jun 23 too -")
print(" if they show similar mm-so-far vs. historical average NOW as they did")
print(" then, that's consistent with a real, worsening pattern, not a bug.)")
watch_districts = ['Ambala', 'Panchkula', 'Hoshiarpur']
print(df[df['district'].isin(watch_districts)][present_cols].to_string(index=False))