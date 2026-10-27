# =============================================================
# ANNA - Step 45: Validate Rice (Kharif 2026) signals vs IMD
# =============================================================
# Ground-truth check on the WEATHER signals (official 2026 yields are
# not published yet, so yield itself cannot be validated today).
#
# Compares, per state, ANNA's Kharif-2026 rainfall departure and
# drought/flood flags against IMD's reported state rainfall departures.
#
# Run: python src/45_validate_rice_vs_imd.py
# =============================================================

import pandas as pd

df = pd.read_csv('data/processed/current_rice_predictions_full.csv')

# IMD reference departures from normal (%), season rainfall 1 Jun - 29 Sep 2026
# (IMD category-wise report as reported by Down To Earth, 30 Sep 2026).
# Positive = surplus. State-level only - district detail is NOT validated here.
IMD = {
    'Punjab': -41,
    'Haryana': -23,
    'Uttar Pradesh': 8,
}

need = ['cur_rain_kharif', 'hist_mean']
missing_cols = [c for c in need if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Columns missing from the rice predictions file: {missing_cols}")

# v3: prefer the departure vs each district's own live 2020-25 history
# (what script 44 now computes); fall back to the old baseline ratio.
if 'rain_departure_pct' in df.columns:
    df['departure_pct'] = df['rain_departure_pct']
else:
    df['departure_pct'] = (df['cur_rain_kharif'] / df['hist_mean'] - 1) * 100

print("=" * 78)
print("ANNA Kharif-2026 weather signal vs IMD, by state")
print("=" * 78)
rows = []
for state, g in df.groupby('state'):
    rows.append({
        'state': state,
        'districts': len(g),
        'ANNA_rain_mm': round(g['cur_rain_kharif'].mean()),
        'training_scale_normal_mm': round(g['hist_mean'].mean()),
        'ANNA_departure_%': round(g['departure_pct'].mean(), 1),
        'IMD_departure_%': IMD.get(state, float('nan')),
        'drought_flagged': int(g['drought_flag'].sum()),
        'flood_flagged': int(g['flood_flag'].sum()),
        'anomaly_missing': int(g['rainfall_anomaly_index'].isna().sum()),
    })
out = pd.DataFrame(rows)
print(out.to_string(index=False))

print("\nVerdict checks")
print("-" * 78)
for state, imd in IMD.items():
    r = out[out['state'] == state]
    if r.empty:
        print(f"[?] {state}: not found in the file (check the spelling)")
        continue
    r = r.iloc[0]
    gap = r['ANNA_departure_%'] - imd
    ok_dir = (r['ANNA_departure_%'] < 0) == (imd < 0)
    flag_ok = (imd < -15 and r['drought_flagged'] > 0) or (imd >= -15)
    print(f"{state:15s} IMD {imd:+.0f}% | ANNA {r['ANNA_departure_%']:+.1f}% | "
          f"direction {'OK' if ok_dir else 'WRONG'} | gap {gap:+.0f} pts | "
          f"drought flags {'plausible' if flag_ok else 'SUSPICIOUS (0 flags despite deficit)'}")

print("\nGeneral checks")
print("-" * 78)
print(f"Total drought-flagged: {int(df['drought_flag'].sum())} | flood-flagged: {int(df['flood_flag'].sum())} of {len(df)}")
print(f"Districts with kharif_days_observed < 140: {int((df.get('kharif_days_observed', pd.Series([999]*len(df))) < 140).sum())}")
print("\nHow to read this: NASA POWER is satellite/model rainfall, not IMD gauges, so a gap of")
print("10-15 points is normal. The DIRECTION must match, and a state running 20-35% short")
print("should not show zero drought flags while showing flood flags.")