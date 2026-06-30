# =============================================================
# NAIP Project - Step 28 (FIXED, v2) — District Watch Snapshot & Diff
# =============================================================
# Each run:
#   1. Builds a fresh snapshot from current_predictions_full.csv
#      (this is always the REAL current model output)
#   2. Looks for the most recently saved PRIOR snapshot (captured
#      BEFORE writing the new one, so we never diff against ourselves)
#   3. Saves the new snapshot
#   4. Diffs new vs prior, classifies each district by genuine
#      change magnitude (not absolute severity alone)
# =============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

SNAPSHOT_DIR = 'data/snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

print("Building new snapshot from current model outputs...")

current = pd.read_csv('data/processed/current_predictions_full.csv')
nasa_seasonal = pd.read_csv('data/processed/nasa_seasonal_ALL_DISTRICTS.csv')

rai_volatility = nasa_seasonal.groupby('district')['rainfall_anomaly_index'].agg(
    hist_rai_mean='mean', hist_rai_std='std'
).reset_index()

new_snapshot = current.merge(rai_volatility, on='district', how='left')
new_snapshot['rai_severity_relative'] = (
    (new_snapshot['rainfall_anomaly_index'] - new_snapshot['hist_rai_mean']) / new_snapshot['hist_rai_std']
).abs()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
new_snapshot['snapshot_timestamp'] = timestamp

snapshot_cols = ['state', 'district', 'crop', 'current_pred_yield', 'rainfall_anomaly_index',
                  'rai_severity_relative', 'drought_flag', 'flood_flag', 'change_vs_2019',
                  'snapshot_timestamp']

# Capture existing snapshots BEFORE saving the new one
existing_snapshots = sorted([
    f for f in os.listdir(SNAPSHOT_DIR) if f.startswith('snapshot_') and f.endswith('.csv')
])
print(f"Existing snapshots found before this run: {len(existing_snapshots)}")
previous_file = existing_snapshots[-1] if len(existing_snapshots) >= 1 else None

new_path = f'{SNAPSHOT_DIR}/snapshot_{timestamp}.csv'
new_snapshot[snapshot_cols].to_csv(new_path, index=False)
print(f"Saved new snapshot: {new_path}")

if previous_file is None:
    print("\nThis is the FIRST snapshot ever — no comparison possible.")
    new_snapshot['alert_level'] = 'baseline'
    new_snapshot['change_note'] = 'First snapshot — establishing baseline.'
    alert_feed = new_snapshot.sort_values('rai_severity_relative', ascending=False)
else:
    print(f"Diffing against previous snapshot: {previous_file}")
    previous = pd.read_csv(f'{SNAPSHOT_DIR}/{previous_file}')

    diff = new_snapshot.merge(
        previous[['district', 'crop', 'current_pred_yield', 'rainfall_anomaly_index', 'rai_severity_relative']],
        on=['district', 'crop'], how='left', suffixes=('', '_prev')
    )

    diff['yield_change_since_last'] = diff['current_pred_yield'] - diff['current_pred_yield_prev']
    diff['rai_change_since_last'] = diff['rainfall_anomaly_index'] - diff['rainfall_anomaly_index_prev']
    diff['severity_change_since_last'] = diff['rai_severity_relative'] - diff['rai_severity_relative_prev']

    YIELD_WATCH_THRESHOLD = 150
    YIELD_URGENT_THRESHOLD = 300
    SEVERITY_CHANGE_THRESHOLD = 0.3

    def classify_alert(row):
        if pd.isna(row['rai_severity_relative_prev']):
            return 'new', 'New district data added since last snapshot.'

        yield_chg = row['yield_change_since_last']
        sev_chg = row['severity_change_since_last']

        no_real_change = abs(yield_chg) < 5 and abs(sev_chg) < 0.05
        if no_real_change:
            status = 'elevated_stable' if row['rai_severity_relative'] > 1.2 else 'stable'
            note = (f"No change since last check (still elevated severity {row['rai_severity_relative']:.2f})."
                    if status == 'elevated_stable' else "No significant change since last check.")
            return status, note

        worsening = sev_chg > SEVERITY_CHANGE_THRESHOLD or yield_chg < -YIELD_WATCH_THRESHOLD

        if abs(yield_chg) > YIELD_URGENT_THRESHOLD and worsening:
            return 'urgent', f"Predicted yield shifted {yield_chg:+.0f} kg/ha since last check (worsening)."
        if row['rai_severity_relative'] > 2.0 and sev_chg > SEVERITY_CHANGE_THRESHOLD:
            return 'urgent', f"Drought/flood severity increasing: {row['rai_severity_relative_prev']:.2f} -> {row['rai_severity_relative']:.2f}."
        if abs(yield_chg) > YIELD_WATCH_THRESHOLD or abs(sev_chg) > SEVERITY_CHANGE_THRESHOLD:
            direction = "improved" if yield_chg > 0 else "declined"
            return 'watch', f"Yield estimate {direction} {yield_chg:+.0f} kg/ha since last check."

        # BUG FIX (2026-06-29): everything between the no_real_change
        # floor (5 kg/ha) and the watch threshold (150 kg/ha) used to
        # fall through silently to 'stable' - conflating TRUE zero
        # change with real, moderate, just-not-alert-worthy change.
        # Caught by the automated sanity gate (script 33): a District
        # Watch run showed 98% 'stable' while the underlying
        # yield_change_since_last had real variance (std=51.5,
        # range=245.3) - exactly because many districts were landing
        # in this unlabeled gap, not because of a calculation error.
        # 'stable' now means what it says: genuinely no change.
        direction = "up" if yield_chg > 0 else "down"
        return 'minor_change', f"Minor change since last check ({yield_chg:+.0f} kg/ha, below alert threshold)."

    diff[['alert_level', 'change_note']] = diff.apply(lambda r: pd.Series(classify_alert(r)), axis=1)

    alert_order = {'urgent': 0, 'watch': 1, 'new': 2, 'elevated_stable': 3, 'stable': 4}
    diff['alert_sort'] = diff['alert_level'].map(alert_order)
    alert_feed = diff.sort_values(['alert_sort', 'rai_severity_relative'], ascending=[True, False])

print(f"\nAlert summary:")
print(alert_feed['alert_level'].value_counts())

print(f"\nDistricts with ACTUAL change flagged (urgent/watch only):")
real_alerts = alert_feed[alert_feed['alert_level'].isin(['urgent', 'watch'])]
if len(real_alerts) > 0:
    print(real_alerts[['district', 'crop', 'alert_level', 'change_note']].to_string(index=False))
else:
    print("  (none)")

alert_feed.to_csv('outputs/metrics/district_watch_feed.csv', index=False)
print(f"\nSaved to outputs/metrics/district_watch_feed.csv")
print("\n[OK] District Watch complete.")