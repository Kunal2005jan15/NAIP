# =============================================================
# NAIP Project - Step 29: Generate Test Second Snapshot
# =============================================================
# PURPOSE: This is a TESTING/DEMO utility, not part of the
# production pipeline. It takes the existing snapshot and
# perturbs a handful of districts to simulate a realistic
# change (e.g., a few days passing with a drought worsening
# in one place and easing in another). This lets you DEMO
# and VALIDATE the diff/alert logic in script 28 without
# waiting for a real multi-day gap between weather refreshes.
#
# For your actual submission, you would instead let real time
# pass and re-run scripts 22 -> 25 -> 27 -> 28 with genuinely
# new NASA data. Document this honestly: this script exists
# to validate the alerting logic works correctly, not to
# fabricate real predictions.
# =============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import time

SNAPSHOT_DIR = 'data/snapshots'

all_snapshots = sorted([
    f for f in os.listdir(SNAPSHOT_DIR) if f.startswith('snapshot_') and f.endswith('.csv')
])
latest = all_snapshots[-1]
print(f"Loading latest snapshot to perturb: {latest}")

df = pd.read_csv(f'{SNAPSHOT_DIR}/{latest}')

np.random.seed(42)

# Pick 5 districts to simulate a WORSENING drought signal
worsen_idx = df.sample(5, random_state=1).index
df.loc[worsen_idx, 'rainfall_anomaly_index'] -= np.random.uniform(0.5, 1.0, size=5)
df.loc[worsen_idx, 'rai_severity_relative'] += np.random.uniform(0.6, 1.2, size=5)
df.loc[worsen_idx, 'current_pred_yield'] -= np.random.uniform(150, 350, size=5)
df.loc[worsen_idx, 'drought_flag'] = 1

# Pick 3 different districts to simulate IMPROVING conditions
improve_idx = df.drop(worsen_idx).sample(3, random_state=2).index
df.loc[improve_idx, 'rainfall_anomaly_index'] += np.random.uniform(0.3, 0.6, size=3)
df.loc[improve_idx, 'current_pred_yield'] += np.random.uniform(80, 180, size=3)

print(f"\nSimulated WORSENING in: {df.loc[worsen_idx, 'district'].tolist()}")
print(f"Simulated IMPROVEMENT in: {df.loc[improve_idx, 'district'].tolist()}")

# Save as a new, later-timestamped snapshot
time.sleep(1)  # ensure a distinct timestamp
new_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df['snapshot_timestamp'] = new_timestamp

new_path = f'{SNAPSHOT_DIR}/snapshot_{new_timestamp}.csv'
df.to_csv(new_path, index=False)
print(f"\nSaved simulated second snapshot: {new_path}")
print("\nNow run src/28_district_watch.py again to see the diff/alert logic fire.")