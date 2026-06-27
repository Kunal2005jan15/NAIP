import pandas as pd
import os

SNAPSHOT_DIR = 'data/snapshots'
all_snapshots = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.startswith('snapshot_') and f.endswith('.csv')])
print("All snapshots on disk, in order:")
for f in all_snapshots:
    print(f"  {f}")

print(f"\nTotal: {len(all_snapshots)}")

# Load the two we THINK we're comparing
current_file = all_snapshots[-1]
previous_file = all_snapshots[-2]
print(f"\nScript 28 would compare:")
print(f"  current:  {current_file}")
print(f"  previous: {previous_file}")

current = pd.read_csv(f'{SNAPSHOT_DIR}/{current_file}')
previous = pd.read_csv(f'{SNAPSHOT_DIR}/{previous_file}')

print(f"\ncurrent shape: {current.shape}, columns: {current.columns.tolist()}")
print(f"previous shape: {previous.shape}")

# Check for duplicate district+crop keys (would cause merge explosion/misalignment)
print(f"\nDuplicate (district, crop) pairs in current: {current.duplicated(subset=['district','crop']).sum()}")
print(f"Duplicate (district, crop) pairs in previous: {previous.duplicated(subset=['district','crop']).sum()}")

# Check the specific districts we expect to see changed
test_districts = ['Sambhal', 'Baghpat', 'Sonipat', 'Faridabad', 'Banda']
print(f"\nChecking our 5 deliberately-perturbed districts:")
for d in test_districts:
    cur_row = current[current['district'] == d]
    prev_row = previous[previous['district'] == d]
    print(f"\n  {d}:")
    print(f"    in current ({len(cur_row)} rows):  yield={cur_row['current_pred_yield'].values if len(cur_row) else 'MISSING'}")
    print(f"    in previous ({len(prev_row)} rows): yield={prev_row['current_pred_yield'].values if len(prev_row) else 'MISSING'}")

# Check the actually-flagged districts to see why THEY got flagged
print(f"\n\nChecking the districts that DID get flagged (unexpectedly):")
for d in ['Kheri', 'Sitapur']:
    cur_row = current[current['district'] == d]
    prev_row = previous[previous['district'] == d]
    print(f"\n  {d}:")
    print(f"    current rows:\n{cur_row[['district','crop','current_pred_yield','rainfall_anomaly_index']].to_string(index=False)}")
    print(f"    previous rows:\n{prev_row[['district','crop','current_pred_yield','rainfall_anomaly_index']].to_string(index=False)}")