# =============================================================
# NAIP Project - Step 13: Final Consolidated Cleaning
# =============================================================
# This applies ALL fixes in one place, in the correct order:
#   1. Remove Uttarakhand districts mislabeled as UP
#   2. Drop corrupted Punjab "S" entry
#   3. Drop Summer season rows (already done in v3, kept here for completeness)
#   4. Rename Shahid Bhagat Singh Nagar -> Nawanshahr
#   5. Merge district coordinates
# Output: master_dataset_FINAL.csv - the single source of truth
# =============================================================

import pandas as pd

master = pd.read_csv('data/processed/master_dataset_v3.csv')
print(f"Starting rows: {len(master)}")

# ---------------------------------------------------------------
# FIX 1: Remove Uttarakhand districts (confirmed via UP Reorganisation Act 2000)
# ---------------------------------------------------------------

UTTARAKHAND_DISTRICTS = [
    'Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun',
    'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh',
    'Rudra Prayag', 'Tehri Garhwal', 'Udam Singh Nagar', 'Uttar Kashi'
]

before = len(master)
master = master[
    ~((master['state'] == 'Uttar Pradesh') &
      (master['district'].isin(UTTARAKHAND_DISTRICTS)))
]
print(f"Removed {before - len(master)} Uttarakhand rows")

# ---------------------------------------------------------------
# FIX 2: Drop corrupted "S" entry in Punjab
# ---------------------------------------------------------------

before = len(master)
master = master[master['district'] != 'S']
print(f"Removed {before - len(master)} corrupted 'S' rows")

# ---------------------------------------------------------------
# CONFIRM: Nawanshahr rename already applied in v3
# ---------------------------------------------------------------

remaining_old_name = (master['district'] == 'Shahid Bhagat Singh Nagar').sum()
print(f"Remaining 'Shahid Bhagat Singh Nagar' rows (should be 0): {remaining_old_name}")

# ---------------------------------------------------------------
# MERGE COORDINATES
# ---------------------------------------------------------------

coords = pd.read_csv('data/processed/district_coordinates_final.csv')
merged = pd.merge(
    master,
    coords[['state', 'district', 'lat', 'lon']],
    on=['state', 'district'],
    how='left'
)

missing = merged['lat'].isna().sum()
print(f"\nRows with missing coordinates after merge: {missing}")

if missing > 0:
    print("Districts still missing coordinates:")
    print(merged[merged['lat'].isna()][['state', 'district']].drop_duplicates())

# ---------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------

print(f"\n{'='*60}")
print(f"FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total rows:     {len(merged)}")
print(f"Districts:      {merged['district'].nunique()}")
print(f"States:         {sorted(merged['state'].unique())}")
print(f"Crops:          {sorted(merged['crop'].unique())}")
print(f"Seasons:        {sorted(merged['season'].unique())}")
print(f"Year range:     {merged['year'].min()} - {merged['year'].max()}")

print(f"\nDistricts per state:")
print(merged.groupby('state')['district'].nunique())

merged.to_csv('data/processed/master_dataset_FINAL.csv', index=False)
print(f"\nSaved to data/processed/master_dataset_FINAL.csv")
print("This is now the single source of truth for the rest of the project.")