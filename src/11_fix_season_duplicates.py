# =============================================================
# NAIP Project - Step 11: Properly Fix Season Duplicates
# =============================================================
# Finding: Kharif and Summer rice are genuinely DIFFERENT crops
#          (Summer = secondary, <2% of area). NOT duplicates.
# Decision: Drop "Summer" season rows entirely.
#           Keep only Kharif (rice) and Rabi (wheat) — the two
#           dominant, agronomically meaningful seasons that match
#           our project's defined scope.
# This REPLACES the incorrect averaging done in script 09.
# =============================================================

import pandas as pd

# Rebuild master dataset from the ORIGINAL clean files,
# not from the incorrectly-averaged master_dataset_v2.csv

kaggle_clean_path = 'data/processed/master_dataset.csv'  # before bad averaging
master = pd.read_csv(kaggle_clean_path)

print(f"Before season fix: {len(master)} rows")
print(master['season'].value_counts())

# Drop Summer season rows
before = len(master)
master = master[master['season'] != 'Summer'].copy()
print(f"\nDropped {before - len(master)} 'Summer' season rows")

# Now apply the Nawanshahr rename fix correctly (this part was right)
count_renamed = (master['district'] == 'Shahid Bhagat Singh Nagar').sum()
master.loc[master['district'] == 'Shahid Bhagat Singh Nagar', 'district'] = 'Nawanshahr'
print(f"Renamed {count_renamed} rows from 'Shahid Bhagat Singh Nagar' to 'Nawanshahr'")

# Now check for TRUE duplicates (same district+crop+year+season)
dupes = master.duplicated(subset=['state', 'district', 'crop', 'year', 'season'], keep=False)
print(f"\nTrue duplicate rows remaining: {dupes.sum()}")

if dupes.sum() > 0:
    print("Sample true duplicates:")
    print(master[dupes].sort_values(['district','crop','year']).head(10))
    # These would be genuine exact duplicates - safe to average
    master = master.groupby(
        ['state', 'district', 'crop', 'year', 'season'], as_index=False
    ).mean(numeric_only=True)

print(f"\nFinal clean master dataset: {len(master)} rows")
print(f"\nSeason breakdown:")
print(master['season'].value_counts())
print(f"\nCrop breakdown:")
print(master['crop'].value_counts())

master.to_csv('data/processed/master_dataset_v3.csv', index=False)
print(f"\nSaved CORRECT version to data/processed/master_dataset_v3.csv")
print("\nThis is now the authoritative master dataset going forward.")