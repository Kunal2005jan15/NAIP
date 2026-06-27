# =============================================================
# NAIP Project - Step 7a: Get Full District List
# =============================================================
import pandas as pd

df = pd.read_csv('data/processed/master_dataset.csv')

districts = df[['state', 'district']].drop_duplicates().sort_values(['state', 'district'])

print(f"Total unique district-state combinations: {len(districts)}")
print(f"\nBy state:")
print(districts['state'].value_counts())

districts.to_csv('data/processed/district_list.csv', index=False)
print(f"\nSaved to data/processed/district_list.csv")
print(f"\nFull list:")
print(districts.to_string(index=False))