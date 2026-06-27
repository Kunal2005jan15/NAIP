# =============================================================
# NAIP Project - Step 12: Merge District Coordinates
# =============================================================
import pandas as pd

master = pd.read_csv('data/processed/master_dataset_v3.csv')
coords = pd.read_csv('data/processed/district_coordinates_final.csv')

print(f"Master rows: {len(master)}")
print(f"Coordinate rows: {len(coords)}")

# Merge on district + state
merged = pd.merge(
    master,
    coords[['state', 'district', 'lat', 'lon']],
    on=['state', 'district'],
    how='left'
)

print(f"\nMerged rows: {len(merged)}")
print(f"Rows with missing coordinates: {merged['lat'].isna().sum()}")

if merged['lat'].isna().sum() > 0:
    print("\nDistricts missing coordinates:")
    print(merged[merged['lat'].isna()][['state', 'district']].drop_duplicates())

merged.to_csv('data/processed/master_dataset_v4.csv', index=False)
print(f"\nSaved data/processed/master_dataset_v4.csv with coordinates attached")

# Also save a unique district-coordinate list for the NASA fetch script
unique_districts = merged[['state', 'district', 'lat', 'lon']].drop_duplicates().dropna()
unique_districts.to_csv('data/processed/districts_for_nasa_fetch.csv', index=False)
print(f"Saved {len(unique_districts)} unique districts to data/processed/districts_for_nasa_fetch.csv")