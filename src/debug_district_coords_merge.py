import pandas as pd

test_preds = pd.read_csv('outputs/metrics/test_predictions_FINAL.csv')
coords = pd.read_csv('data/processed/district_coordinates_final.csv')

print("test_preds columns:")
print(test_preds.columns.tolist())
print(f"\ntest_preds shape: {test_preds.shape}")
print(f"\nSample district/state values from test_preds:")
print(test_preds[['district', 'state']].drop_duplicates().head(10))

print("\n" + "="*60)
print("coords columns:")
print(coords.columns.tolist())
print(f"\ncoords shape: {coords.shape}")
print(f"\nSample district/state values from coords:")
print(coords[['district', 'state']].drop_duplicates().head(10))

print("\n" + "="*60)
merged = test_preds.merge(
    coords[['district', 'state', 'lat', 'lon']],
    on=['district', 'state'],
    how='left'
)
print(f"Merged shape: {merged.shape}")
print(f"Merged columns: {merged.columns.tolist()}")
print(f"Non-null lat values: {merged['lat'].notna().sum() if 'lat' in merged.columns else 'lat column missing!'}")