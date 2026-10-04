import pandas as pd
df = pd.read_csv('data/processed/kharif_2026_risk_flags.csv')
print(df['days_elapsed'].describe())
print()
print(df['days_elapsed'].value_counts().sort_index())
print()
print("Districts with the FEWEST valid days (most likely to be unfairly penalized):")
print(df.nsmallest(10, 'days_elapsed')[['state','district','days_elapsed','kharif_risk_zscore','kharif_risk_level']].to_string(index=False))