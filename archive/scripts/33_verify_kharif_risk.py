import pandas as pd

df = pd.read_csv('data/processed/kharif_2026_risk_flags.csv')

print("Sample size (n_years) distribution across districts:")
print(df['n_years'].value_counts().sort_index())

print(f"\nZ-score distribution:")
print(df['kharif_risk_zscore'].describe())

print(f"\nSample of 'high' risk districts with their actual numbers:")
high_risk = df[df['kharif_risk_level'] == 'high'].head(5)
print(high_risk[['district', 'actual_partial_rainfall_mm', 'window_hist_mean',
                  'window_hist_std', 'n_years', 'kharif_risk_zscore']].to_string(index=False))

print(f"\nThe 2 'low' risk districts (worth eyeballing for sanity):")
low_risk = df[df['kharif_risk_level'] == 'low']
print(low_risk[['district', 'actual_partial_rainfall_mm', 'window_hist_mean', 'kharif_risk_zscore']].to_string(index=False))