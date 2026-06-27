# =============================================================
# NAIP Project - Step 10: Check Kharif vs Summer Rice Duplicates
# =============================================================
# Question: Is "Summer" rice a real separate crop, or is it
# a data artifact duplicating the Kharif row?
# =============================================================

import pandas as pd

df = pd.read_csv('data/raw/India Agriculture Crop Production.csv')

# Look at one specific case: Agra, Rice, 2008
subset = df[
    (df['District'].str.strip().str.title() == 'Agra') &
    (df['Crop'] == 'Rice') &
    (df['Year'] == '2008-09')
]
print("Agra Rice 2008-09 — all season rows:")
print(subset[['State','District','Crop','Year','Season','Area','Production','Yield']].to_string(index=False))

print("\n" + "="*60)

# Broader check: for UP rice rows, how many districts have
# BOTH a Kharif and a Summer row with IDENTICAL Area+Production?
up_rice = df[
    (df['State'].str.strip() == 'Uttar Pradesh') &
    (df['Crop'] == 'Rice') &
    (df['Season'].isin(['Kharif', 'Summer']))
].copy()

pivot = up_rice.pivot_table(
    index=['District', 'Year'],
    columns='Season',
    values=['Area', 'Production'],
    aggfunc='first'
)

# Flag rows where both seasons exist and Area is identical
both = pivot.dropna()
if len(both) > 0:
    identical = (both[('Area','Kharif')] == both[('Area','Summer')]).sum()
    print(f"District-years with BOTH Kharif and Summer rice rows: {len(both)}")
    print(f"Of those, rows where Area is IDENTICAL between seasons: {identical}")
    print(f"Percentage identical: {identical/len(both)*100:.1f}%")
else:
    print("No district-years found with both Kharif and Summer rows.")