import pandas as pd

# Check Kaggle CSV
df = pd.read_csv('data/raw/India Agriculture Crop Production.csv', nrows=5)
print('=== KAGGLE CSV ===')
print('Columns:', df.columns.tolist())
print(df.head(3).to_string())
print()

# Check Mendeley XLS
xls = pd.read_excel('data/raw/main merge (droped _merge==2) (560 dist 1990-2015).xls', nrows=5)
print('=== MENDELEY XLS ===')
print('Columns:', xls.columns.tolist())
print(xls.head(3).to_string())