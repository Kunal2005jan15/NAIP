# =============================================================
# NAIP Project - Step 40: Statistical Model Validation
# =============================================================
# Goes beyond the single train/valid/test split used for day-to-day
# development. For the paper's methodology section, a single split's
# R²/MAPE is one sample - this script answers "how stable is that
# number across different train/test boundaries" and "is the model
# well-behaved across the full range of predictions, or only on
# average."
#
# Three things:
#   1. WALK-FORWARD CROSS-VALIDATION: train on years up to Y, test
#      on Y+1, slide forward. Reports mean +/- std across folds -
#      a much more defensible accuracy claim than one split.
#   2. RESIDUAL DIAGNOSTICS: residual mean (should be ~0, i.e. not
#      systematically biased), residual skew, and residuals broken
#      down by crop/state - flags any systematic under/over-
#      prediction pattern a single aggregate RMSE would hide.
#   3. PER-DECILE INTERVAL CALIBRATION: the published 81% coverage
#      is an AVERAGE - this checks whether it holds across the
#      whole yield range or only in the middle of the distribution.
# =============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test = pd.read_csv('data/processed/test_v2.csv')
full = pd.concat([train, valid, test]).sort_values('year').reset_index(drop=True)

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [l.strip() for l in f.readlines()]

with open('outputs/metrics/xgb_best_params.json') as f:
    BEST_PARAMS = json.load(f)

TARGET = 'yield_kg_ha'

# =============================================================
# 1. WALK-FORWARD CROSS-VALIDATION
# =============================================================
print("=" * 60)
print("WALK-FORWARD CROSS-VALIDATION")
print("=" * 60)

years = sorted(full['year'].unique())
MIN_TRAIN_YEARS = 10  # don't test folds with too little training history
fold_results = []

test_years = [y for y in years if y >= years[0] + MIN_TRAIN_YEARS]

for test_year in test_years:
    train_fold = full[full['year'] < test_year]
    test_fold = full[full['year'] == test_year]
    if len(test_fold) < 5:  # skip near-empty fold years
        continue

    model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, verbosity=0)
    model.fit(train_fold[FEATURES], train_fold[TARGET])
    pred = model.predict(test_fold[FEATURES])

    rmse = np.sqrt(mean_squared_error(test_fold[TARGET], pred))
    mape = np.mean(np.abs((test_fold[TARGET] - pred) / test_fold[TARGET])) * 100
    r2 = r2_score(test_fold[TARGET], pred) if len(test_fold) > 1 else np.nan

    fold_results.append({
        'test_year': test_year, 'n_train': len(train_fold), 'n_test': len(test_fold),
        'rmse': rmse, 'mape': mape, 'r2': r2
    })
    print(f"  Test year {test_year}: n_train={len(train_fold):4d}  n_test={len(test_fold):3d}  "
          f"RMSE={rmse:6.1f}  MAPE={mape:5.1f}%  R2={r2:.3f}")

fold_df = pd.DataFrame(fold_results)
print(f"\n  Across {len(fold_df)} folds:")
print(f"  RMSE: {fold_df['rmse'].mean():.1f} +/- {fold_df['rmse'].std():.1f} kg/ha")
print(f"  MAPE: {fold_df['mape'].mean():.1f} +/- {fold_df['mape'].std():.1f}%")
print(f"  R2:   {fold_df['r2'].mean():.3f} +/- {fold_df['r2'].std():.3f}")
print(f"\n  Compare this spread against the single test_v2 split's numbers -")
print(f"  if they're close, the single-split result is representative;")
print(f"  if the walk-forward std is large, say so explicitly in the paper.")

fold_df.to_csv('outputs/metrics/walk_forward_cv_results.csv', index=False)

# =============================================================
# 2. RESIDUAL DIAGNOSTICS (on the standard test_v2 split, for
#    consistency with the headline reported numbers)
# =============================================================
print("\n" + "=" * 60)
print("RESIDUAL DIAGNOSTICS")
print("=" * 60)

train_full = pd.concat([train, valid])
final_model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, verbosity=0)
final_model.fit(train_full[FEATURES], train_full[TARGET])
test_pred = final_model.predict(test[FEATURES])
residuals = test[TARGET].values - test_pred

print(f"  Residual mean:  {residuals.mean():+.1f} kg/ha "
      f"({'no systematic bias' if abs(residuals.mean()) < 20 else 'WORTH INVESTIGATING - systematic bias'})")
print(f"  Residual std:   {residuals.std():.1f} kg/ha")
print(f"  Residual skew:  {stats.skew(residuals):.2f} "
      f"({'roughly symmetric' if abs(stats.skew(residuals)) < 0.5 else 'notably skewed - check for asymmetric error pattern'})")

test_with_resid = test.copy()
test_with_resid['residual'] = residuals

print("\n  Mean residual by crop (systematic over/under-prediction check):")
print(test_with_resid.groupby('crop')['residual'].agg(['mean', 'std', 'count']).round(1).to_string())

print("\n  Mean residual by state:")
print(test_with_resid.groupby('state')['residual'].agg(['mean', 'count']).round(1).to_string())

test_with_resid[['state', 'district', 'crop', 'year', TARGET, 'residual']].to_csv(
    'outputs/metrics/residuals_detail.csv', index=False
)

# =============================================================
# 3. PER-DECILE INTERVAL CALIBRATION
# =============================================================
print("\n" + "=" * 60)
print("PER-DECILE PREDICTION INTERVAL CALIBRATION")
print("=" * 60)
print("  Checking whether the published ~81% coverage holds across")
print("  the whole yield range, or only on average.\n")

with open('outputs/models/xgb_lower.pkl', 'rb') as f:
    xgb_lower = pickle.load(f)
with open('outputs/models/xgb_upper.pkl', 'rb') as f:
    xgb_upper = pickle.load(f)

test_eval = test.copy()
test_eval['pred'] = test_pred
test_eval['pred_lower'] = xgb_lower.predict(test[FEATURES])
test_eval['pred_upper'] = xgb_upper.predict(test[FEATURES])
test_eval['covered'] = (test_eval[TARGET] >= test_eval['pred_lower']) & (test_eval[TARGET] <= test_eval['pred_upper'])

test_eval['pred_decile'] = pd.qcut(test_eval['pred'], 5, labels=['lowest', 'low', 'mid', 'high', 'highest'])
calib_by_decile = test_eval.groupby('pred_decile')['covered'].agg(['mean', 'count'])
calib_by_decile.columns = ['coverage', 'n']
print(calib_by_decile.round(3).to_string())

worst_decile = calib_by_decile['coverage'].idxmin()
worst_coverage = calib_by_decile['coverage'].min()
if worst_coverage < 0.6:
    print(f"\n  [FLAG] '{worst_decile}' predictions have only {worst_coverage:.0%} coverage - "
          f"meaningfully worse than the overall average. State this range explicitly "
          f"as a limitation rather than letting the overall average hide it.")
else:
    print(f"\n  Calibration looks reasonably consistent across the yield range "
          f"(worst decile: {worst_decile} at {worst_coverage:.0%}).")

print("\n✓ Statistical validation complete.")
print("  Saved: outputs/metrics/walk_forward_cv_results.csv")
print("  Saved: outputs/metrics/residuals_detail.csv")