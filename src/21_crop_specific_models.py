# =============================================================
# NAIP Project - Step 21: Crop-Specific Models (Wheat / Rice)
# =============================================================
# WHY: SHAP analysis showed Rice (Kharif) has consistently worse
# accuracy than Wheat (Rabi) in the combined model (8.9% vs 6.6%
# MAPE in earlier analysis). A single model with `crop_encoded`
# as one feature has to compromise between two crops whose yield
# is driven by different seasons, different weather sensitivities,
# and different management practices. Splitting into two models
# lets each one specialize, instead of crop_encoded just nudging
# a shared set of tree splits.
#
# This does NOT throw away the combined model - it's kept as the
# baseline/benchmark. This script trains Wheat-only and Rice-only
# models on the SAME corrected v2 feature set (post window-fix,
# post agronomic-feature additions) and reports whether the split
# actually helps each crop, or whether pooling was fine all along.
# =============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import json
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test  = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES_FULL = [line.strip() for line in f.readlines()]

# crop_encoded is constant within a crop-specific subset - drop it
# so it can't show up as a spurious "important" feature with zero
# real variance, and so SHAP rankings are honest.
FEATURES = [f for f in FEATURES_FULL if f != 'crop_encoded']

TARGET = 'yield_kg_ha'
train_full = pd.concat([train, valid]).sort_values('year').reset_index(drop=True)

PARAM_GRID = {
    'n_estimators':      [300, 600, 1000],
    'max_depth':         [4, 6, 8],
    'learning_rate':     [0.01, 0.05, 0.1],
    'min_child_weight':  [1, 5],
    'reg_alpha':         [0, 0.1],
    'reg_lambda':        [0.5, 1.0],
    'subsample':         [0.8],
    'colsample_bytree':  [0.8],
}

results_summary = {}

for crop in ['Wheat', 'Rice']:
    print(f"\n{'='*60}")
    print(f"CROP: {crop}")
    print(f"{'='*60}")

    crop_train = train_full[train_full['crop'] == crop].sort_values('year').reset_index(drop=True)
    crop_test = test[test['crop'] == crop].reset_index(drop=True)

    X_train, y_train = crop_train[FEATURES], crop_train[TARGET]
    X_test, y_test = crop_test[FEATURES], crop_test[TARGET]

    print(f"Train rows: {len(crop_train)} ({crop_train['year'].min()}-{crop_train['year'].max()})")
    print(f"Test rows:  {len(crop_test)} ({crop_test['year'].min() if len(crop_test) else 'n/a'}-{crop_test['year'].max() if len(crop_test) else 'n/a'})")

    tscv = TimeSeriesSplit(n_splits=4)  # smaller than combined (5) - half the rows per crop
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0),
        param_distributions=PARAM_GRID,
        n_iter=40,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = np.mean(np.abs(y_test - pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    r2 = r2_score(y_test, pred)

    print(f"\n{crop}-SPECIFIC MODEL — TEST SET RESULTS")
    print(f"  RMSE : {rmse:.1f} kg/ha")
    print(f"  MAE  : {mae:.1f} kg/ha")
    print(f"  MAPE : {mape:.1f}%")
    print(f"  R²   : {r2:.4f}")
    print(f"  Best params: {search.best_params_}")

    model_path = f'outputs/models/xgb_{crop.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Saved: {model_path}")

    with open(f'outputs/metrics/xgb_{crop.lower()}_best_params.json', 'w') as f:
        json.dump(search.best_params_, f, indent=2)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    imp.to_csv(f'outputs/metrics/shap_importance_{crop.lower()}.csv', index=False)

    print(f"\n  Top 8 SHAP features for {crop}:")
    print(imp.head(8).to_string(index=False))

    results_summary[crop] = {'rmse': rmse, 'mape': mape, 'r2': r2, 'n_test': len(crop_test)}

# ---------------------------------------------------------------
# COMBINED (CROP-SPLIT) vs SINGLE-MODEL COMPARISON
# ---------------------------------------------------------------
# Weighted by test-set size, to compare fairly against the single
# combined model's overall test RMSE/MAPE/R2 (run script 19/20 for
# that number, then compare against this weighted figure by hand or
# paste both outputs back for a side-by-side read).

n_wheat = results_summary['Wheat']['n_test']
n_rice = results_summary['Rice']['n_test']
n_total = n_wheat + n_rice

weighted_rmse = (results_summary['Wheat']['rmse'] * n_wheat + results_summary['Rice']['rmse'] * n_rice) / n_total
weighted_mape = (results_summary['Wheat']['mape'] * n_wheat + results_summary['Rice']['mape'] * n_rice) / n_total

print(f"\n{'='*60}")
print("CROP-SPLIT MODELS — COMBINED (TEST-SIZE-WEIGHTED) SUMMARY")
print(f"{'='*60}")
print(f"  Wheat: RMSE={results_summary['Wheat']['rmse']:.1f}  MAPE={results_summary['Wheat']['mape']:.1f}%  R²={results_summary['Wheat']['r2']:.4f}  (n={n_wheat})")
print(f"  Rice:  RMSE={results_summary['Rice']['rmse']:.1f}  MAPE={results_summary['Rice']['mape']:.1f}%  R²={results_summary['Rice']['r2']:.4f}  (n={n_rice})")
print(f"  Weighted RMSE: {weighted_rmse:.1f} kg/ha")
print(f"  Weighted MAPE: {weighted_mape:.1f}%")
print(f"\n  Compare against the SINGLE combined model's test RMSE/MAPE")
print(f"  (from script 19/20) to see whether the split actually helped.")

with open('outputs/metrics/crop_split_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ Crop-specific model training complete.")