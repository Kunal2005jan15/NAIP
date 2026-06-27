# =============================================================
# NAIP Project - Step 19: Tune XGBoost Hyperparameters
# =============================================================
# Tests whether early stopping / regularization were too
# conservative for this dataset size (~3,900 training rows)
#
# Uses TimeSeriesSplit for cross-validation - NOT random KFold -
# to preserve our no-leakage methodology even during tuning
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test  = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

TARGET = 'yield_kg_ha'

# Combine train+valid for cross-validated tuning
# (test set stays completely untouched until final evaluation)
train_full = pd.concat([train, valid]).sort_values('year').reset_index(drop=True)

X_train_full = train_full[FEATURES]
y_train_full = train_full[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

print(f"Tuning data: {X_train_full.shape}")
print(f"Year range for tuning: {train_full['year'].min()}-{train_full['year'].max()}")

# ---------------------------------------------------------------
# TIME-SERIES CROSS VALIDATION
# ---------------------------------------------------------------
# Splits training data into sequential folds - each fold trains
# on earlier years, validates on later years. No shuffling.

tscv = TimeSeriesSplit(n_splits=5)

# ---------------------------------------------------------------
# PARAMETER GRID — testing theory: less aggressive regularization,
# more trees allowed, less aggressive early stopping
# ---------------------------------------------------------------

param_grid = {
    'n_estimators':      [300, 600, 1000],
    'max_depth':         [4, 6, 8],
    'learning_rate':     [0.01, 0.05, 0.1],
    'min_child_weight':  [1, 5],
    'reg_alpha':         [0, 0.1],
    'reg_lambda':        [0.5, 1.0],
    'subsample':         [0.8],
    'colsample_bytree':  [0.8],
}

print(f"\nGrid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
print("This will take several minutes — using RandomizedSearch for efficiency...")

from sklearn.model_selection import RandomizedSearchCV

xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0)

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_grid,
    n_iter=40,                  # test 40 random combinations instead of all
    scoring='neg_root_mean_squared_error',
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_full, y_train_full)

print(f"\n{'='*60}")
print(f"BEST PARAMETERS FOUND")
print(f"{'='*60}")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV RMSE: {-search.best_score_:.1f} kg/ha")

# ---------------------------------------------------------------
# EVALUATE TUNED MODEL ON TEST SET
# ---------------------------------------------------------------

best_model = search.best_estimator_
test_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, test_pred))
mae  = np.mean(np.abs(y_test - test_pred))
mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
r2   = r2_score(y_test, test_pred)

print(f"\n{'='*60}")
print(f"TUNED XGBOOST — TEST SET RESULTS")
print(f"{'='*60}")
print(f"  RMSE : {rmse:.1f} kg/ha")
print(f"  MAE  : {mae:.1f} kg/ha")
print(f"  MAPE : {mape:.1f}%")
print(f"  R²   : {r2:.4f}")

print(f"\n{'='*60}")
print(f"COMPARISON: Default vs Tuned XGBoost")
print(f"{'='*60}")
print(f"  Default XGBoost (v2):  R²=0.5864  RMSE=589.5")
print(f"  Tuned XGBoost:         R²={r2:.4f}  RMSE={rmse:.1f}")
print(f"  Random Forest (v2):    R²=0.8602  RMSE=342.8  <- benchmark to beat")

import pickle
with open('outputs/models/xgb_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

import json
with open('outputs/metrics/xgb_best_params.json', 'w') as f:
    json.dump(search.best_params_, f, indent=2)

print(f"\nSaved tuned model and best parameters.")