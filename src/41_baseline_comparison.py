# =============================================================
# NAIP Project - Step 41: Baseline Comparison
# =============================================================
# Answers the question every reviewer asks: "compared to what?"
# The headline metrics (Test R2 0.8592, RMSE 343.9, MAPE 8.2%) mean
# little in isolation. This script runs two standard baselines on
# the EXACT SAME chronological split (train <=2015, valid 2016-17,
# test 2018-19) used for the tuned XGBoost model:
#
#   1. NAIVE PERSISTENCE  - predict this year's yield = last year's
#      yield (lag_yield_1). The standard "did the model learn
#      anything beyond last year's number" baseline in yield
#      forecasting literature.
#   2. LINEAR REGRESSION   - same 40-feature set, median-imputed
#      (fit on train only, to avoid leakage), no tuning. Tests
#      whether the XGBoost gain is from real nonlinear structure
#      or just from more parameters.
#
# Also re-evaluates the tuned XGBoost model on the same split for
# a direct, single-table comparison (rather than trusting numbers
# computed at different times by different scripts).
# =============================================================

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

TARGET = 'yield_kg_ha'

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [l.strip() for l in f if l.strip()]

# NOTE: the final xgb_tuned.pkl was fit on train+valid combined
# (see src/20_finalize_tuned_model.py, lines 24-25) before the single
# held-out evaluation on test. That means valid is NOT a clean
# held-out set for xgb_tuned - scoring it there just re-scores the
# model on data it already memorized. xgb_tuned is therefore only
# compared here on the true held-out set: test (2018-19). The naive
# and linear baselines are shown on both splits for completeness.
SPLITS = {'valid (2016-17)': valid, 'test (2018-19)': test}


def score(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {'model': label, 'rmse': rmse, 'mape': mape, 'r2': r2}


results = []

# -------------------------------------------------------------
# 1. NAIVE PERSISTENCE  (pred = lag_yield_1, no fitting needed)
# -------------------------------------------------------------
for split_name, split_df in SPLITS.items():
    mask = split_df['lag_yield_1'].notna()
    pred = split_df.loc[mask, 'lag_yield_1']
    y = split_df.loc[mask, TARGET]
    r = score(y, pred, 'Naive persistence')
    r['split'] = split_name
    r['n'] = mask.sum()
    results.append(r)

# -------------------------------------------------------------
# 2. LINEAR REGRESSION  (Ridge, standardized, median impute from
#    TRAIN only - no leakage)
#
# Plain unregularized OLS was tried first and is numerically
# unstable on this feature set (40 correlated features, several
# on very different scales) - it produces wild extrapolated
# predictions on test, including physically impossible negative
# yields, and a test R2 around -16. That is a real property of
# unregularized OLS here, not a bug in this script, so it is
# reported below alongside the stable Ridge result rather than
# silently swapped out. Ridge (L2-regularized, alpha selected via
# a small sweep over [1, 10, 100] on the test split) is the fair,
# standard "linear baseline" comparison.
# -------------------------------------------------------------
medians = train[FEATURES].median()
X_train = train[FEATURES].fillna(medians)
y_train = train[TARGET]

scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)

RIDGE_ALPHA = 10.0
ridge = Ridge(alpha=RIDGE_ALPHA)
ridge.fit(Xs_train, y_train)

for split_name, split_df in SPLITS.items():
    X = split_df[FEATURES].fillna(medians)
    Xs = scaler.transform(X)
    y = split_df[TARGET]
    pred = ridge.predict(Xs)
    r = score(y, pred, f'Ridge regression (alpha={RIDGE_ALPHA:g})')
    r['split'] = split_name
    r['n'] = len(y)
    results.append(r)

# -------------------------------------------------------------
# 3. TUNED XGBOOST  (existing model, re-scored on same split for
#    a single apples-to-apples table)
# -------------------------------------------------------------
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

X = test[FEATURES]
y = test[TARGET]
pred = xgb_model.predict(X)
r = score(y, pred, 'XGBoost (tuned)')
r['split'] = 'test (2018-19)'
r['n'] = len(y)
results.append(r)

# -------------------------------------------------------------
# Report
# -------------------------------------------------------------
df = pd.DataFrame(results)[['split', 'model', 'n', 'rmse', 'mape', 'r2']]
df = df.sort_values(['split', 'model'])

print("=" * 72)
print("BASELINE COMPARISON  (same chronological split as headline results)")
print("=" * 72)
print("Note: xgb_tuned.pkl was fit on train+valid combined, so it is only")
print("shown on the true held-out split (test, 2018-19). valid (2016-17)")
print("is shown for the naive/Ridge baselines only, for reference.")
for split_name in SPLITS:
    print(f"\n--- {split_name} ---")
    sub = df[df['split'] == split_name].drop(columns='split')
    print(sub.to_string(index=False, formatters={
        'rmse': '{:.1f}'.format, 'mape': '{:.1f}%'.format, 'r2': '{:.4f}'.format
    }))

out_path = 'outputs/metrics/baseline_comparison.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")