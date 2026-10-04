# =============================================================
# NAIP Project - Step 42: Lagged Flood/Drought Feature Experiment
# =============================================================
# Follow-up to outputs/metrics/weak_fold_diagnosis_2014.md.
#
# Hypothesis: the 2014-2016 walk-forward folds (R2 0.19/0.63/0.59,
# vs 0.92/0.85 on either side) are weak because 2014 crashes hard
# right after 2013's extreme flood (73% of districts flagged,
# RAI 1.45), and the model has no LAGGED flood/drought feature to
# represent carry-over damage - only same-season flags.
#
# This script:
#   1. Builds district-year-level lag-1 versions of flood_flag,
#      drought_flag, and rainfall_anomaly_index (these are
#      confirmed district-year level, identical across crop rows
#      within a year - verified before writing this script).
#   2. Merges them into train/valid/test_v2 alongside the existing
#      40 features.
#   3. Re-runs walk-forward CV with the extra 3 features and
#      compares fold-by-fold against the baseline
#      (outputs/metrics/walk_forward_cv_results.csv), with 2014-16
#      as the folds of interest.
#   4. Also reports the standard train<=2015/valid/test(2018-19)
#      split for a direct before/after on the headline metrics.
#
# This is EXPERIMENTAL - it does not overwrite feature_list_v2.txt,
# master_dataset_v5, or xgb_tuned.pkl. It only writes a new,
# separately-named comparison file so the existing pipeline/model
# is untouched until you decide whether to adopt the result.
# =============================================================

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

TARGET = 'yield_kg_ha'

master = pd.read_csv('data/processed/master_dataset_v5_full_weather.csv')

# The 40 engineered features (state_encoded, lag_yield_1, etc.) live in
# train/valid/test_v2, not in the raw master dataset - load those and
# merge the lag lookup onto them instead of re-deriving everything.
train_eng = pd.read_csv('data/processed/train_v2.csv')
valid_eng = pd.read_csv('data/processed/valid_v2.csv')
test_eng = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    BASE_FEATURES = [l.strip() for l in f if l.strip()]

with open('outputs/metrics/xgb_best_params.json') as f:
    BEST_PARAMS = json.load(f)

# -------------------------------------------------------------
# 1. Build district-year-level lag features
# -------------------------------------------------------------
dy = (
    master[['district', 'year', 'flood_flag', 'drought_flag', 'rainfall_anomaly_index']]
    .drop_duplicates(subset=['district', 'year'])
    .sort_values(['district', 'year'])
)

dy['flood_flag_lag1'] = dy.groupby('district')['flood_flag'].shift(1)
dy['drought_flag_lag1'] = dy.groupby('district')['drought_flag'].shift(1)
dy['rai_lag1'] = dy.groupby('district')['rainfall_anomaly_index'].shift(1)

lag_cols = dy[['district', 'year', 'flood_flag_lag1', 'drought_flag_lag1', 'rai_lag1']]

# First year on record for each district has no prior year - fill
# with 0 (flood_flag_lag1/drought_flag_lag1) or 0.0 (rai_lag1,
# "no known anomaly") rather than leaving NaN, matching how the
# rest of the pipeline treats missing-history cases (e.g.
# lag_yield_1 for a district's first year).
lag_cols = lag_cols.fillna({'flood_flag_lag1': 0, 'drought_flag_lag1': 0, 'rai_lag1': 0.0})

full = pd.concat([train_eng, valid_eng, test_eng], ignore_index=True)
full = full.merge(lag_cols, on=['district', 'year'], how='left')
# Rows for years outside master's range (shouldn't happen here, but
# be defensive) fall back to the same "no known history" default.
full[['flood_flag_lag1', 'drought_flag_lag1', 'rai_lag1']] = full[
    ['flood_flag_lag1', 'drought_flag_lag1', 'rai_lag1']
].fillna({'flood_flag_lag1': 0, 'drought_flag_lag1': 0, 'rai_lag1': 0.0})

NEW_FEATURES = BASE_FEATURES + ['flood_flag_lag1', 'drought_flag_lag1', 'rai_lag1']

print(f"Base features: {len(BASE_FEATURES)}  ->  With lag features: {len(NEW_FEATURES)}")

# -------------------------------------------------------------
# 2. Walk-forward CV, baseline features vs. +lag features
# -------------------------------------------------------------
full = full.sort_values('year').reset_index(drop=True)
years = sorted(full['year'].unique())
MIN_TRAIN_YEARS = 10
test_years = [y for y in years if y >= years[0] + MIN_TRAIN_YEARS]

def walk_forward(features):
    rows = []
    for test_year in test_years:
        train_fold = full[full['year'] < test_year]
        test_fold = full[full['year'] == test_year]
        if len(test_fold) < 5:
            continue
        model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, verbosity=0)
        model.fit(train_fold[features], train_fold[TARGET])
        pred = model.predict(test_fold[features])
        rmse = np.sqrt(mean_squared_error(test_fold[TARGET], pred))
        mape = np.mean(np.abs((test_fold[TARGET] - pred) / test_fold[TARGET])) * 100
        r2 = r2_score(test_fold[TARGET], pred) if len(test_fold) > 1 else np.nan
        rows.append({'test_year': test_year, 'n_train': len(train_fold),
                      'n_test': len(test_fold), 'rmse': rmse, 'mape': mape, 'r2': r2})
    return pd.DataFrame(rows)

print("\nRunning walk-forward CV: baseline (40 features)...")
baseline_cv = walk_forward(BASE_FEATURES)

print("Running walk-forward CV: +3 lag features (43 features)...")
lagged_cv = walk_forward(NEW_FEATURES)

compare = baseline_cv[['test_year', 'r2', 'rmse', 'mape']].merge(
    lagged_cv[['test_year', 'r2', 'rmse', 'mape']],
    on='test_year', suffixes=('_baseline', '_with_lag')
)
compare['r2_delta'] = compare['r2_with_lag'] - compare['r2_baseline']

print("\n" + "=" * 78)
print("WALK-FORWARD COMPARISON: baseline vs. +lagged flood/drought features")
print("=" * 78)
print(compare.to_string(index=False, formatters={
    'r2_baseline': '{:.3f}'.format, 'r2_with_lag': '{:.3f}'.format,
    'r2_delta': '{:+.3f}'.format, 'rmse_baseline': '{:.1f}'.format,
    'rmse_with_lag': '{:.1f}'.format, 'mape_baseline': '{:.1f}'.format,
    'mape_with_lag': '{:.1f}'.format,
}))

print("\n2014-2016 folds specifically (the weak folds under investigation):")
print(compare[compare['test_year'].isin([2014, 2015, 2016])].to_string(index=False, formatters={
    'r2_baseline': '{:.3f}'.format, 'r2_with_lag': '{:.3f}'.format, 'r2_delta': '{:+.3f}'.format,
}))

out_path = 'outputs/metrics/lagged_feature_walk_forward_comparison.csv'
compare.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# -------------------------------------------------------------
# 3. Standard split (train<=2015 / valid 2016-17 / test 2018-19)
#    - direct before/after on the headline metrics
# -------------------------------------------------------------
train = full[full['year'] <= 2015]
valid = full[(full['year'] >= 2016) & (full['year'] <= 2017)]
test = full[(full['year'] >= 2018) & (full['year'] <= 2019)]

def fit_score(features, label):
    model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, verbosity=0)
    model.fit(pd.concat([train, valid])[features], pd.concat([train, valid])[TARGET])
    pred = model.predict(test[features])
    rmse = np.sqrt(mean_squared_error(test[TARGET], pred))
    mape = np.mean(np.abs((test[TARGET] - pred) / test[TARGET])) * 100
    r2 = r2_score(test[TARGET], pred)
    print(f"{label:30s}  RMSE={rmse:7.1f}  MAPE={mape:5.1f}%  R2={r2:.4f}")
    return {'model': label, 'rmse': rmse, 'mape': mape, 'r2': r2}

print("\n" + "=" * 78)
print("HEADLINE SPLIT (train+valid -> test 2018-19), matching xgb_tuned's own eval")
print("=" * 78)
res_base = fit_score(BASE_FEATURES, 'Baseline (40 features)')
res_lag = fit_score(NEW_FEATURES, '+3 lag features (43 features)')

pd.DataFrame([res_base, res_lag]).to_csv(
    'outputs/metrics/lagged_feature_headline_comparison.csv', index=False
)
print("Saved: outputs/metrics/lagged_feature_headline_comparison.csv")