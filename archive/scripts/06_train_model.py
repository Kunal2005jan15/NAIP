# =============================================================
# NAIP Project - Step 6: Model Training
# =============================================================
# Models trained:
#   1. XGBoost (primary)
#   2. Random Forest (benchmark)
#   3. LightGBM (benchmark)
#
# Evaluation metrics:
#   - RMSE  (root mean squared error - main metric)
#   - MAE   (mean absolute error)
#   - MAPE  (mean absolute percentage error)
#   - R²    (coefficient of determination)
#
# Novel additions:
#   - Quantile regression for prediction intervals (10th/90th percentile)
#   - Time-correct validation (no data leakage)
#   - SHAP explainability
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

print("Loading data...")
train = pd.read_csv('data/processed/train.csv')
valid = pd.read_csv('data/processed/valid.csv')
test  = pd.read_csv('data/processed/test.csv')

FEATURES = [
    'state_encoded', 'crop_encoded', 'season_encoded', 'time_trend',
    'lag_yield_1', 'lag_yield_2', 'lag_yield_3',
    'rolling_yield_3yr', 'yield_trend', 'yield_gap_vs_state',
    'log_area',
    'fertilizer_per_ha', 'nitrogen_per_ha', 'irrigation_pct',
    'rainfall_kharif_mm', 'rainfall_rabi_mm',
    'temp_max_kharif', 'temp_max_rabi', 'evapotrans_kharif',
]

TARGET = 'yield_kg_ha'

X_train = train[FEATURES]
y_train = train[TARGET]
X_valid = valid[FEATURES]
y_valid = valid[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")

# ---------------------------------------------------------------
# METRICS FUNCTION
# ---------------------------------------------------------------

def evaluate(name, y_true, y_pred):
    """Compute and print all evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    
    print(f"\n  {name}:")
    print(f"    RMSE : {rmse:.1f} kg/ha")
    print(f"    MAE  : {mae:.1f} kg/ha")
    print(f"    MAPE : {mape:.1f}%")
    print(f"    R²   : {r2:.4f}")
    
    return {'model': name, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

# ---------------------------------------------------------------
# MODEL 1: XGBOOST (Primary Model)
# ---------------------------------------------------------------
# Why these hyperparameters?
# n_estimators=500    - enough trees without overfitting
# max_depth=6         - standard for tabular data
# learning_rate=0.05  - slow learning = better generalization
# subsample=0.8       - use 80% of rows per tree (prevents overfitting)
# colsample_bytree=0.8 - use 80% of features per tree
# early_stopping      - stop if validation doesn't improve for 30 rounds

print("\n" + "="*60)
print("MODEL 1: XGBoost")
print("="*60)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    random_state=42,
    early_stopping_rounds=30,
    eval_metric='rmse',
    verbosity=0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

print(f"Best iteration: {xgb_model.best_iteration}")

xgb_pred_valid = xgb_model.predict(X_valid)
xgb_pred_test  = xgb_model.predict(X_test)

metrics_results = []
print("\nVALIDATION SET (2016-2017):")
metrics_results.append(evaluate("XGBoost (Valid)", y_valid, xgb_pred_valid))
print("\nTEST SET (2018-2019) — FINAL:")
metrics_results.append(evaluate("XGBoost (Test)", y_test, xgb_pred_test))

# ---------------------------------------------------------------
# MODEL 2: RANDOM FOREST (Benchmark)
# ---------------------------------------------------------------

print("\n" + "="*60)
print("MODEL 2: Random Forest (Benchmark)")
print("="*60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# For RF we combine train+valid for fitting, test for evaluation
X_trainval = pd.concat([X_train, X_valid])
y_trainval = pd.concat([y_train, y_valid])

rf_model.fit(X_trainval, y_trainval)
rf_pred_test = rf_model.predict(X_test)

print("\nTEST SET (2018-2019):")
metrics_results.append(evaluate("Random Forest (Test)", y_test, rf_pred_test))

# ---------------------------------------------------------------
# MODEL 3: LIGHTGBM (Benchmark)
# ---------------------------------------------------------------

print("\n" + "="*60)
print("MODEL 3: LightGBM (Benchmark)")
print("="*60)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    random_state=42,
    verbosity=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(30, verbose=False)]
)

lgb_pred_test = lgb_model.predict(X_test)

print("\nTEST SET (2018-2019):")
metrics_results.append(evaluate("LightGBM (Test)", y_test, lgb_pred_test))

# ---------------------------------------------------------------
# MODEL 4: XGBOOST QUANTILE REGRESSION (Prediction Intervals)
# ---------------------------------------------------------------
# This is your novel contribution #2
# We train two additional models:
#   - Lower bound: predicts 10th percentile (worst case)
#   - Upper bound: predicts 90th percentile (best case)
# Together they give an 80% prediction interval

print("\n" + "="*60)
print("MODEL 4: Quantile Regression (Prediction Intervals)")
print("="*60)

# Lower bound model (10th percentile = worst case scenario)
xgb_lower = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:quantileerror',
    quantile_alpha=0.10,   # 10th percentile
    random_state=42,
    verbosity=0,
)
xgb_lower.fit(X_train, y_train)

# Upper bound model (90th percentile = best case scenario)
xgb_upper = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:quantileerror',
    quantile_alpha=0.90,   # 90th percentile
    random_state=42,
    verbosity=0,
)
xgb_upper.fit(X_train, y_train)

pred_lower = xgb_lower.predict(X_test)
pred_upper = xgb_upper.predict(X_test)
pred_mid   = xgb_model.predict(X_test)

# Coverage: what % of actual values fall within our interval?
coverage = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))
avg_width = np.mean(pred_upper - pred_lower)

print(f"\n  Prediction interval coverage: {coverage:.1%}")
print(f"  Average interval width:       {avg_width:.0f} kg/ha")
print(f"  (Good coverage = 70-90% for 80% interval)")

# Show example predictions with intervals
test_sample = test.copy()
test_sample['pred_yield']  = pred_mid
test_sample['pred_lower']  = pred_lower
test_sample['pred_upper']  = pred_upper
test_sample['actual_yield'] = y_test.values

print(f"\n  Sample predictions with intervals:")
sample_display = test_sample[
    ['district', 'crop', 'year', 'actual_yield', 
     'pred_lower', 'pred_yield', 'pred_upper']
].head(8)
print(sample_display.to_string(index=False))

# ---------------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------------

print("\n" + "="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

print("Computing SHAP values (this takes ~30 seconds)...")

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP = global feature importance
shap_importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n  SHAP Feature Importance (Impact on yield prediction):")
print(f"  {'Feature':<25} {'Importance':>12} {'% of Total':>10}")
print(f"  {'-'*50}")
total_importance = shap_importance['importance'].sum()
for _, row in shap_importance.iterrows():
    pct = row['importance'] / total_importance * 100
    bar = '█' * int(pct / 2)
    print(f"  {row['feature']:<25} {row['importance']:>10.1f}   {pct:>6.1f}%  {bar}")

# ---------------------------------------------------------------
# PLOT 1: Actual vs Predicted
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Scatter plot
axes[0].scatter(y_test, xgb_pred_test, alpha=0.4, color='steelblue', s=20)
min_val = min(y_test.min(), xgb_pred_test.min())
max_val = max(y_test.max(), xgb_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Yield (kg/ha)', fontsize=12)
axes[0].set_ylabel('Predicted Yield (kg/ha)', fontsize=12)
axes[0].set_title('XGBoost: Actual vs Predicted\n(Test Set 2018-2019)', fontsize=13)
axes[0].legend()
r2 = r2_score(y_test, xgb_pred_test)
axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: SHAP bar chart
top_features = shap_importance.head(10)
axes[1].barh(top_features['feature'][::-1], 
             top_features['importance'][::-1], 
             color='steelblue')
axes[1].set_xlabel('Mean |SHAP Value| (kg/ha)', fontsize=12)
axes[1].set_title('Top 10 Feature Importance\n(SHAP Values)', fontsize=13)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/01_model_performance.png', dpi=150, bbox_inches='tight')
print("\n  Plot saved: outputs/plots/01_model_performance.png")

# ---------------------------------------------------------------
# PLOT 2: Prediction Intervals for sample districts
# ---------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

# Show one district's yield over time with intervals
sample_district = test_sample[
    (test_sample['district'] == test_sample['district'].iloc[0]) &
    (test_sample['crop'] == 'Wheat')
]

if len(sample_district) > 0:
    years = sample_district['year']
    ax.fill_between(years, 
                    sample_district['pred_lower'],
                    sample_sample['pred_upper'] if False else sample_district['pred_upper'],
                    alpha=0.3, color='steelblue', label='80% Prediction Interval')
    ax.plot(years, sample_district['pred_yield'], 
            'b-o', linewidth=2, label='Predicted Yield')
    ax.plot(years, sample_district['actual_yield'], 
            'r-s', linewidth=2, label='Actual Yield')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Yield (kg/ha)', fontsize=12)
    ax.set_title(f"Prediction Intervals — {sample_district['district'].iloc[0]} Wheat", 
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient district data for interval plot',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('outputs/plots/02_prediction_intervals.png', dpi=150, bbox_inches='tight')
print("  Plot saved: outputs/plots/02_prediction_intervals.png")

# ---------------------------------------------------------------
# SAVE METRICS AND MODELS
# ---------------------------------------------------------------

metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv('outputs/metrics/model_comparison.csv', index=False)

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(metrics_df.to_string(index=False))

# Save predictions with intervals
test_sample.to_csv('outputs/metrics/test_predictions.csv', index=False)

# Save SHAP importance
shap_importance.to_csv('outputs/metrics/shap_importance.csv', index=False)

import pickle
with open('outputs/models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('outputs/models/xgb_lower.pkl', 'wb') as f:
    pickle.dump(xgb_lower, f)
with open('outputs/models/xgb_upper.pkl', 'wb') as f:
    pickle.dump(xgb_upper, f)

print("\n✓ Models saved to outputs/models/")
print("✓ Training complete.")