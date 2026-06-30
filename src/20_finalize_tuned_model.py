# =============================================================
# NAIP Project - Step 20: Finalize Tuned Model (SHAP + Intervals)
# =============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test  = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

TARGET = 'yield_kg_ha'

X_train = pd.concat([train, valid])[FEATURES]
y_train = pd.concat([train, valid])[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

# Load the tuned model and its best params
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    xgb_tuned = pickle.load(f)

with open('outputs/metrics/xgb_best_params.json') as f:
    best_params = json.load(f)

print("Using tuned hyperparameters:")
print(json.dumps(best_params, indent=2))

# ---------------------------------------------------------------
# SHAP ON TUNED MODEL
# ---------------------------------------------------------------

print("\n" + "="*60)
print("SHAP — Tuned XGBoost")
print("="*60)

explainer   = shap.TreeExplainer(xgb_tuned)
shap_values = explainer.shap_values(X_test)

shap_importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

total = shap_importance['importance'].sum()
print(f"\n  {'Feature':<25} {'Importance':>12} {'% of Total':>10}")
print(f"  {'-'*50}")
for _, row in shap_importance.iterrows():
    pct = row['importance'] / total * 100
    bar = '█' * int(pct / 2)
    print(f"  {row['feature']:<25} {row['importance']:>10.1f}   {pct:>6.1f}%  {bar}")

shap_importance.to_csv('outputs/metrics/shap_importance_TUNED.csv', index=False)

# ---------------------------------------------------------------
# PREDICTION INTERVALS USING TUNED HYPERPARAMETERS
# ---------------------------------------------------------------

print("\n" + "="*60)
print("Quantile Regression — Using Tuned Hyperparameters")
print("="*60)

import xgboost as xgb

quantile_params = {k: v for k, v in best_params.items()}

xgb_lower = xgb.XGBRegressor(
    **quantile_params,
    objective='reg:quantileerror',
    quantile_alpha=0.06,
    random_state=42, verbosity=0,
)
xgb_lower.fit(X_train, y_train)

# BUG FIX (2026-06-28): this script trains xgb_lower/xgb_upper on the
# correct v2 feature set, but previously never saved them to disk -
# so outputs/models/xgb_lower.pkl and xgb_upper.pkl were silently left
# as STALE leftovers from the pre-v2 pipeline (06_train_model.py,
# June 17), trained on old non-"nasa_"-prefixed feature names. Any
# code loading those files for live inference (e.g. the dashboard)
# would get a feature_names mismatch. Saving them properly now.
with open('outputs/models/xgb_lower.pkl', 'wb') as f:
    pickle.dump(xgb_lower, f)
print("Saved: outputs/models/xgb_lower.pkl (v2 features, quantile_alpha=0.06)")

xgb_upper = xgb.XGBRegressor(
    **quantile_params,
    objective='reg:quantileerror',
    quantile_alpha=0.94,
    random_state=42, verbosity=0,
)
xgb_upper.fit(X_train, y_train)

with open('outputs/models/xgb_upper.pkl', 'wb') as f:
    pickle.dump(xgb_upper, f)
print("Saved: outputs/models/xgb_upper.pkl (v2 features, quantile_alpha=0.94)")

pred_lower = xgb_lower.predict(X_test)
pred_upper = xgb_upper.predict(X_test)
pred_mid   = xgb_tuned.predict(X_test)

coverage  = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))
avg_width = np.mean(pred_upper - pred_lower)

print(f"\n  Prediction interval coverage: {coverage:.1%}")
print(f"  Average interval width:       {avg_width:.0f} kg/ha")
print(f"  Target: 70-90% coverage for an 80% interval")

# Compare to old (untuned) interval results
print(f"\n  BEFORE (untuned): coverage=55.1%, width=866 kg/ha")
print(f"  AFTER  (tuned):    coverage={coverage:.1%}, width={avg_width:.0f} kg/ha")

# ---------------------------------------------------------------
# SAVE FINAL PREDICTIONS TABLE
# ---------------------------------------------------------------

test_final = test.copy()
test_final['pred_yield']   = pred_mid
test_final['pred_lower']   = pred_lower
test_final['pred_upper']   = pred_upper
test_final['actual_yield'] = y_test.values
test_final['error']        = test_final['actual_yield'] - test_final['pred_yield']
test_final['error_pct']    = (test_final['error'] / test_final['actual_yield'] * 100)

test_final.to_csv('outputs/metrics/test_predictions_FINAL.csv', index=False)

# ---------------------------------------------------------------
# FINAL PLOTS
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, pred_mid, alpha=0.4, color='steelblue', s=20)
min_val, max_val = y_test.min(), y_test.max()
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Yield (kg/ha)')
axes[0].set_ylabel('Predicted Yield (kg/ha)')
axes[0].set_title(f'Tuned XGBoost: Actual vs Predicted\nR² = {shap_values.shape}')
from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred_mid)
axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
             fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].legend()

top10 = shap_importance.head(10)
axes[1].barh(top10['feature'][::-1], top10['importance'][::-1], color='steelblue')
axes[1].set_xlabel('Mean |SHAP Value| (kg/ha)')
axes[1].set_title('Top 10 Features — Tuned Model')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/03_tuned_model_final.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/plots/03_tuned_model_final.png")

print("\n✓ Day 3 model finalization complete.")