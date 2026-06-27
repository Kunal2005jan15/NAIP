# =============================================================
# NAIP Project - Step 18: Retrain with Full Weather Coverage
# =============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

train = pd.read_csv('data/processed/train_v2.csv')
valid = pd.read_csv('data/processed/valid_v2.csv')
test  = pd.read_csv('data/processed/test_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

TARGET = 'yield_kg_ha'

X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")

def evaluate(name, y_true, y_pred):
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

# ---------------- XGBOOST ----------------
print("\n" + "="*60 + "\nMODEL 1: XGBoost (v2 - full weather)\n" + "="*60)

xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    early_stopping_rounds=30, eval_metric='rmse', verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(f"Best iteration: {xgb_model.best_iteration}")

xgb_pred_test = xgb_model.predict(X_test)
metrics_results = [evaluate("XGBoost v2 (Test)", y_test, xgb_pred_test)]

# ---------------- RANDOM FOREST ----------------
print("\n" + "="*60 + "\nMODEL 2: Random Forest (v2 - full weather)\n" + "="*60)

rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
X_trainval = pd.concat([X_train, X_valid])
y_trainval = pd.concat([y_train, y_valid])
rf_model.fit(X_trainval, y_trainval)
rf_pred_test = rf_model.predict(X_test)
metrics_results.append(evaluate("Random Forest v2 (Test)", y_test, rf_pred_test))

# ---------------- LIGHTGBM ----------------
print("\n" + "="*60 + "\nMODEL 3: LightGBM (v2 - full weather)\n" + "="*60)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    random_state=42, verbosity=-1,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
lgb_pred_test = lgb_model.predict(X_test)
metrics_results.append(evaluate("LightGBM v2 (Test)", y_test, lgb_pred_test))

# ---------------- SHAP ----------------
print("\n" + "="*60 + "\nSHAP EXPLAINABILITY (v2)\n" + "="*60)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap_importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

total_importance = shap_importance['importance'].sum()
print(f"\n  {'Feature':<25} {'Importance':>12} {'% of Total':>10}")
print(f"  {'-'*50}")
for _, row in shap_importance.iterrows():
    pct = row['importance'] / total_importance * 100
    bar = '█' * int(pct / 2)
    print(f"  {row['feature']:<25} {row['importance']:>10.1f}   {pct:>6.1f}%  {bar}")

# ---------------- COMPARISON TABLE ----------------
metrics_df = pd.DataFrame(metrics_results)
print("\n" + "="*60 + "\nMODEL COMPARISON (v2 - full weather)\n" + "="*60)
print(metrics_df.to_string(index=False))

# Compare to OLD results
print("\n" + "="*60)
print("BEFORE vs AFTER comparison")
print("="*60)
old_results = {
    'XGBoost':       {'rmse': 585.0, 'r2': 0.5785},
    'Random Forest': {'rmse': 333.3, 'r2': 0.8632},
    'LightGBM':      {'rmse': 584.0, 'r2': 0.5800},
}
for model_name, old in old_results.items():
    new_row = metrics_df[metrics_df['model'].str.contains(model_name.split()[0])]
    if len(new_row) > 0:
        new_r2 = new_row['r2'].values[0]
        new_rmse = new_row['rmse'].values[0]
        print(f"  {model_name:15s}  R²: {old['r2']:.4f} -> {new_r2:.4f}   "
              f"RMSE: {old['rmse']:.1f} -> {new_rmse:.1f}")

metrics_df.to_csv('outputs/metrics/model_comparison_v2.csv', index=False)
shap_importance.to_csv('outputs/metrics/shap_importance_v2.csv', index=False)

with open('outputs/models/xgb_model_v2.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('outputs/models/rf_model_v2.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n✓ v2 training complete.")