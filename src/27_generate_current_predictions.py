# =============================================================
# NAIP Project - Step 27: Generate Current-Conditions Predictions
# =============================================================
# Runs the tuned XGBoost model on the current_prediction_inputs.csv
# built in script 25 - producing genuine 2026 predictions for
# Wheat (Rabi), using REAL current weather + last-known yield trend.
# =============================================================

import pandas as pd
import numpy as np
import pickle
import shap
import json

print("Loading tuned model and current-conditions inputs...")

with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

current_inputs = pd.read_csv('data/processed/current_prediction_inputs.csv')

# Only Wheat rows have real current-season weather substituted
# (Rabi just completed). Rice/Kharif 2026 is still in-season,
# so we exclude it from "current predictions" for now - that
# would need to wait until Kharif completes (~October 2026).
wheat_current = current_inputs[current_inputs['crop'] == 'Wheat'].copy()
print(f"Wheat districts with current-season prediction ready: {len(wheat_current)}")

missing_features = [f for f in FEATURES if f not in wheat_current.columns]
if missing_features:
    print(f"WARNING: missing features in current inputs: {missing_features}")

X_current = wheat_current[FEATURES].astype(float)

print("\nGenerating predictions...")
predictions = model.predict(X_current)
wheat_current['current_pred_yield'] = predictions

# SHAP explanations for current predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_current)
wheat_current_shap = pd.DataFrame(shap_values, columns=[f'shap_{f}' for f in FEATURES])
wheat_current = pd.concat([wheat_current.reset_index(drop=True), wheat_current_shap], axis=1)

print(f"\nPrediction summary:")
print(f"  Mean predicted yield: {wheat_current['current_pred_yield'].mean():.0f} kg/ha")
print(f"  Districts predicted:  {len(wheat_current)}")
print(f"  Drought-flagged:      {wheat_current['drought_flag'].sum()}")
print(f"  Flood-flagged:        {wheat_current['flood_flag'].sum()}")

# Compare against last known actual (2019) to see directional shift
wheat_current['change_vs_2019'] = wheat_current['current_pred_yield'] - wheat_current['yield_kg_ha']
print(f"\n  Districts predicted HIGHER than 2019: {(wheat_current['change_vs_2019'] > 0).sum()}")
print(f"  Districts predicted LOWER than 2019:  {(wheat_current['change_vs_2019'] < 0).sum()}")
print(f"  Mean change vs 2019: {wheat_current['change_vs_2019'].mean():+.0f} kg/ha")

output_cols = ['state', 'district', 'crop', 'lat', 'lon', 'yield_kg_ha',
               'current_pred_yield', 'change_vs_2019', 'drought_flag', 'flood_flag',
               'rainfall_anomaly_index', 'baseline_yield_year', 'weather_data_as_of']
wheat_current[output_cols].to_csv('outputs/metrics/current_predictions_2026.csv', index=False)

# Full version with SHAP values for dashboard use
wheat_current.to_csv('data/processed/current_predictions_full.csv', index=False)

print(f"\nSaved outputs/metrics/current_predictions_2026.csv (summary)")
print(f"Saved data/processed/current_predictions_full.csv (with SHAP, for dashboard)")
print("\n✓ Current-conditions predictions generated for all Wheat districts.")
print("  Rice (Kharif 2026) predictions will be available once monsoon season completes (~Oct 2026).")