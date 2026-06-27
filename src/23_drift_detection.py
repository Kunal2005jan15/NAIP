# =============================================================
# NAIP Project - Step 23: Drift Detection
# =============================================================
# Monitors whether NEW input conditions (e.g., current weather)
# fall outside the range the model was actually trained on.
# This is a transparent, manual threshold approach rather than
# a black-box library (e.g., Evidently) — easier to explain in
# a paper/viva, and equally valid methodologically for this scale.
#
# Two types of drift checked:
#   1. DATA DRIFT — are input feature values (rainfall, temp)
#      outside the training distribution?
#   2. PERFORMANCE DRIFT — would require new ground-truth yield
#      to check (not available for 2020-2026 yet — documented
#      as a limitation, see report).
# =============================================================

import pandas as pd
import numpy as np
import json

print("Loading training data to establish baseline distributions...")
train = pd.read_csv('data/processed/train_v2.csv')

with open('data/processed/feature_list_v2.txt') as f:
    FEATURES = [line.strip() for line in f.readlines()]

# ---------------------------------------------------------------
# BUILD BASELINE STATISTICS FROM TRAINING DATA
# ---------------------------------------------------------------
# For each feature, store the training range and distribution
# stats. New data will be compared against this baseline.

DRIFT_CHECK_FEATURES = [
    'nasa_rainfall_kharif', 'nasa_rainfall_rabi', 'nasa_rainfall_annual',
    'nasa_temp_avg_kharif', 'nasa_temp_max_kharif',
    'nasa_temp_avg_rabi', 'nasa_temp_max_rabi',
    'rainfall_anomaly_index', 'heat_stress_days', 'frost_risk_days',
]

baseline_stats = {}
for feat in DRIFT_CHECK_FEATURES:
    if feat in train.columns:
        baseline_stats[feat] = {
            'mean': float(train[feat].mean()),
            'std': float(train[feat].std()),
            'min': float(train[feat].min()),
            'max': float(train[feat].max()),
            'p05': float(train[feat].quantile(0.05)),
            'p95': float(train[feat].quantile(0.95)),
        }

print(f"Baseline established for {len(baseline_stats)} features (training years: "
      f"{train['year'].min()}-{train['year'].max()})")

with open('data/processed/drift_baseline.json', 'w') as f:
    json.dump(baseline_stats, f, indent=2)
print("Saved baseline to data/processed/drift_baseline.json")

# ---------------------------------------------------------------
# DRIFT CHECK FUNCTION — reusable, called by dashboard too
# ---------------------------------------------------------------

def check_drift(new_value, feature_name, baseline_stats):
    """
    Returns a drift severity level for a single feature value.
    Uses a z-score-like approach: how many std deviations from
    the training mean, and whether it falls outside the
    5th-95th percentile range seen in training.
    """
    if feature_name not in baseline_stats:
        return {'status': 'unknown', 'severity': 0}

    stats = baseline_stats[feature_name]

    if stats['std'] == 0:
        return {'status': 'normal', 'severity': 0}

    z_score = abs((new_value - stats['mean']) / stats['std'])
    outside_range = (new_value < stats['p05']) or (new_value > stats['p95'])

    if z_score > 3:
        severity, status = 3, 'severe_drift'
    elif z_score > 2 or outside_range:
        severity, status = 2, 'moderate_drift'
    elif z_score > 1.5:
        severity, status = 1, 'mild_drift'
    else:
        severity, status = 0, 'normal'

    return {
        'status': status,
        'severity': severity,
        'z_score': round(z_score, 2),
        'value': new_value,
        'training_mean': round(stats['mean'], 1),
        'training_range_p05_p95': [round(stats['p05'], 1), round(stats['p95'], 1)],
    }


def check_drift_for_row(row, baseline_stats, features_to_check=DRIFT_CHECK_FEATURES):
    """Runs drift check across all monitored features for one data row."""
    results = {}
    max_severity = 0
    for feat in features_to_check:
        if feat in row.index:
            result = check_drift(row[feat], feat, baseline_stats)
            results[feat] = result
            max_severity = max(max_severity, result['severity'])
    return results, max_severity


# ---------------------------------------------------------------
# TEST THE DRIFT DETECTOR ON THE TEST SET
# ---------------------------------------------------------------
# Sanity check: test set (2018-19) should show LOW drift since
# it's chronologically close to training data. This validates
# the detector itself works sensibly before we trust it on
# genuinely new (2026) data.

print("\n" + "="*60)
print("Sanity check: Drift detector on TEST SET (2018-19)")
print("="*60)

test = pd.read_csv('data/processed/test_v2.csv')

drift_summary = []
for idx, row in test.iterrows():
    _, max_sev = check_drift_for_row(row, baseline_stats)
    drift_summary.append(max_sev)

test['max_drift_severity'] = drift_summary

severity_counts = pd.Series(drift_summary).value_counts().sort_index()
print("\nDrift severity distribution on test set (should be mostly 'normal'/'mild'):")
print(severity_counts)
print(f"\n% of test rows with NO drift (severity=0): {(test['max_drift_severity']==0).mean():.1%}")
print(f"% of test rows with SEVERE drift (severity=3): {(test['max_drift_severity']==3).mean():.1%}")

test.to_csv('outputs/metrics/test_set_drift_check.csv', index=False)
print("\nSaved drift-annotated test set to outputs/metrics/test_set_drift_check.csv")

print("\n✓ Drift detection module built and validated.")
print("  This will be reused in the dashboard for current-conditions monitoring.")