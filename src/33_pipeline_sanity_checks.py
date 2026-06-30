# =============================================================
# NAIP Project - Step 33: Automated Pipeline Sanity Checks
# =============================================================
# WHY THIS EXISTS: three real bugs in this project were caught by
# manual diagnosis, one at a time, after the fact:
#   1. Kharif early-warning pacing bug (117/118 districts "high
#      risk" simultaneously - a structurally biased comparison)
#   2. Stale xgb_lower.pkl/xgb_upper.pkl quantile models (trained
#      on an old pre-v2 feature schema, silently mismatched)
#   3. Rabi season-window misalignment (missing the actual
#      harvest-period weather for the labeled crop-year)
#
# Every one of these has a SHAPE: a check that's cheap to automate
# and would have flagged it in seconds instead of requiring a
# multi-step manual investigation. This script is that automated
# gate - run it after regenerating ANY model or prediction output,
# before trusting it in the dashboard, the paper, or a deployment.
#
# It does not replace judgment - it catches the class of mistake
# that judgment already caught manually, so it doesn't have to be
# re-caught manually every time the pipeline changes.
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
import sys

FAILURES = []
WARNINGS = []

def fail(msg):
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")

def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")

def ok(msg):
    print(f"  [OK]   {msg}")


print("=" * 60)
print("CHECK 1: Model / feature-list schema consistency")
print("=" * 60)
# Catches bug #2 (stale quantile models): any model file whose
# feature_names don't EXACTLY match the current feature_list_v2.txt
# is either stale or trained on a different schema - never silently
# use it for live inference.

with open('data/processed/feature_list_v2.txt') as f:
    current_features = [l.strip() for l in f.readlines()]

model_files = {
    'xgb_tuned.pkl': 'main tuned model',
    'xgb_lower.pkl': 'lower quantile model',
    'xgb_upper.pkl': 'upper quantile model',
}

for fname, label in model_files.items():
    path = f'outputs/models/{fname}'
    if not os.path.exists(path):
        warn(f"{label} ({fname}) not found - skipping schema check")
        continue
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model_features = None
    if hasattr(model, 'get_booster'):
        try:
            model_features = model.get_booster().feature_names
        except Exception:
            pass
    if model_features is None:
        warn(f"{label} ({fname}): could not introspect feature_names - cannot verify schema")
        continue
    if model_features != current_features:
        missing = set(current_features) - set(model_features)
        extra = set(model_features) - set(current_features)
        fail(f"{label} ({fname}): feature schema MISMATCH with current feature_list_v2.txt. "
             f"Missing from model: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
             f"Extra in model: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}. "
             f"This model is STALE - retrain before using it for live inference.")
    else:
        ok(f"{label} ({fname}): schema matches current feature list ({len(current_features)} features)")


print("\n" + "=" * 60)
print("CHECK 2: Classification diversity (catches the Kharif-bug shape)")
print("=" * 60)
# Catches bug #1: any time a categorical risk/alert output has
# >85% of rows in a SINGLE bucket, that's either a genuinely
# extreme real-world event (rare) or - far more likely - a
# systematic bug in the comparison logic. Either way, it should
# stop and get a human look before being trusted.

DIVERSITY_CHECKS = [
    ('data/processed/kharif_2026_risk_flags.csv', 'kharif_risk_level', 'Kharif early-warning risk', None),
    ('outputs/metrics/district_watch_feed.csv', 'alert_level', 'District Watch alert', 'yield_change_since_last'),
]

for path, col, label, magnitude_col in DIVERSITY_CHECKS:
    if not os.path.exists(path):
        warn(f"{label} file not found ({path}) - skipping diversity check")
        continue
    df = pd.read_csv(path)
    if col not in df.columns:
        warn(f"{label}: column '{col}' not found in {path}")
        continue
    counts = df[col].value_counts(normalize=True)
    top_share = counts.iloc[0]

    if top_share <= 0.85 or len(df) <= 10:
        ok(f"{label}: distribution looks plausible (top category = {top_share:.0%} of {len(df)} rows)")
        continue

    # REFINEMENT (2026-06-29): a uniform classification is only the
    # Kharif-bug SHAPE if the underlying CONTINUOUS driver also shows
    # real variance that the classifier failed to reflect. If the
    # underlying magnitude itself has low variance (e.g. comparing two
    # near-identical model versions after a small feature addition),
    # uniform "stable" is the CORRECT result, not a bug - this exact
    # case came up after adding NDVI (small SHAP contribution -> most
    # districts genuinely didn't move enough to cross any threshold).
    if magnitude_col and magnitude_col in df.columns:
        mag_std = df[magnitude_col].std()
        mag_range = df[magnitude_col].max() - df[magnitude_col].min()
        if mag_std < 20 and mag_range < 100:
            ok(f"{label}: {top_share:.0%} in one category, but the underlying "
               f"'{magnitude_col}' has low variance too (std={mag_std:.1f}, range={mag_range:.1f}) - "
               f"genuinely small change between snapshots, not a biased classifier. Not flagging.")
            continue
        else:
            fail(f"{label}: {top_share:.0%} of {len(df)} rows fall into a single category "
                 f"('{counts.index[0]}'), but the underlying '{magnitude_col}' has REAL variance "
                 f"(std={mag_std:.1f}, range={mag_range:.1f}). This is the Kharif-bug SHAPE: real "
                 f"underlying variance that the classification logic isn't reflecting. Diagnose "
                 f"the threshold/comparison logic before trusting this output.")
            continue

    fail(f"{label}: {top_share:.0%} of {len(df)} rows fall into a single category "
         f"('{counts.index[0]}'). This is the EXACT shape of the Kharif pacing bug - "
         f"a near-uniform classification almost always means a structurally biased "
         f"comparison, not a real finding. Diagnose before trusting this output.")


print("\n" + "=" * 60)
print("CHECK 3: Live feature values within historical plausible range")
print("=" * 60)
# Catches the general class of "live data pipeline silently broke"
# (e.g. a future fill-value bug, a bad NASA POWER pull, a unit
# error). Compares each live numeric feature's current value against
# the historical training distribution per district; flags if an
# implausible SHARE of districts are simultaneously far outside
# their own historical range - the same "too uniform to be real"
# logic as Check 2, applied to raw inputs instead of model outputs.

hist_path = 'data/processed/model_ready_v2.csv'
live_path = 'data/processed/current_predictions_full.csv'

if os.path.exists(hist_path) and os.path.exists(live_path):
    hist = pd.read_csv(hist_path)
    live = pd.read_csv(live_path)

    check_features = ['nasa_rainfall_rabi', 'nasa_temp_avg_rabi', 'gdd_rabi',
                       'max_dry_streak_rabi', 'water_balance_rabi', 'rainfall_anomaly_index']
    check_features = [c for c in check_features if c in hist.columns and c in live.columns]

    flagged_features = []
    for feat in check_features:
        hist_mean, hist_std = hist[feat].mean(), hist[feat].std()
        if hist_std == 0 or pd.isna(hist_std):
            continue
        z = (live[feat] - hist_mean) / hist_std
        extreme_share = (z.abs() > 3).mean()
        if extreme_share > 0.5:
            flagged_features.append((feat, extreme_share))

    if flagged_features:
        for feat, share in flagged_features:
            fail(f"'{feat}': {share:.0%} of live districts are >3 std from the historical "
                 f"mean SIMULTANEOUSLY. Check the live data pull before trusting predictions "
                 f"built on this feature.")
    else:
        ok(f"All {len(check_features)} checked live features fall within plausible "
           f"historical ranges across districts")
else:
    warn("Historical or live prediction file not found - skipping feature-range check")


print("\n" + "=" * 60)
print("CHECK 4: Required output files present and non-empty")
print("=" * 60)

REQUIRED_FILES = [
    'data/processed/current_predictions_full.csv',
    'outputs/metrics/district_watch_feed.csv',
    'data/processed/kharif_2026_risk_flags.csv',
    'outputs/models/xgb_tuned.pkl',
    'outputs/models/xgb_lower.pkl',
    'outputs/models/xgb_upper.pkl',
]
for path in REQUIRED_FILES:
    if not os.path.exists(path):
        fail(f"Required file missing: {path}")
    elif os.path.getsize(path) == 0:
        fail(f"Required file is empty: {path}")
    else:
        ok(f"Present: {path}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
if FAILURES:
    print("\n  FAILED CHECKS:")
    for f in FAILURES:
        print(f"    - {f}")
    print("\n  Do NOT trust the flagged outputs in the dashboard, paper, or a deployment")
    print("  until these are resolved.")
    sys.exit(1)
else:
    print("\n  All checks passed. Outputs are safe to use.")
    sys.exit(0)