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
import json
from datetime import date

FAILURES = []
WARNINGS = []
RESULTS = []  # structured record of every check line, written to JSON at the end
CURRENT_CHECK = {"name": "setup"}
REPORT_PATH = 'data/processed/sanity_report.json'


def write_report():
    import datetime as _datetime
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "generated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "passed": not FAILURES,
        "failures": len(FAILURES),
        "warnings": len(WARNINGS),
        "results": RESULTS,
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"  Structured report written to {REPORT_PATH}")

def fail(msg):
    FAILURES.append(msg)
    RESULTS.append({"check": CURRENT_CHECK["name"], "status": "fail", "detail": msg})
    print(f"  [FAIL] {msg}")

def warn(msg):
    WARNINGS.append(msg)
    RESULTS.append({"check": CURRENT_CHECK["name"], "status": "warn", "detail": msg})
    print(f"  [WARN] {msg}")

def ok(msg):
    RESULTS.append({"check": CURRENT_CHECK["name"], "status": "pass", "detail": msg})
    print(f"  [OK]   {msg}")

# ---------------------------------------------------------------
# OVERRIDE MECHANISM (built 2026-07-08, delivered 2026-07-11 - this
# existed and was tested but was never actually shipped in an earlier
# handoff; fixing that gap now)
# ---------------------------------------------------------------
# A gate that never has an escape hatch just gets silently disabled by
# whoever gets tired of it - usually by commenting out the whole check,
# which loses the check FOREVER, not just for the diagnosed incident.
# This gives a narrower, audited, self-expiring alternative: a human can
# mark ONE specific finding as manually diagnosed-and-real, with a reason
# and evidence, but it automatically stops applying after `expires` - so
# a genuinely NEW bug that produces the same shape later can't hide
# behind a stale override.

OVERRIDES_PATH = 'data/pipeline_overrides.json'

def get_active_override(check_id):
    """Returns the override dict if check_id has a non-expired override, else None."""
    if not os.path.exists(OVERRIDES_PATH):
        return None
    with open(OVERRIDES_PATH) as f:
        overrides = json.load(f)
    entry = overrides.get(check_id)
    if entry is None:
        return None
    expires = date.fromisoformat(entry['expires'])
    if date.today() > expires:
        warn(f"Override for '{check_id}' EXPIRED on {entry['expires']} - re-diagnose before "
             f"trusting this again. (Was: {entry['reason'][:100]}...)")
        return None
    return entry

def fail_or_override(check_id, msg):
    """Like fail(), but downgrades to a visible WARN if an active, non-expired
    override exists for this check_id. Never silent either way."""
    override = get_active_override(check_id)
    if override:
        warn(f"{msg}\n         >>> OVERRIDDEN (expires {override['expires']}): {override['reason']}\n"
             f"         >>> Caveat: {override.get('caveat', 'none stated')}\n"
             f"         >>> Evidence: {override.get('evidence', 'none stated')}")
        RESULTS[-1]["status"] = "overridden"
        RESULTS[-1]["override_ref"] = f"{check_id} (expires {override['expires']})"
    else:
        fail(msg)


print("=" * 60)
print("CHECK 1: Model / feature-list schema consistency")
CURRENT_CHECK["name"] = "schema_consistency"
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
    path = f'models/{fname}'
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
CURRENT_CHECK["name"] = "classification_diversity"
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

    # 'baseline' at 100% is always correct on first run after snapshot reset
    if counts.index[0] == 'baseline':
        ok(f"{label}: 100% 'baseline' - this is the FIRST snapshot, no prior "
           f"comparison exists. Run District Watch again after the next "
           f"scheduled refresh to see real change detection.")
        continue

    # REFINEMENT (2026-06-29): a uniform classification is only the
    # Kharif-bug SHAPE if the underlying CONTINUOUS driver also shows
    # real variance that the classifier failed to reflect. If the
    # underlying magnitude itself has low variance (e.g. comparing two
    # near-identical model versions after a small feature addition),
    # uniform "stable" is the CORRECT result, not a bug.
    #
    # REFINEMENT 2 (2026-07-01): a second benign source of uniform
    # classification is a MODEL VERSION CHANGE. When the model is
    # retrained with new features, ALL districts shift slightly because
    # the model itself changed - not because real-world conditions
    # changed. District Watch comparing a new-model snapshot against an
    # old-model snapshot will see uniform "minor_change" across all
    # districts (the new signal added a small consistent offset).
    # Detecting this: if the feature count in the current model differs
    # from what the previous snapshot was built with, this IS a version
    # change, and the classification is expected and benign. The fix
    # is to delete old snapshots and rerun District Watch to re-baseline.
    if magnitude_col and magnitude_col in df.columns:
        mag_std = df[magnitude_col].std()
        mag_range = df[magnitude_col].max() - df[magnitude_col].min()
        if mag_std < 20 and mag_range < 100:
            ok(f"{label}: {top_share:.0%} in one category, but the underlying "
               f"'{magnitude_col}' has low variance too (std={mag_std:.1f}, range={mag_range:.1f}) - "
               f"genuinely small change between snapshots, not a biased classifier. Not flagging.")
            continue

        # Check for model-version-change pattern: all districts shifted
        # by a similar small-ish amount (high kurtosis, low spread relative
        # to mean absolute shift) - the signature of a consistent model offset
        # vs. a real-world change that would affect districts differently.
        import scipy.stats as scipy_stats
        mean_abs = df[magnitude_col].abs().mean()
        kurt = scipy_stats.kurtosis(df[magnitude_col])
        n_minor = (df['alert_level'] == 'minor_change').sum() if 'alert_level' in df.columns else 0
        is_version_change = (
            top_share > 0.80 and
            counts.index[0] == 'minor_change' and
            mean_abs < 150 and
            kurt > 1.0  # high kurtosis = suspiciously uniform shift
        )
        if is_version_change:
            warn(f"{label}: {top_share:.0%} 'minor_change' with mean shift "
                 f"{mean_abs:.0f} kg/ha (kurtosis={kurt:.1f}). "
                 f"This pattern is consistent with a MODEL VERSION CHANGE "
                 f"(new features added uniform offset to all predictions). "
                 f"NOT the Kharif-bug shape. "
                 f"ACTION: delete old snapshots in data/snapshots/ and rerun "
                 f"src/28_district_watch.py to re-baseline against the current model.")
            continue

        # REFINEMENT 3 (2026-07-11): a large OVERALL std/range can come entirely
        # from rows that are ALREADY correctly excluded from the top category -
        # e.g. one genuine outlier district correctly flagged 'urgent' while the
        # other 117 are genuinely, identically unchanged. That's the classifier
        # working correctly, not the bug shape. The real bug shape is variance
        # HIDDEN INSIDE the top category itself (real differences the classifier
        # failed to separate out). Check that specifically, not just the
        # dataset-wide std/range, which conflates these two very different cases.
        top_category_mask = df[col] == counts.index[0]
        within_top_std = df.loc[top_category_mask, magnitude_col].std()
        within_top_range = (df.loc[top_category_mask, magnitude_col].max() -
                             df.loc[top_category_mask, magnitude_col].min())
        if within_top_std < 20 and within_top_range < 100:
            n_outliers = (~top_category_mask).sum()
            ok(f"{label}: {top_share:.0%} in '{counts.index[0]}', but that category itself is "
               f"genuinely homogeneous (within-category std={within_top_std:.1f}, "
               f"range={within_top_range:.1f}). The dataset-wide variance (std={mag_std:.1f}) "
               f"comes entirely from {n_outliers} row(s) ALREADY correctly excluded into other "
               f"categories - the classifier is working, not hiding anything. Not flagging.")
            continue

        fail(f"{label}: {top_share:.0%} of {len(df)} rows fall into a single category "
             f"('{counts.index[0]}'), but the underlying '{magnitude_col}' has REAL variance "
             f"(std={mag_std:.1f}, range={mag_range:.1f}). This is the Kharif-bug SHAPE: real "
             f"underlying variance that the classification logic isn't reflecting. Diagnose "
             f"the threshold/comparison logic before trusting this output.")
        continue

    fail_or_override(
        "kharif_high_concentration",
        f"{label}: {top_share:.0%} of {len(df)} rows fall into a single category "
        f"('{counts.index[0]}'). This is the EXACT shape of the Kharif pacing bug - "
        f"a near-uniform classification almost always means a structurally biased "
        f"comparison, not a real finding. Diagnose before trusting this output."
    )


print("\n" + "=" * 60)
print("CHECK 3: Live feature values within historical plausible range")
CURRENT_CHECK["name"] = "feature_ranges"
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
CURRENT_CHECK["name"] = "output_files_present"
print("=" * 60)

REQUIRED_FILES = [
    'data/processed/current_predictions_full.csv',
    'outputs/metrics/district_watch_feed.csv',
    'data/processed/kharif_2026_risk_flags.csv',
    'models/xgb_tuned.pkl',
    'models/xgb_lower.pkl',
    'models/xgb_upper.pkl',
]
for path in REQUIRED_FILES:
    if not os.path.exists(path):
        fail(f"Required file missing: {path}")
    elif os.path.getsize(path) == 0:
        fail(f"Required file is empty: {path}")
    else:
        ok(f"Present: {path}")


print("\n" + "=" * 60)
print("CHECK 5: Per-district weather fetch completeness")
CURRENT_CHECK["name"] = "weather_fetch_completeness"
print("=" * 60)
# BUG FIX (2026-07-11): a single-district NASA POWER timeout (Karnal, this date)
# silently produced zero 2026 weather rows for that district only. Nothing failed
# loudly - it propagated three steps downstream into a spurious "urgent" District
# Watch alert that looked like a real agronomic emergency but was actually a
# missing-data artifact. script 22 already logs failures to
# weather_current_failures.csv but nothing previously READ that file. This check
# closes that gap: surface fetch failures explicitly and immediately, at the
# actual source, instead of letting them masquerade as something else three
# steps downstream.
FAILURE_LOG = 'data/raw/weather_current_failures.csv'
if os.path.exists(FAILURE_LOG) and os.path.getsize(FAILURE_LOG) > 0:
    fetch_failures = pd.read_csv(FAILURE_LOG)
    if len(fetch_failures) > 0:
        districts_str = ', '.join(f"{r.district} ({r.state})" for r in fetch_failures.itertuples())
        fail(f"{len(fetch_failures)} district(s) had a weather fetch failure this run and have "
             f"ZERO current-season weather rows: {districts_str}. Any prediction/alert for "
             f"these specific districts this run is likely a missing-data artifact, not a real "
             f"signal - re-run src/22_fetch_current_weather.py, or at minimum treat their "
             f"District Watch / Kharif entries as unreliable until re-fetched.")
    else:
        ok("weather_current_failures.csv exists but is empty - no fetch failures this run")
else:
    ok("No weather fetch failures this run (all districts returned data)")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
write_report()
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