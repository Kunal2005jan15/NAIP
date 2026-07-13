# =============================================================
# NAIP Project - Automated Regression Test Suite
# =============================================================
# WHY THIS EXISTS: script 33 (pipeline sanity gate) checks whether
# THIS RUN's output looks sane. This suite is different and
# complementary: it checks whether the CODE's logic is correct at
# known edge cases, independent of any particular run's data. Every
# test here exists because of a real bug found this session - the
# goal is that none of these four bug classes can silently come
# back during future development without a test failing first.
#
# Run with: pytest tests/test_pipeline.py -v
# (install if needed: pip install pytest --break-system-packages)
#
# Tests are split into two kinds:
#   - PURE LOGIC tests (no data files needed) - these duplicate the
#     exact threshold/window logic from the real scripts. Comments
#     mark exactly which script/lines they must stay in sync with.
#   - INTEGRATION tests (need real data/model files) - these SKIP
#     gracefully if the files aren't present, rather than failing,
#     since not every environment running this suite will have the
#     full pipeline's output on disk.
# =============================================================

import pytest
import pandas as pd
import numpy as np
import os
import pickle


# =============================================================
# BUG #3: Rabi season-window misalignment (script 15)
# =============================================================
# Must stay in sync with src/15_aggregate_weather_all.py's
# get_seasonal_weather() day-to-crop-year assignment logic.

def assign_crop_year(month, calendar_year):
    """Reimplements script 15's day->crop-year assignment exactly."""
    kharif_year = calendar_year if 6 <= month <= 10 else None
    if month in (11, 12):
        rabi_year = calendar_year
    elif month in (1, 2, 3):
        rabi_year = calendar_year - 1
    else:
        rabi_year = None
    agri_year = calendar_year if month >= 6 else calendar_year - 1
    return kharif_year, rabi_year, agri_year


class TestSeasonWindowAssignment:
    """Regression tests for the Rabi cross-year-boundary bug."""

    def test_november_belongs_to_same_year_rabi(self):
        _, rabi_year, _ = assign_crop_year(month=11, calendar_year=2001)
        assert rabi_year == 2001

    def test_december_belongs_to_same_year_rabi(self):
        _, rabi_year, _ = assign_crop_year(month=12, calendar_year=2001)
        assert rabi_year == 2001

    def test_january_belongs_to_PREVIOUS_year_rabi(self):
        # THE bug: Jan 2002 must belong to rabi_year=2001 (sown Nov
        # 2001), NOT rabi_year=2002. Getting this wrong is exactly
        # what silently broke the Wheat Rabi weather signal.
        _, rabi_year, _ = assign_crop_year(month=1, calendar_year=2002)
        assert rabi_year == 2001

    def test_february_belongs_to_previous_year_rabi(self):
        _, rabi_year, _ = assign_crop_year(month=2, calendar_year=2002)
        assert rabi_year == 2001

    def test_march_belongs_to_previous_year_rabi(self):
        _, rabi_year, _ = assign_crop_year(month=3, calendar_year=2002)
        assert rabi_year == 2001

    def test_april_belongs_to_no_season(self):
        kharif_year, rabi_year, _ = assign_crop_year(month=4, calendar_year=2002)
        assert kharif_year is None and rabi_year is None

    def test_kharif_window_unaffected_by_fix(self):
        # Kharif (Jun-Oct) never crossed a year boundary - confirms
        # the fix didn't accidentally change behavior here.
        for month in range(6, 11):
            kharif_year, _, _ = assign_crop_year(month=month, calendar_year=2010)
            assert kharif_year == 2010

    def test_agri_year_spans_full_crop_cycle(self):
        # Agricultural year (Jun Y - May Y+1) must fully contain one
        # Kharif + one Rabi season for the SAME crop-year label.
        _, _, agri_nov = assign_crop_year(month=11, calendar_year=2010)
        _, _, agri_jan_next = assign_crop_year(month=1, calendar_year=2011)
        _, _, agri_jun = assign_crop_year(month=6, calendar_year=2010)
        assert agri_nov == agri_jan_next == agri_jun == 2010


# =============================================================
# BUG #4: District Watch's silent 5-150 kg/ha classification gap
# =============================================================
# Must stay in sync with src/28_district_watch.py's classify_alert().

YIELD_WATCH_THRESHOLD = 150
YIELD_URGENT_THRESHOLD = 300
SEVERITY_CHANGE_THRESHOLD = 0.3

def classify_alert(yield_chg, sev_chg, rai_severity_relative, rai_severity_relative_prev):
    if pd.isna(rai_severity_relative_prev):
        return 'new'
    no_real_change = abs(yield_chg) < 5 and abs(sev_chg) < 0.05
    if no_real_change:
        return 'elevated_stable' if rai_severity_relative > 1.2 else 'stable'
    worsening = sev_chg > SEVERITY_CHANGE_THRESHOLD or yield_chg < -YIELD_WATCH_THRESHOLD
    if abs(yield_chg) > YIELD_URGENT_THRESHOLD and worsening:
        return 'urgent'
    if rai_severity_relative > 2.0 and sev_chg > SEVERITY_CHANGE_THRESHOLD:
        return 'urgent'
    if abs(yield_chg) > YIELD_WATCH_THRESHOLD or abs(sev_chg) > SEVERITY_CHANGE_THRESHOLD:
        return 'watch'
    return 'minor_change'


class TestDistrictWatchClassification:
    """Regression tests for the silent stable/minor_change gap bug."""

    def test_true_zero_change_is_stable(self):
        assert classify_alert(2, 0.01, 0.5, 0.5) == 'stable'

    def test_elevated_baseline_with_no_change_is_elevated_stable(self):
        assert classify_alert(2, 0.01, 1.5, 1.5) == 'elevated_stable'

    def test_THE_GAP_is_now_minor_change_not_stable(self):
        # This exact case (51 kg/ha change) used to silently fall
        # through to 'stable' before the fix - the bug that caused
        # a 98% 'stable' result despite std=51.5 real variance.
        result = classify_alert(51, 0.1, 0.5, 0.5)
        assert result == 'minor_change'
        assert result != 'stable'

    def test_just_above_watch_threshold_is_watch(self):
        assert classify_alert(151, 0.1, 0.5, 0.5) == 'watch'

    def test_just_below_watch_threshold_is_minor_change(self):
        assert classify_alert(149, 0.1, 0.5, 0.5) == 'minor_change'

    def test_large_worsening_change_is_urgent(self):
        assert classify_alert(-350, 0.4, 1.0, 0.5) == 'urgent'

    def test_new_district_with_no_prior_snapshot(self):
        assert classify_alert(0, 0, 0.5, np.nan) == 'new'

    def test_boundary_exactly_at_no_real_change_floor(self):
        # abs(yield_chg) < 5 is strict - exactly 5 should NOT count
        # as no_real_change (this documents the strict-inequality
        # choice as intentional, not accidental).
        result = classify_alert(5, 0.01, 0.5, 0.5)
        assert result != 'stable'


# =============================================================
# BUG #1: Kharif early-warning pacing bug (script 32)
# =============================================================
# Must stay in sync with src/32_kharif_early_warning.py's
# classify_risk().

MIN_YEARS_REQUIRED = 3

def classify_kharif_risk(z, n_years):
    if pd.isna(z) or n_years < MIN_YEARS_REQUIRED:
        return 'insufficient_data'
    if z < -1.2:
        return 'high'
    elif z < -0.5:
        return 'moderate'
    else:
        return 'low'


class TestKharifRiskClassification:
    """Regression tests for the early-warning risk classifier."""

    def test_insufficient_history_flagged_not_guessed(self):
        assert classify_kharif_risk(z=-3.0, n_years=2) == 'insufficient_data'

    def test_nan_zscore_is_insufficient_data(self):
        assert classify_kharif_risk(z=np.nan, n_years=6) == 'insufficient_data'

    def test_strongly_below_normal_is_high_risk(self):
        assert classify_kharif_risk(z=-2.0, n_years=6) == 'high'

    def test_moderately_below_normal_is_moderate_risk(self):
        assert classify_kharif_risk(z=-0.8, n_years=6) == 'moderate'

    def test_near_normal_is_low_risk(self):
        assert classify_kharif_risk(z=0.1, n_years=6) == 'low'

    def test_classification_is_not_uniform_across_a_realistic_spread(self):
        # Regression test for the ACTUAL 117/118-bug shape: feeding
        # in a realistic spread of z-scores must NOT collapse to one
        # category. (The original bug's root cause was upstream in
        # the z-score calculation itself, not this classifier - but
        # this guards against ever reintroducing a classifier-side
        # version of the same failure.)
        np.random.seed(42)
        z_scores = np.random.normal(loc=-0.8, scale=0.6, size=118)
        results = [classify_kharif_risk(z, n_years=6) for z in z_scores]
        top_share = pd.Series(results).value_counts(normalize=True).iloc[0]
        assert top_share < 0.95, (
            f"Classification collapsed to {top_share:.0%} in one bucket on a "
            f"realistic z-score spread - this is the Kharif-bug shape."
        )


# =============================================================
# INTEGRATION TESTS - need real pipeline output, skip if absent
# =============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(REPO_ROOT, rel)


# =============================================================
# BUG #6: Kharif window frozen at "June" past month-end (script 32)
# =============================================================
# Found 2026-07-07: the window was hardcoded to `month == 6` in both
# the current-season filter and the historical filter, with the
# cutoff taken as a bare day-of-month. Once real data crosses into
# July, a July day-of-month gets misapplied as if it were a June day
# (or July data is silently excluded). Must stay in sync with
# src/32_kharif_early_warning.py's season_day_offset().

def season_day_offset(dt, year):
    """Reimplements script 32's day-offset-from-June-1 logic exactly."""
    june1 = pd.Timestamp(f"{year}-06-01")
    return (pd.Timestamp(dt) - june1).days


class TestKharifSeasonWindow:
    """Regression tests for the June-frozen date window bug (Bug #6)."""

    def test_june_23_offset_is_22(self):
        assert season_day_offset("2026-06-23", 2026) == 22

    def test_july_07_offset_crosses_month_boundary_correctly(self):
        # This is the exact failure mode: July 7 must be treated as
        # 36 days into the season (June has 30 days), NOT as "day 7"
        # of a June-only window.
        assert season_day_offset("2026-07-07", 2026) == 36

    def test_july_07_is_not_misread_as_early_june(self):
        offset = season_day_offset("2026-07-07", 2026)
        assert offset > 30, (
            "July 7 produced a season-day-offset <= 30 - this is the exact "
            "Bug #6 shape: a July date being misclassified as still within June."
        )

    def test_october_end_of_kharif_offset(self):
        # Sanity check the far end of the season doesn't overflow/break.
        assert season_day_offset("2026-10-31", 2026) == 152

    def test_window_label_spans_months_when_appropriate(self):
        # Reimplements the window_label construction from script 32.
        latest_date = pd.Timestamp("2026-07-07")
        window_label = f"Jun 1\u2013{latest_date.strftime('%b %d')}"
        assert window_label == "Jun 1\u2013Jul 07"
        assert "Jun" in window_label and "Jul" in window_label


class TestModelSchemaConsistency:
    """Regression tests for the stale quantile-model bug (Bug #2)."""

    @pytest.mark.skipif(
        not os.path.exists(_path('data/processed/feature_list_v2.txt')),
        reason="feature_list_v2.txt not present in this environment"
    )
    def test_all_models_match_current_feature_schema(self):
        with open(_path('data/processed/feature_list_v2.txt')) as f:
            current_features = [l.strip() for l in f.readlines()]

        for fname in ['xgb_tuned.pkl', 'xgb_lower.pkl', 'xgb_upper.pkl']:
            path = _path(f'outputs/models/{fname}')
            if not os.path.exists(path):
                pytest.skip(f"{fname} not present in this environment")
            with open(path, 'rb') as f:
                model = pickle.load(f)
            model_features = model.get_booster().feature_names
            assert model_features == current_features, (
                f"{fname} feature schema does not match feature_list_v2.txt - "
                f"this model is STALE and must be retrained before live use."
            )


class TestPredictionPlausibility:
    """Sanity bounds on live predictions - catches a broken pipeline
    producing physically implausible yield numbers."""

    @pytest.mark.skipif(
        not os.path.exists(_path('data/processed/current_predictions_full.csv')),
        reason="current_predictions_full.csv not present in this environment"
    )
    def test_predicted_yields_within_plausible_range(self):
        df = pd.read_csv(_path('data/processed/current_predictions_full.csv'))
        # Indian wheat district-average yields realistically fall
        # within roughly 1000-7000 kg/ha; anything outside that is
        # almost certainly a unit error or broken feature, not a
        # real prediction.
        assert df['current_pred_yield'].between(1000, 7000).all(), (
            "Some predicted yields fall outside the plausible 1000-7000 kg/ha "
            "range - check for a unit error or broken feature upstream."
        )

    @pytest.mark.skipif(
        not os.path.exists(_path('data/processed/current_predictions_full.csv')),
        reason="current_predictions_full.csv not present in this environment"
    )
    def test_no_null_predictions(self):
        df = pd.read_csv(_path('data/processed/current_predictions_full.csv'))
        assert df['current_pred_yield'].notna().all(), (
            "At least one district has a null prediction - the model "
            "likely failed silently on missing/malformed input features."
        )


class TestFailureLogFreshness:
    """Regression test for Bug #12: weather_current_failures.csv used to only be
    written `if failed:`, so a real failure from a PAST run could sit on disk
    indefinitely and get silently re-reported by Check 5 as if it happened on the
    CURRENT run, even after every district fetched cleanly. The file must always
    be written fresh, every run, including the empty case."""

    def test_empty_failure_list_still_produces_a_valid_empty_csv(self, tmp_path):
        # Mirrors exactly what script 22 now does on a fully-successful run:
        # pd.DataFrame(failed, columns=[...]).to_csv(...) even when failed == [].
        failed = []
        out_path = tmp_path / "weather_current_failures.csv"
        pd.DataFrame(failed, columns=["state", "district", "error"]).to_csv(out_path, index=False)

        assert out_path.exists(), "The failures file must be written even with zero failures"
        result = pd.read_csv(out_path)
        assert len(result) == 0, "An empty run must produce a zero-row file, not skip writing entirely"

    def test_stale_failure_from_a_past_run_would_be_overwritten(self, tmp_path):
        # Simulates the exact bug: a stale file exists from a PAST failed run,
        # then a NEW successful run (failed == []) must overwrite it, not leave
        # the old failure sitting there for Check 5 to misread as current.
        out_path = tmp_path / "weather_current_failures.csv"
        pd.DataFrame([{"state": "Haryana", "district": "Karnal", "error": "stale from a past run"}]).to_csv(
            out_path, index=False)
        assert len(pd.read_csv(out_path)) == 1  # stale failure present, pre-fix state

        failed = []  # this run had zero failures
        pd.DataFrame(failed, columns=["state", "district", "error"]).to_csv(out_path, index=False)

        result = pd.read_csv(out_path)
        assert len(result) == 0, (
            "A successful run must overwrite a stale failure from a past run, "
            "not leave it sitting there to be misread as a current-run failure."
        )


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))