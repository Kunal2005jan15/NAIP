"""Golden-value tests for pipeline.features.weather.

The synthetic block is hand-checkable: 10 days, 20 C average, 30/10 C
max/min, 20 MJ solar, rain on days 4 and 7 only (5 mm and 10 mm).
Expected values were computed once from the original script-15 formulas.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.features.weather import season_block_features


@pytest.fixture
def block():
    return pd.DataFrame({
        'rainfall_mm': [0, 0, 0, 5, 0, 0, 10, 0, 0, 0],
        'temp_avg_c': [20.0] * 10,
        'temp_max_c': [30.0] * 10,
        'temp_min_c': [10.0] * 10,
        'solar_radiation': [20.0] * 10,
    })


def test_golden_values(block):
    out = season_block_features(block, 'rabi')
    assert out['gdd_rabi'] == pytest.approx(100.0)
    assert out['max_dry_streak_rabi'] == 3
    assert out['rainfall_cv_rabi'] == pytest.approx(2.249828525701843)
    assert out['water_balance_rabi'] == pytest.approx(-62.76149998553268)


def test_prefix_names(block):
    out = season_block_features(block, 'rabi', prefix='current_')
    assert list(out.index) == [
        'current_gdd_rabi', 'current_max_dry_streak_rabi',
        'current_rainfall_cv_rabi', 'current_water_balance_rabi',
    ]


def test_historical_and_live_names_share_values(block):
    hist = season_block_features(block, 'rabi')
    live = season_block_features(block, 'rabi', prefix='current_')
    assert hist.values == pytest.approx(live.values)


def test_no_dry_days(block):
    block['rainfall_mm'] = 5.0
    out = season_block_features(block, 'kharif')
    assert out['max_dry_streak_kharif'] == 0


def test_zero_rain_gives_nan_cv(block):
    block['rainfall_mm'] = 0.0
    out = season_block_features(block, 'kharif')
    assert np.isnan(out['rainfall_cv_kharif'])
    assert out['max_dry_streak_kharif'] == 10