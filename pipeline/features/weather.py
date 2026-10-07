"""Shared seasonal weather features.

Single source of truth for the agronomic features computed from one
season block of daily weather rows. Used by:
  - src/15_aggregate_weather_all.py        (historical training features)
  - src/25_current_conditions.py           (live features)
  - later: the scenario engine

Keeping one implementation prevents train/serve mismatch: a formula
changed here changes training and live features together.

Required columns in `block` (daily rows, one season of one district):
  rainfall_mm, temp_avg_c, temp_max_c, temp_min_c, solar_radiation
"""

import numpy as np
import pandas as pd

GDD_BASE_TEMP_C = 10.0
DRY_DAY_THRESHOLD_MM = 1.0


def season_block_features(block: pd.DataFrame, season_label: str, prefix: str = "") -> pd.Series:
    """Compute GDD, longest dry streak, rainfall CV and water balance.

    Output keys are f"{prefix}{metric}_{season_label}", e.g.
    season_block_features(b, "rabi")                    -> gdd_rabi, ...
    season_block_features(b, "rabi", prefix="current_") -> current_gdd_rabi, ...

    Metrics:
      gdd            Growing Degree Days (base 10 C).
      max_dry_streak Longest run of consecutive days with < 1 mm rain.
      rainfall_cv    std/mean of daily rainfall (NaN if mean is 0).
      water_balance  Season rainfall minus simplified Hargreaves ET0,
                     using measured solar radiation in place of Ra.
    """
    gdd = np.maximum(block['temp_avg_c'] - GDD_BASE_TEMP_C, 0).sum()

    is_dry = (block['rainfall_mm'] < DRY_DAY_THRESHOLD_MM).values
    if is_dry.any():
        # run-length encode consecutive True values
        change = np.diff(np.concatenate(([0], is_dry.astype(int), [0])))
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        max_dry_streak = (ends - starts).max() if len(starts) else 0
    else:
        max_dry_streak = 0

    rain_mean = block['rainfall_mm'].mean()
    rain_std = block['rainfall_mm'].std()
    rainfall_cv = (rain_std / rain_mean) if rain_mean and rain_mean > 0 else np.nan

    trange = (block['temp_max_c'] - block['temp_min_c']).clip(lower=0)
    et0_daily = 0.0023 * block['solar_radiation'] * (block['temp_avg_c'] + 17.8) * np.sqrt(trange)
    water_balance = block['rainfall_mm'].sum() - et0_daily.sum()

    return pd.Series({
        f'{prefix}gdd_{season_label}': gdd,
        f'{prefix}max_dry_streak_{season_label}': max_dry_streak,
        f'{prefix}rainfall_cv_{season_label}': rainfall_cv,
        f'{prefix}water_balance_{season_label}': water_balance,
    })