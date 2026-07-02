# =============================================================
# NAIP Project - Step 38: Groundwater Analysis (Paper Section)
# =============================================================
# 17% coverage (28/118 districts) is too sparse and non-random to
# include gw_depth as a model FEATURE without introducing selection
# bias. This script instead treats groundwater as a validated
# FINDING: for the 28 monitored districts, does the model's
# residual error correlate with groundwater depth trends?
#
# If yes: groundwater stress is a real, unmodeled signal - this is
# a genuine, novel finding and a clear direction for future work
# (collect more monitoring wells, add groundwater as a feature
# once coverage reaches ~70%+).
#
# If no: groundwater depth doesn't explain the model's errors
# independently of the weather/NDVI features already in the model.
# Also a publishable finding.
#
# Either way this is better for the paper than forcing a sparse,
# potentially biased feature into the model.
# =============================================================

import pandas as pd
import numpy as np
from scipy import stats

# Load model residuals (from script 40) and groundwater
residuals = pd.read_csv('outputs/metrics/residuals_detail.csv')
gw = pd.read_csv('data/processed/groundwater_district.csv')

# Join on district-year with case-insensitive matching to handle
# naming inconsistencies between Kuruva et al. and NAIP residuals
residuals['district_lower'] = residuals['district'].str.lower().str.strip()
gw['district_lower'] = gw['district'].str.lower().str.strip()

merged = residuals.merge(
    gw[['district_lower', 'year', 'gw_depth_premonsoon_m',
        'gw_depth_postmonsoon_m', 'n_wells_premonsoon']],
    on=['district_lower', 'year'],
    how='inner'
)
print(f"Matched rows: {len(merged)} (district-years with both residuals + GW data)")
print(f"Districts covered: {merged['district_lower'].nunique()}")
print()

# Per-district GW depth trend - explicit loop avoids groupby.apply
# fragility across pandas versions (include_groups parameter didn't
# exist before pandas 2.2, causing silent failures on older installs)
gw_trend_rows = []
for d_lower, grp in gw.groupby('district_lower'):
    sub = grp.dropna(subset=['gw_depth_premonsoon_m'])
    if len(sub) > 3:
        slope = stats.linregress(sub['year'], sub['gw_depth_premonsoon_m'])[0]
        gw_trend_rows.append({'district_lower': d_lower, 'gw_depth_trend_m_per_year': slope})
gw_trend = pd.DataFrame(gw_trend_rows)

dist_errors = residuals.groupby('district_lower')['residual'].agg(
    abs_error=lambda x: x.abs().mean(),
    n=len
).reset_index()

dist_analysis = dist_errors.merge(gw_trend, on='district_lower', how='inner')
dist_analysis = dist_analysis.dropna(subset=['gw_depth_trend_m_per_year'])

print(f"Districts with GW trend + model error: {len(dist_analysis)}")
print()

# Correlation: do districts with declining water tables have higher model error?
if len(dist_analysis) >= 5:
    r, p = stats.pearsonr(
        dist_analysis['gw_depth_trend_m_per_year'],
        dist_analysis['abs_error']
    )
    print(f"Pearson r (GW decline rate vs abs model error): {r:.3f}")
    print(f"p-value: {p:.4f}")
    if p < 0.05:
        print(f"SIGNIFICANT: districts with faster groundwater decline have")
        print(f"{'higher' if r > 0 else 'lower'} model error - groundwater IS an unmodeled signal.")
        print("Paper finding: add monitoring coverage and include GW as a feature in future work.")
    else:
        print("NOT significant: groundwater decline rate does not independently")
        print("explain model error beyond what weather/NDVI already captures.")
        print("Paper finding: current feature set adequately proxies water availability.")
    print()
else:
    print(f"Too few districts ({len(dist_analysis)}) for correlation - printing what we have.")
    print()

if len(dist_analysis) > 0:
    print("Top 5 districts by groundwater decline rate (m/year, positive = deeper):")
    print(dist_analysis.nlargest(min(5, len(dist_analysis)), 'gw_depth_trend_m_per_year')[
        ['district_lower', 'gw_depth_trend_m_per_year', 'abs_error', 'n']
    ].to_string(index=False))
else:
    print("No districts matched between GW trend data and residuals - check district name alignment.")
    print("GW districts:", sorted(gw['district_lower'].unique()))
    print("Residual districts:", sorted(residuals['district_lower'].unique())[:10], "...")

print()
print("Summary for paper:")
print("  - Groundwater data available for 28/118 districts (Kuruva et al. 2025)")
print("  - Coverage too sparse (17%) for model feature inclusion without selection bias")
print("  - Analysis above determines whether GW explains residual error independently")
print("  - Either result is publishable and specific to this study")