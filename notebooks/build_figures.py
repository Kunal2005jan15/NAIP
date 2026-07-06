# %% [markdown]
# # NAIP — Figure Generation for Synopsis / Report / Paper
# Open this file in VS Code with the Python/Jupyter extension and run cell-by-cell
# (each `# %%` block is one cell — click "Run Cell" or use Shift+Enter).
# Every number plotted here is computed directly from your own outputs/ and data/ files —
# nothing is hardcoded from the Word docs, so if a figure disagrees with something you
# wrote earlier, trust this figure and update the doc, not the other way around.
#
# Assumes this file lives at NAIP/notebooks/build_figures.py (repo root one level up).

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Earth & Grain palette (matches the dashboard's design system)
WHEAT_GOLD = "#C9A24B"
SOIL_BROWN = "#6B4A32"
IRRIGATION_GREEN = "#4F7749"
ALERT_TERRACOTTA = "#C1543A"
INK = "#2B2420"
PAPER = "#FAF6EE"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": PAPER,
})

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"saved {name}.png")

# %% [markdown]
# ## Figure 1 — Actual vs. Predicted Yield (Held-Out Test Set, 2018–19)
# For: Report + Paper (Results). This is the single most important figure — it's your
# chronologically-validated, never-touched-until-final-eval test set.

# %%
tp = pd.read_csv(METRICS / "test_predictions_FINAL.csv")

r2 = r2_score(tp["actual_yield"], tp["pred_yield"])
rmse = np.sqrt(mean_squared_error(tp["actual_yield"], tp["pred_yield"]))
mape = (np.abs((tp["actual_yield"] - tp["pred_yield"]) / tp["actual_yield"])).mean() * 100
print(f"Authoritative final test metrics (computed here, not copied from any doc):")
print(f"  R2 = {r2:.4f}   RMSE = {rmse:.1f} kg/ha   MAPE = {mape:.2f}%   n={len(tp)}")

fig, ax = plt.subplots(figsize=(6, 6))
crop_colors = {"Wheat": WHEAT_GOLD, "Rice": IRRIGATION_GREEN}
for crop, sub in tp.groupby("crop"):
    ax.scatter(sub["actual_yield"], sub["pred_yield"], s=18, alpha=0.65,
               label=crop, color=crop_colors.get(crop, SOIL_BROWN), edgecolors="none")
lims = [tp[["actual_yield", "pred_yield"]].min().min() * 0.95,
        tp[["actual_yield", "pred_yield"]].max().max() * 1.05]
ax.plot(lims, lims, "--", color=INK, linewidth=1, alpha=0.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Yield (kg/ha)"); ax.set_ylabel("Predicted Yield (kg/ha)")
ax.set_title(f"Actual vs. Predicted Yield — Test Set 2018–19\n$R^2$={r2:.3f}, RMSE={rmse:.0f} kg/ha, MAPE={mape:.1f}%")
ax.legend(frameon=False)
save(fig, "fig1_actual_vs_predicted")
plt.show()

# %% [markdown]
# ## Figure 2 — Residual Distribution and Residuals by Year
# For: Report + Paper (error analysis). Uses residuals_detail.csv directly.

# %%
res = pd.read_csv(METRICS / "residuals_detail.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(res["residual"], bins=30, color=WHEAT_GOLD, edgecolor=INK, linewidth=0.5)
axes[0].axvline(0, color=ALERT_TERRACOTTA, linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Residual, Actual − Predicted (kg/ha)"); axes[0].set_ylabel("Count")
axes[0].set_title("Residual Distribution")

for crop, sub in res.groupby("crop"):
    axes[1].scatter(sub["year"], sub["residual"], s=14, alpha=0.5,
                     label=crop, color=crop_colors.get(crop, SOIL_BROWN))
axes[1].axhline(0, color=INK, linewidth=1, alpha=0.4)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Residual (kg/ha)")
axes[1].set_title("Residuals by Year"); axes[1].legend(frameon=False)
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
save(fig, "fig2_residual_analysis")
plt.show()

# %% [markdown]
# ## Figure 3 — Top 15 SHAP Feature Importances (Final Tuned Model)
# For: Synopsis (simplify to top 8–10) + Report + Paper

# %%
shap_tuned = pd.read_csv(METRICS / "shap_importance_TUNED.csv").sort_values("importance", ascending=True)
top15 = shap_tuned.tail(15)
bar_colors = [IRRIGATION_GREEN if any(k in f for k in ("lag", "trend", "rolling")) else WHEAT_GOLD
              for f in top15["feature"]]

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top15["feature"], top15["importance"], color=bar_colors, edgecolor=INK, linewidth=0.4)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 15 Feature Importances — Tuned XGBoost Model\n(green = autoregressive/trend, gold = everything else)")
save(fig, "fig3_shap_importance_top15")
plt.show()

# %% [markdown]
# ## Figure 4 — Model Comparison: XGBoost (untuned) vs Random Forest vs LightGBM vs XGBoost (Tuned)
# For: Report + Paper (justifies model + tuning choice).
# XGBoost(untuned)/RF/LightGBM come from model_comparison.csv (full test set, pre-tuning).
# "XGBoost (Tuned)" is computed directly from test_predictions_FINAL.csv above — this is
# the fair, apples-to-apples final comparison.

# %%
base = pd.read_csv(METRICS / "model_comparison.csv")
base_test = base[base["model"].str.contains("Test")].copy()
base_test["model"] = base_test["model"].str.replace(" (Test)", "", regex=False)
tuned_row = pd.DataFrame([{"model": "XGBoost (Tuned)", "rmse": rmse, "mae": np.nan, "mape": mape, "r2": r2}])
comp = pd.concat([base_test, tuned_row], ignore_index=True)
print(comp)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
bar_c = [SOIL_BROWN, IRRIGATION_GREEN, WHEAT_GOLD, ALERT_TERRACOTTA]
for ax, col, title in zip(axes, ["r2", "rmse", "mape"],
                          ["R\u00b2 (higher is better)", "RMSE, kg/ha (lower is better)", "MAPE, % (lower is better)"]):
    ax.bar(comp["model"], comp[col], color=bar_c, edgecolor=INK, linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
fig.suptitle("Model Comparison — Full Test Set", y=1.03)
fig.tight_layout()
save(fig, "fig4_model_comparison")
plt.show()

# %% [markdown]
# ## Figure 5 — Walk-Forward Cross-Validation (Chronological, 12 Folds 2008–2019)
# For: Paper — your strongest evidence against the "random split inflates accuracy" critique.

# %%
wf = pd.read_csv(METRICS / "walk_forward_cv_results.csv").sort_values("test_year")
print(f"Walk-forward R2 range: {wf['r2'].min():.2f}–{wf['r2'].max():.2f}, mean={wf['r2'].mean():.2f}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(wf["test_year"], wf["r2"], marker="o", color=IRRIGATION_GREEN, linewidth=2, label="R\u00b2")
ax1.axhline(wf["r2"].mean(), color=IRRIGATION_GREEN, linestyle=":", alpha=0.6,
            label=f"Mean R\u00b2 = {wf['r2'].mean():.2f}")
ax1.set_xlabel("Test Year"); ax1.set_ylabel("R\u00b2", color=IRRIGATION_GREEN)
ax1.set_title(f"Walk-Forward Validation R\u00b2 by Year (range: {wf['r2'].min():.2f}\u2013{wf['r2'].max():.2f})")
ax1.tick_params(axis="y", labelcolor=IRRIGATION_GREEN)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2 = ax1.twinx()
ax2.plot(wf["test_year"], wf["mape"], marker="s", color=ALERT_TERRACOTTA, linewidth=1.5, alpha=0.8, label="MAPE (%)")
ax2.set_ylabel("MAPE (%)", color=ALERT_TERRACOTTA)
ax2.tick_params(axis="y", labelcolor=ALERT_TERRACOTTA)
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="lower right", frameon=False, fontsize=9)
save(fig, "fig5_walk_forward_cv")
plt.show()

# %% [markdown]
# ## Figure 6 — Weather Signal's SHAP Share Across Pipeline Stages
# For: Paper (methodology/ablation narrative).
#
# ⚠️ HONESTY NOTE — read before captioning this in the paper:
# Each stage below changes MULTIPLE things at once (weather source, feature count, AND
# tuning together), so this does NOT isolate "the season-window fix alone." If you want
# that precise isolated claim (e.g. "fixing the window alone doubled weather SHAP share"),
# you'd need to rerun SHAP once with the bug deliberately reintroduced and once fixed,
# with everything else held constant, then diff those two runs. What IS honestly supported:
# weather-derived features' combined SHAP share grew substantially as the pipeline matured.

# %%
weather_keywords = ["rainfall", "temp", "humidity", "solar", "gdd", "dry_streak",
                     "water_balance", "frost_risk", "heat_stress", "drought_flag", "flood_flag"]

def weather_share(path):
    d = pd.read_csv(path)
    total = d["importance"].sum()
    mask = d["feature"].str.lower().apply(lambda f: any(k in f for k in weather_keywords))
    return 100 * d.loc[mask, "importance"].sum() / total

stages = [
    ("Partial Weather\n(Mendeley, v1)", weather_share(METRICS / "shap_importance.csv")),
    ("Full NASA Weather\n(untuned, v2)", weather_share(METRICS / "shap_importance_v2.csv")),
    ("Full Weather + Agronomic\n+ Satellite (Tuned)", weather_share(METRICS / "shap_importance_TUNED.csv")),
]
labels, values = zip(*stages)
for l, v in stages:
    print(f"{l.splitlines()[0]}: {v:.2f}%")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
bars = ax.bar(labels, values, color=[ALERT_TERRACOTTA, WHEAT_GOLD, IRRIGATION_GREEN], edgecolor=INK, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Weather-Derived Features' Share of Total SHAP Importance (%)")
ax.set_title("Weather Signal Share Across Pipeline Stages")
save(fig, "fig6_weather_shap_share_by_stage")
plt.show()

# %% [markdown]
# ## Figure 7 — Wheat vs. Rice: Crop-Specific Test Performance
# For: Report + Paper

# %%
with open(METRICS / "crop_split_summary.json") as f:
    crop_split = json.load(f)
crops = list(crop_split.keys())

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, metric, title in zip(axes, ["r2", "rmse", "mape"], ["R\u00b2", "RMSE (kg/ha)", "MAPE (%)"]):
    vals = [crop_split[c][metric] for c in crops]
    ax.bar(crops, vals, color=[WHEAT_GOLD, IRRIGATION_GREEN], edgecolor=INK)
    ax.set_title(title, fontsize=10)
fig.suptitle("Crop-Specific Test Performance (Tuned Model)", y=1.03)
fig.tight_layout()
save(fig, "fig7_crop_split_performance")
plt.show()

# %% [markdown]
# ## Figure 8 — SHAP Importance: Wheat vs. Rice (Top 8 Each)
# For: Paper (crop-specific driver discussion)

# %%
shap_wheat = pd.read_csv(METRICS / "shap_importance_wheat.csv").head(8)
shap_rice = pd.read_csv(METRICS / "shap_importance_rice.csv").head(8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(shap_wheat["feature"][::-1], shap_wheat["pct"][::-1], color=WHEAT_GOLD, edgecolor=INK)
axes[0].set_title("Wheat — Top 8 SHAP Features (%)")
axes[1].barh(shap_rice["feature"][::-1], shap_rice["pct"][::-1], color=IRRIGATION_GREEN, edgecolor=INK)
axes[1].set_title("Rice — Top 8 SHAP Features (%)")
for ax in axes:
    ax.set_xlabel("% of Total SHAP Importance")
fig.tight_layout()
save(fig, "fig8_shap_wheat_vs_rice")
plt.show()

# %% [markdown]
# ## Figure 9 — 80% Prediction Interval Calibration
# For: Paper (uncertainty quantification). Coverage computed directly, not copied from a doc.

# %%
tpc = tp.copy()
tpc["inside_interval"] = (tpc["actual_yield"] >= tpc["pred_lower"]) & (tpc["actual_yield"] <= tpc["pred_upper"])
coverage = 100 * tpc["inside_interval"].mean()
print(f"Empirical 80% interval coverage: {coverage:.2f}% (target band: 70-90%)")

sample = tpc.sort_values("actual_yield").reset_index(drop=True)
x = np.arange(len(sample))
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(x, sample["pred_lower"], sample["pred_upper"], color=WHEAT_GOLD, alpha=0.35, label="80% Prediction Interval")
ax.scatter(x, sample["actual_yield"], s=8, color=INK, alpha=0.6, label="Actual Yield", zorder=3)
ax.plot(x, sample["pred_yield"], color=IRRIGATION_GREEN, linewidth=1, alpha=0.8, label="Point Prediction")
ax.set_xlabel("Test Samples (sorted by actual yield)"); ax.set_ylabel("Yield (kg/ha)")
ax.set_title(f"80% Prediction Interval Calibration — Empirical Coverage: {coverage:.1f}%")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig9_prediction_interval_calibration")
plt.show()

# %% [markdown]
# ## Figure 10 — Rainfall Anomaly Index: Real-World Validation
# For: Paper (validates RAI against documented drought/flood years).
# ⚠️ Verified directly against master_dataset_v5_full_weather.csv's own drought_flag/flood_flag
# columns before plotting — only years the DATA itself flags are highlighted:
#   Ambala: drought_flag=1 in 1999 AND 2002 (RAI ≈ -1.27 and -1.00)
#   Karnal: flood_flag=1 in 2010 AND 2013 (RAI ≈ +1.74 and +1.81); 2012 is NOT flagged
#           (RAI ≈ +0.17, neutral) — if an earlier doc said "Karnal 2012/2013" as a pair,
#           only 2013 is actually supported by this dataset.

# %%
master = pd.read_csv(DATA / "master_dataset_v5_full_weather.csv")
ambala = master[(master["district"] == "Ambala") & (master["year"].between(1997, 2010))][
    ["year", "rainfall_anomaly_index", "drought_flag", "flood_flag"]].drop_duplicates().sort_values("year")
karnal = master[(master["district"] == "Karnal") & (master["year"].between(2005, 2019))][
    ["year", "rainfall_anomaly_index", "drought_flag", "flood_flag"]].drop_duplicates().sort_values("year")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, d, district in [(axes[0], ambala, "Ambala"), (axes[1], karnal, "Karnal")]:
    colors_bar = [ALERT_TERRACOTTA if (row.drought_flag or row.flood_flag) else IRRIGATION_GREEN
                  for row in d.itertuples()]
    ax.bar(d["year"], d["rainfall_anomaly_index"], color=colors_bar, edgecolor=INK, linewidth=0.4)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(-1.0, color=INK, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title(f"{district} — RAI by Year (red = drought/flood flagged by the model)", fontsize=9.5)
    ax.set_ylabel("Rainfall Anomaly Index (Rabi)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
save(fig, "fig10_rai_validation")
plt.show()

# %% [markdown]
# ## Figure 11 — Kharif 2026 Early-Warning Risk Distribution
# For: Report (demonstrates live monitoring capability)

# %%
kharif = pd.read_csv(DATA / "kharif_2026_risk_flags.csv")
counts = kharif["kharif_risk_level"].value_counts().reindex(["low", "moderate", "high"]).fillna(0)
print(counts)

fig, ax = plt.subplots(figsize=(5.5, 5))
risk_colors = {"low": IRRIGATION_GREEN, "moderate": WHEAT_GOLD, "high": ALERT_TERRACOTTA}
ax.bar(counts.index, counts.values, color=[risk_colors[i] for i in counts.index], edgecolor=INK)
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(int(v)), ha="center", fontweight="bold")
ax.set_ylabel("Number of Districts")
ax.set_title(f"Kharif 2026 Early-Warning Risk Classification\n({int(counts.sum())} of 118 districts, June rainfall vs. 2020–25 baseline)")
save(fig, "fig11_kharif_risk_distribution")
plt.show()

# %% [markdown]
# ## Figure 12 — Literature Comparison (Novelty Positioning)
# For: Paper (Related Work / novelty section).
# ⚠️ The published-paper R² values below are numbers YOU already gathered and cited in your
# own Literature Survey (Rao et al. 2019; Gandge 2017 / Kumar et al. 2020) — re-verify each
# against the original paper before submission, since I have not independently re-checked
# their primary sources here. "This Work" values are both computed directly above.

# %%
lit_comparison = pd.DataFrame([
    {"study": "Rao et al. 2019\n(District Rice, AP)", "r2": 0.72, "split": "Random/N.A."},
    {"study": "Gandge 2017 /\nKumar et al. 2020\n(State-level)", "r2": 0.80, "split": "Random"},
    {"study": "This Work —\nUntuned XGBoost", "r2": comp.loc[comp["model"] == "XGBoost", "r2"].values[0], "split": "Chronological"},
    {"study": "This Work —\nTuned XGBoost", "r2": r2, "split": "Chronological"},
])
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors2 = [SOIL_BROWN if ("Random" in s or "N.A." in s) else IRRIGATION_GREEN for s in lit_comparison["split"]]
bars = ax.bar(lit_comparison["study"], lit_comparison["r2"], color=bar_colors2, edgecolor=INK)
for b, v in zip(bars, lit_comparison["r2"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("R\u00b2"); ax.set_ylim(0, 1.0)
ax.set_title("R\u00b2 vs. Published Indian Crop-Yield Studies\n(brown = random split/leakage risk, green = chronological)")
ax.tick_params(axis="x", labelsize=8.5)
save(fig, "fig12_literature_comparison")
plt.show()

print("\nAll figures generated in:", FIG_DIR)