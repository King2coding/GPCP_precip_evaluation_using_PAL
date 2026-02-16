#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
path_to_dat = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs_27Jan2026'
df = pd.read_pickle(os.path.join(path_to_dat, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260216.pkl'))

#%%
# -----------------------------
# 0) Clean + define "available"
# -----------------------------
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["region", "ID", "date"])

# Available = rain_rate is not NaN AND non-negative
df["is_available"] = df["rain_rate"].notna() & (df["rain_rate"] >= 0)

# keep only available rows for availability accounting
dfa = df[df["is_available"]].copy()

# -----------------------------
# 1) Define analysis window
# -----------------------------
t0 = pd.Timestamp("2000-06-01")
t1 = pd.Timestamp("2020-12-31")

# constrain to analysis window
dfa = dfa[(dfa["date"] >= t0) & (dfa["date"] <= t1)].copy()

all_days = pd.date_range(t0, t1, freq="D")
n_expected_days = len(all_days)

# -----------------------------
# 2) Region-wise buoy counts
# -----------------------------
n_buoys_used = dfa.groupby("region")["ID"].nunique().rename("n_buoys_used")

# -----------------------------
# 3) Region-day coverage (at least one buoy)
# -----------------------------
region_day = (
    dfa.groupby(["region", "date"])
       .size()
       .rename("n_obs_that_day")
       .reset_index()
)

n_days_any = region_day.groupby("region")["date"].nunique().rename("n_days_any_buoy")
frac_days_any = (n_days_any / n_expected_days).rename("frac_days_any_buoy")

# -----------------------------
# 4) Region–buoy-day completeness
# -----------------------------
n_buoy_days_obs = dfa.groupby("region").size().rename("n_buoy_days_obs")

region_expected_buoy_days = (n_buoys_used * n_expected_days).rename("n_expected_buoy_days")
frac_buoy_days = (n_buoy_days_obs / region_expected_buoy_days).rename("frac_buoy_days")

# -----------------------------
# 5) Complete-record buoys (>=95% of days)
# -----------------------------
per_buoy_days = (
    dfa.groupby(["region", "ID"])["date"]
       .nunique()
       .rename("n_days_present")
       .reset_index()
)

per_buoy_days["frac_days_present"] = per_buoy_days["n_days_present"] / n_expected_days
per_buoy_days["is_complete_95pct"] = per_buoy_days["frac_days_present"] >= 0.95

n_complete_95 = (
    per_buoy_days.groupby("region")["is_complete_95pct"]
    .sum()
    .astype(int)
    .rename("n_complete_buoys_95pct")
)
pct_complete_95 = (n_complete_95 / n_buoys_used).rename("pct_complete_buoys_95pct")

# -----------------------------
# 6) Start/end dates (region + buoy-level medians)
# -----------------------------
region_min = dfa.groupby("region")["date"].min().rename("region_start_date")
region_max = dfa.groupby("region")["date"].max().rename("region_end_date")

per_buoy_start_end = (
    dfa.groupby(["region", "ID"])["date"]
       .agg(buoy_start="min", buoy_end="max")
       .reset_index()
)

median_buoy_start = per_buoy_start_end.groupby("region")["buoy_start"].median().rename("median_buoy_start")
median_buoy_end   = per_buoy_start_end.groupby("region")["buoy_end"].median().rename("median_buoy_end")

# -----------------------------
# 7) Assemble summary table
# -----------------------------
summary = pd.concat(
    [
        n_buoys_used,
        region_expected_buoy_days,
        n_days_any,
        frac_days_any,
        n_buoy_days_obs,
        frac_buoy_days,
        n_complete_95,
        pct_complete_95,
        region_min,
        region_max,
        median_buoy_start,
        median_buoy_end,
    ],
    axis=1,
).reset_index()

# nicer formatting columns (optional)
summary["analysis_start"] = t0.date()
summary["analysis_end"] = t1.date()
summary["n_expected_days"] = n_expected_days

# order columns
summary = summary[
    [
        "region",
        "analysis_start", "analysis_end", "n_expected_days",
        "n_buoys_used",
        "n_days_any_buoy", "frac_days_any_buoy",
        "n_buoy_days_obs", "n_expected_buoy_days", "frac_buoy_days",
        "n_complete_buoys_95pct", "pct_complete_buoys_95pct",
        "region_start_date", "region_end_date",
        "median_buoy_start", "median_buoy_end",
    ]
].sort_values("region")

display(summary)

# -----------------------------
# 8) Plot: bar comparisons
# -----------------------------
regions = summary["region"].tolist()

# Stacked buoy counts: complete vs incomplete
complete = summary["n_complete_buoys_95pct"].to_numpy()
total = summary["n_buoys_used"].to_numpy()
incomplete = total - complete

# Completeness fractions
frac_buoy = summary["frac_buoy_days"].to_numpy()
frac_any  = summary["frac_days_any_buoy"].to_numpy()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=True)

# Panel 1: buoy counts
axes[0].bar(regions, incomplete, label="Incomplete buoys (<95% days)")
axes[0].bar(regions, complete, bottom=incomplete, label="Complete buoys (≥95% days)")
axes[0].set_ylabel("Number of buoys")
axes[0].set_title("Buoy availability by region (2000–2020)")
axes[0].legend(frameon=False)

# Panel 2: completeness fractions
axes[1].bar(regions, frac_buoy, label="Buoy-day completeness \n(obs / expected buoy-days)")
axes[1].plot(regions, frac_any, marker="o", linestyle="None", label="Fraction of days with ≥1 buoy")
axes[1].set_ylabel("Fraction")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Coverage fractions by region")
axes[1].legend(frameon=False)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# 9) Optional: export table for email/slide
# -----------------------------
# summary.to_csv("buoy_coverage_summary_2000_2020.csv", index=False)
