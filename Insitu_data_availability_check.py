#%%
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%%
path_to_dat = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs_27Jan2026'
df = pd.read_pickle(os.path.join(path_to_dat, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260216.pkl'))

#%%
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

def fmt_k(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)

#%%

# -----------------------------
# Inputs
# -----------------------------
# df = buoy_sate_daily_rainfall_colasped_df.copy()
# your inventory counts (all buoys known per region)
full_buoy_count = pd.DataFrame(index=['ENP', 'WNP', 'IND', 'ATL'], columns=["Count"])
full_buoy_count.loc["ENP", "Count"] = 31
full_buoy_count.loc["WNP", "Count"] = 19
full_buoy_count.loc["IND", "Count"] = 21
full_buoy_count.loc["ATL", "Count"] = 20
full_buoy_count["Count"] = full_buoy_count["Count"].astype(int)

# -----------------------------
# 1) Clean + define availability
# -----------------------------
df["date"] = pd.to_datetime(df["date"])
df["is_available"] = df["rain_rate"].notna() & (df["rain_rate"] >= 0)
dfa = df[df["is_available"]].copy()

# -----------------------------
# 2) Analysis window
# -----------------------------
t0 = pd.Timestamp("2000-06-01")
t1 = pd.Timestamp("2020-12-31")
dfa = dfa[(dfa["date"] >= t0) & (dfa["date"] <= t1)].copy()

# expected days in window (matches your current setup)
n_expected_days = len(pd.date_range(t0, t1, freq="D"))

# -----------------------------
# 3) Core region metrics
# -----------------------------
# Buoys used in analysis window (have at least 1 available obs)
n_buoys_used = dfa.groupby("region")["ID"].nunique().rename("n_buoys_used")

# Observed buoy-days (sum over all buoys & days)
n_buoy_days_obs = dfa.groupby("region").size().rename("n_buoy_days_obs")

# Expected buoy-days per region = N_buoys_used × expected days
region_expected_buoy_days = (n_buoys_used * n_expected_days).rename("n_expected_buoy_days")

# Buoy-day completeness (bars)
frac_buoy = (n_buoy_days_obs / region_expected_buoy_days).rename("frac_buoy_days")

# Fraction of days with >=1 buoy (dots)
region_day = (
    dfa.groupby(["region", "date"])
       .size()
       .rename("n_obs_that_day")
       .reset_index()
)
n_days_any_buoy = region_day.groupby("region")["date"].nunique().rename("n_days_any_buoy")
frac_any = (n_days_any_buoy / n_expected_days).rename("frac_days_any_buoy")

# -----------------------------
# 4) Align regions consistently
# -----------------------------
regions = ["ATL", "ENP", "IND", "WNP"]  # your preferred order

# inventory (total) vs used
total_buoys = full_buoy_count["Count"].reindex(regions).fillna(0).astype(int).to_numpy()
used_buoys  = n_buoys_used.reindex(regions).fillna(0).astype(int).to_numpy()
missing_buoys = total_buoys - used_buoys

# fractions
frac_buoy_arr = frac_buoy.reindex(regions).to_numpy()
frac_any_arr  = frac_any.reindex(regions).to_numpy()

# annotation counts for frac plot
obs_counts = n_buoy_days_obs.reindex(regions).fillna(0).astype(int).to_numpy()
exp_counts = region_expected_buoy_days.reindex(regions).fillna(0).astype(int).to_numpy()


# ============================================================
# SLIDE A: Total buoys vs buoys used in analysis window
# ============================================================
fig1, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=200)

# stacked: used + missing (so total is visible)
bars_missing = ax.bar(regions, missing_buoys, bottom=used_buoys,
                      label="Not used (outside 2000–2020 window)")
bars_used    = ax.bar(regions, used_buoys, label="Used in 2000–2020")

ax.set_title("Buoy counts by region: inventory vs used (2000–2020)")
ax.set_ylabel("Number of buoys")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotate totals on top + used inside
for r, u, t in zip(regions, used_buoys, total_buoys):
    ax.text(r, t + 0.3, f"total={t}", ha="center", va="bottom", fontsize=9)
    ax.text(r, u/2, f"used={u}", ha="center", va="center", fontsize=9)

# small note on window/availability
# ax.text(
#     0.01, 0.02,
#     f"Window: {t0.date()}–{t1.date()} ({n_expected_days:,} days).  Available = rain_rate ≥ 0 and not NaN.",
#     transform=ax.transAxes, ha="left", va="bottom", fontsize=8
# )

# ---- Legend OUTSIDE (bottom) ----
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),   # centered below axes
    ncol=2
)

# make room at the bottom for the legend
fig1.subplots_adjust(bottom=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# SLIDE B: Coverage fractions by region (separate figure)
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 4.5), dpi=200)

bars = ax2.bar(regions, frac_buoy_arr, label="Buoy-day completeness (obs / expected buoy-days)")
ax2.plot(regions, frac_any_arr, marker="o", linestyle="None", label="Fraction of days with ≥1 buoy")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Fraction")
ax2.set_title("Coverage fractions by region (2000–2020)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(frameon=False)

# annotate obs/expected buoy-days on each bar
for rect, obs, exp in zip(bars, obs_counts, exp_counts):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    ax2.text(x, y + 0.02, f"{fmt_k(obs)} / {fmt_k(exp)}", ha="center", va="bottom", fontsize=9)

# compact definitions
ax2.text(
    0.01, 0.98,
    f"Expected days: {n_expected_days:,}   |   Expected buoy-days = N_used × expected days",
    transform=ax2.transAxes, ha="left", va="top", fontsize=9
)

plt.tight_layout()
plt.show()
# -----------------------------
# 9) Optional: export table for email/slide
# -----------------------------
# summary.to_csv("buoy_coverage_summary_2000_2020.csv", index=False)

#%%
# %%

rd = region_day.copy()
rd["date"] = pd.to_datetime(rd["date"])

# Monthly mean number of buoys contributing per day
rd_m = (rd.set_index("date")
          .groupby("region")["n_obs_that_day"]
          .resample("MS")
          .mean()
          .rename("mean_buoys_per_day")
          .reset_index())

regions = sorted(rd_m["region"].unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200, sharex=True, sharey=True)
axes = axes.flatten()

for ax, reg in zip(axes, regions):
    tmp = rd_m[rd_m["region"] == reg]
    ax.plot(tmp["date"], tmp["mean_buoys_per_day"])
    ax.set_title(reg)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Mean # buoys reporting (monthly)")
axes[2].set_ylabel("Mean # buoys reporting (monthly)")
for ax in axes[2:]:
    ax.set_xlabel("Year")

plt.tight_layout()
plt.show()

#%%

n_days_per_id = (
    dfa.groupby(["region", "ID"])["date"]
       .nunique()
       .rename("n_days_present")
       .reset_index()
)

n_days_per_id["frac_days_present"] = n_days_per_id["n_days_present"] / n_expected_days
per_buoy_span = (
    dfa.groupby(["region", "ID"])["date"]
       .agg(first_date="min", last_date="max")
       .reset_index()
)

n_days_per_id = n_days_per_id.merge(per_buoy_span, on=["region","ID"], how="left")

bins = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.01]
labels = ["<25%", "25–50%", "50–75%", "75–90%", "90–95%", "≥95%"]

n_days_per_id["coverage_bin"] = pd.cut(
    n_days_per_id["frac_days_present"],
    bins=bins, labels=labels, right=False
)

coverage_counts = (
    n_days_per_id.groupby(["region", "coverage_bin"])["ID"]
    .nunique()
    .unstack(fill_value=0)
)

coverage_fracs = coverage_counts.div(coverage_counts.sum(axis=1), axis=0)

top10 = n_days_per_id.sort_values(["region","frac_days_present"], ascending=[True, False]).groupby("region").head(10)
bottom10 = n_days_per_id.sort_values(["region","frac_days_present"], ascending=[True, True]).groupby("region").head(10)

per_buoy_report = n_days_per_id.sort_values(["region", "frac_days_present"], ascending=[True, False])
per_buoy_report.head()
# per_buoy_report.to_csv("buoy_per_id_coverage_2000_2020.csv", index=False)
# ---- 2x2 plot ----
regions = sorted(n_days_per_id["region"].unique())
fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=200, sharey=True)
axes = axes.flatten()

thr = 0.50  # change to 0.95 if you want strict completeness line

for ax, reg in zip(axes, regions):
    tmp = n_days_per_id[n_days_per_id["region"] == reg].sort_values("frac_days_present", ascending=False)
    
    ax.bar(tmp["ID"].astype(str), tmp["frac_days_present"])
    ax.axhline(thr, linestyle="--", linewidth=1.2)
    
    ax.set_title(f"{reg} (N={tmp['ID'].nunique()} buoys)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of \ndays present")
    
    # tidy axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

# If you have fewer than 4 regions, hide extra axes
for j in range(len(regions), 4):
    axes[j].axis("off")

fig.suptitle("Per-buoy daily availability within each region (2000–2020)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

#%% PAL Data availability check
path_to_dat = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs_27Jan2026'

df_pal = pd.read_pickle(os.path.join(path_to_dat, f'pal_sate_daily_rainfall_from_all_regions_and_all_tracks_20260128.pkl'))


full_pal_count = pd.DataFrame(index=['TNEP', 'TSEP', 'TNWP', 'ETNP', 'TNIO', 'STNA'], columns=["Count"])
full_pal_count.loc["TNEP", "Count"] = 20
full_pal_count.loc["TSEP", "Count"] = 6
full_pal_count.loc["TNWP", "Count"] = 7
full_pal_count.loc["ETNP", "Count"] = 4
full_pal_count.loc["TNIO", "Count"] = 3
full_pal_count.loc["STNA", "Count"] = 18
full_pal_count["Count"] = full_pal_count["Count"].astype(int)


# -----------------------------
# 1) Clean + define availability
# -----------------------------
df_pal["date"] = pd.to_datetime(df_pal["date"])
df_pal["is_available"] = df_pal["rain_rate"].notna() & (df_pal["rain_rate"] >= 0)
dfa_pal = df_pal[df_pal["is_available"]].copy()

# -----------------------------
# 2) Analysis window
# -----------------------------
pal_t0 = pd.Timestamp("2010-01-01")
pal_t1 = pd.Timestamp("2021-12-31")
dfa_pal = dfa_pal[(dfa_pal["date"] >= pal_t0) & (dfa_pal["date"] <= pal_t1)].copy()

# expected days in window (matches your current setup)
pal_n_expected_days = len(pd.date_range(pal_t0, pal_t1, freq="D"))

# -----------------------------
# 3) Core region metrics
# -----------------------------
# PAL used in analysis window (have at least 1 available obs)
n_pal_used = dfa_pal.groupby("region")["track_PAL_id"].nunique().rename("n_pal_used")

# Observed PAL-days (sum over all PALs & days)
n_pal_days_obs = dfa_pal.groupby("region").size().rename("n_pal_days_obs")

# Expected PAL-days per region = N_PALs_used × expected days
region_expected_pal_days = (n_pal_used * pal_n_expected_days).rename("n_expected_pal_days")

# PAL-day completeness (bars)
frac_pal = (n_pal_days_obs / region_expected_pal_days).rename("frac_pal_days")

# Fraction of days with >=1 PAL (dots)
pal_region_day = (
    dfa_pal.groupby(["region", "date"])
       .size()
       .rename("n_obs_that_day")
       .reset_index()
)
n_days_any_pal = pal_region_day.groupby("region")["date"].nunique().rename("n_days_any_pal")
pal_frac_any = (n_days_any_pal / pal_n_expected_days).rename("frac_days_any_pal")


# -----------------------------
# 4) Align regions consistently
# -----------------------------
pal_regions = pal_frac_any.index.to_list()

# inventory (total) vs used
total_pal = full_pal_count["Count"].reindex(pal_regions).fillna(0).astype(int).to_numpy()
used_pal  = n_pal_used.reindex(pal_regions).fillna(0).astype(int).to_numpy()
missing_pal = total_pal - used_pal

# fractions
frac_pal_arr = frac_pal.reindex(pal_regions).to_numpy()
frac_pal_any_arr  = pal_frac_any.reindex(pal_regions).to_numpy()

# annotation counts for frac plot
pal_obs_counts = n_pal_days_obs.reindex(pal_regions).fillna(0).astype(int).to_numpy()
pal_exp_counts = region_expected_pal_days.reindex(pal_regions).fillna(0).astype(int).to_numpy()

# ============================================================
# SLIDE A: Total PAL vs PALs used in analysis window
# ============================================================
fig1, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=200)

# stacked: used + missing (so total is visible)
bars_missing = ax.bar(pal_regions, missing_pal, bottom=used_pal,
                      label="Not used (outside 2010–2021 window)")
bars_used    = ax.bar(pal_regions, used_pal, label="Used in 2010–2021")

ax.set_title("PAL counts by region: inventory vs used (2010–2021)",y=1.05)
ax.set_ylabel("Number of PALs")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotate totals on top + used inside
for r, u, t in zip(pal_regions, used_pal, total_pal):
    ax.text(r, t + 0.3, f"total={t}", ha="center", va="bottom", fontsize=9)
    ax.text(r, u/2, f"used={u}", ha="center", va="center", fontsize=9)

# small note on window/availability
# ax.text(
#     0.01, 0.02,
#     f"Window: {pal_t0.date()}–{pal_t1.date()} ({pal_n_expected_days:,} days).  Available = rain_rate ≥ 0 and not NaN.",
#     transform=ax.transAxes, ha="left", va="bottom", fontsize=8
# )

# ---- Legend OUTSIDE (bottom) ----
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),   # centered below axes
    ncol=2
)

# make room at the bottom for the legend
fig1.subplots_adjust(bottom=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# SLIDE B: Coverage fractions by region (separate figure)
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 4.5), dpi=200)

bars = ax2.bar(pal_regions, frac_pal_arr, label="PAL-day completeness (obs / expected PAL-days)")
ax2.plot(pal_regions, frac_pal_any_arr, marker="o", linestyle="None", label="Fraction of days with ≥1 PAL")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Fraction")
ax2.set_title("Coverage fractions by region (2010–2021)", fontsize=14, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.18), loc="upper center", fontsize=12)

# annotate obs/expected PAL-days on each bar
for rect, obs, exp in zip(bars, pal_obs_counts, pal_exp_counts):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    ax2.text(x, y + 0.02, f"{fmt_k(obs)} / {fmt_k(exp)}", ha="center", va="bottom", fontsize=9)

# compact definitions
ax2.text(
    0.01, 0.98,
    f"Expected days: {fmt_k(pal_n_expected_days)}   |   Expected PAL-days = N_used × expected days",
    transform=ax2.transAxes, ha="left", va="top", fontsize=9
)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------
# 5) Monthly mean number of buoys contributing per day
pal_rd = pal_region_day.copy()
pal_rd["date"] = pd.to_datetime(pal_rd["date"])

# Monthly mean number of buoys contributing per day
pal_rd_m = (pal_rd.set_index("date")
          .groupby("region")["n_obs_that_day"]
          .resample("MS")
          .mean()
          .rename("mean_pal_per_day")
          .reset_index())

# regions = sorted(rd_m["region"].unique())

fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=200, sharex=True, sharey=False)
axes = axes.flatten()

for ax, reg in zip(axes, pal_regions):
    tmp = pal_rd_m[pal_rd_m["region"] == reg]
    ax.plot(tmp["date"], tmp["mean_pal_per_day"])
    ax.set_title(reg)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.75, 0.98, f"Region Inventory: {full_pal_count.loc[reg, 'Count']:,}", 
            transform=ax.transAxes, ha="left", va="top", fontsize=9)

axes[0].set_ylabel("Mean # PAL\nreporting (monthly)")
axes[1].set_ylabel("Mean # PAL\nreporting (monthly)")
axes[2].set_ylabel("Mean # PAL\nreporting (monthly)")
axes[3].set_ylabel("Mean # PAL\nreporting (monthly)")
axes[4].set_ylabel("Mean # PAL\nreporting (monthly)")
axes[5].set_ylabel("Mean # PAL\nreporting (monthly)")

for ax in axes[2:]:
    ax.set_xlabel("Year")

plt.tight_layout()
plt.show()

#-------------------------------------------------------------


pal_n_days_per_id = (
    dfa_pal.groupby(["region", "track_PAL_id"])["date"]
       .nunique()
       .rename("n_days_present")
       .reset_index()
)
bins = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.01]
labels = ["<25%", "25–50%", "50–75%", "75–90%", "90–95%", "≥95%"]

pal_n_days_per_id["frac_days_present"] = pal_n_days_per_id["n_days_present"] / pal_n_expected_days
per_pal_span = (
    dfa_pal.groupby(["region", "track_PAL_id"])["date"]
       .agg(first_date="min", last_date="max")
       .reset_index()
)

pal_n_days_per_id = pal_n_days_per_id.merge(per_pal_span, on=["region","track_PAL_id"], how="left")

pal_n_days_per_id["coverage_bin"] = pd.cut(
    pal_n_days_per_id["frac_days_present"],
    bins=bins, labels=labels, right=False
)

pal_coverage_counts = (
    pal_n_days_per_id.groupby(["region", "coverage_bin"])["track_PAL_id"]
    .nunique()
    .unstack(fill_value=0)
)

pal_coverage_fracs = pal_coverage_counts.div(pal_coverage_counts.sum(axis=1), axis=0)

# top10 = pal_n_days_per_id.sort_values(["region","frac_days_present"], ascending=[True, False]).groupby("region").head(10)
# bottom10 = pal_n_days_per_id.sort_values(["region","frac_days_present"], ascending=[True, True]).groupby("region").head(10)

# per_pal_report = pal_n_days_per_id.sort_values(["region", "frac_days_present"], ascending=[True, False])
# per_pal_report.head()
# per_pal_report.to_csv("buoy_per_id_coverage_2000_2020.csv", index=False)
# ---- 2x2 plot ----
# regions = sorted(n_days_per_id["region"].unique())
fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=200, sharey=True)
axes = axes.flatten()

thr = 0.50  # change to 0.95 if you want strict completeness line

for ax, reg in zip(axes, pal_regions):
    tmp = pal_n_days_per_id[pal_n_days_per_id["region"] == reg].sort_values("frac_days_present", ascending=False)
    tmp_id = [str(i).replace("PAL_precip_wind_", "") for i in tmp["track_PAL_id"]]
    
    ax.bar(tmp_id, tmp["frac_days_present"])
    ax.axhline(thr, linestyle="--", linewidth=1.2)
    
    ax.set_title(f"{reg} (N={tmp['track_PAL_id'].nunique()} buoys)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of \ndays present")
    
    # tidy axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

# If you have fewer than 6 regions, hide extra axes
for j in range(len(pal_regions), 6):
    axes[j].axis("off")

fig.suptitle("Per PAL daily availability\nwithin each region (2010–2021)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()