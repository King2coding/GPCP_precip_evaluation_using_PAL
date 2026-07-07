#%%
from IEPPO_utils import *

#%%

path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_buoy_dirs = [os.path.join(moored_bouys_paf, d) for d in os.listdir(moored_bouys_paf) if os.path.isdir(os.path.join(moored_bouys_paf, d))]

path_to_ocRain = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/OceanRain'

#%% Floating variables
def fmt_k(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)

#-------------------------------------------------------------------------------------------

# CLASSIFY BUOY FILES BY REGION
pacific_buoy_dir = next((d for d in all_buoy_dirs if "PACIFIC" in d.upper()), None)
indian_buoy_dir = next((d for d in all_buoy_dirs if "INDIAN" in d.upper()), None)
atlantic_buoy_dir = next((d for d in all_buoy_dirs if "ATLANTIC" in d.upper()), None)

# Ensure directories were found
if not pacific_buoy_dir:
    raise ValueError("No directory found for PACIFIC region.")
if not indian_buoy_dir:
    raise ValueError("No directory found for INDIAN region.")
if not atlantic_buoy_dir:
    raise ValueError("No directory found for ATLANTIC region.")

# Get all files for each region
pacific_buoy_files = sorted([os.path.join(pacific_buoy_dir, f) for f in os.listdir(pacific_buoy_dir) if f.endswith('.cdf')])

# define buoy files by regions
pacific_buoy_regions = ['ENP', 'WNP']

# FIRST GROUP BUOY FILES BY REGION BASED ON THEIR LONGITUDE
# Define longitude bounds for ENP and WNP
pacific_region_bounds = {
    'ENP': (-180, -60),  # Longitude range for Eastern North Pacific in [-180, 180]
    'WNP': (120, 180)      # Longitude range for Western North Pacific in [-180, 180]
}

# Group buoy files by region
buoy_files_by_region = {'ENP': [], 'WNP': []}
for buoy_file in pacific_buoy_files:
    with xr.open_dataset(buoy_file) as ds:
        buoy_lon = ds['lon'].values[0]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

    for region, bounds in pacific_region_bounds.items():
        if bounds[0] <= buoy_lon <= bounds[1]:
            buoy_files_by_region[region].append(buoy_file)
            break

# add the india and atlantic buoys
buoy_files_by_region['IND'] = sorted([os.path.join(indian_buoy_dir, f) for f in os.listdir(indian_buoy_dir) if f.endswith('.cdf')])
buoy_files_by_region['ATL'] = sorted([os.path.join(atlantic_buoy_dir, f) for f in os.listdir(atlantic_buoy_dir) if f.endswith('.cdf')])
print("✅ BUOY CLASSIFICATION SUMMARY COMPLETED")
print("\n" + "="*50)
gc.collect()

regional_buoy_sate_dfs_daily_lst = []
for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
        # store Buoy and GPCP dataframes

    # LOAD Buoy DATA
    for b, b_file in enumerate(buoy_files):
        b_df, b_lat, b_lon = grab_Buoy_data_df(b_file)
        b_df["month"] = b_df["time"].dt.month
        b_df['year'] = b_df['time'].dt.year
        # metadata
        buoy_id = os.path.basename(b_file).split(".")[0]
        b_df["region"] = region_name
        b_df["ID"] = buoy_id

        # append to list
        regional_buoy_sate_dfs_daily_lst.append(b_df)
#-------------------------------------------------------------------------------------------
# CLASSIFY PAL FILES BY REGION
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, 
                                                               PAL_region_bounds)

regional_PAL_sate_dfs_daily_lst = []
for region_name, pal_files in list(pals_classed_by_region.items())[:-1]:
  
    print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")      

     

    # LOAD PAL DATA
    for i,pal_file in enumerate(pal_files):
        pal_ds = xr.open_dataset(pal_file)        
        pal_rain_df = grab_PAL_rain_and_wind_df(pal_ds)  
        pal_rain_df["month"] = pal_rain_df["time"].dt.month
        pal_rain_df['year'] = pal_rain_df['time'].dt.year

        # metadata
        pal_id = os.path.basename(pal_file).split(".")[0]
        pal_rain_df["region"] = region_name
        pal_rain_df["ID"] = pal_id

        # append to list
        regional_PAL_sate_dfs_daily_lst.append(pal_rain_df)
gc.collect()


#-------------------------------------------------------------------------------------------
ocRain = np.load(os.path.join(path_to_ocRain, "OceanRAIN_MINUTE_coordinates_and_data_Kingsley_20260210.npz"))

# Access the keys in the .npz file
keys = ocRain.files
print("Keys in the .npz file:", keys)
# Create a DataFrame from the .npz file
ocRain_df = pd.DataFrame({key: ocRain[key] for key in keys})

print("Data files listed and datasets loaded.")


df_qc = oceanrain_step0_qc_precip_main(
    ocRain_df,
    drop_harbor_inop=True,
    drop_spurious_flag2_11=True,
    keep_true_zero=True,
    keep_flag2_12_zero_precip=False,
    min_flag2_positive=13,
    prob_thr=None,
    wind_max=None,
    qclip_hi=None,
)

# gpcp_lat_1d = np.arange(89.75, -90.0, -0.5, dtype=np.float32)
# gpcp_lon_1d = np.arange(-179.75, 180.0, 0.5, dtype=np.float32)


daily_or_all, daily_or_usable = oceanrain_daily_aggregate_to_gpcp_main(
    oc_df_minute=df_qc,
    gpcp_lat_1d=np.arange(89.75, -90.0, -0.5, dtype=np.float32),
    gpcp_lon_1d=np.arange(-179.75, 180.0, 0.5, dtype=np.float32),
    lat_abs_min=45.0,
    coverage_frac=0.50,
)

gc.collect()
#%% Monthly Availability of Number of In situ per Region

# ---------------------------------------------------------
# 1. BUILD MONTHLY UNIQUE SHIP COUNTS BY HEMISPHERE
# ---------------------------------------------------------
def build_oceanrain_monthly_ship_availability(usable_days):
    """
    From OceanRAIN usable daily data, count unique ships per month by hemisphere.
    One ship contributes at most once per hemisphere-month.
    Missing months are filled with zero.
    """
    df = usable_days.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # one ship counted once per hemi-month
    monthly = (
        df[["hemi", "ship", "month_start"]]
        .drop_duplicates()
        .groupby(["hemi", "month_start"])
        .size()
        .reset_index(name="n_ships")
    )

    hemi_order = ["NH", "SH"]

    full_months = pd.date_range(
        monthly["month_start"].min(),
        monthly["month_start"].max(),
        freq="MS"
    )

    full_index = pd.MultiIndex.from_product(
        [hemi_order, full_months],
        names=["hemi", "month_start"]
    )

    monthly = (
        monthly.set_index(["hemi", "month_start"])
               .reindex(full_index, fill_value=0)
               .reset_index()
    )

    return monthly

# =========================================================
# 2. MONTHLY UNIQUE-PLATFORM AVAILABILITY
# =========================================================
def build_monthly_platform_availability(df, region_order=None):
    """
    Count unique platform IDs per month in each region.
    One platform can contribute at most once per region-month.
    Missing months are filled with zero.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["month_start"] = out["time"].dt.to_period("M").dt.to_timestamp()

    # one row per region-ID-month
    out = (
        out[["region", "ID", "month_start"]]
        .drop_duplicates()
        .groupby(["region", "month_start"])
        .size()
        .reset_index(name="n_platforms")
    )

    if region_order is None:
        region_order = sorted(out["region"].unique())

    # full monthly range
    full_months = pd.date_range(
        out["month_start"].min(),
        out["month_start"].max(),
        freq="MS"
    )

    full_index = pd.MultiIndex.from_product(
        [region_order, full_months],
        names=["region", "month_start"]
    )

    out = (
        out.set_index(["region", "month_start"])
           .reindex(full_index, fill_value=0)
           .reset_index()
    )

    return out

def build_oceanrain_raw_monthly_ship_availability(df_raw):
    """
    Build monthly OceanRAIN ship availability by hemisphere
    from raw observational presence, not analysis-screened data.

    Rules:
    - require valid time, lat, lon, ship
    - hemisphere from raw latitude
    - one ship counted at most once per hemisphere-month
    """
    df = df_raw.copy()

    # basic validity only
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon", "ship"]).copy()

    # month and hemisphere
    df["month_start"] = df["time_utc"].dt.to_period("M").dt.to_timestamp()
    df["hemi"] = np.where(df["lat"] >= 0, "NH", "SH")

    # one ship counted once per hemi-month
    monthly = (
        df[["hemi", "ship", "month_start"]]
        .drop_duplicates()
        .groupby(["hemi", "month_start"])
        .size()
        .reset_index(name="n_ships")
    )

    hemi_order = ["NH", "SH"]

    full_months = pd.date_range(
        monthly["month_start"].min(),
        monthly["month_start"].max(),
        freq="MS"
    )

    full_index = pd.MultiIndex.from_product(
        [hemi_order, full_months],
        names=["hemi", "month_start"]
    )

    monthly = (
        monthly.set_index(["hemi", "month_start"])
               .reindex(full_index, fill_value=0)
               .reset_index()
    )

    return monthly
# ------------------------------------------------------------------
# REFINED PLOT FUNCTION
# ------------------------------------------------------------------
def plot_monthly_availability_lines(
    monthly_df,
    region_order,
    region_name_map,
    region_colors=None,
    title="Monthly Data Availability",
    ylabel="Monthly Available Platforms",
    figsize=(13, 5),
    lw=1.6,
    ylim=(0, 21),
    ytick_values=(0, 5, 10, 15, 20),
    xlim=None,
    major_year_interval=5,
    minor_year_interval=1,
    tick_fontsize=15,
    title_fontsize=22,
    label_fontsize=17,
    legend_fontsize=15,
    legend_ncol=2,
    step_where="mid",
    ax=None
):
    """
    Plot monthly availability lines for regions on one axis.
    Legend is placed outside below the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if region_colors is None:
        region_colors = {}

    # plot each region
    for region in region_order:
        sub = (
            monthly_df[monthly_df["region"] == region]
            .sort_values("month_start")
        )

        ax.step(
            sub["month_start"],
            sub["n_platforms"],
            where=step_where,
            lw=lw,
            color=region_colors.get(region, None),
            label=region_name_map.get(region, region)
        )

    # titles and labels
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

    # y-axis
    ax.set_ylim(*ylim)
    ax.set_yticks(list(ytick_values))

    # x-axis
    if xlim is not None:
        ax.set_xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))

    ax.xaxis.set_major_locator(mdates.YearLocator(base=major_year_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(base=minor_year_interval))

    # ticks
    ax.tick_params(axis="both", which="major",
                   labelsize=tick_fontsize, length=6, width=1.1,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   length=3.5, width=0.9,
                   direction="in", top=True, right=True)

    # spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    # no margins on x
    ax.margins(x=0)

    # legend outside below
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=legend_ncol,
        frameon=False,
        fontsize=legend_fontsize
    )

    plt.tight_layout()
    return fig, ax

def plot_oceanrain_monthly_availability_raw(
    monthly_df,
    hemi_name_map=None,
    hemi_colors=None,
    title="OceanRAIN Number",
    ylabel="Monthly Available Ships",
    figsize=(13, 5),
    lw=1.8,
    ylim=None,
    ytick_values=None,
    xlim=None,
    major_year_interval=1,
    minor_year_interval=1,
    tick_fontsize=18,
    title_fontsize=28,
    label_fontsize=20,
    legend_fontsize=18,
    step_where="mid"
):
    if hemi_name_map is None:
        hemi_name_map = {
            "NH": "Northern Hemisphere",
            "SH": "Southern Hemisphere"
        }

    if hemi_colors is None:
        hemi_colors = {
            "NH": "#555555",
            "SH": "#cc6666"
        }

    fig, ax = plt.subplots(figsize=figsize)

    for hemi in ["NH", "SH"]:
        sub = monthly_df[monthly_df["hemi"] == hemi].sort_values("month_start")

        ax.step(
            sub["month_start"],
            sub["n_ships"],
            where=step_where,
            lw=lw,
            color=hemi_colors.get(hemi, None),
            label=hemi_name_map.get(hemi, hemi)
        )

    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

    if ylim is not None:
        ax.set_ylim(*ylim)
    if ytick_values is not None:
        ax.set_yticks(list(ytick_values))

    if xlim is not None:
        ax.set_xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))

    ax.xaxis.set_major_locator(mdates.YearLocator(base=major_year_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(base=minor_year_interval))

    ax.tick_params(axis="both", which="major",
                   labelsize=tick_fontsize, length=6, width=1.1,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   length=3.5, width=0.9,
                   direction="in", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.margins(x=0)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize
    )
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    plt.tight_layout()
    return fig, ax
# ------------------------------------------------------------------

path_to_plots = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots_27Jan2026'

# =========================================================
# 3. BUILD MONTHLY AVAILABILITY FOR BUOY AND PAL
# =========================================================
# Buoy
Buoy_dfs = pd.concat(regional_buoy_sate_dfs_daily_lst, axis=0).copy()
buoy_region_order = ["ENP", "WNP", "IND", "ATL"]   # adjust if needed

Buoy_monthly_avail = build_monthly_platform_availability(
    Buoy_dfs,
    region_order=buoy_region_order
)

# PAL
Pal_dfs = pd.concat(regional_PAL_sate_dfs_daily_lst, axis=0).copy()
pal_region_order = sorted(Pal_dfs["region"].dropna().unique())   # or provide your own order

Pal_monthly_avail = build_monthly_platform_availability(
    Pal_dfs,
    region_order=pal_region_order
)


# =========================================================
# 4. OPTIONAL COLORS
#    Set your own if you want consistency across figures


# =========================================================
# 5. PLOT BUOY
# =========================================================
fig_buoy, ax_buoy = plot_monthly_availability_lines(
    monthly_df=Buoy_monthly_avail,
    region_order=buoy_region_order,
    region_name_map=Buoy_REGION_NAMES,
    region_colors=Buoy_region_colors,
    title="Buoy Number",
    ylabel="Monthly Available Buoys",
    figsize=(13.2, 5.2),
    lw=1.6,
    ylim=(0, 21),
    ytick_values=(0, 5, 10, 15, 20),
    xlim=("1998-01-01", "2025-06-01"),   # adjust if needed
    major_year_interval=5,
    minor_year_interval=1,
    tick_fontsize=15,
    title_fontsize=22,
    label_fontsize=17,
    legend_fontsize=14,
    legend_ncol=4
)

svnme = os.path.join(path_to_plots, f'Figure_S2_buoy_availability_{cde_run_dte}.png')
fig_buoy.savefig(svnme, dpi=150, bbox_inches='tight')
# plt.show()

gc.collect()

# =========================================================
# 6. PLOT PAL
# =========================================================
fig_pal, ax_pal = plot_monthly_availability_lines(
    monthly_df=Pal_monthly_avail,
    region_order=pal_region_order,
    region_name_map=PAL_REGION_NAMES,
    region_colors=PAL_region_colors,
    title="PAL Number",
    ylabel="Monthly\nAvailable PALs",
    figsize=(13.2, 5.2),
    lw=2.6,
    ylim=(0, 21),
    ytick_values=(0, 5, 10, 15, 20),
    xlim=("2010-01-01", "2021-12-01"),   # adjust if needed
    major_year_interval=1,
    minor_year_interval=1,
    tick_fontsize=15,
    title_fontsize=22,
    label_fontsize=17,
    legend_fontsize=13,
    legend_ncol=3
)
svnme = os.path.join(path_to_plots, f'Figure_S1_pal_availability_{cde_run_dte}.png')
fig_pal.savefig(svnme, dpi=150, bbox_inches='tight')
# plt.show()
gc.collect()

# Example:
# daily_all, usable_days = oceanrain_daily_aggregate_to_gpcp_main(...)

OceanRAIN_monthly_raw_avail = build_oceanrain_raw_monthly_ship_availability(ocRain_df)

fig_or, ax_or = plot_oceanrain_monthly_availability_raw(
    OceanRAIN_monthly_raw_avail,
    hemi_name_map={
        "NH": "Northern Hemisphere",
        "SH": "Southern Hemisphere"
    },
    hemi_colors={
        "NH": "#555555",
        "SH": "#cc6666"
    },
    title="OceanRAIN Number",
    ylabel="Monthly\nAvailable Ships",
    figsize=(13.2, 5.2),
    ylim=(0, 5),                 # adjust after checking actual max
    ytick_values=(0, 1, 2, 3, 4, 5),
    xlim=("2010-01-01", "2017-12-01"),
    major_year_interval=1,
    minor_year_interval=1,
    tick_fontsize=17,
    title_fontsize=22,
    label_fontsize=17,
    legend_fontsize=18
)
svnme = os.path.join(path_to_plots, f'Figure_S4_oceanrain_availability_{cde_run_dte}.png')
fig_or.savefig(svnme, dpi=150, bbox_inches='tight')
# plt.show()

gc.collect()
#%%
path_to_dat = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs_27Jan2026'
df = pd.read_pickle(os.path.join(path_to_dat, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260216.pkl'))

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