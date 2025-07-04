#%% IMPORT LIBRARIES

import importlib
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])

from util_functions import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.legend import Legend
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker


#%% DEBUG: Check current region bounds and test overlap function
print("CURRENT REGION BOUNDS:")
print("="*50)
for region, bounds in region_bounds.items():
    print(f"{region}: {bounds}")

print("\nTEST OVERLAP FUNCTION:")
print("-" * 30)
# Test PAL 19412: Lat 1.04-3.09, Lon 164.90-174.91 vs TNWP
pal_lat_min, pal_lat_max = 1.04, 3.09
pal_lon_min, pal_lon_max = 164.90, 174.91

tnwp_bounds = region_bounds["TNWP"]
print(f"PAL 19412: Lat {pal_lat_min}-{pal_lat_max}, Lon {pal_lon_min}-{pal_lon_max}")
print(f"TNWP: {tnwp_bounds}")

overlap_result = simple_box_check(pal_lat_min, pal_lat_max, pal_lon_min, pal_lon_max,
                                 tnwp_bounds["lat_min"], tnwp_bounds["lat_max"],
                                 tnwp_bounds["lon_min"], tnwp_bounds["lon_max"])
print(f"Should overlap with TNWP: {overlap_result}")

# Test PAL 6874 vs TNEP  
pal2_lat_min, pal2_lat_max = 1.50, 4.59
pal2_lon_min, pal2_lon_max = -165.53, -140.20

tnep_bounds = region_bounds["TNEP"]
print(f"\nPAL 6874: Lat {pal2_lat_min}-{pal2_lat_max}, Lon {pal2_lon_min}-{pal2_lon_max}")
print(f"TNEP: {tnep_bounds}")

overlap_result2 = simple_box_check(pal2_lat_min, pal2_lat_max, pal2_lon_min, pal2_lon_max,
                                  tnep_bounds["lat_min"], tnep_bounds["lat_max"],
                                  tnep_bounds["lon_min"], tnep_bounds["lon_max"])
print(f"Should overlap with TNEP: {overlap_result2}")
print("="*50)

#%% DEFINE PATH TO DATA
path_to_pal_data = r'/ra1/pubdat/GPCP_eval_with_PAL/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/GPCP_eval_with_PAL/data/GPCP/GPCP_v1_pnt_3_2010_2020'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/GPCP_eval_with_PAL/data/GPCP/GPCP_v3_pnt_2_2010_2020'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/GPCP_eval_with_PAL/data/GPCP/GPCP_v3_pnt_3_2010_2020'

path_to_put_plts = r'/home/kkumah/Projects/GPCP_ocean_evaluation_study/Results/plots'
#%% DEFINE GLOBAL VARIABLES
all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
# read all GPCP into a single xr data
gpcp_ds_v1pt3_xr = xr.open_mfdataset(all_gpcp_v1pt3_files, combine='by_coords', parallel=True)
gpcp_ds_v1pt3_xr = ds_swaplon(gpcp_ds_v1pt3_xr)

gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcp_v3pt2_files, combine='by_coords', parallel=True)
gpcp_ds_v3pt2_xr = ds_swaplon(gpcp_ds_v3pt2_xr)

gpcp_ds_v3pt3_xr = xr.open_mfdataset(all_gpcp_v3pt3_files, combine='by_coords', parallel=True)
gpcp_ds_v3pt3_xr = ds_swaplon(gpcp_ds_v3pt3_xr)

gc.collect()  # Clean up memory
#%% CLASSIFY AND GROUP PAL FILES
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

# CLASSIFICATION SUMMARY
# Print classification summary
print("\n" + "="*50)
print("PAL CLASSIFICATION SUMMARY")
print("="*50)
for region, files in pals_classed_by_region.items():
    if region != "Unclassified":
        print(f"{region}: {len(files)} PALs")
print("="*50)

# Expected counts from your figure:
expected_counts = {
    "ETNP": 4,   # Extratropical North Pacific
    "TNEP": 20,  # Tropical Northeastern Pacific  
    "TSEP": 6,   # Tropical Southeastern Pacific
    "STNA": 18,  # Subtropical North Atlantic
    "TNIO": 3,   # Tropical North Indian Ocean
    "TNWP": 7    # Tropical Northwestern Pacific
}

print("\nCOMPARISON WITH EXPECTED COUNTS:")
print("-" * 30)
total_expected = sum(expected_counts.values())
total_actual = sum(len(files) for region, files in pals_classed_by_region.items() if region != "Unclassified")

for region in expected_counts:
    actual = len(pals_classed_by_region.get(region, []))
    expected = expected_counts[region]
    status = "✓" if actual == expected else "✗"
    print(f"{region}: Expected {expected}, Got {actual} {status}")

print(f"\nTotal Expected: {total_expected}")
print(f"Total Actual: {total_actual}")
print(f"Difference: {total_actual - total_expected}")

gc.collect()  # Clean up memory
#%% PLOT - FIGURE 1
# === Plot ===
# Set font to Times New Roman and bold for all texts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
# Remove unavailable Times New Roman to avoid findfont warnings

fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for region, files in pals_classed_by_region.items():
    if region == "Unclassified" or len(files) == 0:
        continue  # Skip unclassified and empty regions for plotting bounds
    
    color = region_colors[region]
    for file in files:
        # Load only lat and lon efficiently, downsample by slicing
        ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
        lat = ds['lat'].values[::50]  # every 50th point for performance
        lon = ds['lon'].values[::50]  # every 50th point for performance
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=3)
        ds.close()

# Add grid lines for major ticks
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.7, linestyle='--')

# Create legend with full region names and PAL counts
legend_regions = [r for r in region_colors.keys() if r != "Unclassified"]
handles = [plt.Line2D([0], [0], color=region_colors[r], lw=2) for r in legend_regions]

# Full region names mapping
full_region_names = {
    "ETNP": "Extratropical North Pacific",
    "TNEP": "Tropical Northeastern Pacific", 
    "TSEP": "Tropical Southeastern Pacific",
    "STNA": "Subtropical North Atlantic",
    "TNIO": "Tropical North Indian Ocean",
    "TNWP": "Tropical Northwestern Pacific"
}

# Add full names and PAL counts to legend labels
labels = [f"{region}: ({full_region_names[region]} ({len(pals_classed_by_region.get(region, []))})" for region in legend_regions]
leg = plt.legend(
    handles, labels, title="Regions", loc="lower center", bbox_to_anchor=(0.5, -0.35), 
    fontsize=14, title_fontsize=14, ncol=3, frameon=False
)
# Set legend fontweight to bold
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

# Set ticks and format them with degree symbols and N/S/E/W
xticks = range(-180, 181, 60)
yticks = range(-30, 61, 15)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

ax.tick_params(labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

ax.set_title("PAL Trajectories by Ocean Region", fontsize=20, 
             fontweight='bold', fontname='Times New Roman')

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom for legend

svname = os.path.join(path_to_put_plts, 'PAL_trajectories_by_region.png')
plt.savefig(svname, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()  # Clean up memory


#%% DO DATA INVENTORY PER REGION
# COUNT THE TOTAL NUMBER OF DAYS PER YEAR WITH NON NAN DATA FOR EACH REGION
regional_inventory = []
for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")

        # store PAL and GPCP dataframes
        region_pal_gpcp_dfs = []     

        # LOAD PAL DATA
        for pal_file in pal_files:
            pal_ds = xr.open_dataset(pal_file)

            # Process PAL data as needed
            df = pd.DataFrame({
                'time': pd.to_datetime(pal_ds['time'].values),
                'lat': pal_ds['lat'].values,
                'lon': pal_ds['lon'].values,
                'rain_rate': pal_ds['rain_rate'].values
            })

            df['date'] = df['time'].dt.date  # Extract date from time

            df['year'] = df['time'].dt.year  # Extract year from time

            df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values           

            # Normalize longitude to [-180, 180]
            df['lon'] = (df['lon'] + 360) % 360
            df['lon'][df['lon'] > 180] -= 360

            df_avg = df.groupby(['year', 'date'])[['rain_rate', 'lat', 'lon']].mean()

            df_avg['region'] = region_name  # Add region name for clarity
            df_avg['pal_file'] = os.path.basename(pal_file)  # Add PAL file name for clarity


            # Count unique dates with non-NaN rain_rate
            # unique_dates = df['date'].nunique()
            
            # Store the count in the inventory list
            regional_inventory.append(df_avg)

            pal_ds.close()
gc.collect()  # Clean up memory

# Combine all region PAL-GPCP dataframes into a single dataframe
regional_inventory_df = pd.concat(regional_inventory)
region_counts = regional_inventory_df.groupby('region')['pal_file'].nunique().to_dict()

# groupby region and  year, then count all days with non-NaN rain_rate
regional_inventory_df = regional_inventory_df.groupby(['region', 'year'])[['rain_rate']].count().reset_index()

# plot bar plot of year on x axis and count of days with non-NaN rain_rate on y axis
# comparing regions
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
# Removed 'Times New Roman' to avoid findfont warnings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18

fg, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=regional_inventory_df, x='year', y='rain_rate', hue='region', 
            palette=region_colors, ax=ax)
ax.set_xlabel('Year', fontsize=18, fontweight='bold')
ax.set_ylabel('Total Number of\n  daily observations', fontsize=15, fontweight='bold')
ax.set_title('Yearly distribution of daily observations by Region', fontsize=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=18, )
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.grid(True, alpha=0.3)
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_zorder(2)
ax.spines['left'].set_zorder(2)
ax.set_zorder(1)
# Slant the x-axis tick labels for readability
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
# Set y-axis to log scale for readability
# ax.set_yscale('log')

# Add legend with PAL counts per region (remove the default legend first)
handles, labels_ = ax.get_legend_handles_labels()
ax.legend_.remove()  # Remove the default legend

labels_with_counts = [
    f"{label} ({region_counts.get(label, 0)} PALs)" for label in labels_
]
ax.legend(handles, labels_with_counts, title='Regions', loc='upper center', bbox_to_anchor=(0.5, -0.25), 
          fontsize=14, title_fontsize=14, ncol=3, frameon=False)
# # Add legend
# handles, labels_ = ax.get_legend_handles_labels()
# ax.legend(handles, labels_, title='Regions', loc='upper center', bbox_to_anchor=(0.5, -0.15), 
#           fontsize=14, title_fontsize=14, ncol=3, frameon=False)
plt.tight_layout()
# Set legend fontweight to bold
for text in ax.get_legend().get_texts():
    text.set_fontweight('bold')
svname = os.path.join(path_to_put_plts, 'region_daily_observation_inventory.png')
plt.savefig(svname, dpi=500, bbox_inches='tight')
gc.collect()  # Clean up memory
#%% SPATIOTEMPORAL MATCHING OF PAL AND GPCP DATA
regional_PAL_GPCP_dfs_daily_mean = {}
regional_PAL_GPCP_dfs_daily_lst = []
for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")

        # store PAL and GPCP dataframes
        region_pal_gpcp_dfs = []     

        # LOAD PAL DATA
        for pal_file in pal_files:
            pal_ds = xr.open_dataset(pal_file)

            # Process PAL data as needed
            df = pd.DataFrame({
                'time': pd.to_datetime(pal_ds['time'].values),
                'lat': pal_ds['lat'].values,
                'lon': pal_ds['lon'].values,
                'rain_rate': pal_ds['rain_rate'].values
            })

            df['date'] = df['time'].dt.date  # Extract date from time

            # nan_df = df.copy()  # Keep a copy for debugging
            # nan_df = nan_df[nan_df.isna().any(axis=1)]  # Find rows with NaN values

            df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values           

            # Normalize longitude to [-180, 180]
            df['lon'] = (df['lon'] + 360) % 360
            df['lon'][df['lon'] > 180] -= 360

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            # Process GPCP data with PAL
            
            pal_df_gpcpv1pt3 = df.copy()

            # get the resolution of the GPCP data
            resol_gpcpv1pt3 = np.unique(np.diff(gpcp_ds_v1pt3_xr['longitude'].values))[0]

            pal_gpcpv1pt3_daily_avg = process_gpcp_with_PAL(pal_file, region_name, 
                                                            pal_df_gpcpv1pt3, gpcp_ds_v1pt3_xr, 
                                                            resol_gpcpv1pt3, 'GPCP_v1pt3') 
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            pal_df_gpcpv3pt2 = df.copy()   

            resol_gpcpv3pt2 = np.unique(np.diff(gpcp_ds_v3pt2_xr['lon'].values))[0]
            
            pal_gpcpv3pt2_daily_avg = process_gpcp_with_PAL(pal_file, region_name, 
                                                            pal_df_gpcpv3pt2, gpcp_ds_v3pt2_xr, resol_gpcpv3pt2,
                                                            'GPCP_v3pt2')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            pal_df_gpcpv3pt3 = df.copy()           

            resol_gpcpv3pt3 = np.unique(np.diff(gpcp_ds_v3pt3_xr['lon'].values))[0]

            pal_gpcpv3pt3_daily_avg = process_gpcp_with_PAL(pal_file, region_name,
                                                            pal_df_gpcpv3pt3, gpcp_ds_v3pt3_xr, 
                                                            resol_gpcpv3pt3, 'GPCP_v3pt3') 

            # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data
            # Use pd.merge to combine on 'date' after selecting only relevant columns
            pal_df_combined = pal_gpcpv1pt3_daily_avg.copy()
            pal_df_combined = pal_df_combined[['rain_rate', 'region', 'track_PAL_id', 'GPCP_v1pt3']].copy()
            # pal_df_combined['region'] = region_name  # Add region name for clarity            
            
            # Merge GPCP_v3pt2, always retain prob_liq, but avoid duplicate columns
            pal_df_combined = pal_df_combined.merge(
                pal_gpcpv3pt2_daily_avg[['GPCP_v3pt2', 'prob_liq']], 
                left_index=True, right_index=True, how='left', suffixes=('', '_v3pt2')
            )
            # Remove any duplicate columns from previous merges
            for col in ['GPCP_v3pt2_v3pt2', 'prob_liq_v3pt2']:
                if col in pal_df_combined.columns:
                    pal_df_combined.drop(columns=col, inplace=True)

            # Merge GPCP_v3pt3, avoid duplicate columns
            pal_df_combined = pal_df_combined.merge(
                pal_gpcpv3pt3_daily_avg[['GPCP_v3pt3']], 
                left_index=True, right_index=True, how='left', suffixes=('', '_v3pt3')
            )
            if 'GPCP_v3pt3_v3pt3' in pal_df_combined.columns:
                pal_df_combined.drop(columns=['GPCP_v3pt3_v3pt3'], inplace=True)
            
            # multiply PAL rain rate by 24 to get daily average
            # pal_df_combined['rain_rate'] *= 24

            # retain only columns where prob_liq is == 100
            pal_df_combined = pal_df_combined[pal_df_combined['prob_liq'] == 100]

            region_pal_gpcp_dfs.append(pal_df_combined)

            pal_ds.close()

        # Combine all region PAL-GPCP dataframes into a single dataframe
        region_pal_gpcp_df = pd.concat(region_pal_gpcp_dfs)
        
        
        # calculate daily mean per track_PAL_id
        region_pal_gpcp_df_daily_mean = region_pal_gpcp_df.groupby(['track_PAL_id'])[['rain_rate', 'GPCP_v1pt3', 
                                                                           'GPCP_v3pt2', 'GPCP_v3pt3']].mean().reset_index()
        region_pal_gpcp_df_daily_mean['region'] = region_name  # Add region name for clarity
        regional_PAL_GPCP_dfs_daily_mean[region_name] = region_pal_gpcp_df_daily_mean

        # Append to the list for later processing
        regional_PAL_GPCP_dfs_daily_lst.append(region_pal_gpcp_df)

gc.collect()  # Clean up memory


# # Example: Retrieve the fill value (used to represent missing values) in the first GPCP file
# with xr.open_dataset(all_gpcp_v3pt2_files[0]) as ds:
#     # Access the variable (e.g., 'precip')
#     var = ds["precip"]

#     # Get the missing_value or _FillValue attribute
#     missing_val = var.attrs.get("missing_value") or var.attrs.get("_FillValue")

#     print(f"Missing value: {missing_val}")
#%%
# calculate RB, RMSE and CC using all PAL-GPCP pairs in axes plot and show this in the plot
# as RB = , RMSE = , CC =
#- - - - -   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
pal_gpcv1_3_cmp = pd.concat([df[['rain_rate', 'GPCP_v1pt3']] for df in regional_PAL_GPCP_dfs_daily_mean.values()], ignore_index=True)
pal_gpcv3_2_cmp = pd.concat([df[['rain_rate', 'GPCP_v3pt2']] for df in regional_PAL_GPCP_dfs_daily_mean.values()], ignore_index=True)
pal_gpcv3_3_cmp = pd.concat([df[['rain_rate', 'GPCP_v3pt3']] for df in regional_PAL_GPCP_dfs_daily_mean.values()], ignore_index=True)

# calcute metrics for all regions combined
rb_v1pt3, rmse_v1pt3, cc_v1pt3 = calculate_metrics(pal_gpcv1_3_cmp['rain_rate'], pal_gpcv1_3_cmp['GPCP_v1pt3'])
rb_v3pt2, rmse_v3pt2, cc_v3pt2 = calculate_metrics(pal_gpcv3_2_cmp['rain_rate'], pal_gpcv3_2_cmp['GPCP_v3pt2'])
rb_v3pt3, rmse_v3pt3, cc_v3pt3 = calculate_metrics(pal_gpcv3_3_cmp['rain_rate'], pal_gpcv3_3_cmp['GPCP_v3pt3'])

#- - - - -   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# MAKE SCATTER PLOTS (PAL vs GPCP) - DAILY MEAN - ALL REGIONS - FIGURE 2
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
# Removed 'Times New Roman' to avoid findfont warnings
mpl.rcParams['ytick.labelsize'] = 18

fg, ax = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'wspace': 0.35})  # Increased wspace for wider interval

for region, df in regional_PAL_GPCP_dfs_daily_mean.items():
    reg_col = region_colors[region]
    pal_gpcv1_3 = df[['rain_rate', 'GPCP_v1pt3']]
    pal_gpcv3_2 = df[['rain_rate', 'GPCP_v3pt2']]
    pal_gpcv3_3 = df[['rain_rate', 'GPCP_v3pt3']]

    # Plot GPCP v1.3
    ax[0].scatter(pal_gpcv1_3['rain_rate'], pal_gpcv1_3['GPCP_v1pt3'],
                  color=reg_col, label=region, s=80)
    # Plot GPCP v3.2
    ax[1].scatter(pal_gpcv3_2['rain_rate'] , pal_gpcv3_2['GPCP_v3pt2'],
                  color=reg_col, label=region, s=80)
    # Plot GPCP v3.3
    ax[2].scatter(pal_gpcv3_3['rain_rate'], pal_gpcv3_3['GPCP_v3pt3'],
                  color=reg_col, label=region, s=80)

# Set axes limits, ticks, grids, and major ticks for all subplots
for i, a in enumerate(ax):
    a.set_xlim(0, 15)
    a.set_ylim(0, 15)
    a.set_xticks([0, 5, 10, 15])
    a.set_yticks([0, 5, 10, 15])
    a.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    a.minorticks_on()
    a.tick_params(axis='both', which='major', length=7, width=1.2, labelsize=18)
    a.tick_params(axis='both', which='minor', length=4, width=0.8)
    # add 1:1 line
    x = np.linspace(0, 15, 100)
    a.plot(x, x, color='gray', linestyle='--')
    # Set axis labels and title with bold fontweight
    if i == 0:
        a.set_xlabel('PAL Observations [mm/day]', fontsize=18, fontweight='bold')
        a.set_ylabel('GPCP v1.3 Estimates [mm/day]', fontsize=18, fontweight='bold')
        a.set_title('GPCP v1.3 vs PAL', fontsize=20, fontweight='bold')
    elif i == 1:
        a.set_xlabel('PAL Observations [mm/day]', fontsize=18, fontweight='bold')
        a.set_ylabel('GPCP v3.2 Estimates [mm/day]', fontsize=18, fontweight='bold')
        a.set_title('GPCP v3.2 vs PAL', fontsize=20, fontweight='bold')
    elif i == 2:
        a.set_xlabel('PAL Observations [mm/day]', fontsize=18, fontweight='bold')
        a.set_ylabel('GPCP v3.3 Estimates [mm/day]', fontsize=18, fontweight='bold')
        a.set_title('GPCP v3.3 vs PAL', fontsize=20, fontweight='bold')
    # Make tick labels bold
    for label in a.get_xticklabels() + a.get_yticklabels():
        label.set_fontweight('bold')

# Add metrics text to each subplot, with RMSE unit, bold font
ax[0].text(
    0.05, 0.95,
    f'RB: {rb_v1pt3:.2f}%\nRMSE: {rmse_v1pt3:.2f} mm/day\nCC: {cc_v1pt3:.2f}',
    transform=ax[0].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)
ax[1].text(
    0.05, 0.95,
    f'RB: {rb_v3pt2:.2f}%\nRMSE: {rmse_v3pt2:.2f} mm/day\nCC: {cc_v3pt2:.2f}',
    transform=ax[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)
ax[2].text(
    0.05, 0.95,
    f'RB: {rb_v3pt3:.2f}%\nRMSE: {rmse_v3pt3:.2f} mm/day\nCC: {cc_v3pt3:.2f}',
    transform=ax[2].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

# Add legend
handles, labels_ = ax[0].get_legend_handles_labels()
unique_labels = dict(zip(labels_, handles))  # Remove duplicates
# Place a common legend below and outside the plot
leg = fg.legend(
    unique_labels.values(), unique_labels.keys(),
    loc='lower center', bbox_to_anchor=(0.5, -0.15),
    fontsize=18,  ncol=6, frameon=False
)
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

# plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend
# save the figure
svnme = os.path.join(path_to_put_plts, 'PAL_GPCP_scatter_plots.png')
plt.savefig(svnme, bbox_inches='tight', dpi=500)
gc.collect()  # Clean up memory

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# calculate monhtly mean per region
# Calculate monthly statistics (mean, Q1, Q3) for each region and product
monthly_stats = {}

# Prepare a DataFrame with all daily data
all_daily = pd.concat(regional_PAL_GPCP_dfs_daily_lst)
all_daily = all_daily.reset_index()  # Ensure 'date' is a column after concat
if 'date' not in all_daily.columns:
    all_daily['date'] = pd.to_datetime(all_daily['index'])
else:
    all_daily['date'] = pd.to_datetime(all_daily['date'])
all_daily['month'] = all_daily['date'].dt.month
all_daily['year'] = all_daily['date'].dt.year

regions = [r for r in regional_PAL_GPCP_dfs_daily_mean.keys()]

for region in regions:
    df = all_daily[all_daily['region'] == region].copy()
    # Compute monthly accumulation per PAL (track_PAL_id)
    pal_monthly = df.groupby(['track_PAL_id', 'year', 'month'])[['rain_rate', 'GPCP_v1pt3', 'GPCP_v3pt2', 'GPCP_v3pt3']].sum().reset_index()
    # Now, for each month, compute mean, Q1, Q3 across PALs (i.e., for each month, use all PALs' accumulations)
    stats = pal_monthly.groupby('month').agg({
        'rain_rate': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        'GPCP_v1pt3': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        'GPCP_v3pt2': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        'GPCP_v3pt3': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
    })
    stats.columns = ['_'.join(col).rstrip('_') for col in stats.columns.values]
    monthly_stats[region] = stats.reset_index()
    # Also store the PAL monthly accumulations for scatter/vertical lines
    monthly_stats[region + '_pal_monthly'] = pal_monthly

# Plotting (mimic the attached figure)

region_titles = {
    "TNEP": "(a) Tropical Northeastern Pacific (TNEP)",
    "TSEP": "(b) Tropical Southeastern Pacific (TSEP)",
    "TNWP": "(c) Tropical Northwestern Pacific (TNWP)",
    "ETNP": "(d) Extratropical North Pacific (ETNP)",
    "TNIO": "(e) Tropical North Indian Ocean (TNIO)",
    "STNA": "(f) Subtropical North Atlantic (STNA)"
}
region_order = ["TNEP", "TNWP", "TNIO", "TSEP", "ETNP", "STNA"]

# Use a serif font as a fallback if Times New Roman is not available
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

fig, axs = plt.subplots(2, 3, figsize=(25, 10), sharex=True)
axs = axs.flatten()
months = np.arange(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i, region in enumerate(region_order):
    ax = axs[i]
    stats = monthly_stats[region]
    pal_monthly = monthly_stats[region + '_pal_monthly']
    # PAL: for each month, plot mean as a point, and Q1-Q3 as vertical line
    pal_mean = stats['rain_rate_mean'].values
    pal_q1 = stats['rain_rate_q1'].values
    pal_q3 = stats['rain_rate_q3'].values
    # Plot PAL mean as scatter, Q1-Q3 as vertical line
    for m_idx, m in enumerate(months):
        mean = pal_mean[m_idx]
        q1 = pal_q1[m_idx]
        q3 = pal_q3[m_idx]
        # Only plot if not nan
        if not np.isnan(mean):
            ax.scatter(m, mean, color='k', s=40, zorder=4, label='PAL Obs. Mean' if m_idx == 0 else None)
            ax.vlines(m, q1, q3, color='gray', lw=2, zorder=3, label='PAL Obs. [Q1,Q3]' if m_idx == 0 else None)
    # GPCP v3.2
    ax.plot(months, stats['GPCP_v3pt2_mean'], color='b', label='GPCP v3.2 Est. Mean')
    ax.fill_between(months, stats['GPCP_v3pt2_q1'], stats['GPCP_v3pt2_q3'], color='b', alpha=0.2, label='GPCP v3.2 Est. [Q1,Q3]')
    # GPCP v1.3
    ax.plot(months, stats['GPCP_v1pt3_mean'], color='r', label='GPCP v1.3 Est. Mean')
    ax.fill_between(months, stats['GPCP_v1pt3_q1'], stats['GPCP_v1pt3_q3'], color='r', alpha=0.2, label='GPCP v1.3 Est. [Q1,Q3]')
    # GPCP v3.3
    ax.plot(months, stats['GPCP_v3pt3_mean'], color='g', label='GPCP v3.3 Est. Mean')
    ax.fill_between(months, stats['GPCP_v3pt3_q1'], stats['GPCP_v3pt3_q3'], color='g', alpha=0.2, label='GPCP v3.3 Est. [Q1,Q3]')
    # Title, labels
    ax.set_title(region_titles[region], fontsize=18, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels,  fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel('Rainfall [mm]', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Make y-axis tick labels bold
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add N= count
    n_pal = len(pals_classed_by_region[region])
    ax.text(0.02, 0.95, f'N = {n_pal} PALs', 
            transform=ax.transAxes, fontsize=18, 
            fontweight='bold', va='top')

# Collect legend handles/labels from the first axis
handles, labels = axs[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
# Create the legend object
leg = fig.legend(
    unique.values(), unique.keys(),
    loc='lower center', bbox_to_anchor=(0.5, -0.05),
    fontsize=20, ncol=4, frameon=False
)

# Set fontweight to bold for all legend texts
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

plt.tight_layout(rect=[0, 0.08, 1, 1])
# Save plot to disk
monthly_plot_path = os.path.join(path_to_put_plts, 'PAL_GPCP_monthly_stats_by_region.png')
plt.savefig(monthly_plot_path, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()  # Clean up memory
#%% MINIMAL TEST: 1 PAL + 1 GPCP FILE
# import pandas as pd
# from datetime import datetime

# print("MINIMAL TEST: 1 PAL + 1 GPCP FILE")
# print("="*50)

# # Step 1: Pick just 1 PAL file for testing
# test_pal_file = None
# test_region = None

# for region_name, pal_files in pals_classed_by_region.items():
#     if region_name != "Unclassified" and len(pal_files) > 0:
#         test_pal_file = pal_files[0]  # Just take the first PAL
#         test_region = region_name
#         break

# if test_pal_file is None:
#     print("No PAL files found!")
# else:
#     print(f"Test PAL file: {os.path.basename(test_pal_file)}")
#     print(f"Test region: {test_region}")

# # Step 2: Pick just 1 GPCP file for testing
# test_gpcp_file = all_gpcp_v1pt3_files[0] if len(all_gpcp_v1pt3_files) > 0 else None

# if test_gpcp_file is None:
#     print("No GPCP files found!")
# else:
#     print(f"Test GPCP file: {os.path.basename(test_gpcp_file)}")

# print("="*50)
# print("="*50)

# # Now let's examine both files step by step
# if test_pal_file and test_gpcp_file:
#     print("\nStep 1: Examining PAL file structure...")
#     pal_ds = xr.open_dataset(test_pal_file)
#     print(f"PAL variables: {list(pal_ds.variables.keys())}")
#     print(f"PAL time range: {pal_ds.time.values[0]} to {pal_ds.time.values[-1]}")
#     print(f"PAL data points: {len(pal_ds.time.values)}")
    
#     # Get first few data points as example
#     pal_time = pd.to_datetime(pal_ds['time'].values)
#     pal_lat = pal_ds['lat'].values
#     pal_lon = pal_ds['lon'].values
#     pal_rain = pal_ds['rain_rate'].values
    
#     print(f"Sample PAL data (first 3 points):")
#     for i in range(min(3, len(pal_time))):
#         print(f"  {pal_time[i]}: Lat={pal_lat[i]:.2f}, Lon={pal_lon[i]:.2f}, Rain={pal_rain[i]:.3f}")
    
#     print("\nStep 2: Examining GPCP file structure...")
#     gpcp_ds = xr.open_dataset(test_gpcp_file)
#     print(f"GPCP variables: {list(gpcp_ds.variables.keys())}")
#     print(f"GPCP dimensions: {dict(gpcp_ds.dims)}")
    
#     # Check coordinates
#     if 'time' in gpcp_ds.variables:
#         print(f"GPCP time range: {gpcp_ds.time.values[0]} to {gpcp_ds.time.values[-1]}")
    
#     # Check lat/lon
#     lat_var = 'latitude' if 'latitude' in gpcp_ds.variables else 'lat'
#     lon_var = 'longitude' if 'longitude' in gpcp_ds.variables else 'lon'
    
#     if lat_var in gpcp_ds.variables and lon_var in gpcp_ds.variables:
#         print(f"GPCP lat range: {gpcp_ds[lat_var].values.min():.2f} to {gpcp_ds[lat_var].values.max():.2f}")
#         print(f"GPCP lon range: {gpcp_ds[lon_var].values.min():.2f} to {gpcp_ds[lon_var].values.max():.2f}")
    
#     # Close datasets
#     pal_ds.close()
#     gpcp_ds.close()
    
#     print("\nMinimal test setup complete!")
#     print("Next step: Implement actual matching logic with these files.")

# else:
#     print("Cannot proceed - missing test files!")

# gc.collect()


