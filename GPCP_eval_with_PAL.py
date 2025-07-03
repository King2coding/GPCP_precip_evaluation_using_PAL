#%% IMPORT LIBRARIES
import warnings
warnings.filterwarnings("ignore")
import importlib
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])

from util_functions import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
#%% DEFINE GLOBAL VARIABLES
all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])

# read all GPCP into a single xr data
gpcp_ds_v1pt3_xr = xr.open_mfdataset(all_gpcp_v1pt3_files, combine='by_coords', parallel=True)
gpcp_ds_v1pt3_xr = ds_swaplon(gpcp_ds_v1pt3_xr)

gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcp_v3pt2_files, combine='by_coords', parallel=True)
gpcp_ds_v3pt2_xr = ds_swaplon(gpcp_ds_v3pt2_xr)

gpcp_ds_v3pt3_xr = xr.open_mfdataset(all_gpcp_v3pt3_files, combine='by_coords', parallel=True)
gpcp_ds_v3pt3_xr = ds_swaplon(gpcp_ds_v3pt3_xr)

#%% CLASSIFY AND GROUP PAL FILES
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

#%% PLOT
# === Plot ===
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
        lat = ds['lat'].values[::10]  # every 10th point for performance
        lon = ds['lon'].values[::10]  # every 10th point for performance
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)
        ds.close()

    bounds = region_bounds[region]
    # Comment out dashed rectangular boundaries for now - difficult to achieve programmatically
    # rect = Rectangle(
    #     (bounds["lon_min"], bounds["lat_min"]),
    #     bounds["lon_max"] - bounds["lon_min"],
    #     bounds["lat_max"] - bounds["lat_min"],
    #     linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--',
    #     transform=ccrs.PlateCarree()
    # )
    # ax.add_patch(rect)

    # Remove region name and count labels from the plot
    # label_lon = bounds["lon_min"] + 2
    # label_lat = bounds["lat_max"] - 5
    # ax.text(label_lon, label_lat, f"{region}: {len(files)}", fontsize=15, color=color, transform=ccrs.PlateCarree())

# Remove unclassified files plotting since there are none
# if "Unclassified" in pals_classed_by_region and len(pals_classed_by_region["Unclassified"]) > 0:
#     color = region_colors["Unclassified"]
#     for file in pals_classed_by_region["Unclassified"]:
#         ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
#         lat = ds['lat'].values[::10]
#         lon = ds['lon'].values[::10]
#         ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)
#         ds.close()
#     
#     # Add text annotation for unclassified
#     ax.text(-170, -25, f"Unclassified: {len(pals_classed_by_region['Unclassified'])}", 
#             fontsize=15, color=color, transform=ccrs.PlateCarree())

# Add grid lines
ax.grid(True, linewidth=0.5, color='grey', alpha=0.7, linestyle='--')

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
labels = [f"{full_region_names[region]} ({len(pals_classed_by_region.get(region, []))})" for region in legend_regions]
plt.legend(handles, labels, title="Regions", loc="lower center", bbox_to_anchor=(0.5, -0.35), 
          fontsize=12, title_fontsize=14, ncol=3, frameon=False)

ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(range(-30, 61, 15), crs=ccrs.PlateCarree())  # Changed to 15 degree intervals
ax.tick_params(labelsize=18)  # Increased font size for axis tick labels
ax.set_title("PAL Trajectories by Ocean Region", fontsize=20)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Add extra space at the bottom for legend
plt.show()
gc.collect()  # Clean up memory

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

#%% SPATIOTEMPORAL MATCHING OF PAL AND GPCP DATA
regional_PAL_GPCP_dfs = {}
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
            pal_df_combined['rain_rate'] *= 24

            # retain only columns where prob_liq is == 100
            pal_df_combined = pal_df_combined[pal_df_combined['prob_liq'] == 100]

            region_pal_gpcp_dfs.append(pal_df_combined)

            pal_ds.close()

        # Combine all region PAL-GPCP dataframes into a single dataframe
        region_pal_gpcp_df = pd.concat(region_pal_gpcp_dfs, ignore_index=True)
        # calculate mean per track_PAL_id
        region_pal_gpcp_df = region_pal_gpcp_df.groupby(['track_PAL_id'])[['rain_rate', 'GPCP_v1pt3', 
                                                                           'GPCP_v3pt2', 'GPCP_v3pt3']].mean().reset_index()
        region_pal_gpcp_df['region'] = region_name  # Add region name for clarity
        regional_PAL_GPCP_dfs[region_name] = region_pal_gpcp_df
        
# Example: Retrieve the fill value (used to represent missing values) in the first GPCP file
with xr.open_dataset(all_gpcp_v1pt3_files[0]) as ds:
    fill_values = {}
    for var in ds.data_vars:
        attrs = ds[var].attrs
        # Common attribute names for fill values: '_FillValue' or 'missing_value'
        fill_val = attrs.get('_FillValue', attrs.get('missing_value', None))
        fill_values[var] = fill_val
    print("Fill value (used for missing data) per variable:")
    for var, fill_val in fill_values.items():
        print(f"{var}: {fill_val}")

#%% MINIMAL TEST: 1 PAL + 1 GPCP FILE
import pandas as pd
from datetime import datetime

print("MINIMAL TEST: 1 PAL + 1 GPCP FILE")
print("="*50)

# Step 1: Pick just 1 PAL file for testing
test_pal_file = None
test_region = None

for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        test_pal_file = pal_files[0]  # Just take the first PAL
        test_region = region_name
        break

if test_pal_file is None:
    print("No PAL files found!")
else:
    print(f"Test PAL file: {os.path.basename(test_pal_file)}")
    print(f"Test region: {test_region}")

# Step 2: Pick just 1 GPCP file for testing
test_gpcp_file = all_gpcp_v1pt3_files[0] if len(all_gpcp_v1pt3_files) > 0 else None

if test_gpcp_file is None:
    print("No GPCP files found!")
else:
    print(f"Test GPCP file: {os.path.basename(test_gpcp_file)}")

print("="*50)
print("="*50)

# Now let's examine both files step by step
if test_pal_file and test_gpcp_file:
    print("\nStep 1: Examining PAL file structure...")
    pal_ds = xr.open_dataset(test_pal_file)
    print(f"PAL variables: {list(pal_ds.variables.keys())}")
    print(f"PAL time range: {pal_ds.time.values[0]} to {pal_ds.time.values[-1]}")
    print(f"PAL data points: {len(pal_ds.time.values)}")
    
    # Get first few data points as example
    pal_time = pd.to_datetime(pal_ds['time'].values)
    pal_lat = pal_ds['lat'].values
    pal_lon = pal_ds['lon'].values
    pal_rain = pal_ds['rain_rate'].values
    
    print(f"Sample PAL data (first 3 points):")
    for i in range(min(3, len(pal_time))):
        print(f"  {pal_time[i]}: Lat={pal_lat[i]:.2f}, Lon={pal_lon[i]:.2f}, Rain={pal_rain[i]:.3f}")
    
    print("\nStep 2: Examining GPCP file structure...")
    gpcp_ds = xr.open_dataset(test_gpcp_file)
    print(f"GPCP variables: {list(gpcp_ds.variables.keys())}")
    print(f"GPCP dimensions: {dict(gpcp_ds.dims)}")
    
    # Check coordinates
    if 'time' in gpcp_ds.variables:
        print(f"GPCP time range: {gpcp_ds.time.values[0]} to {gpcp_ds.time.values[-1]}")
    
    # Check lat/lon
    lat_var = 'latitude' if 'latitude' in gpcp_ds.variables else 'lat'
    lon_var = 'longitude' if 'longitude' in gpcp_ds.variables else 'lon'
    
    if lat_var in gpcp_ds.variables and lon_var in gpcp_ds.variables:
        print(f"GPCP lat range: {gpcp_ds[lat_var].values.min():.2f} to {gpcp_ds[lat_var].values.max():.2f}")
        print(f"GPCP lon range: {gpcp_ds[lon_var].values.min():.2f} to {gpcp_ds[lon_var].values.max():.2f}")
    
    # Close datasets
    pal_ds.close()
    gpcp_ds.close()
    
    print("\nMinimal test setup complete!")
    print("Next step: Implement actual matching logic with these files.")

else:
    print("Cannot proceed - missing test files!")

gc.collect()


