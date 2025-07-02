#%% IMPORT LIBRARIES
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

for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")

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

            df['row'], df['col'] = assign_to_gpcp_grid(df['lat'], df['lon'], 1.0)

            # Now df contains the PAL data with GPCP grid assignments
            # average daily rainfall
            
            daily_avg = df.groupby(['date', 'row', 'col'])[['rain_rate', 'lat', 'lon']].mean().reset_index()
            daily_avg = daily_avg.set_index('date')
            daily_avg['region'] = region_name
            daily_avg['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

            # collect GPCP data for this PAL
            # Vectorized approach for speed
            # Prepare arrays for lookup
            gpcp_times = gpcp_ds_v1pt3_xr['time'].values
            gpcp_lats = gpcp_ds_v1pt3_xr['latitude'].values
            gpcp_lons = gpcp_ds_v1pt3_xr['longitude'].values

            # Map PAL dates to nearest GPCP time index
            pal_dates = pd.to_datetime(daily_avg.index)
            gpcp_time_idx = np.searchsorted(gpcp_times, pal_dates)
            gpcp_time_idx = np.clip(gpcp_time_idx, 0, len(gpcp_times) - 1)

            # Map PAL lat/lon to nearest GPCP grid index
            pal_lats = daily_avg['lat'].values
            pal_lons = daily_avg['lon'].values

            gpcp_lat_idx = np.abs(gpcp_lats[:, None] - pal_lats).argmin(axis=0)
            gpcp_lon_idx = np.abs(gpcp_lons[:, None] - pal_lons).argmin(axis=0)

            # Extract GPCP values in a vectorized way
            gpcp_precip = gpcp_ds_v1pt3_xr['precip'].values
            matched_vals = gpcp_precip[gpcp_time_idx, gpcp_lat_idx, gpcp_lon_idx]

            daily_avg['gpcp_v1pt3'] = matched_vals

            pal_ds.close()

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


