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

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc')])
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
#%% ASSIGN PAL TO GPCP GRID AND CALCULATE SPACE-TIME MEAN OF PAL
import pandas as pd
from datetime import datetime

# Initialize list to store all PAL-GPCP paired data
all_pal_gpcp_data = []

print("Processing PAL-GPCP matching (TESTING WITH SAMPLE)...")
print("="*50)

# TESTING: Process only 1-2 PALs from each region for debugging
for region_name, pal_files in pals_classed_by_region.items():
    if region_name == "Unclassified" or len(pal_files) == 0:
        continue
    
    # LIMIT TO FIRST 2 FILES PER REGION FOR TESTING
    sample_files = pal_files[:2]  # Only process first 2 PALs per region
    
    print(f"\nProcessing {region_name} region ({len(sample_files)} PALs - SAMPLE)...")
    
    for pal_idx, pal_file in enumerate(sample_files):
        pal_id = os.path.basename(pal_file).split('.')[0]  # Extract PAL ID from filename
        print(f"  Processing PAL {pal_id} ({pal_idx+1}/{len(pal_files)})")
        
        # Load PAL data
        pal_ds = xr.open_dataset(pal_file)
        
        # Get PAL data (assuming standard variable names - adjust as needed)
        pal_time = pd.to_datetime(pal_ds['time'].values)
        pal_lat = pal_ds['lat'].values
        pal_lon = pal_ds['lon'].values
        
        # Normalize PAL longitude to [-180, 180]
        pal_lon = (pal_lon + 360) % 360
        pal_lon[pal_lon > 180] -= 360
        
        # Get precipitation variable (adjust variable name as needed)
        precip_vars = ['rain_rate', 'precip', 'precipitation', 'rain', 'rainfall', 'pcp']
        pal_precip = None
        for var in precip_vars:
            if var in pal_ds.variables:
                pal_precip = pal_ds[var].values
                break
        
        if pal_precip is None:
            print(f"    Warning: No precipitation variable found in {pal_id}")
            print(f"    Available variables: {list(pal_ds.variables.keys())}")
            continue
        
        # Create PAL DataFrame
        pal_df = pd.DataFrame({
            'time': pal_time,
            'lat': pal_lat,
            'lon': pal_lon,
            'pal_precip': pal_precip,
            'pal_id': pal_id,
            'region': region_name
        })
        
        # Remove invalid data
        pal_df = pal_df.dropna()
        
        # Add date column for daily grouping
        pal_df['date'] = pal_df['time'].dt.date
        
        # Group by date and calculate daily means for PAL
        daily_pal = pal_df.groupby('date').agg({
            'lat': 'mean',
            'lon': 'mean', 
            'pal_precip': 'mean',
            'pal_id': 'first',
            'region': 'first'
        }).reset_index()
        
        # TESTING: Limit to first 10 days for faster processing
        daily_pal = daily_pal.head(10)
        print(f"    TESTING: Processing only {len(daily_pal)} daily observations")
        
        # Get PAL operational period
        pal_start_date = pal_time.min().date()
        pal_end_date = pal_time.max().date()
        print(f"    PAL operational period: {pal_start_date} to {pal_end_date}")
        
        # Filter GPCP files to only those within PAL operational period
        relevant_gpcp_files = []
        
        # TESTING: Only process GPCP files from a limited year range for faster testing
        test_year_start = 2010
        test_year_end = 2012
        print(f"    TESTING: Only processing GPCP files from {test_year_start}-{test_year_end}")
        
        for gpcp_file in all_gpcp_v1pt3_files[:20]:  # Also limit the total files to check
            # Extract year and month from GPCP filename (adjust pattern as needed)
            filename = os.path.basename(gpcp_file)
            # Assuming filename format like: GPCP_v1.3_2010_01.nc or similar
            try:
                # Extract year-month from filename
                if 'GPCP' in filename:
                    parts = filename.split('_')
                    for i, part in enumerate(parts):
                        if part.isdigit() and len(part) == 4:  # Found year
                            year = int(part)
                            
                            # TESTING: Only process files within test year range
                            if year < test_year_start or year > test_year_end:
                                break
                                
                            if i + 1 < len(parts) and parts[i + 1].replace('.nc', '').isdigit():
                                month = int(parts[i + 1].replace('.nc', ''))
                                file_date = pd.Timestamp(year, month, 1).date()
                                
                                # Check if this GPCP file overlaps with PAL period
                                if (file_date.year >= pal_start_date.year and 
                                    file_date.year <= pal_end_date.year):
                                    relevant_gpcp_files.append(gpcp_file)
                                break
            except:
                continue
        
        print(f"    Found {len(relevant_gpcp_files)} relevant GPCP files for PAL period")
        
        # TESTING: Limit to first 3 GPCP files for faster testing
        relevant_gpcp_files = relevant_gpcp_files[:3]
        print(f"    TESTING: Processing only {len(relevant_gpcp_files)} GPCP files")
        
        if len(relevant_gpcp_files) == 0:
            print(f"    No GPCP files found for PAL period!")
            continue
        
        # Load relevant GPCP data once
        gpcp_matches = []
        for gpcp_file in relevant_gpcp_files:
            try:
                print(f"    Processing GPCP file: {os.path.basename(gpcp_file)}")
                gpcp_ds = xr.open_dataset(gpcp_file)
                gpcp_ds = ds_swaplon(gpcp_ds)  # Normalize longitude
                
                # Get GPCP coordinates
                gpcp_lat_vals = gpcp_ds['latitude'].values if 'latitude' in gpcp_ds else gpcp_ds['lat'].values
                gpcp_lon_vals = gpcp_ds['longitude'].values if 'longitude' in gpcp_ds else gpcp_ds['lon'].values
                gpcp_time = pd.to_datetime(gpcp_ds['time'].values)
                
                # Find precipitation variable
                gpcp_precip_vars = ['precip', 'precipitation', 'pcp', 'PRECIP']
                gpcp_precip_var = None
                for var in gpcp_precip_vars:
                    if var in gpcp_ds.variables:
                        gpcp_precip_var = var
                        break
                
                if gpcp_precip_var is None:
                    print(f"    No precipitation variable found in {os.path.basename(gpcp_file)}")
                    print(f"    Available variables: {list(gpcp_ds.variables.keys())}")
                    gpcp_ds.close()
                    continue
                
                # Match with daily PAL observations
                for _, daily_row in daily_pal.iterrows():
                    target_date = daily_row['date']
                    target_lat = daily_row['lat']
                    target_lon = daily_row['lon']
                    
                    # Check if this date is in current GPCP file
                    gpcp_dates = [t.date() for t in gpcp_time]
                    if target_date not in gpcp_dates:
                        continue
                    
                    # Find time index
                    time_idx = gpcp_dates.index(target_date)
                    
                    # Find nearest GPCP grid point
                    lat_idx = np.argmin(np.abs(gpcp_lat_vals - target_lat))
                    lon_idx = np.argmin(np.abs(gpcp_lon_vals - target_lon))
                    
                    # Extract GPCP precipitation value
                    gpcp_precip_val = gpcp_ds[gpcp_precip_var].values[time_idx, lat_idx, lon_idx]
                    
                    # Handle missing values
                    if np.isnan(gpcp_precip_val) or np.ma.is_masked(gpcp_precip_val):
                        continue
                    
                    gpcp_matches.append({
                        'date': target_date,
                        'pal_lat': target_lat,
                        'pal_lon': target_lon,
                        'gpcp_lat': gpcp_lat_vals[lat_idx],
                        'gpcp_lon': gpcp_lon_vals[lon_idx],
                        'pal_precip': daily_row['pal_precip'],
                        'gpcp_precip': float(gpcp_precip_val),
                        'pal_id': daily_row['pal_id'],
                        'region': daily_row['region'],
                        'gpcp_version': 'v1.3'
                    })
                
                gpcp_ds.close()
                
            except Exception as e:
                print(f"    Error processing GPCP file {os.path.basename(gpcp_file)}: {e}")
                continue
        
        # Add matches to overall dataset
        all_pal_gpcp_data.extend(gpcp_matches)
        pal_ds.close()
        
        print(f"    Found {len(gpcp_matches)} daily matches for PAL {pal_id}")
        
        # Break after successful processing for testing
        if len(gpcp_matches) > 0:
            print(f"    SUCCESS! Breaking after successful PAL processing for testing...")
            all_pal_gpcp_data.extend(gpcp_matches)
            pal_ds.close()
            break  # Break from PAL loop
    
    # Break from region loop after processing one region successfully
    if len(all_pal_gpcp_data) > 0:
        print(f"Breaking after processing {region_name} region for testing...")
        break

# Convert to DataFrame
if all_pal_gpcp_data:
    pal_gpcp_df = pd.DataFrame(all_pal_gpcp_data)
    print(f"\nTotal PAL-GPCP paired observations: {len(pal_gpcp_df)}")
    print(f"Unique PALs: {pal_gpcp_df['pal_id'].nunique()}")
    print("\nSample of paired data:")
    print(pal_gpcp_df.head())
else:
    print("No PAL-GPCP matches found!")

gc.collect()  # Clean up memory

# First, let's examine the structure of GPCP files
print("EXAMINING GPCP FILE STRUCTURE:")
print("="*50)
if len(all_gpcp_v1pt3_files) > 0:
    sample_gpcp_file = all_gpcp_v1pt3_files[0]
    print(f"Sample GPCP file: {os.path.basename(sample_gpcp_file)}")
    
    try:
        sample_gpcp = xr.open_dataset(sample_gpcp_file)
        print(f"GPCP variables: {list(sample_gpcp.variables.keys())}")
        print(f"GPCP dimensions: {list(sample_gpcp.dims.keys())}")
        
        # Check coordinate names
        if 'time' in sample_gpcp.variables:
            print(f"Time range: {sample_gpcp.time.values[0]} to {sample_gpcp.time.values[-1]}")
        
        # Check lat/lon coordinate names
        lat_coord = None
        lon_coord = None
        for coord in ['lat', 'latitude', 'LAT', 'LATITUDE']:
            if coord in sample_gpcp.variables:
                lat_coord = coord
                break
        for coord in ['lon', 'longitude', 'LON', 'LONGITUDE']:
            if coord in sample_gpcp.variables:
                lon_coord = coord
                break
                
        if lat_coord and lon_coord:
            print(f"Lat coordinate: {lat_coord}, range: {sample_gpcp[lat_coord].values.min():.2f} to {sample_gpcp[lat_coord].values.max():.2f}")
            print(f"Lon coordinate: {lon_coord}, range: {sample_gpcp[lon_coord].values.min():.2f} to {sample_gpcp[lon_coord].values.max():.2f}")
        
        sample_gpcp.close()
    except Exception as e:
        print(f"Error examining GPCP file: {e}")
        
print("="*50)


