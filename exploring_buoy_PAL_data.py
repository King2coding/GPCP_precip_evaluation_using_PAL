#%%
import importlib
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])

from util_functions import *
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.legend import Legend
import dask
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import pandas as pd
from rasterio.transform import from_origin
from rasterio.transform import rowcol
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

#%%
# define the path to the data
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

#%%
# define global variables
cde_run_dte = str(date.today().strftime('%Y%m%d'))

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# read buoys data
# Define directories for each region
# List all directories in the parent folder
all_buoy_dirs = [os.path.join(moored_bouys_paf, d) for d in os.listdir(moored_bouys_paf) if os.path.isdir(os.path.join(moored_bouys_paf, d))]

# Filter directories based on region names
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
indian_buoy_files = sorted([os.path.join(indian_buoy_dir, f) for f in os.listdir(indian_buoy_dir) if f.endswith('.cdf')])
atlantic_buoy_files = sorted([os.path.join(atlantic_buoy_dir, f) for f in os.listdir(atlantic_buoy_dir) if f.endswith('.cdf')])

gc.collect()
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

gc.collect()


#%%
# NOW WE WILL COLLECT ALL INSTANCES AND DATA FOR WHERE PAL AND BUOY OCCUR IN THE SAME GPCP V3.3 PIXEL
focus_regions = ['TNEP', 'TNWP']

# FIRST GROUP BUOY FILES BY REGION BASED ON THEIR LONGITUDE
# Define longitude bounds for TNEP and TNWP
region_bounds = {
    'TNEP': (-180, -60),  # Longitude range for Tropical Northeastern Pacific
    'TNWP': (60, 180)     # Longitude range for Tropical Northwestern Pacific
}

# Group buoy files by region
buoy_files_by_region = {'TNEP': [], 'TNWP': []}
for buoy_file in pacific_buoy_files:
    with xr.open_dataset(buoy_file) as ds:
        buoy_lon = ds['lon'].values[0]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]
    
    for region, bounds in region_bounds.items():
        if bounds[0] <= buoy_lon <= bounds[1]:
            buoy_files_by_region[region].append(buoy_file)
            break

# Initialize an empty list to store matches
matches = []

# Define the GPCP resolution
gpcp_resolution = 0.5

# Iterate over the focus regions
for region in focus_regions:
    print(f"Processing region: {region}")
    pal_files = pals_classed_by_region.get(region, [])
    buoy_files = buoy_files_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    print(f"Number of Buoy files in {region}: {len(buoy_files)}")
    
    # Iterate over PAL files
    for pal_file in pal_files:
        print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL data into a DataFrame
        with xr.open_dataset(pal_file) as ds:
            pal_time = ds['time'].values
            pal_lat = ds['lat'].values
            pal_lon = ds['lon'].values
            pal_rain_rate = ds['rain_rate'].values  # Assuming 'rain_rate' is the variable name
            
        # Normalize longitude to [-180, 180]
        pal_lon = (pal_lon + 180) % 360 - 180
        
        # Create a DataFrame
        pal_df = pd.DataFrame({
            "time": pal_time,
            "lat": pal_lat,
            "lon": pal_lon,
            "rain_rate": pal_rain_rate
        })

        # Filter out rows with NaN values or negative rain_rate
        pal_df = pal_df.dropna(subset=['lat', 'lon', 'rain_rate'])
        pal_df = pal_df[pal_df['rain_rate'] >= 0]

        # Compute date from time and group by date to calculate daily means
        pal_df['date'] = pd.to_datetime(pal_df['time']).dt.date
        # daily_pal_df = pal_df.groupby('date').mean().reset_index()
        
        pal_id = os.path.basename(pal_file)  # Extract PAL file ID from filename
        
        # Iterate over Buoy files in the same region
        for buoy_file in buoy_files:
            # print(f"Comparing with Buoy file: {os.path.basename(buoy_file)}")
            # Extract Buoy lat/lon
            with xr.open_dataset(buoy_file) as ds:
                buoy_lat = ds['lat'].values[0]
                buoy_lon = ds['lon'].values[0]
                # Normalize longitude to [-180, 180]
            buoy_lon = (buoy_lon + 180) % 360 - 180

            buoy_row, buoy_col = assign_to_gpcp_grid(buoy_lat, buoy_lon, gpcp_resolution)
            buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            
    
            # Check if any daily PAL lat/lon falls in the same GPCP pixel as the Buoy
            pal_df['row_col'] = pal_df.apply(
                lambda row: assign_to_gpcp_grid(row['lat'], row['lon'], gpcp_resolution), axis=1
            )
            
            # Check for matches
            match_indices = pal_df['row_col'].apply(
                lambda rc: rc == (buoy_row, buoy_col)
            )
            
            if match_indices.any():
                print(f"Match found between PAL file {pal_id} and Buoy file {buoy_id} in region {region} at row {buoy_row}, col {buoy_col}")
                # Store the match details
                match_df = pal_df[match_indices].copy()
                match_df['PAL_File'] = pal_file
                match_df['Buoy_File'] = buoy_file
                match_df['Buoy_Row'] = buoy_row
                match_df['Buoy_Col'] = buoy_col
                match_df['PAL_ID'] = pal_id
                match_df['Buoy_ID'] = buoy_id
                match_df['PAL_Region'] = region
                match_df['Buoy_Region'] = region
                
                # Append the match DataFrame to the matches list
                matches.append(match_df)
            # else:
            #     print(f"No match found for PAL file {pal_id} with Buoy file {buoy_id}")

# Concatenate all match DataFrames into a single DataFrame
if matches:
    pal_buoy_matches = pd.concat(matches, ignore_index=True)
    print("Matching process completed. Here are the first few matches:")
    print(pal_buoy_matches.head())
else:
    print("No matches found between PAL and Buoy files.")

gc.collect()


#%%
# NEW METHOD
dfs_by_region = {}
for region in focus_regions:
    print(f"Processing region: {region}")
    
    pal_files = pals_classed_by_region.get(region, [])
    buoy_files = buoy_files_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    # print(f"Number of Buoy files in {region}: {len(buoy_files)}")

    # Initialize a list to store DataFrames for this region
    region_dfs = []
    
    # Iterate over PAL files
    for pal_file in pal_files:
        # print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL data into a DataFrame
        with xr.open_dataset(pal_file) as ds:
            pal_time = ds['time'].values
            pal_lat = ds['lat'].values
            pal_lon = ds['lon'].values
            pal_rain_rate = ds['rain_rate'].values  # Assuming 'rain_rate' is the variable name
            
        # Normalize longitude to [-180, 180]
        pal_lon = (pal_lon + 180) % 360 - 180
        
        # Create a DataFrame
        pal_df = pd.DataFrame({
            "lat": pal_lat,
            "lon": pal_lon,
        })
        # Filter out rows with NaN values or negative rain_rate
        pal_df = pal_df.dropna(subset=['lat', 'lon'])

        pal_df['row_col'] = pal_df.apply(
                lambda row: assign_to_gpcp_grid(row['lat'], row['lon'], gpcp_resolution), axis=1
            )

        pal_df['PAL_File'] = pal_file
        pal_df['PAL_ID'] = os.path.basename(pal_file)
        pal_df['PAL_Region'] = region

        # Append the DataFrame for this PAL file to the list for the region
        region_dfs.append(pal_df)

    # Concatenate all DataFrames for this region into a single DataFrame
    if region_dfs:
       dfs_by_region[region] = pd.concat(region_dfs, ignore_index=True)

gc.collect()

#%%
# Now, iterate over Buoy files and check for matches with PAL DataFrames
region_match_buoy_pal_df = {}
for region, pal_df in dfs_by_region.items():
    print(f"Processing Buoy files for region: {region}")    
    
    buoy_files = buoy_files_by_region.get(region, [])
    
    # Iterate over Buoy files in the same region
    for buoy_file in buoy_files:
        match_buoy_pal_df = []
        # print(f"Comparing with Buoy file: {os.path.basename(buoy_file)}")
        # Extract Buoy lat/lon
        with xr.open_dataset(buoy_file) as ds:
            buoy_lat = ds['lat'].values[0]
            buoy_lon = ds['lon'].values[0]
            # Normalize longitude to [-180, 180]
        buoy_lon = (buoy_lon + 180) % 360 - 180

        buoy_row, buoy_col = assign_to_gpcp_grid(buoy_lat, buoy_lon, gpcp_resolution)
        buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            
        
        # Check if any PAL lat/lon falls in the same GPCP pixel as the Buoy
        # Ensure row_col is converted to tuples for comparison
        pal_df['row_col'] = pal_df['row_col'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        match_pal_df = pal_df.loc[pal_df['row_col'] == (buoy_row, buoy_col)]
                
        if match_pal_df.shape[0] > 0:
            print(f"Match found between PAL file {pal_df['PAL_ID'].iloc[0]} and Buoy file {buoy_id} in region {region} at row {buoy_row}, col {buoy_col}")
            # Store the match details
            
            match_pal_df['Buoy_File'] = buoy_file
            match_pal_df['Buoy_Row'] = buoy_row
            match_pal_df['Buoy_Col'] = buoy_col
            match_pal_df['Buoy_ID'] = buoy_id
            match_pal_df['Buoy_Region'] = region

            # Append the match DataFrame to the matches list
            match_buoy_pal_df.append(match_pal_df)
    region_match_buoy_pal_df[region] = pd.concat(match_buoy_pal_df, ignore_index=True) if match_buoy_pal_df else pd.DataFrame()
gc.collect()