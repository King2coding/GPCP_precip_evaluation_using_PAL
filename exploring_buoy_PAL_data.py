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
focus_regions = [ 'TNEP', 'TNWP']

# FIRST FIND ALL PAL AND BUOY STATIONS OCCURING IN THE SAME GPCP V3.3 PIXEL
# WE USE THE "assign_to_gpcp_grid" THAT RETURNS THE ROW COL INDEX OF THE GPCP PIXEL

# Initialize an empty list to store matches
matches = []

# Define the GPCP resolution
gpcp_resolution = 0.5

# Iterate over the focus regions
for region in focus_regions:
    print(f"Processing region: {region}")
    pal_files = pals_classed_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    
    # Iterate over PAL files
    for pal_file in pal_files:
        print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL lat/lon trajectory (assuming util_functions has a function to extract lat/lon from PAL files)
        with xr.open_dataset(pal_file) as ds:
            pal_lat_list = ds['lat'].values
            pal_lon_list = ds['lon'].values
            # make lon between 180 and -180
        pal_lon_list = (pal_lon_list + 180) % 360 - 180
        # Combine PAL lat/lon into an array and remove rows with NaN values
        pal_lat_lon_array = np.array(list(zip(pal_lat_list, pal_lon_list)))
        pal_lat_lon_array = pal_lat_lon_array[~np.isnan(pal_lat_lon_array).any(axis=1)]
        
        # Convert the cleaned array back to a list of tuples
        pal_lat_lon_list = list(map(tuple, pal_lat_lon_array))

        pal_id = os.path.basename(pal_file)  # Extract PAL file ID from filename
        
        # Iterate over Pacific buoy files
        for buoy_file in pacific_buoy_files:
            print(f"Comparing with Buoy file: {os.path.basename(buoy_file)}")
            # Extract Buoy lat/lon (assuming util_functions has a function to extract lat/lon from Buoy files)
            with xr.open_dataset(buoy_file) as ds:
                buoy_lat = ds['lat'].values[0]
                buoy_lon = ds['lon'].values[0]
                # make lon between 180 and -180
            buoy_lon = (buoy_lon + 180) % 360 - 180

            buoy_row, buoy_col = assign_to_gpcp_grid(buoy_lat, buoy_lon, gpcp_resolution)
            buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            

            # Check if any PAL lat/lon falls in the same GPCP pixel as the Buoy
            # Vectorize the assignment of PAL lat/lon to GPCP grid
            pal_rows_cols = np.array([assign_to_gpcp_grid(lat, lon, gpcp_resolution) for lat, lon in pal_lat_lon_list])
            
            # Check if any PAL grid cell matches the Buoy grid cell
            match_indices = np.where((pal_rows_cols[:, 0] == buoy_row) & (pal_rows_cols[:, 1] == buoy_col))[0]
            
            if match_indices.size > 0:
                print(f"Match found between PAL file {pal_id} and Buoy file {buoy_id}")
                # Store the match details
                # Create a DataFrame for the current match
                match_df = pd.DataFrame({
                    "PAL_Row": pal_rows_cols[match_indices, 0],
                    "PAL_Col": pal_rows_cols[match_indices, 1],
                    "PAL_File": pal_file,
                    "Buoy_File": buoy_file,
                    "Buoy_Row": buoy_row,
                    "Buoy_Col": buoy_col,
                    "PAL_ID": pal_id,
                    "Buoy_ID": buoy_id
                })

                # Append the match DataFrame to the matches list
                matches.append(match_df)
            else:
                print(f"No match found for PAL file {pal_id} with Buoy file {buoy_id}")

# Concatenate all match DataFrames into a single DataFrame
if matches:
    pal_buoy_matches = pd.concat(matches, ignore_index=True)
    print("Matching process completed. Here are the first few matches:")
    print(pal_buoy_matches.head())
else:
    print("No matches found between PAL and Buoy files.")

gc.collect()
