#%%
# important links 
# https://data.pmel.noaa.gov/generic/erddap/info/pmelTaoDyRain/index.html
# buoy data download link: https://www.pmel.noaa.gov/tao/drupal/disdel/

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

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'
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
    'TNEP': (-180, -60),  # Longitude range for Tropical Northeastern Pacific in [-180, 180]
    'TNWP': (120, 180)      # Longitude range for Tropical Northwestern Pacific in [-180, 180]
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

#%%
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
        pal_lon = (pal_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180] for compatibility with GPCP grid
        
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
            buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

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

gpcp_resolution = 1.0
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

        pal_df[['PAL_row', 'PAL_col']] = pal_df.apply(
            lambda row: pd.Series(assign_to_gpcp_grid(row['lat'], row['lon'], gpcp_resolution)), axis=1
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

    match_buoy_pal_df = []
    
    # Iterate over Buoy files in the same region
    for buoy_file in buoy_files:
        
        # Extract Buoy lat/lon
        # with xr.open_dataset(buoy_file) as ds:
        ds  = xr.open_dataset(buoy_file) #as ds
        buoy_lat = ds['lat'].values[0]
        buoy_lon = ds['lon'].values[0]
            # Normalize longitude to [-180, 180]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180] for consistency

        buoy_row, buoy_col = assign_to_gpcp_grid(buoy_lat, buoy_lon, gpcp_resolution)
        buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            
        
        # Check if any PAL lat/lon falls in the same GPCP pixel as the Buoy
        match_pal_df = pal_df.loc[(pal_df['PAL_row'] == buoy_row) & (pal_df['PAL_col'] == buoy_col)]
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

#%%
# lets compare rainfall rates between PAL and Buoy data for the two focus regions
# make pal and buoy dfs by region
pal_dfs_by_region = {}
buoy_dfs_by_region = {}

for region in focus_regions:
    print(f"Processing region: {region}")
    
    pal_files = pals_classed_by_region.get(region, [])
    buoy_files = buoy_files_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    print(f"Number of Buoy files in {region}: {len(buoy_files)}")

    # Initialize lists to store DataFrames for this region
    pal_region_dfs = []
    buoy_region_dfs = []
    
    # Iterate over PAL files
    for pal_file in pal_files:
        # print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL data into a DataFrame
        # with xr.open_dataset(pal_file) as ds:
        pal_ds = xr.open_dataset(pal_file)
        pal_df = pd.DataFrame({
            'time': pd.to_datetime(pal_ds['time'].values),            
            'rain_rate': pal_ds['rain_rate'].values,
            'wind_speed': pal_ds['wind_speed'].values,
        })

        # Filter out rows with negative rain_rate
        pal_df = pal_df[pal_df['rain_rate'] >= 0]

        # drop rows where wind speed is >= 15 m/s
        pal_df = pal_df.drop(pal_df[pal_df['wind_speed'] >= 15].index)

        pal_df['date'] = pal_df['time'].dt.date  # Extract date from time

        pal_df = pal_df.dropna(axis=0, how='any')

        # groupby date and get mean of rain_rate and GPCP data
        daily_avg_rain = pal_df.groupby('date').mean(['rain_rate',]).reset_index()

        # multiply PAL rain rate by 24 to get daily average
        daily_avg_rain['rain_rate'] *= 24
        daily_avg_rain['ID'] = os.path.basename(os.path.basename(pal_file))  # Extract PAL file ID from filename

        # append the DataFrame for this PAL file to the list for the region
        pal_region_dfs.append(daily_avg_rain)

    # Concatenate all DataFrames for this region into a single DataFrame
    pal_dfs_by_region[region] = pd.concat(pal_region_dfs)

    # Iterate over Buoy files in the same region
    for buoy_file in buoy_files:
        buoy_ds = xr.open_dataset(buoy_file)
        buoy_df = pd.DataFrame({
            'time': pd.to_datetime(buoy_ds['time'].values),            
            'rain_rate': buoy_ds['RN_485'].values.flatten(),
            'quality_code': buoy_ds['QRN_5485'].values.flatten(),
        })

        # Filter out rows with negative rain_rate
        buoy_df = buoy_df[buoy_df['rain_rate'] >= 0]

        # buoys cover longer time period than pal so select date time period covering pal datetime period range
        pal_start_date = pal_dfs_by_region[region]['date'].min()
        pal_end_date = pal_dfs_by_region[region]['date'].max()
        buoy_df = buoy_df[(buoy_df['time'] >= pd.to_datetime(pal_start_date)) & (buoy_df['time'] <= pd.to_datetime(pal_end_date))]

        # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
        buoy_df = buoy_df[(buoy_df['quality_code'] >= 1) & (buoy_df['quality_code'] <= 3)]

        # daily rain rate
        buoy_df['rain_rate'] *= 24
        buoy_df['ID'] = os.path.basename(os.path.basename(buoy_file))  # Extract Buoy file ID from filenamebuoy_df['time'].dt.date  # Extract date from time

        # append the DataFrame for this Buoy file to the list for the region
        buoy_region_dfs.append(buoy_df)

    # Concatenate all DataFrames for this region into a single DataFrame
    buoy_dfs_by_region[region] = pd.concat(buoy_region_dfs)

gc.collect()


#%%
bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']

tnep_pal_pdfc_pdfv = compute_pdf_elements(pal_dfs_by_region['TNEP'], bin_values)
tnwp_pal_pdfc_pdfv = compute_pdf_elements(pal_dfs_by_region['TNWP'], bin_values)

tnep_buoy_pdfc_pdfv = compute_pdf_elements(buoy_dfs_by_region['TNEP'], bin_values)
tnwp_buoy_pdfc_pdfv = compute_pdf_elements(buoy_dfs_by_region['TNWP'], bin_values)


# make a 2by 2 line plot of the pdfc and pdfv for pal and buoy data by region
# in the 2by2 plot, the first row is TNEP and the second row is TNWP
# first column is pdfc and the second column is pdfv
# Update matplotlib parameters for consistent styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

fig, axs = plt.subplots(2, 2, figsize=(16, 10), 
                        sharex=False, sharey=False, dpi=1000)

# Set common x-axis ticks and labels
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']
bin_positions = range(len(bin_labels))
lw = 2
# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Line Plot TNEP PDFc
axs[0, 0].plot(bin_positions, tnep_pal_pdfc_pdfv['pdfc'], label='PAL', marker='o',lw=lw)
axs[0, 0].plot(bin_positions, tnep_buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='x',lw=lw)
axs[0, 0].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0, 0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[0, 0].set_xticks(bin_positions)
axs[0, 0].set_xticklabels(bin_labels)
axs[0, 0].legend(fontsize=18, frameon=False)

# Line Plot TNEP PDFv
axs[0, 1].plot(bin_positions, tnep_pal_pdfc_pdfv['pdfv'], label='PAL', marker='o', lw=lw)
axs[0, 1].plot(bin_positions, tnep_buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='x', lw=lw)
axs[0, 1].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0, 1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[0, 1].set_xticks(bin_positions)
axs[0, 1].set_xticklabels(bin_labels)
axs[0, 1].legend(fontsize=18, frameon=False)

# Line Plot TNWP PDFc
axs[1, 0].plot(bin_positions, tnwp_pal_pdfc_pdfv['pdfc'], label='PAL', marker='o')
axs[1, 0].plot(bin_positions, tnwp_buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='x')
axs[1, 0].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1, 0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[1, 0].set_xticks(bin_positions)
axs[1, 0].set_xticklabels(bin_labels)
axs[1, 0].legend(fontsize=18, frameon=False)

# Line Plot TNWP PDFv
axs[1, 1].plot(bin_positions, tnwp_pal_pdfc_pdfv['pdfv'], label='PAL', marker='o', lw=lw)
axs[1, 1].plot(bin_positions, tnwp_buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='x', lw=lw)
axs[1, 1].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1, 1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[1, 1].set_xticks(bin_positions)
axs[1, 1].set_xticklabels(bin_labels)
axs[1, 1].legend(fontsize=18, frameon=False)

# Adjust layout
plt.tight_layout()

# save the figure
svnme = os.path.join(path_to_put_plts, f'pdfc_pdfv_comparison_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight')

#%%
# calculate multiyear monthly mean rainfall rate for PAL and Buoy data

# Call the function
pal_monthly_means_by_region = calculate_multiyear_monthly_mean_rainfall_by_region(pal_dfs_by_region,'date')
buoy_monthly_means_by_region = calculate_multiyear_monthly_mean_rainfall_by_region(buoy_dfs_by_region,'time')


# Plotting the multiyear monthly mean rainfall rate for PAL and Buoy data
# Create a 2x1 plot for multiyear monthly mean rainfall rate comparison
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=False, sharey=True, dpi=1000)

# Update matplotlib parameters for consistent styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# Month labels for x-axis
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_positions = range(1, 13)
lw = 2

# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Plot TNEP data
axs[0].plot(month_positions, pal_monthly_means_by_region['TNEP']['rain_rate'], label='PAL', marker='o', lw=lw)
axs[0].plot(month_positions, buoy_monthly_means_by_region['TNEP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[0].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')
axs[0].set_xticks(month_positions)
axs[0].set_xticklabels(month_labels)
axs[0].legend(fontsize=18, frameon=False)

# Plot TNWP data
axs[1].plot(month_positions, pal_monthly_means_by_region['TNWP']['rain_rate'], label='PAL', marker='o', lw=lw)
axs[1].plot(month_positions, buoy_monthly_means_by_region['TNWP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[1].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')
axs[1].set_xticks(month_positions)
axs[1].set_xticklabels(month_labels)
axs[1].legend(fontsize=18, frameon=False)

# Adjust layout
plt.tight_layout()
gc.collect()

# Save the figure
svnme = os.path.join(path_to_put_plts, f'multiyear_monthly_mean_comparison_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight')


#%%
# scatterplot of PAL and Buoy data for TNEP and TNWP using the monthly means
# Calculate metrics for TNEP and TNWP
tnep_rb, tnep_rmse, tnep_cc = calculate_metrics(pal_monthly_means_by_region['TNEP']['rain_rate'], 
                                                buoy_monthly_means_by_region['TNEP']['rain_rate'])

tnwp_rb, tnwp_rmse, tnwp_cc = calculate_metrics(pal_monthly_means_by_region['TNWP']['rain_rate'], 
                                                buoy_monthly_means_by_region['TNWP']['rain_rate'])
# plot scatter plot of PAL and Buoy data for TNEP and TNWP using the monthly means
fig, axs = plt.subplots(1, 2, figsize=(18, 10), sharex=False, sharey=False, dpi=1000)
# Update matplotlib parameters for consistent styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
# Removed 'Times New Roman' to avoid findfont warnings
mpl.rcParams['ytick.labelsize'] = 18    

# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Scatter Plot TNEP
axs[0].scatter(pal_monthly_means_by_region['TNEP']['rain_rate'], buoy_monthly_means_by_region['TNEP']['rain_rate'], 
               label='TNEP', color='blue', alpha=0.7, edgecolors='w', s=100)
axs[0].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0].set_xlabel('PAL Rainfall [mm]', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Buoy Rainfall [mm]', fontsize=18, fontweight='bold')
axs[0].set_xlim(0, 250)
axs[0].set_ylim(0, 250)
# Add 1:1 line
axs[0].plot([0, 250], [0, 250], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
# axs[0].legend(fontsize=18, frameon=False)

axs[0].text(
    0.05, 0.95,
    f'RB: {tnep_rb:.2f}%\nRMSE: {tnep_rmse:.2f} mm\nCC: {tnep_cc:.2f}',
    transform=axs[0].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)

# Scatter Plot TNWP
axs[1].scatter(pal_monthly_means_by_region['TNWP']['rain_rate'], buoy_monthly_means_by_region['TNWP']['rain_rate'], 
               label='TNWP', color='red', alpha=0.7, edgecolors='w', s=100)
axs[1].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1].set_xlabel('PAL Rainfall [mm]', fontsize=18, fontweight='bold')
axs[1].set_ylabel('Buoy Rainfall [mm]', fontsize=18, fontweight='bold')
axs[1].set_xlim(0, 250)
axs[1].set_ylim(0, 250)
# Add 1:1 line
axs[1].plot([0, 250], [0, 250], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
# axs[1].legend(fontsize=18, frameon=False)

axs[1].text(
    0.05, 0.95,
    f'RB: {tnwp_rb:.2f}%\nRMSE: {tnwp_rmse:.2f} mm\nCC: {tnwp_cc:.2f}',
    transform=axs[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)

# Adjust layout
plt.tight_layout()
gc.collect()

# Save the figure
svnme = os.path.join(path_to_put_plts, f'multiyear_monthly_mean_scatter_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight')