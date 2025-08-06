#%%
# important links 
# https://data.pmel.noaa.gov/generic/erddap/info/pmelTaoDyRain/index.html
# buoy data download link: https://www.pmel.noaa.gov/tao/drupal/disdel/

#%% IMPORT LIBRARIES

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

import seaborn as sns

# Import memory management utilities
from memory_management_improvements import (
    setup_memory_management, memory_safe_batch_processing, 
    safe_dataset_operation, monitor_memory_usage, fast_setup
)
from kernel_recovery import (
    KernelStateManager, save_data_loading_checkpoint, 
    save_analysis_checkpoint, check_what_needs_reloading
)

# Set up minimal memory management - server friendly
print("Setting up minimal memory management...")
try:
    # Use minimal setup with limited resources for shared server
    dask.config.set({'scheduler': 'threads', 'num_workers': 1})
    print("✓ Minimal memory management setup complete")
except Exception as e:
    print(f"Warning: Could not set up memory management: {e}")
    print("Continuing with default configuration...")

dask_client = None

#%% DEBUG: Check current region bounds and test overlap function
print("CURRENT REGION BOUNDS:")
print("="*50)
for region, bounds in buoy_region_bounds.items():
    print(f"{region}: {bounds}")
print("="*50)

#%% DEFINE PATH TO DATA
# path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v1_pnt_3_2010_2020'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2010_2020'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_2010_2020'

path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_daily_1998-2025'

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'

path_to_put_dfs = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs'

#%% DEFINE GLOBAL VARIABLES

cde_run_dte = str(date.today().strftime('%Y%m%d'))

# all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])

all_imerg_files = sorted([os.path.join(path_to_imerg, f) for f in os.listdir(path_to_imerg) if f.endswith('.nc4')])

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



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
# read all GPCP into a single xr data
# Limit the number of simultaneously open files to avoid kernel crash

# Simple data loading without CPU-intensive optimizations (server-friendly)
print("Starting data loading...")

# Simple dask configuration - minimal CPU usage
dask.config.set({
    'array.chunk-size': '128MB',  # Reasonable chunk size
    'scheduler': 'threads',       # Use threads instead of processes
    'num_workers': 2              # Limit workers to be server-friendly
})

# Use moderate batch size
batch_size = 30

# Process GPCP v1.3 files in smaller batches with better error handling
print(f"Processing GPCP v1.3 files in batches of {batch_size}...")
gpcp_v1pt3_batches = [all_gpcp_v1pt3_files[i:i + batch_size] for i in range(0, len(all_gpcp_v1pt3_files), batch_size)]
gpcp_ds_v1pt3_xr_list = []

for i, batch in enumerate(gpcp_v1pt3_batches):
    if i % 30 == 0:
        # Print progress every 30 batches
        print(f"Processing GPCP v1.3 batch {i+1}/{len(gpcp_v1pt3_batches)}")
    
    processed_batch = simple_process_gpcp_batch(batch, "v1.3")
    if processed_batch is not None:
        gpcp_ds_v1pt3_xr_list.append(processed_batch)
    
    # Simple garbage collection
    gc.collect()

# Combine all processed batches into a single xarray dataset - simple version
if gpcp_ds_v1pt3_xr_list:
    gpcp_ds_v1pt3_xr = xr.concat(gpcp_ds_v1pt3_xr_list, dim="time")
    print("GPCP v1.3 loading complete")
else:
    print("Warning: No GPCP v1.3 data was successfully loaded")
    gpcp_ds_v1pt3_xr = None
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# Process GPCP v3.2 files in smaller batches with better error handling
print(f"Processing GPCP v3.2 files in batches of {batch_size}...")
gpcp_v3pt2_batches = [all_gpcp_v3pt2_files[i:i + batch_size] for i in range(0, len(all_gpcp_v3pt2_files), batch_size)]
gpcp_ds_v3pt2_xr_list = []

for i, batch in enumerate(gpcp_v3pt2_batches):
    if i % 30 == 0:
        print(f"Processing GPCP v3.2 batch {i+1}/{len(gpcp_v3pt2_batches)}")
    
    processed_batch = simple_process_gpcp_batch(batch, "v3.2")
    if processed_batch is not None:
        gpcp_ds_v3pt2_xr_list.append(processed_batch)
    
    # Simple garbage collection
    gc.collect()

# Combine all processed batches into a single xarray dataset - simple version
if gpcp_ds_v3pt2_xr_list:
    gpcp_ds_v3pt2_xr = xr.concat(gpcp_ds_v3pt2_xr_list, dim="time")
    print("GPCP v3.2 loading complete")
else:
    print("Warning: No GPCP v3.2 data was successfully loaded")
    gpcp_ds_v3pt2_xr = None
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# Process GPCP v3.3 files in smaller batches with better error handling
print(f"Processing GPCP v3.3 files in batches of {batch_size}...")
gpcp_v3pt3_batches = [all_gpcp_v3pt3_files[i:i + batch_size] for i in range(0, len(all_gpcp_v3pt3_files), batch_size)]
gpcp_ds_v3pt3_xr_list = []

for i, batch in enumerate(gpcp_v3pt3_batches):
    if i % 30 == 0:
        print(f"Processing GPCP v3.3 batch {i+1}/{len(gpcp_v3pt3_batches)}")

    processed_batch = simple_process_gpcp_batch(batch, "v3.3")
    if processed_batch is not None:
        gpcp_ds_v3pt3_xr_list.append(processed_batch)
    
    # Simple garbage collection
    gc.collect()

# Combine all processed batches into a single xarray dataset - simple version
if gpcp_ds_v3pt3_xr_list:
    gpcp_ds_v3pt3_xr = xr.concat(gpcp_ds_v3pt3_xr_list, dim="time")
    print("GPCP v3.3 loading complete")
else:
    print("Warning: No GPCP v3.3 data was successfully loaded")
    gpcp_ds_v3pt3_xr = None
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 


# Process IMERG files - memory-efficient version
print(f"Processing IMERG files in batches of {batch_size}...")
imerg_batches = [all_imerg_files[i:i + batch_size] for i in range(0, len(all_imerg_files), batch_size)]
imerg_ds_xr_list = []

# Use smaller batch size for IMERG to reduce memory pressure
imerg_batch_size = min(batch_size, 100)  # Limit IMERG batch size
print(f"Using IMERG batch size: {imerg_batch_size}")

# Re-create batches with smaller size
imerg_batches = [all_imerg_files[i:i + imerg_batch_size] for i in range(0, len(all_imerg_files), imerg_batch_size)]

for i, batch in enumerate(imerg_batches):
    if i % 10 == 0:  # More frequent progress updates
        print(f"Processing IMERG batch {i+1}/{len(imerg_batches)} ({len(batch)} files)")
    
    try:
        # Use memory-efficient processing
        processed_batch = simple_process_imerg_batch_memory_efficient(
            batch, 
            product="imerg_fn", 
            processing_mode="auto"
        )
        if processed_batch is not None:
            imerg_ds_xr_list.append(processed_batch)
            print(f"  Batch {i+1} completed successfully")
        else:
            print(f"  Warning: Batch {i+1} returned None")
    except Exception as e:
        print(f"Error processing IMERG batch {i+1}: {e}")
        # Continue with next batch instead of stopping
        continue
    
    # Aggressive garbage collection
    import gc
    gc.collect()
    
    # Optional: Print memory usage if psutil is available
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            print(f"  Warning: Memory usage at {memory_percent:.1f}%")
    except ImportError:
        pass

# Combine all processed batches into a single xarray dataset
if imerg_ds_xr_list:
    print(f"Combining {len(imerg_ds_xr_list)} IMERG batches...")
    imerg_ds_xr = xr.concat(imerg_ds_xr_list, dim="time")
    print("IMERG loading complete")
    
    # Clean up batch list to free memory
    del imerg_ds_xr_list
    gc.collect()
else:
    print("Warning: No IMERG data was successfully loaded")
    imerg_ds_xr = None

gc.collect()  # Clean up memory
print("Data loading phase complete!")

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
ax.set_extent([-181, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Initialize buoy counts
buoy_counts = {}
for region, bfiles in buoy_files_by_region.items():
    buoy_counts[region] = len(bfiles)

    # set region markers
    if (region == "ENP") or (region == "WNP"):
        marker = '*'
    elif region == "ATL":
        marker = 'd'
    elif region == "IND":
        marker = 's'
    
    for fl in bfiles: 
        xrfile = xr.open_dataset(fl)       
        # Plot moored buoy data
        lat = xrfile['lat'].values[0]
        lon = xrfile['lon'].values[0]
        # make lon between 180 and -180
        lon = (lon + 180) % 360 - 180

        if lon == -180:
            # shift lon slightly for plotting
            lon = -178

        ax.scatter(lon, lat, color=buoy_region_colors[region], 
                   s=200, marker=marker, label=f'{region} Buoys', 
                   transform=ccrs.PlateCarree())

# Add grid lines for major ticks
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.7, linestyle='--')

# Create legend with full region names and buoy counts
legend_regions = [r for r in buoy_files_by_region.keys()]

# Full region names mapping
full_region_names = {    
    "ENP": "Eastern Pacific",
    "WNP": "Western Pacific",
    "IND": "Indian Ocean",
    "ATL": "Atlantic Ocean",    
}

# Add full names and buoy counts to legend labels
handles = []
labels = []
for region in legend_regions:
    if region in ["ENP", "WNP"]:
        marker = '*'
    elif region == "ATL":
        marker = 'd'
    elif region == "IND":
        marker = 's'
    handles.append(plt.Line2D([0], [0], color=buoy_region_colors[region], marker=marker, markersize=10, linestyle='None'))
    labels.append(f"{full_region_names[region]} ({buoy_counts[region]} buoys)")

leg = plt.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.35), 
    fontsize=18, ncol=2, frameon=False
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

# Try to enforce Times New Roman, but fallback gracefully if not available
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
ax.set_title("Buoy Locations by Ocean Region", fontsize=20, 
             fontweight='bold', fontname='Times New Roman')

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom for legend

svname = os.path.join(path_to_put_plts, f'Buoy_by_region_{cde_run_dte}.png')
plt.savefig(svname, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()  # Clean up memory

#%% SPATIOTEMPORAL MATCHING OF Buoy, IMERG AND GPCP DATA
regional_buoy_sate_dfs_daily_mean = {}
regional_buoy_sate_dfs_daily_lst = []
for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
        # store Buoy and GPCP dataframes
    region_buoy_sate_dfs = []  

    # LOAD Buoy DATA
    for b_file in buoy_files:
        b_ds = xr.open_dataset(b_file)
        b_lat = b_ds['lat'].values[0]
        b_lon = b_ds['lon'].values[0]
        b_lon = (b_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

        b_df = b_ds[['time', 'RN_485', 'QRN_5485']].to_dataframe().reset_index()
        b_df.rename(columns={'RN_485': 'rain_rate', 'QRN_5485': 'quality_flag'}, inplace=True)
        b_df['date'] = pd.to_datetime(b_df['time']).dt.date  # Extract date from time
        # Filter out rows with negative rain_rate
        b_df = b_df[b_df['rain_rate'] >= 0]

        print(f"Processing Buoy file: {os.path.basename(b_file)}")
       
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        # Process GPCP data with Buoy
        # GPCP v1.3 - Memory efficient version (Note: this appears to use v3.2 dataset)
        # print(f"Extracting GPCP v1.3 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
        # b_rain_gpcpv1pt3_df = extract_buoy_gpcp_data_memory_efficient(
        #     gpcp_ds_v3pt2_xr, b_lat, b_lon, version="v1pt3", chunk_size=200
        # )
        
        # if b_rain_gpcpv1pt3_df is None:
        #     print("Warning: Failed to extract GPCP v1.3 data, creating empty dataframe")
        #     b_rain_gpcpv1pt3_df = pd.DataFrame(columns=['date', 'GPCP_v1pt3'])
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -

        # Process GPCP v3.2 - Memory efficient version
        # print(f"Extracting GPCP v3.2 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
        # b_rain_gpcpv3pt2_df = extract_buoy_gpcp_data_memory_efficient(
        #     gpcp_ds_v3pt2_xr, b_lat, b_lon, version="v3pt2", chunk_size=200
        # )
        
        # if b_rain_gpcpv3pt2_df is None:
        #     print("Warning: Failed to extract GPCP v3.2 data, creating empty dataframe")
        #     b_rain_gpcpv3pt2_df = pd.DataFrame(columns=['date', 'GPCP_v3pt2'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        # Process GPCP v3.3 - Memory efficient version
        print(f"Extracting GPCP v3.3 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
        b_rain_gpcpv3pt3_df = extract_buoy_gpcp_data_memory_efficient(
            gpcp_ds_v3pt3_xr, b_lat, b_lon, version="v3pt3", chunk_size=200
        )
        
        if b_rain_gpcpv3pt3_df is None:
            print("Warning: Failed to extract GPCP v3.3 data, creating empty dataframe")
            b_rain_gpcpv3pt3_df = pd.DataFrame(columns=['date', 'GPCP_v3pt3'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process IMERG data with Buoy - Memory efficient version (optimized for 0.1° resolution)
        print(f"Extracting IMERG data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
        # Use smaller chunk size for IMERG due to high spatial resolution (0.1° vs 0.5°/1° for GPCP)
        b_rain_imerg_df = extract_buoy_imerg_data_memory_efficient(
            imerg_ds_xr, b_lat, b_lon, chunk_size=50  # Reduced from 100 to 50 for IMERG
        )
        
        if b_rain_imerg_df is None:
            print("Warning: Failed to extract IMERG data, creating empty dataframe")
            b_rain_imerg_df = pd.DataFrame(columns=['date', 'IMERG'])
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -    

        # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data        
        # COMBINE BY RAINFALL RATE
        b_df_combined_rain = b_df[['date', 'rain_rate', 'quality_flag']].copy()
        
        # merge GPCP v3.1 data            
        # b_df_combined_rain = b_df_combined_rain.merge(
        #     b_rain_gpcpv1pt3_df[['date','GPCP_v3pt1']], 
        #     on='date', how='left', suffixes=('', '_v3pt1')
        # )
         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.2 data
        # b_df_combined_rain = b_df_combined_rain.merge(
        #     b_rain_gpcpv3pt2_df[['date','GPCP_v3pt2', 'PLP_GPCP_v3pt2']], 
        #     on='date', how='left', suffixes=('', '_v3pt2')
        # )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.3 data
        b_df_combined_rain = b_df_combined_rain.merge(
            b_rain_gpcpv3pt3_df[['date','GPCP_v3pt3', 'PLP_GPCP_v3pt3']], 
            on='date', how='left', suffixes=('', '_v3pt3')
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge IMERG data - merge on 'date' column instead of index
        b_df_combined_rain = b_df_combined_rain.merge(
            b_rain_imerg_df[['date', 'IMERG']], 
            on='date', how='left', suffixes=('', '_IMERG')
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # retain only columns where PLP_GPCP_v3pt2 is == 100
        # b_df_combined_rain = b_df_combined_rain[b_df_combined_rain['PLP_GPCP_v3pt2'] == 100]
        # retain only columns where PLP_GPCP_v3pt3 is == 100
        b_df_combined_rain = b_df_combined_rain[b_df_combined_rain['PLP_GPCP_v3pt3'] == 100]


        # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
        b_df_combined_rain = b_df_combined_rain[(b_df_combined_rain['quality_flag'] >= 1) & \
                                                    (b_df_combined_rain['quality_flag'] <= 3)]


    
        # multiply PAL rain rate by 24 to get daily average
        b_df_combined_rain['rain_rate'] *= 24
        # add region name and track_PAL_id to the dataframe
        b_df_combined_rain['region'] = region_name  # Add region name for clarity
        b_df_combined_rain['ID'] = os.path.basename(b_file).split('.')[0]

        region_buoy_sate_dfs.append(b_df_combined_rain)

        b_ds.close()

    # Combine all region PAL-GPCP dataframes into a single dataframe
    region_buoy_sate_df = pd.concat(region_buoy_sate_dfs)

    # calculate daily mean per track_PAL_id
    region_buoy_sate_df_daily_mean = region_buoy_sate_df.groupby(['ID'])[['rain_rate', 
                                                                        # 'GPCP_v1pt3', 
                                                                        # 'GPCP_v3pt2', 
                                                                        'GPCP_v3pt3', 
                                                                        'IMERG']].mean().reset_index()  # , 'IMERG'IMERG']].mean().reset_index()
    region_buoy_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
    regional_buoy_sate_dfs_daily_mean[region_name] = region_buoy_sate_df_daily_mean

    # Append to the list for later processing
    regional_buoy_sate_dfs_daily_lst.append(region_buoy_sate_df)


gc.collect()  # Clean up memory

#%%
# calculate RB, RMSE and CC using all PAL-GPCP pairs in axes plot and show this in the plot
# as RB = , RMSE = , CC =
#- - - - -   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# b_gpcv1_3_cmp = pd.concat([df[['rain_rate', 'GPCP_v1pt3']] for df in regional_buoy_sate_dfs_daily_mean.values()], ignore_index=True)
# b_gpcv3_2_cmp = pd.concat([df[['rain_rate', 'GPCP_v3pt2']] for df in regional_buoy_sate_dfs_daily_mean.values()], ignore_index=True)
b_gpcv3_3_cmp = pd.concat([df[['rain_rate', 'GPCP_v3pt3']] for df in regional_buoy_sate_dfs_daily_mean.values()], ignore_index=True)
b_imerg_cmp = pd.concat([df[['rain_rate', 'IMERG']] for df in regional_buoy_sate_dfs_daily_mean.values()], ignore_index=True)
# calcute metrics for all regions combined
# rb_v1pt3, rmse_v1pt3, cc_v1pt3 = calculate_metrics(b_gpcv1_3_cmp['rain_rate'], b_gpcv1_3_cmp['GPCP_v1pt3'])
# rb_v3pt2, rmse_v3pt2, cc_v3pt2 = calculate_metrics(b_gpcv3_2_cmp['rain_rate'], b_gpcv3_2_cmp['GPCP_v3pt2'])
rb_v3pt3, rmse_v3pt3, cc_v3pt3 = calculate_metrics(b_gpcv3_3_cmp['rain_rate'], b_gpcv3_3_cmp['GPCP_v3pt3'])
rb_img, rmse_img, cc_img = calculate_metrics(b_imerg_cmp['rain_rate'], b_imerg_cmp['IMERG'])
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

fg, ax = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'wspace': 0.35})  # Increased wspace for wider interval

# Get PAL counts per region for legend
region_pal_counts = {region: len(buoy_files_by_region[region]) for region in regional_buoy_sate_dfs_daily_mean.keys()}

for region, df in regional_buoy_sate_dfs_daily_mean.items():
    reg_col = buoy_region_colors[region]
    # b_gpcv1_3 = df[['rain_rate', 'GPCP_v1pt3']]
    # b_gpcpv3_2 = df[['rain_rate', 'GPCP_v3pt2']]
    b_gpcv3_3 = df[['rain_rate', 'GPCP_v3pt3']]
    b_img_df = df[['rain_rate', 'IMERG']]

    # # Plot GPCP v1.3
    # ax[0].scatter(b_gpcv1_3['rain_rate'], b_gpcv1_3['GPCP_v1pt3'],
    #               color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)
    # # Plot GPCP v3.2
    # ax[1].scatter(b_gpcpv3_2['rain_rate'], b_gpcpv3_2['GPCP_v3pt2'],
    #               color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)
    # Plot GPCP v3.3
    ax[0].scatter(b_gpcv3_3['rain_rate'], b_gpcv3_3['GPCP_v3pt3'],
                  color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)
    # Plot IMERG
    ax[1].scatter(b_img_df['rain_rate'], b_img_df['IMERG'],
                  color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)

# Set axes limits, ticks, grids, and major ticks for all subplots
for i, a in enumerate(ax):
    a.set_xlim(0, 18)
    a.set_ylim(0, 18)
    a.set_xticks([0, 3, 6, 9, 12, 15, 18])
    a.set_yticks([0, 3, 6, 9, 12, 15, 18])
    a.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    a.minorticks_on()
    a.tick_params(axis='both', which='major', length=7, width=1.2, labelsize=18)
    a.tick_params(axis='both', which='minor', length=4, width=0.8)
    # add 1:1 line
    x = np.linspace(0, 18, 100)
    a.plot(x, x, color='gray', linestyle='--')
    # Set axis labels and title with bold fontweight
    if i == 0:
        a.set_xlabel('Buoy Observations [mm/day]', fontsize=18, fontweight='bold')
        a.set_ylabel('GPCP v3.3 Estimates [mm/day]', fontsize=18, fontweight='bold')
        a.set_title('GPCP v3.3 vs Buoy', fontsize=20, fontweight='bold')
    elif i == 1:
        a.set_xlabel('Buoy Observations [mm/day]', fontsize=18, fontweight='bold')
        a.set_ylabel('IMERG Estimates [mm/day]', fontsize=18, fontweight='bold')
        a.set_title('IMERG vs Buoy', fontsize=20, fontweight='bold')
    # Make tick labels bold
    for label in a.get_xticklabels() + a.get_yticklabels():
        label.set_fontweight('bold')

# Add metrics text to each subplot, with RMSE unit, bold font
# ax[0].text(
#     0.05, 0.95,
#     f'RB: {rb_v1pt3:.2f}%\nRMSE: {rmse_v1pt3:.2f} mm/day\nCC: {cc_v1pt3:.2f}',
#     transform=ax[0].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
#     bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
# )
# ax[1].text(
#     0.05, 0.95,
#     f'RB: {rb_v3pt2:.2f}%\nRMSE: {rmse_v3pt2:.2f} mm/day\nCC: {cc_v3pt2:.2f}',
#     transform=ax[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
#     bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
# )
ax[0].text(
    0.05, 0.95,
    f'RB: {rb_v3pt3:.2f}%\nRMSE: {rmse_v3pt3:.2f} mm/day\nCC: {cc_v3pt3:.2f}',
    transform=ax[0].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)

ax[1].text(
    0.05, 0.95,
    f'RB: {rb_img:.2f}%\nRMSE: {rmse_img:.2f} mm/day\nCC: {cc_img:.2f}',
    transform=ax[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)

# Add legend
handles, labels_ = ax[0].get_legend_handles_labels()
unique_labels = dict(zip(labels_, handles))  # Remove duplicates
# Place a common legend below and outside the plot
leg = fg.legend(
    unique_labels.values(), unique_labels.keys(),
    loc='lower center', bbox_to_anchor=(0.5, -0.15),
    fontsize=18,  ncol=6, frameon=False,
   
) #  title="Region (N PALs)"
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

# plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend
# save the figure
svnme = os.path.join(path_to_put_plts, f'Buoy_IMERG_GPCP_satellite_scatter_plots_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight', dpi=500)
gc.collect()  # Clean up memory

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# calculate monhtly mean per region
# Calculate monthly statistics (mean, Q1, Q3) for each region and product
monthly_stats = {}

# Prepare a DataFrame with all daily data
all_daily = pd.concat(regional_buoy_sate_dfs_daily_lst)
all_daily = all_daily.reset_index()  # Ensure 'date' is a column after concat
if 'date' not in all_daily.columns:
    all_daily['date'] = pd.to_datetime(all_daily['index'])
else:
    all_daily['date'] = pd.to_datetime(all_daily['date'])
all_daily['month'] = all_daily['date'].dt.month
all_daily['year'] = all_daily['date'].dt.year


# save all daily data to a csv file
all_daily.to_csv(os.path.join(path_to_put_dfs, f'all_daily_Buoy_sate_data_{cde_run_dte}.csv'), index=False)

regions = [r for r in regional_buoy_sate_dfs_daily_mean.keys()]

for region in regions:
    df = all_daily[all_daily['region'] == region].copy()
    # Compute monthly accumulation per buoy (ID)
    buoy_monthly = df.groupby(['ID', 'year', 'month'])[['rain_rate', 
                                                        # 'GPCP_v1pt3', 
                                                    #  'GPCP_v3pt2', 
                                                        'GPCP_v3pt3',
                                                         'IMERG'
                                                                 ]].sum().reset_index()
    # buoy_yr_monthly = df.groupby(['ID','year', 'month'])[['rain_rate', 
    #                                                             # 'GPCP_v1pt3', 
    #                                                             # 'GPCP_v3pt2', 
    #                                                             'GPCP_v3pt3', 
    #                                                             'IMERG']].sum().reset_index()
    # buoy_yr_monthly = buoy_yr_monthly.groupby(['year', 'month'])[['rain_rate', 
    #                                                             # 'GPCP_v1pt3', 
    #                                                             # 'GPCP_v3pt2', 
    #                                                             'GPCP_v3pt3', 
    #                                                             'IMERG']].mean().reset_index  
    # Now, for each month, compute mean, Q1, Q3 across buoys (i.e., for each month, use all buoys' accumulations)
    stats_ = buoy_monthly.groupby('month').agg({
        'rain_rate': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        # 'GPCP_v1pt3': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        # 'GPCP_v3pt2': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        'GPCP_v3pt3': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
        'IMERG': ['mean', ('q1', lambda x: np.percentile(x, 25)), ('q3', lambda x: np.percentile(x, 75))],
    })
    stats_.columns = ['_'.join(col).rstrip('_') for col in stats_.columns.values]
    monthly_stats[region] = stats_.reset_index()
    # Also store the PAL monthly accumulations for scatter/vertical lines
    monthly_stats[region + '_buoy_monthly'] = buoy_monthly

# Plotting (mimic the attached figure)

region_titles = {
    "ENP": "(a) Eastern Pacific (ENP)",
    "WNP": "(b) Western Pacific (WNP)",
    "IND": "(c) Indian Ocean (IND)",
    "ATL": "(d) Atlantic Ocean (ATL)"
}
region_order = ["ENP", "WNP", "IND", "ATL"]

# Use a serif font as a fallback if Times New Roman is not available
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

# fig, axs = plt.subplots(2, 3, figsize=(25, 10))
# axs = axs.flatten()
# months = np.arange(1, 13)
# month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# for i, region in enumerate(region_order):
#     ax = axs[i]
#     stats_ = monthly_stats[region]
#     buoy_monthly = monthly_stats[region + '_buoy_monthly']
#     # PAL: for each month, plot mean as a point, and Q1-Q3 as vertical line
#     pal_mean = stats_['rain_rate_mean'].values
#     pal_q1 = stats_['rain_rate_q1'].values
#     pal_q3 = stats_['rain_rate_q3'].values
#     # Plot PAL mean as scatter, Q1-Q3 as vertical line
#     for m_idx, m in enumerate(months):
#         mean = pal_mean[m_idx]
#         q1 = pal_q1[m_idx]
#         q3 = pal_q3[m_idx]
#         # Only plot if not nan
#         if not np.isnan(mean):
#             ax.scatter(m, mean, color='k', s=40, zorder=4, label='PAL Obs. Mean' if m_idx == 0 else None)
#             ax.vlines(m, q1, q3, color='gray', lw=2, zorder=3, label='PAL Obs. [Q1,Q3]' if m_idx == 0 else None)
#     # GPCP v3.2
#     # ax.plot(months, stats_['GPCP_v3pt2_mean'], color='b', label='GPCP v3.2 Est. Mean')
#     # ax.fill_between(months, stats_['GPCP_v3pt2_q1'], stats_['GPCP_v3pt2_q3'], color='b', alpha=0.2, label='GPCP v3.2 Est. [Q1,Q3]')
#     # GPCP v1.3
#     # ax.plot(months, stats_['GPCP_v1pt3_mean'], color='r', label='GPCP v1.3 Est. Mean')
#     # ax.fill_between(months, stats_['GPCP_v1pt3_q1'], stats_['GPCP_v1pt3_q3'], color='r', alpha=0.2, label='GPCP v1.3 Est. [Q1,Q3]')
#     # GPCP v3.3
#     ax.plot(months, stats_['GPCP_v3pt3_mean'], color='g', label='GPCP v3.3 Est. Mean')
#     ax.fill_between(months, stats_['GPCP_v3pt3_q1'], stats_['GPCP_v3pt3_q3'], color='g', alpha=0.2, label='GPCP v3.3 Est. [Q1,Q3]')
#     # IMERG
#     ax.plot(months, stats_['IMERG_mean'], color='orange', label='IMERG Est. Mean')
#     ax.fill_between(months, stats_['IMERG_q1'], stats_['IMERG_q3'], color='orange', alpha=0.2, label='IMERG Est. [Q1,Q3]')
#     # Title, labels
#     ax.set_title(region_titles[region], fontsize=18, fontweight='bold')
#     ax.set_xticks(months)
#     ax.set_xticklabels(month_labels,  fontsize=20, fontweight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.set_ylabel('Rainfall [mm]', fontsize=20, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     # Make y-axis tick labels bold
#     for label in ax.get_yticklabels():
#         label.set_fontweight('bold')
#     # Add N= count
#     n_buoy = len(buoy_files_by_region[region])
#     ax.text(0.02, 0.95, f'N = {n_buoy} Buoys', 
#             transform=ax.transAxes, fontsize=18, 
#             fontweight='bold', va='top')

# # Collect legend handles/labels from the first axis
# handles, labels = axs[0].get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# # Create the legend object
# leg = fig.legend(
#     unique.values(), unique.keys(),
#     loc='lower center', bbox_to_anchor=(0.5, -0.05),
#     fontsize=20, ncol=4, frameon=False
# )

# # Set fontweight to bold for all legend texts
# for text in leg.get_texts():
#     text.set_fontweight('bold')
# if leg.get_title() is not None:
#     leg.get_title().set_fontweight('bold')

# plt.tight_layout(rect=[0, 0.08, 1, 1])
# # Save plot to disk
# monthly_plot_path = os.path.join(path_to_put_plts, f'PAL_satellite_monthly_stats_by_region_{cde_run_dte}.png')
# plt.savefig(monthly_plot_path, bbox_inches='tight', dpi=500)
# plt.show()
# gc.collect()  # Clean up memory


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# plot just the monthly mean
fig, axs = plt.subplots(2, 2, figsize=(25, 10))
axs = axs.flatten()
months = np.arange(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i, region in enumerate(region_order):
    ax = axs[i]
    stats_monthly = monthly_stats[region]
    buoy_monthly = monthly_stats[region + '_buoy_monthly']
    # PAL: plot monthly mean as a dashed black line
    ax.plot(months, stats_monthly['rain_rate_mean'], color='black', linestyle='--', label='Buoy Obs. Mean', linewidth=2)
    # GPCP v3.2
    # ax.plot(months, stats_monthly['GPCP_v3pt2_mean'], color='b', label='GPCP v3.2 Est. Mean', linewidth=2)
    # GPCP v1.3
    # ax.plot(months, stats_monthly['GPCP_v1pt3_mean'], color='r', label='GPCP v1pt3 Est. Mean', linewidth=2)
    # GPCP v3.3
    ax.plot(months, stats_monthly['GPCP_v3pt3_mean'], color='g', label='GPCP v3.3 Est. Mean', linewidth=2)
    # IMERG
    ax.plot(months, stats_monthly['IMERG_mean'], color='orange', label='IMERG Est. Mean', linewidth=2)
    # Title, labels
    ax.set_title(region_titles[region], fontsize=18, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel('Rainfall [mm]', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Make y-axis tick labels bold
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add N= count
    n_buoy = len(buoy_files_by_region[region])
    ax.text(0.02, 0.95, f'N = {n_buoy} Buoys', 
            transform=ax.transAxes, fontsize=18, 
            fontweight='bold', va='top')
    # Add minor ticks to y-axis (no labels)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    # Add major ticks (no labels) for top row
    if i < 3:
        ax.xaxis.set_tick_params(which='major', bottom=True, top=True, labelbottom=True, labeltop=False)
    # Add major and minor ticks (no labels) for right y-axis
    ax.yaxis.set_tick_params(which='major', right=True, labelright=False)
    ax.yaxis.set_tick_params(which='minor', right=True, labelright=False)
    # Remove top and right ticks for all axes
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)

# Collect legend handles/labels from the first axis
handles, labels = axs[0].get_legend_handles_labels()
unique = sorted(dict(zip(labels, handles)).items())
# Create the legend object
leg = fig.legend(
    [item[1] for item in unique], [item[0] for item in unique],
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
monthly_plot_path = os.path.join(path_to_put_plts, f'Buoy_satellite_monthly_means_by_region_{cde_run_dte}.png')
plt.savefig(monthly_plot_path, bbox_inches='tight', dpi=500)
# plt.show()
gc.collect()


#%%

bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']

# for region_name in buoy_files_by_region.keys():
    # df2cmpt = all_daily.loc[all_daily['region'] == region_name].copy()
buoy_pdfc_pdfv = compute_pdf_elements(all_daily, 'rain_rate', bin_values)
img_pdfc_pdfv = compute_pdf_elements(all_daily, 'IMERG', bin_values)
gpcp_pdfc_pdfv = compute_pdf_elements(all_daily, 'GPCP_v3pt3', bin_values)

# Plotting PDF for Buoy, IMERG, and GPCP v3.3

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=500)

# Set common x-axis ticks and labels
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']
bin_positions = range(len(bin_labels))
lw = 2

# Add grid lines and customize ticks
for ax in axs:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Plot PDFc for all products in ax[0]
axs[0].plot(bin_positions, buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='o', lw=lw)
axs[0].plot(bin_positions, img_pdfc_pdfv['pdfc'], label='IMERG', marker='x', lw=lw)
axs[0].plot(bin_positions, gpcp_pdfc_pdfv['pdfc'], label='GPCP v3.3', marker='s', lw=lw)
# axs[0].set_title('PDFc Comparison', fontsize=18, fontweight='bold')
axs[0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[0].set_xticks(bin_positions)
axs[0].set_xticklabels(bin_labels, fontsize=16, fontweight='bold')
axs[0].legend(fontsize=16, frameon=False)

# Plot PDFv for all products in ax[1]
axs[1].plot(bin_positions, buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='o', lw=lw)
axs[1].plot(bin_positions, img_pdfc_pdfv['pdfv'], label='IMERG', marker='x', lw=lw)
axs[1].plot(bin_positions, gpcp_pdfc_pdfv['pdfv'], label='GPCP v3.3', marker='s', lw=lw)
# axs[1].set_title('PDFv Comparison', fontsize=18, fontweight='bold')
axs[1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[1].set_xticks(bin_positions)
axs[1].set_xticklabels(bin_labels, fontsize=16, fontweight='bold')
axs[1].legend(fontsize=16, frameon=False)

# Set common x-axis label
for ax in axs:
    ax.set_xlabel('Rain Rate Bins [mm/day]', fontsize=18, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the figure
svnme = os.path.join(path_to_put_plts, f'pdfc_pdfv_comparison_buoy_imerg_gpcp_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight')
#%%  A CONCENTRATED ANALYSIS OF PAL RAIN RATE WITH ITS WIND SPEED
# windspd_obs_ana_by_region = {}
# rr_obs_ana_by_region = {}
# thr_rr_obs_ana_by_region = {}
# for region_name, pal_files in pals_classed_by_region.items():
#     if region_name != "Unclassified" and len(pal_files) > 0:
#         print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")

#         # store PAL and GPCP dataframes        
#         regional_observations = []

#         # LOAD PAL DATA
#         for pal_file in pal_files:
#             print(f"Processing PAL file: {os.path.basename(pal_file)}")
#             pal_ds = xr.open_dataset(pal_file)

#             # create an analytical df
#             ana_df = pd.DataFrame({
#                 'time': pal_ds['time'].values,
#                 'rain_rate': pal_ds['rain_rate'].values.flatten(),
#                 'wind_speed': pal_ds['wind_speed'].values.flatten()
#             })

#             # set values of both rain rate and wind speed to NaN if they are less than 0
#             ana_df.loc[ana_df['rain_rate'] < 0, 'rain_rate'] = np.nan
#             ana_df.loc[ana_df['wind_speed'] < 0, 'wind_speed'] = np.nan

#             # drop rows with NaN values in either rain_rate or wind_speed
#             ana_df = ana_df.dropna(subset=['rain_rate', 'wind_speed'], axis=0)

#             regional_observations.append(ana_df)

#         # Combine all PAL dataframes for the region
#         regional_dfs = pd.concat(regional_observations, axis=0)
#         # Filter out invalid values
#         # regional_dfs = regional_dfs[(regional_dfs['rain_rate'] >= 0) & (regional_dfs['wind_speed'] >= 0)]
#         # filter rain above 0

#         # Define wind speed bins and labels
#         wind_speed_bins = pd.cut(
#             regional_dfs['wind_speed'], 
#             bins=[-1, 5, 10, 15, float('inf')], 
#             labels=['0-5', '5-10', '10-15', '>15'], 
#             duplicates='drop'
#         )
        
#         # Count of wind speed observations by wind speed bins
#         wind_speed_counts = wind_speed_bins.value_counts().sort_index()
#         wind_speed_counts = wind_speed_counts.reset_index()
#         wind_speed_counts.columns = ['wind_speed_bin', 'count']
#         wind_speed_counts['percentage'] = ((wind_speed_counts['count'] / wind_speed_counts['count'].sum()) * 100).round(2)

#         # Count of rain rate observations by wind speed bins
#         regional_dfs['wind_speed_bins'] = wind_speed_bins
#         rain_rate_counts = regional_dfs.groupby('wind_speed_bins')['rain_rate'].count()
#         rain_rate_counts = rain_rate_counts.reset_index()
#         rain_rate_counts.columns = ['wind_speed_bin', 'count']
#         rain_rate_counts['percentage'] = ((rain_rate_counts['count'] / rain_rate_counts['count'].sum()) * 100).round(2)

#         # Count of rain rate > 0 observations by wind speed bins
#         rain_rate_above_threshold_counts = regional_dfs[regional_dfs['rain_rate'] > 0].groupby('wind_speed_bins')['rain_rate'].count()
#         rain_rate_above_threshold_counts = rain_rate_above_threshold_counts.reset_index()
#         rain_rate_above_threshold_counts.columns = ['wind_speed_bin', 'count']
#         rain_rate_above_threshold_counts['percentage'] = ((rain_rate_above_threshold_counts['count'] / rain_rate_above_threshold_counts['count'].sum()) * 100).round(2)

#         # append to dict
#         windspd_obs_ana_by_region[region_name] = wind_speed_counts
#         rr_obs_ana_by_region[region_name] = rain_rate_counts
#         thr_rr_obs_ana_by_region[region_name] = rain_rate_above_threshold_counts

# gc.collect()  # Clean up memory

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # Plotting wind speed bin comparison across regions

# svnme = os.path.join(path_to_put_plts, f'wind_speed_bin_comparison_{cde_run_dte}.png')
# plot_wind_speed_bin_comparison(
#     windspd_obs_ana_by_region, region_colors,
#     title='Wind Speed Bin Comparison Across Regions',
#     ylabel='Count of Observations',
#     ylabrot=0,
#     output_path=svnme
# )
# # - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - 

# svnme = os.path.join(path_to_put_plts, f'rain_rate_bin_comparison_{cde_run_dte}.png')
# plot_wind_speed_bin_comparison(
#     rr_obs_ana_by_region, region_colors,
#     title='Rain Rate Count per Wind Speed Bin\n Comparison Across Regions',
#     ylabel='Count of Observations',
#     ylabrot=0,
#     output_path=svnme
# )
# # - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - 

# svnme = os.path.join(path_to_put_plts, f'rain_rate_above_threshold_bin_comparison_{cde_run_dte}.png')
# plot_wind_speed_bin_comparison(
#     thr_rr_obs_ana_by_region, region_colors,
#     title='> 0 mm/h Rain Rate Count per Wind Speed Bin\n Comparison Across Regions',
#     ylabel='Count of Observations',
#     ylabrot=0,
#     output_path=svnme
# )

gc.collect()  # Clean up memory

#%% MINIMAL TEST: 1 PAL + 1 GPCP FILE
# import concurrent.futures

# def process_pal_file(args):
#     pal_file, region_name = args
#     pal_ds = xr.open_dataset(pal_file)

#     df = pd.DataFrame({
#         'time': pd.to_datetime(pal_ds['time'].values),
#         'lat': pal_ds['lat'].values,
#         'lon': pal_ds['lon'].values,
#         'rain_rate': pal_ds['rain_rate'].values
#     })

#     df['date'] = df['time'].dt.date
#     df = df.dropna(axis=0, how='any')
#     df['lon'] = (df['lon'] + 360) % 360
#     df['lon'][df['lon'] > 180] -= 360

#     df['row_idx'], df['col_idx'] = assign_to_gpcp_grid(df['lat'], df['lon'], 0.5)
#     df['rain_rate'] = df['rain_rate'].where(df['rain_rate'] >= 0, np.nan)
#     df['region'] = region_name
#     df['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

#     pal_dates = pd.to_datetime(df['date'])
#     pal_lats = df['lat'].values
#     pal_lons = df['lon'].values

#     gpcp_precip = gpcp_ds_v3pt2_xr['precip'].interp(
#         time=("points", pal_dates), lat=("points", pal_lats), lon=("points", pal_lons), method="nearest"
#     )
#     gpcp_plp = gpcp_ds_v3pt2_xr['probability_liquid_phase'].interp(
#         time=("points", pal_dates), lat=("points", pal_lats), lon=("points", pal_lons), method="nearest"
#     )

#     gpcp_plp_daily_avg = pd.DataFrame({
#         'GPCP_v3pt2': gpcp_precip.values,
#         'prob_liq': gpcp_plp.values,
#         'date': pd.to_datetime(gpcp_precip['time'].values).date,
#     })

#     df_cpy = df[['time', 'date', 'rain_rate', 'region', 'track_PAL_id']].copy()
#     gpcp_df_merged = pd.merge(df_cpy, gpcp_plp_daily_avg, on=['date'], how='left')

#     gpcp_plpdf_daily_avg = gpcp_df_merged.groupby('date')[['rain_rate', 'GPCP_v3pt2', 'prob_liq']].mean().reset_index('date')
#     gpcp_plpdf_daily_avg['region'] = region_name
#     gpcp_plpdf_daily_avg['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]
#     gpcp_plpdf_daily_avg['rain_rate'] = gpcp_plpdf_daily_avg['rain_rate'] * 24

#     gpcp_plpdf_daily_avg = gpcp_plpdf_daily_avg[gpcp_plpdf_daily_avg['prob_liq'] == 100]

#     pal_ds.close()
#     return gpcp_plpdf_daily_avg

# regional_PAL_GPCP_dfs_daily_mean = {}
# regional_PAL_GPCP_dfs_daily_lst = []

# import concurrent.futures

# for region_name, pal_files in pals_classed_by_region.items():
#     if region_name == "Unclassified" or len(pal_files) == 0:
#         continue

#     print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")
#     region_pal_gpcp_dfs = []

#     with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
#         results = list(executor.map(process_pal_file, [(pal_file, region_name) for pal_file in pal_files]))
#         region_pal_gpcp_dfs = [res for res in results if res is not None and not res.empty]

#     if not region_pal_gpcp_dfs:
#         continue

#     region_pal_gpcp_df = pd.concat(region_pal_gpcp_dfs)
#     region_pal_gpcp_df_daily_mean = region_pal_gpcp_df.groupby(['track_PAL_id'])[['rain_rate', 'GPCP_v3pt2']].mean().reset_index()
#     region_pal_gpcp_df_daily_mean['region'] = region_name

#     regional_PAL_GPCP_dfs_daily_mean[region_name] = region_pal_gpcp_df_daily_mean
#     regional_PAL_GPCP_dfs_daily_lst.append(region_pal_gpcp_df)

# gc.collect()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


#- - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# regional_PAL_GPCP_dfs_daily_mean = {}
# regional_PAL_GPCP_dfs_daily_lst = []
# for region_name, pal_files in pals_classed_by_region.items():
#     if region_name == "Unclassified" or len(pal_files) == 0:
#         continue

#     print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")
#     region_pal_gpcp_dfs = []

#     # LOAD PAL DATA
#     for pal_file in pal_files:
#         pal_ds = xr.open_dataset(pal_file)
#         print(f"Processing PAL file: {os.path.basename(pal_file)}")

#         # Process PAL data as needed
#         df = pd.DataFrame({
#             'time': pd.to_datetime(pal_ds['time'].values),
#             'lat': pal_ds['lat'].values,
#             'lon': pal_ds['lon'].values,
#             'rain_rate': pal_ds['rain_rate'].values
#         })

#         df['date'] = df['time'].dt.date
#         df = df.dropna(axis=0, how='any')
#         df['lon'] = (df['lon'] + 360) % 360
#         df['lon'][df['lon'] > 180] -= 360

#         # df['row_idx'], df['col_idx'] = assign_to_gpcp_grid(df['lat'], df['lon'], 0.5)
#         df['rain_rate'] = df['rain_rate'].where(df['rain_rate'] >= 0, np.nan)
#         # df['region'] = region_name
#         # df['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]  

#         pal_df_gpcpv3pt2 = df.copy()

#         pal_gpcpv3pt2_daily_avg = process_gpcp_with_PAL2(pal_file, region_name, pal_df_gpcpv3pt2, 
#                            pal_df_gpcpv3pt2, 'GPCP_v3pt2')

#         # pal_dates = pd.to_datetime(df['date'])
#         # pal_lats = df['lat'].values
#         # pal_lons = df['lon'].values        

#         # # Convert dask-backed DataArray to a regular (in-memory) DataArray if needed
#         # # gpcp_precip = gpcp_ds_v3pt2_xr['precip'].compute()
#         # gpcp_precip = gpcp_ds_v3pt2_xr['precip'].interp(
#         #     time=("points", pal_dates), lat=("points", pal_lats), lon=("points", pal_lons), method="nearest"
#         # )
#         # # Set places where the values are less than 0 to NaN
#         # gpcp_precip = gpcp_precip.where(gpcp_precip >= 0, np.nan)

#         # # do same for probability of liquid phase
#         # # gpcp_plp = gpcp_ds_v3pt2_xr['probability_liquid_phase'].compute()
#         # gpcp_plp = gpcp_ds_v3pt2_xr['probability_liquid_phase'].interp(
#         #     time=("points", pal_dates), lat=("points", pal_lats), lon=("points", pal_lons), method="nearest"
#         # )
#         # # Set places where the values are less than 0 to NaN
#         # gpcp_plp = gpcp_plp.where(gpcp_plp >= 0, np.nan)
        

#         # # Store matched values in the DataFrame
#         # df['GPCP_v3pt2'] = gpcp_precip

#         # # Store matched values in the DataFrame
#         # df['PLP_v3pt2'] = gpcp_plp

#         # # sel only where PLP_v3pt2 is 100
#         # df = df[df['PLP_v3pt2'] == 100]

#         # daily_avg = df.groupby('date').agg({
#         #     'rain_rate': 'mean',
#         #     'GPCP_v3pt2': 'mean',
#         #     'PLP_v3pt2': 'mean',            
#         # })
#         # # convert pal rainrate to daily average
#         # daily_avg['rain_rate'] *= 24  # Convert to daily average
#         # daily_avg['region'] = region_name
#         # daily_avg['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]       

#         region_pal_gpcp_dfs.append(pal_gpcpv3pt2_daily_avg)

#     # Combine all region PAL-GPCP dataframes into a single dataframe
#     region_pal_gpcp_df = pd.concat(region_pal_gpcp_dfs)
    
    
#     # calculate daily mean per track_PAL_id
#     region_pal_gpcp_df_daily_mean = region_pal_gpcp_df.groupby(['track_PAL_id'])[['rain_rate', 
#                                                                         'GPCP_v3pt2']].mean()
#     region_pal_gpcp_df_daily_mean['region'] = region_name  # Add region name for clarity
#     regional_PAL_GPCP_dfs_daily_mean[region_name] = region_pal_gpcp_df_daily_mean

#     # Append to the list for later processing
#     regional_PAL_GPCP_dfs_daily_lst.append(region_pal_gpcp_df)

#%%
# pal_gpcv3_2_cmp = pd.concat([df[['rain_rate', 'GPCP_v3pt2']] for df in regional_PAL_GPCP_dfs_daily_mean.values()], ignore_index=True)

# # calculate metrics for all regions combined
# rb_v3pt2, rmse_v3pt2, cc_v3pt2 = calculate_metrics(pal_gpcv3_2_cmp['rain_rate'], pal_gpcv3_2_cmp['GPCP_v3pt2'])

# # make scatter plot for the single PAL and GPCP v3.2
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
# mpl.rcParams['font.weight'] = 'bold'        
# mpl.rcParams['axes.labelweight'] = 'bold'
# mpl.rcParams['axes.titleweight'] = 'bold'
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['xtick.labelsize'] = 14
# mpl.rcParams['ytick.labelsize'] = 14        
# mpl.rcParams['legend.fontsize'] = 12
# mpl.rcParams['legend.title_fontsize'] = 14

# fg, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(pal_gpcv3_2_cmp['rain_rate'], pal_gpcv3_2_cmp['GPCP_v3pt2'], c='blue', alpha=0.5)
# # Set axis limits and ticks
# ax.set_xlim(0, 15)
# ax.set_ylim(0, 15)
# ax.set_xticks([0, 5, 10, 15])
# ax.set_yticks([0, 5, 10, 15])
# # 1:1 line from (0,0) to (15,15)
# ax.plot([0, 15], [0, 15], color='gray', linestyle='--')
# ax.set_xlabel('PAL Observations [mm/day]', fontsize=16, fontweight='bold')
# ax.set_ylabel('GPCP v3.2 Estimates [mm/day]', fontsize=16, fontweight='bold')
# ax.set_title('GPCP v3.2 vs PAL', fontsize=18, fontweight='bold')
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.tick_params(axis='both', which='minor', labelsize=14)
# ax.grid(True, alpha=0.3)
# # Add metrics text to the plot
# ax.text(0.05, 0.95,
#     f'RB: {rb_v3pt2:.2f}%\nRMSE: {rmse_v3pt2:.2f} mm/day\nCC: {cc_v3pt2:.2f}',
#     transform=ax.transAxes, fontsize=14, verticalalignment='top')
# ax.legend(loc='upper left', fontsize=12)
# plt.tight_layout()