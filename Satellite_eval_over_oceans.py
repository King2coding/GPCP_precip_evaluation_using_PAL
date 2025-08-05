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
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v1_pnt_3_2010_2020'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2010_2020'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_2010_2020'

path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_daily_1998-2025'

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'
#%% DEFINE GLOBAL VARIABLES

cde_run_dte = str(date.today().strftime('%Y%m%d'))

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

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
indian_buoy_files = sorted([os.path.join(indian_buoy_dir, f) for f in os.listdir(indian_buoy_dir) if f.endswith('.cdf')])
atlantic_buoy_files = sorted([os.path.join(atlantic_buoy_dir, f) for f in os.listdir(atlantic_buoy_dir) if f.endswith('.cdf')])


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

def simple_process_gpcp_batch(batch, version):
    """Simple processing of GPCP files without intensive optimizations"""
    try:
        processed_batch = xr.open_mfdataset(
            batch, 
            combine='by_coords', 
            parallel=False,  # Disable parallel processing to reduce CPU load
            engine='netcdf4'
        )
        processed_batch = ds_swaplon(processed_batch)
        return processed_batch
    except Exception as e:
        print(f"Error processing {version} batch: {e}")
        return None

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

def simple_process_imerg_batch(batch):
    """Simple IMERG processing without parallel jobs for server-friendly operation"""
    try:
        # Use the existing process_imerg but we'll load files sequentially instead
        processed_batch = xr.open_mfdataset(
            batch, 
            combine='by_coords', 
            parallel=False,  # No parallel processing
            engine='netcdf4'
        )
        return processed_batch
    except Exception as e:
        print(f"Error processing IMERG batch: {e}")
        return None

# Process IMERG files - server friendly version
print(f"Processing IMERG files in batches of {batch_size}...")
imerg_batches = [all_imerg_files[i:i + batch_size] for i in range(0, len(all_imerg_files), batch_size)]
imerg_ds_xr_list = []

for i, batch in enumerate(imerg_batches):
    if i % 30 == 0:
        print(f"Processing IMERG batch {i+1}/{len(imerg_batches)}")
    
    try:
        # Use simple processing instead of the CPU-intensive process_imerg function
        processed_batch = simple_process_imerg_batch(batch)
        if processed_batch is not None:
            imerg_ds_xr_list.append(processed_batch)
    except Exception as e:
        print(f"Error processing IMERG batch {i+1}: {e}")
    
    # Simple garbage collection
    gc.collect()

# Combine all processed batches into a single xarray dataset - simple version
if imerg_ds_xr_list:
    imerg_ds_xr = xr.concat(imerg_ds_xr_list, dim="time")
    print("IMERG loading complete")
else:
    print("Warning: No IMERG data was successfully loaded")
    imerg_ds_xr = None

# pacific_buoy_xr = xr.open_mfdataset(
#     pacific_buoy_files, combine='nested', parallel=False, engine='netcdf4', chunks={}
# )

# indian_buoy_xr = xr.open_mfdataset(
#     indian_buoy_files, combine='by_coords', parallel=False, engine='netcdf4', chunks={}
# )

# atlantic_buoy_xr = xr.open_mfdataset(
#     atlantic_buoy_files, combine='by_coords', parallel=False, engine='netcdf4', chunks={}
# )

# pacific_buoy_xr

gc.collect()  # Clean up memory
print("Data loading phase complete!")
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

# Save checkpoint after data loading and classification
print("Saving data loading checkpoint...")
save_data_loading_checkpoint(
    gpcp_ds_v1pt3_xr=gpcp_ds_v1pt3_xr,
    gpcp_ds_v3pt2_xr=gpcp_ds_v3pt2_xr, 
    gpcp_ds_v3pt3_xr=gpcp_ds_v3pt3_xr,
    imerg_ds_xr=imerg_ds_xr,
    pals_classed_by_region=pals_classed_by_region
)
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

for region, files in pals_classed_by_region.items():
    if region == "Unclassified" or len(files) == 0:
        continue  # Skip unclassified and empty regions for plotting bounds
    
    color = region_colors[region]
    for file in files:
        # Load only lat and lon efficiently, downsample by slicing
        ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
        lat = ds['lat'].values[::50]
        lon = ds['lon'].values[::50]
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=3)
        ds.close()

# Initialize buoy counts
buoy_counts = {"PACIFIC": len(pacific_buoy_files), "INDIAN": len(indian_buoy_files), "ATLANTIC": len(atlantic_buoy_files)}

for bouy_reg, buoy_files, marker in [("PACIFIC", pacific_buoy_files, '*'), 
                                     ("INDIAN", indian_buoy_files, 's'), 
                                     ("ATLANTIC", atlantic_buoy_files, 'd')]:
    for fl in buoy_files: 
        xrfile = xr.open_dataset(fl)       
        # Plot moored buoy data
        lat = xrfile['lat'].values[0]
        lon = xrfile['lon'].values[0]
        # make lon between 180 and -180
        lon = (lon + 180) % 360 - 180

        if lon == -180:
            # shift lon slightly for plotting
            lon = -178

        # print(lon)
        
        ax.scatter(lon, lat, color='black', s=100, marker=marker, label=f'{bouy_reg} Buoys', transform=ccrs.PlateCarree())

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
labels = [f"{region}: ({full_region_names[region]} ({len(pals_classed_by_region.get(region, []))} PALs)" for region in legend_regions]

# Add buoy markers and counts to the legend
handles.extend([
    plt.Line2D([0], [0], color='black', marker='*', markersize=10, linestyle='None'),
    plt.Line2D([0], [0], color='black', marker='s', markersize=10, linestyle='None'),
    plt.Line2D([0], [0], color='black', marker='d', markersize=10, linestyle='None')
])
labels.extend([
    f"PACIFIC ({buoy_counts['PACIFIC']} buoys)",
    f"INDIAN ({buoy_counts['INDIAN']} buoys)",
    f"ATLANTIC ({buoy_counts['ATLANTIC']} buoys)"
])

leg = plt.legend(
    handles, labels,  loc="lower center", bbox_to_anchor=(0.5, -0.35), 
    fontsize=14,ncol=3, frameon=False
) # title="Regions and Buoys",  title_fontsize=14, 
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
ax.set_title("PAL Trajectories and Buoy Locations by Ocean Region", fontsize=20, 
             fontweight='bold', fontname='Times New Roman')

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom for legend

svname = os.path.join(path_to_put_plts, f'PAL_trajectories_by_region_{cde_run_dte}.png')
plt.savefig(svname, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()  # Clean up memory


#%% DO DATA INVENTORY PER REGION
# COUNT THE TOTAL NUMBER OF DAYS PER YEAR WITH NON NAN DATA FOR EACH REGION
# Build inventory: count number of non-NaN daily rain_rate observations per region per year
regional_inventory = []
for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")
        for pal_file in pal_files:
            pal_ds = xr.open_dataset(pal_file)
            df = pd.DataFrame({
                'time': pd.to_datetime(pal_ds['time'].values),
                'rain_rate': pal_ds['rain_rate'].values
            })
            df = df.dropna(subset=['rain_rate'])  # Only keep rows with valid rain_rate
            if not df.empty:
                df.set_index('time', inplace=True)
                # Resample to daily, taking the mean rain_rate per day
                daily_df = df.resample('D').mean()
                daily_df = daily_df.dropna(subset=['rain_rate'])  # Only keep days with valid mean
                daily_df = daily_df.reset_index()
                daily_df['date'] = daily_df['time'].dt.date
                daily_df['year'] = daily_df['time'].dt.year
                daily_df['region'] = region_name
                daily_df['pal_file'] = os.path.basename(pal_file)
                regional_inventory.append(daily_df[['region', 'year', 'date', 'pal_file']])
            pal_ds.close()
gc.collect()  # Clean up memory

# Combine all PALs' valid daily records
regional_inventory_df = pd.concat(regional_inventory, ignore_index=True)
# PAL count per region for legend
region_counts = regional_inventory_df.groupby('region')['pal_file'].nunique().to_dict()

# Count number of valid daily observations per region per year
regional_inventory_df = (
    regional_inventory_df
    .groupby(['region', 'year'])
    .agg(rain_rate=('date', 'count'))
    .reset_index()
)



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
# Set y-axis ticks to show as 2000, 4000, 6000, 8000, etc.
# yticks = np.arange(0, regional_inventory_df['rain_rate'].max() + 2000, 2000)
# ax.set_yticks(yticks)
# ax.set_yticklabels([f"{int(y):,}" for y in yticks])

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
svname = os.path.join(path_to_put_plts, f'region_daily_observation_inventory_{cde_run_dte}.png')
plt.savefig(svname, dpi=500, bbox_inches='tight')
gc.collect()  # Clean up memory


#%% SPATIOTEMPORAL MATCHING OF PAL AND GPCP DATA
# regional_PAL_GPCP_wind_dfs_dict = {}
# regional_PAL_GPCP_wind_dfs_lst = []
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

            print(f"Processing PAL file: {os.path.basename(pal_file)}")

            # gran required dfs
            # , pal_wind_df
            pal_rain_df = grab_PAL_rain_and_wind_df(pal_ds)             

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            # Process GPCP data with PAL
            
            pal_rain_gpcpv1pt3_df = pal_rain_df.copy()
            
            pal_gpcpv1pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                            pal_rain_gpcpv1pt3_df,                                             
                                            gpcp_ds_v1pt3_xr, 'GPCP_v1pt3')
            #  , pal_gpcpv1pt3_df_wind, pal_wind_gpcpv1pt3_df,
            
            pal_gpcpv1pt3_df_rain.index = pd.to_datetime(pal_gpcpv1pt3_df_rain['time'])
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            pal_rain_gpcpv3pt2_df = pal_rain_df.copy()   
            
            pal_gpcpv3pt2_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                            pal_rain_gpcpv3pt2_df,
                                            gpcp_ds_v3pt2_xr, 'GPCP_v3pt2') # , pal_wind_gpcpv3pt2_df,

            pal_gpcpv3pt2_df_rain.index = pd.to_datetime(pal_gpcpv3pt2_df_rain['time'])  # Ensure index is datetime

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
            pal_rain_gpcpv3pt3_df = pal_rain_df.copy()           
            
            pal_gpcpv3pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                            pal_rain_gpcpv3pt3_df,
                                            gpcp_ds_v3pt3_xr, 'GPCP_v3pt3')  # , pal_wind_gpcpv3pt3_df

            pal_gpcpv3pt3_df_rain.index = pd.to_datetime(pal_gpcpv3pt3_df_rain['time'])        

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
            # Process IMERG data with PAL - DISABLED for server-friendly operation
            # Re-enable with single-threaded version later
            # print(f"Skipping IMERG-PAL matching for {os.path.basename(pal_file)} (server-friendly mode)")
            # Use the new fast vectorized IMERG matching function
            pal_imerg_df_rain = process_imerg_with_PAL_rainV2(pal_rain_df, imerg_ds_xr)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -    

        # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data        
        # COMBINE BY RAINFALL RATE
        pal_df_combined_rain = pal_gpcpv1pt3_df_rain.copy()
        pal_df_combined_rain = pal_df_combined_rain[['date','rain_rate', 'GPCP_v1pt3']].copy()

        # merge GPCP v3.2 data            
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv3pt2_df_rain[['date','GPCP_v3pt2', 'PLP_GPCP_v3pt2']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt2')
        )
        
        # Remove any duplicate columns from previous merges
        for col in ['GPCP_v3pt2_v3pt2', 'PLP_GPCP_v3pt2_v3pt2', 'date_v3pt2']:
            if col in pal_df_combined_rain.columns:
                pal_df_combined_rain.drop(columns=col, inplace=True)

        # merge GPCP v3.3 data
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv3pt3_df_rain[['date','GPCP_v3pt3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt3')
        )

        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['GPCP_v3pt3_v3pt3', 'date_v3pt3']], 
                                                inplace=True)

        # retain only columns where PLP_GPCP_v3pt2 is == 100
        pal_df_combined_rain = pal_df_combined_rain[pal_df_combined_rain['PLP_GPCP_v3pt2'] == 100]

        # groupby date and get mean of rain_rate and GPCP data
        daily_avg_rain = pal_df_combined_rain.groupby('date').mean([['rain_rate', 'GPCP_v1pt3', 'GPCP_v3pt2', 'GPCP_v3pt3']]).reset_index()
        # multiply PAL rain rate by 24 to get daily average
        daily_avg_rain['rain_rate'] *= 24
        # add region name and track_PAL_id to the dataframe
        daily_avg_rain['region'] = region_name  # Add region name for clarity
        daily_avg_rain['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

        region_pal_gpcp_dfs.append(daily_avg_rain)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -             

        

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

    # Append wind data by region
    # regional_PAL_GPCP_wind_dfs_dict[region_name] = pd.concat(regional_PAL_GPCP_wind_dfs_lst, ignore_index=True)

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

# Get PAL counts per region for legend
region_pal_counts = {region: len(pals_classed_by_region[region]) for region in regional_PAL_GPCP_dfs_daily_mean.keys()}

for region, df in regional_PAL_GPCP_dfs_daily_mean.items():
    reg_col = region_colors[region]
    pal_gpcv1_3 = df[['rain_rate', 'GPCP_v1pt3']]
    pal_gpcv3_2 = df[['rain_rate', 'GPCP_v3pt2']]
    pal_gpcv3_3 = df[['rain_rate', 'GPCP_v3pt3']]

    # Plot GPCP v1.3
    ax[0].scatter(pal_gpcv1_3['rain_rate'], pal_gpcv1_3['GPCP_v1pt3'],
                  color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)
    # Plot GPCP v3.2
    ax[1].scatter(pal_gpcv3_2['rain_rate'] , pal_gpcv3_2['GPCP_v3pt2'],
                  color=reg_col, label=f"{region} ({region_pal_counts[region]})", s=80)
    # Plot GPCP v3.3
    ax[2].scatter(pal_gpcv3_3['rain_rate'], pal_gpcv3_3['GPCP_v3pt3'],
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
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)
ax[1].text(
    0.05, 0.95,
    f'RB: {rb_v3pt2:.2f}%\nRMSE: {rmse_v3pt2:.2f} mm/day\nCC: {cc_v3pt2:.2f}',
    transform=ax[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
    bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
)
ax[2].text(
    0.05, 0.95,
    f'RB: {rb_v3pt3:.2f}%\nRMSE: {rmse_v3pt3:.2f} mm/day\nCC: {cc_v3pt3:.2f}',
    transform=ax[2].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
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
    title="Region (N PALs)"
)
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

# plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend
# save the figure
svnme = os.path.join(path_to_put_plts, f'PAL_GPCP_scatter_plots_{cde_run_dte}.png')
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

fig, axs = plt.subplots(2, 3, figsize=(25, 10))
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
monthly_plot_path = os.path.join(path_to_put_plts, f'PAL_GPCP_monthly_stats_by_region_{cde_run_dte}.png')
plt.savefig(monthly_plot_path, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()  # Clean up memory


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# plot just the monthly mean
fig, axs = plt.subplots(2, 3, figsize=(25, 10))
axs = axs.flatten()
months = np.arange(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i, region in enumerate(region_order):
    ax = axs[i]
    stats = monthly_stats[region]
    pal_monthly = monthly_stats[region + '_pal_monthly']
    # PAL: plot monthly mean as a dashed black line
    ax.plot(months, stats['rain_rate_mean'], color='black', linestyle='--', label='PAL Obs. Mean', linewidth=2)
    # GPCP v3.2
    ax.plot(months, stats['GPCP_v3pt2_mean'], color='b', label='GPCP v3.2 Est. Mean', linewidth=2)
    # GPCP v1.3
    ax.plot(months, stats['GPCP_v1pt3_mean'], color='r', label='GPCP v1pt3 Est. Mean', linewidth=2)
    # GPCP v3.3
    ax.plot(months, stats['GPCP_v3pt3_mean'], color='g', label='GPCP v3.3 Est. Mean', linewidth=2)
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
    n_pal = len(pals_classed_by_region[region])
    ax.text(0.02, 0.95, f'N = {n_pal} PALs', 
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
monthly_plot_path = os.path.join(path_to_put_plts, f'PAL_GPCP_monthly_means_by_region_{cde_run_dte}.png')
plt.savefig(monthly_plot_path, bbox_inches='tight', dpi=500)
plt.show()
gc.collect()


#%% DO SOME WINDY ANALYSIS
# col2ana = ['rain_rate', 'wind_speed', 'GPCP_v1pt3', 'GPCP_v3pt2', 'GPCP_v3pt3']
# for region_name,data_df in regional_PAL_GPCP_wind_dfs_dict.items():
#     print(f"Region: {region_name}, Number of records: {(data_df.shape[0])}")

#     data2ana = data_df[col2ana].copy()
#     data2ana = data2ana.dropna(axis=0, how='any')

#     rr_data2ana = data2ana[['rain_rate', 'GPCP_v1pt3', 'GPCP_v3pt2', 'GPCP_v3pt3']].copy()

#     # DO PDFC AND PDFV
#     pdfc_pdfv_results = compute_rainfall_fraction_and_volume_by_windspeed_bins(
#         data2ana, col2ana, bn_size=2, threshold=0.5)
    

#%%  A CONCENTRATED ANALYSIS OF PAL RAIN RATE WITH ITS WIND SPEED
windspd_obs_ana_by_region = {}
rr_obs_ana_by_region = {}
thr_rr_obs_ana_by_region = {}
for region_name, pal_files in pals_classed_by_region.items():
    if region_name != "Unclassified" and len(pal_files) > 0:
        print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")

        # store PAL and GPCP dataframes        
        regional_observations = []

        # LOAD PAL DATA
        for pal_file in pal_files:
            print(f"Processing PAL file: {os.path.basename(pal_file)}")
            pal_ds = xr.open_dataset(pal_file)

            # create an analytical df
            ana_df = pd.DataFrame({
                'time': pal_ds['time'].values,
                'rain_rate': pal_ds['rain_rate'].values.flatten(),
                'wind_speed': pal_ds['wind_speed'].values.flatten()
            })

            # set values of both rain rate and wind speed to NaN if they are less than 0
            ana_df.loc[ana_df['rain_rate'] < 0, 'rain_rate'] = np.nan
            ana_df.loc[ana_df['wind_speed'] < 0, 'wind_speed'] = np.nan

            # drop rows with NaN values in either rain_rate or wind_speed
            ana_df = ana_df.dropna(subset=['rain_rate', 'wind_speed'], axis=0)

            regional_observations.append(ana_df)

        # Combine all PAL dataframes for the region
        regional_dfs = pd.concat(regional_observations, axis=0)
        # Filter out invalid values
        # regional_dfs = regional_dfs[(regional_dfs['rain_rate'] >= 0) & (regional_dfs['wind_speed'] >= 0)]
        # filter rain above 0

        # Define wind speed bins and labels
        wind_speed_bins = pd.cut(
            regional_dfs['wind_speed'], 
            bins=[-1, 5, 10, 15, float('inf')], 
            labels=['0-5', '5-10', '10-15', '>15'], 
            duplicates='drop'
        )
        
        # Count of wind speed observations by wind speed bins
        wind_speed_counts = wind_speed_bins.value_counts().sort_index()
        wind_speed_counts = wind_speed_counts.reset_index()
        wind_speed_counts.columns = ['wind_speed_bin', 'count']
        wind_speed_counts['percentage'] = ((wind_speed_counts['count'] / wind_speed_counts['count'].sum()) * 100).round(2)

        # Count of rain rate observations by wind speed bins
        regional_dfs['wind_speed_bins'] = wind_speed_bins
        rain_rate_counts = regional_dfs.groupby('wind_speed_bins')['rain_rate'].count()
        rain_rate_counts = rain_rate_counts.reset_index()
        rain_rate_counts.columns = ['wind_speed_bin', 'count']
        rain_rate_counts['percentage'] = ((rain_rate_counts['count'] / rain_rate_counts['count'].sum()) * 100).round(2)

        # Count of rain rate > 0 observations by wind speed bins
        rain_rate_above_threshold_counts = regional_dfs[regional_dfs['rain_rate'] > 0].groupby('wind_speed_bins')['rain_rate'].count()
        rain_rate_above_threshold_counts = rain_rate_above_threshold_counts.reset_index()
        rain_rate_above_threshold_counts.columns = ['wind_speed_bin', 'count']
        rain_rate_above_threshold_counts['percentage'] = ((rain_rate_above_threshold_counts['count'] / rain_rate_above_threshold_counts['count'].sum()) * 100).round(2)

        # append to dict
        windspd_obs_ana_by_region[region_name] = wind_speed_counts
        rr_obs_ana_by_region[region_name] = rain_rate_counts
        thr_rr_obs_ana_by_region[region_name] = rain_rate_above_threshold_counts

gc.collect()  # Clean up memory

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plotting wind speed bin comparison across regions

svnme = os.path.join(path_to_put_plts, f'wind_speed_bin_comparison_{cde_run_dte}.png')
plot_wind_speed_bin_comparison(
    windspd_obs_ana_by_region, region_colors,
    title='Wind Speed Bin Comparison Across Regions',
    ylabel='Count of Observations',
    ylabrot=0,
    output_path=svnme
)
# - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - 

svnme = os.path.join(path_to_put_plts, f'rain_rate_bin_comparison_{cde_run_dte}.png')
plot_wind_speed_bin_comparison(
    rr_obs_ana_by_region, region_colors,
    title='Rain Rate Count per Wind Speed Bin\n Comparison Across Regions',
    ylabel='Count of Observations',
    ylabrot=0,
    output_path=svnme
)
# - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - 

svnme = os.path.join(path_to_put_plts, f'rain_rate_above_threshold_bin_comparison_{cde_run_dte}.png')
plot_wind_speed_bin_comparison(
    thr_rr_obs_ana_by_region, region_colors,
    title='> 0 mm/h Rain Rate Count per Wind Speed Bin\n Comparison Across Regions',
    ylabel='Count of Observations',
    ylabrot=0,
    output_path=svnme
)

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