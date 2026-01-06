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

from multiprocessing import Pool
from rasterio.warp import Resampling
from pyproj import CRS

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

#%% DEFINE PATH TO DATA
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v1_pnt_3_2010_2020'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/GPCP/v3.2/daily'
# r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2000_2020'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/GPCP/v3.3/daily'
# r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_1998_2024'

path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_daily_1998-2025'

path_to_era5_tp = r'/ra1/pubdat/ECMWF/ERA5/daily'

path_to_merra2 = r'/ra1/pubdat/MERRA/Daily'

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'

path_to_put_dfs = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs'


#%% DEFINE GLOBAL VARIABLES

cde_run_dte = str(date.today().strftime('%Y%m%d'))

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])
# all_gpcp_v3pt3_files_2010_2020 = [
#                                   f for f in all_gpcp_v3pt3_files 
#                                   if 2010 <= int(os.path.basename(f).split('_')[2][:4]) <= 2020]

all_imerg_files = sorted([os.path.join(path_to_imerg, f) for f in os.listdir(path_to_imerg) if f.endswith('.nc4')])

all_era5_tp_files = sorted([os.path.join(path_to_era5_tp, f) for f in os.listdir(path_to_era5_tp) if f'era5_tp_' in f and f.endswith('.nc')])

all_merra2_files = sorted([os.path.join(path_to_merra2, f) for f in os.listdir(path_to_merra2) if f.endswith('.nc4')])


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


# - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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

# %%
# === TNWP/TNEP PALs + PACIFIC buoys inside GPCP boxes (0..360°) ===
# with correct E/W tick labels for central_longitude=180 and coastal context

import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# ------------------- font sizes -------------------
TITLE_FONTSIZE  = 13
BOX_LABEL_SIZE   = 11
LEGEND_FONTSIZE  = 11
LINEWIDTH_PAL    = 2.6
LINEWIDTH_BOX    = 2.2
STAR_SIZE        = 70

# ------------------- map window -------------------
LON_MIN, LON_MAX = 125, 210
LAT_MIN, LAT_MAX = -20, 20

# ------------------- helpers -------------------
def to_0360(lon):
    """Force longitude(s) into [0, 360). Works for scalar/array."""
    a = np.asarray(lon, dtype=float)
    return np.mod(a % 360.0, 360.0)

# Dateline inclusion toggle:
# False => 160 <= lon < 180 (right edge OPEN); 180.0 is EXCLUDED
# True  => 160 <= lon <= 180 (right edge CLOSED); 180.0 is INCLUDED
INCLUDE_RIGHT_EDGE = False

def inside_box(lat, lon, b):
    lat_ok = (lat >= b["lat_min"]) & (lat <= b["lat_max"])
    if INCLUDE_RIGHT_EDGE:
        lon_ok = (lon >= b["lon_min"]) & (lon <= b["lon_max"])
    else:
        lon_ok = (lon >= b["lon_min"]) & (lon <  b["lon_max"])
    return lat_ok & lon_ok

def any_box_mask(lat, lon, boxes):
    m = np.zeros_like(lat, dtype=bool)
    for b in boxes:
        m |= inside_box(lat, lon, b)
    return m

# CORRECT longitude labeler for central_longitude=180
# Cartopy gives ticks in a frame offset by -180; undo that, then label.
def lon_label_wrap(x, pos=None):
    true_lon = (x + 180.0) % 360.0         # back to geodetic 0..360
    if np.isclose(true_lon, 180.0):
        return "180°"                      # show the dateline plainly
    if true_lon < 180.0:
        return f"{int(round(true_lon))}°E"
    return f"{int(round(360.0 - true_lon))}°W"

def lat_label_ns(y, pos=None):
    if np.isclose(y, 0.0): return "0°"
    hemi = "N" if y > 0 else "S"
    return f"{int(round(abs(y)))}°{hemi}"

# ------------------- boxes (0…360°) -------------------
# boxes = [
#     dict(name="Box 1", lat_min=  5, lat_max= 15, lon_min=160, lon_max=175, color="k",          ls="--"),
#     dict(name="Box 2", lat_min=-15, lat_max= -5, lon_min=170, lon_max=200, color="tab:green",  ls="--"),
#     dict(name="Box 3", lat_min= -5, lat_max=  5, lon_min=160, lon_max=180, color="royalblue",  ls="--"),
#     dict(name="Box 4", lat_min=  5, lat_max= 15, lon_min=135, lon_max=160, color="orange",     ls="--"),
# ]
boxes = [
    dict(name="Box 1", lat_min=  4, lat_max= 16, lon_min=159, lon_max=180, color="k",          ls="--"),
    dict(name="Box 2", lat_min=-16, lat_max= -7, lon_min=170, lon_max=200, color="tab:green",  ls="--"),
    dict(name="Box 3", lat_min= -7, lat_max=  4, lon_min=159, lon_max=180, color="royalblue",  ls="--"),
    dict(name="Box 4", lat_min=  0.5, lat_max= 16, lon_min=125, lon_max=159, color="orange",     ls="--"),
]
# which PAL regions to include
include_tnep = True
regions      = ["TNWP"] + (["TNEP"] if include_tnep else [])
pal_colors   = {"TNWP": "#2ecc71", "TNEP": "#1f77b4"}  # green / blue

# ------------------- figure / map -------------------
proj = ccrs.PlateCarree(central_longitude=180)
fig  = plt.figure(figsize=(9.2, 5.0), dpi=200)
ax   = plt.axes(projection=proj)

# exact window
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

# context: land & coastlines (switch to with_scale("50m") or "10m" if desired)
ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)

# gridlines as lines only
xgrid = np.arange(LON_MIN, LON_MAX + 1, 10)
ygrid = np.arange(LAT_MIN, LAT_MAX + 1,  5)
gl = ax.gridlines(linewidth=0.4, color="grey", alpha=0.55, linestyle="--",
                  xlocs=xgrid, ylocs=ygrid)
gl.top_labels = gl.right_labels = False

# axis ticks + custom formatters (now correct)
ax.set_xticks(xgrid, crs=ccrs.PlateCarree())
ax.set_yticks(ygrid, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_label_wrap))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_label_ns))
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontsize(11); lab.set_fontweight("bold")

# draw boxes + colored names
for b in boxes:
    rect = Rectangle((b["lon_min"], b["lat_min"]),
                     b["lon_max"]-b["lon_min"], b["lat_max"]-b["lat_min"],
                     fill=False, ec=b["color"], ls=b["ls"], lw=LINEWIDTH_BOX,
                     transform=ccrs.PlateCarree(), zorder=3)
    ax.add_patch(rect)
    # ax.text(b["lon_min"]+0.8, b["lat_max"]-1.2, b["name"],
    #         transform=ccrs.PlateCarree(), fontsize=BOX_LABEL_SIZE, weight="bold",
    #         color=b["color"], bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    #         zorder=5)

# ------------------- PALs (only in-box segments) -------------------
def plot_pals_segmented(pals_classed_by_region):
    for region in regions:
        color = pal_colors[region]
        for f in pals_classed_by_region.get(region, []):
            with xr.open_dataset(f) as ds:
                lat = np.asarray(ds["lat"]).astype(float)[::25]
                lon = to_0360(np.asarray(ds["lon"]).astype(float)[::25])
            m = any_box_mask(lat, lon, boxes)
            if not m.any(): 
                continue
            lat_seg, lon_seg = lat.copy(), lon.copy()
            lat_seg[~m] = np.nan; lon_seg[~m] = np.nan
            ax.plot(lon_seg, lat_seg, color=color, lw=LINEWIDTH_PAL, alpha=0.95,
                    transform=ccrs.PlateCarree(), zorder=2)

# ------------------- PACIFIC buoys (only those in boxes) -------------------
def plot_pacific_buoys(pacific_buoy_files):
    for fl in pacific_buoy_files:
        with xr.open_dataset(fl) as dsb:
            blat = float(dsb["lat"].values[0])
            blon = float(dsb["lon"].values[0])
        blon = to_0360(np.round(blon, 6))  # safety + tiny rounding near 180
        if any(inside_box(blat, blon, b) for b in boxes):
            ax.scatter(blon, blat, s=STAR_SIZE, c="k", marker="*",
                       transform=ccrs.PlateCarree(), zorder=4)

# ---- plot
plot_pals_segmented(pals_classed_by_region)
plot_pacific_buoys(pacific_buoy_files)

# ------------------- legend -------------------
handles = [
    plt.Line2D([0],[0], color="k",         ls="--", lw=LINEWIDTH_BOX, label="Box 1 (4–16°N, 159–180°E)"),
    plt.Line2D([0],[0], color="tab:green", ls="--", lw=LINEWIDTH_BOX, label="Box 2 (7–16°S, 170–200°E)"),
    plt.Line2D([0],[0], color="royalblue", ls="--", lw=LINEWIDTH_BOX, label="Box 3 (4°N–7°S, 159–180°E)"),
    plt.Line2D([0],[0], color="orange",    ls="--", lw=LINEWIDTH_BOX, label="Box 4 (0.5–16°N, 125–159°E)"),
    plt.Line2D([0],[0], color=pal_colors["TNWP"], lw=LINEWIDTH_PAL, label="TNWP PALs (in-box segments)")
]
if include_tnep:
    handles.append(plt.Line2D([0],[0], color=pal_colors["TNEP"], lw=LINEWIDTH_PAL, label="TNEP PALs (in-box segments)"))
handles.append(plt.Line2D([0],[0], color="k", marker="*", linestyle="None", markersize=10, label="PACIFIC buoys"))

leg = ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2,
                loc="lower center", bbox_to_anchor=(0.5, -0.45), handlelength=2.6, columnspacing=1.3)
for t in leg.get_texts():
    t.set_fontweight("bold")

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.show()
gc.collect()
#%%# Read and process satellite precipitation data into xr datasets

# 1) Process GPCP v3.2 - Memory efficient version
all_gpcpv3pt2_files_2010_2021 = [
    f for f in all_gpcp_v3pt2_files 
    if 2010 <= int(os.path.basename(f).split('_')[2][:4]) <= 2021
]
print(f"Processing GPCP v3.2 files in total number of {len(all_gpcpv3pt2_files_2010_2021)}...")

gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcpv3pt2_files_2010_2021,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                    cache=False
                                    )

gpcp_ds_v3pt2_xr = ds_swaplon(gpcp_ds_v3pt2_xr)

print("GPCP v3.2 loading complete!")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# 2) Process GPCP v3.3 - Memory efficient version
all_gpcpv3pt3_files_2010_2021 = [
    f for f in all_gpcp_v3pt3_files 
    if 2010 <= int(os.path.basename(f).split('_')[2][:4]) <= 2021
]
print(f"Processing GPCP v3.3 files in total number of {len(all_gpcpv3pt3_files_2010_2021)}...")

gpcp_ds_v3pt3_xr = xr.open_mfdataset(all_gpcpv3pt3_files_2010_2021,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                    
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                    cache=False
                                    )
print("GPCP v3.3 loading complete!")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# 3) Process IMERG files - Memory efficient version
all_imerg_files_2010_2021 = [
    f for f in all_imerg_files 
    if 2010 <= int(os.path.basename(f).split('.')[4][:4]) <= 2021
]
print(f"Processing IMERG files in total number of {len(all_imerg_files_2010_2021)}...")

imerg_ds_xr_list = []

imerg_xr_data = process_imerg(all_imerg_files_2010_2021, product="imerg_fn")

imerg_xr_data = harmonize_to_target(imerg_xr_data, gpcp_ds_v3pt2_xr)

gc.collect()  # Clean up memory
print("Data loading phase complete!")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
# 4) Process ERA5 files - Multiprocessing version
all_era5_tp_files_2010_2021 = [
    f for f in all_era5_tp_files 
    if 2010 <= int(os.path.basename(f).split('_')[2]) <= 2021
]
print(f"Processing ERA5 files in total number of {len(all_era5_tp_files_2010_2021)}...")
def process_era5_file(file_info):
    import rioxarray 
    idx, file_path = file_info
    if idx % 5 == 0:
        print(f"Processing ERA5 file {idx+1}/{len(all_era5_tp_files_2010_2021)}")
    era5_xr = xr.open_dataset(file_path, engine='netcdf4')
    era5_xr = ds_swaplon(era5_xr)
    era5_xr = era5_xr.rename({'valid_time': 'time'})

    # data units are in m per day, convert to mm/day using 1000 factor
    era5_xr['tp'] = era5_xr['tp'] * 1000  # mm/h
    era5_xr['tp'] = era5_xr['tp'] * 24  # mm/day
    # resample to 0.5 degree resolution
    cc = CRS.from_authority(code=4326, auth_name='EPSG')
    era5_xr.rio.write_crs(cc.to_string(), inplace=True)
    # era5_xr = era5_xr.rio.reproject(
    #     era5_xr.rio.crs,
    #     shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    #     resampling=Resampling.bilinear,
    # )
    return era5_xr


# Use multiprocessing to process files in parallel
era5_ds_xr_list = []
with Pool(processes=6) as pool:  # Adjust the number of processes as needed
    era5_ds_xr_list = pool.map(process_era5_file, enumerate(all_era5_tp_files_2010_2021))
# Combine all processed batches into a single xarray dataset - simple version
if era5_ds_xr_list:
    era5_ds_xr = xr.concat(era5_ds_xr_list, dim="time")
    era5_ds_xr = harmonize_to_target(era5_ds_xr, gpcp_ds_v3pt2_xr)
    print("ERA5 loading complete")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
# Merra2 processingg using multiprocessing
all_merra2_files_2010_2021 = [
    f for f in all_merra2_files 
    if 2010 <= int(os.path.basename(f).split('.')[5][:4]) <= 2021
]
print(f"Processing MERRA2 files in total number of {len(all_merra2_files_2010_2021)}...")
def process_merra2_file(file_info):
    idx, file_path = file_info
    try:
        if idx % 1000 == 0:
            print(f"Processing MERRA2 file {idx+1}/{len(all_merra2_files)}")
        mer2_xr = xr.open_dataset(file_path, engine='netcdf4')
        # convert units in kg m-2 s-1 to mm/day by a factor of 3600*24
        mer2_xr = mer2_xr['PRECTOTCORR'] * 3600
        mer2_xr = mer2_xr.mean(dim='time')
        mer2_xr = mer2_xr * 24  # convert to mm/day
        # Add a time dimension based on the file name or metadata
        time = pd.to_datetime(os.path.basename(file_path).split('.')[5], format='%Y%m%d')
        mer2_xr = mer2_xr.expand_dims(time=[time])
        # resample to 0.5 degree resolution
        cc = CRS.from_authority(code=4326, auth_name='EPSG')
        mer2_xr.rio.write_crs(cc.to_string(), inplace=True)
        # Set spatial dimensions explicitly
        mer2_xr = mer2_xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        # mer2_xr = mer2_xr.rio.reproject(
        #     mer2_xr.rio.crs,
        #     shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:],  # (360, 720), set the shape as the GPCP data
        #     resampling=Resampling.bilinear,
        # )
        return mer2_xr
    except Exception as e:
        print(f"Error processing file: {os.path.basename(file_path)}")
        print(f"Error details: {e}")
        return None

# Use multiprocessing to process files in parallel
mer2_ds_xr_list = []
with Pool(processes=6) as pool:  # Adjust the number of processes as needed
    mer2_ds_xr_list = pool.map(process_merra2_file, enumerate(all_merra2_files_2010_2021))
# Combine all processed batches into a single xarray dataset - simple version
if mer2_ds_xr_list:
    mer2_ds_xr_list = [ds for ds in mer2_ds_xr_list if ds is not None]  # Filter out None values
    mer2_ds_xr = xr.concat(mer2_ds_xr_list, dim="time")
    mer2_ds_xr = harmonize_to_target(mer2_ds_xr, gpcp_ds_v3pt2_xr)
    print("MERRA2 loading complete")

#%% Match PALS and Buoys to Satellite grid boxes

# 1) Define analysis box boundaries for WP boxes

pals_classed_by_wp_bx = classify_and_group_files_bounding_box(pals_classed_by_region['TNWP'], box_ana_bnds_WP)
pals_classed_by_ep_bx = classify_and_group_files_bounding_box(pals_classed_by_region['TNEP'], box_ana_bnds_WP)

pals_classed_by_bx = {
    'Box 1': pals_classed_by_wp_bx['Box 1'] + pals_classed_by_ep_bx['Box 1'],
    'Box 2': pals_classed_by_wp_bx['Box 2'] + pals_classed_by_ep_bx['Box 2'],
    'Box 3': pals_classed_by_wp_bx['Box 3'] + pals_classed_by_ep_bx['Box 3'],
    'Box 4': pals_classed_by_wp_bx['Box 4'] + pals_classed_by_ep_bx['Box 4']
}

buoy_classed_by_wp_bx = classify_and_group_files_bounding_box(buoy_files_by_region['WNP'], box_ana_bnds_WP)

#---------------------------------------------------------------------------------
# 2) Begin extraction of satellite data for PALs and Buoys in boxes

for bx, pal_files in list(pals_classed_by_bx.items())[:-1]:
    
    print(f"\nProcessing region: {bx} with {len(pal_files)} PAL files")      

    # store PAL and GPCP dataframes
    bx_pal_sate_dfs = []     

    # LOAD PAL DATA
    for i,pal_file in enumerate(pal_files):
        
        pal_ds = xr.open_dataset(pal_file)
        
        pal_rain_df = grab_PAL_rain_and_wind_df(pal_ds) 

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        pal_rain_gpcpv3pt2_df = pal_rain_df.copy()           
        
        pal_gpcpv3pt2_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                        pal_rain_gpcpv3pt2_df,
                                        gpcp_ds_v3pt2_xr, 'GPCP_v3pt2') 

        pal_gpcpv3pt2_df_rain.index = pd.to_datetime(pal_gpcpv3pt2_df_rain['time'])  # Ensure index is datetime

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        pal_rain_gpcpv3pt3_df = pal_rain_df.copy()          
        
        pal_gpcpv3pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                        pal_rain_gpcpv3pt3_df,
                                        gpcp_ds_v3pt3_xr, 'GPCP_v3pt3')  # , pal_wind_gpcpv3pt3_df

        pal_gpcpv3pt3_df_rain.index = pd.to_datetime(pal_gpcpv3pt3_df_rain['time'])    

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        # Process ERA5 data with PAL 
        pal_rain_era5_df = pal_rain_df.copy()         
        pal_era5_df_rain = process_era5_with_PAL_rain_and_wind_v1(pal_rain_era5_df, era5_ds_xr)   
        pal_era5_df_rain.index = pd.to_datetime(pal_era5_df_rain['time'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        # Process IMERG data with PAL
        pal_rain_imerg_df = pal_rain_df.copy()
        pal_imerg_df_rain = process_imerg_with_PAL_rain_simple(pal_rain_imerg_df, imerg_xr_data)
        pal_imerg_df_rain.index = pd.to_datetime(pal_imerg_df_rain['time'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        # Process MERRA2 data with PAL
        pal_rain_merra2_df = pal_rain_df.copy()
        pal_merra2_df_rain = process_merra2_with_PAL_rain_simple(pal_rain_merra2_df, mer2_ds_xr)
        pal_merra2_df_rain.index = pd.to_datetime(pal_merra2_df_rain['time'])

        # Combine all data into a single dataframe
        pal_df_combined_rain = pal_gpcpv3pt2_df_rain.copy()
        pal_df_combined_rain = pal_df_combined_rain[['time','date','rain_rate', 
                                                     'GPCP_v3pt2','PLP_GPCP_v3pt2']].copy()
        
        pal_gpcpv3pt3_daily = pal_gpcpv3pt3_df_rain.copy()        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv3pt3_df_rain[['date','GPCP_v3pt3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt3')
        )

        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['GPCP_v3pt3_v3pt3', 'date_v3pt3']], 
                                                inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # ERA5 merge
        pal_era5_daily = pal_era5_df_rain.copy()
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_era5_df_rain[['date','ERA5']], 
            on='date', how='left', suffixes=('', '_ERA5')
        )
        
        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['ERA5_ERA5', 'date_ERA5']], 
                                                inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # IMERG merge
        pal_imerg_daily = pal_imerg_df_rain.copy()
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_imerg_df_rain[['date','IMERG']], 
            on='date', how='left', suffixes=('', '_IMERG')
        )
        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['IMERG_IMERG', 'date_IMERG']], 
                                                inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # MERRA2 merge
        pal_merra2_daily = pal_merra2_df_rain.copy()
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_merra2_df_rain[['date','MERRA2']], 
            on='date', how='left', suffixes=('', '_MERRA2')
        )
        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['MERRA2_MERRA2', 'date_MERRA2']], 
                                                inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # groupby date and get mean of rain_rate and GPCP data 'GPCP_v1pt3',
        grp = pal_df_combined_rain.groupby('date')
        daily_avg_rain = grp.mean([['rain_rate', 
                                    'GPCP_v3pt2', 'GPCP_v3pt3', 
                                    'ERA5', 'IMERG', 'MERRA2']]) \
        .join(pal_df_combined_rain.groupby('time')['rain_rate'] \
        .count() \
        .to_frame('n_min')
        ).reset_index()

        daily_avg_rain['cov_hr'] = daily_avg_rain['n_min'] / 60.0
        daily_avg_rain = daily_avg_rain[daily_avg_rain['cov_hr'] >= 12] 

        daily_avg_rain['rain_rate'] *= 24
        daily_avg_rain['Box'] = bx  # Add box for clarity
        daily_avg_rain['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

        # region_pal_sate_dfs.append(daily_avg_rain)
        bx_pal_sate_dfs.append(daily_avg_rain)

        pal_ds.close()
    
    # Combine all PAL-satellite dataframes in a box into a single dataframe
    box_pal_sate_df = pd.concat(bx_pal_sate_dfs)        
    
    # calculate daily mean per track_PAL_id
    bx_pal_sate_df_daily_mean = box_pal_sate_df.groupby(['track_PAL_id'])[['rain_rate', 'GPCP_v3pt2', 
                                                                                  'GPCP_v3pt3','ERA5']].mean().reset_index()
    box_pal_sate_df_daily_mean['Box'] = bx  # Add box for clarity
    box_PAL_sate_dfs_daily_mean[bx] = box_pal_sate_df_daily_mean
        

    

 



# ---- Box definitions (reuse anywhere) ----
BOXES_180 = {
    "box1": {"lat": ( 5, 15),  "lon": [(160, 175)]},
    "box2": {"lat": (-15, -5), "lon": [(170, 180), (-180, -160)]},  # dateline split
    "box3": {"lat": (-5,  5),  "lon": [(160, 180)]},
    "box4": {"lat": ( 5, 15),  "lon": [(135, 160)]},
}

# ---- Examples ----
# ds = ...  # your Dataset/DataArray with coords 'lat','lon' on [-180,180]

box3_ds_gpcpv3pt2 = subset_box(gpcp_ds_v3pt2_xr, "box3",                     
                     lat="lat", 
                     lon="lon").compute()  # no split
box3_ds_gpcpv3pt3 = subset_box(gpcp_ds_v3pt3_xr, "box3",                     
                     lat="lat", 
                     lon="lon").compute()  # no split

# LOAD Buoy DATA
for b_file in pacific_buoy_files:
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
    print(f"Extracting GPCP v1.3 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
    # Check if the buoy's lat/lon falls within the bounds of the GPCP v1.3 dataset
    if b_lat < box3_ds_gpcpv3pt3['latitude'].min() or b_lat > box3_ds_gpcpv1pt3['latitude'].max() or \
       b_lon < box3_ds_gpcpv3pt3['longitude'].min() or b_lon > box3_ds_gpcpv1pt3['longitude'].max():
        print(f"Warning: Buoy at lat={b_lat:.2f}, lon={b_lon:.2f} is outside GPCP v1.3 data bounds, skipping extraction")
        b_rain_gpcpv1pt3_df = pd.DataFrame(columns=['date', 'GPCP_v1pt3'])
    else:
        # Extract data if within bounds
        b_rain_gpcpv3pt3_df = extract_buoy_gpcp_data_memory_efficient(
            box3_ds_gpcpv1pt3, b_lat, b_lon, version="v1pt3", chunk_size=200
        )
        
        if b_rain_gpcpv1pt3_df is None:
            print("Warning: Failed to extract GPCP v1.3 data, creating empty dataframe")
            b_rain_gpcpv1pt3_df = pd.DataFrame(columns=['date', 'GPCP_v1pt3'])
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -

    # Process GPCP v3.2 - Memory efficient version
    print(f"Extracting GPCP v3.2 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
    b_rain_gpcpv3pt2_df = extract_buoy_gpcp_data_memory_efficient(
        box3_ds_gpcpv3pt2, b_lat, b_lon, version="v3pt2", chunk_size=200
    )
    
    if b_rain_gpcpv3pt2_df is None:
        print("Warning: Failed to extract GPCP v3.2 data, creating empty dataframe")
        b_rain_gpcpv3pt2_df = pd.DataFrame(columns=['date', 'GPCP_v3pt2'])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
    # Process GPCP v3.3 - Memory efficient version
    print(f"Extracting GPCP v3.3 data for buoy at lat={b_lat:.2f}, lon={b_lon:.2f}")
    b_rain_gpcpv3pt3_df = extract_buoy_gpcp_data_memory_efficient(
        gpcp_ds_v3pt3_xr, b_lat, b_lon, version="v3pt3", chunk_size=200
    )

box2_ds = subset_box(gpcp_ds_v1pt3_xr, "box2",                     
                     lat="latitude", 
                     lon="longitude")  # handles split