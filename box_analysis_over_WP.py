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

#%% DEFINE PATH TO DATA
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v1_pnt_3_2010_2020'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2010_2020'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_2010_2020'

path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_daily_1998-2025'

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'

path_to_put_dfs = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs'


#%% DEFINE GLOBAL VARIABLES

cde_run_dte = str(date.today().strftime('%Y%m%d'))

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])
all_gpcp_v3pt3_files_2010_2020 = [
                                  f for f in all_gpcp_v3pt3_files 
                                  if 2010 <= int(os.path.basename(f).split('_')[2][:4]) <= 2020]

all_imerg_files = sorted([os.path.join(path_to_imerg, f) for f in os.listdir(path_to_imerg) if f.endswith('.nc4')])
all_imerg_files_2010_2020 = [
    f for f in all_imerg_files 
    if 2010 <= int(os.path.basename(f).split('.')[4][:4]) <= 2020
]

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

# - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# ---- Define boxes in their original 0–360 form (as in your figure) ----
EPS = 1e-6

# Boxes already in −180…180
BOXES_ = [
    dict(name='Box 1', lat=( 5, 15), lon_segments=[(160, 175)],                  style=dict(ls='--', lw=2, color='k')),
    dict(name='Box 2', lat=(-15, -5), lon_segments=[(170, 180-EPS), (-180+EPS, -160)], style=dict(ls='--', lw=2, color='g')),
    dict(name='Box 3', lat=(-5,  5), lon_segments=[(160, 180-EPS)],              style=dict(ls='--', lw=2, color='b')),
    dict(name='Box 4', lat=( 5, 15), lon_segments=[(135, 160)],                  style=dict(ls='--', lw=2, color='orange')),
]

def draw_boxes(ax):
    for b in BOXES_:
        print(f"Drawing {b['name']}")
        (lat0, lat1) = b['lat']
        for lo, hi in b['lon_segments']:
            ax.add_patch(Rectangle((lo, lat0), hi-lo, lat1-lat0,
                                   transform=ccrs.PlateCarree(), fill=False, **b['style']))

# --- A) World frame (−180…180)
fig = plt.figure(figsize=(10, 2.8), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([130, 210, -20, 20], crs=ccrs.PlateCarree())  # same window as your original plot
ax.coastlines('110m', linewidth=0.7)
ax.add_feature(cfeature.LAND, facecolor='0.9', edgecolor='none')
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
draw_boxes(ax)
plt.title("Boxes in −180…180° (dateline handled)", pad=6)
plt.tight_layout(); plt.show()

# --- B) Pacific-centred (no wrap headaches)
fig = plt.figure(figsize=(10, 2.8), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-50, 30, -20, 20], crs=ccrs.PlateCarree())   # 130–210E in this frame
ax.coastlines('110m', linewidth=0.7)
ax.add_feature(cfeature.LAND, facecolor='0.9', edgecolor='none')
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
draw_boxes(ax)
plt.title("Boxes with central_longitude=180", pad=6)
plt.tight_layout(); plt.show()


# --- imports ---
# === TNWP/TNEP PALs + PACIFIC buoys inside GPCP boxes (0…360°) ===
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

# ------------------- styling -------------------
FONT = "DejaVu Serif"  # use "Times New Roman" if you have it
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT, "Times", "serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
BOX_LABEL_SIZE   = 11
LEGEND_FONTSIZE  = 11
LINEWIDTH_PAL    = 2.6
LINEWIDTH_BOX    = 2.2
STAR_SIZE        = 70

# ------------------- map window -------------------
LON_MIN, LON_MAX = 130, 210
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
boxes = [
    dict(name="Box 1", lat_min=  5, lat_max= 15, lon_min=160, lon_max=175, color="k",          ls="--"),
    dict(name="Box 2", lat_min=-15, lat_max= -5, lon_min=170, lon_max=200, color="tab:green",  ls="--"),
    dict(name="Box 3", lat_min= -5, lat_max=  5, lon_min=160, lon_max=180, color="royalblue",  ls="--"),
    dict(name="Box 4", lat_min=  5, lat_max= 15, lon_min=135, lon_max=160, color="orange",     ls="--"),
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
    ax.text(b["lon_min"]+0.8, b["lat_max"]-1.2, b["name"],
            transform=ccrs.PlateCarree(), fontsize=BOX_LABEL_SIZE, weight="bold",
            color=b["color"], bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=5)

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
    plt.Line2D([0],[0], color="k",         ls="--", lw=LINEWIDTH_BOX, label="Box 1 (5–15°N, 160–175°E)"),
    plt.Line2D([0],[0], color="tab:green", ls="--", lw=LINEWIDTH_BOX, label="Box 2 (5–15°S, 170–200°E)"),
    plt.Line2D([0],[0], color="royalblue", ls="--", lw=LINEWIDTH_BOX, label="Box 3 (5°N–5°S, 160–180°E)"),
    plt.Line2D([0],[0], color="orange",    ls="--", lw=LINEWIDTH_BOX, label="Box 4 (5–15°N, 135–160°E)"),
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
#%%# read all GPCP into a single xr data
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
gpcp_v3pt3_batches = [all_gpcp_v3pt3_files_2010_2020[i:i + batch_size] for i \
                      in range(0, len(all_gpcp_v3pt3_files_2010_2020), batch_size)]
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
# print(f"Processing IMERG files in batches of {batch_size}...")
# imerg_batches = [all_imerg_files_2010_2020[i:i + batch_size] for i \
#                   in range(0, len(all_imerg_files_2010_2020), batch_size)]
imerg_ds_xr_list = []

# Use smaller batch size for IMERG to reduce memory pressure
imerg_batch_size = 15 #min(batch_size, 100)  # Limit IMERG batch size
print(f"Using IMERG batch size: {imerg_batch_size}")

# Re-create batches with smaller size
imerg_batches = [all_imerg_files_2010_2020[i:i + imerg_batch_size] for i \
                 in range(0, len(all_imerg_files_2010_2020), 
                          imerg_batch_size)]

for i, batch in enumerate(imerg_batches):
    if i % 25 == 0:  # More frequent progress updates
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


#%%
# ---- Box definitions (reuse anywhere) ----
BOXES_180 = {
    "box1": {"lat": ( 5, 15),  "lon": [(160, 175)]},
    "box2": {"lat": (-15, -5), "lon": [(170, 180), (-180, -160)]},  # dateline split
    "box3": {"lat": (-5,  5),  "lon": [(160, 180)]},
    "box4": {"lat": ( 5, 15),  "lon": [(135, 160)]},
}

# ---- Examples ----
# ds = ...  # your Dataset/DataArray with coords 'lat','lon' on [-180,180]
box3_ds_gpcpv1pt3 = subset_box(gpcp_ds_v1pt3_xr, "box3",                     
                     lat="latitude", 
                     lon="longitude").compute()  # no split
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
    if b_lat < box3_ds_gpcpv1pt3['latitude'].min() or b_lat > box3_ds_gpcpv1pt3['latitude'].max() or \
       b_lon < box3_ds_gpcpv1pt3['longitude'].min() or b_lon > box3_ds_gpcpv1pt3['longitude'].max():
        print(f"Warning: Buoy at lat={b_lat:.2f}, lon={b_lon:.2f} is outside GPCP v1.3 data bounds, skipping extraction")
        b_rain_gpcpv1pt3_df = pd.DataFrame(columns=['date', 'GPCP_v1pt3'])
    else:
        # Extract data if within bounds
        b_rain_gpcpv1pt3_df = extract_buoy_gpcp_data_memory_efficient(
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