'''
Functions, packages and floating variables used for IEPPO study.
'''
#%%
import warnings
warnings.filterwarnings("ignore")

import gc
import os
from typing import Optional, Tuple, Sequence
from collections import defaultdict
from datetime import date
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.ticker import FixedLocator, FuncFormatter, FormatStrFormatter
from matplotlib.colors import BoundaryNorm
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature


import seaborn as sns
from scipy.stats import linregress

import HydroErr as he

import xarray as xr

from pyproj import CRS
from rasterio.warp import Resampling


from rasterio.enums import Resampling

from multiprocessing import Pool

#%% DEFINE GLOBAL VARIABLES
product_markers = {
    "GPCP v1.3": "o",
    "GPCP v3.2": "s",
    "GPCP v3.3": "^",
    "IMERG v07": "D",
    "ERA5": "P",
    "MERRA-2": "X",
}

# Wider spacing for clarity
product_offsets = {
    "GPCP v1.3": (-12.0,  3.0),
    "GPCP v3.2": ( 0.0,  4.0),
    "GPCP v3.3": ( 10.0,  3.0),
    "IMERG v07": (-6.0, -1.0),
    "ERA5":      ( 0.0, -8.0),
    "MERRA-2":    ( 12.0, -3.0),
}


METRIC_STYLE = {
    # categorical
    "POD": {
        "label": "POD",
        "bounds": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "cmap": plt.cm.viridis
    },
    "FAR": {
        "label": "FAR",
        "bounds": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "cmap": plt.cm.viridis_r
    },
    "HSS": {
        "label": "HSS",
        "bounds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "cmap": plt.cm.viridis
    },
    "Bias_det": {
        "label": "Bias",
        "bounds": [0.0, 0.5, 0.8, 1.0, 1.2, 1.6, 2.2],
        "cmap": plt.cm.RdYlBu_r
    },

    # quantitative
    "CC": {
        "label": "CC",
        "bounds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "cmap": plt.cm.viridis
    },
    "RMSE": {
        "label": "RMSE [mm/day]",
        "bounds": [0, 5, 10, 15,20],
        "cmap": plt.cm.viridis_r
    },
    "MAE": {
        "label": "MAE [mm/day]",
        "bounds": [0, 2, 4, 6, 8, 10],
        "cmap": plt.cm.viridis_r
    },
    "Bias": {
        "label": "Bias [%]",
        "bounds": [-100, -50, -20, 20, 50, 100],
        "cmap": plt.cm.RdBu_r
    },
}

region_label_offsets = {
    "ETNP": (0.0, 15.0),
    "TNEP": (0.0, 15.0),
    "TNWP": (0.0, 15.0),
    "TSEP": (0.0, 15.0),
    "STNA": (0.0, 15.0),
    "TNIO": (0.0, 15.0),
    "ENP":  (0.0, 15.0),
    "WNP":  (0.0, 15.0),
    "IND":  (0.0, 15.0),
    "ATL":  (0.0, 15.0),
}


products = [
    "rain_rate",     # Buoy
    "GPCP v3.2",
    "GPCP v3.3",
    "ERA5",
    "IMERG v07",
    "MERRA-2",
]

Buoy_PRODUCT_COLS = {
    "Buoy": "rain_rate",
    "GPCP v3.2": "GPCP v3.2",
    "GPCP v3.3": "GPCP v3.3",
    "ERA5": "ERA5",
    "IMERG v07": "IMERG v07",
    "MERRA-2": "MERRA-2",
}

cc = CRS.from_authority(code=4326, auth_name='EPSG')

cde_run_dte = str(date.today().strftime('%Y%m%d'))

year_colors = {
    2010: 'black',
    2011: 'magenta',
    2012: 'green',
    2013: 'lime',
    2014: 'orange',
    2015: 'cyan',
    2016: 'blue',
    2017: 'red'
}

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Set plot parameters
# font to Times New Roman and bold for all texts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# DEFINE REGIONS AND THEIR BOUNDARIES (based on Figure 1 and PAL data coverage)
PAL_region_bounds = {
    "ETNP": {"lon_min": -170, "lon_max": -120, "lat_min": 30, "lat_max": 60},  # Extratropical North Pacific 
    "TNEP": {"lon_min": -180, "lon_max": -80, "lat_min": 0, "lat_max": 30}, # Tropical Northeastern Pacific (northern hemisphere only, extended to capture SPURS2 and Caribbean PALs)
    "TNWP": {"lon_min": 120,  "lon_max": 180,  "lat_min": -5, "lat_max": 30},  # Tropical Northwestern Pacific
    "TSEP": {"lon_min": -160, "lon_max": -70,  "lat_min": -25, "lat_max": 0},  # Tropical Southeastern Pacific (southern hemisphere only)
    "STNA": {"lon_min": -70,  "lon_max": -10,  "lat_min": 15, "lat_max": 45},  # Subtropical North Atlantic
    "TNIO": {"lon_min": 60,   "lon_max": 100,  "lat_min": -5, "lat_max": 20},  # Tropical North Indian Ocean
    
}
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Buoy_region_bounds = {
    "ENP": {"lon_min": -180, "lon_max": -60, "lat_min": -30, "lat_max": 15},  # Eastern Pacific
    "WNP": {"lon_min": 120, "lon_max": 180, "lat_min": -15, "lat_max": 15},    # Western Pacific
    "IND": {"lon_min": 40, "lon_max": 110, "lat_min": -15, "lat_max": 30},     # Indian Ocean
    "ATL": {"lon_min": -70, "lon_max": 20, "lat_min": -30, "lat_max": 30},     # Atlantic Ocean
}

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Define global constants
PAL_region_colors = {
    "TNEP": "#3366ff",      # blue
    "TNWP": "#33cc33",      # green
    "TSEP": "#66ccff",      # light blue    
    "ETNP": "#888888",      # gray
    "TNIO": "#ffcc33",      # yellow/orange
    "STNA": "#b35959",      # brown/red
    "Unclassified": "black",
}

product_colors = {
    "GPCP v1.3": "#9467bd",   # purple
    "GPCP v2.3": "#17becf",   # cyan
    "GPCP v3.2": "#4c4c4c",   # dark gray
    "GPCP v3.3": "#1f77b4",   # blue
    "ERA5": "#d62728",       # red
    "IMERG v07": "#2ca02c",  # green
    "MERRA-2": "#ff7f0e",     # orange
    'PAL': "#0820d4",         # deep blue
    'Buoy': "#0820d4",       # deep blue
}

PAL_region_markers = {
    "ETNP": "*",
    "TNEP": "o",
    "TNWP": "^",
    "TSEP": "s",   
    "TNIO": "D",
    "STNA": "P",
}

PAL_REGION_NAMES = {
    "ETNP": "Extratropical \nNorth Pacific",
    "TNEP": "       Tropical \nNortheastern Pacific",
    "TNWP": "       Tropical \nNorthwestern Pacific",
    "TSEP": "       Tropical \nSoutheastern Pacific",    
    "TNIO": "Tropical North \nIndian Ocean",
    "STNA": "Subtropical North \n   Atlantic",
}

Buoy_region_markers = {
    "ENP": "o",
    "WNP": "^",
    "IND": "D",
    "ATL": "P",
}

Buoy_REGION_NAMES = {
    "ENP": "Eastern Pacific",
    "WNP": "Western Pacific",
    "IND": "Indian Ocean",
    "ATL": "Atlantic",
}

Buoy_region_colors = {
    "ENP": "#3366ff",      # blue
    "WNP": "#33cc33",      # green
    "IND": "#ffcc33",      # yellow/orange
    "ATL": "#b35959",      # brown/red
    "Unclassified": "black",
}

#%% DEFINE CUSTOM FUNCTIONS
def p_corr(obs,model):
    return he.pearson_r(model,obs)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# relative bias
def relative_bias(obs, model):
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)

    m = np.isfinite(obs) & np.isfinite(model)
    if m.sum() == 0:
        return np.nan

    mu_obs = np.mean(obs[m])
    if mu_obs == 0:
        return np.nan  # or np.inf / raise, depending on your preference

    mu_residuals = np.mean(model[m] - obs[m])
    return mu_residuals / mu_obs
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# rmse
# Range 0 RMSE < inf, smaller is better.
# Notes: The standard deviation of the residuals. A lower spread indicates that the points are better concentrated
# around the line of best fit (linear). Random errors do not cancel. This metric will highlights larger errors.

def rmsqe(obs,model):
    return he.rmse(model,obs)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def calculate_metrics(df, xcol, ycol):
    df = df.dropna(axis=0, how='any', subset=[xcol, ycol])  # Drop rows with NaN in specified columns
    xdf = df[xcol].values
    ydf = df[ycol].values
    
    
    # if len(xdf) == 0 or len(ydf) == 0:
    #     return np.nan, np.nan, np.nan
    
    # Calculate metrics
    rb = round(relative_bias(xdf, ydf) * 100,1)  # Relative Bias in %
    rmse = round(rmsqe(xdf, ydf), 2)  # Root Mean Square Error
    cc = round(p_corr(xdf, ydf), 2)   # Pearson Correlation Coefficient    
    mae  = round(np.mean(np.abs(ydf - xdf)), 2) # Mean Absolute Error

    return {
        'Bias':rb, 
        'RMSE':rmse, 
        'CC':cc, 
        'MAE':mae
        } # rb, rmse, cc, mae

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def ds_swaplon(ds):
    """
    Swap longitude coordinates from [0, 360] to [-180, 180] and sort.
    Handles both 'lon' and 'longitude' coordinate names.
    """
    var = 'lon' if 'lon' in ds.coords else 'longitude' if 'longitude' in ds.coords else None
    if var is None:
        raise ValueError("No longitude coordinate found in dataset (expected 'lon' or 'longitude').")
    new_lon = (((ds[var] + 180) % 360) - 180)
    ds = ds.assign_coords({var: new_lon})
    ds = ds.sortby(var)
    return ds
# ------------------------------------------------------------
# helpers
def infer_time_var(ds):
    for v in ['time_utc', 'time', 'datetime', 'date_time', 'TIME', 'Time']:
        if v in ds.variables or v in ds.coords:
            return v
    raise KeyError(f"No time variable found. Available: {list(ds.variables)}")

# ------------------------------------------------------------
def wrap_lon(lon):
    """Convert longitude to [-180, 180)."""
    lon = np.asarray(lon)
    return (lon + 180) % 360 - 180

def split_track_on_jumps(lon, lat, max_jump_deg=8):
    """
    Split ship track into segments when there is a large jump
    in lon/lat between consecutive points.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    good = np.isfinite(lon) & np.isfinite(lat)
    lon = lon[good]
    lat = lat[good]

    if len(lon) == 0:
        return []

    segments = []
    start = 0
    for i in range(1, len(lon)):
        dlon = abs(lon[i] - lon[i - 1])
        dlat = abs(lat[i] - lat[i - 1])

        # dateline crossing or unrealistic jump
        if dlon > 100 or dlat > max_jump_deg:
            if i - start > 1:
                segments.append((lon[start:i], lat[start:i]))
            start = i

    if len(lon) - start > 1:
        segments.append((lon[start:], lat[start:]))

    return segments

def split_track_on_jumps_wrapped(lon, lat, max_jump_deg=12):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    good = np.isfinite(lon) & np.isfinite(lat)
    lon = lon[good]
    lat = lat[good]

    if len(lon) == 0:
        return []

    segments = []
    start = 0

    for i in range(1, len(lon)):
        raw_dlon = abs(lon[i] - lon[i - 1])
        dlon = min(raw_dlon, 360 - raw_dlon)   # wrapped longitude jump
        dlat = abs(lat[i] - lat[i - 1])

        if dlat > max_jump_deg or dlon > 25:
            if i - start > 1:
                segments.append((lon[start:i], lat[start:i]))
            start = i

    if len(lon) - start > 1:
        segments.append((lon[start:], lat[start:]))

    return segments
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def simple_box_check(lat_min_file, lat_max_file, lon_min_file, lon_max_file,
                     lat_min_r, lat_max_r, lon_min_r, lon_max_r):
    """
    Alternative simpler check: does the PAL box overlap with region box?
    """
    lat_overlap = not (lat_max_file < lat_min_r or lat_min_file > lat_max_r)
    lon_overlap = not (lon_max_file < lon_min_r or lon_min_file > lon_max_r)
    return lat_overlap and lon_overlap
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# FUNCTION TO CLASSIFY AND GROUP PAL FILES BASED ON BOUNDING BOXES
def classify_and_group_files_bounding_box(file_list, region_bounds_dict=None):
    """
    Classify PAL files into regions based on bounding box overlap.
    Returns a dictionary of {region: list_of_files}
    """
    if region_bounds_dict is None:
        region_bounds_dict = PAL_region_bounds
    
    classification = {region: [] for region in region_bounds_dict}
    classification["Unclassified"] = []

    for file_path in file_list:
        try:
            ds = xr.open_dataset(file_path)
            
            # Try different possible variable names for lat/lon
            lat_vars = ['lat', 'latitude', 'LAT', 'LATITUDE']
            lon_vars = ['lon', 'longitude', 'LON', 'LONGITUDE']
            
            lat = None
            lon = None
            
            for var in lat_vars:
                if var in ds.variables:
                    lat = ds[var].values
                    break
            
            for var in lon_vars:
                if var in ds.variables:
                    lon = ds[var].values
                    break
            
            if lat is None or lon is None:
                print(f"[!] Could not find lat/lon variables in {os.path.basename(file_path)}")
                print(f"    Available variables: {list(ds.variables.keys())}")
                classification["Unclassified"].append(file_path)
                continue

            # Handle fill values
            lat = np.array(lat.filled(np.nan)) if hasattr(lat, "filled") else lat
            lon = np.array(lon.filled(np.nan)) if hasattr(lon, "filled") else lon

            # Normalize longitude to [-180, 180]
            lon = (lon + 360) % 360
            lon[lon > 180] -= 360

            # Get PAL file bounding box
            lat_min_file, lat_max_file = np.nanmin(lat), np.nanmax(lat)
            lon_min_file, lon_max_file = np.nanmin(lon), np.nanmax(lon)

            found = False
            for region, bounds in region_bounds_dict.items():
                lat_min_r = bounds["lat_min"]
                lat_max_r = bounds["lat_max"]
                lon_min_r = bounds["lon_min"]
                lon_max_r = bounds["lon_max"]

                # Use the simpler box check
                overlap = simple_box_check(lat_min_file, lat_max_file,
                                         lon_min_file, lon_max_file,
                                         lat_min_r, lat_max_r,
                                         lon_min_r, lon_max_r)

                if overlap:
                    classification[region].append(file_path)
                    found = True
                    break

            if not found:
                print(f"[!] Not classified: {os.path.basename(file_path)}")
                print(f"    Lat range: {lat_min_file:.2f} to {lat_max_file:.2f}")
                print(f"    Lon range: {lon_min_file:.2f} to {lon_max_file:.2f}")
                classification["Unclassified"].append(file_path)
            
            ds.close()  # Close dataset to free memory

        except Exception as e:
            print(f"[!] Failed to process {os.path.basename(file_path)}: {e}")
            classification["Unclassified"].append(file_path)

    print(f"\nTotal unclassified files: {len(classification['Unclassified'])}")
    return classification

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def region_center_from_bounds(bounds_dict):
    rows = []
    for region, b in bounds_dict.items():
        lon_c = (b["lon_min"] + b["lon_max"]) / 2
        lat_c = (b["lat_min"] + b["lat_max"]) / 2
        rows.append({
            "region": region,
            "lon": lon_c,
            "lat": lat_c,
            "lon_min": b["lon_min"],
            "lon_max": b["lon_max"],
            "lat_min": b["lat_min"],
            "lat_max": b["lat_max"],
        })
    return pd.DataFrame(rows)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def build_obs_df_for_representative_locations(
    pals_classed_by_region,
    buoy_files_by_region,
    pal_stride=25
):
    rows = []

    # ----------------------------------------
    # PAL points
    # ----------------------------------------
    for region, files in pals_classed_by_region.items():
        if region == "Unclassified" or len(files) == 0:
            continue

        for file in files:
            ds = xr.open_dataset(file)

            lat_vals = np.asarray(ds["lat"].values)[::pal_stride]
            lon_vals = wrap_lon(np.asarray(ds["lon"].values)[::pal_stride])

            ds.close()

            n = min(len(lat_vals), len(lon_vals))
            lat_vals = lat_vals[:n]
            lon_vals = lon_vals[:n]

            good = np.isfinite(lat_vals) & np.isfinite(lon_vals)

            for lat, lon in zip(lat_vals[good], lon_vals[good]):
                rows.append({
                    "reference_type": "PAL",
                    "region": region,
                    "lat": float(lat),
                    "lon": float(lon)
                })

    # ----------------------------------------
    # Buoy points
    # map file-group keys to your metric region labels
    # ----------------------------------------
    buoy_region_map = {
        "ENP": "ENP",
        "WNP": "WNP",
        "IND": "IND",
        "ATL": "ATL",
    }

    for raw_region, files in buoy_files_by_region.items():
        region = buoy_region_map.get(raw_region, raw_region)

        for file in files:
            ds = xr.open_dataset(file)

            lat = float(np.asarray(ds["lat"].values).ravel()[0])
            lon = float(wrap_lon(np.asarray(ds["lon"].values).ravel()[0]))

            ds.close()

            if np.isfinite(lat) and np.isfinite(lon):
                rows.append({
                    "reference_type": "Buoy",
                    "region": region,
                    "lat": lat,
                    "lon": lon
                })

    obs_df = pd.DataFrame(rows)
    return obs_df
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def nested_metrics_to_tidy(metrics_dict, reference_type, bounds_dict):
    rows = []
    centers = region_center_from_bounds(bounds_dict)

    for region, prod_dict in metrics_dict.items():
        for product, met_dict in prod_dict.items():
            for metric, value in met_dict.items():
                rows.append({
                    "reference_type": reference_type,
                    "region": region,
                    "product": product,
                    "metric": metric,
                    "value": value
                })

    df = pd.DataFrame(rows)
    df = df.merge(centers, on="region", how="left")
    return df

#=============================================================
# ============================================================
# METRIC COLOR STYLE
# ============================================================

# ============================================================
# METRIC STYLE
# ============================================================
def make_metric_style_dict():
    """
    More discrete color steps, but only a subset of ticks will be labeled later.
    Bounds are chosen to better emphasize observed variability while still using extensions.
    """
    return {
        "POD": {
            # finer internal steps, but colorbar labels can stay sparse
            "bounds": np.arange(0.05, 1.05,0.05), # np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80])
            "cmap": plt.cm.cividis,
            "label": "POD",
            "extend": "both",
            "tick_labels": [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],# [0.1, 0.3, 0.5, 0.7, 0.8]
        },
        "FAR": {
            "bounds": np.arange(0.05, 1.05,0.05), # np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
            "cmap": plt.cm.cividis_r,
            "label": "FAR",
            "extend": "both",
            "tick_labels": [0.15, 0.30, 0.45, 0.60, 0.75, 0.90], # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        },
        "HSS": {
            "bounds": np.arange(0.02, 0.42,0.02), # np.array([0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42])
            "cmap": plt.cm.cividis,
            "label": "HSS",
            "extend": "both",
            "tick_labels": [0.02, 0.08, 0.20, 0.14, 0.20, 0.26, 0.32, 0.38], # [0.10, 0.20, 0.30, 0.40]
        },
        "FreqBias": {
            "bounds": np.arange(0.20, 1.8,0.08), # np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50])
            "cmap": plt.cm.RdBu_r,
            "label": "Frequency bias",
            "extend": "both",
            "tick_labels": [0.20, 0.44, 0.68, 0.92, 1.16, 1.40, 1.64, 1.80],
        },
        "CC": {
            "bounds": np.arange(0.03, 0.60,0.03), # np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
            "cmap": plt.cm.cividis,
            "label": "CC",
            "extend": "both",
            "tick_labels": [0.03, 0.12, 0.21, 0.30, 0.39, 0.48, 0.57], # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        },
        "RMSE": {
            "bounds": np.arange(2.5, 17.5,0.5), # np.array([4, 6, 8, 10, 12, 14, 16, 18, 20])
            "cmap": plt.cm.cividis_r,
            "label": "RMSE [mm/day]",
            "extend": "both",
            "tick_labels": [2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5, 16.0, 17.50], # [4, 8, 12, 16, 20]
        },
        "MAE": {
            "bounds": np.arange(1.5, 12.5,0.5), # np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
            "cmap": plt.cm.cividis_r,
            "label": "MAE [mm/day]",
            "extend": "both",
            "tick_labels": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5], # [2, 4, 6, 8, 10]
        },
        "Bias": {
            "bounds": np.arange(-40, 50, 2), # np.array([-20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50])
            "cmap": plt.cm.RdBu_r,
            "label": "Bias",
            "extend": "both",
            "tick_labels": [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50], # [-20, -10, -5, 0, 5, 10,20,30, 40, 50]
        },
    }
# ============================================================
# REPRESENTATIVE LOCATIONS FROM ACTUAL OBS
# ============================================================

def compute_region_representative_locations(
    obs_df,
    reference_col="reference_type",
    region_col="region",
    lon_col="lon",
    lat_col="lat",
    method="median"
):
    """
    Compute representative lon/lat per (reference_type, region) using
    actual observation coordinates rather than bounding-box centers.

    Notes:
    - Uses a circular mean for longitude to avoid dateline issues.
    - Uses median latitude by default.
    """

    req = [reference_col, region_col, lon_col, lat_col]
    miss = [c for c in req if c not in obs_df.columns]
    if miss:
        raise ValueError(f"obs_df missing required columns: {miss}")

    def circular_mean_deg(lon_deg):
        lon_rad = np.deg2rad(wrap_lon(np.asarray(lon_deg, dtype=float)))
        s = np.nanmean(np.sin(lon_rad))
        c = np.nanmean(np.cos(lon_rad))
        out = np.rad2deg(np.arctan2(s, c))
        return wrap_lon(out)

    rows = []
    for (ref, reg), g in obs_df.groupby([reference_col, region_col]):
        lon_vals = wrap_lon(g[lon_col].astype(float).values)
        lat_vals = g[lat_col].astype(float).values

        if method == "median":
            # longitude median can be awkward near dateline; use circular mean
            rep_lon = circular_mean_deg(lon_vals)
            rep_lat = np.nanmedian(lat_vals)
        elif method == "mean":
            rep_lon = circular_mean_deg(lon_vals)
            rep_lat = np.nanmean(lat_vals)
        else:
            raise ValueError("method must be 'median' or 'mean'")

        rows.append({
            "reference_type": ref,
            "region": reg,
            "lon": rep_lon,
            "lat": rep_lat
        })

    return pd.DataFrame(rows)


def nested_metrics_to_tidy_with_replocs(
    metrics_dict,
    reference_type,
    rep_locs_df
):
    """
    Convert nested metrics dict to tidy dataframe and attach representative
    lon/lat from rep_locs_df.

    rep_locs_df must contain:
      - reference_type
      - region
      - lon
      - lat
    """
    rows = []
    for region, prod_dict in metrics_dict.items():
        for product, met_dict in prod_dict.items():
            for metric, value in met_dict.items():
                rows.append({
                    "reference_type": reference_type,
                    "region": region,
                    "product": product,
                    "metric": metric,
                    "value": value
                })

    df = pd.DataFrame(rows)

    need = {"reference_type", "region", "lon", "lat"}
    if not need.issubset(rep_locs_df.columns):
        raise ValueError(
            f"rep_locs_df must contain {need}, got {set(rep_locs_df.columns)}"
        )

    df = df.merge(
        rep_locs_df[["reference_type", "region", "lon", "lat"]],
        on=["reference_type", "region"],
        how="left"
    )
    return df

def format_lon(x, pos=None):
    x = int(round(x))
    if x == 0:
        return "0°"
    if x < 0:
        return f"{abs(x)}°W"
    return f"{x}°E"

def format_lat(y, pos=None):
    y = int(round(y))
    if y == 0:
        return "EQ"
    if y < 0:
        return f"{abs(y)}°S"
    return f"{y}°N"


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def process_era5_file(file_info):
    idx, file_path = file_info
    if idx % 5 == 0:
        print(f"Processing ERA5 file {idx+1}")
    era5_xr = xr.open_dataset(file_path, engine='netcdf4')
    era5_xr = ds_swaplon(era5_xr)
    # data units are in m per day, convert to mm/day using 1000 factor
    era5_xr['tp'] = era5_xr['tp'] * 1000  # mm/h
    era5_xr['tp'] = era5_xr['tp'] * 24  # mm/day
    # # write crs and resample to gpcp resolution
    era5_xr.rio.write_crs(cc.to_string(), inplace=True)
    era5_xr = era5_xr.rio.reproject(
        era5_xr.rio.crs,
        shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
        resampling=Resampling.average,
    )
    return era5_xr

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def process_merra2_file(file_info):
    idx, file_path = file_info
    try:
        if idx % 2000 == 0:
            print(f"Processing MERRA2 file {idx+1}")
        mer2_xr = xr.open_dataset(file_path, engine='netcdf4')
        # convert units in kg m-2 s-1 to mm/day by a factor of 3600*24
        mer2_xr = mer2_xr['PRECTOT'] * 3600
        mer2_xr = mer2_xr.mean(dim='time')
        mer2_xr = mer2_xr * 24  # convert to mm/day
        # Add a time dimension based on the file name or metadata
        time = pd.to_datetime(os.path.basename(file_path).split('.')[5], format='%Y%m%d')
        mer2_xr = mer2_xr.expand_dims(time=[time])
        # # write crs and resample to gpcp resolution
        mer2_xr.rio.write_crs(cc.to_string(), inplace=True)
        # Set spatial dimensions explicitly
        mer2_xr = mer2_xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        mer2_xr = mer2_xr.rio.reproject(
            mer2_xr.rio.crs,
            shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:],  # (360, 720), set the shape as the GPCP data
            resampling=Resampling.bilinear,
        )
        return mer2_xr
    except Exception as e:
        print(f"Error processing file: {os.path.basename(file_path)}")
        print(f"Error details: {e}")
        return None
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def process_imerg_file(args):
    idx, file_path, version = args
    if idx % 500 == 0:
        print(f"Processing IMERG {version} file {idx+1}")

    imerg_precip_data = xr.open_dataset(file_path,engine='netcdf4')

    if version == 'v06':
        precip_aray = imerg_precip_data.HQprecipitation.data    
        imerg_time = pd.to_datetime(imerg_precip_data.attrs['BeginDate'], format='%Y-%m-%d')
        
    elif version == 'v07':
        precip_aray = imerg_precip_data.precipitation.data    
        imerg_time = pd.to_datetime(imerg_precip_data['time'].values[0],format='%Y-%m-%d') 

    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    precip_aray = precip_aray[np.newaxis, :, :]

    lon = imerg_precip_data.coords['lon'].values
    lat = np.flip(imerg_precip_data.coords['lat']).values    
    imerg_precip_data.close() 

    # Create xarray DataArray
    imerg_xr = xr.DataArray(
        precip_aray,
        coords={
            "time": [imerg_time],
            "lat": lat,
            "lon": lon,
        },
        dims=["time", "lat", "lon"],
        name="precipitation",
    )

    # write crs and resample to gpcp resolution
    imerg_xr.rio.write_crs(cc.to_string(), inplace=True)
    imerg_xr = imerg_xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    imerg_xr = imerg_xr.rio.reproject(
        imerg_xr.rio.crs,
        shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
        resampling=Resampling.average,
    )
    # rename spatial dims back to latlon
    imerg_xr = imerg_xr.rename({'y': 'lat', 'x': 'lon'})
    del(imerg_precip_data,imerg_time, precip_aray,lon,lat)  
    return imerg_xr

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def process_imerg_hdf_file(args):
    idx, file_path = args
    if idx % 500 == 0:
        print(f"Processing IMERG file {idx+1}")

    imerg_precip_data = xr.open_dataset(file_path,engine='netcdf4',group='Grid')

    imerg_precip_data = imerg_precip_data['precipitation']
    precip_aray = imerg_precip_data.data
    imerg_time = pd.Timestamp(imerg_precip_data["time"].values[0].strftime("%Y-%m-%d"))    

    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    precip_aray = precip_aray[np.newaxis, :, :]

    lon = imerg_precip_data.coords['lon'].values
    lat = np.flip(imerg_precip_data.coords['lat']).values    
    imerg_precip_data.close() 

    # Create xarray DataArray
    imerg_xr = xr.DataArray(
        precip_aray,
        coords={
            "time": [imerg_time],
            "lat": lat,
            "lon": lon,
        },
        dims=["time", "lat", "lon"],
        name="precipitation",
    )

    # write crs and resample to gpcp resolution
    imerg_xr.rio.write_crs(cc.to_string(), inplace=True)
    imerg_xr = imerg_xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    imerg_xr = imerg_xr.rio.reproject(
        imerg_xr.rio.crs,
        shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
        resampling=Resampling.average,
    )
    # rename spatial dims back to latlon
    imerg_xr = imerg_xr.rename({'y': 'lat', 'x': 'lon'})
    del(imerg_precip_data,imerg_time, precip_aray,lon,lat)  
    return imerg_xr

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def grab_PAL_rain_and_wind_df(pal_xr_ds):
    """
    Grab rain and wind data from a PAL xarray dataset.
    Parameters:
    - pal_xr_ds: xarray dataset containing PAL data.
    Returns:
    - df_rain: DataFrame containing rain data.
    - df_wind: DataFrame containing wind data.
    """    

    df = pd.DataFrame({
        'time': pd.to_datetime(pal_xr_ds['time'].values),
        'lat': pal_xr_ds['lat'].values,
        'lon': pal_xr_ds['lon'].values,
        'rain_rate': pal_xr_ds['rain_rate'].values,
        'wind_speed': pal_xr_ds['wind_speed'].values,                
    })   

    df['date'] = df['time'].dt.date  # Extract date from time

    df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values

    # Normalize longitude to [-180, 180]
    df['lon'] = (df['lon'] + 360) % 360
    df['lon'][df['lon'] > 180] -= 360

    return  df  
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def process_gpcp_with_PAL_rain_and_wind(df, gpcp_ds_xr, gpcp_version):
    # pal_df_rain, pal_df_wind

    # DO PAL RAIN GPCP MATCHING    

    pal_dates_rain = pd.to_datetime(df['date'])
    pal_lats_rain = df['lat'].values
    pal_lons_rain = df['lon'].values

    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'latitude' in gpcp_ds_xr.dims or 'longitude' in gpcp_ds_xr.dims:
        gpcp_ds_xr = gpcp_ds_xr.rename({'latitude': 'lat', 'longitude': 'lon'})

    gpcp_precip = gpcp_ds_xr['precip'].interp(
        time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
        lon=("points", pal_lons_rain), method="nearest"
    )
    # Set places where the values are less than 0 to NaN
    gpcp_precip = gpcp_precip.where(gpcp_precip >= 0, np.nan)

    # Store matched values in the DataFrame
    df[gpcp_version] = gpcp_precip

    # do same for probability of liquid phase if it exsists in dataset
    if gpcp_version == 'GPCP v3.2':
        gpcp_plp = gpcp_ds_xr['probability_liquid_phase'].interp(
            time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
            lon=("points", pal_lons_rain), method="nearest")
        
        # Set places where the values are less than 0 to NaN
        gpcp_plp = gpcp_plp.where(gpcp_plp >= 0, np.nan)
        
        # Store matched values in the DataFrame
        df[f'PLP_{gpcp_version}'] = gpcp_plp   

    gc.collect()

    return df#pal_df_rain, pal_df_wind

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def process_era5_with_PAL_rain_and_wind_v1(df, era5_ds):    
    # DO PAL RAIN ERA5 MATCHING 
    pal_dates_rain = pd.to_datetime(df['date'])
    pal_lats_rain = df['lat'].values
    pal_lons_rain = df['lon'].values

    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'y' in era5_ds.dims or 'x' in era5_ds.dims:
        era5_ds = era5_ds.rename({'y': 'lat', 'x': 'lon'})

    era5_tp = era5_ds['tp'].interp(
        valid_time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
        lon=("points", pal_lons_rain), method="nearest"
    )
    # Set places where the values are less than 0 to NaN
    era5_tp = era5_tp.where(era5_tp >= 0, np.nan)

    # Store matched values in the DataFrame
    df['ERA5'] = era5_tp    

    gc.collect()

    return df
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def process_imerg_with_PAL_rain_and_wind_v1(df, imerg_ds, imerg_version):    
    # DO PAL RAIN IMERG MATCHING 
    pal_dates_rain = pd.to_datetime(df['date'])
    pal_lats_rain = df['lat'].values
    pal_lons_rain = df['lon'].values

    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'y' in imerg_ds.dims or 'x' in imerg_ds.dims:
        imerg_ds = imerg_ds.rename({'y': 'lat', 'x': 'lon'})

    imerg_pr = imerg_ds.interp(
        time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
        lon=("points", pal_lons_rain), method="nearest"
    )
    # Set places where the values are less than 0 to NaN
    imerg_pr = imerg_pr.where(imerg_pr >= 0, np.nan)

    # Store matched values in the DataFrame
    df[imerg_version] = imerg_pr    

    gc.collect()

    return df

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def process_merra2_with_PAL_rain_and_wind_v1(df, merra2_ds,):    
    # DO PAL RAIN ERA5 MATCHING 
    pal_dates_rain = pd.to_datetime(df['date'])
    pal_lats_rain = df['lat'].values
    pal_lons_rain = df['lon'].values

    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'y' in merra2_ds.dims or 'x' in merra2_ds.dims:
        merra2_ds = merra2_ds.rename({'y': 'lat', 'x': 'lon'})

    merra2_pr = merra2_ds.interp(
        time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
        lon=("points", pal_lons_rain), method="nearest"
    )
    # Set places where the values are less than 0 to NaN
    merra2_pr = merra2_pr.where(merra2_pr >= 0, np.nan)

    # Store matched values in the DataFrame
    df['MERRA2'] = merra2_pr    

    gc.collect()   

    return df

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def grab_Buoy_data_df(buoy_xr_ds):
    """
    Grab Buoy data from a given Buoy NetCDF file and process it into a Pandas DataFrame.

    Parameters
    ----------
    buoy_xr_ds : xr.Dataset
        The Buoy NetCDF file to process.

    Returns
    -------
    b_df : pd.DataFrame
        The processed Buoy data in a Pandas DataFrame format.
    b_lat : float
        The latitude of the buoy.
    b_lon : float
        The longitude of the buoy.
    """
    
    b_ds = xr.open_dataset(buoy_xr_ds)
    b_lat = b_ds['lat'].values[0]
    b_lon = b_ds['lon'].values[0]
    b_lon = (b_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

    b_df = b_ds[['time', 'RN_485', 'QRN_5485']].to_dataframe().reset_index()
    b_df.rename(columns={'RN_485': 'rain_rate', 'QRN_5485': 'quality_flag'}, inplace=True)
    b_df['date'] = pd.to_datetime(b_df['time']).dt.date  # Extract date from time
    # Filter out rows with negative rain_rate
    b_df = b_df[b_df['rain_rate'] >= 0]
    del(b_ds)
    return b_df, b_lat, b_lon
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def _standardize_latlon_names(da):
    """
    Rename common latitude/longitude variants to lat/lon.
    Handles case-insensitive matches and avoids double-renaming.
    """
    rename_map = {}

    for dim in da.dims:
        d = dim.lower()
        if d in ("lat", "latitude", "y") and dim != "lat":
            rename_map[dim] = "lat"
        elif d in ("lon", "longitude", "x") and dim != "lon":
            rename_map[dim] = "lon"

    if rename_map:
        da = da.rename(rename_map)

    return da
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def extract_timeseries_memory_efficient(xr_dataset, lat, lon, var_name=None, chunk_size=100):
    """
    Memory-efficient extraction of time series data from large xarray datasets.
    Processes data in chunks to avoid loading entire dataset into memory.
    
    Parameters:
    - xr_dataset: xarray DataArray or Dataset
    - lat: latitude of the point
    - lon: longitude of the point  
    - var_name: variable name if working with Dataset (None for DataArray)
    - chunk_size: number of time steps to process at once
    
    Returns:
    - pandas DataFrame with time series data
    """
    import gc

    # rename xr_dataset lat/lon if needed
    # coord_lst = list(xr_dataset.coords)
    # if 'latitude' in coord_lst:
    #     xr_dataset = xr_dataset.rename({'latitude': 'lat'})
    
    # if 'longitude' in coord_lst:
    #     xr_dataset = xr_dataset.rename({'longitude': 'lon'})
    xr_dataset = _standardize_latlon_names(xr_dataset)

    # Get time dimension
    if 'valid_time' in xr_dataset.dims:        
        xr_dataset = xr_dataset.rename({'valid_time': 'time'})
    
    # Select the nearest lat/lon point (this doesn't load data yet)
    if var_name:
        selected_data = xr_dataset[var_name].sel(lat=lat, lon=lon, method='nearest')
    else:
        selected_data = xr_dataset.sel(lat=lat, lon=lon, method='nearest')  
        
    times = selected_data.time.values
    total_times = len(times)
    
    print(f"Extracting time series for lat={lat:.2f}, lon={lon:.2f}")
    print(f"Total time steps: {total_times}, processing in chunks of {chunk_size}")
    
    # Initialize list to store chunks
    data_chunks = []
    
    # Process in chunks
    for i in range(0, total_times, chunk_size):
        end_idx = min(i + chunk_size, total_times)
        
        if i % (chunk_size * 10) == 0:  # Progress every 10 chunks
            print(f"  Processing chunk {i//chunk_size + 1}/{(total_times-1)//chunk_size + 1}")
        
        try:
            # Select time slice and compute (this loads only the chunk)
            chunk_data = selected_data.isel(time=slice(i, end_idx)).compute()
            
            # Ensure the DataArray has a name for DataFrame conversion
            if chunk_data.name is None:
                if var_name:
                    chunk_data.name = var_name
                else:
                    # For IMERG data, assign a default name
                    chunk_data.name = 'precipitation'
            
            # Convert to dataframe
            chunk_df = chunk_data.to_dataframe().reset_index()
            data_chunks.append(chunk_df)
            
            # Clean up
            del chunk_data
            gc.collect()
            
        except Exception as e:
            print(f"Error processing chunk {i//chunk_size + 1}: {e}")
            continue
    
    if not data_chunks:
        print("Warning: No data chunks were successfully processed")
        return None
    
    # Combine all chunks
    print("Combining chunks...")
    result_df = pd.concat(data_chunks, ignore_index=True)
    
    # Clean up
    del data_chunks
    gc.collect()
    
    print(f"Extraction complete: {len(result_df)} records")
    return result_df

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def extract_buoy_satellite_data_memory_efficient(ds_xr, 
                                                 b_lat, b_lon, 
                                                 product, var_name,
                                                 chunk_size=100):
    """
    Memory-efficient extraction of satellite data at buoy locations.
    
    Parameters:
    - product: satellite product xarray Dataset
    - b_lat: buoy latitude
    - b_lon: buoy longitude    
    - chunk_size: number of time steps to process at once
    
    Returns:
    - pandas DataFrame with satellite product time series
    """
    try:
        # Extract precipitation data using memory-efficient method
        precip_df = extract_timeseries_memory_efficient(
            ds_xr, b_lat, b_lon, var_name=var_name, chunk_size=chunk_size
        )
        
        if precip_df is None:
            return None
        
        # For GPCP v3.2 and v3.3, also extract probability_liquid_phase
        if product  == 'GPCP v3.2':
            try:
                print(f"  Also extracting probability_liquid_phase for {product}")
                plp_df = extract_timeseries_memory_efficient(
                    ds_xr, b_lat, b_lon, var_name='probability_liquid_phase', chunk_size=chunk_size
                )
                
                if plp_df is not None:
                    # Merge probability_liquid_phase data with precipitation data
                    precip_df = precip_df.merge(
                        plp_df[['time', 'probability_liquid_phase']], 
                        on='time', how='left'
                    )
                    print(f"  ✅ Successfully merged probability_liquid_phase data")
                else:
                    print(f"  ⚠️ Warning: Failed to extract probability_liquid_phase for {product}")
                    
            except Exception as e:
                print(f"  ⚠️ Warning: Could not extract probability_liquid_phase for {product}: {e}")
        
        # Add date column and rename columns
        precip_df['date'] = pd.to_datetime(precip_df['time']).dt.date
        precip_df.rename(columns={'precip': product}, inplace=True)
        
        # Rename probability_liquid_phase if it exists
        if 'probability_liquid_phase' in precip_df.columns:
            precip_df.rename(
                columns={'probability_liquid_phase': f'PLP_{product}'}, 
                inplace=True
            )
        
        return precip_df
        
    except Exception as e:
        print(f"Error extracting satellite {product} data: {e}")
        return None
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _infer_var_columns(ds, varnames):
    """
    Return list of variable column names we expect in the output df.
    Works for xr.Dataset and xr.DataArray.
    """
    if varnames is None:
        if isinstance(ds, xr.Dataset):
            return list(ds.data_vars)
        elif isinstance(ds, xr.DataArray):
            # DataArray has a single "variable": its name (or fallback)
            return [ds.name or "value"]
        else:
            return ["value"]
    else:
        return list(varnames)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def extract_point_timeseries_to_df(
    ds,
    lat,
    lon,
    t0,
    t1,
    varnames="precip",          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
    method="nearest",
    time_name="time",
    lat_name="lat",
    lon_name="lon",
):
    # ---- normalize varnames ----
    if varnames is None:
        varnames = None
    elif isinstance(varnames, str):
        varnames = [varnames]
    else:
        varnames = list(varnames)

    # ---- time bounds: clip to dataset availability ----
    t0 = pd.to_datetime(t0)
    t1 = pd.to_datetime(t1)

    ds_tmin = pd.to_datetime(ds[time_name].values.min())
    ds_tmax = pd.to_datetime(ds[time_name].values.max())

    t0_clip = max(t0, ds_tmin)
    t1_clip = min(t1, ds_tmax)

    if t0_clip > t1_clip:        
        vcols = _infer_var_columns(ds, varnames)
        cols = [time_name] + vcols + [lat_name, lon_name, "lat_sel", "lon_sel"]
        return pd.DataFrame(columns=cols)
        # raise ValueError(f"Requested time window [{t0},{t1}] is outside dataset [{ds_tmin},{ds_tmax}].")

    # ---- lon handling: if dataset is 0..360 but you provide -180..180 ----
    lon_vals = ds[lon_name].values
    if np.nanmin(lon_vals) >= 0 and lon < 0:
        lon = lon % 360

    # ---- select vars + subset ----
    if varnames is not None:
        ds_sub = ds[varnames].sel(
                    {lat_name: float(lat), lon_name: float(lon)},
                    method=method
                ).sel({time_name: slice(t0_clip, t1_clip)})
    else:
        ds_sub = ds.sel(
                    {lat_name: float(lat), lon_name: float(lon)},
                    method=method
                ).sel({time_name: slice(t0_clip, t1_clip)})

    # ---- to DataFrame ----
    df = ds_sub.to_dataframe().reset_index()

    # optional: record the actual gridpoint selected
    df["lat_sel"] = float(ds_sub[lat_name].values)
    df["lon_sel"] = float(ds_sub[lon_name].values)

    return df
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_metrics_by_intensity_for_df(
    df,
    products,
    rainfall_bins,
    obs_col="rain_rate",
):
    """
    Compute categorical and quantitative metrics by rain-rate threshold
    for one collocated dataframe.

    Returns
    -------
    cat_met_by_prdt : dict
        cat_met_by_prdt[product] -> DataFrame(index=rainfall_bins, columns=cat_metrics)
    qt_met_by_prdt : dict
        qt_met_by_prdt[product]  -> DataFrame(index=rainfall_bins, columns=qt_metrics)
    """
    cat_metrics = ['POD', 'FAR', 'Bias', 'HSS']
    qt_metrics  = ['CC', 'RMSE', 'MAE', 'RB']

    cat_met_by_prdt = {}
    qt_met_by_prdt  = {}

    for product in products:
        cat_met_prdt = pd.DataFrame(index=rainfall_bins, columns=cat_metrics, dtype=float)
        qt_met_prdt  = pd.DataFrame(index=rainfall_bins, columns=qt_metrics,  dtype=float)

        forecast = df[product]
        observed = df[obs_col]

        for r_bin in rainfall_bins:
            # categorical: all valid pairs at this threshold
            cat_mets = categorical_stats(forecast, observed, r_bin)
            cat_met_prdt.loc[r_bin, 'POD']  = cat_mets.get('POD',  np.nan)
            cat_met_prdt.loc[r_bin, 'FAR']  = cat_mets.get('FAR',  np.nan)
            cat_met_prdt.loc[r_bin, 'Bias'] = cat_mets.get('Bias', np.nan)
            cat_met_prdt.loc[r_bin, 'HSS']  = cat_mets.get('HSS',  np.nan)

            # quantitative: only intensity-matched rainy cases
            bin_df = df[(df[obs_col] >= r_bin) & (df[product] >= r_bin)].copy()
            qt_mets = calculate_metrics(bin_df, obs_col, product)

            qt_met_prdt.loc[r_bin, 'CC']   = qt_mets.get('CC',   np.nan)
            qt_met_prdt.loc[r_bin, 'RMSE'] = qt_mets.get('RMSE', np.nan)
            qt_met_prdt.loc[r_bin, 'MAE']  = qt_mets.get('MAE',  np.nan)
            qt_met_prdt.loc[r_bin, 'RB']   = qt_mets.get('Bias', np.nan)  # relative bias [%]

        cat_met_by_prdt[product] = cat_met_prdt
        qt_met_by_prdt[product]  = qt_met_prdt

    return cat_met_by_prdt, qt_met_by_prdt
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def categorical_stats(forecast, observation, threshold):
    """
    Compute categorical verification statistics following WMO/JWGNE definitions.
    """

    forecast = np.asarray(forecast)
    observation = np.asarray(observation)

    # Binary events
    f_event = forecast >= threshold
    o_event = observation >= threshold

    # Contingency table
    H = np.sum(f_event & o_event)        # Hits
    M = np.sum(~f_event & o_event)       # Misses
    F = np.sum(f_event & ~o_event)       # False alarms
    C = np.sum(~f_event & ~o_event)      # Correct negatives

    # Basic metrics
    POD = H / (H + M) if (H + M) > 0 else np.nan
    FAR = F / (H + F) if (H + F) > 0 else np.nan
    CSI = H / (H + M + F) if (H + M + F) > 0 else np.nan
    Bias = (H + F) / (H + M) if (H + M) > 0 else np.nan
    POFD = F / (F + C) if (F + C) > 0 else np.nan
    Accuracy = (H + C) / (H + M + F + C) if (H + M + F + C) > 0 else np.nan

    # Heidke Skill Score (WMO)
    denom = (H + M) * (M + C) + (H + F) * (F + C)
    HSS = (2 * (H * C - M * F) / denom) if denom > 0 else np.nan

    return {
        "Hits": H,
        "Misses": M,
        "False_Alarms": F,
        "Correct_Negatives": C,
        "POD": POD,
        "FAR": FAR,
        "CSI": CSI,
        "Bias": Bias,
        "POFD": POFD,
        "Accuracy": Accuracy,
        "HSS": HSS
    }

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def continous_cat_metrics_array_based(obs, sim,target_val):
    '''
    continous_cat_metrics_array_based is is an array based implementation of continous_cat_metrics, which is df based

    it requires 
    obs = observation (x variable)
    sim = simulated (y variable)   

    target value  = the value/class label in the observed to be evaluated 

    returns a dataframe of catagorical metrics Hits, Miss, False alarms, POD, FAR, POFD,
    ACC, CSI, HSS, ETS

    These computations were aquird from Wilks, D.S. Statistical Methods in the Atmospheric Sciences; Elsevier: Amsterdam, The Netherlands, 2011; p. 627.
    '''

    if target_val == 0:
        a_hit = np.count_nonzero((obs > target_val) & (sim > target_val))

        b_false = np.count_nonzero((obs == target_val) & (sim > target_val))

        c_miss = np.count_nonzero((obs > target_val) & (sim == target_val))  

        d_cor_neg = np.count_nonzero((obs == target_val) & (sim == target_val))

    elif target_val > 0:

        a_hit = np.count_nonzero((obs > target_val) & (sim > target_val))

        b_false = np.count_nonzero((obs < target_val) & (sim > target_val))

        c_miss = np.count_nonzero((obs > target_val) & (sim < target_val))  

        d_cor_neg = np.count_nonzero((obs < target_val) & (sim < target_val))

    n = a_hit + b_false + c_miss + d_cor_neg # total number of calssifications

    a_hit_rand_hss = ((a_hit + c_miss)*(a_hit + b_false) + (d_cor_neg + c_miss)*(d_cor_neg + b_false))/n # random hits for HSS

    #a_hit_rand_ets = (a_hit + c_miss)*(a_hit + b_false)/n # random hits for ETSS   
    # a_ref = (a_hit + b_false)*(a_hit + c_miss) / n    

    # Calculate metrics with safe handling for division by zero
    pod = round(a_hit / (a_hit + c_miss), 2) if (a_hit + c_miss) > 0 else np.nan
    far = round(b_false / (a_hit + b_false), 2) if (a_hit + b_false) > 0 else np.nan
    bias = round((a_hit + b_false) / (a_hit + c_miss), 2) if (a_hit + c_miss) > 0 else np.nan
    csi = round(a_hit / (a_hit + b_false + c_miss), 2) if (a_hit + b_false + c_miss) > 0 else np.nan


    # pod = round(a_hit/(a_hit + c_miss),2)
    # the hit rate is the ratio of correct forecasts to the number of times this event occurred
    # range  between 0 (worse) to 1 (perfect)

    # far = round(b_false/(a_hit + b_false),2)
    # That is, FAR is the fraction of yes forecasts that turn out to be wrong, or that proportion of the forecast events that fail to materialize
    # range  between 0 (perfect) to 1 (worse case)

    pofd = round(b_false/(b_false + d_cor_neg),2)
    # ratio of false alarms to the total number of nonoccurrences of the event

    # bias = round((a_hit + b_false)/(a_hit + c_miss),2)
    # bias is simply the ratio of the number of yes forecasts to the number of yes observations.
    # bias = 1 means Unbiased forecasts,indicating that the event was forecast the same number of times that it was observed
    # bias > 1 = indicates that the event was forecast more often than observed, which is called overforecasting
    #  bias < 1 = one indicates that the event was forecast less often than observed, or was underforecast

    acc = round((a_hit + d_cor_neg)/ n,2)
    #This is simply the fraction of the n forecast occasions for which the nonprobabilistic forecast
    # correctly anticipated the subsequent event or non event

    # csi = round(a_hit/(a_hit + b_false + c_miss),2)
    # csi Range: 0 to 1, 0 indicates no skill. Perfect score: 1. It can be viewed as a proportion
    # correct for the quantity being forecast, after removing correct no forecasts from consideration.

    hss = round((2*((a_hit*d_cor_neg) - (b_false*c_miss)))/(((a_hit+c_miss)*(c_miss+d_cor_neg))+((a_hit+b_false)*(b_false+d_cor_neg))),2)
    # hss Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
    #round(((a_hit + d_cor_false)-(a_hit_rand_hss))/(N - a_hit_rand_hss),2)

    # ets = round((a_hit - a_ref)/(a_hit - a_ref + b_false + c_miss),2) 
    # ets Range: -1/3 to 1, 0 indicates no skill. Perfect score: 1.

    # Hanssen and Kuipers discriminant (true skill statistic, Peirce's skill score)
    #hk = round((a_hit/a_hit + b_miss) - (c_false/d_cor_false + c_false),3)

    return {
        "Hits": a_hit,
        "Misses": c_miss,
        "False_Alarms": b_false,
        "Correct_Negatives": d_cor_neg,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "Bias": bias,
        "POFD": pofd,
        "Accuracy": acc,
        "HSS": hss
    }

    # return pod, far, bias, csi,  (a_hit, b_false, c_miss, d_cor_neg,n) #  pofd,hss, ets, acc, 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_pdf_elements(data, colname, bins):
    pdfc = []  # PDF by occurrence
    pdfv = []  # PDF by volume
    bin_labels = []  # Bin labels for the DataFrame

    total_count = len(data)
    total_volume = 0

    # Loop through bins to compute PDFc and PDFv
    for i, bn in enumerate(bins):
        if i == 0:
            bin_data = data[data[colname] <= bn]
        else:
            bin_data = data[(data[colname] > bins[i - 1]) & (data[colname] <= bn)]

        # PDFc: Percentage of occurrences in the bin
        bin_count = len(bin_data)
        pdfc.append((bin_count / total_count) * 100)

        # PDFv: Percentage of volume in the bin
        if bin_count > 0:
            bin_mean = bin_data[colname].mean()
            bin_volume = bin_count * bin_mean
        else:
            bin_volume = 0

        total_volume += bin_volume
        pdfv.append(bin_volume)

        # Add bin label
        if i == 0:
            bin_labels.append(f"<= {bn}")
        else:
            bin_labels.append(f"{bins[i - 1]} - {bn}")

    # Normalize PDFv to percentages
    pdfv = [(volume / total_volume) * 100 for volume in pdfv]

    # Create a DataFrame with bin, pdfc, and pdfv
    pdf_df = pd.DataFrame({
        "bin": bins,
        "pdfc": pdfc,
        "pdfv": pdfv
    })

    return pdf_df

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_pdf_bundle_for_insitu_df(
    df,
    *,
    obs_col="rain_rate",
    products=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    bin_values=(0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    year_range=None,
    date_col="date",
):
    """
    Compute PDFc/PDFv tables for one in situ-based collocated dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input collocated dataframe containing obs_col + product columns.
    obs_col : str
        In situ rainfall column, usually 'rain_rate'.
    products : tuple/list
        Product columns to process.
    bin_values : tuple/list
        Bin upper edges passed to compute_pdf_elements().
    year_range : tuple or None
        Optional year filter, e.g. (2000, 2020).
    date_col : str
        Date column used only if year_range is provided.

    Returns
    -------
    pdf_dict : dict
        pdf_dict['insitu'] = PDF table for obs_col
        pdf_dict[product]  = PDF table for each product
    """
    dff = df.copy()

    if year_range is not None:
        y0, y1 = year_range
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff["year"] = dff[date_col].dt.year.astype(int)
        dff = dff[(dff["year"] >= y0) & (dff["year"] <= y1)].copy()

    pdf_dict = {}
    pdf_dict["insitu"] = compute_pdf_elements(dff, obs_col, list(bin_values))

    for product in products:
        pdf_dict[product] = compute_pdf_elements(dff, product, list(bin_values))

    return pdf_dict
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def make_monthly_clim_anoms(monthly_clim_by_region, buoy_col="rain_rate", products=None):
    """
    For each region: add anomaly columns prod_anom = prod - buoy_col
    Returns dict(region -> dataframe with anomaly columns)
    """
    anom_by_region = {}

    for region, clim in monthly_clim_by_region.items():
        c = clim.copy()

        for prod in products:
            if prod == buoy_col:
                continue
            c[f"{prod}_anom"] = c[prod] - c[buoy_col]

        anom_by_region[region] = c

    return anom_by_region

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_monthly_climatology_equal_station_weight(
    df,
    products,
    id_col="ID",
    region_col="region",
    date_col="date",
):
    """
    Compute monthly climatology in 2 steps:
      (1) Station-month means: mean over time for each (region, ID, month)
      (2) Region-month means: mean over stations for each (region, month)

    Returns:
      monthly_clim_by_region: dict[region] -> DataFrame with columns ["month"] + products
      station_month: DataFrame of station-month climatology (useful for diagnostics)
    """
    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col])
    dfx["month"] = dfx[date_col].dt.month

    # --- Step 1: station-month climatology (equal-weight within each station) ---
    station_month = (
        dfx
        .groupby([region_col, id_col, "month"], as_index=False)[products]
        .mean()
    )

    # --- Step 2: region-month climatology (equal-weight across stations) ---
    region_month = (
        station_month
        .groupby([region_col, "month"], as_index=False)[products]
        .mean()
    )

    monthly_clim_by_region = {
        reg: sub.sort_values("month").reset_index(drop=True)
        for reg, sub in region_month.groupby(region_col)
    }

    return monthly_clim_by_region, station_month

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def rename_monthnum_for_plotting(monthly_clim_by_region):
    out = {}
    for region, clim in monthly_clim_by_region.items():
        c = clim.copy()
        if "month_num" in c.columns:
            c = c.rename(columns={"month_num": "month"})
        out[region] = c
    return out
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_monthly_climatology_from_monthly_buoy_df(
    df,
    products,
    *,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="Buoy",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=None,
    equal_weight_by_buoy=True,
):
    """
    Compute regional monthly climatology from the monthly buoy-product dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Master monthly buoy-product dataframe.
    products : list
        Product columns including buoy reference, e.g.
        ["Buoy", "GPCP v2.3", "GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"]
    min_days_per_month : int
        Minimum buoy daily count required for a monthly buoy value to be used.
    min_buoys_per_month : int or None
        If given, require at least this many active buoys in a region-month before using that row.
    equal_weight_by_buoy : bool
        If True, first compute each buoy's monthly climatology, then average across buoys.
        If False, average all rows directly.

    Returns
    -------
    monthly_clim_by_region : dict
        region -> DataFrame with columns ["month_num"] + products
    """
    dff = df.copy()
    dff[month_col] = pd.to_datetime(dff[month_col])

    # keep only sufficiently sampled buoy-months
    if n_days_col in dff.columns:
        dff = dff[dff[n_days_col] >= min_days_per_month].copy()

    # add calendar month
    dff["month_num"] = dff[month_col].dt.month

    # optional: require enough active buoys per region-month
    if min_buoys_per_month is not None:
        active_counts = (
            dff.groupby([region_col, month_col], as_index=False)
               .agg(n_active_buoys=(id_col, "nunique"))
        )

        dff = dff.merge(active_counts, on=[region_col, month_col], how="left")
        dff = dff[dff["n_active_buoys"] >= min_buoys_per_month].copy()

    monthly_clim_by_region = {}

    for region in dff[region_col].dropna().unique():
        dfr = dff[dff[region_col] == region].copy()

        if equal_weight_by_buoy:
            # Step 1: monthly climatology per buoy
            buoy_month_clim = (
                dfr.groupby([id_col, "month_num"], as_index=False)[products]
                   .mean()
            )

            # Step 2: average climatology across buoys
            clim = (
                buoy_month_clim.groupby("month_num", as_index=False)[products]
                              .mean()
            )
        else:
            # direct row-wise average
            clim = (
                dfr.groupby("month_num", as_index=False)[products]
                   .mean()
            )

        monthly_clim_by_region[region] = clim.sort_values("month_num").reset_index(drop=True)

    return monthly_clim_by_region
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ============================================================
# BUILD ANNUAL REGIONAL SERIES FROM MONTHLY BUOY-PRODUCT TABLE
# ============================================================
def build_annual_from_monthly_buoy_df(
    df,
    products,
    *,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="Buoy",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=5,
    min_months_per_year=8,
    equal_weight_by_buoy=True,
):
    """
    Build annual regional mean rainfall series from the monthly buoy-product table.

    Workflow
    --------
    1) keep only buoy-months with sufficient daily coverage
    2) optionally require enough active buoys in each region-month
    3) compute region-month means (equal-weight by buoy recommended)
    4) compute annual means from valid region-month means

    Returns
    -------
    annual_by_region : dict
        region -> DataFrame with columns:
        ['year', 'n_valid_months'] + products

    annual_df : pd.DataFrame
        concatenated regional annual dataframe
    """

    dff = df.copy()
    dff[month_col] = pd.to_datetime(dff[month_col])

    # --------------------------------------------------------
    # 1) keep only sufficiently sampled buoy-months
    # --------------------------------------------------------
    if n_days_col in dff.columns:
        dff = dff[dff[n_days_col] >= min_days_per_month].copy()

    if dff.empty:
        return {}, pd.DataFrame()

    # --------------------------------------------------------
    # 2) require enough active buoys in each region-month
    # --------------------------------------------------------
    if min_buoys_per_month is not None:
        active_counts = (
            dff.groupby([region_col, month_col], as_index=False)
               .agg(n_active_buoys=(id_col, "nunique"))
        )

        dff = dff.merge(active_counts, on=[region_col, month_col], how="left")
        dff = dff[dff["n_active_buoys"] >= min_buoys_per_month].copy()

    if dff.empty:
        return {}, pd.DataFrame()

    # --------------------------------------------------------
    # 3) compute region-month series
    # --------------------------------------------------------
    if equal_weight_by_buoy:
        # one value per buoy-month already exists; now average equally across buoys
        monthly_region = (
            dff.groupby([region_col, month_col], as_index=False)[products]
               .mean()
        )
    else:
        monthly_region = (
            dff.groupby([region_col, month_col], as_index=False)[products]
               .mean()
        )

    monthly_region["year"] = monthly_region[month_col].dt.year
    monthly_region["month_num"] = monthly_region[month_col].dt.month

    # --------------------------------------------------------
    # 4) annual means from monthly regional values
    # --------------------------------------------------------
    annual_rows = []

    for region, dfr in monthly_region.groupby(region_col):
        for year, dfy in dfr.groupby("year"):
            row = {
                region_col: region,
                "year": int(year),
                "n_valid_months": int(dfy[buoy_col].notna().sum())
            }

            if row["n_valid_months"] >= min_months_per_year:
                for p in products:
                    row[p] = dfy[p].mean()
            else:
                for p in products:
                    row[p] = np.nan

            annual_rows.append(row)

    annual_df = pd.DataFrame(annual_rows).sort_values([region_col, "year"]).reset_index(drop=True)

    annual_by_region = {
        reg: sub.drop(columns=[region_col]).reset_index(drop=True)
        for reg, sub in annual_df.groupby(region_col)
    }

    return annual_by_region, annual_df

#-------------------------------------------------------------------
def build_annual_from_monthly_buoy_df_with_sample_counts(

    df,
    products,
    *,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="Buoy",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=5,
    min_months_per_year=8,
    equal_weight_by_buoy=True,
    year_min=None,
    year_max=None,
    region_year_limits=None,

):

    """

    Build annual regional mean rainfall series from monthly buoy-product table,

    while also storing sample-support diagnostics.

    Returns

    -------

    annual_by_region : dict

        region -> DataFrame

    annual_df : pd.DataFrame

        concatenated annual dataframe with:

        [region, year, n_valid_months, n_active_buoys_year, n_valid_buoy_months] + products

    """

    dff = df.copy()

    dff[month_col] = pd.to_datetime(dff[month_col])

    dff["year"] = dff[month_col].dt.year

    dff["month_num"] = dff[month_col].dt.month

    # --------------------------------------------------------

    # 0) optional year filtering first

    # --------------------------------------------------------

    if year_min is not None:

        dff = dff[dff["year"] >= year_min].copy()

    if year_max is not None:

        dff = dff[dff["year"] <= year_max].copy()

    if region_year_limits is not None:

        keep_parts = []

        for region, sub in dff.groupby(region_col):

            if region in region_year_limits:

                yr0, yr1 = region_year_limits[region]

                sub = sub[(sub["year"] >= yr0) & (sub["year"] <= yr1)].copy()

            keep_parts.append(sub)

        dff = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame()

    if dff.empty:

        return {}, pd.DataFrame()

    # --------------------------------------------------------

    # 1) keep only sufficiently sampled buoy-months

    # --------------------------------------------------------

    if n_days_col in dff.columns:

        dff = dff[dff[n_days_col] >= min_days_per_month].copy()

    if dff.empty:

        return {}, pd.DataFrame()

    # --------------------------------------------------------

    # 2) require enough active buoys in each region-month

    # --------------------------------------------------------

    if min_buoys_per_month is not None:

        active_counts = (

            dff.groupby([region_col, month_col], as_index=False)

               .agg(n_active_buoys_month=(id_col, "nunique"))

        )

        dff = dff.merge(active_counts, on=[region_col, month_col], how="left")

        dff = dff[dff["n_active_buoys_month"] >= min_buoys_per_month].copy()

    if dff.empty:

        return {}, pd.DataFrame()

    # --------------------------------------------------------

    # 3) compute region-month series

    # --------------------------------------------------------

    if equal_weight_by_buoy:

        monthly_region = (

            dff.groupby([region_col, month_col], as_index=False)[products]

               .mean()

        )

    else:

        monthly_region = (

            dff.groupby([region_col, month_col], as_index=False)[products]

               .mean()

        )

    monthly_region["year"] = monthly_region[month_col].dt.year

    monthly_region["month_num"] = monthly_region[month_col].dt.month

    # count valid region-months contributing to annual means

    monthly_counts = (

        monthly_region.groupby([region_col, "year"], as_index=False)

        .agg(n_valid_months=("month_num", "nunique"))

    )

    # --------------------------------------------------------

    # 4) annual sample diagnostics from buoy-level valid records

    # --------------------------------------------------------

    annual_support = (

        dff.groupby([region_col, "year"], as_index=False)
           .agg(
               n_active_buoys_year=(id_col, "nunique"),
               n_valid_buoy_months=(id_col, "size"),
           )
    )

    # optional: theoretical max possible buoy-months given active buoys that year

    annual_support["max_possible_buoy_months"] = 12 * annual_support["n_active_buoys_year"]

    # optional: coverage fraction

    annual_support["buoy_month_coverage_frac"] = (
        annual_support["n_valid_buoy_months"] / annual_support["max_possible_buoy_months"]
    )

    # --------------------------------------------------------
    # 5) annual means from valid region-month means
    # --------------------------------------------------------

    annual_means = (
        monthly_region.groupby([region_col, "year"], as_index=False)[products]
        .mean()
    )

    annual_df = (
        annual_means
        .merge(monthly_counts, on=[region_col, "year"], how="left")
        .merge(annual_support, on=[region_col, "year"], how="left")
    )

    # if too few valid months, blank out annual means

    bad = annual_df["n_valid_months"] < min_months_per_year

    annual_df.loc[bad, products] = np.nan

    annual_df = annual_df.sort_values([region_col, "year"]).reset_index(drop=True)

    annual_by_region = {

        reg: sub.drop(columns=[region_col]).reset_index(drop=True)

        for reg, sub in annual_df.groupby(region_col)

    }

    return annual_by_region, annual_df
#-------------------------------------------------------------------
# ============================================================
# Monthly anomaly scatter from the MONTHLY buoy-product table
# Consistent with:
#   - min 20 valid buoy days per month
#   - optional min active buoys per region-month
#   - equal weighting by buoy
# ============================================================

def build_monthly_region_series_from_monthly_buoy_df(
    df,
    products,
    *,
    region_col="region",
    id_col="ID",
    month_col="month",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=2,
    equal_weight_by_buoy=True,
):
    """
    Build regional monthly series from the monthly buoy-product dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly buoy-product dataframe, e.g. all_buoy_product_monthly_df
    products : list
        Columns to retain, e.g.
        ["Buoy","GPCP v2.3","GPCP v3.2","GPCP v3.3","IMERG v07","ERA5","MERRA2"]
    min_days_per_month : int
        Minimum number of valid buoy days required in a buoy-month.
    min_buoys_per_month : int or None
        Require at least this many active buoys in a region-month.
    equal_weight_by_buoy : bool
        If True:
          1) keep one row per buoy-month
          2) region-month = mean across buoy monthly values
        If False:
          direct mean across all rows in a region-month.

    Returns
    -------
    monthly_region : pd.DataFrame
        Columns:
          region, month_start, n_active_buoys, products...
    """
    dff = df.copy()
    dff[month_col] = pd.to_datetime(dff[month_col])

    # -----------------------------
    # 1) keep only buoy-months with enough daily coverage
    # -----------------------------
    if n_days_col in dff.columns:
        dff = dff[dff[n_days_col] >= min_days_per_month].copy()

    if dff.empty:
        return pd.DataFrame(columns=[region_col, "month_start", "n_active_buoys"] + list(products))

    # -----------------------------
    # 2) count active buoys per region-month
    # -----------------------------
    active_counts = (
        dff.groupby([region_col, month_col], as_index=False)
           .agg(n_active_buoys=(id_col, "nunique"))
           .rename(columns={month_col: "month_start"})
    )

    # -----------------------------
    # 3) compute region-month means
    # -----------------------------
    if equal_weight_by_buoy:
        # one value per buoy-month is already present in the input table
        # region-month = average across buoy monthly values
        monthly_region = (
            dff.groupby([region_col, month_col], as_index=False)[products]
               .mean()
               .rename(columns={month_col: "month_start"})
        )
    else:
        monthly_region = (
            dff.groupby([region_col, month_col], as_index=False)[products]
               .mean()
               .rename(columns={month_col: "month_start"})
        )

    monthly_region = monthly_region.merge(
        active_counts,
        on=[region_col, "month_start"],
        how="left"
    )

    # -----------------------------
    # 4) optional minimum buoy count filter
    # -----------------------------
    if min_buoys_per_month is not None:
        monthly_region = monthly_region[
            monthly_region["n_active_buoys"] >= min_buoys_per_month
        ].copy()

    monthly_region = monthly_region.sort_values([region_col, "month_start"]).reset_index(drop=True)

    return monthly_region
#-------------------------------------------------------------------
# QC for OceanRAIN data

def oceanrain_step0_qc(
    df: pd.DataFrame,
    *,
    keep_cols=None,
    drop_harbor_inop=True,
    drop_spurious=True,
    min_flag2=13,          # e.g., 14 to keep >=0.1 mm/h (keeps true_zero too unless you exclude)
    prob_thr=None,           # e.g., 0.9 for high-confidence phase (optional)
    wind_max=15,           # e.g., 15.0 if you want a wind limit (optional)
    wind_col_preference=("true_wind_speed", "u10", "rel_wind_speed"),
    # sanity caps (conservative)
    dsd_cap_mmph=300.0,
    gag_cap_mmph=50.0,
    odm_cap_mmph=400.0,
    qclip_hi=None,          # e.g., 0.99 to clip top 1% (set >q to NaN); None disables
    qclip_cols=("dsd", "gag"),  # which groups to clip: any of {"dsd","gag"}
) -> pd.DataFrame:
    """
    Minute-level OceanRAIN QC for your *new* extracted dataframe.

    Expected columns (subset ok):
      time_utc, lat, lon, ship,
      rate_rain_dsd_mmph, rate_snow_dsd_mmph, rate_gag_mmph, rate_odm_mmph,
      precip_flag, precip_flag2,
      rain_prob, snow_prob, mixed_prob,
      true_wind_speed/u10/rel_wind_speed

    Returns minute-resolution dataframe ready for:
      - optional geographic subset (e.g., |lat|>=45)
      - GPCP pixel mapping
      - daily aggregation
    """

    df = df.copy()

    # ----------------------------
    # 1) Datetime + required cols
    # ----------------------------
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["time_utc", "lat", "lon"])

    # ----------------------------
    # 2) Replace common fill values with NaN
    # ----------------------------
    fill_vals = [-99.99, -99.9, -999.99, -999.9, -9999, -99999, -999.0, -9.9]
    num_cols = [
        "rate_rain_dsd_mmph", "rate_snow_dsd_mmph", "rate_gag_mmph", "rate_odm_mmph",
        "rain_prob", "snow_prob", "mixed_prob",
        "true_wind_speed", "u10", "rel_wind_speed"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].replace(fill_vals, np.nan)

    # Flags can carry fill values
    for c in ["precip_flag", "precip_flag2"]:
        if c in df.columns:
            df[c] = df[c].replace([9, 99, -99, -999], np.nan)

    # Use nullable integer for flags (safe with NaNs)
    if "precip_flag" in df.columns:
        df["precip_flag"] = df["precip_flag"].astype("Int64")
    if "precip_flag2" in df.columns:
        df["precip_flag2"] = df["precip_flag2"].astype("Int64")

    # ----------------------------
    # 3) Core QC filtering (minute-level)
    # ----------------------------
    m = pd.Series(True, index=df.index)

    # drop inoperative + harbor
    if drop_harbor_inop and "precip_flag" in df.columns:
        # 4=inoperative, 5=harbor
        m &= ~df["precip_flag"].isin([4, 5])

    # drop spurious/unknown precip_flag2
    if drop_spurious and "precip_flag2" in df.columns:
        # 11 = spurious_unknown
        m &= (df["precip_flag2"] != 11)

    # optional minimum intensity gate using precip_flag2
    # flag_values:
    # 10 true_zero
    # 12 precip_0.00
    # 13 precip_0.01-0.09
    # 14 precip_0.1-0.99
    # 15 precip_1.0-9.99
    # 16 precip_10.0-49.99
    # 17 precip_gt_50
    if (min_flag2 is not None) and ("precip_flag2" in df.columns):
        min_flag2 = int(min_flag2)
        # Keep true_zero (10) always; otherwise enforce >= min_flag2
        # m &= (df["precip_flag2"].isin([10, 12]) | (df["precip_flag2"] >= min_flag2))
        m &= (df["precip_flag2"].isin([10]) | (df["precip_flag2"] >= min_flag2))

    # optional wind filter
    wind_col = None
    for wc in wind_col_preference:
        if wc in df.columns:
            wind_col = wc
            break
    if (wind_max is not None) and (wind_col is not None):
        m &= (df[wind_col].isna() | (df[wind_col] <= float(wind_max)))

    df = df.loc[m].copy()

    # ----------------------------
    # 4) Rate sanity masks (do NOT drop rows; set bad values to NaN)
    # ----------------------------
    # DSD rain/snow: allow big values but remove absolute junk
    for c in ["rate_rain_dsd_mmph", "rate_snow_dsd_mmph"]:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > float(dsd_cap_mmph)), c] = np.nan

    # ODM rate (optional diagnostic)
    if "rate_odm_mmph" in df.columns:
        df.loc[(df["rate_odm_mmph"] < 0) | (df["rate_odm_mmph"] > float(odm_cap_mmph)), "rate_odm_mmph"] = np.nan

    # Gauge: remove placeholders/spikes
    if "rate_gag_mmph" in df.columns:
        df.loc[(df["rate_gag_mmph"] < 0), "rate_gag_mmph"] = np.nan
        df.loc[np.isclose(df["rate_gag_mmph"], 99.99, atol=1e-6), "rate_gag_mmph"] = np.nan
        df.loc[df["rate_gag_mmph"] >= float(gag_cap_mmph), "rate_gag_mmph"] = np.nan
        # ----------------------------
    # 4b) Optional high-end quantile clipping (set extreme values to NaN)
    #     Applied AFTER caps/QC; does not drop rows.
    # ----------------------------
    if qclip_hi is not None:
        q = float(qclip_hi)
        if not (0.0 < q < 1.0):
            raise ValueError("qclip_hi must be between 0 and 1 (e.g., 0.99)")

        # DSD columns
        if "dsd" in qclip_cols:
            for c in ["rate_rain_dsd_mmph", "rate_snow_dsd_mmph"]:
                if c in df.columns:
                    thr = df[c].quantile(q, interpolation="linear")
                    if pd.notna(thr):
                        df.loc[df[c] > thr, c] = np.nan

        # Gauge column
        if "gag" in qclip_cols:
            if "rate_gag_mmph" in df.columns:
                thr = df["rate_gag_mmph"].quantile(q, interpolation="linear")
                if pd.notna(thr):
                    df.loc[df["rate_gag_mmph"] > thr, "rate_gag_mmph"] = np.nan

    # ----------------------------
    # 5) Optional: high-confidence phase subsets via probabilities
    # (we still do NOT drop rows unless prob is present and <thr)
    # ----------------------------
    if prob_thr is not None:
        thr = float(prob_thr)

        if "rain_prob" in df.columns and "precip_flag" in df.columns:
            df = df[~((df["precip_flag"] == 0) & (df["rain_prob"].notna()) & (df["rain_prob"] < thr))]

        if "snow_prob" in df.columns and "precip_flag" in df.columns:
            df = df[~((df["precip_flag"] == 1) & (df["snow_prob"].notna()) & (df["snow_prob"] < thr))]

        if "mixed_prob" in df.columns and "precip_flag" in df.columns:
            df = df[~((df["precip_flag"] == 2) & (df["mixed_prob"].notna()) & (df["mixed_prob"] < thr))]

        df = df.copy()

    # ----------------------------
    # 6) Keep only needed columns (memory efficiency)
    # ----------------------------
    default_cols = [
        "time_utc", "lat", "lon", "ship",
        "rate_rain_dsd_mmph", "rate_snow_dsd_mmph",
        "rate_gag_mmph", "rate_odm_mmph",
        "precip_flag", "precip_flag2",
        "rain_prob", "snow_prob", "mixed_prob",
        "true_wind_speed", "u10", "rel_wind_speed"
    ]
    if keep_cols is None:
        keep_cols = [c for c in default_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)


#-------------------------------------------------------------------

def oceanrain_step0_qc_v2(
    df: pd.DataFrame,
    *,
    keep_cols=None,

    # core flag handling
    drop_harbor_inop: bool = True,
    keep_true_zero: bool = True,
    keep_spurious_flag2_11: bool = False,
    min_flag2: Optional[int] = 13,

    # optional confidence gates
    prob_thr: Optional[float] = None,
    wind_max: Optional[float] = None,
    wind_col_preference: Tuple[str, ...] = (
        "true_wind_speed", "wind_speed_in_10m_height", "relative_wind_speed"
    ),

    # sanity caps
    dsd_cap_mmph: float = 350.0,
    gag_cap_mmph: float = 60.0,
    odm_cap_mmph: float = 450.0,

    # optional: quantile clipping AFTER phase-consistency
    qclip_hi: Optional[float] = None,
) -> pd.DataFrame:

    df = df.copy()

    # 1) Datetime + basics
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=[c for c in ["time_utc", "lat", "lon"] if c in df.columns])

    # 2) Fill/sentinel -> NaN (expanded)
    fill_vals = {
        -99.9999, -99.99, -99.9, -99,
        -999.9999, -999.99, -999.9, -999,
        -9999, -99999,
        -9.9, -9,
        99.99, 999.99, 999.98999, 999.989990234375
    }

    num_cols = [
        "rate_rain_dsd_mmph", "rate_snow_dsd_mmph", "rate_gag_mmph", "rate_odm_mmph",
        "rain_prob", "snow_prob", "mixed_prob",
        "true_wind_speed", "wind_speed_in_10m_height", "relative_wind_speed"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].replace(list(fill_vals), np.nan)

    # special true-zero sentinel used in some OceanRAIN-W diagnostics
    for c in ["true_wind_speed", "relative_wind_speed"]:
        if c in df.columns:
            df[c] = df[c].replace([-888.88], np.nan)

    # flags -> Int64 with NaNs
    for c in ["precip_flag", "precip_flag2"]:
        if c in df.columns:
            df[c] = df[c].replace([9, 99, -99, -999], np.nan).astype("Int64")

    # 3) Core row filters
    m = pd.Series(True, index=df.index)

    if "precip_flag" in df.columns:
        if drop_harbor_inop:
            m &= ~df["precip_flag"].isin([4, 5])
        if not keep_true_zero:
            m &= (df["precip_flag"] != 3)

    if "precip_flag2" in df.columns:
        if not keep_spurious_flag2_11:
            m &= (df["precip_flag2"] != 11)

        if min_flag2 is not None:
            min_flag2 = int(min_flag2)
            if keep_true_zero:
                m &= (df["precip_flag2"] == 10) | (df["precip_flag2"] >= min_flag2)
            else:
                m &= (df["precip_flag2"] >= min_flag2)

    # optional wind filter
    wind_col = None
    for wc in wind_col_preference:
        if wc in df.columns:
            wind_col = wc
            break
    if (wind_max is not None) and (wind_col is not None):
        m &= (df[wind_col].isna() | (df[wind_col] <= float(wind_max)))

    df = df.loc[m].copy()

    # 4) Phase-consistent rate
    df["rate_best_mmph"] = np.nan

    if "precip_flag" in df.columns:
        if "rate_rain_dsd_mmph" in df.columns:
            df.loc[df["precip_flag"] == 0, "rate_best_mmph"] = df.loc[df["precip_flag"] == 0, "rate_rain_dsd_mmph"]

        if "rate_snow_dsd_mmph" in df.columns:
            df.loc[df["precip_flag"] == 1, "rate_best_mmph"] = df.loc[df["precip_flag"] == 1, "rate_snow_dsd_mmph"]
            df.loc[df["precip_flag"] == 2, "rate_best_mmph"] = df.loc[df["precip_flag"] == 2, "rate_snow_dsd_mmph"]

        df.loc[df["precip_flag"] == 3, "rate_best_mmph"] = 0.0

        # null-out phase-inconsistent theoretical cols to prevent accidental use
        if "rate_rain_dsd_mmph" in df.columns:
            df.loc[~df["precip_flag"].isin([0, 2]), "rate_rain_dsd_mmph"] = np.nan
        if "rate_snow_dsd_mmph" in df.columns:
            df.loc[~df["precip_flag"].isin([1, 2]), "rate_snow_dsd_mmph"] = np.nan

    # 5) Sanity caps (set junk to NaN; do not drop rows)
    for c, cap in [
        ("rate_rain_dsd_mmph", dsd_cap_mmph),
        ("rate_snow_dsd_mmph", dsd_cap_mmph),
        ("rate_odm_mmph",      odm_cap_mmph),
        ("rate_gag_mmph",      gag_cap_mmph),
        ("rate_best_mmph",     max(dsd_cap_mmph, odm_cap_mmph)),
    ]:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > float(cap)), c] = np.nan

    # 6) Optional probability confidence gate
    if prob_thr is not None and "precip_flag" in df.columns:
        thr = float(prob_thr)
        if "rain_prob" in df.columns:
            df = df[~((df["precip_flag"] == 0) & df["rain_prob"].notna() & (df["rain_prob"] < thr))]
        if "snow_prob" in df.columns:
            df = df[~((df["precip_flag"] == 1) & df["snow_prob"].notna() & (df["snow_prob"] < thr))]
        if "mixed_prob" in df.columns:
            df = df[~((df["precip_flag"] == 2) & df["mixed_prob"].notna() & (df["mixed_prob"] < thr))]
        df = df.copy()

    # 7) Optional quantile clip (usually apply to rate_best only)
    if qclip_hi is not None:
        q = float(qclip_hi)
        if not (0.0 < q < 1.0):
            raise ValueError("qclip_hi must be between 0 and 1 (e.g., 0.995)")

        for c in ["rate_best_mmph"]:
            if c in df.columns:
                thr = df[c].quantile(q, interpolation="linear")
                if pd.notna(thr):
                    df.loc[df[c] > thr, c] = np.nan

    # 8) Keep cols
    default_cols = [
        "time_utc", "lat", "lon", "ship",
        "precip_flag", "precip_flag2",
        "rain_prob", "snow_prob", "mixed_prob",
        "rate_best_mmph",
        "rate_rain_dsd_mmph", "rate_snow_dsd_mmph",
        "rate_gag_mmph", "rate_odm_mmph",
        "true_wind_speed", "wind_speed_in_10m_height", "relative_wind_speed"
    ]
    if keep_cols is None:
        keep_cols = [c for c in default_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)
# -------------------------------------------------------------------


def oceanrain_step0_qc_precip_main(
    df: pd.DataFrame,
    *,
    keep_cols: Optional[Sequence[str]] = None,

    # core row filtering
    drop_harbor_inop: bool = True,
    drop_spurious_flag2_11: bool = True,
    drop_flag2_17: bool = False,
    keep_true_zero: bool = True,

    # whether to keep flag2=12 (precipitation minutes with 0.00 mm/h)
    # useful choice depends on analysis:
    #   True  -> keeps borderline/light precip occurrence minutes
    #   False -> stricter for quantitative rate validation
    keep_flag2_12_zero_precip: bool = False,

    # optional minimum flag2 threshold for positive precipitation classes
    # None: no extra threshold
    # 13 : keep 0.01-0.09 and above (plus true-zero if keep_true_zero=True)
    # 14 : keep 0.1 and above     (plus true-zero if keep_true_zero=True)
    min_flag2_positive: Optional[int] = 13,

    # optional confidence gate by phase probabilities
    # not recommended as default for total-precip use
    prob_thr: Optional[float] = None,

    # optional wind sensitivity filter (not core QC)
    wind_max: Optional[float] = None,
    wind_col_preference: Tuple[str, ...] = (
        "true_wind_speed",
        "wind_speed_in_10m_height",
        "relative_wind_speed",
        "u10",
        "rel_wind_speed",
    ),

    # conservative physical caps (set bad values to NaN; do not drop rows)
    dsd_cap_mmph: float = 350.0,
    gag_cap_mmph: float = 60.0,
    odm_cap_mmph: float = 400.0,

    # optional upper-tail clipping (off by default)
    qclip_hi: Optional[float] = None,
):
    """
    OceanRAIN minute-level QC for the current study goal:
    use OceanRAIN's main precipitation variable directly as the primary reference,
    without splitting rain vs snow for the main analysis.

    Expected columns (subset okay):
      time_utc, lat, lon, ship,
      precip_flag, precip_flag2,
      rate_odm_mmph,
      rate_rain_dsd_mmph, rate_snow_dsd_mmph, rate_gag_mmph,
      rain_prob, snow_prob, mixed_prob,
      true_wind_speed / wind_speed_in_10m_height / relative_wind_speed / u10 / rel_wind_speed

    Returns:
      dataframe with:
        - main variable: rate_main_mmph   (QC'd ODM precipitation rate)
        - optional diagnostic: rate_phasebest_mmph
        - original selected columns for traceability
    """

    df = df.copy()

    # --------------------------------------------------
    # 1) Datetime / geolocation basics
    # --------------------------------------------------
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)

    needed = [c for c in ["time_utc", "lat", "lon"] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed).copy()

    # --------------------------------------------------
    # 2) Replace common sentinel / fill values with NaN
    # --------------------------------------------------
    fill_vals = {
        -99.9999, -99.99, -99.9, -99,
        -999.9999, -999.99, -999.9, -999,
        -9999, -99999,
        -9.9, -9,
        99.99, 999.99, 999.98999, 999.989990234375,
    }

    num_cols = [
        "rate_odm_mmph",
        "rate_rain_dsd_mmph",
        "rate_snow_dsd_mmph",
        "rate_gag_mmph",
        "rain_prob",
        "snow_prob",
        "mixed_prob",
        "true_wind_speed",
        "wind_speed_in_10m_height",
        "relative_wind_speed",
        "u10",
        "rel_wind_speed",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].replace(list(fill_vals), np.nan)

    # In OceanRAIN-W, true-zero minutes can use -888.88 for some ODM diagnostics
    for c in ["true_wind_speed", "relative_wind_speed", "rel_wind_speed"]:
        if c in df.columns:
            df[c] = df[c].replace([-888.88], np.nan)

    # flags -> nullable integer
    for c in ["precip_flag", "precip_flag2"]:
        if c in df.columns:
            df[c] = df[c].replace([9, 99, -99, -999], np.nan).astype("Int64")

    # --------------------------------------------------
    # 3) Core row filtering based on official flags
    # --------------------------------------------------
    m = pd.Series(True, index=df.index)

    # precip_flag:
    # 0 rain, 1 snow, 2 mixed, 3 true-zero, 4 inoperative, 5 harbor
    if "precip_flag" in df.columns:
        if drop_harbor_inop:
            m &= ~df["precip_flag"].isin([4, 5])

        if not keep_true_zero:
            m &= (df["precip_flag"] != 3)

    # precip_flag2:
    # 10 true-zero, 11 spurious, 12 precip but 0.00 mm/h, 13..17 increasing positive intensity
    if "precip_flag2" in df.columns:
        if drop_spurious_flag2_11:
            m &= (df["precip_flag2"] != 11)

        if drop_flag2_17:
            m &= (df["precip_flag2"] != 17)

        allowed_flag2 = pd.Series(False, index=df.index)

        if keep_true_zero:
            allowed_flag2 |= (df["precip_flag2"] == 10)

        if keep_flag2_12_zero_precip:
            allowed_flag2 |= (df["precip_flag2"] == 12)

        if min_flag2_positive is None:
            # no extra threshold: allow all non-spurious positive/intensity classes
            allowed_flag2 |= df["precip_flag2"].isin([13, 14, 15, 16, 17])
        else:
            min_flag2_positive = int(min_flag2_positive)
            allowed_flag2 |= (df["precip_flag2"] >= min_flag2_positive)

        m &= allowed_flag2

    # optional wind filter
    wind_col = None
    for wc in wind_col_preference:
        if wc in df.columns:
            wind_col = wc
            break

    if (wind_max is not None) and (wind_col is not None):
        m &= (df[wind_col].isna() | (df[wind_col] <= float(wind_max)))

    df = df.loc[m].copy()

    # --------------------------------------------------
    # 4) Build main precipitation variable
    # --------------------------------------------------
    # Main variable = QC'd ODM precipitation rate
    # For true-zero minutes, force to 0.0
    df["rate_main_mmph"] = np.nan

    if "rate_odm_mmph" in df.columns:
        df["rate_main_mmph"] = df["rate_odm_mmph"]

    if "precip_flag" in df.columns:
        df.loc[df["precip_flag"] == 3, "rate_main_mmph"] = 0.0

    # --------------------------------------------------
    # 5) Keep optional diagnostic phase-based rate
    # --------------------------------------------------
    # This is NOT the main variable for the current study.
    # It is kept only for later checks / sensitivity / interpretation.
    df["rate_phasebest_mmph"] = np.nan

    if "precip_flag" in df.columns:
        if "rate_rain_dsd_mmph" in df.columns:
            df.loc[df["precip_flag"] == 0, "rate_phasebest_mmph"] = df.loc[
                df["precip_flag"] == 0, "rate_rain_dsd_mmph"
            ]

        if "rate_snow_dsd_mmph" in df.columns:
            df.loc[df["precip_flag"] == 1, "rate_phasebest_mmph"] = df.loc[
                df["precip_flag"] == 1, "rate_snow_dsd_mmph"
            ]

            # OceanRAIN paper notes mixed-phase currently uses snowfall intensity
            df.loc[df["precip_flag"] == 2, "rate_phasebest_mmph"] = df.loc[
                df["precip_flag"] == 2, "rate_snow_dsd_mmph"
            ]

        df.loc[df["precip_flag"] == 3, "rate_phasebest_mmph"] = 0.0

    # --------------------------------------------------
    # 6) Sanity caps (set impossible values to NaN)
    # --------------------------------------------------
    for c, cap in [
        ("rate_odm_mmph", odm_cap_mmph),
        ("rate_main_mmph", odm_cap_mmph),
        ("rate_rain_dsd_mmph", dsd_cap_mmph),
        ("rate_snow_dsd_mmph", dsd_cap_mmph),
        ("rate_phasebest_mmph", dsd_cap_mmph),
        ("rate_gag_mmph", gag_cap_mmph),
    ]:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > float(cap)), c] = np.nan

    # re-impose true-zero after caps, just in case
    if "precip_flag" in df.columns:
        df.loc[df["precip_flag"] == 3, "rate_main_mmph"] = 0.0
        df.loc[df["precip_flag"] == 3, "rate_phasebest_mmph"] = 0.0

    # --------------------------------------------------
    # 7) Optional probability confidence filter
    # --------------------------------------------------
    # Mostly useful as a sensitivity test, not default production QC.
    if (prob_thr is not None) and ("precip_flag" in df.columns):
        thr = float(prob_thr)

        if "rain_prob" in df.columns:
            df = df[
                ~(
                    (df["precip_flag"] == 0)
                    & df["rain_prob"].notna()
                    & (df["rain_prob"] < thr)
                )
            ]

        if "snow_prob" in df.columns:
            df = df[
                ~(
                    (df["precip_flag"] == 1)
                    & df["snow_prob"].notna()
                    & (df["snow_prob"] < thr)
                )
            ]

        if "mixed_prob" in df.columns:
            df = df[
                ~(
                    (df["precip_flag"] == 2)
                    & df["mixed_prob"].notna()
                    & (df["mixed_prob"] < thr)
                )
            ]

        df = df.copy()

    # --------------------------------------------------
    # 8) Optional upper-tail clip
    # --------------------------------------------------
    if qclip_hi is not None:
        q = float(qclip_hi)
        if not (0.0 < q < 1.0):
            raise ValueError("qclip_hi must be between 0 and 1, e.g. 0.995")

        for c in ["rate_main_mmph", "rate_phasebest_mmph"]:
            if c in df.columns:
                thr = df[c].quantile(q, interpolation="linear")
                if pd.notna(thr):
                    df.loc[df[c] > thr, c] = np.nan

        # again preserve true-zero explicitly
        if "precip_flag" in df.columns:
            df.loc[df["precip_flag"] == 3, "rate_main_mmph"] = 0.0
            df.loc[df["precip_flag"] == 3, "rate_phasebest_mmph"] = 0.0

    # --------------------------------------------------
    # 9) Keep output columns
    # --------------------------------------------------
    default_cols = [
        "time_utc", "lat", "lon", "ship",
        "precip_flag", "precip_flag2",
        "rain_prob", "snow_prob", "mixed_prob",
        "rate_main_mmph",          # <-- recommended primary variable
        "rate_phasebest_mmph",     # diagnostic only
        "rate_odm_mmph",
        "rate_rain_dsd_mmph",
        "rate_snow_dsd_mmph",
        "rate_gag_mmph",
        "true_wind_speed",
        "wind_speed_in_10m_height",
        "relative_wind_speed",
        "u10",
        "rel_wind_speed",
    ]
    if keep_cols is None:
        keep_cols = [c for c in default_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)
#-------------------------------------------------------------------
# Nearest grid-index mapping (your current function is fine)
def map_to_gpcp_idx(arr1d, values):
    a = np.asarray(arr1d)
    v = np.asarray(values)
    if a.ndim != 1: a = a.ravel()
    if v.ndim != 1: v = v.ravel()

    v_finite = np.where(np.isfinite(v), v, np.nan)

    asc = bool(a[0] <= a[-1])
    a_work = a if asc else a[::-1]

    idx = np.searchsorted(a_work, v_finite)
    idx0 = np.clip(idx - 1, 0, a_work.size - 1)
    idx1 = np.clip(idx, 0, a_work.size - 1)

    choose_left = (np.abs(v_finite - a_work[idx0]) <= np.abs(v_finite - a_work[idx1]))
    out_rev = np.where(choose_left, idx0, idx1)

    out = out_rev if asc else (a_work.size - 1) - out_rev

    if np.issubdtype(v.dtype, np.floating):
        nanmask = ~np.isfinite(v)
        if nanmask.any():
            out = out.astype("int64")
            out[nanmask] = 0

    return out

# -------------------------------------------------------------------
# Daily aggregation (phase-consistent)
def oceanrain_daily_aggregate_to_gpcp_v2(
    oc_df_minute: pd.DataFrame,
    gpcp_lat_1d,
    gpcp_lon_1d,
    *,
    lat_abs_min=45.0,
    coverage_frac=0.10,         # e.g. 0.10 * 1440 = 144 minutes
    phase_frac_thr=0.20,        # "mostly rain/snow" fraction threshold
    include_mixed_in_all=False, # if True, include mixed in all-precip total
    mixed_rule="rain",          # "rain" or "ignore" (if include_mixed_in_all=True)
    wind_mean_col_preference=("true_wind_speed", "u10", "rel_wind_speed"),
):
    """
    Step 1:
      - (Assumes Step0 QC already done)
      - poleward subset
      - map to (ilat, ilon)
      - daily ship+pixel aggregation
      - adds lat_c, lon_c, hemi for NH/SH analysis

    Required columns in oc_df_minute:
      time_utc, lat, lon, ship, precip_flag,
      rate_rain_dsd_mmph, rate_snow_dsd_mmph
    Optional:
      rate_gag_mmph, true_wind_speed/u10/rel_wind_speed
    """

    df = oc_df_minute.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon"])
    df["date"] = df["time_utc"].dt.floor("D")

    # poleward subset
    if lat_abs_min is not None:
        df = df[df["lat"].abs() >= float(lat_abs_min)].copy()

    # map to gpcp indices (lon wrap to [-180,180))
    lon_wrapped = ((df["lon"].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0
    df["ilat"] = map_to_gpcp_idx(np.asarray(gpcp_lat_1d), df["lat"].to_numpy())
    df["ilon"] = map_to_gpcp_idx(np.asarray(gpcp_lon_1d), lon_wrapped)

    # choose wind column
    wind_col = None
    for wc in wind_mean_col_preference:
        if wc in df.columns:
            wind_col = wc
            break

    grp_keys = ["date", "ilat", "ilon", "ship"]

    # ---------- coverage + phase fractions ----------
    # fractions: mean(boolean) works because True=1, False=0
    daily_cov = (
        df.groupby(grp_keys, as_index=False)
          .agg(
              n_min_total=("precip_flag", "size"),
              n_min_rain=("precip_flag", lambda s: np.sum(s.to_numpy() == 0)),
              n_min_snow=("precip_flag", lambda s: np.sum(s.to_numpy() == 1)),
              n_min_mixed=("precip_flag", lambda s: np.sum(s.to_numpy() == 2)),

              frac_rain=("precip_flag", lambda s: np.mean(s.to_numpy() == 0)),
              frac_snow=("precip_flag", lambda s: np.mean(s.to_numpy() == 1)),
              frac_mixed=("precip_flag", lambda s: np.mean(s.to_numpy() == 2)),

              wind_mean=(wind_col, "mean") if wind_col is not None else ("lat", "mean"),
          )
    )

    # ---------- phase-consistent rates ----------
    d2_cols = grp_keys + ["precip_flag", "rate_rain_dsd_mmph", "rate_snow_dsd_mmph"]
    if "rate_gag_mmph" in df.columns:
        d2_cols += ["rate_gag_mmph"]

    d2 = df[d2_cols].copy()

    # rain-only and snow-only minute series
    d2["dsd_rain_mmph"] = d2["rate_rain_dsd_mmph"].where(d2["precip_flag"] == 0)
    d2["dsd_snow_mmph"] = d2["rate_snow_dsd_mmph"].where(d2["precip_flag"] == 1)

    # all-phase precip (liquid-equivalent using the appropriate theoretical rate)
    d2["dsd_all_mmph"] = np.nan
    d2.loc[d2["precip_flag"] == 0, "dsd_all_mmph"] = d2.loc[d2["precip_flag"] == 0, "rate_rain_dsd_mmph"]
    d2.loc[d2["precip_flag"] == 1, "dsd_all_mmph"] = d2.loc[d2["precip_flag"] == 1, "rate_snow_dsd_mmph"]

    if include_mixed_in_all:
        if mixed_rule == "rain":
            d2.loc[d2["precip_flag"] == 2, "dsd_all_mmph"] = d2.loc[d2["precip_flag"] == 2, "rate_rain_dsd_mmph"]
        elif mixed_rule == "ignore":
            pass
        else:
            raise ValueError("mixed_rule must be 'rain' or 'ignore'")

    def _mmday_from_mmph(series):
        # minute sampling: mm/day = sum(mm/h)/60
        return np.nansum(series.to_numpy()) / 60.0

    daily_rates = (
        d2.groupby(grp_keys, as_index=False)
          .agg(
              dsd_mean_all_mmph=("dsd_all_mmph", "mean"),
              dsd_mean_rain_mmph=("dsd_rain_mmph", "mean"),
              dsd_mean_snow_mmph=("dsd_snow_mmph", "mean"),

              dsd_mmday_all=("dsd_all_mmph", _mmday_from_mmph),
              dsd_mmday_rain=("dsd_rain_mmph", _mmday_from_mmph),
              dsd_mmday_snow=("dsd_snow_mmph", _mmday_from_mmph),

              gag_mean_mmph=("rate_gag_mmph", "mean") if "rate_gag_mmph" in d2.columns else ("dsd_all_mmph", "mean"),
              gag_mmday=("rate_gag_mmph", _mmday_from_mmph) if "rate_gag_mmph" in d2.columns else ("dsd_all_mmph", _mmday_from_mmph),
          )
    )

    daily_or = daily_cov.merge(daily_rates, on=grp_keys, how="left")

    # ---------- attach coordinate centers + hemisphere ----------
    g_lat = np.asarray(gpcp_lat_1d)
    g_lon = np.asarray(gpcp_lon_1d)

    daily_or["lat_c"] = g_lat[daily_or["ilat"].to_numpy()]
    daily_or["lon_c"] = g_lon[daily_or["ilon"].to_numpy()]
    daily_or["hemi"] = np.where(daily_or["lat_c"] >= 0, "NH", "SH")

    # ---------- thresholds for "usable" days ----------
    min_minutes = float(coverage_frac) * (24.0 * 60.0)

    mostly_rain = daily_or["frac_rain"] >= float(phase_frac_thr)
    mostly_snow = daily_or["frac_snow"] >= float(phase_frac_thr)

    rain_days = daily_or[(daily_or["n_min_rain"] >= min_minutes) & mostly_rain].copy()
    snow_days = daily_or[(daily_or["n_min_snow"] >= min_minutes) & mostly_snow].copy()

    return daily_or, rain_days, snow_days

# ------------------------------------------------------------
# helper: minute-rate series (mm/h) -> daily accumulation (mm/day)
# ------------------------------------------------------------
def _mmday_from_mmph(series):
    # OceanRAIN-W is 1-minute data:
    # accumulation over a minute = (mm/h) * (1/60 h)
    # daily total = sum(rate_mmph / 60)
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return np.nansum(x) / 60.0

def _mmday_from_mmph_mean24(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    if np.isfinite(x).sum() == 0:
        return np.nan
    return np.nanmean(x) * 24.0

#------------------------------------------------------------

def oceanrain_minute_match_common1deg_then_daily_xr(
    oc_df_minute: pd.DataFrame,
    *,
    products,
    grid_lat_1d,
    grid_lon_1d,
    time_col="time_utc",
    date_col="date",
    lat_col="lat",
    lon_col="lon",
    ship_col="ship",
    obs_rate_col="rate_main_mmph",
    precip_flag_col="precip_flag",
    lat_abs_min=45.0,
    coverage_frac=0.5,
    method="nearest",
    tolerance_time=None,
    dropna_product_cols=False,
    product_dropna_mode="all",
    chunk_size=200_000,
):
    """
    PAL-like OceanRAIN common-1° workflow.

    This explicitly matches each OceanRAIN minute sample to the common 1°
    analysis grid and attaches product values at the minute level before
    daily aggregation.

    Workflow:
        OceanRAIN minute observation
            -> nearest common 1° grid cell
            -> product value at that date/grid cell
            -> daily ship-grid-cell aggregation

    This avoids using a daily mean ship lat/lon to sample products.
    """

    import numpy as np
    import pandas as pd
    import xarray as xr

    df = oc_df_minute.copy()

    # -------------------------------------------------------------
    # 1. Basic cleaning
    # -------------------------------------------------------------
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col, ship_col]).copy()

    if obs_rate_col not in df.columns:
        raise ValueError(f"Input dataframe must contain '{obs_rate_col}'.")

    df[obs_rate_col] = pd.to_numeric(df[obs_rate_col], errors="coerce")
    df = df[np.isfinite(df[obs_rate_col])].copy()

    if lat_abs_min is not None:
        df = df[df[lat_col].abs() >= float(lat_abs_min)].copy()

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Daily product timestamp
    df[date_col] = df[time_col].dt.floor("D").dt.tz_localize(None)

    # Wrap OceanRAIN longitude to -180..180
    df[lon_col] = ((df[lon_col].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0

    # -------------------------------------------------------------
    # 2. Assign each minute to nearest common 1° grid cell using xarray
    # -------------------------------------------------------------
    grid_lat_1d = np.asarray(grid_lat_1d)
    grid_lon_1d = np.asarray(grid_lon_1d)

    grid_da = xr.DataArray(
        np.zeros((len(grid_lat_1d), len(grid_lon_1d)), dtype=np.float32),
        coords={"lat": grid_lat_1d, "lon": grid_lon_1d},
        dims=("lat", "lon"),
        name="grid_id_dummy",
    )

    lat_points = xr.DataArray(df[lat_col].to_numpy(dtype="float64"), dims="points")
    lon_points = xr.DataArray(df[lon_col].to_numpy(dtype="float64"), dims="points")

    grid_sel = grid_da.sel(
        lat=lat_points,
        lon=lon_points,
        method="nearest",
    )

    df["lat_c"] = grid_sel["lat"].to_numpy()
    df["lon_c"] = grid_sel["lon"].to_numpy()
    df["lon_c"] = ((df["lon_c"].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0
    df["hemi"] = np.where(df["lat_c"] >= 0, "NH", "SH")

    # Optional integer indices, useful for stable grouping and debugging
    lat_index = {float(v): i for i, v in enumerate(grid_lat_1d)}
    lon_index = {float(((v + 180.0) % 360.0) - 180.0): i for i, v in enumerate(grid_lon_1d)}

    df["ilat"] = [lat_index[float(v)] for v in df["lat_c"].to_numpy()]
    df["ilon"] = [lon_index[float(v)] for v in df["lon_c"].to_numpy()]

    # -------------------------------------------------------------
    # 3. Attach product values at minute level
    # -------------------------------------------------------------
    minute_matched = df.copy()

    if isinstance(products, dict):
        iterable = products.items()
    elif isinstance(products, list):
        iterable = products
    else:
        raise TypeError(f"`products` must be dict or list, got {type(products)}")

    for item in iterable:
        if isinstance(products, dict):
            name, (xr_obj, var_map) = item
        else:
            name, xr_obj, var_map = item

        print(f"Minute common-1° attaching: {name}")

        minute_matched = attach_satellite_vars_pointwise_chunked(
            minute_matched,
            xr_obj,
            var_map=var_map,
            date_col=date_col,
            lat_col="lat_c",
            lon_col="lon_c",
            method=method,
            tolerance_time=tolerance_time,
            chunk_size=chunk_size,
        )

    # Product columns
    product_cols = []
    if isinstance(products, dict):
        for _, (_, var_map) in products.items():
            product_cols.extend(list(var_map.keys()))
    else:
        for _, _, var_map in products:
            product_cols.extend(list(var_map.keys()))

    product_cols = list(dict.fromkeys(product_cols))

    # Mask negative product values
    for col in product_cols:
        if col in minute_matched.columns:
            minute_matched[col] = pd.to_numeric(minute_matched[col], errors="coerce")
            minute_matched.loc[minute_matched[col] < 0, col] = np.nan

    # Optional product NaN filtering
    if dropna_product_cols and len(product_cols) > 0:
        if product_dropna_mode == "all":
            minute_matched = minute_matched.dropna(subset=product_cols, how="all").copy()
        elif product_dropna_mode == "any":
            minute_matched = minute_matched.dropna(subset=product_cols, how="any").copy()
        else:
            raise ValueError("product_dropna_mode must be 'all' or 'any'.")

    if minute_matched.empty:
        empty = pd.DataFrame()
        return empty, empty, minute_matched

    # -------------------------------------------------------------
    # 4. Aggregate matched minute records to daily ship-grid-cell values
    # -------------------------------------------------------------
    grp_keys = [date_col, "ilat", "ilon", ship_col]

    agg_dict = {
        "n_min_total": (obs_rate_col, "size"),
        "n_min_valid_rate": (
            obs_rate_col,
            lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce"))),
        ),
        "main_mean_mmph": (obs_rate_col, "mean"),
        "main_median_mmph": (obs_rate_col, "median"),
        "main_max_mmph": (obs_rate_col, "max"),
        "main_mmday": (obs_rate_col, _mmday_from_mmph),
        "lat_mean": (lat_col, "mean"),
        "lon_mean": (lon_col, "mean"),
        "lat_c": ("lat_c", "first"),
        "lon_c": ("lon_c", "first"),
        "hemi": ("hemi", "first"),
    }

    if precip_flag_col in minute_matched.columns:
        agg_dict.update({
            "n_min_zero": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 3)),
            "n_min_rain": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 0)),
            "n_min_snow": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 1)),
            "n_min_mixed": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 2)),
            "frac_zero": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 3)),
            "frac_rain": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 0)),
            "frac_snow": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 1)),
            "frac_mixed": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 2)),
        })

    if "rate_phasebest_mmph" in minute_matched.columns:
        agg_dict.update({
            "phasebest_mean_mmph": ("rate_phasebest_mmph", "mean"),
            "phasebest_mmday": ("rate_phasebest_mmph", _mmday_from_mmph),
        })

    if "rate_odm_mmph" in minute_matched.columns:
        agg_dict.update({
            "odm_mean_mmph": ("rate_odm_mmph", "mean"),
            "odm_mmday": ("rate_odm_mmph", _mmday_from_mmph),
        })

    if "rate_gag_mmph" in minute_matched.columns:
        agg_dict.update({
            "gag_mean_mmph": ("rate_gag_mmph", "mean"),
            "gag_mmday": ("rate_gag_mmph", _mmday_from_mmph),
        })

    if "rain_prob" in minute_matched.columns:
        agg_dict["rain_prob_mean"] = ("rain_prob", "mean")

    if "snow_prob" in minute_matched.columns:
        agg_dict["snow_prob_mean"] = ("snow_prob", "mean")

    if "mixed_prob" in minute_matched.columns:
        agg_dict["mixed_prob_mean"] = ("mixed_prob", "mean")

    # Product values are daily precipitation fields.
    # Mean over matched minutes gives the ship-sampled daily product value.
    for col in product_cols:
        if col in minute_matched.columns:
            agg_dict[col] = (col, "mean")

    daily_all = (
        minute_matched
        .groupby(grp_keys, as_index=False)
        .agg(**agg_dict)
    )

    daily_all["coverage_frac_day"] = daily_all["n_min_total"] / 1440.0

    if all(c in daily_all.columns for c in ["n_min_rain", "n_min_snow", "n_min_mixed"]):
        daily_all["n_min_precip_phase"] = (
            daily_all["n_min_rain"] +
            daily_all["n_min_snow"] +
            daily_all["n_min_mixed"]
        )

        daily_all["frac_precip_phase"] = (
            daily_all["frac_rain"] +
            daily_all["frac_snow"] +
            daily_all["frac_mixed"]
        )

        phase_cols = ["frac_rain", "frac_snow", "frac_mixed", "frac_zero"]
        phase_names = ["rain", "snow", "mixed", "zero"]

        phase_arr = daily_all[phase_cols].to_numpy(dtype="float64")
        phase_idx = np.nanargmax(phase_arr, axis=1)
        daily_all["dominant_phase"] = np.array(phase_names, dtype=object)[phase_idx]

    min_minutes = float(coverage_frac) * 1440.0
    daily_usable = daily_all[daily_all["n_min_total"] >= min_minutes].copy()

    return daily_all, daily_usable, minute_matched

# ============================================================
# helper: categorical metrics dict -> tidy table
# ============================================================
def compute_spread_ratio_from_daily_pairs(
    df,
    *,
    obs_col="main_mmday",
    product_cols=None,
    hemi_col="hemi",
    hemis=("NH", "SH"),
):
    rows = []

    if product_cols is None:
        raise ValueError("product_cols must be provided.")

    for hemi in hemis:
        dfh = df[df[hemi_col] == hemi].copy()

        for prod in product_cols:
            sub = dfh[[obs_col, prod]].replace([np.inf, -np.inf], np.nan).dropna().copy()

            if len(sub) < 2:
                rows.append({
                    "hemi": hemi,
                    "product": prod,
                    "N_pairs": len(sub),
                    "obs_std": np.nan,
                    "prod_std": np.nan,
                    "spread_ratio": np.nan,
                })
                continue

            obs_std = sub[obs_col].std(ddof=1)
            prod_std = sub[prod].std(ddof=1)

            spread_ratio = np.nan
            if np.isfinite(obs_std) and obs_std != 0:
                spread_ratio = prod_std / obs_std

            rows.append({
                "hemi": hemi,
                "product": prod,
                "N_pairs": len(sub),
                "obs_std": obs_std,
                "prod_std": prod_std,
                "spread_ratio": spread_ratio,
            })

    return pd.DataFrame(rows)
#------------------------------------------------------------
def quant_dict_to_table_with_spread(
    qt_metrics_dict,
    spread_df,
    products_order=None,
):
    rows = []

    for hemi, prod_dict in qt_metrics_dict.items():
        for prod, mets in prod_dict.items():
            rows.append({
                "hemi": hemi,
                "product": prod,
                "CC": mets.get("CC", np.nan),
                "Bias_pct": mets.get("Bias", np.nan),
                "RMSE": mets.get("RMSE", np.nan),
                "MAE": mets.get("MAE", np.nan),
                "N_event_pairs": mets.get("N", np.nan),
            })

    qt_df = pd.DataFrame(rows)

    out = qt_df.merge(
        spread_df[["hemi", "product", "N_pairs", "obs_std", "prod_std", "spread_ratio"]],
        on=["hemi", "product"],
        how="left"
    )

    if products_order is not None:
        out["product"] = pd.Categorical(out["product"], categories=products_order, ordered=True)
        out = out.sort_values(["hemi", "product"]).reset_index(drop=True)

    return out

#------------------------------------------------------------
def plot_quant_summary_cc_bias_spread(
    quant_summary_df,
    *,
    products_order,
    product_colors,
    figsize=(12, 9),
):
    metrics = [
        ("CC", "CC"),
        ("Bias_pct", "Bias [%]"),
        ("spread_ratio", r"Spread ratio [$\sigma_p / \sigma_{OR}$]"),
    ]

    hemis = ["NH", "SH"]

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(hemis),
        figsize=figsize,
        sharex="col",
        squeeze=False
    )

    for j, hemi in enumerate(hemis):
        dfh = quant_summary_df[quant_summary_df["hemi"] == hemi].copy()
        dfh = dfh.set_index("product").reindex(products_order).reset_index()

        x = np.arange(len(products_order))

        for i, (col, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            vals = dfh[col].values.astype(float)

            ax.bar(
                x,
                vals,
                color=[product_colors.get(p, "0.7") for p in products_order],
                edgecolor="black",
                linewidth=0.5
            )

            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

            if col == "CC":
                ax.set_ylim(0, max(0.6, np.nanmax(vals) * 1.15 if np.isfinite(np.nanmax(vals)) else 0.6))
            elif col == "spread_ratio":
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
            elif col == "Bias_pct":
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)

            if i == 0:
                ax.set_title(hemi, fontsize=15, fontweight="bold")

            if i == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(products_order, rotation=25, ha="right", fontsize=11, fontweight="bold")
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.tick_params(axis="y", labelsize=11)

    handles = [Patch(facecolor=product_colors.get(p, "0.7"), edgecolor="black", label=p) for p in products_order]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(products_order), 6),
        frameon=False,
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

#------------------------------------------------------------
def build_oceanrain_descriptive_stats_table(
    df,
    *,
    obs_col="main_mmday",
    product_cols=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    hemi_col="hemi",
):
    rows = []

    for hemi in ["NH", "SH"]:
        dsub = df[df[hemi_col] == hemi].copy()

        datasets = [("OceanRAIN", obs_col)] + [(p, p) for p in product_cols]

        for name, col in datasets:
            s = pd.to_numeric(dsub[col], errors="coerce").dropna()

            if len(s) == 0:
                continue

            rows.append({
                "hemi": hemi,
                "dataset": name,
                # "N": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(ddof=1),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
            })

    out = pd.DataFrame(rows)
    return out

#------------------------------------------------------------
def plot_oceanrain_quant_summary_panel(
    qt_metrics_hemi,
    products,
    product_colors,
    *,
    hemis=("NH", "SH"),
    metrics=("CC", "Bias", "RMSE"),
    figsize=(12, 9),
    ylims=None,
    ylabel_map=None,
    add_zero_line_for_bias=True,
    auto_ylim=True,
    tick_fontsize=12,
    axis_label_fontsize=14,
    title_fontsize=16,
    legend_fontsize=13,
):
    """
    Compact NH/SH quantitative summary panel for OceanRAIN comparison.

    Uses independent y-axis limits for each hemisphere/metric panel unless
    ylims are explicitly supplied.

    ylims can be either:
        {"CC": (0, 0.7), "Bias": (-60, 40), "RMSE": (0, 18)}

    or hemisphere-specific:
        {
            "CC": {"NH": (0, 0.6), "SH": (0, 0.5)},
            "Bias": {"NH": (-60, 5), "SH": (-20, 20)},
            "RMSE": {"NH": (0, 18), "SH": (0, 12)}
        }
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if ylabel_map is None:
        ylabel_map = {
            "CC": "CC",
            "Bias": "Bias [%]",
            "RMSE": "RMSE\n[mm day$^{-1}$]",
            "MAE": "MAE\n[mm day$^{-1}$]",
        }

    nrows = len(metrics)
    ncols = len(hemis)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    x = np.arange(len(products))

    def _auto_ylim(vals, metric):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            return None

        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

        if metric == "CC":
            lo = max(0.0, vmin - 0.08)
            hi = min(1.0, vmax + 0.08)
            if hi <= lo:
                hi = min(1.0, lo + 0.2)
            return lo, hi

        if metric in ["RMSE", "MAE"]:
            lo = 0.0
            hi = vmax * 1.15 if vmax > 0 else 1.0
            return lo, hi

        if metric == "Bias":
            # Always include zero for bias.
            vmin = min(vmin, 0.0)
            vmax = max(vmax, 0.0)
            pad = 0.15 * (vmax - vmin) if vmax > vmin else 5.0
            return vmin - pad, vmax + pad

        pad = 0.12 * (vmax - vmin) if vmax > vmin else 1.0
        return vmin - pad, vmax + pad

    def _get_ylim(metric, hemi, vals):
        if ylims is not None and metric in ylims:
            # Option 1: ylims["CC"] = (0, 0.7)
            if isinstance(ylims[metric], tuple):
                return ylims[metric]

            # Option 2: ylims["CC"]["NH"] = (0, 0.6)
            if isinstance(ylims[metric], dict) and hemi in ylims[metric]:
                return ylims[metric][hemi]

        if auto_ylim:
            return _auto_ylim(vals, metric)

        return None

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for j, hemi in enumerate(hemis):
        for i, met in enumerate(metrics):
            ax = axes[i, j]

            vals = []
            for p in products:
                d = qt_metrics_hemi.get(hemi, {}).get(p, {})
                vals.append(d.get(met, np.nan) if isinstance(d, dict) else np.nan)

            colors = [
                _get_product_color(p, product_colors, default="0.7")
                for p in products
            ]

            ax.bar(
                x,
                vals,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )

            panel_idx = i * len(hemis) + j

            ax.text(
                0.02,
                0.95,
                panel_labels[panel_idx],
                transform=ax.transAxes,
                fontsize=axis_label_fontsize,
                fontweight="bold",
                ha="left",
                va="top",
            )

            if i == 0:
                ax.set_title(
                    hemi,
                    fontsize=title_fontsize,
                    fontweight="bold",
                )

            ax.set_ylabel(
                ylabel_map.get(met, met),
                fontsize=axis_label_fontsize,
                fontweight="bold",
            )

            ax.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.35,
            )

            if met == "Bias" and add_zero_line_for_bias:
                ax.axhline(
                    0,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                )

            panel_ylim = _get_ylim(met, hemi, vals)
            if panel_ylim is not None:
                ax.set_ylim(*panel_ylim)

            ax.tick_params(
                axis="both",
                labelsize=tick_fontsize,
                direction="in",
                top=True,
                right=True,
            )

            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
                tick.set_fontsize(tick_fontsize)

            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    products,
                    rotation=25,
                    ha="right",
                    fontsize=tick_fontsize + 1,
                    fontweight="bold",
                )
            else:
                ax.tick_params(axis="x", labelbottom=False)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_get_product_color(p, product_colors, default="0.7"),
            edgecolor="black",
            label=p,
        )
        for p in products
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(products),
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    return fig, axes

#------------------------------------------------------------
def round_metric_table(df, cols, ndigits=3):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).round(ndigits)
    return out
#------------------------------------------------------------
def categorical_dict_to_table(cat_metrics_dict, products_order=None):
    rows = []
    for hemi, prod_dict in cat_metrics_dict.items():
        for prod, mets in prod_dict.items():
            rows.append({
                "hemi": hemi,
                "product": prod,
                "N": mets.get("Hits", 0) + mets.get("Misses", 0) + mets.get("False_Alarms", 0) + mets.get("Correct_Negatives", 0),
                "POD": mets.get("POD", np.nan),
                "FAR": mets.get("FAR", np.nan),
                "Bias": mets.get("Bias", np.nan),
                "HSS": mets.get("HSS", np.nan),
            })
    out = pd.DataFrame(rows)

    if products_order is not None:
        out["product"] = pd.Categorical(out["product"], categories=products_order, ordered=True)
        out = out.sort_values(["hemi", "product"]).reset_index(drop=True)

    return out
# ------------------------------------------------------------
# daily aggregation using the new OceanRAIN main precip variable
# ------------------------------------------------------------
def oceanrain_daily_aggregate_to_gpcp_main(
    oc_df_minute: pd.DataFrame,
    gpcp_lat_1d,
    gpcp_lon_1d,
    *,
    lat_abs_min=45.0,
    coverage_frac=0.10,      # e.g. 10% of day = 144 minutes
    wind_mean_col_preference=("true_wind_speed", "u10", "rel_wind_speed"),
    keep_only_usable_days=False,
):
    """
    Aggregate QC'd OceanRAIN minute data to daily ship+GPCP-pixel values.

    Intended for the *new* OceanRAIN workflow:
      - primary minute-level precip variable is `rate_main_mmph`
      - `rate_phasebest_mmph` is kept only as a diagnostic
      - phase fractions are retained for interpretation, not primary screening

    Required columns in oc_df_minute:
      time_utc, lat, lon, ship, precip_flag, rate_main_mmph

    Optional useful columns:
      precip_flag2, rate_phasebest_mmph, rate_odm_mmph, rate_gag_mmph,
      true_wind_speed / u10 / rel_wind_speed,
      rain_prob, snow_prob, mixed_prob

    Returns
    -------
    daily_all : pd.DataFrame
        Daily ship+pixel aggregates with coverage, phase fractions, totals, etc.
    usable_days : pd.DataFrame
        Subset passing minimum daily coverage threshold.
    """

    df = oc_df_minute.copy()

    # --------------------------------------------------------
    # 1) basic cleaning / datetime
    # --------------------------------------------------------
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon", "ship"]).copy()

    if "rate_main_mmph" not in df.columns:
        raise ValueError("Input dataframe must contain 'rate_main_mmph'.")

    df["date"] = df["time_utc"].dt.floor("D")

    # poleward subset if desired
    if lat_abs_min is not None:
        df = df[df["lat"].abs() >= float(lat_abs_min)].copy()

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    # --------------------------------------------------------
    # 2) map each minute to nearest GPCP pixel
    # --------------------------------------------------------
    lon_wrapped = ((df["lon"].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0
    df["ilat"] = map_to_gpcp_idx(np.asarray(gpcp_lat_1d), df["lat"].to_numpy(dtype="float64"))
    df["ilon"] = map_to_gpcp_idx(np.asarray(gpcp_lon_1d), lon_wrapped)

    # --------------------------------------------------------
    # 3) choose wind column if available
    # --------------------------------------------------------
    wind_col = None
    for wc in wind_mean_col_preference:
        if wc in df.columns:
            wind_col = wc
            break

    grp_keys = ["date", "ilat", "ilon", "ship"]

    # --------------------------------------------------------
    # 4) minute coverage / phase composition diagnostics
    # --------------------------------------------------------
    # precip_flag meaning:
    # 0 rain, 1 snow, 2 mixed, 3 true-zero, 4 inoperative, 5 harbor
    # step0 QC should already have removed 4 and 5
    daily_cov = (
        df.groupby(grp_keys, as_index=False)
          .agg(
              n_min_total=("precip_flag", "size"),
              n_min_valid_rate=("rate_main_mmph", lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce")))),
              n_min_zero=("precip_flag", lambda s: np.sum(s.to_numpy() == 3)),
              n_min_rain=("precip_flag", lambda s: np.sum(s.to_numpy() == 0)),
              n_min_snow=("precip_flag", lambda s: np.sum(s.to_numpy() == 1)),
              n_min_mixed=("precip_flag", lambda s: np.sum(s.to_numpy() == 2)),
              frac_zero=("precip_flag", lambda s: np.mean(s.to_numpy() == 3)),
              frac_rain=("precip_flag", lambda s: np.mean(s.to_numpy() == 0)),
              frac_snow=("precip_flag", lambda s: np.mean(s.to_numpy() == 1)),
              frac_mixed=("precip_flag", lambda s: np.mean(s.to_numpy() == 2)),
              lat_mean=("lat", "mean"),
              lon_mean=("lon", "mean"),
              wind_mean=(wind_col, "mean") if wind_col is not None else ("lat", "mean"),
          )
    )

    # coverage relative to a full day
    daily_cov["coverage_frac_day"] = daily_cov["n_min_total"] / 1440.0

    # --------------------------------------------------------
    # 5) daily precipitation metrics
    # --------------------------------------------------------
    agg_dict = {
        "main_mean_mmph": ("rate_main_mmph", "mean"),
        "main_median_mmph": ("rate_main_mmph", "median"),
        "main_max_mmph": ("rate_main_mmph", "max"),
        "main_mmday": ("rate_main_mmph", _mmday_from_mmph),
    }

    if "rate_phasebest_mmph" in df.columns:
        agg_dict.update({
            "phasebest_mean_mmph": ("rate_phasebest_mmph", "mean"),
            "phasebest_mmday": ("rate_phasebest_mmph", _mmday_from_mmph),
        })

    if "rate_odm_mmph" in df.columns:
        agg_dict.update({
            "odm_mean_mmph": ("rate_odm_mmph", "mean"),
            "odm_mmday": ("rate_odm_mmph", _mmday_from_mmph),
        })

    if "rate_gag_mmph" in df.columns:
        agg_dict.update({
            "gag_mean_mmph": ("rate_gag_mmph", "mean"),
            "gag_mmday": ("rate_gag_mmph", _mmday_from_mmph),
        })

    if "rain_prob" in df.columns:
        agg_dict["rain_prob_mean"] = ("rain_prob", "mean")
    if "snow_prob" in df.columns:
        agg_dict["snow_prob_mean"] = ("snow_prob", "mean")
    if "mixed_prob" in df.columns:
        agg_dict["mixed_prob_mean"] = ("mixed_prob", "mean")

    daily_rates = (
        df.groupby(grp_keys, as_index=False)
          .agg(**agg_dict)
    )

    # --------------------------------------------------------
    # 6) merge coverage + rates
    # --------------------------------------------------------
    daily_all = daily_cov.merge(daily_rates, on=grp_keys, how="left")

    # --------------------------------------------------------
    # 7) attach target grid-cell centers and hemisphere label
    # --------------------------------------------------------
    g_lat = np.asarray(gpcp_lat_1d)
    g_lon = np.asarray(gpcp_lon_1d)

    daily_all["lat_c"] = g_lat[daily_all["ilat"].to_numpy()]
    daily_all["lon_c"] = g_lon[daily_all["ilon"].to_numpy()]
    daily_all["hemi"] = np.where(daily_all["lat_c"] >= 0, "NH", "SH")

    # --------------------------------------------------------
    # 8) optional convenience diagnostics
    # --------------------------------------------------------
    # "precip-present" minutes = non-zero phase minutes
    daily_all["n_min_precip_phase"] = (
        daily_all["n_min_rain"] + daily_all["n_min_snow"] + daily_all["n_min_mixed"]
    )
    daily_all["frac_precip_phase"] = (
        daily_all["frac_rain"] + daily_all["frac_snow"] + daily_all["frac_mixed"]
    )

    # some users like a dominant phase tag for interpretation only
    phase_cols = ["frac_rain", "frac_snow", "frac_mixed", "frac_zero"]
    phase_names = ["rain", "snow", "mixed", "zero"]

    phase_arr = daily_all[phase_cols].to_numpy(dtype="float64")
    phase_idx = np.nanargmax(phase_arr, axis=1)
    daily_all["dominant_phase"] = np.array(phase_names, dtype=object)[phase_idx]

    # --------------------------------------------------------
    # 9) usable-day filter based on minimum coverage
    # --------------------------------------------------------
    min_minutes = float(coverage_frac) * 1440.0
    usable_days = daily_all[daily_all["n_min_total"] >= min_minutes].copy()

    if keep_only_usable_days:
        daily_all = usable_days.copy()

    return daily_all, usable_days

# -----------------------------------------------------------------------------
# STEP 2A: Add pixel-center coordinates + hemisphere from ilat/ilon
# -----------------------------------------------------------------------------
def add_pixel_coords_and_hemi(
    df: pd.DataFrame,
    lat_grid_1d: np.ndarray,   # e.g., gpcp_ds.lat.values
    lon_grid_1d: np.ndarray,   # e.g., gpcp_ds.lon.values
    *,
    ilat_col="ilat",
    ilon_col="ilon",
    lon_wrap=True,
) -> pd.DataFrame:
    """
    Convert (ilat, ilon) indices -> pixel-center (lat_c, lon_c), and add hemisphere label.

    Works for any table that already contains ilat/ilon (daily_or, snow_days, rain_days).
    """
    out = df.copy()

    ilat = out[ilat_col].to_numpy().astype(int)
    ilon = out[ilon_col].to_numpy().astype(int)

    out["lat_c"] = np.asarray(lat_grid_1d)[ilat]
    out["lon_c"] = np.asarray(lon_grid_1d)[ilon]

    if lon_wrap:
        out["lon_c"] = ((out["lon_c"].to_numpy() + 180.0) % 360.0) - 180.0

    out["hemi"] = np.where(out["lat_c"].to_numpy() >= 0, "NH", "SH")
    return out

# -----------------------------------------------------------------------------
# STEP 2B: Robust xarray dim inference (Dataset or DataArray)
# -----------------------------------------------------------------------------
def _infer_dims(da: xr.DataArray):
    dims = list(da.dims)

    for cand in ["time", "valid_time", "date", "datetime"]:
        if cand in dims:
            tdim = cand
            break
    else:
        raise ValueError(f"Could not infer time dim from {dims}")

    for cand in ["lat", "latitude", "y"]:
        if cand in dims:
            ydim = cand
            break
    else:
        raise ValueError(f"Could not infer lat dim from {dims}")

    for cand in ["lon", "longitude", "x"]:
        if cand in dims:
            xdim = cand
            break
    else:
        raise ValueError(f"Could not infer lon dim from {dims}")

    return tdim, ydim, xdim

def _to_naive_datetime64ns(x):
    """Convert anything datetime-like (including tz-aware) to naive UTC datetime64[ns]."""
    s = pd.to_datetime(x, errors="coerce", utc=True)
    # utc=True ensures tz-aware; now strip tz to make naive
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")

def _ensure_xr_time_naive(da: xr.DataArray, tdim: str) -> xr.DataArray:
    """Ensure da[tdim] is naive datetime64[ns]."""
    t = pd.to_datetime(da[tdim].values, errors="coerce", utc=True)
    t = t.tz_convert("UTC").tz_localize(None)
    return da.assign_coords({tdim: t.to_numpy(dtype="datetime64[ns]")})

#-----------------------------------------------------------------------------
def ensure_pixel_coords_and_hemi(
    df: pd.DataFrame,
    gpcp_lat_1d,
    gpcp_lon_1d,
    *,
    ilat_col="ilat",
    ilon_col="ilon",
    latc_col="lat_c",
    lonc_col="lon_c",
    hemi_col="hemi",
):
    out = df.copy()

    g_lat = np.asarray(gpcp_lat_1d)
    g_lon = np.asarray(gpcp_lon_1d)

    if latc_col not in out.columns:
        out[latc_col] = g_lat[out[ilat_col].to_numpy()]
    if lonc_col not in out.columns:
        out[lonc_col] = g_lon[out[ilon_col].to_numpy()]
    if hemi_col not in out.columns:
        out[hemi_col] = np.where(out[latc_col] >= 0, "NH", "SH")

    return out

#-----------------------------------------------------------------------------
def attach_daily_vars(
    df: pd.DataFrame,
    xr_obj,                       # xr.Dataset or xr.DataArray
    *,
    var_map: dict,                # {"new_col": "xr_varname"}; for DataArray: {"new_col": None}
    date_col="date",
    lat_col="lat_c",
    lon_col="lon_c",
    lon_wrap=True,
    method="nearest",             # "nearest" is fine; can be "exact" if you prefer
):
    """
    Attach DAILY gridded values from xr_obj onto df using (date, lat_c, lon_c).
    Assumes products are already harmonized to GPCP grid, so coords should match.
    """
    out = df.copy()
    out[date_col] = _to_naive_datetime64ns(out[date_col])

    # Build DataArrays
    if isinstance(xr_obj, xr.DataArray):
        if len(var_map) < 1:
            raise ValueError("For DataArray input, var_map must have at least one output column.")
        only_key = next(iter(var_map.keys()))
        das = {only_key: xr_obj}
    elif isinstance(xr_obj, xr.Dataset):
        das = {new_col: xr_obj[varname] for new_col, varname in var_map.items()}
    else:
        raise TypeError("xr_obj must be xr.Dataset or xr.DataArray")

    da0 = next(iter(das.values()))
    tdim, ydim, xdim = _infer_dims(da0)

    # vector indexers
    latv = out[lat_col].to_numpy(dtype="float64")
    lonv = out[lon_col].to_numpy(dtype="float64")
    if lon_wrap:
        lonv = ((lonv + 180.0) % 360.0) - 180.0

    t_indexer = xr.DataArray(out[date_col], dims="points")
    y_indexer = xr.DataArray(latv, dims="points")
    x_indexer = xr.DataArray(lonv, dims="points")

    for new_col, da in das.items():
        da = _ensure_xr_time_naive(da, tdim)
        vals = da.sel({tdim: t_indexer, ydim: y_indexer, xdim: x_indexer}, method=method).values
        out[new_col] = vals

    return out

def _ensure_datetime_coord_naive(da: xr.DataArray, tdim: str) -> xr.DataArray:
    """
    Ensure da[tdim] is datetime64[ns] *naive*.
    """
    tvals = da[tdim].values
    # pd.to_datetime handles cftime-ish + numpy datetimes well in most cases
    t = pd.to_datetime(tvals, errors="coerce")
    # if tz-aware, strip it to naive
    if isinstance(t, pd.DatetimeIndex) and t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    t64 = t.to_numpy(dtype="datetime64[ns]")
    return da.assign_coords({tdim: t64})

#----------------------------------------------------------------------------
def attach_satellite_vars(
    df: pd.DataFrame,
    xr_obj,
    *,
    var_map: dict,
    date_col="date",
    lat_col="lat_c",
    lon_col="lon_c",
    lon_wrap=True,
    method="nearest",
    tolerance_time=None,
) -> pd.DataFrame:
    """
    Attach values from xr_obj to df rows using (date, lat_c, lon_c).

    Robust behavior:
    - time tolerance applied only to time dimension
    - rows with unmatched times get NaN instead of crashing
    """

    out = df.copy()
    out[date_col] = _to_naive_datetime64ns(out[date_col])

    if isinstance(xr_obj, xr.DataArray):
        if len(var_map) < 1:
            raise ValueError("For DataArray input, var_map must have at least one output column.")
        only_key = next(iter(var_map.keys()))
        das = {only_key: xr_obj}
    elif isinstance(xr_obj, xr.Dataset):
        das = {}
        for new_col, varname in var_map.items():
            if varname is None:
                if len(xr_obj.data_vars) != 1:
                    raise ValueError(
                        f"For Dataset input with var_map[{new_col}] = None, dataset must have exactly one data variable."
                    )
                only_var = list(xr_obj.data_vars)[0]
                das[new_col] = xr_obj[only_var]
            else:
                das[new_col] = xr_obj[varname]
    else:
        raise TypeError("xr_obj must be xr.Dataset or xr.DataArray")

    da0 = next(iter(das.values()))
    tdim, ydim, xdim = _infer_dims(da0)

    latv = out[lat_col].to_numpy(dtype="float64")
    lonv = out[lon_col].to_numpy(dtype="float64")
    if lon_wrap:
        lonv = ((lonv + 180.0) % 360.0) - 180.0

    for new_col, da in das.items():
        da = _ensure_datetime_coord_naive(da, tdim)

        # initialize output with NaN
        vals_out = np.full(len(out), np.nan, dtype="float64")

        # determine which rows fall inside product time range
        tcoord = pd.to_datetime(da[tdim].values, errors="coerce")
        tmin = tcoord.min().to_datetime64()
        tmax = tcoord.max().to_datetime64()

        valid_time_mask = (out[date_col].to_numpy() >= tmin) & (out[date_col].to_numpy() <= tmax)

        if valid_time_mask.any():
            sub = out.loc[valid_time_mask].copy()

            t_indexer = xr.DataArray(sub[date_col].to_numpy().astype("datetime64[ns]"), dims="points")
            y_indexer = xr.DataArray(sub[lat_col].to_numpy(dtype="float64"), dims="points")
            xvals = sub[lon_col].to_numpy(dtype="float64")
            if lon_wrap:
                xvals = ((xvals + 180.0) % 360.0) - 180.0
            x_indexer = xr.DataArray(xvals, dims="points")

            try:
                # time selection first
                if tolerance_time is None:
                    da_t = da.sel({tdim: t_indexer}, method=method)
                else:
                    da_t = da.sel({tdim: t_indexer}, method=method, tolerance=tolerance_time)

                # then space
                vals = da_t.sel({ydim: y_indexer, xdim: x_indexer}, method=method).values

                vals_out[valid_time_mask] = np.asarray(vals).reshape(-1)

            except KeyError:
                # fallback: leave unmatched rows as NaN
                pass

        out[new_col] = vals_out

    return out

#-----------------------------------------------------------------------------
def subset_to_common_time_range(
    df: pd.DataFrame,
    products: dict,
    *,
    date_col="date",
):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").values.astype("datetime64[ns]")

    prod_mins = []
    prod_maxs = []

    for name, (xr_obj, var_map) in products.items():
        if isinstance(xr_obj, xr.DataArray):
            da0 = xr_obj
        else:
            # pick first mapped variable with a real variable name
            first_var = next(iter(var_map.values()))
            if first_var is None:
                # fallback: first data var in dataset
                first_var = list(xr_obj.data_vars)[0]
            da0 = xr_obj[first_var]

        tdim, _, _ = _infer_dims(da0)
        t = pd.to_datetime(da0[tdim].values, errors="coerce")
        t = pd.DatetimeIndex(t).tz_localize(None) if getattr(t, "tz", None) is not None else pd.DatetimeIndex(t)

        prod_mins.append(t.min().to_datetime64())
        prod_maxs.append(t.max().to_datetime64())

    common_start = max(prod_mins)
    common_end = min(prod_maxs)

    out = out[(out[date_col] >= common_start) & (out[date_col] <= common_end)].copy()

    return out, common_start, common_end

# -----------------------------------------------------------------------------
# STEP 2D: Example usage (NH vs SH)
# -----------------------------------------------------------------------------
# Assume you already have:
#   - snow_days, rain_days  (from your Step 1 aggregation)
#   - gpcp_ds_v3pt2_al (or any product on the same 0.5° grid)
# and you want to attach GPCP precip + pliq (Dataset case)

def step2_attach_all_products_example(
    snow_days: pd.DataFrame,
    rain_days: pd.DataFrame,
    gpcp_ds: xr.Dataset,
    *,
    gpcp_precip_var="precip",
    gpcp_pliq_var="probability_liquid_phase",
):
    # 1) add pixel centers + hemisphere
    snow_days2 = add_pixel_coords_and_hemi(
        snow_days, gpcp_ds["lat"].values, gpcp_ds["lon"].values
    )
    rain_days2 = add_pixel_coords_and_hemi(
        rain_days, gpcp_ds["lat"].values, gpcp_ds["lon"].values
    )

    # 2) attach satellite vars
    snow_days2 = attach_satellite_vars(
        snow_days2,
        gpcp_ds,
        var_map={"gpcp_mmday": gpcp_precip_var, "gpcp_pliq": gpcp_pliq_var},
        date_col="date",
        lat_col="lat_c",
        lon_col="lon_c",
        method="nearest",
        tolerance_time=np.timedelta64(12, "h"),  # optional safety
    )
    rain_days2 = attach_satellite_vars(
        rain_days2,
        gpcp_ds,
        var_map={"gpcp_mmday": gpcp_precip_var, "gpcp_pliq": gpcp_pliq_var},
        date_col="date",
        lat_col="lat_c",
        lon_col="lon_c",
        method="nearest",
        tolerance_time=np.timedelta64(12, "h"),
    )

    # 3) split NH/SH
    snow_NH = snow_days2[snow_days2["hemi"] == "NH"].copy()
    snow_SH = snow_days2[snow_days2["hemi"] == "SH"].copy()
    rain_NH = rain_days2[rain_days2["hemi"] == "NH"].copy()
    rain_SH = rain_days2[rain_days2["hemi"] == "SH"].copy()

    return snow_days2, rain_days2, (snow_NH, snow_SH, rain_NH, rain_SH)

def step2_attach_and_split(
    snow_days: pd.DataFrame,
    rain_days: pd.DataFrame,
    *,
    gpcp_grid_ds: xr.Dataset,            # used ONLY for lat/lon arrays
    products: dict,                      # {"name": (xr_obj, var_map)}
    ilat_col="ilat",
    ilon_col="ilon",
    date_col="date",
):
    """
    products example:
      {
        "GPCP": (gpcp_ds, {"gpcp_mmday":"precip", "gpcp_pliq":"probability_liquid_phase"}),
        "ERA5": (era5_ds, {"era5_mmday":"tp"}),
        "IMERG": (imerg_da, {"imerg_mmday": None}),   # if DataArray
      }
    """
    g_lat = gpcp_grid_ds["lat"].values
    g_lon = gpcp_grid_ds["lon"].values

    # add pixel coords + hemi
    snow2 = add_pixel_coords_and_hemi(snow_days, g_lat, g_lon, ilat_col=ilat_col, ilon_col=ilon_col)
    rain2 = add_pixel_coords_and_hemi(rain_days, g_lat, g_lon, ilat_col=ilat_col, ilon_col=ilon_col)

    # attach all products
    for _, (xr_obj, var_map) in products.items():
        snow2 = attach_daily_vars(snow2, xr_obj, var_map=var_map, date_col=date_col, lat_col="lat_c", lon_col="lon_c")
        rain2 = attach_daily_vars(rain2, xr_obj, var_map=var_map, date_col=date_col, lat_col="lat_c", lon_col="lon_c")

    # split NH/SH
    snow_NH = snow2[snow2["hemi"] == "NH"].copy()
    snow_SH = snow2[snow2["hemi"] == "SH"].copy()
    rain_NH = rain2[rain2["hemi"] == "NH"].copy()
    rain_SH = rain2[rain2["hemi"] == "SH"].copy()

    return snow2, rain2, (snow_NH, snow_SH, rain_NH, rain_SH)

# -----------------------------------------------------------------------------
def step2_attach_products_oceanrain(
    daily_or_df: pd.DataFrame,
    *,
    gpcp_grid_ds: xr.Dataset,
    products,
    date_col="date",
    ilat_col="ilat",
    ilon_col="ilon",
    lat_col="lat_c",
    lon_col="lon_c",
    ref_col="main_mmday",
    method="nearest",
    tolerance_time=None,
):
    out = daily_or_df.copy()

    print("INSIDE function - type(products):", type(products))

    if ref_col not in out.columns:
        raise ValueError(f"Expected OceanRAIN reference column '{ref_col}' not found.")

    out = ensure_pixel_coords_and_hemi(
        out,
        gpcp_grid_ds["lat"].values,
        gpcp_grid_ds["lon"].values,
        ilat_col=ilat_col,
        ilon_col=ilon_col,
        latc_col=lat_col,
        lonc_col=lon_col,
        hemi_col="hemi",
    )

    # robust iteration
    if isinstance(products, dict):
        iterable = products.items()
    elif isinstance(products, list):
        iterable = products
    else:
        raise TypeError(f"`products` must be dict or list, got {type(products)}")

    for item in iterable:
        if isinstance(products, dict):
            name, (xr_obj, var_map) = item
        else:
            name, xr_obj, var_map = item

        print("Attaching:", name)

        out = attach_satellite_vars(
            out,
            xr_obj,
            var_map=var_map,
            date_col=date_col,
            lat_col=lat_col,
            lon_col=lon_col,
            method=method,
            tolerance_time=tolerance_time,
        )

    return out

#-----------------------------------------------------------------------------
def split_oceanrain_attached_by_hemi(
    df: pd.DataFrame,
    *,
    hemi_col="hemi",
):
    df_nh = df[df[hemi_col] == "NH"].copy()
    df_sh = df[df[hemi_col] == "SH"].copy()
    return df_nh, df_sh 

# -----------------------------------------------------------------------------
def subset_valid_pairs(
    df: pd.DataFrame,
    *,
    ref_col="main_mmday",
    product_cols=None,
):
    out = df.copy()

    if product_cols is None:
        product_cols = []

    needed = [ref_col] + list(product_cols)
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not found in dataframe.")

    return out.dropna(subset=needed).copy()
#-----------------------------------------------------------------------------
def build_metric_table(phase_based_cat_metrics_hemi, products, metrics=("POD","FAR","Bias","HSS")):
    tbl = {}
    for hemi in ["SH", "NH"]:
        tbl.setdefault(hemi, {})
        for met in metrics:
            tbl[hemi].setdefault(met, {"Rain": {}, "Snow": {}})

    for hemi in ["NH", "SH"]:
        for phse in ["Rain", "Snow"]:
            for product in products:
                d = phase_based_cat_metrics_hemi[hemi][phse][product]
                for met in metrics:
                    tbl[hemi][met][phse][product] = d.get(met, np.nan)
    return tbl

#-----------------------------------------------------------------------------

def build_oceanrain_ship_year_month_means(
    df: pd.DataFrame,
    *,
    obs_col="main_mmday",
    product_cols=None,
    date_col="date",
    hemi_col="hemi",
    ship_col="ship",
    min_days_per_month=1,
):
    """
    Build monthly mean daily rainfall from daily collocations.

    Each output row = one (hemi, ship, year, month) point.

    Parameters
    ----------
    df : pd.DataFrame
        Usually daily_or_attached.
    obs_col : str
        OceanRAIN reference column, e.g. 'main_mmday'.
    product_cols : list[str]
        Product columns to average.
    min_days_per_month : int
        Minimum number of valid OceanRAIN daily values required in a ship-year-month bin.
    """

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, hemi_col, ship_col, obs_col]).copy()

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month

    if product_cols is None:
        product_cols = []

    grp = [hemi_col, ship_col, "year", "month"]

    agg_dict = {
        obs_col: (obs_col, "mean"),
        "n_days_obs": (obs_col, lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce")))),
    }

    for p in product_cols:
        if p in out.columns:
            agg_dict[p] = (p, "mean")
            agg_dict[f"n_days_{p}"] = (p, lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce"))))

    monthly = out.groupby(grp, as_index=False).agg(**agg_dict)

    monthly = monthly[monthly["n_days_obs"] >= int(min_days_per_month)].copy()

    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly["month_name"] = monthly["month"].map(month_map)

    # nice label if you want to annotate later
    monthly["year_month"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)

    return monthly.sort_values([hemi_col, ship_col, "year", "month"]).reset_index(drop=True)
#-----------------------------------------------------------------------------
def build_oceanrain_ship_month_climatology(
    df: pd.DataFrame,
    *,
    obs_col="main_mmday",
    product_cols=None,
    min_days_per_ship_month=10,
    date_col="date",
    hemi_col="hemi",
    ship_col="ship",
):
    """
    Build multi-year mean monthly daily-rainfall climatology by ship.

    Each output row = one (hemi, ship, month) climatological point.

    Parameters
    ----------
    df : pd.DataFrame
        Usually daily_or_attached.
    obs_col : str
        OceanRAIN reference column, e.g. 'main_mmday'.
    product_cols : list[str]
        Satellite/reanalysis columns to average alongside OceanRAIN.
    min_days_per_ship_month : int
        Minimum number of valid daily collocations required for a ship-month bin.
    """

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, hemi_col, ship_col, obs_col]).copy()
    out["month"] = out[date_col].dt.month

    if product_cols is None:
        product_cols = []

    # only keep needed cols
    use_cols = [date_col, hemi_col, ship_col, "month", obs_col] + product_cols
    use_cols = [c for c in use_cols if c in out.columns]
    out = out[use_cols].copy()

    # helper to compute valid-day counts per product
    grp = [hemi_col, ship_col, "month"]

    # OceanRAIN count
    obs_count = (
        out.groupby(grp, as_index=False)
           .agg(n_days_obs=(obs_col, lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce")))))
    )

    # mean OceanRAIN climatology
    obs_mean = (
        out.groupby(grp, as_index=False)
           .agg(**{obs_col: (obs_col, "mean")})
    )

    # product means + valid counts
    prod_frames = []
    for p in product_cols:
        if p not in out.columns:
            continue

        tmp = (
            out.groupby(grp, as_index=False)
               .agg(
                   **{
                       p: (p, "mean"),
                       f"n_days_{p}": (p, lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce"))))
                   }
               )
        )
        prod_frames.append(tmp)

    # merge all pieces
    clim = obs_mean.merge(obs_count, on=grp, how="left")

    for tmp in prod_frames:
        clim = clim.merge(tmp, on=grp, how="left")

    # retain only ship-month bins with enough OceanRAIN days
    clim = clim[clim["n_days_obs"] >= int(min_days_per_ship_month)].copy()

    # month labels
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    clim["month_name"] = clim["month"].map(month_map)

    # optional: total valid pairs across all years for each ship-month
    # this helps later for annotation / filtering
    return clim.sort_values([hemi_col, ship_col, "month"]).reset_index(drop=True)
# -----------------------------------------------------------------------------
def compute_metrics_from_ship_month_climatology(
    ship_month_clim: pd.DataFrame,
    *,
    products,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
):
    """
    Compute quantitative metrics using ship-month climatology points.
    """

    metrics_hemi = {}

    for hemi in hemis:
        dfh = ship_month_clim[ship_month_clim["hemi"] == hemi].copy()
        metrics_hemi[hemi] = {}

        print(f"\nHemisphere = {hemi}, total ship-month points = {len(dfh)}")

        for p in products:
            sub = dfh[[obs_col, p]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            print(f"  {p}: N = {len(sub)}")

            if len(sub) == 0:
                metrics_hemi[hemi][p] = {
                    "CC": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "RB": np.nan,
                    "N": 0,
                }
                continue

            qt = calculate_metrics(sub, obs_col, p)

            # add relative bias if your calculate_metrics does not already
            if isinstance(qt, dict):
                if "RB" not in qt:
                    obs_mean = sub[obs_col].mean()
                    pred_mean = sub[p].mean()
                    qt["RB"] = np.nan if obs_mean == 0 else 100.0 * (pred_mean - obs_mean) / obs_mean
                qt["N"] = len(sub)

            metrics_hemi[hemi][p] = qt

    return metrics_hemi
# ------------------------------------------------------------
def compute_metrics_from_ship_year_month_means(
    monthly_df: pd.DataFrame,
    *,
    products,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
):
    metrics_hemi = {}

    for hemi in hemis:
        dfh = monthly_df[monthly_df["hemi"] == hemi].copy()
        metrics_hemi[hemi] = {}

        print(f"\nHemisphere = {hemi}, total monthly points = {len(dfh)}")

        for p in products:
            sub = dfh[[obs_col, p]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            print(f"  {p}: N = {len(sub)}")

            if len(sub) == 0:
                metrics_hemi[hemi][p] = {
                    "CC": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "RB": np.nan,
                    "N": 0,
                }
                continue

            qt = calculate_metrics(sub, obs_col, p)

            if isinstance(qt, dict):
                if "RB" not in qt:
                    obs_mean = sub[obs_col].mean()
                    pred_mean = sub[p].mean()
                    qt["RB"] = np.nan if obs_mean == 0 else 100.0 * (pred_mean - obs_mean) / obs_mean
                qt["N"] = len(sub)

            metrics_hemi[hemi][p] = qt

    return metrics_hemi
# ------------------------------------------------------------
# 1) Build monthly-climatology-based anomalies for each region
# ------------------------------------------------------------
def make_region_monthly_anomaly_df(
    df,
    products,
    region_col="region",
    date_col="date",
):
    """
    Create anomaly columns for each product using region-specific monthly climatology:
        anomaly = daily value - region/month mean

    Returns
    -------
    df_anom : DataFrame
        Original df plus columns like:
        rain_rate_anom, GPCP v3.2_anom, ...
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["month"] = d[date_col].dt.month

    # region-month climatology for each product
    clim = (
        d.groupby([region_col, "month"])[products]
         .mean()
         .reset_index()
    )

    # merge climatology back
    d = d.merge(
        clim,
        on=[region_col, "month"],
        suffixes=("", "_clim"),
        how="left"
    )

    # create anomaly columns
    for p in products:
        d[f"{p}_anom"] = d[p] - d[f"{p}_clim"]

    return d

# -----------------------------------------------------------------------------
def make_monthly_region_series(df, products, date_col="date", region_col="region"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["month_start"] = d[date_col].dt.to_period("M").dt.to_timestamp()

    out = (
        d.groupby([region_col, "month_start"])[products]
         .mean()
         .reset_index()
         .sort_values([region_col, "month_start"])
    )
    return out

# -----------------------------------------------------------------------------
def deseasonalize_monthly(df_monthly, products, time_col="month_start", region_col="region"):
    d = df_monthly.copy()
    d["month"] = pd.to_datetime(d[time_col]).dt.month

    clim = (
        d.groupby([region_col, "month"])[products]
         .mean()
         .reset_index()
    )

    d = d.merge(clim, on=[region_col, "month"], suffixes=("", "_clim"))

    for p in products:
        d[f"{p}_anom"] = d[p] - d[f"{p}_clim"]

    return d
#-----------------------------------------------------------------------------
def make_doy_climatology_pooled(
    df_daily: pd.DataFrame,
    *,
    date_col: str = "date",
    obs_col: str,
    product_cols: list,
    min_pairs: int = 1,           # <-- set to 1 (or 2/3) for your sparse ship sampling
    drop_feb29: bool = True,
    shift_leap_after_feb28: bool = True,
    reindex_365: bool = False,    # if True -> returns 365 rows (NaNs where missing)
):
    d = df_daily.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d[d[date_col].notna()].copy()

    # DOY with leap-handling
    doy = d[date_col].dt.dayofyear
    is_leap = d[date_col].dt.is_leap_year
    feb29 = (d[date_col].dt.month == 2) & (d[date_col].dt.day == 29)

    if drop_feb29:
        d = d.loc[~feb29].copy()
        doy = doy.loc[~feb29]

    if shift_leap_after_feb28 and drop_feb29:
        after_feb28 = (d[date_col].dt.month > 2) & is_leap.loc[d.index]
        doy = doy.copy()
        doy.loc[after_feb28] = doy.loc[after_feb28] - 1

    d["doy"] = doy.astype(int)

    # Means per DOY (pooled across ships/years/hemis)
    mean_cols = [obs_col] + product_cols
    clim = d.groupby("doy", as_index=False)[mean_cols].mean()

    # Enforce minimum sample count per DOY based on obs availability
    n_obs = d.groupby("doy")[obs_col].apply(lambda s: s.notna().sum()).rename("n_obs").reset_index()
    clim = clim.merge(n_obs, on="doy", how="left")
    clim = clim[clim["n_obs"] >= min_pairs].drop(columns=["n_obs"]).copy()

    # Optional: force 365 rows
    if reindex_365:
        clim = clim.set_index("doy").reindex(range(1, 366)).reset_index()

    return clim
#%% THE PLOT FUNCTIONS
def build_metric_table(phase_based_cat_metrics_hemi, products, metrics=("POD","FAR","Bias","HSS")):
    tbl = {}
    for hemi in ["SH", "NH"]:
        tbl.setdefault(hemi, {})
        for met in metrics:
            tbl[hemi].setdefault(met, {"Rain": {}, "Snow": {}})

    for hemi in ["NH", "SH"]:
        for phse in ["Rain", "Snow"]:
            for product in products:
                d = phase_based_cat_metrics_hemi[hemi][phse][product]
                for met in metrics:
                    tbl[hemi][met][phse][product] = d.get(met, np.nan)
    return tbl
# -----------------------------------------------------------------------------
def split_metrics_by_hemi(df_phase, phase_name, products, obs_col, thr_mmday=1.0):
    """
    Returns:
      cat_hemi[hemi][product] -> cat metric dict
      qt_hemi[hemi][product]  -> quant metric dict
    Requires:
      df_phase has columns: ['hemi', obs_col] + product columns
      categorical_stats(forecast, observed, thr)
      calculate_metrics(df, obs_col, product)
    """
    cat_hemi = {h: {} for h in ["SH", "NH"]}
    qt_hemi  = {h: {} for h in ["SH", "NH"]}

    for hemi in ["SH", "NH"]:
        dH = df_phase[df_phase["hemi"] == hemi].copy()

        for product in products:
            f = dH[product]
            o = dH[obs_col]

            cat_hemi[hemi][product] = categorical_stats(f, o, thr_mmday)
            qt_hemi[hemi][product]  = calculate_metrics(dH, obs_col, product)

    return cat_hemi, qt_hemi

# -----------------------------------------------------------------------------

def plot_phase_cat_metrics_NH_SH(
    cat_hemi,
    products,
    product_colors,
    *,
    phase_label="Rain",
    metrics=("POD", "FAR", "Bias", "HSS"),
    hemis=("SH", "NH"),
    figsize=(12, 10),
    legend_ncol=None
):
    """
    cat_hemi[hemi][product][metric] -> value
    Produces: rows=metrics, cols=hemis, x=products
    """
    if legend_ncol is None:
        legend_ncol = len(products)

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=len(hemis),
        figsize=figsize, sharex=True
    )

    if len(metrics) == 1 and len(hemis) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = np.array([axes])
    elif len(hemis) == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(products))
    width = 0.75  # single bars per product

    for c, hemi in enumerate(hemis):
        for r, met in enumerate(metrics):
            ax = axes[r, c]

            vals = [cat_hemi.get(hemi, {}).get(p, {}).get(met, np.nan) for p in products]
            colors = [product_colors.get(p, None) for p in products]

            ax.bar(x, vals, width=width, color=colors)

            ax.set_ylabel(met)
            ax.grid(True, axis="y", alpha=0.3)

            if r == 0:
                ax.set_title(f"{phase_label} – {hemi}")

            if r == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(products, rotation=25, ha="right")

    # Legend once (top)
    handles = [plt.Rectangle((0, 0), 1, 1, color=product_colors.get(p, None)) for p in products]
    fig.legend(handles, products, loc="upper center", ncol=legend_ncol, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
# -----------------------------------------------------------------------------

def plot_oceanrain_monthly_ship_scatter_by_hemi(
    monthly_df: pd.DataFrame,
    *,
    hemi,
    products,
    obs_col="main_mmday",
    ship_col="ship",
    ship_colors=None,
    ship_markers=None,
    figsize=(15, 7.5),
    xylim=(0, 20),
    ncols=3,
    add_titles=False,
):
    ships = sorted(monthly_df[ship_col].dropna().unique())

    if ship_colors is None:
        cmap = plt.get_cmap("tab10")
        ship_colors = {s: cmap(i % 10) for i, s in enumerate(ships)}

    if ship_markers is None:
        marker_list = ["o", "s", "^", "D", "P", "X", "v", "*"]
        ship_markers = {s: marker_list[i % len(marker_list)] for i, s in enumerate(ships)}

    sub_hemi = monthly_df[monthly_df["hemi"] == hemi].copy()

    n = len(products)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    xmin, xmax = xylim
    ymin, ymax = xylim

    for i, product in enumerate(products):
        ax = axes[i]

        sub = sub_hemi[[obs_col, product, ship_col, "year_month"]].replace([np.inf, -np.inf], np.nan).dropna()

        for ship in ships:
            ss = sub[sub[ship_col] == ship]
            if len(ss) == 0:
                continue

            ax.scatter(
                ss[obs_col], ss[product],
                color=ship_colors[ship],
                marker=ship_markers[ship],
                s=70,
                edgecolor="k",
                linewidth=0.45,
                alpha=0.9,
                label=ship
            )

        ax.plot([xmin, xmax], [ymin, ymax], "--", color="0.55", lw=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if len(sub) > 0:
            met = calculate_metrics(sub, obs_col, product)

            txt = (
                f"RB: {met['Bias']:.1f}%\n"
                f"RMSE: {met['RMSE']:.2f} mm/day\n"
                f"CC: {met['CC']:.2f}\n"
                f"N: {len(sub)}"
            )

            ax.text(
                0.04, 0.94, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=11, fontweight="bold"
            )

        if add_titles:
            ax.set_title(product, fontsize=16)

        ax.set_xlabel("OceanRAIN monthly mean [mm day$^{-1}$]", fontsize=13)
        ax.set_ylabel(f"{product} [mm day$^{-1}$]", fontsize=13)
        ax.grid(True, alpha=0.28)

    for j in range(len(products), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(len(by_label), 5),
        frameon=False,
        fontsize=13,
        handletextpad=0.5,
        columnspacing=1.5
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
#-----------------------------------------------------------------------------
# def plot_hemi_phase_metrics_barpanel(
#     phase_based_cat_metrics_hemi,
#     products,
#     product_colors,
#     *,
#     metrics=("POD", "FAR", "Bias", "HSS"),
#     phases=("Rain", "Snow"),
#     hemis=("SH", "NH"),
# ):
#     tbl = build_metric_table(phase_based_cat_metrics_hemi, products, metrics=metrics)

#     fig, axes = plt.subplots(
#         nrows=len(metrics), ncols=len(hemis),
#         figsize=(12, 10),
#         sharex=True
#     )

#     x = np.arange(len(phases))
#     width = 0.14
#     offsets = (np.arange(len(products)) - (len(products) - 1) / 2) * width

#     for c, hemi in enumerate(hemis):
#         for r, met in enumerate(metrics):
#             ax = axes[r, c]

#             for j, product in enumerate(products):
#                 vals = [tbl[hemi][met][ph][product] for ph in phases]
#                 ax.bar(
#                     x + offsets[j],
#                     vals,
#                     width=width,
#                     label=product,
#                     color=product_colors.get(product, None)  # <- uses your mapping
#                 )

#             ax.set_ylabel(met)
#             ax.grid(True, axis="y", alpha=0.3)

#             if r == 0:
#                 ax.set_title(f"{hemi} metrics")

#             if r == len(metrics) - 1:
#                 ax.set_xticks(x)
#                 ax.set_xticklabels(phases)

#     # One legend for whole figure (keep your product order)
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=len(products), frameon=False)

#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     return fig

#-----------------------------------------------------------------------------


def plot_hemi_phase_metrics_barpanel(
    phase_based_cat_metrics_hemi,
    products,
    product_colors,
    *,
    metrics=("POD", "FAR", "Bias", "HSS"),
    phases=("Rain", "Snow"),
    hemis=("SH", "NH"),
    figsize=(12, 10),
    ylims=None,
):
    """
    Layout:
      - columns = phases (Rain, Snow)
      - rows    = metrics
      - x-axis  = products
      - bars    = hemispheres (SH vs NH) for each product

    Requires build_metric_table(...) -> tbl[hemi][metric][phase][product] -> value
    """

    # Hemisphere styling (same product color, different appearance)
    hemi_style = {
        "SH": {"alpha": 0.95, "hatch": None},
        "NH": {"alpha": 0.55, "hatch": "///"},
    }

    tbl = build_metric_table(phase_based_cat_metrics_hemi, products, metrics=metrics)

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=len(phases),
        figsize=figsize,
        sharex=True
    )

    # ensure 2D axes array
    if len(metrics) == 1 and len(phases) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = np.array([axes])
    elif len(phases) == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(products))  # products on x
    width = 0.35                  # enough for 2 hemis
    offsets = (np.arange(len(hemis)) - (len(hemis) - 1) / 2) * width  # [-w/2, +w/2]

    for c, ph in enumerate(phases):         # columns = phases
        for r, met in enumerate(metrics):   # rows = metrics
            ax = axes[r, c]

            # ---- plot BOTH hemispheres (this must be inside the loop) ----
            for j, hemi in enumerate(hemis):
                vals = [tbl[hemi][met][ph][p] for p in products]
                style = hemi_style.get(hemi, {"alpha": 0.9, "hatch": None})

                ax.bar(
                    x + offsets[j],
                    vals,
                    width=width,
                    color=[product_colors.get(p, None) for p in products],
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    edgecolor="k" if style["hatch"] else "none",
                    linewidth=0.3 if style["hatch"] else 0.0,
                )

            ax.set_ylabel(met, fontsize=14)
            ax.grid(True, axis="y", alpha=0.3)

            if ylims is not None and met in ylims:
                ax.set_ylim(*ylims[met])

            if r == 0:
                ax.set_title(ph)

            if r == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(products, rotation=25, ha="right")

    # ---- Product legend (colors) ----
    prod_handles = [
        mpatches.Patch(facecolor=product_colors.get(p, "0.7"), label=p)
        for p in products
    ]
    fig.legend(
        handles=prod_handles,
        labels=products,
        loc="upper center",
        ncol=len(products),
        frameon=False,
        fontsize=12
    )

    # ---- Hemisphere legend (style) ----
    hemi_handles = [
        mpatches.Patch(facecolor="0.3",
                       alpha=hemi_style["SH"]["alpha"],
                       hatch=hemi_style["SH"]["hatch"],
                       label="SH"),
        mpatches.Patch(facecolor="0.3",
                       alpha=hemi_style["NH"]["alpha"],
                       hatch=hemi_style["NH"]["hatch"],
                       label="NH"),
    ]
    axes[0, 1].legend(
        handles=hemi_handles,
        title="Hemi",
        loc="upper left",
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
#-----------------------------------------------------------------------------

def plot_oceanrain_ship_month_scatter(
    ship_month_clim: pd.DataFrame,
    *,
    products,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
    ship_colors=None,
    ship_markers=None,
    figsize=(14, 10),
    xlim=None,
    ylim=None,
):
    """
    Multi-panel scatter:
      rows/cols over products
      columns show products
      within each panel: NH and SH combined? No.
    
    Here we do:
      one subplot per product
      points colored/marked by ship
      NH and SH shown in separate rows
    """

    nprod = len(products)
    ncols = 3
    nrows = int(np.ceil(nprod / ncols))

    fig, axes = plt.subplots(
        nrows=nrows * len(hemis),
        ncols=ncols,
        figsize=figsize,
        squeeze=False
    )

    ships = sorted(ship_month_clim["ship"].dropna().unique())

    if ship_colors is None:
        cmap = plt.get_cmap("tab10")
        ship_colors = {s: cmap(i % 10) for i, s in enumerate(ships)}

    if ship_markers is None:
        marker_list = ["o", "s", "^", "D", "P", "X", "v", "*"]
        ship_markers = {s: marker_list[i % len(marker_list)] for i, s in enumerate(ships)}

    for h, hemi in enumerate(hemis):
        dfh = ship_month_clim[ship_month_clim["hemi"] == hemi].copy()

        for i, product in enumerate(products):
            rr = h * nrows + (i // ncols)
            cc = i % ncols
            ax = axes[rr, cc]

            sub = dfh[[obs_col, product, "ship", "month"]].replace([np.inf, -np.inf], np.nan).dropna().copy()

            for ship in ships:
                ss = sub[sub["ship"] == ship]
                if len(ss) == 0:
                    continue

                ax.scatter(
                    ss[obs_col],
                    ss[product],
                    s=55,
                    color=ship_colors.get(ship, None),
                    marker=ship_markers.get(ship, "o"),
                    edgecolor="k",
                    linewidth=0.4,
                    alpha=0.85,
                    label=ship if (i == 0 and h == 0) else None,
                )

            # 1:1 line
            xymax = np.nanmax([
                sub[obs_col].max() if len(sub) else np.nan,
                sub[product].max() if len(sub) else np.nan
            ])
            if np.isfinite(xymax):
                lim_max = xymax * 1.05
                ax.plot([0, lim_max], [0, lim_max], "--", color="0.5", lw=1)

                if xlim is None:
                    ax.set_xlim(0, lim_max)
                else:
                    ax.set_xlim(*xlim)

                if ylim is None:
                    ax.set_ylim(0, lim_max)
                else:
                    ax.set_ylim(*ylim)

            # metrics
            if len(sub) >= 2:
                obs = sub[obs_col]
                pred = sub[product]
                rb = 100 * (pred.mean() - obs.mean()) / obs.mean() if obs.mean() != 0 else np.nan
                rmse = np.sqrt(np.mean((pred - obs) ** 2))
                ccv = np.corrcoef(obs, pred)[0, 1] if len(sub) > 1 else np.nan

                txt = f"RB: {rb:.2f}%\nRMSE: {rmse:.2f} mm/day\nCC: {ccv:.2f}\nN: {len(sub)}"
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=10, fontweight="bold")

            ax.set_title(f"{product} ({hemi})", fontsize=13)
            ax.set_xlabel("OceanRAIN ship-month climatology [mm day$^{-1}$]")
            ax.set_ylabel(f"{product} [mm day$^{-1}$]")
            ax.grid(True, alpha=0.3)

    # turn off unused axes
    total_axes = axes.shape[0] * axes.shape[1]
    used_axes = len(hemis) * len(products)
    for k in range(used_axes, total_axes):
        rr = k // axes.shape[1]
        cc = k % axes.shape[1]
        axes[rr, cc].axis("off")

    # legend
    handles = [
        plt.Line2D([0], [0],
                   marker=ship_markers[s],
                   color="w",
                   markerfacecolor=ship_colors[s],
                   markeredgecolor="k",
                   markersize=8,
                   linestyle="None",
                   label=s)
        for s in ships
    ]
    fig.legend(handles=handles, labels=ships, loc="upper center", ncol=min(len(ships), 5), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
#-----------------------------------------------------------------------------

def plot_hemi_phase_quant_metrics_panel(
    phase_based_qt_metrics_hemi: dict,
    products: list,
    product_colors: dict,
    *,
    metrics=("CC", "RMSE", "MAE", "Bias"),
    phases=("Rain", "Snow"),
    hemis=("SH", "NH"),
    figsize=(12, 10),
    sharex=True,
    ylims=None,
    ylabel_map=None,
):
    """
    Layout:
      - columns = phases
      - rows    = metrics
      - x-axis  = products
      - bars    = hemis (SH/NH) for each product

    Expected:
      phase_based_qt_metrics_hemi[hemi][phase][product] -> dict with keys in metrics
    """
    hemi_style = {
    "SH": {"alpha": 0.95, "hatch": None},
    "NH": {"alpha": 0.55, "hatch": "///"},
    }

    if ylabel_map is None:
        ylabel_map = {
            "CC": "CC",
            "RMSE": "RMSE \n[mm day$^{-1}$]",
            "MAE":  "MAE \n[mm day$^{-1}$]",
            "Bias": "Bias"
        }

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=len(phases),
        figsize=figsize,
        sharex=sharex
    )

    if len(metrics) == 1 and len(phases) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = np.array([axes])
    elif len(phases) == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(products))
    width = 0.35
    offsets = (np.arange(len(hemis)) - (len(hemis)-1)/2) * width

    for c, ph in enumerate(phases):
        for r, met in enumerate(metrics):
            ax = axes[r, c]

            for j, hemi in enumerate(hemis):
                vals = []
                for p in products:
                    d = phase_based_qt_metrics_hemi.get(hemi, {}).get(ph, {}).get(p, {})
                    vals.append(d.get(met, np.nan))

                style = hemi_style[hemi]  # hemi loop variable
                ax.bar(
                    x + offsets[j],
                    vals,
                    width=width,
                    color=[product_colors.get(p, None) for p in products],
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    edgecolor="k" if style["hatch"] else "none",
                    linewidth=0.0 if style["hatch"] is None else 0.3,
                )

            ax.set_ylabel(ylabel_map.get(met, met), fontsize=14)
            ax.grid(True, axis="y", alpha=0.3)

            if ylims is not None and met in ylims:
                ax.set_ylim(*ylims[met])

            if r == 0:
                ax.set_title(f"{ph}")

            if r == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(products, rotation=25, ha="right")

    # apply alpha split for hemis
    for c, ph in enumerate(phases):
        for r, met in enumerate(metrics):
            ax = axes[r, c]
            bars = ax.patches
            nP = len(products)
            for j, hemi in enumerate(hemis):
                for iP in range(nP):
                    b = bars[j*nP + iP]
                    b.set_alpha(0.9 if hemi == hemis[0] else 0.45)

    # product legend at top
    prod_handles = [plt.Rectangle((0, 0), 1, 1, color=product_colors.get(p, None)) for p in products]
    fig.legend(prod_handles, products, loc="upper center", 
               ncol=len(products), frameon=False, fontsize=14)

    # hemi legend (alpha-based proxy)
    # axes[0, 0].legend(
    #     handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.6", alpha=0.9),
    #              plt.Rectangle((0, 0), 1, 1, facecolor="0.6", alpha=0.45)],
    #     labels=[hemis[0], hemis[1]],
    #     loc="upper left",
    #     frameon=False,
    #     title="Hemi"
    # )
    

    hemi_handles = [
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["SH"]["alpha"],
                    hatch=hemi_style["SH"]["hatch"], label="SH"),
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["NH"]["alpha"],
                    hatch=hemi_style["NH"]["hatch"], label="NH"),
    ]

    # Put it inside top-left axis (or wherever you like)
    axes[0, 1].legend(handles=hemi_handles, title="Hemi", loc="upper left", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

#------------------------------------------------------------------------------
def compute_hemi_metrics_oceanrain(
    daily_or_attached: pd.DataFrame,
    *,
    products,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
    cat_thr=1.0,
    qt_sample="all_valid",  # options: "all_valid", "hits"
):
    cat_metrics_hemi = {}
    qt_metrics_hemi = {}

    for hemi in hemis:
        dfh = daily_or_attached[daily_or_attached["hemi"] == hemi].copy()

        cat_metrics_hemi.setdefault(hemi, {})
        qt_metrics_hemi.setdefault(hemi, {})

        print(f"\nProcessing hemisphere = {hemi}, N_total = {len(dfh)}")

        for product in products:
            sub = dfh[[obs_col, product]].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

            print(f"  {product}: N_valid = {len(sub)}")

            if len(sub) == 0:
                cat_metrics_hemi[hemi][product] = {}
                qt_metrics_hemi[hemi][product] = {}
                continue

            observed = sub[obs_col]
            forecast = sub[product]

            # categorical metrics on all valid pairs
            cat_metrics_hemi[hemi][product] = categorical_stats(
                forecast, observed, cat_thr
            )

            # quantitative metrics
            if qt_sample == "hits":
                sub_qt = sub[(sub[obs_col] >= cat_thr) & (sub[product] >= cat_thr)].copy()
            elif qt_sample == "all_valid":
                sub_qt = sub.copy()
            else:
                raise ValueError("qt_sample must be 'all_valid' or 'hits'.")

            if len(sub_qt) == 0:
                qt_metrics_hemi[hemi][product] = {
                    "CC": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "N": 0,
                }
            else:
                qt = calculate_metrics(sub_qt, obs_col, product)
                if isinstance(qt, dict):
                    qt["N"] = len(sub_qt)
                qt_metrics_hemi[hemi][product] = qt

    return cat_metrics_hemi, qt_metrics_hemi
#------------------------------------------------------------------------------

def plot_hemi_cat_metrics_barpanel(
    cat_metrics_hemi,
    products,
    product_colors,
    *,
    metrics=("POD", "FAR", "Bias", "HSS"),
    hemis=("SH", "NH"),
    figsize=(10, 10),
    ylims=None,
):
    """
    Rows = metrics
    X-axis = products
    Bars = hemispheres
    """

    hemi_style = {
        "SH": {"alpha": 0.95, "hatch": None},
        "NH": {"alpha": 0.55, "hatch": "///"},
    }

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1,
        figsize=figsize, sharex=True
    )

    if len(metrics) == 1:
        axes = np.array([axes])

    x = np.arange(len(products))
    width = 0.35
    offsets = (np.arange(len(hemis)) - (len(hemis) - 1) / 2) * width

    for r, met in enumerate(metrics):
        ax = axes[r]

        for j, hemi in enumerate(hemis):
            vals = []
            for p in products:
                d = cat_metrics_hemi.get(hemi, {}).get(p, {})
                vals.append(d.get(met, np.nan) if isinstance(d, dict) else np.nan)

            style = hemi_style[hemi]

            ax.bar(
                x + offsets[j],
                vals,
                width=width,
                color=[product_colors.get(p, None) for p in products],
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor="k" if style["hatch"] else "none",
                linewidth=0.3 if style["hatch"] else 0.0,
            )

        ax.set_ylabel(met, fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)

        if ylims is not None and met in ylims:
            ax.set_ylim(*ylims[met])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(products, rotation=25, ha="right")

    prod_handles = [
        mpatches.Patch(facecolor=product_colors.get(p, "0.7"), label=p)
        for p in products
    ]
    fig.legend(
        handles=prod_handles,
        labels=products,
        loc="upper center",
        ncol=len(products),
        frameon=False,
        fontsize=11
    )

    hemi_handles = [
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["SH"]["alpha"],
                       hatch=hemi_style["SH"]["hatch"], label="SH"),
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["NH"]["alpha"],
                       hatch=hemi_style["NH"]["hatch"], label="NH"),
    ]
    axes[0].legend(handles=hemi_handles, title="Hemi", loc="upper left", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
#------------------------------------------------------------------------------
def plot_hemi_quant_metrics_barpanel(
    qt_metrics_hemi,
    products,
    product_colors,
    *,
    metrics=("CC", "RMSE", "MAE", "Bias"),
    hemis=("SH", "NH"),
    figsize=(10, 10),
    ylims=None,
    ylabel_map=None,
):
    """
    Rows = metrics
    X-axis = products
    Bars = hemispheres
    """

    hemi_style = {
        "SH": {"alpha": 0.95, "hatch": None},
        "NH": {"alpha": 0.55, "hatch": "///"},
    }

    if ylabel_map is None:
        ylabel_map = {
            "CC": "CC",
            "RMSE": "RMSE\n[mm day$^{-1}$]",
            "MAE": "MAE\n[mm day$^{-1}$]",
            "Bias": "Bias"
        }

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1,
        figsize=figsize, sharex=True
    )

    if len(metrics) == 1:
        axes = np.array([axes])

    x = np.arange(len(products))
    width = 0.35
    offsets = (np.arange(len(hemis)) - (len(hemis) - 1) / 2) * width

    for r, met in enumerate(metrics):
        ax = axes[r]

        for j, hemi in enumerate(hemis):
            vals = []
            for p in products:
                d = qt_metrics_hemi.get(hemi, {}).get(p, {})
                vals.append(d.get(met, np.nan) if isinstance(d, dict) else np.nan)

            style = hemi_style[hemi]

            ax.bar(
                x + offsets[j],
                vals,
                width=width,
                color=[product_colors.get(p, None) for p in products],
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor="k" if style["hatch"] else "none",
                linewidth=0.3 if style["hatch"] else 0.0,
            )

        ax.set_ylabel(ylabel_map.get(met, met), fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)

        if ylims is not None and met in ylims:
            ax.set_ylim(*ylims[met])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(products, rotation=25, ha="right")

    prod_handles = [
        mpatches.Patch(facecolor=product_colors.get(p, "0.7"), label=p)
        for p in products
    ]
    fig.legend(
        handles=prod_handles,
        labels=products,
        loc="upper center",
        ncol=len(products),
        frameon=False,
        fontsize=11
    )

    hemi_handles = [
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["SH"]["alpha"],
                       hatch=hemi_style["SH"]["hatch"], label="SH"),
        mpatches.Patch(facecolor="0.2", alpha=hemi_style["NH"]["alpha"],
                       hatch=hemi_style["NH"]["hatch"], label="NH"),
    ]
    axes[0].legend(handles=hemi_handles, title="Hemi", loc="upper left", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
#-------------------------------------------------------------------------------


def plot_satellite_vs_groundtruth(
    df_all_regs,
    truth_col,
    product_cols,
    product_labels=None,
    truth_label=None,
    max_val=18,
    ticks=(0, 6, 12, 18),
    figsize_per_col=6,
    figsize_per_row=5,
    region_markers=None,
    region_colors=None,
    region_labels=None,
    savepath=None,
):
    """
    Generic scatter-plot function for satellite vs in situ evaluation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing truth and satellite columns
    truth_col : str
        Column name for ground truth (x-axis)
    product_cols : list of str
        Column names for satellite products (y-axis)
    product_labels : list of str, optional
        Display names for products (defaults to column names)
    """

    n_prod = len(product_cols)
    ncols = min(3, n_prod)
    nrows = math.ceil(n_prod / ncols)

    if product_labels is None:
        product_labels = product_cols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (prod, label) in enumerate(zip(product_cols, product_labels)):
        # df  = pd.concat([dff[[truth_col, prod]] for dff in df_dict.values()], ignore_index=True)
        ax = axes[i]

        # x = df[truth_col].values
        # y = df[prod].values

        qt_met = calculate_metrics(df_all_regs,truth_col, prod)

        for region in df_all_regs['region'].unique():

            dff = df_all_regs[df_all_regs['region'] == region]

            marker = region_markers.get(region, "o")

            ax.scatter(
                dff[truth_col].values,
                dff[prod].values,
                c=region_colors[region],#"k",
                marker=marker,
                s=90,
                edgecolor="k",
                linewidth=0.8,
                alpha=0.9,
                label=region
            )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.minorticks_on()

        ax.tick_params(axis='both', which='major', length=7, width=1.2, labelsize=16)
        ax.tick_params(axis='both', which='minor', length=4, width=0.8)

        # 1:1 line
        xx = np.linspace(0, max_val, 100)
        ax.plot(xx, xx, '--', color='gray')

        ax.set_xlabel(truth_label + ' [mm day$^{-1}$]', fontsize=16, fontweight='bold')
        ax.set_ylabel(label + ' Estimates \n [mm day$^{-1}$]', fontsize=16, fontweight='bold')
        ax.set_title(f'{label} vs {truth_col}', fontsize=18, fontweight='bold')

        # Stats annotation
        ax.text(
            0.05, 0.97,
            f'RB: {qt_met["Bias"]:.2f}%\nRMSE: {qt_met["RMSE"]:.2f} mm/day\nCC: {qt_met["CC"]:.2f}',
            transform=ax.transAxes,
            fontsize=15,
            fontweight='bold',
            verticalalignment='top'
        )

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    # Remove empty panels
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Legend
    # Build region legend handles
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker=region_markers[r],
            color=region_colors[r],#"k",
            linestyle='None',
            markersize=10,
            markeredgewidth=1,
            label=region_labels[r]
        )
        for r in region_markers
    ]

    # Make room at the bottom for the legend
    fig.subplots_adjust(bottom=0.15)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=16,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if savepath:
        plt.savefig(savepath, dpi=500, bbox_inches='tight')

    return fig

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import string
import numpy as np
import matplotlib.pyplot as plt


def plot_combined_daily_mean_pal_buoy_scatter_black(
    pal_df,
    buoy_df,
    truth_col,
    product_cols,
    product_labels=None,
    pal_truth_label="PAL Observations",
    buoy_truth_label="Buoy Observations",
    pal_max_val=18,
    buoy_max_val=15,
    pal_ticks=(0, 6, 12, 18),
    buoy_ticks=(0, 5, 10, 15),
    figsize=(18, 20),
    point_size=70,
    point_alpha=0.85,
    savepath=None,
):
    """
    Manuscript-style combined scatter figure for multi-year mean daily precipitation.

    Layout:
        Panels (a)-(f): Product estimates versus PAL observations
        Panels (g)-(l): Product estimates versus buoy observations

    All points are plotted as filled black circles to avoid regional legends
    and reduce visual complexity.
    """

    if product_labels is None:
        product_labels = product_cols

    if len(product_cols) != len(product_labels):
        raise ValueError("product_cols and product_labels must have the same length.")

    if len(product_cols) != 6:
        raise ValueError("This layout expects exactly six products.")

    fig, axes = plt.subplots(
        4, 3,
        figsize=figsize,
        squeeze=False
    )

    axes_flat = axes.flatten()
    panel_letters = list(string.ascii_lowercase)

    def _plot_one_panel(
        ax,
        df,
        prod,
        label,
        truth_label,
        max_val,
        ticks,
        panel_letter,
    ):
        # Calculate metrics using your existing function.
        qt_met = calculate_metrics(df, truth_col, prod)

        ax.scatter(
            df[truth_col].values,
            df[prod].values,
            marker="o",
            s=point_size,
            facecolor="black",
            edgecolor="black",
            linewidth=0.5,
            alpha=point_alpha,
        )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.minorticks_on()

        ax.tick_params(axis="both", which="major", length=7, width=1.2, labelsize=14)
        ax.tick_params(axis="both", which="minor", length=4, width=0.8)

        # 1:1 line
        xx = np.linspace(0, max_val, 100)
        ax.plot(xx, xx, "--", color="0.5", linewidth=1.2)

        # No subplot title; axes carry product/reference information.
        ax.set_xlabel(
            f"{truth_label} [mm day$^{{-1}}$]",
            fontsize=18,
            fontweight="bold"
        )
        ax.set_ylabel(
            f"{label} Estimates\n[mm day$^{{-1}}$]",
            fontsize=18,
            fontweight="bold"
        )

        # Panel label above upper-left corner.
        ax.text(
            0.02, 1.04,
            f"({panel_letter})",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Stats annotation.
        ax.text(
            0.05, 0.95,
            f"RB: {qt_met['Bias']:.2f}%\n"
            f"RMSE: {qt_met['RMSE']:.2f} mm/day\n"
            f"CC: {qt_met['CC']:.2f}",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            verticalalignment="top",
        )

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    # Panels (a)-(f): PAL
    for i, (prod, label) in enumerate(zip(product_cols, product_labels)):
        _plot_one_panel(
            ax=axes_flat[i],
            df=pal_df,
            prod=prod,
            label=label,
            truth_label=pal_truth_label,
            max_val=pal_max_val,
            ticks=pal_ticks,
            panel_letter=panel_letters[i],
        )

    # Panels (g)-(l): Buoys
    for j, (prod, label) in enumerate(zip(product_cols, product_labels)):
        idx = j + len(product_cols)
        _plot_one_panel(
            ax=axes_flat[idx],
            df=buoy_df,
            prod=prod,
            label=label,
            truth_label=buoy_truth_label,
            max_val=buoy_max_val,
            ticks=buoy_ticks,
            panel_letter=panel_letters[idx],
        )

    # Group labels.
    fig.text(
        0.015, 0.735,
        "PAL",
        rotation=90,
        fontsize=25,
        fontweight="bold",
        va="center",
        ha="center",
    )

    fig.text(
        0.015, 0.285,
        "Buoys",
        rotation=90,
        fontsize=25,
        fontweight="bold",
        va="center",
        ha="center",
    )

    plt.tight_layout(rect=[0.035, 0.02, 1, 0.98])

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_categorical_metrics_by_region(
    metrics_dict,
    products,
    product_colors,
    region_labels=None,
    metrics=("POD", "FAR", "Bias", "HSS"),
    figsize=(18, 16),
    bar_width=0.18,
):
    """
    4x1 bar plot of categorical metrics by region, colored by product.
    """

    regions = list(PAL_REGION_NAMES.keys())#list(metrics_dict.keys())
    n_regions = len(regions)
    n_products = len(products)

    x = np.arange(n_regions)

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=figsize, sharex=True
    )

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, product in enumerate(products):
            values = [
                metrics_dict[reg][product][metric]
                if product in metrics_dict[reg]
                else np.nan
                for reg in regions
            ]

            ax.bar(
                    x + j * bar_width,
                    values,
                    width=bar_width,
                    color=product_colors[product],
                    edgecolor="black",
                    linewidth=0.8,
                    label=product if i == 0 else None,
                )
            # Metric-specific limits
            if metric in ["POD", "FAR"]:
                ax.set_ylim(0, 1)
            elif metric == "HSS":
                ax.set_ylim(0, 0.4)   # zoom in to show variability
            # (optional) leave Bias auto-scaled unless you want fixed bounds
            
            metric_label = metric + ' [mm day$^{-1}$]' if metric in ["MAE", "RMSE"] else metric

            metric_label = metric + ' [%]' if metrics == ("CC", "RMSE", "MAE", "Bias") else metric
                

        ax.set_ylabel(metric_label, fontsize=18, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Metric-specific limits
        if metric in ["POD", "FAR"]:
            ax.set_ylim(0, 1)

        ax.tick_params(axis="both", labelsize=15)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        

    # X-axis
    axes[-1].set_xticks(x + bar_width * (n_products - 1) / 2)
    axes[-1].set_xticklabels([PAL_REGION_NAMES[k] for k in regions], 
                             fontsize=15, fontweight="bold", rotation=25, ha="center")
    # axes[-1].set_xlabel("Region", fontsize=15, fontweight="bold")

    # Legend (top, single row)
    axes[0].legend(
        ncol=len(products),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        fontsize=18,
        frameon=False,
    )

    plt.tight_layout()
    return fig

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_monthly_climatology_2x2(
    monthly_clim_by_region,
    regions,
    region_labels,
    products,
    product_colors,
    ref_col="Buoy",
    ref_label="Buoy",
    figsize=(12, 9),
    lw=3.5,
    ncol_legend=7
):
    """
    Plot monthly climatology in 2x2 subplots (no shared axes).
    Makes multiple figures if len(regions) > 4.
    X-axis uses month integers (1–12).
    """

    months = np.arange(1, 13)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(regions):
            ax.axis("off")
            continue

        region = regions[i]
        clim = monthly_clim_by_region[region]

        # ---- reference ----
        ax.plot(
            clim["month"], clim[ref_col],
            lw=lw, color="b", label=ref_label
        )

        # ---- products ----
        for prod in products:
            if prod == ref_col:
                continue
            ax.plot(
                clim["month"], clim[prod],
                lw=lw, color=product_colors[prod], label=prod
            )

        ax.set_title(region_labels[region], fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=16, fontweight="bold")
        ax.set_ylabel("[mm day$^{-1}$]", fontsize=16, fontweight="bold")

        ax.set_xticks(months)
        ax.set_xlim(1, 12)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.004),
    ncol=ncol_legend,
    frameon=False,
    fontsize=16
)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_monthly_climatology_anoms_2x2(
    monthly_anom_by_region,
    regions,
    region_labels,
    products,                 # include buoy_col as first element, same as your list
    product_colors,
    buoy_col="rain_rate",
    figsize=(12, 9),
    lw=3.2,
    ncol_legend=3,
    ylim=None,                # e.g., (-3, 3) if you want fixed range
):
    """
    2x2 plot of monthly climatology anomalies (Product - Buoy) per region.
    No shared axes. Month numbers 1–12.
    Creates multiple figures if regions > 4.
    """

    months = np.arange(1, 13)

    # chunk into groups of 4
    for k in range(0, len(regions), 4):
        regs = regions[k:k+4]

        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(regs):
                ax.axis("off")
                continue

            region = regs[i]
            clim = monthly_anom_by_region[region]

            # plot each product anomaly
            for prod in products:
                if prod == buoy_col:
                    continue
                ax.plot(
                    clim["month"],
                    clim[f"{prod}_anom"],
                    lw=lw,
                    color=product_colors[prod],
                    label=prod
                )

            ax.axhline(0, color="k", lw=1.5, ls ='--' ,alpha=0.8)  # zero reference
            ax.set_title(region_labels[region], fontsize=15, fontweight="bold")
            ax.set_xlabel("Month", fontsize=12, fontweight="bold")
            ax.set_ylabel("Product − Buoy [mm day$^{-1}$]", fontsize=12, fontweight="bold")

            ax.set_xticks(months)
            ax.set_xlim(1, 12)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", labelsize=11)

            if ylim is not None:
                ax.set_ylim(*ylim)

        # One legend for the whole figure (use first active axis)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=ncol_legend,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, 0.98)
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        yield fig  # <-- generator: yields each figure
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_monthly_climatology_stack(
    monthly_clim_by_region,
    regions,
    products,
    product_colors,
    figsize=(14, 14),
    month_as_numbers=True,
    savepath=None,
):
    """
    Stacked (n_regions x 1) monthly climatology plot by region.
    """
    fig, axes = plt.subplots(
        nrows=len(regions),
        ncols=1,
        figsize=figsize,
        sharex=True,
        dpi=300
    )

    if len(regions) == 1:
        axes = [axes]

    if month_as_numbers:
        month_ticks = list(range(1, 13))
        month_labels = [str(m) for m in month_ticks]
    else:
        month_ticks = list(range(1, 13))
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    for ax, region in zip(axes, regions):
        clim = monthly_clim_by_region[region]

        # Buoy reference
        ax.plot(
            clim["month"],
            clim["rain_rate"],
            lw=4,
            color="b",
            label="Buoy"
        )

        # Satellite / reanalysis
        for prod in products[1:]:
            ax.plot(
                clim["month"],
                clim[prod],
                lw=4,
                color=product_colors[prod],
                label=prod
            )

        ax.set_title(region, fontsize=14, fontweight="bold")
        ax.set_ylabel("Monthly Mean Rainfall (mm/day)", fontsize=12)
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", labelsize=11)

    axes[-1].set_xlabel("Month", fontsize=13)

    # One clean legend
    axes[0].legend(
        ncol=3,
        fontsize=11,
        frameon=False
    )

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


#%%

def plot_doy_climatology_scatter_panels_with_metrics(
    clim_df: pd.DataFrame,
    *,
    obs_col: str,                 # x-axis (OceanRAIN DOY mean)
    product_cols: list,           # y-axis products
    product_colors: dict = None,
    title: str = "",
    max_val: float = 18,
    ticks=(0, 6, 12, 18),
):
    d = clim_df.copy()

    n_prod = len(product_cols)
    ncols = min(3, n_prod)
    nrows = math.ceil(n_prod / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        dpi=200
    )
    axes = axes.flatten()

    for i, prod in enumerate(product_cols):
        ax = axes[i]

        dd = d[[obs_col, prod]].dropna()
        x = dd[obs_col].to_numpy()
        y = dd[prod].to_numpy()

        ax.scatter(
            x, y,
            s=70,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.6,
            color=(product_colors.get(prod, None) if product_colors else None)
        )

        # 1:1 line and axes
        lim = (0, max_val)
        ax.plot(lim, lim, "--", color="gray", lw=1.2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

        ax.set_xlabel("OceanRAIN DOY mean [mm day$^{-1}$]", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{prod} DOY mean [mm day$^{-1}$]", fontsize=14, fontweight="bold")
        ax.set_title(prod, fontsize=15, fontweight="bold")

        # metrics annotation (CC, RMSE, Bias)
        if len(dd) >= 2:
            m = calculate_metrics(dd, obs_col, prod)
            ax.text(
                0.05, 0.97,
                f'RB: {m["Bias"]:.2f}%\nRMSE: {m["RMSE"]:.2f} mm/day\nCC: {m["CC"]:.2f}',
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top"
            )
        else:
            ax.text(
                0.05, 0.97,
                "n<2",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top"
            )

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    # remove unused panels
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, y=0.98, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
# additional analysis functions
def _prob_from_hist(x, bin_edges, eps=1e-12):
    """Histogram -> probability mass per bin (sums to 1)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    # keep nonnegative rain only (optional)
    x = x[x >= 0]

    h, _ = np.histogram(x, bins=bin_edges)
    p = h.astype(float)
    p = p + eps               # smoothing to avoid zeros
    p = p / p.sum()
    return p

def kl_divergence(p, q):
    """KL(P||Q) for discrete distributions."""
    return np.sum(p * np.log(p / q))

def js_divergence(p, q, base=2):
    """Jensen–Shannon divergence (bounded, symmetric)."""
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    if base == 2:
        js = js / np.log(2)
    return js

def wasserstein_binned(p, q, bin_centers):
    """
    1D Wasserstein distance for binned discrete distributions.
    Uses CDF difference integrated over bin spacing.
    """
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    # bin widths for integration
    centers = np.asarray(bin_centers)
    # approximate widths from centers (works for log bins too)
    widths = np.empty_like(centers)
    widths[1:-1] = 0.5 * (centers[2:] - centers[:-2])
    widths[0]    = centers[1] - centers[0]
    widths[-1]   = centers[-1] - centers[-2]

    return np.sum(np.abs(cdf_p - cdf_q) * widths)

def pdf_distance_table(df, region, ref_col, product_cols, bin_edges):
    """
    Returns a table of PDF distances between each product and the reference
    for one region.
    """
    dfr = df[df["region"] == region].copy()

    # reference distribution
    p_ref = _prob_from_hist(dfr[ref_col].values, bin_edges)

    # bin centers
    bin_centers = 0.5 * (np.asarray(bin_edges[:-1]) + np.asarray(bin_edges[1:]))

    rows = []
    for prod in product_cols:
        p_prod = _prob_from_hist(dfr[prod].values, bin_edges)

        rows.append({
            "region": region,
            "product": prod,
            "JSD": js_divergence(p_ref, p_prod, base=2),       # 0..1
            "KL(ref||prod)": kl_divergence(p_ref, p_prod),     # >=0
            "Wasserstein": wasserstein_binned(p_ref, p_prod, bin_centers)
        })

    out = pd.DataFrame(rows).sort_values("JSD")
    return out


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def circular_month_lag(prod_peak, ref_peak):
    """
    Return lag in months wrapped to [-6, +6].
    Positive means product peaks later than reference.
    """
    lag = int(prod_peak) - int(ref_peak)
    # wrap to shortest direction on a 12-month circle
    if lag > 6:
        lag -= 12
    elif lag < -6:
        lag += 12
    return lag

def robust_peak_month(months, y, tol=0.02):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    months = np.asarray(months)[m]
    y = y[m]
    ymax = y.max()
    peak_months = months[y >= (1 - tol) * ymax]
    return int(np.median(peak_months))

def amp_and_phase_scores(monthly_clim_by_region, ref_col="rain_rate",
                         product_cols=("GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2")):
    rows = []

    for region, clim in monthly_clim_by_region.items():
        clim = clim.sort_values("month")

        ref = clim[ref_col].values.astype(float)
        ref_amp = np.nanmax(ref) - np.nanmin(ref)
        ref_peak_month = int(clim.loc[np.nanargmax(ref), "month"])
        # ref_peak_month = robust_peak_month(clim["month"].values, ref)

        for prod in product_cols:
            y = clim[prod].values.astype(float)

            prod_amp = np.nanmax(y) - np.nanmin(y)
            amp_ratio = np.nan if ref_amp == 0 else prod_amp / ref_amp

            prod_peak_month = int(clim.loc[np.nanargmax(y), "month"])
            # prod_peak_month = robust_peak_month(clim["month"].values, y)
            phase_lag = circular_month_lag(prod_peak_month, ref_peak_month)

            rows.append({
                "region": region,
                "product": prod,
                "buoy_amp": ref_amp,
                "prod_amp": prod_amp,
                "amp_ratio": amp_ratio,
                "buoy_peak_month": ref_peak_month,
                "prod_peak_month": prod_peak_month,
                "phase_lag_months": phase_lag
            })

    return pd.DataFrame(rows)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def _text_color_for_value(val, vmin, vmax, cmap, threshold=0.55):
    """
    Choose white/black text based on background luminance at this value.
    threshold ~0.5–0.6 works well for viridis-like maps.
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    r, g, b, a = cmap(norm(val))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < threshold else "black"


def plot_amp_phase_heatmaps(
    scores,
    region_order=("ENP", "WNP", "IND", "ATL"),
    product_order=("GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    figsize=(12, 4.2),
    cmap="Spectral_r",
    annot_fontsize=12,      # <-- bigger numbers
    annot_weight="bold",
    savepath=None
):
    """
    Make two heatmaps:
      (1) Amplitude ratio (Prod/Buoy)
      (2) Phase lag (months)

    Annotations auto-switch between black/white for readability.
    """

    # Pivot to matrices (Region x Product)
    amp = (
        scores.pivot(index="region", columns="product", values="amp_ratio")
        .reindex(index=region_order, columns=product_order)
    )
    lag = (
        scores.pivot(index="region", columns="product", values="phase_lag_months")
        .reindex(index=region_order, columns=product_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
    cmap_obj = plt.get_cmap(cmap)

    # -----------------------------
    # Amplitude ratio heatmap
    # -----------------------------
    ax0 = axes[0]
    vmin0 = np.nanmin(amp.values)
    vmax0 = np.nanmax(amp.values)

    im0 = ax0.imshow(
        amp.values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin0,
        vmax=vmax0,
    )

    ax0.set_title("Amplitude ratio (Prod / Buoy)", fontweight="bold")
    ax0.set_yticks(np.arange(len(amp.index)))
    ax0.set_yticklabels(amp.index, fontweight="bold")
    ax0.set_xticks(np.arange(len(amp.columns)))
    ax0.set_xticklabels(amp.columns, rotation=30, ha="right", fontweight="bold")

    for i in range(amp.shape[0]):
        for j in range(amp.shape[1]):
            v = amp.values[i, j]
            if np.isfinite(v):
                txt_color = _text_color_for_value(v, vmin0, vmax0, cmap_obj)
                ax0.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    fontweight=annot_weight,
                    color=txt_color
                )

    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("Ratio", rotation=90, fontweight="bold")

    # -----------------------------
    # Phase lag heatmap
    # -----------------------------
    ax1 = axes[1]
    vmin1 = np.nanmin(lag.values)
    vmax1 = np.nanmax(lag.values)

    im1 = ax1.imshow(
        lag.values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin1,
        vmax=vmax1,
    )

    ax1.set_title("Phase lag (months)", fontweight="bold")
    ax1.set_yticks(np.arange(len(lag.index)))
    ax1.set_yticklabels(lag.index, fontweight="bold")
    ax1.set_xticks(np.arange(len(lag.columns)))
    ax1.set_xticklabels(lag.columns, rotation=30, ha="right", fontweight="bold")

    for i in range(lag.shape[0]):
        for j in range(lag.shape[1]):
            v = lag.values[i, j]
            if np.isfinite(v):
                txt_color = _text_color_for_value(v, vmin1, vmax1, cmap_obj)
                ax1.text(
                    j, i, f"{int(v):+d}",
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    fontweight=annot_weight,
                    color=txt_color
                )

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Months", rotation=90, fontweight="bold")

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig

#-----------------------------------------------------------------------------

def make_annual_means(df, products, date_col="date", region_col="region"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["year"] = d[date_col].dt.year

    annual = (
        d.groupby([region_col, "year"])[products]
         .mean()
         .reset_index()
         .sort_values([region_col, "year"])
    )
    return annual
#-----------------------------------------------------------------------------
# -----------------------------
# helpers: linear trend + bootstrap CI
# -----------------------------
def linreg_slope_intercept(x, y):
    """Least-squares y = a + b*x. Returns (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan, np.nan
    b = np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0)
    a = np.mean(y) - b * np.mean(x)
    return a, b

#-----------------------------------------------------------------------------
def slope_only_on_common_years(dfr, xcol, ycol, ok_mask=None):
    """
    OLS slope of ycol vs xcol using finite values.
    If ok_mask is provided, applies it first (e.g., buoy-valid years).
    Returns slope (b) in units of y per x.
    """
    x = dfr[xcol].values.astype(float)
    y = dfr[ycol].values.astype(float)

    m = np.isfinite(x) & np.isfinite(y)
    if ok_mask is not None:
        m = m & ok_mask

    if m.sum() < 2:
        return np.nan

    a, b = linreg_slope_intercept(x[m], y[m])
    return b
#-----------------------------------------------------------------------------
def add_slope_stack(
    ax,
    slope_dict,
    color_dict,
    x=0.02,
    y=0.96,
    dy=0.055,
    fmt="{name}: {slope:+.3f}",
    units=" mm day$^{-1}$ yr$^{-1}$",
    fontsize=10,
    title="Slopes:"
):
    """
    Writes a small list of colored slope values inside the axes.
    slope_dict: {name: slope_value}
    color_dict: {name: color}
    """
    ax.text(x, y, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, fontweight="bold")

    yy = y - dy
    for name, s in slope_dict.items():
        if not np.isfinite(s):
            txt = f"{name}: n/a"
        else:
            txt = fmt.format(name=name, slope=s) + units
        ax.text(x, yy, txt, transform=ax.transAxes,
                ha="left", va="top",
                fontsize=fontsize,
                color=color_dict.get(name, "k"))
        yy -= dy
#-----------------------------------------------------------------------------

def bootstrap_trend_band(x, y, n_boot=2000, ci=95, seed=42):
    """
    Bootstrap linear trend on (x,y). Returns dict with xgrid, mid, lo, hi, slope stats.
    Uses resampling WITH replacement of the (year,value) pairs.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return None

    xgrid = np.linspace(x.min(), x.max(), 200)
    preds = np.empty((n_boot, len(xgrid)), dtype=float)
    slopes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = linreg_slope_intercept(x[idx], y[idx])
        preds[i, :] = a + b * xgrid
        slopes[i] = b

    lo = np.percentile(preds, (100 - ci) / 2, axis=0)
    hi = np.percentile(preds, 100 - (100 - ci) / 2, axis=0)
    mid = np.mean(preds, axis=0)

    slo = np.percentile(slopes, (100 - ci) / 2)
    shi = np.percentile(slopes, 100 - (100 - ci) / 2)
    smid = np.mean(slopes)

    return {
        "xgrid": xgrid,
        "mid": mid,
        "lo": lo,
        "hi": hi,
        "slope_mean": smid,
        "slope_lo": slo,
        "slope_hi": shi,
    }

# -----------------------------
# Option B: split series into contiguous year segments
# -----------------------------
def contiguous_segments(years_obs, max_gap=1):
    """
    Split sorted integer years into contiguous segments.
    max_gap=1 => break if year jump > 1.
    Returns list of arrays of years.
    """
    years_obs = np.asarray(years_obs, dtype=int)
    if years_obs.size == 0:
        return []
    years_obs = np.sort(np.unique(years_obs))
    breaks = np.where(np.diff(years_obs) > max_gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(years_obs) - 1]
    return [years_obs[s:e + 1] for s, e in zip(starts, ends)]
#-----------------------------------------------------------------------------
def contiguous_year_segments(years, max_gap=1):
    """
    Split sorted integer years into contiguous segments where gaps <= max_gap.
    Returns list of np.array segments.
    """
    years = np.array(sorted(set(years)), dtype=int)
    if len(years) == 0:
        return []
    segs = [[years[0]]]
    for y in years[1:]:
        if (y - segs[-1][-1]) <= max_gap:
            segs[-1].append(y)
        else:
            segs.append([y])
    return [np.array(s, dtype=int) for s in segs]

#-----------------------------------------------------------------------------
def std_and_corr_table(dfr, ref="rain_rate", products=None):
    rows = []
    for p in products:
        if p == ref:
            continue
        x = dfr[ref].values
        y = dfr[p].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            r = np.nan
        else:
            r = np.corrcoef(x[m], y[m])[0,1]
        rows.append({
            "Product": p,
            "σ(annual)": np.nanstd(y, ddof=1),
            "r vs buoy": r
        })
    tab = pd.DataFrame(rows).sort_values("r vs buoy", ascending=False)
    return tab

#-----------------------------------------------------------------------------
def make_gap_aware_annual_df(annual_df, region, products, start_year=None, end_year=None):
    dfr = annual_df[annual_df["region"] == region].copy()
    yrs = dfr["year"].astype(int)

    if start_year is None: start_year = yrs.min()
    if end_year is None: end_year = yrs.max()

    full_years = pd.DataFrame({"year": np.arange(start_year, end_year + 1)})
    out = full_years.merge(dfr[["year"] + products], on="year", how="left").sort_values("year")
    return out

#-----------------------------------------------------------------------------
def annual_mean_with_coverage(df, products, region, min_days=250, ref="rain_rate", min_days_ref=300):
    d = df[df["region"] == region].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year

    out_rows = []
    for yr, g in d.groupby("year"):
        row = {"region": region, "year": int(yr)}

        for p in products:
            valid = g[p].notna().sum()
            row[f"n_{p}"] = int(valid)

            thr = min_days_ref if p == ref else min_days
            row[p] = g[p].mean() if valid >= thr else np.nan

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("year")
#-----------------------------------------------------------------------------
def plot_interannual_variability_one_region(
    annual_df,
    region,
    products,
    product_colors,
    ref="rain_rate",
    add_trend_for=None,            # trend on buoy by default
    add_trend_band=True,
    add_stats_inset=True,
    figsize=(12,5),
    ):
        if add_trend_for is None:
            add_trend_for = ref

        dfr = annual_df[annual_df["region"] == region].copy().sort_values("year")
        years = dfr["year"].values.astype(float)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

        # plot time series
        # buoy
        ax.plot(years, dfr[ref].values, lw=3.5, color="b", label="Buoy")

        # products
        for p in products:
            if p == ref:
                continue
            ax.plot(years, dfr[p].values, lw=2.8, color=product_colors[p], label=p)

        # trend + CI band
        if add_trend_band and add_trend_for in dfr.columns:
            y = dfr[add_trend_for].values.astype(float)
            out = bootstrap_trend_band(years, y, n_boot=2000, ci=95)
            if out is not None:
                ax.plot(out["xgrid"], out["mid"], lw=2.5, color="k", label="Buoy trend")
                ax.fill_between(out["xgrid"], out["lo"], out["hi"], alpha=0.15)  # default color
                # small text with slope
                ax.text(
                    0.01, 0.98,
                    f"Buoy slope = {out['slope_mean']:.3g} (95% CI [{out['slope_lo']:.3g}, {out['slope_hi']:.3g}])\nmm/day per year",
                    transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold"
                )

        # inset stats table
        if add_stats_inset:
            tab = std_and_corr_table(dfr, ref=ref, products=products)
            # keep it small: top 5 rows
            tab_show = tab.copy()
            tab_show["σ(annual)"] = tab_show["σ(annual)"].map(lambda v: f"{v:.2f}")
            tab_show["r vs buoy"] = tab_show["r vs buoy"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "nan")
            tab_show = tab_show.head(5)

            cell_text = tab_show[["σ(annual)", "r vs buoy"]].values.tolist()
            row_labels = tab_show["Product"].tolist()
            col_labels = ["σ", "r"]

            table = ax.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=col_labels,
                cellLoc="center",
                loc="upper right",
                bbox=[0.73, 0.55, 0.26, 0.40],  # [left, bottom, width, height]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)

        ax.set_title(f"{region}: Interannual variability (annual means)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.set_ylabel("Annual mean rainfall (mm/day)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.legend(ncol=3, fontsize=9, frameon=False)
        plt.tight_layout()
        return fig

#-----------------------------------------------------------------------------
def monthly_mean_with_coverage(
    df, products, region,
    min_days=20,            # for products
    ref="rain_rate",
    min_days_ref=25         # for buoy
):
    d = df[df["region"] == region].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ym"] = d["date"].dt.to_period("M").dt.to_timestamp()

    out = []
    for ym, g in d.groupby("ym"):
        row = {"region": region, "date": ym}
        for p in products:
            valid = g[p].notna().sum()
            thr = min_days_ref if p == ref else min_days
            row[f"n_{p}"] = int(valid)
            row[p] = g[p].mean() if valid >= thr else np.nan
        out.append(row)

    return pd.DataFrame(out).sort_values("date")
#-----------------------------------------------------------------------------
# -------------------------
# 0) Collapse to daily regional means (removes ID multiplicity)
# -------------------------
def daily_region_mean(df, products, date_col="date", region_col="region"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    # mean across IDs (and any repeats) -> one value per region-day
    out = (
        d.groupby([region_col, date_col])[products]
         .mean()
         .reset_index()
         .sort_values([region_col, date_col])
    )
    return out


# -------------------------
# 1) Daily -> monthly mean with coverage in UNIQUE DAYS
# -------------------------
def monthly_mean_with_day_coverage(daily_df, products, region,
                                  min_days=20, ref="rain_rate", min_days_ref=25,
                                  date_col="date", region_col="region"):
    d = daily_df[daily_df[region_col] == region].copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d["ym"] = d[date_col].dt.to_period("M").dt.to_timestamp()

    rows = []
    for ym, g in d.groupby("ym"):
        row = {region_col: region, "date": ym}
        for p in products:
            # count unique days with finite values (<= 31)
            valid_days = g.loc[np.isfinite(g[p].values), date_col].nunique()
            thr = min_days_ref if p == ref else min_days
            row[f"n_{p}_days"] = int(valid_days)
            row[p] = g[p].mean() if valid_days >= thr else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date")

#----------------------------------------------------------------------------
def monthly_from_raw_daily(df_raw, region, products, date_col="date"):
    """
    Raw daily df has multiple rows per day (multiple buoy IDs).
    We first collapse to ONE value per (region, date) by averaging across IDs,
    then compute monthly mean series (one row per month).
    """
    d = df_raw[df_raw["region"] == region].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    # 1) collapse multiple IDs -> one region-day value
    daily_region = (
        d.groupby(date_col)[products]
         .mean()
         .reset_index()
         .rename(columns={date_col: "date"})
    )

    # 2) monthly mean of those region-day values
    monthly = (
        daily_region
        .assign(ym=daily_region["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("ym")[products]
        .mean()
        .reset_index()
        .rename(columns={"ym": "date"})
        .sort_values("date")
    )
    return monthly


# -------------------------
# 2) 13-month centered rolling mean on monthly series
# -------------------------
def add_rm13_monthly(monthly_df, products, date_col="date"):
    """
    13-month centered rolling mean on monthly time series.
    Requires monthly_df to be sorted by date and one row per month.
    """
    d = monthly_df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    for p in products:
        d[p + "_rm13"] = (
            d[p]
            .rolling(window=13, center=True, min_periods=13)
            .mean()
        )
    return d


# -------------------------
# 3) Annual means from rm13 monthly series
# -------------------------
def annual_from_monthly_rm13(monthly_rm13_df, products, date_col="date"):
    """
    Annual mean from the rm13 monthly series.
    """
    rm_cols = [p + "_rm13" for p in products]
    d = monthly_rm13_df.copy()
    d["year"] = pd.to_datetime(d[date_col]).dt.year

    ann = (
        d.groupby("year")[rm_cols]
         .mean()
         .reset_index()
         .rename(columns={"year": "Year"})
         .sort_values("Year")
    )
    return ann


# -------------------------
# 4) Trend stats on annual series (mm/day/decade)
# -------------------------
def trend_stats_yearly(years, y):
    """
    Trend on annual series. Returns slope per decade, p-value, std.
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(years) & np.isfinite(y)
    if m.sum() < 3:
        return None
    res = linregress(years[m], y[m])
    slope_dec = res.slope * 10.0
    return {
        "slope_dec": slope_dec,
        "p": res.pvalue,
        "std": float(np.nanstd(y[m], ddof=1)),
    }
# -------------------------
# FULL: build annual (per region) with this pipeline
# -------------------------
def build_region_annual_series(df_raw, region, products,
                              ref="rain_rate",
                              min_days_month=20, min_days_month_ref=25):
    # 0) daily region mean
    daily = daily_region_mean(df_raw, products)

    # 1) monthly with day coverage
    monthly = monthly_mean_with_day_coverage(
        daily, products, region,
        min_days=min_days_month, ref=ref, min_days_ref=min_days_month_ref
    )

    # 2) rm13
    monthly_rm13 = add_rm13_monthly(monthly, products)

    # 3) annual from rm13
    annual = add_rm13_monthly(monthly_rm13, products, ref=ref)

    return annual  # columns: year, rain_rate_rm13, product_rm13, ...


#-----------------------------------------------------------------------------

def trend_stats_per_decade(dates, y):
    # dates: datetime64; y: float
    m = np.isfinite(y)
    if m.sum() < 10:
        return None

    # decimal years
    dt = pd.to_datetime(dates[m])
    x = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25
    yv = np.asarray(y[m], float)

    res = linregress(x, yv)
    slope_dec = res.slope * 10.0
    pval = res.pvalue
    std = float(np.nanstd(yv, ddof=1))
    return {"slope_dec": slope_dec, "p": pval, "std": std}

#-----------------------------------------------------------------------------
def trend_stats_yearly(years, y):
    """Trend stats on annual series. Returns slope per decade, p-value, std."""
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(years) & np.isfinite(y)
    if m.sum() < 3:
        return None
    res = linregress(years[m], y[m])  # slope is per year
    return {
        "slope_dec": res.slope * 10.0,   # mm/day/decade
        "p": res.pvalue,
        "std": np.nanstd(y[m], ddof=1)
    }

#-----------------------------------------------------------------------------
def bootstrap_trend_band_years(years, y, n_boot=2000, ci=95, seed=42):
    """
    Bootstrap CI band for a linear trend on annual data (x=years).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return None

    xgrid = np.linspace(x.min(), x.max(), 200)
    preds = np.empty((n_boot, len(xgrid)), dtype=float)
    slopes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]; yb = y[idx]
        # linear fit
        b = np.cov(xb, yb, ddof=0)[0, 1] / np.var(xb, ddof=0)
        a = np.mean(yb) - b * np.mean(xb)
        preds[i, :] = a + b * xgrid
        slopes[i] = b

    lo = np.percentile(preds, (100 - ci) / 2, axis=0)
    hi = np.percentile(preds, 100 - (100 - ci) / 2, axis=0)
    mid = np.mean(preds, axis=0)

    slo = np.percentile(slopes, (100 - ci) / 2)
    shi = np.percentile(slopes, 100 - (100 - ci) / 2)
    smid = np.mean(slopes)

    return {"xgrid": xgrid, "mid": mid, "lo": lo, "hi": hi,
            "slope_mean": smid, "slope_lo": slo, "slope_hi": shi}

#-----------------------------------------------------------------
def contiguous_year_segments(years, max_gap=1):
    years = np.array(sorted(set(years)), dtype=int)
    if len(years) == 0:
        return []
    segs = [[years[0]]]
    for y in years[1:]:
        if (y - segs[-1][-1]) <= max_gap:
            segs[-1].append(y)
        else:
            segs.append([y])
    return [np.array(s, dtype=int) for s in segs]

#------------------------------------------------------------------
def reindex_monthly_full(df, date_col="date"):
    """
    Reindex a monthly dataframe to a full monthly timeline so missing months
    remain explicit as NaN and plots show gaps rather than artificial continuity.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    if d.empty:
        return d

    full_idx = pd.date_range(
        d[date_col].min(),
        d[date_col].max(),
        freq="MS"
    )

    d = (
        d.set_index(date_col)
         .reindex(full_idx)
         .rename_axis(date_col)
         .reset_index()
    )
    return d

#------------------------------------------------------------------
def longest_contiguous_month_block(dates, max_gap_months=1):
    """
    Find the longest contiguous block in a monthly datetime series.
    dates: array-like of monthly timestamps
    returns: DatetimeIndex of the longest block
    """
    dt = pd.to_datetime(pd.Series(dates).dropna().unique())
    dt = pd.DatetimeIndex(dt).sort_values()

    if len(dt) == 0:
        return pd.DatetimeIndex([])

    segs = [[dt[0]]]

    for t in dt[1:]:
        prev = segs[-1][-1]
        month_diff = (t.year - prev.year) * 12 + (t.month - prev.month)

        if month_diff <= max_gap_months:
            segs[-1].append(t)
        else:
            segs.append([t])

    segs = [pd.DatetimeIndex(s) for s in segs]
    return max(segs, key=len)

#-----------------------------------------------------------------------------
def plot_interannual_variability_2x2(
    annual_df,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    figsize=(14, 9),
    add_trend_band=True,
    add_stats_inset=True,
    n_boot=2000,
    ci=95,
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300)
    axes = axes.flatten()

    for ax, region in zip(axes, regions):
        dfr = annual_df[annual_df["region"] == region].copy().sort_values("year")
        years = dfr["year"].values.astype(float)

        # --- time series ---
        ax.plot(years, dfr[ref].values, lw=3.2, color="b", label="Buoy")

        for p in products:
            if p == ref:
                continue
            ax.plot(years, dfr[p].values, lw=2.4, color=product_colors[p], label=p)

        # --- buoy trend + CI ---
        if add_trend_band:
            y = dfr[ref].values.astype(float)
            out = bootstrap_trend_band(years, y, n_boot=n_boot, ci=ci)
            if out is not None:
                ax.plot(out["xgrid"], out["mid"], lw=2.2, color="k", label="Buoy trend")
                ax.fill_between(out["xgrid"], out["lo"], out["hi"], alpha=0.15)

        # --- inset σ + r table (optional) ---
        if add_stats_inset:
            tab = std_and_corr_table(dfr, ref=ref, products=products)
            tab_show = tab.copy()
            tab_show["σ(annual)"] = tab_show["σ(annual)"].map(lambda v: f"{v:.2f}")
            tab_show["r vs buoy"] = tab_show["r vs buoy"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "nan")
            tab_show = tab_show.head(5)

            cell_text = tab_show[["σ(annual)", "r vs buoy"]].values.tolist()
            row_labels = tab_show["Product"].tolist()
            col_labels = ["σ", "r"]

            table = ax.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=col_labels,
                cellLoc="center",
                loc="upper right",
                bbox=[0.62, 0.55, 0.36, 0.40],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)

        ax.set_title(region, fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax.set_ylabel("Annual mean rainfall (mm/day)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)

    # turn off unused axes if regions < 4
    for i in range(len(regions), 4):
        axes[i].axis("off")

    # one legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=9, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Interannual variability (annual means): buoy vs products",
                 fontsize=15, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# -----------------------------
# 2x2 interannual plot (gap-aware trend/CI)
# -----------------------------


def add_trend_summary_box(ax, out, ci=95, units="mm day$^{-1}$ yr$^{-1}$"):
    """Bottom-right text box with slope + CI + significance flag."""
    if out is None:
        return
    sig = (out["slope_lo"] > 0) or (out["slope_hi"] < 0)
    sig_txt = "YES" if sig else "NO"

    txt = (
        f"Slope = {out['slope_mean']:.3g}\n"
        f"{ci}% CI = [{out['slope_lo']:.3g}, {out['slope_hi']:.3g}]\n"
        f"Significant? {sig_txt}\n"
        f"({units})"
    )
    ax.text(
        0.55, 0.02, txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

def plot_interannual_variability_2x2_gapaware(
    annual_df,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    figsize=(14, 9),
    add_trend_band=True,
    n_boot=2000,
    ci=95,
    max_gap_years=1,
    min_days_ref=365,
    require_all_products=False,
    legend_ncol=6,
    legend_fontsize=12,
    show_product_slopes=True,   # NEW
    region_names=Buoy_REGION_NAMES,
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300)
    axes = axes.flatten()

    for k, (ax, region) in enumerate(zip(axes, regions)):
        dfr = annual_df[annual_df["region"] == region].copy().sort_values("year")

        # ---- buoy validity mask (coverage + finite) ----
        ncol = f"n_{ref}"
        if ncol in dfr.columns:
            ok_ref = (
                np.isfinite(dfr[ref].values.astype(float)) &
                (dfr[ncol].values.astype(float) >= float(min_days_ref))
            )
        else:
            ok_ref = np.isfinite(dfr[ref].values.astype(float))

        if require_all_products:
            ok_all = ok_ref.copy()
            for p in products:
                if p == ref:
                    continue
                if p in dfr.columns:
                    ok_all &= np.isfinite(dfr[p].values.astype(float))
            ok_ref = ok_all

        dfr["_ok_ref"] = ok_ref

        # ---- restrict plotting to main contiguous buoy block ----
        valid_years = dfr.loc[dfr["_ok_ref"], "year"].dropna().astype(int).values
        segs = contiguous_year_segments(valid_years, max_gap=max_gap_years)

        if len(segs) > 0:
            main_seg = max(segs, key=len)
            first_valid_year = int(main_seg.min())
            last_valid_year  = int(main_seg.max())
            dfr = dfr[(dfr["year"].astype(int) >= first_valid_year) &
                      (dfr["year"].astype(int) <= last_valid_year)].copy()

        if dfr.empty:
            ax.set_title(region, fontsize=13, fontweight="bold")
            ax.text(0.5, 0.5, "No valid annual data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        years = dfr["year"].astype(int).values

        # ---- plot series ----
        ax.plot(years, dfr[ref].values, lw=3.2, color="b", label="Buoy")
        for p in products:
            if p == ref:
                continue
            if p in dfr.columns:
                ax.plot(years, dfr[p].values, lw=2.4,
                        color=product_colors[p], label=p)

        # ---- compute buoy trend band on longest contiguous observed segment ----
        # buoy_out = None
        # if add_trend_band and ref in dfr.columns:
        #     y_buoy = dfr[ref].values.astype(float)
        #     m = np.isfinite(y_buoy) & np.isfinite(years.astype(float))
        #     years_obs = years[m]
        #     buoy_obs  = y_buoy[m]

        #     segs_obs = contiguous_segments(years_obs, max_gap=max_gap_years)
        #     if len(segs_obs) > 0:
        #         main_seg_obs = max(segs_obs, key=len)
        #         seg_mask = np.isin(years_obs, main_seg_obs)
        #         xs = years_obs[seg_mask].astype(float)
        #         ys = buoy_obs[seg_mask]

        #         buoy_out = bootstrap_trend_band(xs, ys, n_boot=n_boot, ci=ci, seed=42)
        #         if buoy_out is not None:
        #             line_label = "Buoy trend" if k == 0 else None
        #             band_label = f"Buoy trend ({ci}% bootstrap CI)" if k == 0 else None
        #             ax.plot(buoy_out["xgrid"], buoy_out["mid"], "k--", lw=2.0, label=line_label)
        #             ax.fill_between(buoy_out["xgrid"], buoy_out["lo"], buoy_out["hi"],
        #                             alpha=0.15, label=band_label)

        # # ---- slopes text stack (Buoy + products) ----
        # if show_product_slopes:
        #     slope_dict = {}

        #     # buoy slope: use bootstrap mean if available; otherwise OLS
        #     if buoy_out is not None:
        #         slope_dict["Buoy"] = buoy_out["slope_mean"]
        #     else:
        #         slope_dict["Buoy"] = slope_only_on_common_years(dfr, "year", ref)

        #     # product slopes: compute on the SAME years where buoy is finite+coverage
        #     # (within the already-trimmed dfr)
        #     ok_mask_local = np.isfinite(dfr[ref].values.astype(float))
        #     ncol_local = f"n_{ref}"
        #     if ncol_local in dfr.columns:
        #         ok_mask_local &= (dfr[ncol_local].values.astype(float) >= float(min_days_ref))

        #     for p in products:
        #         if p == ref:
        #             continue
        #         if p in dfr.columns:
        #             slope_dict[p] = slope_only_on_common_years(
        #                 dfr, "year", p, ok_mask=ok_mask_local
        #             )

        #     color_dict = {"Buoy": "b"}
        #     color_dict.update({p: product_colors[p] for p in products if p != ref})

        #     add_slope_stack(
        #         ax,
        #         slope_dict=slope_dict,
        #         color_dict=color_dict,
        #         x=0.02, y=0.96, dy=0.055,
        #         fmt="{name}: {slope:+.3f}",
        #         units=" mm day$^{-1}$ yr$^{-1}$",
        #         fontsize=10,
        #         title="Slope (OLS):" if buoy_out is None else "Slope:"
        #     )

        #     # optional: add buoy CI line only (one extra line, not too messy)
        #     if buoy_out is not None:
        #         ax.text(
        #             0.02, 0.96 - 0.055*(len(slope_dict)+0.3),
        #             f"Buoy {ci}% CI: [{buoy_out['slope_lo']:+.3f}, {buoy_out['slope_hi']:+.3f}]",
        #             transform=ax.transAxes, ha="left", va="top",
        #             fontsize=9, color="b"
        #         )

        # ---- cosmetics ----
        ax.set_title(region_names.get(region, region), fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.set_ylabel("[mm day$^{-1}$]", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)

        xmin, xmax = int(years.min()), int(years.max())
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(np.arange(xmin, xmax + 1, 2))

    for i in range(len(regions), 4):
        axes[i].axis("off")

    # shared legend below
    handles, labels = [], []
    for ax in axes[:min(len(regions), 4)]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)

    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h

    fig.legend(list(uniq.values()), list(uniq.keys()),
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=legend_ncol, frameon=False, fontsize=legend_fontsize)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# -----------------------------
# Main plotter (raw daily -> monthly -> rm13 -> annual -> plot)
# -----------------------------
def plot_2x2_annual_rm13_with_trends(
    df_raw,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    region_names=None,
    figsize=(14, 9),
    year_min=None,          # e.g. 2000
    year_max=None,          # e.g. 2020
    show_ci_band=True,
    ci=95,
    n_boot=2000,
    xtick_step=2,           # show every other year
    legend_ncol=6,
    legend_fontsize=12,
):
    region_names = region_names or {}

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300)
    axes = axes.flatten()

    for k, (ax, reg) in enumerate(zip(axes, regions)):

        # 1) monthly series from raw daily
        monthly = monthly_from_raw_daily(df_raw, reg, products)

        # 2) rm13 monthly
        monthly_rm13 = add_rm13_monthly(monthly, products)

        # 3) annual mean from rm13 monthly
        ann = annual_from_monthly_rm13(monthly_rm13, products)
        ann = annual_from_monthly_rm13(monthly_rm13, products)

        # --- MINIMAL FIX: drop years where buoy annual rm13 is NaN ---
        ann = ann[np.isfinite(ann[ref + "_rm13"].values)].copy()

        # (optional) also drop years where ALL products are NaN
        # rm_cols = [p + "_rm13" for p in products]
        # ann = ann[ann[rm_cols].notna().any(axis=1)].copy()

        years = ann["Year"].astype(int).values
        years = ann["Year"].astype(int).values

        # optional year range
        if year_min is not None:
            ann = ann[ann["Year"] >= year_min]
        if year_max is not None:
            ann = ann[ann["Year"] <= year_max]

        years = ann["Year"].astype(int).values
        if len(years) == 0:
            ax.set_title(region_names.get(reg, reg), fontsize=13, fontweight="bold")
            ax.text(0.5, 0.5, "No data in range", transform=ax.transAxes,
                    ha="center", va="center")
            ax.axis("off")
            continue

        # 4) plot annual rm13 means
        ax.plot(years, ann[ref + "_rm13"], lw=3.0, color="b", label="Buoy")

        for p in products:
            if p == ref:
                continue
            ax.plot(years, ann[p + "_rm13"], lw=2.2, color=product_colors[p], label=p)

        y_all = np.concatenate([ann[c].values for c in [ref+"_rm13"] + [p+"_rm13" for p in products if p != ref]])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size:
            ypad = 0.08 * (y_all.max() - y_all.min() + 1e-6)
            ax.set_ylim(y_all.min() - ypad, y_all.max() + ypad)

        # # 5) buoy trend + (optional) CI (gap-aware on annual years)
        # yb = ann[ref + "_rm13"].values.astype(float)
        # ok = np.isfinite(yb)
        # years_ok = years[ok]

        # out = None
        # if len(years_ok) >= 3:
        #     segs = contiguous_year_segments(years_ok, max_gap=1)
        #     main_seg = max(segs, key=len) if len(segs) else years_ok
        #     mask_main = np.isin(years, main_seg)
        #     years_main = years[mask_main]
        #     buoy_main  = yb[mask_main]

        #     out = bootstrap_trend_band_years(years_main, buoy_main, n_boot=n_boot, ci=ci, seed=42)
        #     if out is not None:
        #         ax.plot(out["xgrid"], out["mid"], "k--", lw=2.0, label="Buoy trend" if k == 0 else None)
        #         if show_ci_band:
        #             ax.fill_between(out["xgrid"], out["lo"], out["hi"], alpha=0.15,
        #                             label=f"Buoy trend ({ci}% bootstrap CI)" if k == 0 else None)

        # # 6) trend stats text (colored) — per decade
        # # --- trend stats text: 2 stacked groups (same x, different y bands) ---
        # x_text = 0.01

        # top_group = [ref, "GPCP v3.2", "GPCP v3.3"]          # keep up
        # low_group = ["ERA5", "IMERG v07", "MERRA2"]          # push down

        # # (optional) keep only those present in your products list
        # top_group = [p for p in top_group if p in products]
        # low_group = [p for p in low_group if p in products]

        # # y locations in Axes fraction
        # y_top0 = 0.97
        # y_low0 = 0.2          # <-- move this up/down to taste
        # dy_top = 0.075
        # dy_low = 0.075

        # def _fmt(name, s):
        #     return f"{name}: Trend={s['slope_dec']:+.3f}, p={s['p']:.3f}, Std={s['std']:.3f}"

        # # --- top band ---
        # y = y_top0
        # for p in top_group:
        #     yvals = ann[p + "_rm13"].values.astype(float)
        #     s = trend_stats_yearly(years, yvals)
        #     if s is None:
        #         continue
        #     ax.text(
        #         x_text, y, _fmt("Buoy" if p == ref else p, s),
        #         transform=ax.transAxes, ha="left", va="top",
        #         fontsize=12,
        #         color=("b" if p == ref else product_colors[p]),
        #         fontweight=("bold" if p == ref else "normal"),
        #         bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
        #     )
        #     y -= dy_top

        # # --- lower band ---
        # y = y_low0
        # for p in low_group:
        #     yvals = ann[p + "_rm13"].values.astype(float)
        #     s = trend_stats_yearly(years, yvals)
        #     if s is None:
        #         continue
        #     ax.text(
        #         x_text, y, _fmt(p, s),
        #         transform=ax.transAxes, ha="left", va="top",
        #         fontsize=12, color=product_colors[p],
        #         bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.2)
        #     )
        #     y -= dy_low

        # cosmetics
        ax.set_title(region_names.get(reg, reg), fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.set_ylabel("[mm day$^{-1}$]",
                      fontsize=12, fontweight="bold") # "Annual mean rainfall (mm day$^{-1}$)"
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        xmin, xmax = int(years.min()), int(years.max())
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(xmin, xmax + 1, xtick_step))

    # hide unused
    for i in range(len(regions), 4):
        axes[i].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=legend_ncol, frameon=False, fontsize=legend_fontsize)

    # fig.suptitle("Interannual variability (annual mean of 13-month smoothed monthly means): Buoy vs products",
    #              fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig, monthly


#----------------------------------------------------------------------------

def resample_to_new_res(obj, new_shape, xdim="lon", ydim="lat", crs="EPSG:4326",
                        keep_vars=None, drop_nonspatial=True):
    """
    Resample/reproject (in same CRS) to a new pixel shape using average resampling.
    Works for xarray.DataArray or xarray.Dataset.

    - If Dataset: reprojects only vars that have (ydim, xdim) dims.
      Non-spatial vars (e.g., time_bnds) are preserved by default.
    """

    def _prep(o):
        # rename to x/y for rioxarray
        if xdim in o.dims and ydim in o.dims:
            o = o.rename({xdim: "x", ydim: "y"})
        # write CRS + spatial dims
        o = o.rio.write_crs(crs, inplace=False)
        o = o.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        return o

    # -------- DataArray case --------
    if isinstance(obj, xr.DataArray):
        da = _prep(obj)
        da = da.rio.reproject(da.rio.crs, shape=new_shape, resampling=Resampling.bilinear)
        return da.rename({"y": ydim, "x": xdim})

    # -------- Dataset case --------
    if not isinstance(obj, xr.Dataset):
        raise TypeError("Input must be an xarray.DataArray or xarray.Dataset")

    ds = obj

    # choose which vars to resample
    if keep_vars is not None:
        spatial_vars = [v for v in keep_vars if v in ds.data_vars]
    else:
        # only vars that truly have spatial dims
        spatial_vars = [v for v in ds.data_vars if (xdim in ds[v].dims and ydim in ds[v].dims)]

    if len(spatial_vars) == 0:
        raise ValueError(f"No variables found with dims ({ydim}, {xdim}).")

    # keep non-spatial vars (like time_bnds) aside
    nonspatial_vars = [v for v in ds.data_vars if v not in spatial_vars]
    ds_nonspatial = ds[nonspatial_vars] if (drop_nonspatial is False and nonspatial_vars) else None

    # reproject spatial vars only
    out_vars = {}
    for v in spatial_vars:
        da = _prep(ds[v])
        da = da.rio.reproject(da.rio.crs, shape=new_shape, resampling=Resampling.bilinear)
        out_vars[v] = da.rename({"y": ydim, "x": xdim})

    ds_out = xr.Dataset(out_vars, coords={c: ds[c] for c in ds.coords if c in ["time", ydim, xdim] or c in ds.coords})

    # add back non-spatial vars if requested
    if ds_nonspatial is not None:
        ds_out = xr.merge([ds_out, ds_nonspatial])

    # keep attrs
    ds_out.attrs = ds.attrs
    return ds_out



def compute_pdf_hist_1d(values, bins, weights=None, smooth_window=3):
    x = np.asarray(values, dtype=float)
    m = np.isfinite(x)
    x = x[m]

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w[m]
        m2 = np.isfinite(w)
        x = x[m2]
        w = w[m2]

    hist_counts, bin_edges = np.histogram(x, bins=bins, weights=w, density=False)

    bin_widths = np.diff(bin_edges)
    total = np.sum(hist_counts)
    hist_density = hist_counts / (total * bin_widths) if total > 0 else np.zeros_like(hist_counts, dtype=float)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if smooth_window and smooth_window > 1:
        k = int(smooth_window)
        kernel = np.ones(k) / k
        hist_line = np.convolve(hist_density, kernel, mode="same")
    else:
        hist_line = hist_density.copy()

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "hist_bars": hist_density,
        "hist_line": hist_line,
        "hist_counts": hist_counts,
    }

def plot_pdf_from_pdfdict(ax, pdf, label, color, alpha_bar=0.18, lw=4):
    edges = pdf["bin_edges"]
    centers = pdf["bin_centers"]
    widths = np.diff(edges)

    ax.bar(centers, pdf["hist_bars"], width=widths, align="center",
           alpha=alpha_bar, edgecolor="none", color=color)
    ax.plot(centers, pdf["hist_line"], color=color, lw=lw, label=label)

def plot_global_and_regional_pdfs_regioncol(
    df,
    region_name,
    region_col="region",
    product_cols=("rain_rate", "GPCP v3.3"),
    colors=("black", "red"),
    bins=np.arange(0, 15.5, 0.5),
    smooth_window=3,
    weights_col=None,
    figsize=(14, 4.8),
    dpi=160,
):
    # ---- Global: use everything ----
    df_g = df

    # ---- Regional: subset by region ----
    df_r = df[df[region_col] == region_name]

    w_g = df_g[weights_col].values if weights_col is not None else None
    w_r = df_r[weights_col].values if weights_col is not None else None

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax0, ax1 = axes

    ax0.set_title("Global PDF", fontweight="bold")
    for col, c in zip(product_cols, colors):
        lbl = "Buoy" if col == "rain_rate" else col
        pdf = compute_pdf_hist_1d(df_g[col].values, bins=bins, weights=w_g, smooth_window=smooth_window)
        plot_pdf_from_pdfdict(ax0, pdf, label=lbl, color=c)
    ax0.set_xlabel("Precipitation (mm/day)")
    ax0.set_ylabel("Density")
    ax0.legend(frameon=False, loc="upper right")

    region_name_ = Buoy_REGION_NAMES.get(region_name, region_name)

    ax1.set_title(f"Regional PDF: {region_name_}", fontweight="bold")
    for col, c in zip(product_cols, colors):
        lbl = "Buoy" if col == "rain_rate" else col
        pdf = compute_pdf_hist_1d(df_r[col].values, bins=bins, weights=w_r, smooth_window=smooth_window)
        plot_pdf_from_pdfdict(ax1, pdf, label=lbl, color=c)
    ax1.set_xlabel("Precipitation (mm/day)")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    return fig, axes


#-----------------------------------------------------------------------------
def _plot_hist_only(ax, pdf, color="0.6", alpha=0.25, label=None):
    edges = pdf["bin_edges"]
    centers = pdf["bin_centers"]
    widths = np.diff(edges)
    ax.bar(
        centers, pdf["hist_bars"],
        width=widths, align="center",
        alpha=alpha, edgecolor="none", color=color,
        label=label
    )

def _plot_line_only(ax, pdf, label, color, lw=4):
    ax.plot(pdf["bin_centers"], pdf["hist_line"], color=color, lw=lw, label=label)

def plot_global_and_regional_pdfs_insitu_hist_only(
    df,
    region_name,
    region_col="region",
    insitu_col="rain_rate",
    line_cols=("rain_rate", "GPCP v3.2", "GPCP v3.3", "ERA5"),
    line_colors=("black", "blue", "red", "lime"),
    bins=np.arange(0, 15.5, 0.5),
    smooth_window=3,
    weights_col=None,
    hist_color="0.7",
    hist_alpha=0.25,
    figsize=(14, 4.8),
    dpi=160,
    legend_loc="upper right",
):
    """
    Global: uses full df
    Regional: df[df[region_col] == region_name]

    Bars: ONLY insitu_col
    Lines: all in line_cols (can include insitu_col too)
    Legend: includes a patch indicating bars are insitu histogram
    """
    df_g = df
    df_r = df[df[region_col] == region_name]

    w_g = df_g[weights_col].values if weights_col is not None else None
    w_r = df_r[weights_col].values if weights_col is not None else None

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    lbl = "Buoy" if insitu_col == "rain_rate" else insitu_col

    # ---------------- Global panel ----------------
    ax0.set_title("Global PDF", fontweight="bold")

    # insitu bars
    pdf_insitu_g = compute_pdf_hist_1d(df_g[insitu_col].values, bins=bins, weights=w_g, smooth_window=1)
    _plot_hist_only(ax0, pdf_insitu_g, color=hist_color, alpha=hist_alpha)

    # lines
    line_handles_g = []
    for col, c in zip(line_cols, line_colors):
        pdf_line = compute_pdf_hist_1d(df_g[col].values, bins=bins, weights=w_g, smooth_window=smooth_window)
        h = ax0.plot(pdf_line["bin_centers"], pdf_line["hist_line"], color=c, lw=4, label=col)[0]
        line_handles_g.append(h)

    ax0.set_xlabel("Precipitation (mm/day)")
    ax0.set_ylabel("Density")

    # legend with explicit patch for histogram
    hist_patch = Patch(facecolor=hist_color, alpha=hist_alpha, edgecolor="none",
                       label=f"{lbl} histogram")
    ax0.legend(handles=[hist_patch] + line_handles_g, frameon=False, loc=legend_loc)

    # ---------------- Regional panel ----------------
    region_name_ = Buoy_REGION_NAMES.get(region_name, region_name)
    ax1.set_title(f"Regional PDF: {region_name_}", fontweight="bold")

    pdf_insitu_r = compute_pdf_hist_1d(df_r[insitu_col].values, bins=bins, weights=w_r, smooth_window=1)
    _plot_hist_only(ax1, pdf_insitu_r, color=hist_color, alpha=hist_alpha)

    line_handles_r = []
    for col, c in zip(line_cols, line_colors):
        pdf_line = compute_pdf_hist_1d(df_r[col].values, bins=bins, weights=w_r, smooth_window=smooth_window)
        h = ax1.plot(pdf_line["bin_centers"], pdf_line["hist_line"], color=c, lw=4, label=col)[0]
        line_handles_r.append(h)

    ax1.set_xlabel("Precipitation (mm/day)")
    ax1.set_ylabel("Density")

    hist_patch_r = Patch(facecolor=hist_color, alpha=hist_alpha, edgecolor="none",
                         label=f"{lbl} histogram")
    ax1.legend(handles=[hist_patch_r] + line_handles_r, frameon=False, loc=legend_loc)

    plt.tight_layout()
    return fig, (ax0, ax1)

# -----------------------------------------------------------------------------
# ------------------------------------------------------------
# 2) Panel scatter plot: rows=regions, cols=products
# ------------------------------------------------------------
def plot_anomaly_scatter_by_region_product(
    df,
    regions,
    ref_col="rain_rate",
    product_cols=("GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"),
    region_labels=None,
    product_colors=None,
    date_col="date",
    region_col="region",
    use_monthly_clim_anoms=True,
    figsize=(22, 14),
    marker_size=18,
    point_alpha=0.75,
    reg_line_color="red",
    one_to_one_color="black",
    zero_line_color="k",
    xlim=None,
    ylim=None,
    equal_axes=True,
    savepath=None,
):
    """
    Make scatter panels in anomaly space:
        x = ref anomaly
        y = product anomaly

    Layout:
        rows = regions
        cols = products

    Metrics shown in each panel:
        slope, std dev of residuals, correlation
    """

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    all_cols = [ref_col] + list(product_cols)

    # Create anomaly columns
    if use_monthly_clim_anoms:
        d = make_region_monthly_anomaly_df(
            d,
            products=all_cols,
            region_col=region_col,
            date_col=date_col,
        )
    else:
        # simple mean anomalies by region
        for c in all_cols:
            d[f"{c}_anom"] = d[c] - d.groupby(region_col)[c].transform("mean")

    nrows = len(regions)
    ncols = len(product_cols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=250,
        squeeze=False
    )

    for i, region in enumerate(regions):
        dfr = d[d[region_col] == region].copy()

        region_title = region_labels.get(region, region) if region_labels else region

        for j, prod in enumerate(product_cols):
            ax = axes[i, j]

            xcol = f"{ref_col}_anom"
            ycol = f"{prod}_anom"

            dd = dfr[[xcol, ycol]].dropna().copy()
            x = dd[xcol].values.astype(float)
            y = dd[ycol].values.astype(float)

            # scatter
            ax.scatter(
                x, y,
                s=marker_size,
                alpha=point_alpha,
                color=product_colors.get(prod, "0.35") if product_colors else "0.35",
                edgecolor="none"
            )

            # 1:1 line and regression line
            if len(dd) >= 2:
                xmin_ = np.nanmin(x)
                xmax_ = np.nanmax(x)
                ymin_ = np.nanmin(y)
                ymax_ = np.nanmax(y)

                if equal_axes:
                    lim0 = np.nanmin([xmin_, ymin_])
                    lim1 = np.nanmax([xmax_, ymax_])
                    xrng = (lim0, lim1)
                    yrng = (lim0, lim1)
                else:
                    xrng = (xmin_, xmax_)
                    yrng = (ymin_, ymax_)

                # allow manual override
                if xlim is not None:
                    xrng = xlim
                if ylim is not None:
                    yrng = ylim

                # zero lines
                ax.axhline(0, color=zero_line_color, lw=1.0, ls=(0, (2, 4)), alpha=0.8)
                ax.axvline(0, color=zero_line_color, lw=1.0, ls=(0, (2, 4)), alpha=0.8)

                # 1:1 line
                lo = min(xrng[0], yrng[0])
                hi = max(xrng[1], yrng[1])
                ax.plot([lo, hi], [lo, hi], color=one_to_one_color, lw=1.2)

                # regression line
                slope, intercept = np.polyfit(x, y, 1)
                xx = np.linspace(lo, hi, 200)
                yy = slope * xx + intercept
                ax.plot(xx, yy, color=reg_line_color, lw=1.5)

                # stats
                corr = np.corrcoef(x, y)[0, 1]

                # "Std Dev" similar to attached figure:
                # std of residuals around regression line
                resid = y - (slope * x + intercept)
                std_dev = np.std(resid, ddof=1) if len(resid) > 1 else np.nan

                ax.text(
                    0.05, 0.94,
                    f"Slope: {slope:.2f}\nStd Dev: {std_dev:.2f}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12,
                    color="red",
                    fontweight="bold"
                )

                ax.text(
                    0.95, 0.06,
                    f"Correlation: {corr:.2f}",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=12,
                    color="black",
                    fontweight="bold"
                )

                ax.set_xlim(xrng)
                ax.set_ylim(yrng)

            else:
                ax.text(
                    0.5, 0.5, "Insufficient data",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold"
                )

            # titles
            if i == 0:
                ax.set_title(prod, fontsize=15, fontweight="bold")

            if j == 0:
                ax.set_ylabel(
                    f"{region_title}\n{prod} anomaly\n(mm day$^{{-1}}$)",
                    fontsize=13,
                    fontweight="bold"
                )
            else:
                ax.set_ylabel("")

            if i == nrows - 1:
                ax.set_xlabel(
                    "Buoy anomaly (mm day$^{-1}$)",
                    fontsize=13,
                    fontweight="bold"
                )
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="both", labelsize=11, width=1.2)
            ax.minorticks_on()

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    return fig

#-----------------------------------------------------------------------------
def plot_deseasonalized_anomaly_scatter(
    df_anom,
    regions,
    ref_col="rain_rate",
    product_cols=("GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"),
    region_labels=None,
    product_colors=None,
    region_col="region",
    figsize=(20, 12),
    savepath=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    nrows = len(regions)
    ncols = len(product_cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=250, squeeze=False)

    for i, region in enumerate(regions):
        dfr = df_anom[df_anom[region_col] == region].copy()

        for j, prod in enumerate(product_cols):
            ax = axes[i, j]

            xcol = f"{ref_col}_anom"
            ycol = f"{prod}_anom"

            dd = dfr[[xcol, ycol]].dropna()
            x = dd[xcol].values
            y = dd[ycol].values

            if len(dd) < 2:
                ax.text(0.5, 0.5, "Insufficient data",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            # scatter
            ax.scatter(
                x, y,
                s=18,
                alpha=0.75,
                color= 'k',
                edgecolor="none"
            ) # product_colors.get(prod, "0.4") if product_colors else "0.4"

            # symmetric limits around zero
            lim = 5#np.nanmax(np.abs(np.r_[x, y]))
            lim = np.ceil(lim * 1.05)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            # zero lines
            ax.axhline(0, color="k", lw=1.0, ls=(0, (2, 4)))
            ax.axvline(0, color="k", lw=1.0, ls=(0, (2, 4)))

            # 1:1 line
            ax.plot([-lim, lim], [-lim, lim], color="k", lw=1.2)

            # regression
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(-lim, lim, 200)
            ax.plot(xx, slope * xx + intercept, color="red", lw=1.4)

            # metrics
            corr = np.corrcoef(x, y)[0, 1]
            resid = y - (slope * x + intercept)
            std_dev = np.std(resid, ddof=1)

            ax.text(
                0.04, 0.96,
                f"Slope: {slope:.2f}\nStd Dev: {std_dev:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=14, color="k", fontweight="bold"
            )

            ax.text(
                0.96, 0.05,
                f"CC: {corr:.2f}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=14, color="black", fontweight="bold"
            )

            if i == 0:
                ax.set_title(prod, fontsize=15, fontweight="bold")

            if j == 0:
                reglbl = region_labels.get(region, region) if region_labels else region
                ax.set_ylabel(f"{reglbl}\nProduct anomaly\n[mm day$^{{-1}}$]",
                              fontsize=15, fontweight="bold")

            if i == nrows - 1:
                ax.set_xlabel("Buoy anomaly [mm day$^{-1}$]",
                              fontsize=15, fontweight="bold")

            ax.tick_params(labelsize=14)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig


#-----------------------------------------------------------------------------
def plot_region_monthly_anomalies_rm13(
    df_raw,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    region_col="region",
    date_col="date",
    region_names=None,
    min_days_month=20,
    min_days_month_ref=25,
    apply_rm13=True,
    figsize=(14, 9),
    sharex=True,
    sharey=False,
    linewidth_ref=2.8,
    linewidth_prod=2.0,
    alpha=0.95,
    savepath=None,
):
    """
    Plot deseasonalized regional monthly precipitation anomalies,
    optionally smoothed with a 13-month centered running mean.

    Workflow per region:
      raw daily -> regional daily mean -> monthly mean -> deseasonalize -> rm13

    Parameters
    ----------
    df_raw : pd.DataFrame
        Daily matched dataframe with columns [date, region, rain_rate, products...]
    regions : list
        Region names to plot, e.g. ["ENP","WNP","IND","ATL"]
    products : list
        Includes reference column, e.g.
        ["rain_rate","GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2"]
    product_colors : dict
        Color mapping for products
    ref : str
        Reference/in situ column
    region_names : dict or None
        Pretty names for regions
    """

    region_names = region_names or {}

    nreg = len(regions)
    ncols = 2 if nreg > 1 else 1
    nrows = int(np.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=300,
        sharex=sharex,
        sharey=sharey,
        squeeze=False
    )
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]

        # --------------------------------------------------
        # Step 1: collapse raw daily to one daily value/region
        # --------------------------------------------------
        daily_reg = daily_region_mean(
            df_raw,
            products=products,
            date_col=date_col,
            region_col=region_col
        )

        # --------------------------------------------------
        # Step 2: monthly means with coverage
        # --------------------------------------------------
        monthly = monthly_mean_with_day_coverage(
            daily_reg,
            products=products,
            region=region,
            min_days=min_days_month,
            ref=ref,
            min_days_ref=min_days_month_ref,
            date_col=date_col,
            region_col=region_col
        )

        if monthly.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, fontweight="bold")
            ax.set_title(region_names.get(region, region), fontweight="bold")
            continue

        # --------------------------------------------------
        # Step 3: deseasonalize monthly series
        # --------------------------------------------------
        monthly_anom = deseasonalize_monthly(
            df_monthly=monthly,
            products=products,
            time_col="date",
            region_col=region_col
        )

        # --------------------------------------------------
        # Step 4: optional 13-month running mean on anomalies
        # --------------------------------------------------
        anom_cols = [f"{p}_anom" for p in products]

        plot_df = monthly_anom[["date"] + anom_cols].copy()

        if apply_rm13:
            plot_df = add_rm13_monthly(
                plot_df,
                products=anom_cols,
                date_col="date"
            )
            cols_to_plot = [f"{c}_rm13" for c in anom_cols]
        else:
            cols_to_plot = anom_cols

        # --------------------------------------------------
        # Step 5: plot
        # --------------------------------------------------
        t = pd.to_datetime(plot_df["date"])

        # reference
        ref_plot_col = f"{ref}_anom" + ("_rm13" if apply_rm13 else "")
        ax.plot(
            t,
            plot_df[ref_plot_col],
            lw=linewidth_ref,
            color="k",
            alpha=alpha,
            label="Buoy"
        )

        # products
        for p in products:
            if p == ref:
                continue
            pcol = f"{p}_anom" + ("_rm13" if apply_rm13 else "")
            ax.plot(
                t,
                plot_df[pcol],
                lw=linewidth_prod,
                color=product_colors.get(p, None),
                alpha=alpha,
                label=p
            )

        ax.axhline(0, color="0.6", lw=1.0, ls=(0, (4, 4)))
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_title(region_names.get(region, region), fontsize=14, fontweight="bold")
        ax.set_ylabel("Precipitation anomaly\n(mm day$^{-1}$)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)

        # optional correlations placed in panel
        y0 = 0.06
        dy = 0.07
        for k, p in enumerate(products):
            if p == ref:
                continue

            pcol = f"{p}_anom" + ("_rm13" if apply_rm13 else "")
            dd = plot_df[[ref_plot_col, pcol]].dropna()

            if len(dd) >= 2:
                cc = np.corrcoef(dd[ref_plot_col], dd[pcol])[0, 1]
                ax.text(
                    0.02 + (k % 2) * 0.42,
                    y0 + (k // 2) * dy,
                    f"Buoy vs {p}: ({cc:.2f})",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=product_colors.get(p, "k"),
                    fontweight="bold"
                )

        if i >= len(regions) - ncols:
            ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    # turn off unused axes
    for j in range(len(regions), len(axes)):
        axes[j].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=11
    )

    title_txt = "Monthly domain-mean precipitation anomalies"
    if apply_rm13:
        title_txt = "Monthly (13-month-running-mean) domain-mean precipitation anomalies"

    fig.suptitle(title_txt, fontsize=17, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    return fig

#--------------------------------------------------------------
def plot_region_monthly_anomalies_rm13_gapaware(
    df_raw,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    region_col="region",
    date_col="date",
    region_names=None,
    min_days_month=20,
    min_days_month_ref=25,
    apply_rm13=True,
    figsize=(14, 9),
    sharex=False,
    sharey=False,
    linewidth_ref=2.8,
    linewidth_prod=2.0,
    alpha=0.95,
    comparison_mode="buoy",   # "buoy" or "common"
    max_gap_months=1,
    legend_ncol=3,
    savepath=None,
):
    """
    Gap-aware 2x2 plot of deseasonalized monthly anomalies with optional 13-mo RM.

    comparison_mode:
      - "buoy"  -> define valid plotting window from buoy only
      - "common" -> define valid plotting window from months where all products are available
    """

    region_names = region_names or {}

    nreg = len(regions)
    ncols = 2 if nreg > 1 else 1
    nrows = int(np.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=300,
        sharex=sharex,
        sharey=sharey,
        squeeze=False
    )
    axes = axes.flatten()

    # collapse once outside the loop
    daily_reg_all = daily_region_mean(
        df_raw,
        products=products,
        date_col=date_col,
        region_col=region_col
    )

    for i, region in enumerate(regions):
        ax = axes[i]

        # -------------------------------
        # region monthly mean with coverage
        # -------------------------------
        monthly = monthly_mean_with_day_coverage(
            daily_reg_all,
            products=products,
            region=region,
            min_days=min_days_month,
            ref=ref,
            min_days_ref=min_days_month_ref,
            date_col=date_col,
            region_col=region_col
        )

        if monthly.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, fontweight="bold")
            ax.set_title(region_names.get(region, region), fontweight="bold")
            continue

        # -------------------------------
        # deseasonalize
        # -------------------------------
        monthly_anom = deseasonalize_monthly(
            df_monthly=monthly,
            products=products,
            time_col="date",
            region_col=region_col
        )

        anom_cols = [f"{p}_anom" for p in products]
        plot_df = monthly_anom[["date"] + anom_cols].copy()

        # -------------------------------
        # full monthly reindex for gap-aware plotting
        # -------------------------------
        plot_df = reindex_monthly_full(plot_df, date_col="date")

        # -------------------------------
        # 13-month running mean
        # -------------------------------
        if apply_rm13:
            plot_df = add_rm13_monthly(
                plot_df,
                products=anom_cols,
                date_col="date"
            )
            plot_cols = {p: f"{p}_anom_rm13" for p in products}
        else:
            plot_cols = {p: f"{p}_anom" for p in products}

        # -------------------------------
        # choose comparison window
        # -------------------------------
        if comparison_mode == "common":
            valid_mask = np.ones(len(plot_df), dtype=bool)
            for p in products:
                valid_mask &= np.isfinite(plot_df[plot_cols[p]].values)
        else:
            valid_mask = np.isfinite(plot_df[plot_cols[ref]].values)

        valid_dates = plot_df.loc[valid_mask, "date"]

        if len(valid_dates) == 0:
            ax.text(0.5, 0.5, "No valid anomaly period", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, fontweight="bold")
            ax.set_title(region_names.get(region, region), fontweight="bold")
            continue

        main_block = longest_contiguous_month_block(valid_dates, max_gap_months=max_gap_months)

        if len(main_block) > 0:
            t0, t1 = main_block.min(), main_block.max()
            plot_df = plot_df[(plot_df["date"] >= t0) & (plot_df["date"] <= t1)].copy()

        # -------------------------------
        # plotting
        # -------------------------------
        t = pd.to_datetime(plot_df["date"])

        ref_plot_col = plot_cols[ref]
        ax.plot(
            t, plot_df[ref_plot_col],
            lw=linewidth_ref, color="k", alpha=alpha, label="Buoy"
        )

        for p in products:
            if p == ref:
                continue
            ax.plot(
                t, plot_df[plot_cols[p]],
                lw=linewidth_prod,
                color=product_colors.get(p, None),
                alpha=alpha,
                label=p
            )

        ax.axhline(0, color="0.6", lw=1.0, ls=(0, (4, 4)))
        ax.grid(True, linestyle="--", alpha=0.35)

        # correlations inside panel
        y0 = 0.06
        dy = 0.07
        ktxt = 0
        for p in products:
            if p == ref:
                continue
            dd = plot_df[[ref_plot_col, plot_cols[p]]].dropna()
            if len(dd) >= 2:
                cc = np.corrcoef(dd[ref_plot_col], dd[plot_cols[p]])[0, 1]
                ax.text(
                    0.03 + (ktxt % 2) * 0.42,
                    y0 + (ktxt // 2) * dy,
                    f"Buoy vs {p}: ({cc:.2f})",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=product_colors.get(p, "k"),
                    fontweight="bold"
                )
                ktxt += 1

        ax.set_title(region_names.get(region, region), fontsize=14, fontweight="bold")
        ax.set_ylabel("Precipitation anomaly\n(mm day$^{-1}$)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)

        if i >= len(regions) - ncols:
            ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    # turn off unused
    for j in range(len(regions), len(axes)):
        axes[j].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    title_txt = "Monthly domain-mean precipitation anomalies"
    if apply_rm13:
        title_txt = "Monthly (13-month-running-mean) domain-mean precipitation anomalies"

    fig.suptitle(title_txt, fontsize=17, fontweight="bold", y=0.98)

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_ncol,
        frameon=False,
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    return fig


#----------------------------------------------------------------------------
def plot_region_monthly_anomalies_rm13_gapaware_v2(
    df_raw,
    regions,
    products,
    product_colors,
    ref="rain_rate",
    region_col="region",
    date_col="date",
    region_names=None,
    min_days_month=20,
    min_days_month_ref=25,
    apply_rm13=True,
    figsize=(14, 9),
    sharex=False,
    sharey=False,
    linewidth_ref=2.8,
    linewidth_prod=2.0,
    alpha=0.95,
    reindex_full=True,
    year_min=None,
    year_max=None,
    legend_ncol=3,
    savepath=None,
):
    """
    Gap-aware monthly anomaly plot using the SAME philosophy as earlier plots:
      - preserve each region's natural valid period
      - do not trim to longest contiguous block
      - missing months remain NaN and appear as gaps
    """

    region_names = region_names or {}

    nreg = len(regions)
    ncols = 2 if nreg > 1 else 1
    nrows = int(np.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=300,
        sharex=sharex,
        sharey=sharey,
        squeeze=False
    )
    axes = axes.flatten()

    # collapse once, same as your earlier workflow style
    daily_reg_all = daily_region_mean(
        df_raw,
        products=products,
        date_col=date_col,
        region_col=region_col
    )

    for i, region in enumerate(regions):
        ax = axes[i]

        # ----------------------------------------
        # 1) monthly means with coverage by region
        # ----------------------------------------
        monthly = monthly_mean_with_day_coverage(
            daily_reg_all,
            products=products,
            region=region,
            min_days=min_days_month,
            ref=ref,
            min_days_ref=min_days_month_ref,
            date_col=date_col,
            region_col=region_col
        )

        if monthly.empty:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, fontweight="bold"
            )
            ax.set_title(region_names.get(region, region), fontweight="bold")
            continue

        # ----------------------------------------
        # 2) deseasonalize monthly series
        # ----------------------------------------
        monthly_anom = deseasonalize_monthly(
            df_monthly=monthly,
            products=products,
            time_col="date",
            region_col=region_col
        )

        anom_cols = [f"{p}_anom" for p in products]
        plot_df = monthly_anom[["date"] + anom_cols].copy()

        # ----------------------------------------
        # 3) preserve full monthly timeline (gaps explicit)
        # ----------------------------------------
        if reindex_full:
            plot_df = reindex_monthly_full(plot_df, date_col="date")

        # ----------------------------------------
        # 4) optional year clipping (light touch only)
        # ----------------------------------------
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        if year_min is not None:
            plot_df = plot_df[plot_df["date"].dt.year >= year_min]
        if year_max is not None:
            plot_df = plot_df[plot_df["date"].dt.year <= year_max]

        # ----------------------------------------
        # 5) 13-month running mean
        # ----------------------------------------
        if apply_rm13:
            plot_df = add_rm13_monthly(
                plot_df,
                products=anom_cols,
                date_col="date"
            )
            plot_cols = {p: f"{p}_anom_rm13" for p in products}
        else:
            plot_cols = {p: f"{p}_anom" for p in products}

        # ----------------------------------------
        # 6) plot full available range (NO trimming)
        # ----------------------------------------
        t = pd.to_datetime(plot_df["date"])
        ref_plot_col = plot_cols[ref]

        ax.plot(
            t,
            plot_df[ref_plot_col],
            lw=linewidth_ref,
            color="k",
            alpha=alpha,
            label="Buoy"
        )

        for p in products:
            if p == ref:
                continue
            ax.plot(
                t,
                plot_df[plot_cols[p]],
                lw=linewidth_prod,
                color=product_colors.get(p, None),
                alpha=alpha,
                label=p
            )

        ax.axhline(0, color="0.6", lw=1.0, ls=(0, (4, 4)))
        ax.grid(True, linestyle="--", alpha=0.35)

        # correlations inside panel
        y0 = 0.06
        dy = 0.07
        ktxt = 0
        for p in products:
            if p == ref:
                continue
            dd = plot_df[[ref_plot_col, plot_cols[p]]].dropna()
            if len(dd) >= 2:
                cc = np.corrcoef(dd[ref_plot_col], dd[plot_cols[p]])[0, 1]
                ax.text(
                    0.03 + (ktxt % 2) * 0.42,
                    y0 + (ktxt // 2) * dy,
                    f"Buoy vs {p}: ({cc:.2f})",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=product_colors.get(p, "k"),
                    fontweight="bold"
                )
                ktxt += 1

        ax.set_title(region_names.get(region, region), fontsize=14, fontweight="bold")
        ax.set_ylabel("Precipitation anomaly\n(mm day$^{-1}$)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)

        if i >= len(regions) - ncols:
            ax.set_xlabel("Year", fontsize=12, fontweight="bold")

    # turn off unused axes
    for j in range(len(regions), len(axes)):
        axes[j].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    title_txt = "Monthly domain-mean precipitation anomalies"
    if apply_rm13:
        title_txt = "Monthly (13-month-running-mean) domain-mean precipitation anomalies"

    fig.suptitle(title_txt, fontsize=17, fontweight="bold", y=0.98)

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_ncol,
        frameon=False,
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    return fig


#-------------------------------------------------
def plot_region_anomaly_timeseries_from_df(
    df_anom,
    regions=("ENP", "WNP", "IND", "ATL"),
    region_col="region",
    date_col="date",
    ref_col="rain_rate_anom",
    anomaly_cols=(
        "rain_rate_anom",
        "GPCP v3.2_anom",
        "GPCP v3.3_anom",
        "ERA5_anom",
        "IMERG v07_anom",
        "MERRA2_anom",
    ),
    region_names=Buoy_REGION_NAMES,
    product_colors=product_colors,
    figsize=(14, 9),
    sharex=False,
    sharey=False,
    linewidth_ref=3.0,
    linewidth_prod=2.2,
    alpha=0.95,
    title="Monthly (13-month-running-mean) domain-mean precipitation anomalies",
    ylabel="Precipitation anomaly\n[mm day$^{-1}$]",
    legend_ncol=3,
    savepath=None,
):
    """
    Plot regional anomaly time series from an already-prepared anomaly dataframe.

    Expected dataframe columns
    --------------------------
    date, region, and anomaly columns such as:
      rain_rate_anom
      GPCP v3.2_anom
      GPCP v3.3_anom
      ERA5_anom
      IMERG v07_anom
      MERRA2_anom
    """

    d = df_anom.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    if region_names is None:
        region_names = Buoy_REGION_NAMES

    if product_colors is None:
        product_colors = {
            "GPCP v3.2": "#4c4c4c",
            "GPCP v3.3": "#1f77b4",
            "ERA5": "#d62728",
            "IMERG v07": "#2ca02c",
            "MERRA2": "#ff7f0e",
            "Buoy": "#0820d4",
        }

    # map anomaly column -> legend label
    col_to_label = {
        "rain_rate_anom": "Buoy",
        "GPCP v3.2_anom": "GPCP v3.2",
        "GPCP v3.3_anom": "GPCP v3.3",
        "ERA5_anom": "ERA5",
        "IMERG v07_anom": "IMERG v07",
        "MERRA2_anom": "MERRA2",
    }

    nreg = len(regions)
    ncols = 2 if nreg > 1 else 1
    nrows = int(np.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=300,
        sharex=sharex,
        sharey=sharey,
        squeeze=False
    )
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]

        dfr = (
            d[d[region_col] == region]
            .sort_values(date_col)
            .copy()
        )

        if dfr.empty:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, fontweight="bold"
            )
            ax.set_title(region_names.get(region, region), fontsize=14, fontweight="bold")
            continue

        # plot reference first
        if ref_col in dfr.columns:
            ax.plot(
                dfr[date_col],
                dfr[ref_col],
                lw=linewidth_ref,
                color=product_colors.get("Buoy", "b"),
                alpha=alpha,
                label="Buoy",
            )

        # plot other anomaly series
        for col in anomaly_cols:
            if col == ref_col:
                continue
            if col not in dfr.columns:
                continue

            label = col_to_label.get(col, col.replace("_anom", ""))
            ax.plot(
                dfr[date_col],
                dfr[col],
                lw=linewidth_prod,
                color=product_colors.get(label, None),
                alpha=alpha,
                label=label,
            )

        # zero line
        ax.axhline(0, color="0.6", lw=1.0, ls=(0, (4, 4)))

        # title and labels
        ax.set_title(region_names.get(region, region), fontsize=16, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

        # nicer ticks
        ax.tick_params(axis="both", labelsize=10)

        # correlation text inside each panel
        y0 = 0.06
        dy = 0.07
        ktxt = 0

        for col in anomaly_cols:
            if col == ref_col or col not in dfr.columns:
                continue

            dd = dfr[[ref_col, col]].dropna()
            if len(dd) >= 2:
                cc = np.corrcoef(dd[ref_col], dd[col])[0, 1]
                label = col_to_label.get(col, col.replace("_anom", ""))

                ax.text(
                    0.03 + (ktxt % 2) * 0.42,
                    y0 + (ktxt // 2) * dy,
                    f"Buoy vs {label}: ({cc:.2f})",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=product_colors.get(label, "k"),
                    fontweight="bold",
                )
                ktxt += 1

    # turn off unused axes
    for j in range(len(regions), len(axes)):
        axes[j].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_ncol,
        frameon=False,
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    return fig
#------------------------------------------------

def make_discrete_norm_and_cmap_(metric):
    style = METRIC_STYLE[metric]
    bounds = style["bounds"]
    cmap = style["cmap"]
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds, style["label"]

def make_discrete_norm_and_cmap(metric, metric_style):
    style = metric_style[metric]
    bounds = style["bounds"]
    cmap = style["cmap"]
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds, style["label"]



def get_metric_style(metric):
    """
    Returns cmap, norm, and whether higher is better.
    Adjust ranges if needed.
    """
    if metric in ["POD", "HSS", "CC", "CSI", "Accuracy"]:
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
    elif metric in ["FAR", "POFD"]:
        cmap = plt.cm.viridis_r
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
    elif metric in ["RMSE", "MAE"]:
        cmap = plt.cm.viridis_r
        # panel-specific scaling may also be used; this is a generic default
        norm = None
    elif metric == "Bias":
        cmap = plt.cm.RdBu_r
        norm = None
    else:
        cmap = plt.cm.viridis
        norm = None

    return cmap, norm

def compute_panel_norm(df_panel, metric):
    """
    If metric-specific norm was not fixed, define from panel data.
    """
    cmap, norm = get_metric_style(metric)
    if norm is not None:
        return cmap, norm

    vals = df_panel["value"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(vals) == 0:
        return cmap, mpl.colors.Normalize(vmin=0, vmax=1)

    if metric == "Bias":
        vmax = np.nanmax(np.abs(vals))
        vmax = max(vmax, 1.0)
        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return cmap, norm

def add_base_map(
    ax,
    extent=(-180, 180, -30, 60),
    show_left_labels=True,
    show_bottom_labels=True,
    land_color="lightgray",
):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        color="grey",
        alpha=0.45,
        linestyle="--",
        zorder=0
    )

    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-30, 61, 15)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    if show_bottom_labels:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    else:
        ax.set_xticklabels([])

    if show_left_labels:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=12)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

def add_base_map_(
    ax,
    extent=(-180, 180, -30, 60),
    show_left_labels=False,
    show_bottom_labels=False,
):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=1)

    # grid
    ax.grid(True, which="major", linewidth=0.45, color="gray",
            alpha=0.5, linestyle="--")

    # ticks
    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-30, 61, 15)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

    if not show_bottom_labels:
        ax.set_xticklabels([])
    else:
        ax.tick_params(axis="x", labelsize=11)

    if not show_left_labels:
        ax.set_yticklabels([])
    else:
        ax.tick_params(axis="y", labelsize=11)

    return ax

#=----------------------------------------------------------------------------
# ============================================================
# 4) BOTTOM CONTEXT MAP: PAL + BUOYS ONLY
# ============================================================

# ------------------------------------------------------------
# Optional: slightly cleaner context legend placement
# ------------------------------------------------------------
# ============================================================
# CONTEXT MAP
# ============================================================
def plot_pal_buoy_context_map(
    ax,
    pals_classed_by_region,
    buoy_files_by_region,
    PAL_region_colors,
    extent=(-180, 180, -30, 60),
    pal_track_stride=25,
    pal_linewidth=2.0,
    buoy_marker_size=35,
    add_legend=False,   # <- default OFF now
):
    """
    Bottom context panel showing:
      - PAL tracks by PAL region color
      - buoy locations by ocean marker type

    Legend is off by default to keep the figure clean.
    """

    add_base_map(
        ax,
        extent=extent,
        show_left_labels=True,
        show_bottom_labels=True
    )

    # -----------------------------
    # PAL tracks
    # -----------------------------
    for region, files in pals_classed_by_region.items():
        if region == "Unclassified" or len(files) == 0:
            continue

        color = PAL_region_colors.get(region, "tab:blue")

        for file in files:
            ds = xr.open_dataset(file)
            lat = ds["lat"].values[::pal_track_stride]
            lon = wrap_lon(ds["lon"].values[::pal_track_stride])
            ds.close()

            ax.plot(
                lon, lat,
                transform=ccrs.PlateCarree(),
                color=color,
                linewidth=pal_linewidth,
                zorder=4
            )

    # -----------------------------
    # Buoys
    # -----------------------------
    buoy_specs = [
        ("ENP", buoy_files_by_region.get("ENP", []), "*"),
        ("WNP", buoy_files_by_region.get("WNP", []), "P"),
        ("IND", buoy_files_by_region.get("IND", []), "s"),
        ("ATL", buoy_files_by_region.get("ATL", []), "d"),
    ]

    for _, buoy_files, marker in buoy_specs:
        for fl in buoy_files:
            xrfile = xr.open_dataset(fl)
            lat = xrfile["lat"].values[0]
            lon = wrap_lon(xrfile["lon"].values[0])
            xrfile.close()

            if lon == -180:
                lon = -179.8

            ax.scatter(
                lon, lat,
                color="k",
                s=buoy_marker_size,
                linewidths=1.2,
                marker=marker,
                transform=ccrs.PlateCarree(),
                zorder=5
            )

    return ax

#=----------------------------------------------------------------------------
# ============================================================
# ROW COLORBARS
# ============================================================
def add_row_colorbars_clean(
    fig,
    axes,
    metrics,
    row_mappables,
    metric_style,
    ax_context=None,
    *,
    cb_width_frac=0.40,
    cb_height=0.011,
    title_position="top",
    title_fontsize=12,
    tick_fontsize=10,
    tick_pad=1,
    row_cb_y_offsets=None,   # NEW: optional per-row vertical adjustment
):
    """
    One shared horizontal colorbar per row.
    Uses many discrete color steps, but only labels selected ticks.

    row_cb_y_offsets : dict, optional
        Dictionary mapping row index to vertical offset in figure coordinates.
        Negative values move the colorbar down; positive values move it up.
        Example: {5: -0.018}
    """

    fig.canvas.draw()
    nrows = len(metrics)

    if row_cb_y_offsets is None:
        row_cb_y_offsets = {}

    for i, metric in enumerate(metrics):
        mappable = row_mappables[i]
        if mappable is None:
            continue

        style = metric_style[metric]
        bounds = np.asarray(style["bounds"])
        extend = style.get("extend", "neither")
        cbar_label = style["label"]
        tick_labels = style.get("tick_labels", bounds)

        # row geometry
        row_boxes = [ax.get_position() for ax in axes[i, :]]
        row_left = min(bb.x0 for bb in row_boxes)
        row_right = max(bb.x1 for bb in row_boxes)
        row_bottom = min(bb.y0 for bb in row_boxes)
        row_width = row_right - row_left

        cb_width = row_width * cb_width_frac
        cb_left = row_left + 0.5 * (row_width - cb_width)

        # place centered in the gap between this row and the next element below
        if i < nrows - 1:
            next_row_top = max(ax.get_position().y1 for ax in axes[i + 1, :])
            gap_mid = 0.5 * (row_bottom + next_row_top)
            cb_bottom = gap_mid - 0.5 * cb_height
        else:
            if ax_context is not None:
                ctx_top = ax_context.get_position().y1
                gap_mid = 0.5 * (row_bottom + ctx_top)
                cb_bottom = gap_mid - 0.5 * cb_height
            else:
                cb_bottom = row_bottom - 0.03

        # NEW: apply row-specific vertical offset
        cb_bottom = cb_bottom + row_cb_y_offsets.get(i, 0.0)

        cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])

        norm = BoundaryNorm(bounds, mappable.cmap.N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=mappable.cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            ticks=tick_labels,
            extend=extend,
            boundaries=bounds,
            spacing="proportional",
        )

        cbar.ax.tick_params(labelsize=tick_fontsize, pad=tick_pad)

        # label placement
        if title_position == "right":
            cbar.set_label(cbar_label, fontsize=title_fontsize, fontweight="bold", labelpad=2)
            cbar.ax.xaxis.set_label_position("bottom")
        elif title_position == "bottom":
            cbar.set_label(cbar_label, fontsize=title_fontsize, fontweight="bold", labelpad=2)
            cbar.ax.xaxis.set_label_position("bottom")
        else:
            # centered above
            cbar.ax.set_title(cbar_label, fontsize=title_fontsize, fontweight="bold", pad=6)
# ============================================================
# 5) NEW MAIN PLOTTING FUNCTION
# ============================================================

# ------------------------------------------------------------
# NEW: dedicated GridSpec colorbar rows
# ------------------------------------------------------------
# ============================================================
# MAIN SPATIAL PANEL PLOT
# ============================================================
def plot_spatial_skill_panels_with_context(
    df,
    metrics,
    pals_classed_by_region,
    buoy_files_by_region,
    PAL_region_colors,
    reference_types=("PAL", "Buoy"),
    metric_style=None,
    figsize=(16, 15.2),
    extent=(-180, 180, -30, 60),
    marker_size=105,
    marker_edge_width=0.7,
    pal_context_track_stride=25,
    savepath=None,
):
    """
    Cleaner version:
      - shorter row colorbars
      - more vertical spacing
      - no geo-context legend
      - cleaner product legend
    """

    if metric_style is None:
        metric_style = make_metric_style_dict()

    nrows = len(metrics)
    ncols = len(reference_types)

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=[1] * nrows + [0.82],   # slightly taller context map
        hspace=0.72,                          # more row spacing
        wspace=0.02
    )

    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    ax_context = fig.add_subplot(gs[-1, :], projection=ccrs.PlateCarree())

    row_mappables = [None] * nrows

    # -----------------------------
    # metric map rows
    # -----------------------------
    for i, metric in enumerate(metrics):
        style = metric_style[metric]
        cmap = style["cmap"]
        bounds = np.asarray(style["bounds"])
        norm = BoundaryNorm(bounds, cmap.N, clip=False)

        for j, ref in enumerate(reference_types):
            ax = axes[i, j]

            add_base_map(
                ax,
                extent=extent,
                show_left_labels=(j == 0),
                show_bottom_labels=(i == nrows - 1)
            )

            dsub = df[
                (df["metric"] == metric) &
                (df["reference_type"] == ref)
            ].copy()

            for product, dprod in dsub.groupby("product"):
                marker = product_markers.get(product, "o")
                dx, dy = product_offsets.get(product, (0.0, 0.0))

                sc = ax.scatter(
                    dprod["lon"].values + dx,
                    dprod["lat"].values + dy,
                    c=dprod["value"].values,
                    cmap=cmap,
                    norm=norm,
                    s=marker_size,
                    marker=marker,
                    edgecolor="black",
                    linewidth=marker_edge_width,
                    transform=ccrs.PlateCarree(),
                    zorder=4
                )

                if row_mappables[i] is None:
                    row_mappables[i] = sc

            ax.set_title(
                f"{ref} — {style['label']}",
                fontsize=17,
                fontweight="bold",
                pad=10
            )

    # -----------------------------
    # bottom context map
    # -----------------------------
    plot_pal_buoy_context_map(
        ax=ax_context,
        pals_classed_by_region=pals_classed_by_region,
        buoy_files_by_region=buoy_files_by_region,
        PAL_region_colors=PAL_region_colors,
        extent=extent,
        pal_track_stride=pal_context_track_stride,
        add_legend=False
    )

    # -----------------------------
    # product legend only
    # -----------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label=product
        )
        for product, marker in product_markers.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=6,
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
        columnspacing=1.5
    )

    # finalize axes positions before adding row colorbars
    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.97,
        bottom=0.12
    )
    fig.canvas.draw()

    add_row_colorbars_clean(
            fig=fig,
            axes=axes,
            metrics=metrics,
            row_mappables=row_mappables,
            metric_style=metric_style,
            ax_context=ax_context,
            cb_width_frac=0.40,
            cb_height=0.011,
            title_position="top",
            title_fontsize=12,
            tick_fontsize=10,
    )

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes, ax_context

#----------------------------------------------------------------------------

def plot_selected_daily_skill_maps_pal_buoy_main(
    cat_df,
    quant_df,
    pals_classed_by_region,
    buoy_files_by_region,
    PAL_region_colors,
    reference_types=("PAL", "Buoy"),
    selected_rows=None,
    metric_style=None,
    figsize=(16, 14.5),
    extent=(-180, 180, -30, 60),
    marker_size=105,
    marker_edge_width=0.7,
    pal_context_track_stride=25,
    savepath=None,
):
    """
    Main-text selected daily skill maps for PAL and buoy references.

    Expected dataframe columns:
        reference_type, region, product, metric, value, lon, lat

    Default rows:
        1. HSS from categorical metrics
        2. Detection bias / frequency bias from categorical metrics
        3. CC from quantitative metrics
        4. Relative bias from quantitative metrics

    Columns:
        PAL, Buoy
    """

    if metric_style is None:
        metric_style = make_metric_style_dict()

    if selected_rows is None:
        selected_rows = [
            {
                "source": "cat",
                "metric": "HSS",
                "style_key": "HSS",
                "label": "HSS",
            },
            {
                "source": "cat",
                "metric": "FreqBias",
                "style_key": "FreqBias",
                "label": "Detection bias",
            },
            {
                "source": "quant",
                "metric": "CC",
                "style_key": "CC",
                "label": "CC",
            },
            {
                "source": "quant",
                "metric": "Bias",
                "style_key": "Bias",
                "label": "Relative bias [%]",
            },
        ]

    nrows = len(selected_rows)
    ncols = len(reference_types)

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=[1] * nrows + [0.82],
        hspace=0.72,
        wspace=0.02,
    )

    axes = np.empty((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    ax_context = fig.add_subplot(gs[-1, :], projection=ccrs.PlateCarree())

    row_mappables = [None] * nrows

    for i, row in enumerate(selected_rows):
        source = row["source"]
        metric = row["metric"]
        style_key = row["style_key"]
        label = row["label"]

        if source == "cat":
            df_source = cat_df
        elif source == "quant":
            df_source = quant_df
        else:
            raise ValueError("source must be either 'cat' or 'quant'.")

        if style_key not in metric_style:
            raise KeyError(
                f"'{style_key}' is not in metric_style. "
                f"Available keys are: {list(metric_style.keys())}"
            )

        style = metric_style[style_key]
        cmap = style["cmap"]
        bounds = np.asarray(style["bounds"])
        norm = BoundaryNorm(bounds, cmap.N, clip=False)

        for j, ref in enumerate(reference_types):
            ax = axes[i, j]

            add_base_map(
                ax,
                extent=extent,
                show_left_labels=(j == 0),
                show_bottom_labels=(i == nrows - 1),
            )

            dsub = df_source[
                (df_source["metric"] == metric) &
                (df_source["reference_type"] == ref)
            ].copy()

            if dsub.empty:
                print(f"Warning: no data found for reference_type={ref}, metric={metric}")

            for product, dprod in dsub.groupby("product"):
                marker = product_markers.get(product, "o")
                dx, dy = product_offsets.get(product, (0.0, 0.0))

                sc = ax.scatter(
                    dprod["lon"].values + dx,
                    dprod["lat"].values + dy,
                    c=dprod["value"].values,
                    cmap=cmap,
                    norm=norm,
                    s=marker_size,
                    marker=marker,
                    edgecolor="black",
                    linewidth=marker_edge_width,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )

                if row_mappables[i] is None:
                    row_mappables[i] = sc

            ax.set_title(
                f"{ref} — {label}",
                fontsize=17,
                fontweight="bold",
                pad=10,
            )

    # Bottom context map
    plot_pal_buoy_context_map(
        ax=ax_context,
        pals_classed_by_region=pals_classed_by_region,
        buoy_files_by_region=buoy_files_by_region,
        PAL_region_colors=PAL_region_colors,
        extent=extent,
        pal_track_stride=pal_context_track_stride,
        add_legend=False,
    )

    # Product legend
    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label=product,
        )
        for product, marker in product_markers.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=6,
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.97,
        bottom=0.12,
    )

    fig.canvas.draw()

    add_row_colorbars_clean(
        fig=fig,
        axes=axes,
        metrics=[row["style_key"] for row in selected_rows],
        row_mappables=row_mappables,
        metric_style=metric_style,
        ax_context=ax_context,
        cb_width_frac=0.40,
        cb_height=0.011,
        title_position="top",
        title_fontsize=12,
        tick_fontsize=10,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes, ax_context
#----------------------------------------------------------------------------

def plot_main_daily_skill_maps_pal_buoy_six_metrics(
    cat_df,
    quant_df,
    reference_types=("PAL", "Buoy"),
    selected_rows=None,
    metric_style=None,
    figsize=(18, 20),
    extent=(-180, 180, -30, 60),
    marker_size=135,
    marker_edge_width=0.8,
    savepath=None,
):
    """
    Main-text daily skill map figure for PAL and buoy references.

    Expected dataframe columns:
        reference_type, region, product, metric, value, lon, lat

    Default rows:
        1. POD from categorical metrics
        2. HSS from categorical metrics
        3. Detection bias / frequency bias from categorical metrics
        4. CC from quantitative metrics
        5. RMSE from quantitative metrics
        6. Relative bias from quantitative metrics

    Columns:
        PAL, Buoy

    Notes:
        - No bottom context map is included.
        - Panel titles are placed in the upper-left using letters.
        - Product symbols are retained and enlarged for readability.
    """

    if metric_style is None:
        metric_style = make_metric_style_dict()

    if selected_rows is None:
        selected_rows = [
            {
                "source": "cat",
                "metric": "POD",
                "style_key": "POD",
                "label": "POD",
            },
            
            {
                "source": "cat",
                "metric": "FreqBias",
                "style_key": "FreqBias",
                "label": "Frequency bias",
            },
            {
                "source": "cat",
                "metric": "HSS",
                "style_key": "HSS",
                "label": "HSS",
            },
            {
                "source": "quant",
                "metric": "CC",
                "style_key": "CC",
                "label": "CC",
            },
            {
                "source": "quant",
                "metric": "RMSE",
                "style_key": "RMSE",
                "label": "RMSE [mm day$^{-1}$]",
            },
            {
                "source": "quant",
                "metric": "Bias",
                "style_key": "Bias",
                "label": "Relative bias [%]",
            },
        ]

    nrows = len(selected_rows)
    ncols = len(reference_types)

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        hspace=0.82,
        wspace=0.04,
    )

    axes = np.empty((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    row_mappables = [None] * nrows

    panel_letters = "abcdefghijklmnopqrstuvwxyz"

    for i, row in enumerate(selected_rows):
        source = row["source"]
        metric = row["metric"]
        style_key = row["style_key"]
        row_label = row["label"]

        if source == "cat":
            df_source = cat_df
        elif source == "quant":
            df_source = quant_df
        else:
            raise ValueError("source must be either 'cat' or 'quant'.")

        if style_key not in metric_style:
            raise KeyError(
                f"'{style_key}' is not in metric_style. "
                f"Available keys are: {list(metric_style.keys())}"
            )

        style = metric_style[style_key]
        cmap = style["cmap"]
        bounds = np.asarray(style["bounds"])
        norm = BoundaryNorm(bounds, cmap.N, clip=False)

        for j, ref in enumerate(reference_types):
            ax = axes[i, j]

            add_base_map(
                ax,
                extent=extent,
                show_left_labels=(j == 0),
                show_bottom_labels=(i == nrows - 1),
            )

            dsub = df_source[
                (df_source["metric"] == metric) &
                (df_source["reference_type"] == ref)
            ].copy()

            if dsub.empty:
                print(f"Warning: no data found for reference_type={ref}, metric={metric}")

            for product, dprod in dsub.groupby("product"):
                marker = product_markers.get(product, "o")
                dx, dy = product_offsets.get(product, (0.0, 0.0))

                sc = ax.scatter(
                    dprod["lon"].values + dx,
                    dprod["lat"].values + dy,
                    c=dprod["value"].values,
                    cmap=cmap,
                    norm=norm,
                    s=marker_size,
                    marker=marker,
                    edgecolor="black",
                    linewidth=marker_edge_width,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )

                if row_mappables[i] is None:
                    row_mappables[i] = sc

            # Panel label and title in upper-left, not centered.
            panel_id = panel_letters[i * ncols + j]
            ax.text(
                0.01,
                1.06,
                f"({panel_id}) {ref} — {row_label}",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

    # Product legend only.
    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=11,
            label=product,
        )
        for product, marker in product_markers.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),   # lower legend
        ncol=6,
        frameon=False,
        fontsize=18,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.125,   # more bottom space
    )

    fig.canvas.draw()

    add_row_colorbars_clean(
    fig=fig,
    axes=axes,
    metrics=[row["style_key"] for row in selected_rows],
    row_mappables=row_mappables,
    metric_style=metric_style,
    ax_context=None,
    cb_width_frac=0.40,
    cb_height=0.010,
    title_position="top",
    title_fontsize=16,
    tick_fontsize=10,
    row_cb_y_offsets={
        5: -0.020,   # moves only Relative bias colorbar lower
    },
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig, axes
#----------------------------------------------------------------------------
def plot_spatial_metric_panels(
    df,
    metrics,
    reference_types=("PAL", "Buoy"),
    figsize=(16, 12),
    extent=(-180, 180, -30, 60),
    marker_size=120,
    marker_edge_width=0.7,
    savepath=None,
):
    nrows = len(metrics)
    ncols = len(reference_types)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    row_mappables = []
    row_cbar_info = []

    for i, metric in enumerate(metrics):
        cmap, norm, bounds, cbar_label = make_discrete_norm_and_cmap_(metric)
        row_mappables.append(None)
        row_cbar_info.append((bounds, cbar_label))

        for j, ref in enumerate(reference_types):
            ax = axes[i, j]

            show_left = (j == 0)
            show_bottom = (i == nrows - 1)

            add_base_map_(
                ax,
                extent=extent,
                show_left_labels=show_left,
                show_bottom_labels=show_bottom
            )

            dsub = df[(df["metric"] == metric) & (df["reference_type"] == ref)].copy()

            for product, dprod in dsub.groupby("product"):
                marker = product_markers.get(product, "o")
                dx, dy = product_offsets.get(product, (0.0, 0.0))

                x = dprod["lon"].values + dx
                y = dprod["lat"].values + dy
                c = dprod["value"].values

                sc = ax.scatter(
                    x, y,
                    c=c,
                    cmap=cmap,
                    norm=norm,
                    s=marker_size,
                    marker=marker,
                    edgecolor="black",
                    linewidth=marker_edge_width,
                    transform=ccrs.PlateCarree(),
                    zorder=4
                )

                if row_mappables[i] is None:
                    row_mappables[i] = sc

            ax.set_title(
                f"{ref} — {METRIC_STYLE[metric]['label']}",
                fontsize=16,
                fontweight="bold",
                pad=8
            )

    # tighter panel spacing first
    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.97,
        bottom=0.12,
        wspace=0.02,
        hspace=0.20
    )

    # -------- Manual row-wise colorbars --------
    # longer and centered, with explicit vertical positions
    fig.canvas.draw()

    for i in range(nrows):
        bounds, cbar_label = row_cbar_info[i]
        mappable = row_mappables[i]

        # union of the two axes positions for this row
        pos_l = axes[i, 0].get_position()
        pos_r = axes[i, -1].get_position()

        row_left = pos_l.x0
        row_right = pos_r.x1
        row_bottom = min(pos_l.y0, pos_r.y0)

        row_width = row_right - row_left

        # make cbar longer and centered
        cbar_width = row_width * 0.42
        cbar_height = 0.012

        # centered under the row
        cbar_left = row_left + 0.5 * (row_width - cbar_width)

        # vertical placement:
        # for bottom row keep it a bit closer to panels so it doesn't fight legend
        if i == nrows - 1:
            cbar_bottom = row_bottom - 0.040
        else:
            cbar_bottom = row_bottom - 0.055

        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

        cbar = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal",
            ticks=bounds
        )
        cbar.ax.tick_params(labelsize=12, pad=2)
        cbar.set_label(cbar_label, fontsize=13, fontweight="bold", labelpad=2)

    # -------- Larger product legend --------
    legend_handles = []
    for product, marker in product_markers.items():
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=11,
                label=product
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=6,
        frameon=False,
        fontsize=15,
        handletextpad=0.5,
        columnspacing=1.4
    )

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_metric_bars_4x2_by_reference(
    df,
    products,
    product_colors,
    *,
    metrics=("POD", "FAR", "Bias", "HSS"),
    reference_types=("PAL", "Buoy"),
    pal_region_order=("ETNP", "TNEP", "TNWP", "TSEP", "TNIO", "STNA"),
    buoy_region_order=("ENP", "WNP", "IND", "ATL"),
    pal_region_labels=None,
    buoy_region_labels=None,
    figsize=(24, 18),
    bar_width=0.11,
    group_gap=0.24,
    ylabel_fontsize=19,
    title_fontsize=20,
    tick_fontsize=15,
    legend_fontsize=16,
    bottom_tick_fontsize=18,
    max_yticks=5,
    savepath=None,
):
    """
    rows = metrics
    col 0 = PAL
    col 1 = Buoy

    Required df columns:
      reference_type, region, product, metric, value
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    if pal_region_labels is None:
        pal_region_labels = PAL_REGION_NAMES

    if buoy_region_labels is None:
        buoy_region_labels = Buoy_REGION_NAMES

    # share x by column, y by row
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(reference_types),
        figsize=figsize,
        sharex='col',
        sharey='row',
        squeeze=False
    )

    # nicer metric labels
    metric_ylabel_map = {
        "POD": "POD",
        "FAR": "FAR",
        "HSS": "HSS",
        "Bias_det": "Bias",
        "CC": "CC",
        "RMSE": "RMSE [mm/day]",
        "MAE": "MAE [mm/day]",
        "NRMSE": "NRMSE [%]",
        "Bias_q": "Bias [%]",   # optional alias if you rename quantitative bias
    }

    # fixed y-limits where useful
    metric_ylims = {
        "POD": (0, 1.0),
        "FAR": (0, 1.0),
        "HSS": (0, 0.5),       # slightly larger so buoy bars do not clip
        "CC": (0, 0.6),
        # Bias / RMSE / MAE left to auto unless you want to force them
    }

    # precompute region x positions for each column
    region_setup = {}
    for ref in reference_types:
        if ref == "PAL":
            region_order = list(pal_region_order)
            region_labels = [pal_region_labels.get(r, r) for r in region_order]
        else:
            region_order = list(buoy_region_order)
            region_labels = [buoy_region_labels.get(r, r) for r in region_order]

        n_regions = len(region_order)
        n_products = len(products)
        x = np.arange(n_regions) * (n_products * bar_width + group_gap)
        group_center = x + (n_products - 1) * bar_width / 2

        region_setup[ref] = {
            "order": region_order,
            "labels": region_labels,
            "x": x,
            "center": group_center,
        }

    # draw panels
    for i, metric in enumerate(metrics):
        for j, ref in enumerate(reference_types):
            ax = axes[i, j]

            region_order = region_setup[ref]["order"]
            region_labels = region_setup[ref]["labels"]
            x = region_setup[ref]["x"]
            group_center = region_setup[ref]["center"]

            dsub = df[
                (df["reference_type"] == ref) &
                (df["metric"] == metric)
            ].copy()

            for k, product in enumerate(products):
                vals = []
                for region in region_order:
                    dd = dsub[(dsub["region"] == region) & (dsub["product"] == product)]
                    vals.append(dd["value"].iloc[0] if len(dd) > 0 else np.nan)

                ax.bar(
                    x + k * bar_width,
                    vals,
                    width=bar_width,
                    color=product_colors.get(product, "0.7"),
                    edgecolor="black",
                    linewidth=0.55,
                    label=product if (i == 0 and j == 0) else None,
                )

            # titles only on top row
            if i == 0:
                ax.set_title(ref, fontsize=title_fontsize, fontweight="bold", pad=10)

            # y-label only on left column
            if j == 0:
                ylabel = metric_ylabel_map.get(metric, metric)
                # quantitative bias special handling if metric is plain "Bias"
                if metric == "Bias" and any(m in metrics for m in ["CC", "RMSE", "MAE"]):
                    ylabel = "Bias [%]"
                ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight="bold")

            # x ticks only on bottom row
            ax.set_xticks(group_center)
            if i == len(metrics) - 1:
                ax.set_xticklabels(
                    region_labels,
                    fontsize=bottom_tick_fontsize,
                    fontweight="bold",
                    rotation=24,
                    ha="right"
                )
            else:
                ax.tick_params(axis="x", labelbottom=False)

            # grid
            ax.grid(axis="y", linestyle="--", alpha=0.45)

            # metric-specific y-limits
            if metric in metric_ylims:
                ax.set_ylim(*metric_ylims[metric])

            # reduce y tick density
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))
            ax.tick_params(axis="y", labelsize=tick_fontsize)

            for t in ax.get_yticklabels():
                t.set_fontweight("bold")

    # bottom legend
    legend_handles = [
        Patch(facecolor=product_colors.get(p, "0.7"), edgecolor="black", label=p)
        for p in products
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.02),
        ncol=len(products),
        frameon=False,
        fontsize=legend_fontsize
    )

    plt.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.94,
        bottom=0.12,
        wspace=0.12,
        hspace=0.10
    )

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_intensity_metrics_cat_4x2(
    pal_cat,
    buoy_cat,
    rainfall_bins,
    products,
    product_colors,
    figsize=(16, 16),
    linewidth=2.2,
    markersize=6,
    tick_fontsize=20,
    label_fontsize=22,
    title_fontsize=22,
    legend_fontsize=22,
    savepath=None,
):
    cat_metrics = ['POD', 'FAR', 'Bias', 'HSS']

    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=figsize,
        sharex='col',
        sharey='row'
    )

    col_titles = ["PAL", "Buoy"]

    ylabels = {
        "POD": "POD",
        "FAR": "FAR",
        "Bias": "Bias",
        "HSS": "HSS",
    }

    ylims = {
        "POD": (0.0, 1.0),
        "FAR": (0.35, 1.0),
        "Bias": (0.2, 1.8),
        "HSS": (0, 0.50),
    }

    sources = [pal_cat, buoy_cat]

    for j, source in enumerate(sources):
        for i, met in enumerate(cat_metrics):
            ax = axes[i, j]

            for product in products:
                dmet = source[product]
                ax.plot(
                    rainfall_bins,
                    dmet.loc[rainfall_bins, met].astype(float),
                    marker='o',
                    linewidth=linewidth,
                    markersize=markersize,
                    color=product_colors[product],
                    label=product if (i == 0 and j == 0) else None
                )

            if i == 0:
                ax.set_title(col_titles[j], fontsize=title_fontsize, fontweight='bold')

            if j == 0:
                ax.set_ylabel(ylabels[met], fontsize=label_fontsize, fontweight='bold')

            if ylims[met] is not None:
                ax.set_ylim(*ylims[met])
            
            if met == "POD":
                ax.set_yticks([0.0, 0.5, 1.0])
            elif met == "FAR":
                ax.set_yticks([0.4, 0.7, 1.0])
            elif met == "HSS":
                ax.set_yticks([0.0, 0.2, 0.4])

            elif met == "Bias":
                ax.set_yticks([0.4, 0.8, 1.2])

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xscale('log')
            ax.set_xticks(rainfall_bins)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(labelsize=tick_fontsize)

            if i < 3:
                ax.tick_params(axis='x', labelbottom=False)

    for ax in axes[-1, :]:
        ax.set_xlabel('[mm/day]', fontsize=label_fontsize, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=4,
        fontsize=legend_fontsize,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')

    return fig, axes
#---------------------------------------------------------------------------------------------


def plot_intensity_metrics_qt_4x2(
    pal_qt,
    buoy_qt,
    rainfall_bins,
    products,
    product_colors,
    figsize=(16, 16),
    linewidth=2.2,
    markersize=6,
    tick_fontsize=22,
    label_fontsize=22,
    title_fontsize=22,
    legend_fontsize=22,
    savepath=None,
):
    qt_metrics = ['CC', 'RMSE', 'MAE', 'RB']

    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=figsize,
        sharex='col',
        sharey='row'
    )

    col_titles = ["PAL", "Buoy"]

    ylabels = {
        "CC": "CC",
        "RMSE": "RMSE [mm/day]",
        "MAE": "MAE [mm/day]",
        "RB": "Bias [%]",
    }

    ylims = {
        "CC": (-0.04, 0.45),
        "RMSE": [15,50],
        "MAE": [9,35],
        "RB": [-40,30],
    }

    sources = [pal_qt, buoy_qt]

    for j, source in enumerate(sources):
        for i, met in enumerate(qt_metrics):
            ax = axes[i, j]

            for product in products:
                dmet = source[product]
                ax.plot(
                    rainfall_bins,
                    dmet.loc[rainfall_bins, met].astype(float),
                    marker='o',
                    linewidth=linewidth,
                    markersize=markersize,
                    color=product_colors[product],
                    label=product if (i == 0 and j == 0) else None
                )

            if i == 0:
                ax.set_title(col_titles[j], fontsize=title_fontsize, fontweight='bold')

            if j == 0:
                ax.set_ylabel(ylabels[met], fontsize=label_fontsize, fontweight='bold')

            if ylims[met] is not None:
                ax.set_ylim(*ylims[met])

            if met == "CC":
                ax.set_yticks([0.0, 0.15, 0.35])
            elif met == "RMSE":
                ax.set_yticks([20, 30, 40, 50])
            elif met == "MAE":
                ax.set_yticks([10, 20, 30])
            elif met == "RB":
                ax.set_yticks([-30, -15, 0, 15, 30])

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xscale('log')
            ax.set_xticks(rainfall_bins)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(labelsize=tick_fontsize)

            if i < 3:
                ax.tick_params(axis='x', labelbottom=False)

    for ax in axes[-1, :]:
        ax.set_xlabel('[mm/day]', fontsize=label_fontsize, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=3,
        fontsize=legend_fontsize,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')

    return fig, axes

#------------------------------------------------------------------------------------------------------------------------------------------


def plot_intensity_metrics_selected_pal_buoy_4x2(
    pal_cat,
    buoy_cat,
    pal_qt,
    buoy_qt,
    rainfall_bins,
    products,
    product_colors,
    figsize=(17, 16),
    linewidth=3.2,
    markersize=7.5,
    tick_fontsize=22,
    label_fontsize=25,
    title_fontsize=24,
    panel_fontsize=20,
    legend_fontsize=21,
    savepath=None,
):
    """
    Figure 6-style intensity-dependent daily skill plot.

    Layout:
        Row 1: HSS
        Row 2: Frequency bias
        Row 3: CC
        Row 4: Relative bias

    Columns:
        Left: PAL
        Right: Buoy

    Inputs
    ------
    pal_cat, buoy_cat : dict
        Product-keyed dictionaries of categorical metric DataFrames.
        Each product DataFrame should be indexed by rainfall threshold and include:
        POD, FAR, Bias, HSS.
        Here categorical 'Bias' is plotted as frequency bias.

    pal_qt, buoy_qt : dict
        Product-keyed dictionaries of quantitative metric DataFrames.
        Each product DataFrame should be indexed by rainfall threshold and include:
        CC, RMSE, MAE, RB.

    rainfall_bins : list-like
        Rain-rate thresholds used as x-axis values.

    products : list-like
        Product names in plotting order.

    product_colors : dict
        Mapping from product name to color.

    Returns
    -------
    fig, axes
    """

    rows = [
        {
            "source": "cat",
            "metric": "HSS",
            "ylabel": "HSS",
            "ylim": (0.0, 0.46),
            "yticks": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "source": "cat",
            "metric": "Bias",
            "ylabel": "Frequency bias",
            "ylim": (0.2, 1.75),
            "yticks": [0.2, 0.6, 1.0, 1.4],
        },
        {
            "source": "qt",
            "metric": "CC",
            "ylabel": "CC",
            "ylim": (-0.15, 0.45),
            "yticks": [0, 0.2, 0.4],
        },
        {
            "source": "qt",
            "metric": "RB",
            "ylabel": "Relative bias [%]",
            "ylim": (-45, 35),
            "yticks": [-40, -20, 0, 20],
        },
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=figsize,
        sharex="col",
        sharey=False
    )

    col_titles = ["PAL", "Buoy"]
    sources = {
        "PAL": {
            "cat": pal_cat,
            "qt": pal_qt,
        },
        "Buoy": {
            "cat": buoy_cat,
            "qt": buoy_qt,
        },
    }

    panel_letters = list("abcdefghijklmnopqrstuvwxyz")

    for j, ref in enumerate(col_titles):
        for i, row in enumerate(rows):
            ax = axes[i, j]
            source = sources[ref][row["source"]]
            metric = row["metric"]

            for product in products:
                dmet = source[product]

                ax.plot(
                    rainfall_bins,
                    dmet.loc[rainfall_bins, metric].astype(float),
                    marker="o",
                    linewidth=linewidth,
                    markersize=markersize,
                    color=product_colors[product],
                    label=product if (i == 0 and j == 0) else None,
                )

            # Column title only on first row.
            if i == 0:
                ax.set_title(
                    ref,
                    fontsize=title_fontsize,
                    fontweight="bold",
                    pad=12
                )

            # Panel labels inside upper-left.
            panel_id = panel_letters[i * 2 + j]
            ax.text(
                0.025,
                0.94,
                f"({panel_id})",
                transform=ax.transAxes,
                fontsize=panel_fontsize,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=2.0),
            )

            # Y label on both columns because y-axis is not shared.
            ax.set_ylabel(
                row["ylabel"],
                fontsize=label_fontsize,
                fontweight="bold"
            )

            ax.set_ylim(*row["ylim"])
            ax.set_yticks(row["yticks"])

            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xscale("log")
            ax.set_xticks(rainfall_bins)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=tick_fontsize,
                width=1.6,
                length=7
            )

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

            if i < len(rows) - 1:
                ax.tick_params(axis="x", labelbottom=False)

    for ax in axes[-1, :]:
        ax.set_xlabel(
            "[mm day$^{-1}$]",
            fontsize=label_fontsize,
            fontweight="bold"
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=6,
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    plt.tight_layout(rect=[0, 0.075, 1, 1])

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig, axes
#------------------------------------------------------------------------------------------------------------------------------------------

def plot_pdf_bundle_on_axis(
    ax,
    pdf_dict,
    *,
    insitu_label="PAL",
    products=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    product_colors=None,
    pdf_kind="pdfv",   # "pdfv" or "pdfc"
    lw=4,
    insitu_color="b",
    insitu_ls=":",
):
    """
    Plot one PDF bundle on one axis.
    """
    x = pdf_dict["insitu"]["bin"].values

    # in situ line
    ax.plot(
        x,
        pdf_dict["insitu"][pdf_kind],
        lw=lw,
        color=insitu_color,
        ls=insitu_ls,
        label=insitu_label
    )

    # product lines
    for product in products:
        ax.plot(
            x,
            pdf_dict[product][pdf_kind],
            lw=lw,
            color=product_colors[product],
            label=product
        )


def plot_pdf_comparison_pal_buoy(
    pal_df,
    buoy_df,
    *,
    obs_col="rain_rate",
    products=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    product_colors=None,
    bin_values=(0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    pdf_kind="pdfv",          # "pdfv" or "pdfc"
    pal_year_range=None,
    buoy_year_range=(2000, 2020),
    figsize=(16, 6),
    dpi=300,
    lw=5,                    # slightly thicker default
    savepath=None,
):
    """
    Build PAL and Buoy PDFs and plot them side by side.

    Returns
    -------
    fig, axes, pal_pdf_dict, buoy_pdf_dict
    """

    # -----------------------------
    # compute PDFs
    # -----------------------------
    pal_pdf_dict = compute_pdf_bundle_for_insitu_df(
        pal_df,
        obs_col=obs_col,
        products=products,
        bin_values=bin_values,
        year_range=pal_year_range,
    )

    buoy_pdf_dict = compute_pdf_bundle_for_insitu_df(
        buoy_df,
        obs_col=obs_col,
        products=products,
        bin_values=bin_values,
        year_range=buoy_year_range,
    )

    # -----------------------------
    # plot
    # -----------------------------
    # Do not share y-axis, so PAL and buoy panels can scale independently.
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=False)

    # left: PAL
    plot_pdf_bundle_on_axis(
        axes[0],
        pal_pdf_dict,
        insitu_label="PAL",
        products=products,
        product_colors=product_colors,
        pdf_kind=pdf_kind,
        lw=lw,
        insitu_color="b",
        insitu_ls=":"
    )

    # right: Buoy
    plot_pdf_bundle_on_axis(
        axes[1],
        buoy_pdf_dict,
        insitu_label="Buoy",
        products=products,
        product_colors=product_colors,
        pdf_kind=pdf_kind,
        lw=lw,
        insitu_color="b",
        insitu_ls="-"
    )

    # -----------------------------
    # formatting
    # -----------------------------
    panel_labels = ["(a) PAL", "(b) Buoy"]

    for ax, panel_label in zip(axes, panel_labels):
        # Remove centered title; use panel label inside upper-left instead.
        ax.set_title("")

        ax.text(
            0.03,
            0.96,
            panel_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
        )

        ax.set_xscale("log")
        ax.set_xlabel("[mm day$^{-1}$]", fontsize=20, fontweight="bold")
        ax.grid(True, which="major", linestyle="--", alpha=0.6)

        ax.xaxis.set_major_locator(FixedLocator(list(bin_values)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: "0.5" if abs(v - 0.5) < 1e-12 else f"{int(round(v))}")
        )
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis="x", which="minor", bottom=False)

        # Larger tick labels
        ax.tick_params(axis="both", which="major", labelsize=20, width=1.7, length=8)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    # Since y-axis is not shared, give both panels y-axis labels.
    axes[0].set_ylabel("PDF (%)", fontsize=20, fontweight="bold")
    axes[1].set_ylabel("PDF (%)", fontsize=20, fontweight="bold")

    # -----------------------------
    # common legend
    # -----------------------------
    legend_handles = [
        Line2D([0], [0], color="b", lw=lw, ls=":", label="PAL"),
        Line2D([0], [0], color="b", lw=lw, ls="-", label="Buoy"),
    ]

    legend_handles.extend([
        Line2D([0], [0], color=product_colors[p], lw=lw, ls="-", label=p)
        for p in products
    ])

    # More bottom space is needed because the legend has two rows.
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        fontsize=20,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # Reserve enough bottom margin for the legend.
    plt.tight_layout(rect=[0, 0.17, 1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, axes, pal_pdf_dict, buoy_pdf_dict

# ============================================================
# PLOT 2x2 INTERANNUAL VARIABILITY FROM MONTHLY-BASED ANNUAL DF
# ============================================================

def plot_interannual_variability_2x2_from_monthly_df(
    annual_df,
    regions,
    products,
    product_colors,
    *,
    ref="Buoy",
    region_labels=None,
    figsize=(16.5, 9.5), # (12, 9)
    lw_ref=3.5,
    lw_prod=3.0,
    ncol_legend=3,
    year_min=None,
    year_max=None,
    region_year_limits=None
):
    """
    2x2 annual interannual variability plot using annual_df produced from
    build_annual_from_monthly_buoy_df().
    """

    if region_labels is None:
        region_labels = Buoy_REGION_NAMES

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(regions):
            ax.axis("off")
            continue

        region = regions[i]
        dfr = annual_df[annual_df["region"] == region].copy().sort_values("year")
        if region_year_limits is not None and region in region_year_limits:
            yr0, yr1 = region_year_limits[region]
            dfr = dfr[(dfr["year"] >= yr0) & (dfr["year"] <= yr1)].copy()

        if year_min is not None:
            dfr = dfr[dfr["year"] >= year_min].copy()
        if year_max is not None:
            dfr = dfr[dfr["year"] <= year_max].copy()

        if dfr.empty:
            ax.axis("off")
            continue

        # reference
        ax.plot(
            dfr["year"],
            dfr[ref],
            lw=lw_ref,
            color=product_colors.get(ref, "b"),
            label=ref
        )

        # products
        for prod in products:
            if prod == ref:
                continue
            ax.plot(
                dfr["year"],
                dfr[prod],
                lw=lw_prod,
                color=product_colors[prod],
                label=prod
            )

        ax.set_title(region_labels.get(region, region), fontsize=16, fontweight="bold")
        # ax.set_xlabel("Year", fontsize=13, fontweight="bold")
        ax.set_ylabel("[mm day$^{-1}$]", fontsize=15, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=15)

        # integer year ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=ncol_legend,
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    return fig

#------------------------------------------------------------------
def plot_interannual_variability_with_sample_counts_2x2(

    annual_df,
    regions,
    products,
    product_colors,
    *,
    ref=None,
    region_labels=None,
    figsize=(16.5, 9.5),
    lw_ref=3.5,
    lw_prod=3.5,
    lw_count=2.5,
    count_col="n_valid_months",
    count_color="0.35",
    count_ls="--",
    count_marker="o",
    count_label="Valid months",
    show_count_on_all_panels=True,
    right_ylabel="Valid months",
    ncol_legend=3,
    year_min=None,
    year_max=None,
    region_year_limits=None,
    count_ylim=None,   # <-- changed default

    rainfall_ylabel="[mm day$^{-1}$]",

):

    """
    2x2 annual interannual variability plot with sample-count line on right y-axis.
    """

    if region_labels is None:
        region_labels = Buoy_REGION_NAMES

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)

    axes = axes.flatten()
    all_handles = []
    all_labels = []

    # auto right-axis range from selected count column

    if count_ylim is None and count_col in annual_df.columns:
        cmax = annual_df[count_col].max()
        if np.isfinite(cmax):
            count_ylim = (0, cmax * 1.10)
        else:
            count_ylim = None

    for i, ax in enumerate(axes):

        if i >= len(regions):

            ax.axis("off")

            continue

        region = regions[i]
        dfr = annual_df[annual_df["region"] == region].copy().sort_values("year")

        if region_year_limits is not None and region in region_year_limits:
            yr0, yr1 = region_year_limits[region]
            dfr = dfr[(dfr["year"] >= yr0) & (dfr["year"] <= yr1)].copy()

        if year_min is not None:
            dfr = dfr[dfr["year"] >= year_min].copy()

        if year_max is not None:
            dfr = dfr[dfr["year"] <= year_max].copy()

        if dfr.empty:
            ax.axis("off")

            continue

        # --- left axis: rainfall series ---

        if ref is not None and ref in dfr.columns:

            h = ax.plot(
                dfr["year"],
                dfr[ref],
                lw=lw_ref,
                color=product_colors.get(ref, "k"),
                label=ref,
                zorder=3,

            )[0]

            all_handles.append(h)
            all_labels.append(ref)

        for prod in products:

            if prod not in dfr.columns:
                continue

            if ref is not None and prod == ref:
                continue

            h = ax.plot(
                dfr["year"],
                dfr[prod],
                lw=lw_prod,
                color=product_colors[prod],
                label=prod,
                zorder=2,

            )[0]

            all_handles.append(h)
            all_labels.append(prod)

        ax.set_title(region_labels.get(region, region), fontsize=16, fontweight="bold")

        ax.set_ylabel(rainfall_ylabel, fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # --- right axis: count line ---
        axr = ax.twinx()
        if count_col in dfr.columns:

            hc = axr.plot(
                dfr["year"],
                dfr[count_col],
                color=count_color,
                lw=lw_count,
                ls=count_ls,
                marker=count_marker,
                ms=4.5,
                label=count_label if i == 0 else None,
                zorder=1,

            )[0]

            if i == 0:
                all_handles.append(hc)
                all_labels.append(count_label)

        if count_ylim is not None:
            axr.set_ylim(*count_ylim)

        # sensible automatic ticks for right axis

        axr.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axr.tick_params(axis="y", labelsize=16, colors=count_color)
        if show_count_on_all_panels:
            axr.set_ylabel(right_ylabel, fontsize=16, fontweight="bold", color=count_color)

        else:
            if i % 2 == 1:
                axr.set_ylabel(right_ylabel, fontsize=16, fontweight="bold", color=count_color)

            else:
                axr.set_ylabel("")

    # deduplicate legend

    uniq = {}

    for lab, h in zip(all_labels, all_handles):

        if lab not in uniq:
            uniq[lab] = h

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=ncol_legend,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.tight_layout(rect=[0, 0.13, 1, 1])

    return fig

def plot_deseasonalized_anomaly_scatter_from_monthly_buoy_df(
    monthly_buoy_df,
    *,
    products,
    regions,
    ref_col="Buoy",
    product_cols=("GPCP v2.3", "GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"),
    region_labels=None,
    product_colors=None,
    region_col="region",
    month_col="month",
    id_col="ID",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=2,
    equal_weight_by_buoy=True,
    figsize=(20, 12),
    marker_size=18,
    point_alpha=0.75,
    savepath=None,
):
    """
    Build monthly regional series from monthly buoy-product table,
    deseasonalize, then make anomaly scatter.
    """

    # -----------------------------------------
    # 1) build monthly regional series
    # -----------------------------------------
    monthly_region = build_monthly_region_series_from_monthly_buoy_df(
        monthly_buoy_df,
        products=products,
        region_col=region_col,
        id_col=id_col,
        month_col=month_col,
        n_days_col=n_days_col,
        min_days_per_month=min_days_per_month,
        min_buoys_per_month=min_buoys_per_month,
        equal_weight_by_buoy=equal_weight_by_buoy,
    )

    # -----------------------------------------
    # 2) deseasonalize by region
    # -----------------------------------------
    monthly_region_anom = deseasonalize_monthly(
        df_monthly=monthly_region,
        products=products,
        time_col="month_start",
        region_col=region_col
    )

    # -----------------------------------------
    # 3) scatter plot using your existing function
    # -----------------------------------------
    fig = plot_deseasonalized_anomaly_scatter(
        df_anom=monthly_region_anom,
        regions=regions,
        ref_col=ref_col,
        product_cols=product_cols,
        region_labels=region_labels,
        product_colors=product_colors,
        region_col=region_col,
        figsize=figsize,
        savepath=savepath,
    )

    return fig, monthly_region, monthly_region_anom

#------------------------------------------------------------------------------
# Roebber / performance diagram for categorical precipitation metrics
#------------------------------------------------------------------------------


def _get_product_color(product, product_colors, default="0.5"):
    """
    Robust color lookup for product names, including common aliases.
    This avoids losing color when the column name is 'MERRA2' but the
    manuscript/product color dictionary uses 'MERRA-2'.
    """
    aliases = {
        "MERRA2": "MERRA-2",
        "MERRA-2": "MERRA2",
        "IMERG": "IMERG v07",
        "IMERG v07": "IMERG",
    }

    if product in product_colors:
        return product_colors[product]

    alt = aliases.get(product)
    if alt is not None and alt in product_colors:
        return product_colors[alt]

    return default

def draw_perf_background(
    ax,
    *,
    contour_label_fontsize=10,
    bias_label_fontsize=10,
    axis_label_fontsize=14,
    tick_fontsize=12,
    csi_color="brown",
    bias_color="steelblue",
    show_csi_right_axis=True,
    show_bias_label=True,
):
    """
    Draw CSI isolines and frequency-bias lines in POD vs SR space.

    x-axis = Success Ratio, SR = 1 - FAR
    y-axis = POD

    CSI = 1 / (1/SR + 1/POD - 1)
    Frequency Bias = POD / SR
    """

    import numpy as np
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, AutoMinorLocator

    # ---------------------------------------------------------
    # Set limits FIRST and keep them fixed
    # ---------------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)

    # ---------------------------------------------------------
    # CSI contours
    # ---------------------------------------------------------
    sr = np.linspace(0.001, 0.999, 500)
    pod = np.linspace(0.001, 0.999, 500)
    SR, POD = np.meshgrid(sr, pod)

    with np.errstate(divide="ignore", invalid="ignore"):
        CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)

    CSI = np.where(
        np.isfinite(CSI) & (CSI >= 0.0) & (CSI <= 1.0),
        CSI,
        np.nan,
    )

    csi_levels = np.arange(0.1, 1.0, 0.1)

    cs = ax.contour(
        SR,
        POD,
        CSI,
        levels=csi_levels,
        colors=csi_color,
        linewidths=0.9,
        alpha=0.90,
        zorder=2,
    )

    ax.clabel(
        cs,
        fmt="%.1f",
        fontsize=contour_label_fontsize,
        colors=csi_color,
        inline=True,
    )

    # ---------------------------------------------------------
    # Frequency-bias lines
    # Bias = POD / SR, so POD = Bias * SR
    # These converge at 0,0.
    # ---------------------------------------------------------
    sr_line = np.linspace(0.0, 1.0, 1000)
    bias_levels = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    for fb in bias_levels:
        pod_line = fb * sr_line
        valid = pod_line <= 1.0

        ax.plot(
            sr_line[valid],
            pod_line[valid],
            linestyle="--",
            color=bias_color,
            lw=1.0,
            alpha=0.85,
            zorder=1,
        )

        if fb < 1.0:
            xlab = 0.90
            ylab = fb * xlab
        elif np.isclose(fb, 1.0):
            xlab = 0.88
            ylab = 0.88
        else:
            ylab = 0.97
            xlab = ylab / fb

        ax.text(
            xlab,
            ylab,
            f"{fb:g}",
            fontsize=bias_label_fontsize,
            ha="center",
            va="center",
            color=bias_color,
            fontweight="bold",
            alpha=0.95,
            clip_on=False,
            zorder=3,
        )

    if show_bias_label:
        ax.text(
            0.43,
            0.54,
            "Frequency bias",
            color=bias_color,
            fontsize=bias_label_fontsize + 1,
            fontweight="bold",
            ha="center",
            va="bottom",
            rotation=48,
            alpha=0.95,
            zorder=3,
        )

    # ---------------------------------------------------------
    # Main axis formatting
    # ---------------------------------------------------------
    major_ticks = np.arange(0.0, 1.01, 0.2)

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax.set_xlabel(
        "Success Ratio (SR = 1 − FAR)",
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    ax.set_ylabel(
        "POD",
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    ax.grid(ls="--", lw=0.6, alpha=0.35)

    ax.tick_params(
        which="major",
        axis="both",
        direction="in",
        length=5,
        top=True,
        right=True,
        bottom=True,
        left=True,
        labelsize=tick_fontsize,
    )

    ax.tick_params(
        which="minor",
        axis="both",
        direction="in",
        length=2.5,
        top=True,
        right=True,
        bottom=True,
        left=True,
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(tick_fontsize)

    # ---------------------------------------------------------
    # Manual CSI right-axis guide
    # This avoids twinx() problems.
    # ---------------------------------------------------------
    if show_csi_right_axis:
        # for level in csi_levels:
        #     ax.text(
        #         1.01,
        #         level,
        #         f"{level:.1f}",
        #         transform=ax.transData,
        #         color=csi_color,
        #         fontsize=tick_fontsize,
        #         fontweight="bold",
        #         ha="left",
        #         va="center",
        #         clip_on=False,
        #     )

        ax.text(
            1.05,
            0.5,
            "CSI",
            transform=ax.transAxes,
            color=csi_color,
            fontsize=axis_label_fontsize,
            fontweight="bold",
            rotation=270,
            ha="center",
            va="center",
            clip_on=False,
        )

    # ---------------------------------------------------------
    # Orientation labels
    # ---------------------------------------------------------
    ax.text(
        0.03,
        0.95,
        "High detection",
        transform=ax.transAxes,
        fontsize=tick_fontsize - 1,
        ha="left",
        va="top",
        fontweight="bold",
        alpha=0.75,
    )

    ax.text(
        0.97,
        0.03,
        "Low false alarm",
        transform=ax.transAxes,
        fontsize=tick_fontsize - 1,
        ha="right",
        va="bottom",
        fontweight="bold",
        alpha=0.75,
    )

    # Final hard reset
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)

def plot_oceanrain_roebber_diagram(
    cat_metrics_hemi,
    products,
    product_colors,
    *,
    hemis=("NH", "SH"),
    figsize=(13.5, 7.2),
    marker_size=95,
    annotate=True,
    title=None,
    contour_label_fontsize=10,
    bias_label_fontsize=10,
    axis_label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=14,
):
    """
    Plot OceanRAIN categorical metrics on a Roebber/performance diagram.

    Parameters
    ----------
    cat_metrics_hemi : dict
        cat_metrics_hemi[hemi][product] -> dict containing at least:
            POD, FAR, Bias, HSS
        This is the output from compute_hemi_metrics_oceanrain().

    products : list
        Product names in plotting order.

    product_colors : dict
        Existing manuscript/product color dictionary.

    hemis : tuple/list
        Hemisphere panels to plot. Default: ("NH", "SH").

    Returns
    -------
    fig, axes
    """

    ncols = len(hemis)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        constrained_layout=False,
        squeeze=False,
    )

    fig.subplots_adjust(
    left=0.07,
    right=0.90,
    bottom=0.18,
    top=0.90,
    wspace=0.28,
    )
    axes = axes.ravel()

    # different markers make products easier to distinguish in grayscale
    marker_cycle = {
        "GPCP v1.3": "o",
        "GPCP v2.3": "o",
        "GPCP v3.2": "s",
        "GPCP v3.3": "D",
        "IMERG v07": "^",
        "ERA5": "P",
        "MERRA2": "X",
        "MERRA-2": "X",
    }

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    hemi_labels = {
        "NH": "NH",
        "SH": "SH",
    }

    for i, (ax, hemi) in enumerate(zip(axes, hemis)):
        draw_perf_background(
            ax,
            contour_label_fontsize=contour_label_fontsize,
            bias_label_fontsize=bias_label_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            csi_color="brown",
            bias_color="steelblue",
            show_csi_right_axis=True,
            show_bias_label=True,
        )

        for product in products:
            mets = cat_metrics_hemi.get(hemi, {}).get(product, {})

            pod = mets.get("POD", np.nan)
            far = mets.get("FAR", np.nan)

            if not np.isfinite(pod) or not np.isfinite(far):
                continue

            sr = 1.0 - far

            if not np.isfinite(sr):
                continue

            color = _get_product_color(product, product_colors)
            marker = marker_cycle.get(product, "o")

            ax.scatter(
                sr,
                pod,
                s=marker_size,
                marker=marker,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=10,
                label=product,
                clip_on=True,
            )

            if annotate:
                ax.annotate(
                    product,
                    xy=(sr, pod),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    color="black",
                )

        ax.text(
            0.02,
            1.03,
            f"{panel_labels[i]} {hemi_labels.get(hemi, hemi)}",
            transform=ax.transAxes,
            fontsize=axis_label_fontsize + 2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Keep full unit-square after all points are plotted
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_autoscale_on(False)

    if title is not None:
        fig.suptitle(title, fontsize=axis_label_fontsize + 3, fontweight="bold")

    # single product legend
    handles = []
    for product in products:
        color = _get_product_color(product, product_colors)
        marker = marker_cycle.get(product, "o")

        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                label=product,
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(products), 6),
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, -0.04),
    )

    # # Add explanatory mini-legend for background
    # bg_handles = [
    #     mpatches.Patch(facecolor="none", edgecolor="brown", label="CSI contours"),
    #     plt.Line2D([0], [0], color="black", linestyle="--", lw=0.8, label="Frequency-bias lines"),
    # ]

    # fig.legend(
    #     handles=bg_handles,
    #     loc="upper center",
    #     ncol=2,
    #     frameon=False,
    #     fontsize=legend_fontsize - 1,
    #     bbox_to_anchor=(0.5, 1.03),
    # )

    return fig, axes


#-------------------------------------------------------------------------------------------------------
def plot_oceanrain_distribution_boxplot(
    df,
    *,
    obs_col="main_mmday",
    product_cols=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    product_colors=None,
    hemi_col="hemi",
    hemis=("NH", "SH"),
    wet_only=False,
    wet_threshold=0.3,
    figsize=(13, 6),
    ylabel="Daily precipitation [mm day$^{-1}$]",
    ylimit=None,
    use_symlog=False,
    panel_label_fontsize=14,
    axis_label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    median_color="lime",
    median_linewidth=2.8,
    mean_marker_size=36,
    legend_ax_index=1,
    whisker_mode="none",
):
    """
    OceanRAIN daily precipitation distribution boxplot by hemisphere.

    Intended as a visual replacement for the descriptive statistics table.

    Boxplot definition:
        box          = 25th to 75th percentile
        thick line   = median
        black circle = mean
        whiskers     = not shown

    Parameters
    ----------
    df : pandas.DataFrame
        Daily OceanRAIN/product collocation dataframe.

    obs_col : str
        OceanRAIN reference column.

    product_cols : sequence
        Product columns to include.

    product_colors : dict
        Product color dictionary used elsewhere in the manuscript.

    hemi_col : str
        Hemisphere column name.

    hemis : sequence
        Hemisphere panels to plot.

    wet_only : bool
        If True, plot only values >= wet_threshold for each dataset.
        If False, plot all valid daily values including zeros.

    wet_threshold : float
        Threshold used when wet_only=True.

    ylimit : tuple or None
        Optional y-axis limits, e.g. (0, 4).

    use_symlog : bool
        If True, use symmetric log scaling to better show the upper tail
        while retaining zero values.

    median_color : str
        Color used for the median line.

    median_linewidth : float
        Thickness of median line.

    legend_ax_index : int
        Axis index where the legend should be placed.
        Default is 1, usually the SH panel for a two-panel figure.

    Returns
    -------
    fig, axes
    """    

    if product_colors is None:
        product_colors = {}

    datasets = [("OceanRAIN", obs_col)] + [(p, p) for p in product_cols]

    def _color(name):
        if name == "OceanRAIN":
            return "white"

        aliases = {
            "MERRA2": "MERRA-2",
            "MERRA-2": "MERRA2",
            "IMERG": "IMERG v07",
            "IMERG v07": "IMERG",
        }

        if name in product_colors:
            return product_colors[name]

        alt = aliases.get(name)
        if alt is not None and alt in product_colors:
            return product_colors[alt]

        return "0.7"

    fig, axes = plt.subplots(
        1,
        len(hemis),
        figsize=figsize,
        constrained_layout=False,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.24,
        top=0.88,
        wspace=0.18,
    )

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, (ax, hemi) in enumerate(zip(axes, hemis)):
        dsub = df[df[hemi_col] == hemi].copy()

        box_stats = []
        labels = []
        colors = []
        means = []

        for name, col in datasets:
            if col not in dsub.columns:
                continue

            s = pd.to_numeric(dsub[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()

            if wet_only:
                s = s[s >= wet_threshold]

            if len(s) == 0:
                continue

            q05 = np.nanpercentile(s, 5)
            q25 = np.nanpercentile(s, 25)
            q50 = np.nanpercentile(s, 50)
            q75 = np.nanpercentile(s, 75)
            q95 = np.nanpercentile(s, 95)
            vmin = np.nanmin(s)
            vmax = np.nanmax(s)
            mean = np.nanmean(s)

            if whisker_mode == "none":
                whislo = q25
                whishi = q75

            elif whisker_mode == "p5_p95":
                whislo = q05
                whishi = q95

            elif whisker_mode == "min_max":
                whislo = vmin
                whishi = vmax

            elif whisker_mode == "tukey":
                iqr = q75 - q25
                lower_fence = q25 - 1.5 * iqr
                upper_fence = q75 + 1.5 * iqr

                inlier_vals = s[(s >= lower_fence) & (s <= upper_fence)]

                if len(inlier_vals) > 0:
                    whislo = np.nanmin(inlier_vals)
                    whishi = np.nanmax(inlier_vals)
                else:
                    whislo = q25
                    whishi = q75

            else:
                raise ValueError(
                    "whisker_mode must be one of: 'none', 'p5_p95', 'min_max', 'tukey'"
                )

            box_stats.append({
                "label": name,
                "whislo": whislo,
                "q1": q25,
                "med": q50,
                "q3": q75,
                "whishi": whishi,
                "mean": mean,
                "fliers": [],
            })

            labels.append(name)
            colors.append(_color(name))
            means.append(s.mean())

        positions = np.arange(1, len(box_stats) + 1)

        bp = ax.bxp(
            box_stats,
            positions=positions,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
            showmeans=False,
            manage_ticks=False,
        )

        # Box colors
        for patch, color, name in zip(bp["boxes"], colors, labels):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.78 if name != "OceanRAIN" else 1.0)
            patch.set_linewidth(1.1)

        # Whiskers/caps
        if whisker_mode == "none":
            for element in ["whiskers", "caps"]:
                for artist in bp[element]:
                    artist.set_alpha(0.0)
                    artist.set_linewidth(0.0)
        else:
            for element in ["whiskers", "caps"]:
                for artist in bp[element]:
                    artist.set_color("black")
                    artist.set_linewidth(1.1)
                    artist.set_alpha(0.90)

        # Median line: lime, thick
        for median in bp["medians"]:
            median.set_color(median_color)
            median.set_linewidth(median_linewidth)
            median.set_zorder(5)

        # Mean overlay
        ax.scatter(
            positions,
            means,
            marker="o",
            s=mean_marker_size,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
            fontsize=tick_fontsize,
            fontweight="bold",
        )

        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
            direction="in",
            length=5,
            right=True,
            top=True,
        )

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(tick_fontsize)

        if i == 0:
            ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, fontweight="bold")

        ax.text(
            0.02,
            1.03,
            f"{panel_labels[i]} {hemi}",
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        ax.grid(axis="y", ls="--", lw=0.6, alpha=0.35)

        if use_symlog:
            ax.set_yscale("symlog", linthresh=0.1)

        if ylimit is not None:
            ax.set_ylim(*ylimit)

    # Legend placed inside one selected panel
    legend_handles = [
        Patch(
            facecolor="0.8",
            edgecolor="black",
            label="Box: 25th–75th percentile",
        ),
        Line2D(
            [0],
            [0],
            color=median_color,
            lw=median_linewidth,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="None",
            markersize=6,
            label="Mean",
        ),
    ]

    legend_ax_index = min(legend_ax_index, len(axes) - 1)

    axes[legend_ax_index].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="0.7",
        fontsize=legend_fontsize,
    )

    return fig, axes

def attach_satellite_vars_pointwise_chunked(
    df: pd.DataFrame,
    xr_obj,
    var_map: dict,
    *,
    date_col="date",
    lat_col="lat",
    lon_col="lon",
    method="nearest",
    tolerance_time=None,
    lon_wrap=True,
    chunk_size=200_000,
):
    """
    Attach gridded product values to point observations using simultaneous
    pointwise time-lat-lon selection.

    This avoids the memory problem caused by selecting time first, which can
    create an intermediate array with shape (points, lat, lon).

    The selection logic is PAL-like:
        da.sel(time=("points", t), lat=("points", y), lon=("points", x))

    Parameters
    ----------
    df : pandas.DataFrame
        Point observation dataframe.

    xr_obj : xarray.Dataset or xarray.DataArray
        Product data.

    var_map : dict
        Mapping from output column names to product variable names.
        If xr_obj is a DataArray, use {output_name: None}.

    chunk_size : int
        Number of point samples to select per chunk.

    Returns
    -------
    pandas.DataFrame
        Copy of df with product columns attached.
    """

    import numpy as np
    import pandas as pd
    import xarray as xr

    out = df.copy()

    # Build DataArray dictionary from Dataset/DataArray input
    das = {}

    if isinstance(xr_obj, xr.Dataset):
        for new_col, varname in var_map.items():
            if varname is None:
                raise ValueError(
                    f"For Dataset input, var_map['{new_col}'] must be a variable name."
                )
            das[new_col] = xr_obj[varname]

    elif isinstance(xr_obj, xr.DataArray):
        for new_col, varname in var_map.items():
            if varname is not None:
                # Allow this, but ignore varname because object is already a DataArray
                pass
            das[new_col] = xr_obj

    else:
        raise TypeError("xr_obj must be an xarray Dataset or DataArray.")

    # Ensure date column is datetime64[ns], timezone-naive
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if getattr(out[date_col].dt, "tz", None) is not None:
        out[date_col] = out[date_col].dt.tz_convert(None)

    out[date_col] = out[date_col].values.astype("datetime64[ns]")

    # Coordinates as arrays
    lat_all = out[lat_col].to_numpy(dtype="float64")
    lon_all = out[lon_col].to_numpy(dtype="float64")
    if lon_wrap:
        lon_all = ((lon_all + 180.0) % 360.0) - 180.0

    time_all = out[date_col].to_numpy(dtype="datetime64[ns]")

    for new_col, da in das.items():
        da = da.copy()

        tdim, ydim, xdim = _infer_dims(da)
        da = _ensure_datetime_coord_naive(da, tdim)

        vals_out = np.full(len(out), np.nan, dtype="float64")

        tcoord = pd.to_datetime(da[tdim].values, errors="coerce")
        tmin = tcoord.min().to_datetime64()
        tmax = tcoord.max().to_datetime64()

        valid_mask = (
            np.isfinite(lat_all) &
            np.isfinite(lon_all) &
            pd.notna(time_all) &
            (time_all >= tmin) &
            (time_all <= tmax)
        )

        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
            out[new_col] = vals_out
            continue

        for start in range(0, len(valid_idx), chunk_size):
            idx = valid_idx[start:start + chunk_size]

            t_indexer = xr.DataArray(time_all[idx], dims="points")
            y_indexer = xr.DataArray(lat_all[idx], dims="points")
            x_indexer = xr.DataArray(lon_all[idx], dims="points")

            try:
                if tolerance_time is None:
                    vals = da.sel(
                        {
                            tdim: t_indexer,
                            ydim: y_indexer,
                            xdim: x_indexer,
                        },
                        method=method,
                    ).values
                else:
                    vals = da.sel(
                        {
                            tdim: t_indexer,
                            ydim: y_indexer,
                            xdim: x_indexer,
                        },
                        method=method,
                        tolerance=tolerance_time,
                    ).values

                vals_out[idx] = np.asarray(vals).reshape(-1)

            except KeyError:
                # Leave this chunk as NaN if no match
                continue

        out[new_col] = vals_out

    return out

def compute_oceanrain_sample_distribution_summary(
    df,
    *,
    obs_col="main_mmday",
    product_cols=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"),
    hemi_col="hemi",
    hemis=("NH", "SH"),
    wet_only=False,
    wet_threshold=0.3,
):
    """
    Compute summary statistics for OceanRAIN and product daily precipitation
    over the matched sampled data.

    This is intended to document the sample population behind the quantitative
    bar-plot metrics.

    Summary statistics:
        N
        mean
        median
        std
        p95
        sum
        sum_ratio_to_OceanRAIN
        sum_bias_percent_vs_OceanRAIN

    Returns
    -------
    pandas.DataFrame
    """

    import numpy as np
    import pandas as pd

    datasets = [("OceanRAIN", obs_col)] + [(p, p) for p in product_cols]

    rows = []

    for hemi in hemis:
        dsub = df[df[hemi_col] == hemi].copy()

        # OceanRAIN reference sum for this hemisphere
        ref = pd.to_numeric(dsub[obs_col], errors="coerce")
        ref = ref.replace([np.inf, -np.inf], np.nan).dropna()

        if wet_only:
            ref = ref[ref >= wet_threshold]

        ref_sum = ref.sum() if len(ref) > 0 else np.nan

        for name, col in datasets:
            if col not in dsub.columns:
                continue

            s = pd.to_numeric(dsub[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()

            if wet_only:
                if wet_threshold == 0:
                    s = s[s > 0]
                else:
                    s = s[s >= wet_threshold]

            if len(s) == 0:
                rows.append({
                    "hemi": hemi,
                    "dataset": name,
                    "N": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "p95": np.nan,
                    "sum": np.nan,
                    "sum_ratio_to_OceanRAIN": np.nan,
                    "sum_bias_percent_vs_OceanRAIN": np.nan,
                })
                continue

            this_sum = s.sum()

            if name == "OceanRAIN":
                sum_ratio = 1.0
                sum_bias = 0.0
            else:
                sum_ratio = this_sum / ref_sum if ref_sum > 0 else np.nan
                sum_bias = 100.0 * (this_sum - ref_sum) / ref_sum if ref_sum > 0 else np.nan

            rows.append({
                "hemi": hemi,
                "dataset": name,
                "N": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "p95": np.nanpercentile(s, 95),
                "sum": this_sum,
                "sum_ratio_to_OceanRAIN": sum_ratio,
                "sum_bias_percent_vs_OceanRAIN": sum_bias,
            })

    return pd.DataFrame(rows)
#------------------------------------------------------------------------------
def oceanrain_attach_products_minute_native_then_daily(
    oc_df_minute,
    *,
    products,
    date_col="date",
    time_col="time_utc",
    lat_col="lat",
    lon_col="lon",
    ship_col="ship",
    obs_rate_col="rate_main_mmph",
    precip_flag_col="precip_flag",
    gpcp_lat_1d=None,
    gpcp_lon_1d=None,
    lat_abs_min=45.0,
    coverage_frac=0.5,
    min_valid_minutes=None,
    group_mode="ship_grid_day",
    method="nearest",
    tolerance_time=None,
    mask_negative_products=True,
):
    """
    PAL-like OceanRAIN sensitivity workflow.

    This function attaches product values at the OceanRAIN minute-level
    locations first, then aggregates OceanRAIN and product values to daily
    samples using the same valid minutes.

    This avoids using a daily mean ship coordinate for product extraction.

    Parameters
    ----------
    oc_df_minute : pandas.DataFrame
        QC-passed OceanRAIN minute-level dataframe.

    products : dict
        {"Product name": (xr_obj, var_map)}
        The product fields should be daily precipitation fields. They may be
        native-grid products or already regridded products.

    date_col : str
        Name of date column to create/use for daily matching.

    time_col : str
        OceanRAIN timestamp column.

    lat_col, lon_col : str
        Minute-level OceanRAIN coordinates.

    obs_rate_col : str
        OceanRAIN minute precipitation rate column in mm h-1.

    gpcp_lat_1d, gpcp_lon_1d : arrays or None
        If group_mode="ship_grid_day", these define the grid used for
        assigning minute samples to daily ship-grid-cell groups.

    group_mode : {"ship_grid_day", "ship_day"}
        ship_grid_day:
            Aggregate by date + ship + common grid cell. This is closest to
            your current OceanRAIN daily design.

        ship_day:
            Aggregate by date + ship only. This follows the moving ship track
            for the whole day and does not split by grid cell.

    method : str
        Product selection method, usually "nearest".

    Returns
    -------
    daily_attached : pandas.DataFrame
        Daily OceanRAIN/product matched dataframe.
    minute_attached : pandas.DataFrame
        Minute-level dataframe with product values attached.
    """

    df = oc_df_minute.copy()

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col, ship_col, obs_rate_col]).copy()

    if lat_abs_min is not None:
        df = df[df[lat_col].abs() >= float(lat_abs_min)].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Daily product time coordinate
    df[date_col] = df[time_col].dt.floor("D").dt.tz_localize(None)

    # Standard lon range for selection
    df[lon_col] = ((df[lon_col].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0

    # Keep only valid finite OceanRAIN rates
    df[obs_rate_col] = pd.to_numeric(df[obs_rate_col], errors="coerce")
    df = df[np.isfinite(df[obs_rate_col])].copy()

    # Add grid indices if grouping by ship-grid-day
    if group_mode == "ship_grid_day":
        if gpcp_lat_1d is None or gpcp_lon_1d is None:
            raise ValueError(
                "gpcp_lat_1d and gpcp_lon_1d are required when group_mode='ship_grid_day'."
            )

        lon_wrapped = ((df[lon_col].to_numpy(dtype="float64") + 180.0) % 360.0) - 180.0

        df["ilat"] = map_to_gpcp_idx(
            np.asarray(gpcp_lat_1d),
            df[lat_col].to_numpy(dtype="float64"),
        )
        df["ilon"] = map_to_gpcp_idx(
            np.asarray(gpcp_lon_1d),
            lon_wrapped,
        )

        g_lat = np.asarray(gpcp_lat_1d)
        g_lon = np.asarray(gpcp_lon_1d)

        df["lat_c"] = g_lat[df["ilat"].to_numpy()]
        df["lon_c"] = g_lon[df["ilon"].to_numpy()]
        df["hemi"] = np.where(df["lat_c"] >= 0, "NH", "SH")

        grp_keys = [date_col, "ilat", "ilon", ship_col]

    elif group_mode == "ship_day":
        df["hemi"] = np.where(df[lat_col] >= 0, "NH", "SH")
        grp_keys = [date_col, ship_col]

    else:
        raise ValueError("group_mode must be either 'ship_grid_day' or 'ship_day'.")

    # Attach products at minute-level OceanRAIN locations
    minute_attached = df.copy()

    for name, (xr_obj, var_map) in products.items():
        print(f"Minute-native attaching: {name}")

        minute_attached = attach_satellite_vars_pointwise_chunked(
        minute_attached,
        xr_obj,
        var_map=var_map,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        method=method,
        tolerance_time=tolerance_time,
        chunk_size=200_000,
    )

    # Product columns from var_map keys
    product_cols = []
    for _, (_, var_map) in products.items():
        product_cols.extend(list(var_map.keys()))

    product_cols = list(dict.fromkeys(product_cols))

    if mask_negative_products:
        for col in product_cols:
            if col in minute_attached.columns:
                minute_attached[col] = pd.to_numeric(minute_attached[col], errors="coerce")
                minute_attached.loc[minute_attached[col] < 0, col] = np.nan

    # Daily aggregation
    if min_valid_minutes is None:
        min_valid_minutes = int(float(coverage_frac) * 1440)

    agg_dict = {
        "n_min_total": (obs_rate_col, "size"),
        "n_min_valid_rate": (obs_rate_col, lambda s: np.sum(np.isfinite(pd.to_numeric(s, errors="coerce")))),
        "main_mean_mmph": (obs_rate_col, "mean"),
        "main_median_mmph": (obs_rate_col, "median"),
        "main_max_mmph": (obs_rate_col, "max"),
        "main_mmday": (obs_rate_col, _mmday_from_mmph),
        "lat_mean": (lat_col, "mean"),
        "lon_mean": (lon_col, "mean"),
        "hemi": ("hemi", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
    }

    if precip_flag_col in minute_attached.columns:
        agg_dict.update({
            "n_min_zero": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 3)),
            "n_min_rain": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 0)),
            "n_min_snow": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 1)),
            "n_min_mixed": (precip_flag_col, lambda s: np.sum(s.to_numpy() == 2)),
            "frac_zero": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 3)),
            "frac_rain": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 0)),
            "frac_snow": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 1)),
            "frac_mixed": (precip_flag_col, lambda s: np.mean(s.to_numpy() == 2)),
        })

    if group_mode == "ship_grid_day":
        agg_dict.update({
            "lat_c": ("lat_c", "first"),
            "lon_c": ("lon_c", "first"),
        })

    # For products: daily estimate is the mean of minute-sampled daily product values
    for col in product_cols:
        if col in minute_attached.columns:
            agg_dict[col] = (col, "mean")

    daily_attached = (
        minute_attached
        .groupby(grp_keys, as_index=False)
        .agg(**agg_dict)
    )

    daily_attached["coverage_frac_day"] = daily_attached["n_min_total"] / 1440.0
    daily_attached = daily_attached[daily_attached["n_min_total"] >= min_valid_minutes].copy()

    if "lon_mean" in daily_attached.columns:
        daily_attached["lon_mean"] = (
            (daily_attached["lon_mean"].to_numpy(dtype="float64") + 180.0) % 360.0
        ) - 180.0

    return daily_attached, minute_attached


def quantitative_dict_to_table(
    qt_metrics_hemi,
    products_order=None,
    hemis=("NH", "SH"),
):
    """
    Convert OceanRAIN quantitative metric dictionary to a flat DataFrame.

    Expected input structure:
        qt_metrics_hemi[hemi][product] = {
            "CC": ...,
            "RB": ... or "RB_%": ...,
            "RMSE": ...,
            "MAE": ...,
            ...
        }

    Returns
    -------
    pandas.DataFrame
        Columns: hemi, product, metric columns
    """

    import pandas as pd
    import numpy as np

    rows = []

    for hemi in hemis:
        if hemi not in qt_metrics_hemi:
            continue

        hemi_dict = qt_metrics_hemi[hemi]

        if products_order is None:
            products = list(hemi_dict.keys())
        else:
            products = products_order

        for product in products:
            if product not in hemi_dict:
                continue

            mets = hemi_dict[product]

            row = {
                "hemi": hemi,
                "product": product,
            }

            for key, val in mets.items():
                row[key] = val

            rows.append(row)

    out = pd.DataFrame(rows)

    # Optional: standardize relative-bias column name if needed
    if "RB" in out.columns and "RB_%" not in out.columns:
        out = out.rename(columns={"RB": "RB_%"})

    return out


def categorical_dict_to_table(
    cat_metrics_hemi,
    products_order=None,
    hemis=("NH", "SH"),
):
    """
    Convert OceanRAIN categorical metric dictionary to a flat DataFrame.

    Expected input structure:
        cat_metrics_hemi[hemi][product] = {
            "POD": ...,
            "FAR": ...,
            "Bias": ...,
            "HSS": ...,
            ...
        }

    Returns
    -------
    pandas.DataFrame
        Columns: hemi, product, metric columns
    """

    import pandas as pd

    rows = []

    for hemi in hemis:
        if hemi not in cat_metrics_hemi:
            continue

        hemi_dict = cat_metrics_hemi[hemi]

        if products_order is None:
            products = list(hemi_dict.keys())
        else:
            products = products_order

        for product in products:
            if product not in hemi_dict:
                continue

            mets = hemi_dict[product]

            row = {
                "hemi": hemi,
                "product": product,
            }

            for key, val in mets.items():
                row[key] = val

            rows.append(row)

    return pd.DataFrame(rows)


#-----------------------------------------------------------
def compute_native_daily_spatial_mean_timeseries(
    product_map,
    *,
    region="highlat_all",
    lat_abs_min=45.0,
    start_date=None,
    end_date=None,
    join="inner",
    mask_negative=True,
):
    """
    Compute native-resolution daily spatial-mean precipitation time series.

    Each product remains on its native grid. For each day, the spatial field is
    collapsed to one area-weighted mean value using cos(latitude) weights.

    Parameters
    ----------
    product_map : dict
        Example:
        {
            "GPCP v3.3": gpcp_ds_v3pt3_al["precip"],
            "IMERG v07": imerg_v07_al,
            "ERA5": era5_ds_al["tp"],
        }

        Values can be xarray DataArrays or (xarray object, variable name).

    region : {"global", "NH", "SH", "highlat_all"}
        Region to average.

    lat_abs_min : float
        Used when region="highlat_all", "NH", or "SH".

    join : {"inner", "outer"}
        How to combine product time series.

    Returns
    -------
    pandas.DataFrame
        Daily time series with columns:
        date, product1, product2, ...
    """ 

    def _as_dataarray(obj):
        if isinstance(obj, tuple):
            xr_obj = obj[0]
            varinfo = obj[1] if len(obj) > 1 else None

            if isinstance(xr_obj, xr.DataArray):
                return xr_obj

            if isinstance(xr_obj, xr.Dataset):
                if isinstance(varinfo, str):
                    return xr_obj[varinfo]

                if isinstance(varinfo, dict):
                    # Use first non-None variable name if provided.
                    varnames = [v for v in varinfo.values() if v is not None]
                    if len(varnames) > 0:
                        return xr_obj[varnames[0]]

                raise ValueError(
                    "Dataset tuple input needs a variable name, e.g. (dataset, 'precip') "
                    "or a var_map with a non-None variable name."
                )

            raise TypeError(f"Unsupported tuple xarray object type: {type(xr_obj)}")

        if isinstance(obj, xr.DataArray):
            return obj

        if isinstance(obj, xr.Dataset):
            raise ValueError(
                "Dataset input needs a variable name: use (dataset, 'varname')."
            )

        raise TypeError(f"Unsupported product_map value type: {type(obj)}")

    def _infer_time_lat_lon_dims(da):
        dims = list(da.dims)

        time_candidates = [
            "time",
            "valid_time",
            "date",
            "datetime",
        ]

        lat_candidates = [
            "lat",
            "latitude",
            "y",
        ]

        lon_candidates = [
            "lon",
            "longitude",
            "x",
        ]

        tdim = next((d for d in time_candidates if d in dims), None)
        ydim = next((d for d in lat_candidates if d in dims), None)
        xdim = next((d for d in lon_candidates if d in dims), None)

        if tdim is None or ydim is None or xdim is None:
            raise ValueError(
                f"Could not infer time/lat/lon dims from {da.dims}. "
                "Expected time/valid_time, lat/y, lon/x."
            )

        return tdim, ydim, xdim

    series_list = []

    for product, obj in product_map.items():
        print(f"Computing native daily spatial mean: {product}")

        da = _as_dataarray(obj).copy()
        tdim, ydim, xdim = _infer_time_lat_lon_dims(da)

        # Rename internally for easier handling
        rename_dict = {}

        if tdim != "time":
            rename_dict[tdim] = "time"

        if ydim != "lat":
            rename_dict[ydim] = "lat"

        if xdim != "lon":
            rename_dict[xdim] = "lon"

        if rename_dict:
            da = da.rename(rename_dict)

        da["time"] = pd.to_datetime(da["time"].values)

        if start_date is not None or end_date is not None:
            da = da.sel(
                time=slice(
                    pd.to_datetime(start_date) if start_date is not None else None,
                    pd.to_datetime(end_date) if end_date is not None else None,
                )
            )

        if mask_negative:
            da = da.where(da >= 0)

        lat = da["lat"]

        if region == "global":
            region_mask = xr.ones_like(lat, dtype=bool)

        elif region == "highlat_all":
            region_mask = abs(lat) >= lat_abs_min

        elif region == "NH":
            region_mask = lat >= lat_abs_min

        elif region == "SH":
            region_mask = lat <= -lat_abs_min

        else:
            raise ValueError(
                "region must be one of: 'global', 'highlat_all', 'NH', 'SH'"
            )

        # Mask data outside selected region
        da_reg = da.where(region_mask)

        # Area weights for regular lat-lon grids.
        # xarray.weighted() cannot accept NaNs in weights, so fill missing values with 0.
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(region_mask, 0.0)
        weights = weights.where(np.isfinite(weights), 0.0)
        weights = weights.clip(min=0.0)
        weights = weights.fillna(0.0)

        ts = da_reg.weighted(weights).mean(
            dim=("lat", "lon"),
            skipna=True,
        )

        s = ts.to_series()
        s.name = product
        series_list.append(s)

    ts_df = pd.concat(series_list, axis=1, join=join)
    ts_df = ts_df.reset_index().rename(columns={"time": "date"})

    return ts_df

def plot_native_daily_spatial_mean_timeseries(
    ts_df,
    *,
    products,
    product_colors,
    date_col="date",
    rolling=None,
    figsize=(13, 5.5),
    ylabel="Area-weighted mean precipitation [mm day$^{-1}$]",
    title=None,
    tick_fontsize=12,
    axis_label_fontsize=13,
    legend_fontsize=12,
):
    """
    Plot native-resolution daily spatial-mean product time series.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    df = ts_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    for product in products:
        if product not in df.columns:
            continue

        y = pd.to_numeric(df[product], errors="coerce")

        if rolling is not None and rolling > 1:
            y = y.rolling(rolling, center=True, min_periods=max(1, rolling // 2)).mean()

        ax.plot(
            df[date_col],
            y,
            label=product,
            color=_get_product_color(product, product_colors, default="0.7"),
            linewidth=3.0,
            alpha=0.95,
        )

    ax.set_ylabel(
        ylabel,
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Date",
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    if title is not None:
        ax.set_title(title, fontsize=axis_label_fontsize + 2, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(tick_fontsize)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(products), 6),
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.tight_layout()

    return fig, ax


def mask_common05deg_product_map(product_map, spatial_mask):
    """
    Apply a common 2-D categorical mask to each product.

    The mask is sampled to each product grid using nearest-neighbor
    selection and then assigned the product's exact coordinate labels
    before xarray alignment.
    """

    masked_map = {}

    for product, obj in product_map.items():

        if isinstance(obj, tuple):
            xr_obj = obj[0]
            var_map = obj[1]
        else:
            xr_obj = obj
            var_map = {product: None}

        if isinstance(xr_obj, xr.DataArray):
            da = xr_obj

        elif isinstance(xr_obj, xr.Dataset):
            source_vars = [
                source_var
                for source_var in var_map.values()
                if source_var is not None
            ]

            if len(source_vars) != 1:
                raise ValueError(
                    f"Could not identify one source variable for {product}"
                )

            da = xr_obj[source_vars[0]]

        else:
            raise TypeError(
                f"Unsupported object for {product}: {type(xr_obj)}"
            )

        # Normalize longitude convention if needed
        if float(da["lon"].max()) > 180:
            da = da.assign_coords(
                lon=((da["lon"] + 180.0) % 360.0) - 180.0
            ).sortby("lon")

        # Sample categorical mask to the product grid
        mask_aligned = spatial_mask.sel(
            lat=xr.DataArray(da["lat"].values, dims="lat"),
            lon=xr.DataArray(da["lon"].values, dims="lon"),
            method="nearest",
        )

        # Critical step:
        # sel(method="nearest") may retain the source mask coordinate labels.
        # Replace them with the product's exact labels before where().
        mask_aligned = mask_aligned.assign_coords(
            lat=da["lat"],
            lon=da["lon"],
        )

        # Remove scalar coordinates introduced by rioxarray, if present
        mask_aligned = mask_aligned.drop_vars(
            ["spatial_ref"],
            errors="ignore",
        )

        masked_da = da.where(mask_aligned.astype(bool))

        print(
            product,
            "product shape:",
            da.shape,
            "mask shape:",
            mask_aligned.shape,
            "masked shape:",
            masked_da.shape,
        )

        masked_map[product] = (
            masked_da,
            var_map,
        )

    return masked_map