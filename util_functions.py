#%% IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
from rasterio.transform import from_origin
from rasterio.transform import rowcol
#%% GLOBAL VARIABLES
# DEFINE REGIONS AND THEIR BOUNDARIES (based on Figure 1 and PAL data coverage)
region_bounds = {
    "ETNP": {"lon_min": -170, "lon_max": -120, "lat_min": 30, "lat_max": 60},  # Extratropical North Pacific 
    "TNEP": {"lon_min": -180, "lon_max": -80, "lat_min": 0, "lat_max": 30}, # Tropical Northeastern Pacific (northern hemisphere only, extended to capture SPURS2 and Caribbean PALs)
    "TSEP": {"lon_min": -160, "lon_max": -70,  "lat_min": -25, "lat_max": 0},  # Tropical Southeastern Pacific (southern hemisphere only)
    "STNA": {"lon_min": -70,  "lon_max": -10,  "lat_min": 15, "lat_max": 45},  # Subtropical North Atlantic
    "TNIO": {"lon_min": 60,   "lon_max": 100,  "lat_min": -5, "lat_max": 20},  # Tropical North Indian Ocean
    "TNWP": {"lon_min": 120,  "lon_max": 180,  "lat_min": -5, "lat_max": 30},  # Tropical Northwestern Pacific
}
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# DEFINE COLORS FOR EACH REGION
region_colors = {
    "ETNP": "blue",
    "TNEP": "green",
    "TSEP": "orange",
    "STNA": "red",
    "TNIO": "purple",
    "TNWP": "brown",
    "Unclassified": "black",
}
#%% DEFINE FUNCTIONS
# FUNCTION TO CLASSIFY AND GROUP PAL FILES BASED ON REGIONS

def assign_to_gpcp_grid(lat,lon, resolution):
    """
    Assign each PAL observation (lat, lon) to a GPCP grid cell using rasterio.

    Parameters:
    - pal_df: pandas.DataFrame with at least columns ['lat', 'lon', 'time', 'rain']
    - gpcp_resolution: float (1.0 for v1.3, 0.5 for v3.2)

    Returns:
    - pandas.DataFrame with additional columns: 'row', 'col', and 'date'
    """
    # Define affine transform for the GPCP grid
    transform = from_origin(west=-180.0, north=90.0, xsize=resolution, ysize=resolution)

    # Use rasterio to compute grid indices
    row, col = rowcol(transform, lon, lat)

    return row, col
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def boxes_overlap(lat_min1, lat_max1, lon_min1, lon_max1,
                  lat_min2, lat_max2, lon_min2, lon_max2):
    """
    Check if two bounding boxes overlap.
    Box 1: PAL trajectory bounding box
    Box 2: Region bounding box
    """
    # Check for NO overlap conditions
    no_overlap = (lat_max1 < lat_min2 or  # PAL is completely south of region
                  lat_min1 > lat_max2 or  # PAL is completely north of region  
                  lon_max1 < lon_min2 or  # PAL is completely west of region
                  lon_min1 > lon_max2)    # PAL is completely east of region
    
    # If there's no overlap, return False; otherwise return True
    return not no_overlap

def simple_box_check(lat_min_file, lat_max_file, lon_min_file, lon_max_file,
                     lat_min_r, lat_max_r, lon_min_r, lon_max_r):
    """
    Alternative simpler check: does the PAL box overlap with region box?
    """
    lat_overlap = not (lat_max_file < lat_min_r or lat_min_file > lat_max_r)
    lon_overlap = not (lon_max_file < lon_min_r or lon_min_file > lon_max_r)
    return lat_overlap and lon_overlap
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def classify_and_group_files_fixed(file_list):
    classification = {region: [] for region in region_bounds}
    classification["Unclassified"] = []

    for file_path in file_list:
        try:
            ds = xr.open_dataset(file_path)
            lat = ds['lat'].values
            lon = ds['lon'].values

            # Handle masked arrays or _FillValue
            lat = np.array(lat.filled(np.nan)) if hasattr(lat, "filled") else lat
            lon = np.array(lon.filled(np.nan)) if hasattr(lon, "filled") else lon

            # Normalize longitude
            lon = (lon + 360) % 360
            lon[lon > 180] -= 360

            med_lat = float(np.nanmedian(lat))
            med_lon = float(np.nanmedian(lon))

            found = False
            for region, bounds in region_bounds.items():
                lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
                lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
                if lat_min <= med_lat <= lat_max and lon_min <= med_lon <= lon_max:
                    classification[region].append(file_path)
                    found = True
                    break
            if not found:
                classification["Unclassified"].append(file_path)
        except Exception as e:
            classification["Unclassified"].append(file_path)
    return classification

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FUNCTION TO CLASSIFY AND GROUP PAL FILES BASED ON BOUNDING BOXES
def classify_and_group_files_bounding_box(file_list, region_bounds_dict=None):
    """
    Classify PAL files into regions based on bounding box overlap.
    Returns a dictionary of {region: list_of_files}
    """
    if region_bounds_dict is None:
        region_bounds_dict = region_bounds
    
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

#%% DEBUG FUNCTION
def debug_overlap_test():
    """Test the boxes_overlap function with specific PAL coordinates"""
    # PAL 19412: Lat 1.04-3.09, Lon 164.90-174.91
    # TNWP: Lat -5 to 30, Lon 120-180
    
    lat_min_file, lat_max_file = 1.04, 3.09
    lon_min_file, lon_max_file = 164.90, 174.91
    
    lat_min_r, lat_max_r = -5, 30
    lon_min_r, lon_max_r = 120, 180
    
    print(f"PAL 19412: Lat {lat_min_file}-{lat_max_file}, Lon {lon_min_file}-{lon_max_file}")
    print(f"TNWP: Lat {lat_min_r}-{lat_max_r}, Lon {lon_min_r}-{lon_max_r}")
    
    # Test individual conditions
    cond1 = lat_max_file < lat_min_r  # 3.09 < -5
    cond2 = lat_min_file > lat_max_r  # 1.04 > 30
    cond3 = lon_max_file < lon_min_r  # 174.91 < 120
    cond4 = lon_min_file > lon_max_r  # 164.90 > 180
    
    print(f"lat_max_file < lat_min_r: {lat_max_file} < {lat_min_r} = {cond1}")
    print(f"lat_min_file > lat_max_r: {lat_min_file} > {lat_max_r} = {cond2}")
    print(f"lon_max_file < lon_min_r: {lon_max_file} < {lon_min_r} = {cond3}")
    print(f"lon_min_file > lon_max_r: {lon_min_file} > {lon_max_r} = {cond4}")
    
    overall = cond1 or cond2 or cond3 or cond4
    result = not overall
    
    print(f"Overall OR: {overall}")
    print(f"NOT overall (should overlap): {result}")
    
    # Test with actual function
    actual_result = boxes_overlap(lat_min_file, lat_max_file, lon_min_file, lon_max_file,
                                 lat_min_r, lat_max_r, lon_min_r, lon_max_r)
    print(f"boxes_overlap function result: {actual_result}")

# Call the debug function
# debug_overlap_test()

