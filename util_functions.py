#%% IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
#%% GLOBAL VARIABLES
# DEFINE REGIONS AND THEIR BOUNDARIES (based on Figure 1)
region_bounds = {
    "ETNP": {"lon_min": -160, "lon_max": -130, "lat_min": 35, "lat_max": 55},  # Extratropical North Pacific
    "TNEP": {"lon_min": -140, "lon_max": -90,  "lat_min": 5,  "lat_max": 25},  # Tropical Northeastern Pacific
    "TSEP": {"lon_min": -120, "lon_max": -70,  "lat_min": -25, "lat_max": -5}, # Tropical Southeastern Pacific
    "STNA": {"lon_min": -70,  "lon_max": -10,  "lat_min": 15, "lat_max": 45},  # Subtropical North Atlantic
    "TNIO": {"lon_min": 60,   "lon_max": 100,  "lat_min": -5, "lat_max": 20},  # Tropical North Indian Ocean
    "TNWP": {"lon_min": 120,  "lon_max": 180,  "lat_min": 5,  "lat_max": 25},  # Tropical Northwestern Pacific
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
}
#%% DEFINE FUNCTIONS
# FUNCTION TO CLASSIFY AND GROUP PAL FILES BASED ON REGIONS

def boxes_overlap(lat_min1, lat_max1, lon_min1, lon_max1,
                  lat_min2, lat_max2, lon_min2, lon_max2):
    return not (lat_max1 < lat_min2 or lat_min1 > lat_max2 or
                lon_max1 < lon_min2 or lon_min1 > lon_max2)
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

                if boxes_overlap(lat_min_file, lat_max_file,
                                 lon_min_file, lon_max_file,
                                 lat_min_r, lat_max_r,
                                 lon_min_r, lon_max_r):
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

