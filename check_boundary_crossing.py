#!/usr/bin/env python3

# Check for PALs that might cross multiple region boundaries
import os
import sys
import xarray as xr

# Import our utilities
from util_functions import *

def check_pal_overlaps(file_path, region_bounds_dict):
    """Check which regions a single PAL file overlaps with"""
    try:
        ds = xr.open_dataset(file_path)
        
        # Get lat/lon
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
            return []
        
        # Calculate bounding box
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        
        # Check overlap with each region
        overlapping_regions = []
        for region, bounds in region_bounds_dict.items():
            overlap = simple_box_check(lat_min, lat_max, lon_min, lon_max,
                                     bounds["lat_min"], bounds["lat_max"],
                                     bounds["lon_min"], bounds["lon_max"])
            if overlap:
                overlapping_regions.append(region)
        
        ds.close()
        return overlapping_regions, (lat_min, lat_max, lon_min, lon_max)
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return [], None

# Get the PAL files
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'
all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]

print("Checking for PALs that overlap multiple regions...")
print("="*60)

multi_region_pals = []

for file_path in all_pal_files:
    filename = os.path.basename(file_path)
    overlaps, bounds = check_pal_overlaps(file_path, region_bounds)
    
    if len(overlaps) > 1:
        print(f"🔄 MULTI-REGION PAL: {filename}")
        print(f"   Overlaps: {overlaps}")
        if bounds:
            lat_min, lat_max, lon_min, lon_max = bounds
            print(f"   Bounds: Lat {lat_min:.2f}-{lat_max:.2f}, Lon {lon_min:.2f}-{lon_max:.2f}")
        print()
        multi_region_pals.append((filename, overlaps, bounds))

if not multi_region_pals:
    print("✅ No PALs found that overlap multiple regions")
else:
    print(f"Found {len(multi_region_pals)} PAL(s) that overlap multiple regions")
