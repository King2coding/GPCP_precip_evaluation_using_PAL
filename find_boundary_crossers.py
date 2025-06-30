#!/usr/bin/env python3

# Find a PAL that crosses region boundaries by checking which one was hardest to classify
import os
from util_functions import *

# Get the PAL files
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'
all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]

# Classify them
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

# Look for PALs that might span boundaries by checking their lat/lon ranges
print("Looking for PALs that might span multiple regions...")
print("="*60)

# Check TNEP and TSEP boundary (at equator)
print("PALs near TNEP/TSEP boundary (equator):")
for file_path in pals_classed_by_region.get("TNEP", []) + pals_classed_by_region.get("TSEP", []):
    filename = os.path.basename(file_path)
    try:
        ds = xr.open_dataset(file_path)
        if 'lat' in ds.variables and 'lon' in ds.variables:
            lat = ds['lat'].values
            lon = ds['lon'].values
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(lat) | np.isnan(lon))
            if valid_mask.any():
                lat_clean = lat[valid_mask]
                lon_clean = lon[valid_mask]
                
                lat_min, lat_max = float(np.min(lat_clean)), float(np.max(lat_clean))
                lon_min, lon_max = float(np.min(lon_clean)), float(np.max(lon_clean))
                
                # Check if it crosses the equator (TNEP/TSEP boundary)
                if lat_min < 0 and lat_max > 0:
                    print(f"🌊 BOUNDARY CROSSER: {filename}")
                    print(f"   Lat: {lat_min:.2f} to {lat_max:.2f} (crosses equator)")
                    print(f"   Lon: {lon_min:.2f} to {lon_max:.2f}")
                    
                    # Check which region it was assigned to
                    for region, files in pals_classed_by_region.items():
                        if file_path in files:
                            print(f"   Assigned to: {region}")
                            break
                    print()
                    
        ds.close()
    except Exception as e:
        print(f"Error checking {filename}: {e}")

# Also check for very wide longitude spans that might cross multiple regions
print("\nPALs with very wide longitude spans:")
for region, files in pals_classed_by_region.items():
    if region == "Unclassified":
        continue
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            ds = xr.open_dataset(file_path)
            if 'lat' in ds.variables and 'lon' in ds.variables:
                lat = ds['lat'].values
                lon = ds['lon'].values
                
                # Filter out NaN values
                valid_mask = ~(np.isnan(lat) | np.isnan(lon))
                if valid_mask.any():
                    lat_clean = lat[valid_mask]
                    lon_clean = lon[valid_mask]
                    
                    lat_min, lat_max = float(np.min(lat_clean)), float(np.max(lat_clean))
                    lon_min, lon_max = float(np.min(lon_clean)), float(np.max(lon_clean))
                    
                    lon_span = lon_max - lon_min
                    
                    # Check for very wide spans (>50 degrees longitude)
                    if lon_span > 50:
                        print(f"🌍 WIDE SPAN PAL: {filename}")
                        print(f"   Lat: {lat_min:.2f} to {lat_max:.2f}")
                        print(f"   Lon: {lon_min:.2f} to {lon_max:.2f} (span: {lon_span:.1f}°)")
                        print(f"   Assigned to: {region}")
                        print()
                        
            ds.close()
        except Exception as e:
            continue

print("Done checking for boundary-crossing PALs")
