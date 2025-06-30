#!/usr/bin/env python3

# Longitude investigation script
import os
import xarray as xr
import numpy as np

# Path to PAL data
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

print("🔍 INVESTIGATING LONGITUDE VALUES IN PAL FILES")
print("="*60)

# Check specific PAL files that we know exist
sample_files = ['PAL_precip_wind_Argo_float_6862_v1.nc', 
                'PAL_precip_wind_TPOS_Argo_float_19412_v1.nc',
                'PAL_precip_wind_Argo_float_6874_v1.nc',
                'PAL_precip_wind_SPURS2_Argo_float_12360_v1.nc']

longitude_formats = {"0_to_360": [], "-180_to_180": [], "mixed": [], "errors": []}
boundary_crossers = []

for filename in sample_files:
    file_path = os.path.join(path_to_pal_data, filename)
    
    if os.path.exists(file_path):
        try:
            ds = xr.open_dataset(file_path)
            
            if 'lon' in ds.variables:
                lon = ds['lon'].values
                lat = ds['lat'].values
                
                # Filter out NaN values
                valid_mask = ~(np.isnan(lat) | np.isnan(lon))
                if valid_mask.any():
                    valid_lon = lon[valid_mask]
                    valid_lat = lat[valid_mask]
                    
                    lon_min, lon_max = float(np.min(valid_lon)), float(np.max(valid_lon))
                    lat_min, lat_max = float(np.min(valid_lat)), float(np.max(valid_lat))
                    
                    print(f"📍 {filename}")
                    print(f"   Lat range: {lat_min:.2f} to {lat_max:.2f}")
                    print(f"   Lon range: {lon_min:.2f} to {lon_max:.2f}")
                    
                    # Classify longitude format
                    if lon_min >= 0 and lon_max <= 360:
                        if lon_max > 180:  # Likely 0-360 format
                            longitude_formats["0_to_360"].append(filename)
                            print(f"   🌐 Format: 0-360 (NEEDS CONVERSION to -180/180)")
                        else:
                            longitude_formats["-180_to_180"].append(filename)
                            print(f"   ➕ Format: 0-180 (positive only)")
                    elif lon_min >= -180 and lon_max <= 180:
                        longitude_formats["-180_to_180"].append(filename)
                        print(f"   ✅ Format: -180 to 180 (correct)")
                    else:
                        longitude_formats["mixed"].append(filename)
                        print(f"   ⚠️  Format: Mixed/Unusual")
                    
                    # Check longitude span
                    lon_span = lon_max - lon_min
                    if lon_span > 100:
                        print(f"   🌍 WIDE SPAN: {lon_span:.1f}° (BOUNDARY CROSSER)")
                        boundary_crossers.append(filename)
                    
                    # Check if crosses the 180/-180 dateline  
                    if (lon_min < -170 and lon_max > 170) or (lon_min < 10 and lon_max > 350):
                        print(f"   🔄 DATELINE CROSSER")
                        boundary_crossers.append(filename)
                        
                    print()
                else:
                    print(f"❌ {filename}: No valid lon/lat values (all NaN)")
                    longitude_formats["errors"].append(filename)
            else:
                print(f"❌ {filename}: No 'lon' variable found")
                longitude_formats["errors"].append(filename)
                
            ds.close()
            
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
            longitude_formats["errors"].append(filename)
    else:
        print(f"❌ File not found: {filename}")

print("📊 LONGITUDE FORMAT SUMMARY:")
print("-" * 40)
for format_type, files in longitude_formats.items():
    if files:
        print(f"{format_type}: {len(files)} files")
        for f in files:
            print(f"  - {f}")

if boundary_crossers:
    print(f"\n🌍 BOUNDARY CROSSING PALs FOUND:")
    print("-" * 40)
    for f in set(boundary_crossers):  # Remove duplicates
        print(f"📍 {f}")
else:
    print("\n✅ No obvious boundary crossers in sample files")

print("\n💡 LONGITUDE CONVERSION ISSUE:")
print("If some PAL files use 0-360° longitude format and others use -180-180°,")
print("this could cause classification problems without proper conversion.")
print("Our util_functions.py should handle this conversion consistently.")
