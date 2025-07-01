#!/usr/bin/env python3
"""
Simple test for PAL-GPCP data matching
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import sys
import importlib

# Import util functions
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])
from util_functions import *

# Define paths
path_to_pal_data = r'/ra1/pubdat/GPCP_eval_with_PAL/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'
path_to_gpcp_v1pt3 = r'/ra1/pubdat/GPCP_eval_with_PAL/data/GPCP/GPCP_v1_pnt_3_2010_2020'

print("=== TESTING PAL-GPCP MATCHING ===")

# Get file lists
all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])
all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

print(f"Found {len(all_pal_files)} PAL files")
print(f"Found {len(all_gpcp_v1pt3_files)} GPCP files")

# Classify PALs
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

# Test with just ONE PAL file from ETNP region
if 'ETNP' in pals_classed_by_region and len(pals_classed_by_region['ETNP']) > 0:
    test_pal_file = pals_classed_by_region['ETNP'][0]
    pal_id = os.path.basename(test_pal_file).split('.')[0]
    
    print(f"\n=== TESTING WITH PAL: {pal_id} ===")
    
    # Load PAL data
    try:
        pal_ds = xr.open_dataset(test_pal_file)
        print(f"PAL variables: {list(pal_ds.variables.keys())}")
        
        # Get PAL time range
        pal_time = pd.to_datetime(pal_ds['time'].values)
        pal_start_date = pal_time.min().date()
        pal_end_date = pal_time.max().date()
        print(f"PAL operational period: {pal_start_date} to {pal_end_date}")
        
        # Get first few data points
        pal_lat = pal_ds['lat'].values[:100]  # First 100 points
        pal_lon = pal_ds['lon'].values[:100]
        pal_precip = pal_ds['rain_rate'].values[:100]
        
        # Normalize longitude
        pal_lon = (pal_lon + 360) % 360
        pal_lon[pal_lon > 180] -= 360
        
        print(f"PAL lat range: {pal_lat.min():.2f} to {pal_lat.max():.2f}")
        print(f"PAL lon range: {pal_lon.min():.2f} to {pal_lon.max():.2f}")
        print(f"PAL precip range: {pal_precip.min():.2f} to {pal_precip.max():.2f}")
        
        # Create simplified daily PAL data
        pal_df = pd.DataFrame({
            'time': pal_time[:100],
            'lat': pal_lat,
            'lon': pal_lon,
            'precip': pal_precip
        })
        pal_df['date'] = pal_df['time'].dt.date
        
        # Get daily means
        daily_pal = pal_df.groupby('date').agg({
            'lat': 'mean',
            'lon': 'mean',
            'precip': 'mean'
        }).reset_index()
        
        print(f"Daily PAL observations: {len(daily_pal)}")
        print("First few daily observations:")
        print(daily_pal.head())
        
        pal_ds.close()
        
        # Now test with ONE GPCP file
        print(f"\n=== TESTING GPCP FILE MATCHING ===")
        
        # Find GPCP files that match PAL dates
        matching_gpcp_files = []
        for gpcp_file in all_gpcp_v1pt3_files[:50]:  # Check first 50 files
            filename = os.path.basename(gpcp_file)
            
            # Parse GPCP filename: gpcp_v01r03_daily_d20100101_c20170814.nc
            try:
                if 'daily_d' in filename:
                    date_part = filename.split('daily_d')[1].split('_')[0]  # Extract YYYYMMDD
                    if len(date_part) == 8 and date_part.isdigit():
                        file_date = pd.to_datetime(date_part, format='%Y%m%d').date()
                        
                        # Check if this date overlaps with PAL period
                        if pal_start_date <= file_date <= pal_end_date:
                            matching_gpcp_files.append((gpcp_file, file_date))
            except:
                continue
        
        print(f"Found {len(matching_gpcp_files)} matching GPCP files")
        
        if len(matching_gpcp_files) > 0:
            # Test with first matching GPCP file
            test_gpcp_file, test_date = matching_gpcp_files[0]
            print(f"Testing with GPCP file for date: {test_date}")
            print(f"File: {os.path.basename(test_gpcp_file)}")
            
            try:
                gpcp_ds = xr.open_dataset(test_gpcp_file)
                gpcp_ds = ds_swaplon(gpcp_ds)  # Normalize longitude
                
                print(f"GPCP variables: {list(gpcp_ds.variables.keys())}")
                
                # Get GPCP coordinates
                if 'latitude' in gpcp_ds:
                    gpcp_lat = gpcp_ds['latitude'].values
                    gpcp_lon = gpcp_ds['longitude'].values
                else:
                    gpcp_lat = gpcp_ds['lat'].values
                    gpcp_lon = gpcp_ds['lon'].values
                
                print(f"GPCP lat range: {gpcp_lat.min():.2f} to {gpcp_lat.max():.2f}")
                print(f"GPCP lon range: {gpcp_lon.min():.2f} to {gpcp_lon.max():.2f}")
                
                # Find precipitation variable
                precip_var = None
                for var in ['precip', 'precipitation', 'PRECIP']:
                    if var in gpcp_ds.variables:
                        precip_var = var
                        break
                
                if precip_var:
                    print(f"Found precipitation variable: {precip_var}")
                    
                    # Test matching with one daily PAL observation
                    if test_date in daily_pal['date'].values:
                        pal_row = daily_pal[daily_pal['date'] == test_date].iloc[0]
                        target_lat = pal_row['lat']
                        target_lon = pal_row['lon']
                        
                        # Find nearest GPCP grid point
                        lat_idx = np.argmin(np.abs(gpcp_lat - target_lat))
                        lon_idx = np.argmin(np.abs(gpcp_lon - target_lon))
                        
                        # Extract GPCP value
                        gpcp_precip = gpcp_ds[precip_var].values[0, lat_idx, lon_idx]  # Assuming first time index
                        
                        print(f"\n=== SUCCESSFUL MATCH ===")
                        print(f"Date: {test_date}")
                        print(f"PAL location: {target_lat:.2f}°N, {target_lon:.2f}°E")
                        print(f"GPCP grid: {gpcp_lat[lat_idx]:.2f}°N, {gpcp_lon[lon_idx]:.2f}°E")
                        print(f"PAL precip: {pal_row['precip']:.2f} mm/day")
                        print(f"GPCP precip: {gpcp_precip:.2f} mm/day")
                        
                    else:
                        print(f"Date {test_date} not found in daily PAL data")
                else:
                    print("No precipitation variable found in GPCP file")
                
                gpcp_ds.close()
                
            except Exception as e:
                print(f"Error processing GPCP file: {e}")
        
        else:
            print("No matching GPCP files found for PAL period")
            
    except Exception as e:
        print(f"Error processing PAL file: {e}")

else:
    print("No ETNP PAL files found for testing")

print("\n=== TEST COMPLETED ===")
