'''
Functions, packages and floating variables used for IEPPO study.
'''
#%%
import warnings
warnings.filterwarnings("ignore")

import gc
import os

from datetime import date
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import xarray as xr

from pyproj import CRS
from rasterio.warp import Resampling

from multiprocessing import Pool

#%% DEFINE GLOBAL VARIABLES

cc = CRS.from_authority(code=4326, auth_name='EPSG')

cde_run_dte = str(date.today().strftime('%Y%m%d'))

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
    "TSEP": {"lon_min": -160, "lon_max": -70,  "lat_min": -25, "lat_max": 0},  # Tropical Southeastern Pacific (southern hemisphere only)
    "STNA": {"lon_min": -70,  "lon_max": -10,  "lat_min": 15, "lat_max": 45},  # Subtropical North Atlantic
    "TNIO": {"lon_min": 60,   "lon_max": 100,  "lat_min": -5, "lat_max": 20},  # Tropical North Indian Ocean
    "TNWP": {"lon_min": 120,  "lon_max": 180,  "lat_min": -5, "lat_max": 30},  # Tropical Northwestern Pacific
}
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Buoy_region_bounds = {
    "ENP": {"lon_min": -180, "lon_max": -60, "lat_min": -30, "lat_max": 15},  # Eastern Pacific
    "WNP": {"lon_min": 120, "lon_max": 180, "lat_min": -15, "lat_max": 15},    # Western Pacific
    "IND": {"lon_min": 40, "lon_max": 110, "lat_min": -15, "lat_max": 30},     # Indian Ocean
    "ATL": {"lon_min": -70, "lon_max": 20, "lat_min": -30, "lat_max": 30},     # Atlantic Ocean
}


#%% DEFINE CUSTOM FUNCTIONS
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
        mer2_xr = mer2_xr['PRECTOTCORR'] * 3600
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
        precip_aray = imerg_precip_data.precipitationCal.data    
        imerg_time = imerg_precip_data.attrs['BeginDate']
        
    elif version == 'v07':
        precip_aray = imerg_precip_data.precipitation.data    
        imerg_time = imerg_precip_data['time'].values[0] 

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
