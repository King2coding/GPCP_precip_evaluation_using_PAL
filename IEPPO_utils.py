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
def process_era5_file(file_info):
    idx, file_path = file_info
    if idx % 5 == 0:
        print(f"Processing ERA5 file {idx+1}")
    era5_xr = xr.open_dataset(file_path, engine='netcdf4')
    era5_xr = ds_swaplon(era5_xr)
    # data units are in m per day, convert to mm/day using 1000 factor
    era5_xr['tp'] = era5_xr['tp'] * 1000  # mm/h
    era5_xr['tp'] = era5_xr['tp'] * 24  # mm/day
    # resample to 0.5 degree resolution
    cc = CRS.from_authority(code=4326, auth_name='EPSG')
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
        # resample to 0.5 degree resolution
        cc = CRS.from_authority(code=4326, auth_name='EPSG')
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

