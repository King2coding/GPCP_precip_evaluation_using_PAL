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
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, FuncFormatter

import seaborn as sns
from scipy.stats import linregress

import HydroErr as he

import xarray as xr

from pyproj import CRS
from rasterio.warp import Resampling

from multiprocessing import Pool

#%% DEFINE GLOBAL VARIABLES

products = [
    "rain_rate",     # Buoy
    "GPCP v3.2",
    "GPCP v3.3",
    "ERA5",
    "IMERG v07",
    "MERRA2",
]

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
    "GPCP v3.2": "#4c4c4c",   # dark gray
    "GPCP v3.3": "#1f77b4",   # blue
    "ERA5": "#d62728",       # red
    "IMERG v07": "#2ca02c",  # green
    "MERRA2": "#ff7f0e",     # orange
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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# QC for OceanRAIN data
def oceanrain_step0_qc(
    df: pd.DataFrame,
    *,
    keep_cols=None,
    drop_harbor_inop=True,
    drop_spurious=True,
    min_flag2=None,          # e.g., 14 to keep >=0.1 mm/h
    prob_thr=None,           # e.g., 0.9 for high-confidence phase (optional)
    wind_max=None            # e.g., 15.0 if you want a wind limit (optional)
) -> pd.DataFrame:
    """
    Minute-level OceanRAIN QC for the 'detailed' dataframe:
    time_utc, lat, lon, rate_dsd_mmph, rate_gag_mmph, precip_flag, precip_flag2,
    mixed_prob, rain_prob, snow_prob, wind_speed, ship

    Returns a filtered dataframe (still minute-resolution) ready for pixel-mapping.
    """

    df = df.copy()

    # ----------------------------
    # 1) Ensure datetime + basic columns
    # ----------------------------
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon"])

    # ----------------------------
    # 2) Replace common fill values with NaN
    # (your merged df may already have NaNs, but be safe)
    # ----------------------------
    fill_vals = [-99.99, -99.9, -999.99, -999.9, -9999, -99999]
    for c in ["rate_dsd_mmph", "rate_gag_mmph", "mixed_prob", "rain_prob", "snow_prob", "wind_speed"]:
        if c in df.columns:
            df[c] = df[c].replace(fill_vals, np.nan)

    # Flags can also carry fill values
    for c in ["precip_flag", "precip_flag2"]:
        if c in df.columns:
            df[c] = df[c].replace([9, 99, -99, -999], np.nan)

    # Cast flags to Int64 (nullable ints)
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
        m &= (df["precip_flag2"] != 11)

    # optional minimum intensity gate using precip_flag2
    # 13: 0.01–0.09, 14: 0.1–0.99, 15+: >=1 mm/h
    if (min_flag2 is not None) and ("precip_flag2" in df.columns):
        m &= (df["precip_flag2"] >= int(min_flag2))

    # optional wind filter
    if (wind_max is not None) and ("wind_speed" in df.columns):
        m &= (df["wind_speed"].isna() | (df["wind_speed"] <= float(wind_max)))

    df = df.loc[m].copy()

    # ----------------------------
    # 4) Rate sanity masks (no flags version fallback)
    # ----------------------------
    # DSD: allow big values but remove absolute junk
    if "rate_dsd_mmph" in df.columns:
        df.loc[(df["rate_dsd_mmph"] < 0) | (df["rate_dsd_mmph"] > 400), "rate_dsd_mmph"] = np.nan

    # Gauge: treat as diagnostic; remove placeholders/spikes
    if "rate_gag_mmph" in df.columns:
        df.loc[(df["rate_gag_mmph"] < 0), "rate_gag_mmph"] = np.nan
        df.loc[np.isclose(df["rate_gag_mmph"], 99.99, atol=1e-6), "rate_gag_mmph"] = np.nan
        # optional physical cap for gauge artefacts
        df.loc[df["rate_gag_mmph"] >= 50, "rate_gag_mmph"] = np.nan

    # ----------------------------
    # 5) Optional: high-confidence phase subsets via probabilities
    # ----------------------------
    if prob_thr is not None:
        thr = float(prob_thr)

        # Only enforce if the probability columns exist
        if "rain_prob" in df.columns and "precip_flag" in df.columns:
            # for rain minutes, require rain_prob >= thr
            df = df[~((df["precip_flag"] == 0) & (df["rain_prob"].notna()) & (df["rain_prob"] < thr))]

        if "snow_prob" in df.columns and "precip_flag" in df.columns:
            df = df[~((df["precip_flag"] == 1) & (df["snow_prob"].notna()) & (df["snow_prob"] < thr))]

        if "mixed_prob" in df.columns and "precip_flag" in df.columns:
            df = df[~((df["precip_flag"] == 2) & (df["mixed_prob"].notna()) & (df["mixed_prob"] < thr))]

        df = df.copy()

    # ----------------------------
    # 6) Keep only the columns you need (memory efficiency)
    # ----------------------------
    default_cols = [
        "time_utc", "lat", "lon", "ship",
        "rate_dsd_mmph", "rate_gag_mmph",
        "precip_flag", "precip_flag2",
        "rain_prob", "snow_prob", "mixed_prob",
        "wind_speed"
    ]
    if keep_cols is None:
        keep_cols = [c for c in default_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)


#------------------------------------------------------------------------
def map_to_gpcp_idx(arr1d, values):
    a = np.asarray(arr1d)
    v = np.asarray(values)
    if a.ndim != 1: a = a.ravel()
    if v.ndim != 1: v = v.ravel()
    v_finite = np.where(np.isfinite(v), v, np.nan)
    asc = bool(a[0] <= a[-1])
    if not asc: a_work = a[::-1]
    else: a_work = a
    idx = np.searchsorted(a_work, v_finite)
    idx0 = np.clip(idx - 1, 0, a_work.size - 1)
    idx1 = np.clip(idx, 0, a_work.size - 1)
    choose_left = (np.abs(v_finite - a_work[idx0]) <= np.abs(v_finite - a_work[idx1]))
    out_rev = np.where(choose_left, idx0, idx1)
    if not asc: out = (a_work.size - 1) - out_rev
    else: out = out_rev
    if np.issubdtype(v.dtype, np.floating):
        nanmask = ~np.isfinite(v)
        if nanmask.any():
            out = out.astype('int64')
            out[nanmask] = 0
    return out
#%% THE PLOT FUNCTIONS
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
def plot_categorical_metrics_by_region(
    metrics_dict,
    products,
    product_colors,
    region_labels=None,
    metrics=("POD", "FAR", "Bias", "HSS"),
    figsize=(16, 14),
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
    axes[-1].set_xticklabels([PAL_REGION_NAMES[k] for k in regions], fontsize=12, fontweight="bold")
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
    figsize=(12, 9),   # close to the 2x2 style you showed
    lw=3.5,
    ncol_legend=3
):
    """
    Plot monthly climatology in 2x2 subplots (no shared axes).
    Makes multiple figures if len(regions) > 4.
    X-axis uses month integers (1–12).
    """

    months = np.arange(1, 13)

    # chunk regions into groups of 4
    for k in range(0, len(regions), 4):
        regs = regions[k:k+4]

        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(regs):
                ax.axis("off")
                continue

            region = regs[i]
            clim = monthly_clim_by_region[region]

            # ---- Buoy (reference) ----
            ax.plot(
                clim["month"], clim["rain_rate"],
                lw=lw, color="b", label="Buoy"
            )

            # ---- Products ----
            for prod in products[1:]:
                ax.plot(
                    clim["month"], clim[prod],
                    lw=lw, color=product_colors[prod], label=prod
                )

            ax.set_title(region_labels[region], fontsize=14, fontweight="bold")
            ax.set_xlabel("Month", fontsize=12, fontweight="bold")
            ax.set_ylabel("Rainfall [mm day$^{-1}$]", fontsize=12, fontweight="bold")

            ax.set_xticks(months)
            ax.set_xlim(1, 12)

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", labelsize=11)

        # one legend for the whole figure (clean)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=ncol_legend,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, 0.05)#(0.5, 0.98)
        )

        fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.92])  # leave room for legend, titlerect=[0, 0, 1, 0.94])  # leave room for legend
        plt.show()

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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    legend_ncol=4,
    legend_fontsize=10,
    show_product_slopes=True,   # NEW
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
        buoy_out = None
        if add_trend_band and ref in dfr.columns:
            y_buoy = dfr[ref].values.astype(float)
            m = np.isfinite(y_buoy) & np.isfinite(years.astype(float))
            years_obs = years[m]
            buoy_obs  = y_buoy[m]

            segs_obs = contiguous_segments(years_obs, max_gap=max_gap_years)
            if len(segs_obs) > 0:
                main_seg_obs = max(segs_obs, key=len)
                seg_mask = np.isin(years_obs, main_seg_obs)
                xs = years_obs[seg_mask].astype(float)
                ys = buoy_obs[seg_mask]

                buoy_out = bootstrap_trend_band(xs, ys, n_boot=n_boot, ci=ci, seed=42)
                if buoy_out is not None:
                    line_label = "Buoy trend" if k == 0 else None
                    band_label = f"Buoy trend ({ci}% bootstrap CI)" if k == 0 else None
                    ax.plot(buoy_out["xgrid"], buoy_out["mid"], "k--", lw=2.0, label=line_label)
                    ax.fill_between(buoy_out["xgrid"], buoy_out["lo"], buoy_out["hi"],
                                    alpha=0.15, label=band_label)

        # ---- slopes text stack (Buoy + products) ----
        if show_product_slopes:
            slope_dict = {}

            # buoy slope: use bootstrap mean if available; otherwise OLS
            if buoy_out is not None:
                slope_dict["Buoy"] = buoy_out["slope_mean"]
            else:
                slope_dict["Buoy"] = slope_only_on_common_years(dfr, "year", ref)

            # product slopes: compute on the SAME years where buoy is finite+coverage
            # (within the already-trimmed dfr)
            ok_mask_local = np.isfinite(dfr[ref].values.astype(float))
            ncol_local = f"n_{ref}"
            if ncol_local in dfr.columns:
                ok_mask_local &= (dfr[ncol_local].values.astype(float) >= float(min_days_ref))

            for p in products:
                if p == ref:
                    continue
                if p in dfr.columns:
                    slope_dict[p] = slope_only_on_common_years(
                        dfr, "year", p, ok_mask=ok_mask_local
                    )

            color_dict = {"Buoy": "b"}
            color_dict.update({p: product_colors[p] for p in products if p != ref})

            add_slope_stack(
                ax,
                slope_dict=slope_dict,
                color_dict=color_dict,
                x=0.02, y=0.96, dy=0.055,
                fmt="{name}: {slope:+.3f}",
                units=" mm day$^{-1}$ yr$^{-1}$",
                fontsize=10,
                title="Slope (OLS):" if buoy_out is None else "Slope:"
            )

            # optional: add buoy CI line only (one extra line, not too messy)
            if buoy_out is not None:
                ax.text(
                    0.02, 0.96 - 0.055*(len(slope_dict)+0.3),
                    f"Buoy {ci}% CI: [{buoy_out['slope_lo']:+.3f}, {buoy_out['slope_hi']:+.3f}]",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color="b"
                )

        # ---- cosmetics ----
        ax.set_title(region, fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax.set_ylabel("Annual mean rainfall (mm day$^{-1}$)", fontsize=11, fontweight="bold")
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
    legend_ncol=4,
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

        # 5) buoy trend + (optional) CI (gap-aware on annual years)
        yb = ann[ref + "_rm13"].values.astype(float)
        ok = np.isfinite(yb)
        years_ok = years[ok]

        out = None
        if len(years_ok) >= 3:
            segs = contiguous_year_segments(years_ok, max_gap=1)
            main_seg = max(segs, key=len) if len(segs) else years_ok
            mask_main = np.isin(years, main_seg)
            years_main = years[mask_main]
            buoy_main  = yb[mask_main]

            out = bootstrap_trend_band_years(years_main, buoy_main, n_boot=n_boot, ci=ci, seed=42)
            if out is not None:
                ax.plot(out["xgrid"], out["mid"], "k--", lw=2.0, label="Buoy trend" if k == 0 else None)
                if show_ci_band:
                    ax.fill_between(out["xgrid"], out["lo"], out["hi"], alpha=0.15,
                                    label=f"Buoy trend ({ci}% bootstrap CI)" if k == 0 else None)

        # 6) trend stats text (colored) — per decade
        # --- trend stats text: 2 stacked groups (same x, different y bands) ---
        x_text = 0.01

        top_group = [ref, "GPCP v3.2", "GPCP v3.3"]          # keep up
        low_group = ["ERA5", "IMERG v07", "MERRA2"]          # push down

        # (optional) keep only those present in your products list
        top_group = [p for p in top_group if p in products]
        low_group = [p for p in low_group if p in products]

        # y locations in Axes fraction
        y_top0 = 0.97
        y_low0 = 0.2          # <-- move this up/down to taste
        dy_top = 0.075
        dy_low = 0.075

        def _fmt(name, s):
            return f"{name}: Trend={s['slope_dec']:+.3f}, p={s['p']:.3f}, Std={s['std']:.3f}"

        # --- top band ---
        y = y_top0
        for p in top_group:
            yvals = ann[p + "_rm13"].values.astype(float)
            s = trend_stats_yearly(years, yvals)
            if s is None:
                continue
            ax.text(
                x_text, y, _fmt("Buoy" if p == ref else p, s),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12,
                color=("b" if p == ref else product_colors[p]),
                fontweight=("bold" if p == ref else "normal"),
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
            )
            y -= dy_top

        # --- lower band ---
        y = y_low0
        for p in low_group:
            yvals = ann[p + "_rm13"].values.astype(float)
            s = trend_stats_yearly(years, yvals)
            if s is None:
                continue
            ax.text(
                x_text, y, _fmt(p, s),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12, color=product_colors[p],
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.2)
            )
            y -= dy_low

        # cosmetics
        ax.set_title(region_names.get(reg, reg), fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax.set_ylabel("Annual mean rainfall (mm day$^{-1}$)",
                      fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)
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
    return fig