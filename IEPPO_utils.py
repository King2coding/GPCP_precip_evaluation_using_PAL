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
import seaborn as sns

import HydroErr as he

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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Define global constants
region_colors = {
    "TNEP": "#3366ff",      # blue
    "TSEP": "#66ccff",      # light blue
    "TNWP": "#33cc33",      # green
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
    "TSEP": "s",
    "TNWP": "^",
    "TNIO": "D",
    "STNA": "P",
}

Buoy_region_markers = {
    "ENP": "*",
    "WNP": "o",
    "IND": "s",
    "ATL": "^",
}

#%% DEFINE CUSTOM FUNCTIONS
def p_corr(obs,model):
    return he.pearson_r(model,obs)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# relative bias
def relative_bias(obs,model):
    mu_residuals = np.nanmean(model - obs)
    mu_obs = np.nanmean(obs)

    return mu_residuals/mu_obs
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

    return rb, rmse, cc

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

        rb, rmse, cc = calculate_metrics(df_all_regs,truth_col, prod)

        for region in df_all_regs['region'].unique():

            dff = df_all_regs[df_all_regs['region'] == region]

            marker = region_markers.get(region, "o")

            ax.scatter(
                dff[truth_col].values,
                dff[prod].values,
                c="k",
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

        ax.set_xlabel(f'{truth_col} [mm day$^{-1}$]', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{label} Estimates [mm day$^{-1}$]', fontsize=16, fontweight='bold')
        ax.set_title(f'{label} vs {truth_col}', fontsize=18, fontweight='bold')

        # Stats annotation
        ax.text(
            0.05, 0.97,
            f'RB: {rb:.2f}%\nRMSE: {rmse:.2f} mm/day\nCC: {cc:.2f}',
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
            color='k',
            linestyle='None',
            markersize=10,
            markeredgewidth=1,
            label=r
        )
        for r in region_markers
    ]

    # Make room at the bottom for the legend
    fig.subplots_adjust(bottom=0.15)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=14,
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
    metrics=("POD", "FAR", "Bias", "HSS"),
    figsize=(16, 14),
    bar_width=0.18,
):
    """
    4x1 bar plot of categorical metrics by region, colored by product.
    """

    regions = list(metrics_dict.keys())
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

        ax.set_ylabel(metric, fontsize=18, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Metric-specific limits
        if metric in ["POD", "FAR"]:
            ax.set_ylim(0, 1)

        ax.tick_params(axis="both", labelsize=15)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        

    # X-axis
    axes[-1].set_xticks(x + bar_width * (n_products - 1) / 2)
    axes[-1].set_xticklabels(regions, fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Region", fontsize=15, fontweight="bold")

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

            ax.set_title(region, fontsize=14, fontweight="bold")
            ax.set_xlabel("Month", fontsize=12, fontweight="bold")
            ax.set_ylabel("Monthly Mean Rainfall [mm day$^{-1}$]", fontsize=12, fontweight="bold")

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
            bbox_to_anchor=(0.5, 0.98)
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for legend
        plt.show()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_monthly_climatology_anoms_2x2(
    monthly_anom_by_region,
    regions,
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
            ax.set_title(region, fontsize=14, fontweight="bold")
            ax.set_xlabel("Month", fontsize=12, fontweight="bold")
            ax.set_ylabel("Δ Monthly Mean \n (Product − Buoy) [mm day$^{-1}$]", fontsize=12, fontweight="bold")

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