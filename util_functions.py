#%% IMPORT LIBRARIES
import warnings

import matplotlib as mpl
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
from evaluation_fucntions_algorithms import *

from rasterio.transform import from_origin
from rasterio.transform import rowcol

# Optional GDAL imports - code will work without these
try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    print("Warning: GDAL not available. Some functions may be limited.")
    HAS_GDAL = False

import subprocess


from scipy import stats
from joblib import Parallel, delayed
import dask
#%% GLOBAL VARIABLES

def format_lon(x, pos=None):
    if x == 0:
        return "0°"
    elif x < 0:
        return f"{abs(int(x))}°W"
    else:
        return f"{int(x)}°E"
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def format_lat(y, pos=None):
    if y == 0:
        return "0°"
    elif y < 0:
        return f"{abs(int(y))}°S"
    else:
        return f"{int(y)}°N"
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def run_gdalinfo(file_path):
    """Run gdalinfo command using subprocess"""
    if not HAS_GDAL:
        print("Warning: GDAL not available. Cannot run gdalinfo.")
        return
    
    # Run the gdalinfo command using subprocess
    result = subprocess.run(['gdalinfo', file_path], capture_output=True, text=True)
    
    # Print the output
    print(result.stdout)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DEFINE REGIONS AND THEIR BOUNDARIES (based on Figure 1 and PAL data coverage)
region_bounds = {
    "ETNP": {"lon_min": -170, "lon_max": -120, "lat_min": 30, "lat_max": 60},  # Extratropical North Pacific 
    "TNEP": {"lon_min": -180, "lon_max": -80, "lat_min": 0, "lat_max": 30}, # Tropical Northeastern Pacific (northern hemisphere only, extended to capture SPURS2 and Caribbean PALs)
    "TSEP": {"lon_min": -160, "lon_max": -70,  "lat_min": -25, "lat_max": 0},  # Tropical Southeastern Pacific (southern hemisphere only)
    "STNA": {"lon_min": -70,  "lon_max": -10,  "lat_min": 15, "lat_max": 45},  # Subtropical North Atlantic
    "TNIO": {"lon_min": 60,   "lon_max": 100,  "lat_min": -5, "lat_max": 20},  # Tropical North Indian Ocean
    "TNWP": {"lon_min": 120,  "lon_max": 180,  "lat_min": -5, "lat_max": 30},  # Tropical Northwestern Pacific
}

buoy_region_bounds = {
    "ENP": {"lon_min": -180, "lon_max": -60, "lat_min": -30, "lat_max": 15},  # Eastern Pacific
    "WNP": {"lon_min": 120, "lon_max": 180, "lat_min": -15, "lat_max": 15},    # Western Pacific
    "IND": {"lon_min": 40, "lon_max": 110, "lat_min": -15, "lat_max": 30},     # Indian Ocean
    "ATL": {"lon_min": -70, "lon_max": 20, "lat_min": -30, "lat_max": 30},     # Atlantic Ocean
}


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# DEFINE COLORS FOR EACH REGION
region_colors = {
    "TNEP": "#3366ff",      # blue
    "TSEP": "#66ccff",      # light blue
    "TNWP": "#33cc33",      # green
    "ETNP": "#888888",      # gray
    "TNIO": "#ffcc33",      # yellow/orange
    "STNA": "#b35959",      # brown/red
    "Unclassified": "black",
}

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DEFINE COLORS FOR EACH BUOY REGION
buoy_region_colors = {
    "ENP": "#3366ff",      # blue
    "WNP": "#66ccff",      # light blue
    "IND": "#33cc33",      # green
    "ATL": "#b35959",      # brown/red
    
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
    - tuple: 'row', 'col'
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
def grab_PAL_rain_and_wind_df(pal_xr_ds):
    """
    Grab rain and wind data from a PAL xarray dataset.
    Parameters:
    - pal_xr_ds: xarray dataset containing PAL data.
    Returns:
    - df_rain: DataFrame containing rain data.
    - df_wind: DataFrame containing wind data.
    """

    # Process PAL data as needed
    # df_rain = pd.DataFrame({
    #     'time': pd.to_datetime(pal_xr_ds['time'].values),
    #     'lat': pal_xr_ds['lat'].values,
    #     'lon': pal_xr_ds['lon'].values,
    #     'rain_rate': pal_xr_ds['rain_rate'].values,
    # })

    # df_rain['date'] = df_rain['time'].dt.date  # Extract date from time               

    # df_rain = df_rain.dropna(axis=0, how='any')  # Drop rows with any NaN values           

    # # Normalize longitude to [-180, 180]
    # df_rain['lon'] = (df_rain['lon'] + 360) % 360
    # df_rain['lon'][df_rain['lon'] > 180] -= 360

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    df = pd.DataFrame({
        'time': pd.to_datetime(pal_xr_ds['time'].values),
        'lat': pal_xr_ds['lat'].values,
        'lon': pal_xr_ds['lon'].values,
        'rain_rate': pal_xr_ds['rain_rate'].values,
        'wind_speed': pal_xr_ds['wind_speed'].values,                
    })

    # drop rows where wind speed is >= 15 m/s
    df = df.drop(df[df['wind_speed'] >= 15].index) 

    df['date'] = df['time'].dt.date  # Extract date from time

    df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values

    # Normalize longitude to [-180, 180]
    df['lon'] = (df['lon'] + 360) % 360
    df['lon'][df['lon'] > 180] -= 360

    return  df   # df_rain,
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
    if gpcp_version == 'GPCP_v3pt2':
        gpcp_plp = gpcp_ds_xr['probability_liquid_phase'].interp(
            time=("points", pal_dates_rain), lat=("points", pal_lats_rain), 
            lon=("points", pal_lons_rain), method="nearest")
        
        # Set places where the values are less than 0 to NaN
        gpcp_plp = gpcp_plp.where(gpcp_plp >= 0, np.nan)
        
        # Store matched values in the DataFrame
        df[f'PLP_{gpcp_version}'] = gpcp_plp
        

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # NOW DO THE SAME FOR WIND SPEED DATA
    # pal_dates_wind = pd.to_datetime(pal_df_wind['date'])
    # pal_lats_wind = pal_df_wind['lat'].values
    # pal_lons_wind = pal_df_wind['lon'].values

    # gpcp_pr_wind_speed = gpcp_ds_xr['precip'].interp(
    #     time=("points", pal_dates_wind), lat=("points", pal_lats_wind), 
    #     lon=("points", pal_lons_wind), method="nearest"
    # )
    # # Set places where the values are less than 0 to NaN
    # gpcp_pr_wind_speed = gpcp_pr_wind_speed.where(gpcp_pr_wind_speed >= 0, np.nan)
    # # Store matched values in the DataFrame
    # pal_df_wind[gpcp_version] = gpcp_pr_wind_speed

    # # do same for probability of liquid phase if it exsists in dataset
    # if gpcp_version == 'GPCP_v3pt2':
    #     gpcp_plp = gpcp_ds_xr['probability_liquid_phase'].interp(
    #         time=("points", pal_dates_wind), lat=("points", pal_lats_wind), 
    #         lon=("points", pal_lons_wind), method="nearest")
    #     # Set places where the values are less than 0 to NaN
    #     gpcp_plp = gpcp_plp.where(gpcp_plp >= 0, np.nan)
    #     # Store matched values in the DataFrame
    #     pal_df_wind[f'PLP_{gpcp_version}'] = gpcp_plp

    return df#pal_df_rain, pal_df_wind

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def process_imerg_with_PAL_rain(df, imerg_ds_xr, chunk_size=10000, n_jobs=20):
    """
    Process IMERG data with PAL rain data in chunks using Dask for lazy evaluation and parallel processing.
    Parameters:
    - df: DataFrame containing PAL rain data.
    - imerg_ds_xr: xarray dataset containing IMERG data.
    - chunk_size: Number of rows to process in each chunk.
    - n_jobs: Number of parallel jobs to run (default is 20 cores).
    Returns:
    - df: DataFrame with IMERG rain data added.
    """

    # Extract relevant columns from the DataFrame
    pal_dates_rain = pd.to_datetime(df['date'])
    pal_lats_rain = df['lat'].values
    pal_lons_rain = df['lon'].values

    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'latitude' in imerg_ds_xr.dims or 'longitude' in imerg_ds_xr.dims:
        imerg_ds_xr = imerg_ds_xr.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Subset IMERG data to PAL's date range
    min_date, max_date = pal_dates_rain.min(), pal_dates_rain.max()
    # Use Dask to subset the dataset lazily
    imerg_ds_xr = imerg_ds_xr.sel(time=slice(min_date, max_date)).chunk({'time': 100})

    # Removed process_chunk as it is redundant

    # Split the data into chunks and process in parallel
    chunks = [(start_idx, min(start_idx + chunk_size, len(df))) for start_idx in range(0, len(df), chunk_size)]
    # Use Dask for lazy evaluation and parallel processing
    import dask.array as da

    def process_chunk_dask(start_idx, end_idx):
        chunk_dates = pal_dates_rain[start_idx:end_idx]
        chunk_lats = pal_lats_rain[start_idx:end_idx]
        chunk_lons = pal_lons_rain[start_idx:end_idx]

        # Interpolate IMERG data for the current chunk
        imerg_precip_chunk = imerg_ds_xr.interp(
            time=("points", chunk_dates), lat=("points", chunk_lats),
            lon=("points", chunk_lons), method="nearest"
        )
        # Set places where the values are less than 0 to NaN
        imerg_precip_chunk = imerg_precip_chunk.where(imerg_precip_chunk >= 0, np.nan)
        return imerg_precip_chunk
    # Use Dask for lazy evaluation and parallel processing
    import dask.array as da
    dask_chunks = [process_chunk_dask(start, min(start + chunk_size, len(df))) for start in range(0, len(df), chunk_size)]
    dask_results = da.concatenate([da.from_array(chunk.values) for chunk in dask_chunks])

    # Compute the results and add to the DataFrame
    df['IMERG'] = dask_results.compute()

    return df

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def simple_process_gpcp_batch(batch, version):
    """Simple processing of GPCP files without intensive optimizations"""
    try:
        processed_batch = xr.open_mfdataset(
            batch, 
            combine='by_coords', 
            parallel=False,  # Disable parallel processing to reduce CPU load
            engine='netcdf4'
        )
        processed_batch = ds_swaplon(processed_batch)
        return processed_batch
    except Exception as e:
        print(f"Error processing {version} batch: {e}")
        return None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def simple_process_imerg_batch(batch):
    """Simple IMERG processing without parallel jobs for server-friendly operation"""
    try:
        # Use the existing process_imerg but we'll load files sequentially instead
        processed_batch = xr.open_mfdataset(
            batch, 
            combine='by_coords', 
            parallel=False,  # No parallel processing
            engine='netcdf4'
        )
        return processed_batch
    except Exception as e:
        print(f"Error processing IMERG batch: {e}")
        return None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def compute_rainfall_fraction_and_volume_by_windspeed_bins(
        rainfall_df,products, bn_size=1, threshold=0.2):
    """
    Compute precipitation fraction for each temperature bin for multiple precipitation products.

    Parameters:
    - temp_data: xarray.DataArray of temperature data.
    - precip_products: List of tuples (name, xarray.DataArray) for precipitation products.
    - seas_mnth: List of months for seasonal analysis (e.g., [12, 1, 2] for DJF). If None, compute for all months.
    - temp_bin_size: Size of the temperature bins (default: 1).
    - threshold: Precipitation threshold to calculate fraction (default: 0.02).

    Returns:
    - precip_fraction_by_bin: Dictionary with product names as keys and DataFrames of precipitation fraction by temperature bin.
    """
    # Define wind speed bins based on the wind speed data
    windspeed_data = rainfall_df['wind_speed'].values.flatten()
    wind_bins = np.arange(windspeed_data.min().item(), windspeed_data.max().item() + bn_size, bn_size)
    wind_bin_labels = (wind_bins[:-1] + wind_bins[1:]) / 2  # Midpoints of bins for labeling

    # Initialize a dictionary to store results
    rain_fraction_and_volume_results= []

    for product_name in [i for i in products if i != 'wind_speed']:
        print(f"Processing {product_name}...")        

        wind_speed_data_cpy = windspeed_data.copy() 
        rr_data = rainfall_df[product_name].values.flatten()      

        # Drop NaNs
        valid_indices = ~np.isnan(wind_speed_data_cpy) & ~np.isnan(rr_data)
        wind_speed_data_cpy = wind_speed_data_cpy[valid_indices]
        rr_data = rr_data[valid_indices]

        if product_name == 'rain_rate':
            product_name = 'PAL'  

        # Create a DataFrame for binning
        df = pd.DataFrame({'Wind_Speed': wind_speed_data_cpy, f'{product_name}_rainfall': rr_data})
        df['Wind_Speed_bin'] = pd.cut(df['Wind_Speed'], bins=wind_bins, labels=wind_bin_labels, include_lowest=True)

        # # PDFc
        fraction_group = df.groupby('Wind_Speed_bin')
        pdfc = fraction_group.apply(lambda x: pd.Series({
            f'{product_name}_Rainfall_Fraction': (x[f'{product_name}_rainfall'] >= threshold).sum() / len(x)
        })).reset_index()

        # PDFv
        volume_group = df.groupby('Wind_Speed_bin')
        volume_group = volume_group.apply(lambda x: pd.Series({
            f'{product_name}_Rainfall_Volume': x[f'{product_name}_rainfall'].sum()
        })).reset_index()

        # Normalize each row by the sum of Rainfall_Volume
        pdfv = volume_group.copy()
        total_volume = pdfv[f'{product_name}_Rainfall_Volume'].sum()
        pdfv[f'{product_name}_Rainfall_Volume'] = pdfv[f'{product_name}_Rainfall_Volume'] / total_volume if total_volume != 0 else 0


        # pdfv = volume_group/volume_group[f'{product_name}_Rainfall_Volume'].sum()
        result = pd.merge(pdfc, pdfv, on='Wind_Speed_bin')

        # Store the results in the dictionary
        rain_fraction_and_volume_results.append(result)

    final_results = pd.concat(rain_fraction_and_volume_results, axis=1)

    return final_results

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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - 

def process_gpcp_with_PAL(pal_file, region_name, pal_df, gpcp_ds_xr, 
                          resolution, gpcp_version):

    coord_lst = list(gpcp_ds_xr.coords)

    missing_val = -9999 # -9999 for both PAL and GPCP data

    # get lat, lon var name
    lat_var = 'latitude' if 'latitude' in coord_lst else 'lat'
    lon_var = 'longitude' if 'longitude' in coord_lst else 'lon'

    pal_df['row_idx'], pal_df['col_idx'] = assign_to_gpcp_grid(pal_df['lat'], pal_df['lon'], resolution)

    # Handle missing values in rain_rate
    pal_df['rain_rate'] = pal_df['rain_rate'].replace(missing_val, np.nan)
    # set values less than 0 to NaN
    pal_df['rain_rate'] = pal_df['rain_rate'].where(pal_df['rain_rate'] >= 0, np.nan)

    # Now df contains the PAL data with GPCP grid assignments
    # average daily rainfall

    daily_avg = pal_df.groupby(['date', 'row_idx', 'col_idx'])[['rain_rate', 'lat', 'lon']].mean().reset_index()
    daily_avg = daily_avg.set_index('date')
    # convert rain_rate to mm/day
    daily_avg['rain_rate'] = daily_avg['rain_rate'] * 24  # convert to mm/day
    daily_avg['region'] = region_name
    daily_avg['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

    # collect GPCP data for this PAL
    # Vectorized approach for speed
    # Prepare arrays for lookup
    gpcp_times = gpcp_ds_xr['time'].values
    gpcp_lats = gpcp_ds_xr[lat_var].values
    gpcp_lons = gpcp_ds_xr[lon_var].values

    # Map PAL dates to nearest GPCP time index
    pal_dates = pd.to_datetime(daily_avg.index)
    gpcp_time_idx = np.searchsorted(gpcp_times, pal_dates)
    gpcp_time_idx = np.clip(gpcp_time_idx, 0, len(gpcp_times) - 1)

    # Map PAL lat/lon to nearest GPCP grid index
    pal_lats = daily_avg['lat'].values
    pal_lons = daily_avg['lon'].values

    gpcp_lat_idx = np.abs(gpcp_lats[:, None] - pal_lats).argmin(axis=0)
    gpcp_lon_idx = np.abs(gpcp_lons[:, None] - pal_lons).argmin(axis=0)

    # Extract GPCP values in a vectorized way
    gpcp_precip = gpcp_ds_xr['precip'].values
    # Handle missing values by replacing with NaN
    gpcp_precip = np.where(gpcp_precip == missing_val, np.nan, gpcp_precip)
    # set values less than 0 to NaN
    gpcp_precip[gpcp_precip < 0] = np.nan
    matched_vals = gpcp_precip[gpcp_time_idx, gpcp_lat_idx, gpcp_lon_idx]

    daily_avg[gpcp_version] = matched_vals

    # Extract probability of Liquid precipitation data if gpcp_version == 'GPCP_v3pt2'
    if gpcp_version == 'GPCP_v3pt2':
        gpcp_prob_liq = gpcp_ds_xr['probability_liquid_phase'].values
        daily_avg['prob_liq'] = gpcp_prob_liq[gpcp_time_idx, gpcp_lat_idx, gpcp_lon_idx]

    return daily_avg

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - 
def calculate_metrics(pal, gpcp):
    pal = pal.dropna()
    gpcp = gpcp.dropna()
    
    if len(pal) == 0 or len(gpcp) == 0:
        return np.nan, np.nan, np.nan
    
    # Calculate metrics
    rb = round(relative_bias(pal, gpcp) * 100,1)  # Relative Bias in %
    rmse = round(rmsqe(pal, gpcp), 2)  # Root Mean Square Error
    cc = round(p_corr(pal, gpcp), 2)   # Pearson Correlation Coefficient

    # Relative Bias (in %): (mean(GPCP) - mean(PAL)) / mean(PAL) * 100
    # mean_pal = np.mean(pal)
    # mean_gpcp = np.mean(gpcp)
    # rb = round(((mean_gpcp - mean_pal) / mean_pal) * 100, 1) if mean_pal != 0 else np.nan

    # # Root Mean Square Error (RMSE)
    # rmse = round(np.sqrt(np.mean((gpcp - pal) ** 2)), 2)

    # # Pearson Correlation Coefficient (CC)
    # if len(pal) > 1 and np.std(pal) > 0 and np.std(gpcp) > 0:
    #     cc = round(np.corrcoef(pal, gpcp)[0, 1], 2)
    # else:
    #     cc = np.nan

    return rb, rmse, cc

# - - -  - - - - - - - - - - - - - - - - - -- - - -  - - - - - - - - - - - - - - - - 
def get_cdf_and_norm_pdf_(arr_input,rnge,binsz):        

    # Filter out NaN values
    arr1d = arr_input[~np.isnan(arr_input)]
    
    rng = rnge
    bns = binsz

    arr_bin_means, arr_bin_edges, _ = stats.binned_statistic(x = arr1d, values = arr1d, statistic = 'mean', 
                                                             bins = bns, range = rng)
    arr_bin_cnt, _, _ = stats.binned_statistic(x = arr1d, values = arr1d, statistic = 'count', 
                                               bins = bns,range = rng)
    
    # noirmalised pdf
    arr_v = arr_bin_cnt * arr_bin_means
    arr_total_v = np.nansum(arr_v)
    norm_pdf = arr_v/arr_total_v

    # cdf
    cdf = np.nancumsum(norm_pdf)
    cdf /= cdf[-1]

    del(arr1d,arr_bin_means,arr_bin_cnt)

    return norm_pdf, cdf, arr_bin_edges

# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def plot_wind_speed_bin_comparison(data_dict, region_colors, title, ylabel, ylabrot, output_path=None):
    """
    Plots a 4x1 subplot bar chart comparing counts across regions for each wind speed bin.

    Parameters:
    - data_dict: dict
        Dictionary where keys are region names and values are DataFrames with columns:
        ['wind_speed_bin', 'count', 'percentage'].
    - region_colors: dict
        Dictionary mapping region names to their respective colors.
    - title: str
        Title for the entire figure.
    - ylabel: str
        Label for the y-axis.
    - output_path: str, optional
        If provided, saves the plot to the specified path.
    """
    import matplotlib.pyplot as plt

    # Set font and style
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    # Define wind speed bins
    wind_speed_bins = ['0-5', '5-10', '10-15', '>15']
    n_bins = len(wind_speed_bins)

    # Create subplots
    fig, axs = plt.subplots(n_bins, 1, figsize=(12, 20), sharex=False)
    fig.suptitle(title, fontsize=22, fontweight='bold')

    # Iterate over each wind speed bin
    for i, wind_bin in enumerate(wind_speed_bins):
        ax = axs[i]
        counts = []
        percentages = []
        regions = []

        # Collect data for the current wind speed bin
        for region_name, df in data_dict.items():
            bin_data = df[df['wind_speed_bin'] == wind_bin]
            if not bin_data.empty:
                counts.append(bin_data['count'].values[0])
                percentages.append(bin_data['percentage'].values[0])
                regions.append(region_name)

        # Plot bar chart
        colors = [region_colors.get(region, 'gray') for region in regions]
        bars = ax.bar(regions, counts, color=colors)

        # Annotate percentage on top of each bar
        for bar, percentage in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{percentage:.2f}%', ha='center', va='bottom',
                fontsize=18, fontweight='bold'
            )

        # Set title, labels, and grid
        ax.set_title(f'Wind Speed Bin: {wind_bin}', fontsize=18, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='x', labelrotation=0, labelsize=18)
        ax.tick_params(axis='y', labelrotation =ylabrot, labelsize=18)

    # Adjust layout and save/show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    gc.collect()  # Clean up memory


# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def compute_pdf_elements(data, bins):
    pdfc = []  # PDF by occurrence
    pdfv = []  # PDF by volume
    bin_labels = []  # Bin labels for the DataFrame

    total_count = len(data)
    total_volume = 0

    # Loop through bins to compute PDFc and PDFv
    for i, bn in enumerate(bins):
        if i == 0:
            bin_data = data[data['rain_rate'] <= bn]
        else:
            bin_data = data[(data['rain_rate'] > bins[i - 1]) & (data['rain_rate'] <= bn)]

        # PDFc: Percentage of occurrences in the bin
        bin_count = len(bin_data)
        pdfc.append((bin_count / total_count) * 100)

        # PDFv: Percentage of volume in the bin
        if bin_count > 0:
            bin_mean = bin_data['rain_rate'].mean()
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

# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
# calculate multiyear monthly mean rainfall rate for PAL and Buoy data
def calculate_multiyear_monthly_mean_rainfall_by_region(data_dict, tme_var):
    monthly_means_by_region = {}

    for region, data in data_dict.items():
        # Ensure 'time' column is datetime
        if tme_var not in data.columns:
            data = data.reset_index()  # Reset index to access 'date' if it's the index
        data[tme_var] = pd.to_datetime(data[tme_var])
        data['year'] = data[tme_var].dt.year
        # Extract month and group by month to calculate mean
        data['month'] = data[tme_var].dt.month
        monthly_mean = data.groupby(['ID', 'year', 'month'])['rain_rate'].sum().reset_index()
        monthly_mean = monthly_mean.groupby('month')['rain_rate'].mean().reset_index()

        # Store the result in the dictionary
        monthly_means_by_region[region] = monthly_mean

    return monthly_means_by_region


# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def read_nc_imger_file(file_path, product):
    imerg_precip_data = xr.open_dataset(file_path)
    if product == 'imerg_fn':
        precip_aray = imerg_precip_data.precipitation.data 
    elif product == 'imerg_mw':
        precip_aray = imerg_precip_data.MWprecipitation.data 
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data.attrs['BeginDate']
    imerg_precip_data.close()
    
    # Convert time to pandas datetime
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')

    del(imerg_precip_data,imerg_time)
    
    return precip_aray, imerg_time_index

# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def process_imerg(files, product):
    img_lon, img_lat = return_imerg_cords(files[0])

    # Use parallel processing to read files
    def process_file(imf):
        imerg_fn = read_nc_imger_file(imf, product)
        return imerg_fn[0], imerg_fn[1]

    results = Parallel(n_jobs=20)(delayed(process_file)(imf) for imf in files)

    # Unpack results
    all_imfn_prcp, all_imfn_tms = zip(*results)

    # Ensure data is sorted by time
    sorted_indices = np.argsort(np.array(all_imfn_tms))
    all_imfn_prcp = np.array(all_imfn_prcp)[sorted_indices]
    all_imfn_tms = np.array(all_imfn_tms)[sorted_indices]

    # Process the files to aggregate data
    imerg_xarr_data = create_xarray(all_imfn_prcp, all_imfn_tms, img_lon, img_lat)

    return imerg_xarr_data

# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def create_xarray(all_precip, all_time_index, lon, lat, attrs=None):
    """
    Create an xarray DataArray from the list of 2D precipitation arrays and add attributes.

    Parameters:
    - all_precip: List or array of 2D precipitation arrays
    - all_time_index: List of timestamps
    - lon: Array of longitudes
    - lat: Array of latitudes
    - attrs: Dictionary of attributes to add to the DataArray (optional)

    Returns:
    - precip_data: xarray DataArray with the specified attributes
    """
    # Create a pandas DatetimeIndex from the list of timestamps
    time_index = pd.to_datetime(all_time_index)
    
    # Create an xarray DataArray from the list of 2D precipitation arrays
    precip_data = xr.DataArray(
        data=all_precip,
        dims=["time", "lat", "lon"],
        coords={
            "time": time_index,
            "lat": lat,
            "lon": lon
        }
    )

    # Add attributes if provided
    if attrs:
        precip_data.attrs.update(attrs)
    
    return precip_data

# - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - 
def return_imerg_cords(file):
    file_dat = xr.open_dataset(file)
    lon = file_dat.coords['lon'].values
    lat = np.flip(file_dat.coords['lat']).values

    del(file_dat)

    return lon, lat
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

def process_imerg_with_PAL_rainV2(pal_rain_df, imerg_ds_xr):
    """
    Fast vectorized IMERG matching with PAL rain data.
    
    This function efficiently matches PAL minute-level data to IMERG daily data by:
    1. Identifying unique date/location combinations to reduce redundant lookups
    2. Using vectorized xarray operations for fast processing
    3. Mapping results back to preserve original data structure
    
    Parameters:
    -----------
    pal_rain_df : pandas.DataFrame
        PAL rain dataframe with columns: 'date', 'lat', 'lon', 'rain_rate', etc.
    imerg_ds_xr : xarray.Dataset
        IMERG dataset with precipitation data
        
    Returns:
    --------
    pandas.DataFrame
        Original PAL dataframe with added 'IMERG' column containing matched precipitation values
    """
    print(f"Starting fast IMERG matching for {len(pal_rain_df)} PAL records...")
    
    # Make a copy to avoid modifying original data
    pal_imerg_df_rain = pal_rain_df.copy()
    
    # Extract arrays for processing
    pal_dates_rain = pd.to_datetime(pal_imerg_df_rain['date'])
    pal_lats_rain = pal_imerg_df_rain['lat'].values
    pal_lons_rain = pal_imerg_df_rain['lon'].values
    
    # Copy and prepare IMERG dataset
    imerg_ds_xr_cpy = imerg_ds_xr.copy(deep=True)
    
    # Rename latitude/longitude dims to 'lat' and 'lon' if needed
    if 'latitude' in imerg_ds_xr_cpy.dims or 'longitude' in imerg_ds_xr_cpy.dims:
        imerg_ds_xr_cpy = imerg_ds_xr_cpy.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    if len(pal_dates_rain) > 0:
        # Subset IMERG data to PAL's date range
        min_date, max_date = pal_dates_rain.min(), pal_dates_rain.max()
        imerg_subset = imerg_ds_xr_cpy.sel(time=slice(min_date, max_date))
        
        # VECTORIZED APPROACH - Much faster than one-by-one processing
        print("Creating unique location-date combinations to reduce redundant lookups...")
        pal_coords = pd.DataFrame({
            'date': pal_dates_rain,
            'lat': pal_lats_rain,
            'lon': pal_lons_rain,
            'original_index': range(len(pal_dates_rain))
        })
        
        # Group by date, lat, lon to find unique combinations
        unique_coords = pal_coords.groupby(['date', 'lat', 'lon']).first().reset_index()
        print(f"Reduced to {len(unique_coords)} unique date/location combinations (from {len(pal_coords)})")
        
        # Use xarray's vectorized selection for all unique points at once
        try:
            print("Performing vectorized IMERG lookup...")
            
            # Create DataArrays for coordinates
            coord_dates = xr.DataArray(unique_coords['date'], dims=['points'])
            coord_lats = xr.DataArray(unique_coords['lat'], dims=['points']) 
            coord_lons = xr.DataArray(unique_coords['lon'], dims=['points'])
            
            # Vectorized selection - much faster than loops
            imerg_results = imerg_subset['precipitation'].sel(
                time=coord_dates,
                lat=coord_lats, 
                lon=coord_lons,
                method='nearest'
            )
            
            # Convert to values and handle missing data
            imerg_unique_values = imerg_results.values
            imerg_unique_values[imerg_unique_values < 0] = np.nan
            
            # Map results back to original dataframe
            unique_coords['IMERG'] = imerg_unique_values
            
            # Merge back with original data
            pal_coords_with_imerg = pal_coords.merge(
                unique_coords[['date', 'lat', 'lon', 'IMERG']], 
                on=['date', 'lat', 'lon'], 
                how='left'
            )
            
            # Sort by original index to maintain order
            pal_coords_with_imerg = pal_coords_with_imerg.sort_values('original_index')
            
            # Assign results
            pal_imerg_df_rain['IMERG'] = pal_coords_with_imerg['IMERG'].values
            
            successful_matches = len([v for v in pal_coords_with_imerg['IMERG'].values if not np.isnan(v)])
            print(f"✓ SUCCESS! Matched {successful_matches} out of {len(pal_imerg_df_rain)} points to IMERG data")
            print(f"✓ Efficiency gain: {len(pal_coords)}/{len(unique_coords)} = {len(pal_coords)/len(unique_coords):.1f}x fewer lookups needed!")
            
        except Exception as e:
            print(f"Vectorized approach failed: {e}")
            print("Falling back to batch processing...")
            
            # Fallback to efficient batch processing
            batch_size = 1000
            imerg_values = []
            
            for i in range(0, len(unique_coords), batch_size):
                end_idx = min(i + batch_size, len(unique_coords))
                batch = unique_coords.iloc[i:end_idx]
                
                try:
                    batch_results = imerg_subset['precipitation'].sel(
                        time=('points', batch['date'].values),
                        lat=('points', batch['lat'].values),
                        lon=('points', batch['lon'].values),
                        method='nearest'
                    ).values
                    
                    batch_results[batch_results < 0] = np.nan
                    imerg_values.extend(batch_results)
                    
                    if (i // batch_size + 1) % 10 == 0:
                        print(f"Processed batch {i // batch_size + 1}/{(len(unique_coords) + batch_size - 1) // batch_size}")
                        
                except Exception as batch_error:
                    print(f"Batch {i // batch_size + 1} failed: {batch_error}")
                    imerg_values.extend([np.nan] * (end_idx - i))
            
            # Map results back
            unique_coords['IMERG'] = imerg_values[:len(unique_coords)]
            pal_coords_with_imerg = pal_coords.merge(
                unique_coords[['date', 'lat', 'lon', 'IMERG']], 
                on=['date', 'lat', 'lon'], 
                how='left'
            )
            pal_coords_with_imerg = pal_coords_with_imerg.sort_values('original_index')
            pal_imerg_df_rain['IMERG'] = pal_coords_with_imerg['IMERG'].values
            
            successful_matches = len([v for v in pal_coords_with_imerg['IMERG'].values if not np.isnan(v)])
            print(f"✓ Batch processing complete: {successful_matches} successful matches")
    else:
        pal_imerg_df_rain['IMERG'] = np.nan
        print("No PAL data to process")
    
    return pal_imerg_df_rain