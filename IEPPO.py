'''
Integrated Evaluation of Precipitation Products Over Oceans
Ground Truth Data Sources:
    - Passive Aquatic Listeners (PALs)
    - Ocean Buoys 
    - Attolls
    - OceanRain

Precipitation products:
    - Satellite-based: GPCP (v3.2, 3.3), IMERG (v06, v07)
    - Reanalysis-based: ERA5, MERRA-2
'''

#%%
from IEPPO_utils import *

#%% DEFINE PATHS AND DIRECTORIES
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_1998_2024'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2000_2020'

path_to_imerg_v06 = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV6/DataV6'

path_to_imerg_v07 = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_daily_1998-2025'

path_to_era5_tp = r'/ra1/pubdat/ECMWF/ERA5/daily'

path_to_merra2 = r'/ra1/pubdat/MERRA/Daily'

#%% LIST AND LOAD DATA FILES

print("Listing and loading data files...")
all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

all_buoy_dirs = [os.path.join(moored_bouys_paf, d) for d in os.listdir(moored_bouys_paf) if os.path.isdir(os.path.join(moored_bouys_paf, d))]

all_gpcp_v3pt2_files = sorted([os.path.join(path_to_gpcp_v3pt2, f) for f in os.listdir(path_to_gpcp_v3pt2) if f.endswith('.nc4')])

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])

all_imerg_v06_files = sorted([os.path.join(path_to_imerg_v06, f) for f in os.listdir(path_to_imerg_v06) if f.endswith('.nc4')])

all_imerg_v07_files = sorted([os.path.join(path_to_imerg_v07, f) for f in os.listdir(path_to_imerg_v07) if f.endswith('.nc4')])

all_era5_tp_files = sorted([os.path.join(path_to_era5_tp, f) for f in os.listdir(path_to_era5_tp) if f'era5_tp_' in f and f.endswith('.nc')])

all_merra2_files = sorted([os.path.join(path_to_merra2, f) for f in os.listdir(path_to_merra2) if f.endswith('.nc4')])

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load GPCP datasets using xarray with chunking for efficiency
gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcp_v3pt2_files,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                    cache=False
                                    )

gpcp_ds_v3pt2_xr = ds_swaplon(gpcp_ds_v3pt2_xr)

gpcp_ds_v3pt3_xr = xr.open_mfdataset(all_gpcp_v3pt3_files,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                    cache=False
                                    )
gpcp_ds_v3pt3_xr = ds_swaplon(gpcp_ds_v3pt3_xr)

print("✅ GPCP loading complete!...")
print("-" * 30 + "\n")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Use multiprocessing to process ERA5 files in parallel
era5_ds_xr_list = []
with Pool(processes=18) as pool:  # Adjust the number of processes as needed
    era5_ds_xr_list = pool.map(process_era5_file, enumerate(all_era5_tp_files))
# Combine all processed batches into a single xarray dataset - simple version
if era5_ds_xr_list:
    era5_ds_xr = xr.concat(era5_ds_xr_list, dim="valid_time")
    print("✅ ERA5 loading complete")
print("-" * 30 + "\n")
del(era5_ds_xr_list)
gc.collect() 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Use multiprocessing to process MERRA2 files in parallel
mer2_ds_xr_list = []
with Pool(processes=18) as pool:  # Adjust the number of processes as needed
    mer2_ds_xr_list = pool.map(process_merra2_file, enumerate(all_merra2_files))
# Combine all processed batches into a single xarray dataset - simple version
if mer2_ds_xr_list:
    mer2_ds_xr_list = [ds for ds in mer2_ds_xr_list if ds is not None]  # Filter out None values
    mer2_ds_xr = xr.concat(mer2_ds_xr_list, dim="time")
    print("✅ MERRA2 loading complete")
print("-" * 30 + "\n")
del(mer2_ds_xr_list)
gc.collect() 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Use multiprocessing to process IMERG files in parallel
imerg_v06_ds_xr_list = []
with Pool(processes=18) as pool:
    imerg_v06_ds_xr_list = pool.map(
        process_imerg_file,
        [(idx, file_path, 'v06') for idx, file_path in enumerate(all_imerg_v06_files)]
    )

# Combine all processed batches into a single xarray dataset - simple version
if imerg_v06_ds_xr_list:
    imerg_v06_ds_xr = xr.concat(imerg_v06_ds_xr_list, dim="time")

imerg_v07_ds_xr_list = []
with Pool(processes=18) as pool:
    imerg_v07_ds_xr_list = pool.map(
        process_imerg_file,
        [(idx, file_path, 'v07') for idx, file_path in enumerate(all_imerg_v07_files)]
    )
# Combine all processed batches into a single xarray dataset - simple version
if imerg_v07_ds_xr_list:
    imerg_v07_ds_xr = xr.concat(imerg_v07_ds_xr_list, dim="time")


print("IMERG loading complete")
print("-" * 50 + "\n")
del(imerg_v06_ds_xr_list, imerg_v07_ds_xr_list)
gc.collect() 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# CLASSIFY PAL FILES BY REGION
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, 
                                                               PAL_region_bounds)

# CLASSIFICATION SUMMARY
# Print classification summary
print("\n" + "="*50)
print("PAL CLASSIFICATION SUMMARY")
print("="*50)
for region, files in pals_classed_by_region.items():
    if region != "Unclassified":
        print(f"{region}: {len(files)} PALs")
print("="*50)

# Expected counts from your figure:
expected_counts = {
    "ETNP": 4,   # Extratropical North Pacific
    "TNEP": 20,  # Tropical Northeastern Pacific  
    "TSEP": 6,   # Tropical Southeastern Pacific
    "STNA": 18,  # Subtropical North Atlantic
    "TNIO": 3,   # Tropical North Indian Ocean
    "TNWP": 7    # Tropical Northwestern Pacific
}

print("\nCOMPARISON WITH EXPECTED COUNTS:")
print("-" * 30)
total_expected = sum(expected_counts.values())
total_actual = sum(len(files) for region, files in pals_classed_by_region.items() if region != "Unclassified")

for region in expected_counts:
    actual = len(pals_classed_by_region.get(region, []))
    expected = expected_counts[region]
    status = "✓" if actual == expected else "✗"
    print(f"{region}: Expected {expected}, Got {actual} {status}")

print(f"\nTotal Expected: {total_expected}")
print(f"Total Actual: {total_actual}")
print(f"Difference: {total_actual - total_expected}")
print('✅ PAL data loading and classification complete!')
print("-" * 30 + "\n")
gc.collect() 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# CLASSIFY BUOY FILES BY REGION
pacific_buoy_dir = next((d for d in all_buoy_dirs if "PACIFIC" in d.upper()), None)
indian_buoy_dir = next((d for d in all_buoy_dirs if "INDIAN" in d.upper()), None)
atlantic_buoy_dir = next((d for d in all_buoy_dirs if "ATLANTIC" in d.upper()), None)

# Ensure directories were found
if not pacific_buoy_dir:
    raise ValueError("No directory found for PACIFIC region.")
if not indian_buoy_dir:
    raise ValueError("No directory found for INDIAN region.")
if not atlantic_buoy_dir:
    raise ValueError("No directory found for ATLANTIC region.")

# Get all files for each region
pacific_buoy_files = sorted([os.path.join(pacific_buoy_dir, f) for f in os.listdir(pacific_buoy_dir) if f.endswith('.cdf')])

# define buoy files by regions
pacific_buoy_regions = ['ENP', 'WNP']

# FIRST GROUP BUOY FILES BY REGION BASED ON THEIR LONGITUDE
# Define longitude bounds for ENP and WNP
pacific_region_bounds = {
    'ENP': (-180, -60),  # Longitude range for Eastern North Pacific in [-180, 180]
    'WNP': (120, 180)      # Longitude range for Western North Pacific in [-180, 180]
}

# Group buoy files by region
buoy_files_by_region = {'ENP': [], 'WNP': []}
for buoy_file in pacific_buoy_files:
    with xr.open_dataset(buoy_file) as ds:
        buoy_lon = ds['lon'].values[0]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

    for region, bounds in pacific_region_bounds.items():
        if bounds[0] <= buoy_lon <= bounds[1]:
            buoy_files_by_region[region].append(buoy_file)
            break

# add the india and atlantic buoys
buoy_files_by_region['IND'] = sorted([os.path.join(indian_buoy_dir, f) for f in os.listdir(indian_buoy_dir) if f.endswith('.cdf')])
buoy_files_by_region['ATL'] = sorted([os.path.join(atlantic_buoy_dir, f) for f in os.listdir(atlantic_buoy_dir) if f.endswith('.cdf')])
print("✅ BUOY CLASSIFICATION SUMMARY COMPLETED")
print("\n" + "="*50)
gc.collect()
print("Data files listed and datasets loaded.")


#%% MATCHING GROUND TRUTH DATA AND GRIDDED PRECIPITATION PRODUCTS
# THE PAL MATCHING
print("\nStarting spatiotemporal matching of PAL and GPCP data...")
resolution = 0.5  # 0.5 degree resolution

# regional_PAL_sate_dfs_daily_mean = {}
regional_PAL_sate_dfs_daily_mean = {}
# regional_PAL_sate_dfs_daily_lst = []
regional_PAL_sate_dfs_daily_lst = []
for region_name, pal_files in list(pals_classed_by_region.items())[:-1]:
  
    print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")      

    # store PAL and GPCP dataframes
    region_pal_sate_dfs = []     

    # LOAD PAL DATA
    for i,pal_file in enumerate(pal_files):
        pal_ds = xr.open_dataset(pal_file)

        if i % 5 == 0:

            print(f"Processing PAL file: {os.path.basename(pal_file)}")

        pal_rain_df = grab_PAL_rain_and_wind_df(pal_ds)     
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        pal_rain_gpcpv3pt2_df = pal_rain_df.copy()           
        
        pal_gpcpv3pt2_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                        pal_rain_gpcpv3pt2_df,
                                        gpcp_ds_v3pt2_xr, 'GPCP v3.2') 

        pal_gpcpv3pt2_df_rain.index = pd.to_datetime(pal_gpcpv3pt2_df_rain['time'])  # Ensure index is datetime

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        pal_rain_gpcpv3pt3_df = pal_rain_df.copy()          
        
        pal_gpcpv3pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                        pal_rain_gpcpv3pt3_df,
                                        gpcp_ds_v3pt3_xr, 'GPCP v3.3')  # , pal_wind_gpcpv3pt3_df

        pal_gpcpv3pt3_df_rain.index = pd.to_datetime(pal_gpcpv3pt3_df_rain['time'])        

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        # # Process ERA5 data with PAL - Ultra memory-efficient approach
        pal_rain_era5_df = pal_rain_df.copy() 
        
        pal_era5_df_rain = process_era5_with_PAL_rain_and_wind_v1(pal_rain_era5_df, era5_ds_xr)        

        pal_era5_df_rain.index = pd.to_datetime(pal_era5_df_rain['time'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        # pal_rain_imerg_df = pal_rain_df.copy() 
        # pal_imerg_v06_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v06_ds_xr, 'v06')
        # pal_imerg_v06_df_rain.index = pd.to_datetime(pal_imerg_v06_df_rain['time'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        pal_rain_imerg_df = pal_rain_df.copy() 
        pal_imerg_v07_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v07_ds_xr, 'v07')
        pal_imerg_v07_df_rain.index = pd.to_datetime(pal_imerg_v07_df_rain['time'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------

        # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data         

        # merge GPCP v3.2 data  
        pal_df_combined_rain = pal_gpcpv3pt2_df_rain.copy()
        pal_df_combined_rain = pal_df_combined_rain[['time','date','rain_rate', 
                                                     'GPCP v3.2','PLP_GPCP v3.2']].copy()       

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.3 data
        # bring the lquid precip data into gpcp v3.3 df
        pal_gpcpv3pt3_daily = pal_gpcpv3pt3_df_rain.copy()        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv3pt3_df_rain[['date','GPCP v3.3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3.3')
        )

        # # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['GPCP v3.3_v3.3', 'date_v3.3']], 
                                                inplace=True)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # ERA5 merge
        pal_era5_daily = pal_era5_df_rain.copy()
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_era5_df_rain[['date','ERA5']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_ERA5')
        )
        # # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['ERA5_ERA5', 'date_ERA5']], 
                                                inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
        # IMERG v06 merge
        # pal_imerg_v06_daily = pal_imerg_v06_df_rain.copy()
        
        # pal_df_combined_rain = pal_df_combined_rain.merge(
        #     pal_imerg_v06_df_rain[['date','IMERG v06']], 
        #     left_index=True, right_index=True, how='left', suffixes=('', '_IMERG v06')
        # )
        # # Remove any duplicate columns from previous merges
        # pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
        #                                         ['IMERG v06_IMERG v06', 'date_IMERG v06']], 
        #                                         inplace=True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # IMERG v07 merge
        pal_imerg_v07_daily = pal_imerg_v07_df_rain.copy()
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_imerg_v07_df_rain[['date','v07']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v07')  
        )
        # # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['v07_v07', 'date_v07']], 
                                                inplace=True)       
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -        
        pal_df_combined_rain = pal_df_combined_rain[pal_df_combined_rain['PLP_GPCP v3.2'] == 100]        

        # groupby date and get mean of rain_rate and GPCP data 'GPCP_v1pt3',
        grp = pal_df_combined_rain.groupby('date')
        daily_avg_rain = grp.mean([['rain_rate', 
                                    'GPCP v3.2', 
                                    'GPCP v3.3', 
                                    # 'IMERG v06',
                                    'IMERG v07',
                                    'ERA5']]) \
        .join(pal_df_combined_rain.groupby('date')['time'] \
        .count() \
        .to_frame('n_min')
        ).reset_index()

        daily_avg_rain['cov_hr'] = daily_avg_rain['n_min'] / 60.0
        daily_avg_rain = daily_avg_rain[daily_avg_rain['cov_hr'] >= 12] 
        
        # # multiply PAL rain rate by 24 to get daily average
        daily_avg_rain['rain_rate'] *= 24
        # # add region name and track_PAL_id to the dataframe
        daily_avg_rain['region'] = region_name  # Add region name for clarity
        daily_avg_rain['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

        daily_avg_rain.drop(columns=['n_min', 'cov_hr',
                                     'PLP_GPCP v3.2',], 
                                     inplace=True)
        
        daily_avg_rain.rename(columns={'v07':'IMERG v07'}, 
                                      inplace=True)

        # region_pal_sate_dfs.append(daily_avg_rain)
        region_pal_sate_dfs.append(daily_avg_rain)

        pal_ds.close()

    # Combine all region PAL-GPCP dataframes into a single dataframe
    region_pal_sate_df = pd.concat(region_pal_sate_dfs)        
    
    # calculate daily mean per track_PAL_id
    region_pal_sate_df_daily_mean = region_pal_sate_df.groupby(['track_PAL_id'])[
                                                               ['rain_rate', 
                                                                'GPCP v3.2', 
                                                                'GPCP v3.3',
                                                                # 'IMERG v06',
                                                                'IMERG v07',
                                                                'ERA5']] \
                                                     .mean() \
                                                     .reset_index()
    region_pal_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
    regional_PAL_sate_dfs_daily_mean[region_name] = region_pal_sate_df_daily_mean

    # Append to the list for later processing
    regional_PAL_sate_dfs_daily_lst.append(region_pal_sate_df)
gc.collect()  # Clean up memory


# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# THE BUOY MATCHING
# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
regional_buoy_sate_dfs_daily_mean = {}
regional_buoy_sate_dfs_daily_lst = []
for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
        # store Buoy and GPCP dataframes
    region_buoy_sate_dfs = []  

    # LOAD Buoy DATA
    for b_file in buoy_files:
        b_df, b_lat, b_lon = grab_Buoy_data_df(b_file)
        print(f"Processing Buoy file: {os.path.basename(b_file)}")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -

        # Process GPCP v3.2 - Memory efficient version
        gpcpv3pt2_df = extract_buoy_satellite_data_memory_efficient(
            gpcp_ds_v3pt2_xr, b_lat, b_lon, 'GPCP v3.2', 'precip',chunk_size=300
        )
        
        if gpcpv3pt2_df is None:
            print("Warning: Failed to extract GPCP v3.2 data, creating empty dataframe")
            b_rain_gpcpv3pt2_df = pd.DataFrame(columns=['date', 'GPCP v3.2'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        # Process GPCP v3.3 - Memory efficient version
        gpcpv3pt3_df = extract_buoy_satellite_data_memory_efficient(
            gpcp_ds_v3pt3_xr, b_lat, b_lon, 'GPCP v3.3', 'precip', chunk_size=300
        )
        
        if gpcpv3pt3_df is None:
            print("Warning: Failed to extract GPCP v3.3 data, creating empty dataframe")
            gpcpv3pt3_df = pd.DataFrame(columns=['date', 'GPCP v3.3'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process ERA5 data with Buoy - Memory efficient version 
        era5_df = extract_buoy_satellite_data_memory_efficient(
            era5_ds_xr, b_lat, b_lon, 'ERA5', 'tp', chunk_size=300
        )
        
        if era5_df is None:
            print("Warning: Failed to extract ERA5 data, creating empty dataframe")
            era5_df = pd.DataFrame(columns=['date', 'ERA5'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process IMERG data with Buoy - Memory efficient version
        imerg_v07_df = extract_buoy_satellite_data_memory_efficient(
            imerg_v07_ds_xr, b_lat, b_lon, 'IMERG v07', None, chunk_size=300  # Reduced from 100 to 50 for IMERG
        )
        
        if imerg_v07_df is None:
            print("Warning: Failed to extract IMERG data, creating empty dataframe")
            imerg_v07_df = pd.DataFrame(columns=['date', 'IMERG v07'])
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -    

        # COMBINE BY RAINFALL RATE
        b_df_combined_rain = b_df[['date', 'rain_rate', 'quality_flag']].copy()        
        
         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.2 data
        b_df_combined_rain = b_df_combined_rain.merge(
            gpcpv3pt2_df[['date','GPCP v3.2', 'PLP_GPCP v3.2']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt2')
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.3 data
        b_df_combined_rain = b_df_combined_rain.merge(
            gpcpv3pt3_df[['date','GPCP v3.3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt3')
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge IMERG data - merge on 'date' column instead of index
        b_df_combined_rain = b_df_combined_rain.merge(
            imerg_v07_df[['date', 'IMERG v07']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_IMERG')
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # retain only columns where PLP_GPCP_v3pt2 is == 100
        # b_df_combined_rain = b_df_combined_rain[b_df_combined_rain['PLP_GPCP_v3pt2'] == 100]
        # retain only columns where PLP_GPCP_v3pt3 is == 100
        b_df_combined_rain = b_df_combined_rain[b_df_combined_rain['PLP_GPCP v3.3'] == 100]


        # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
        b_df_combined_rain = b_df_combined_rain[(b_df_combined_rain['quality_flag'] >= 1) & \
                                                                                                    
                                                (b_df_combined_rain['quality_flag'] <= 3)]


    
        # multiply PAL rain rate by 24 to get daily average
        b_df_combined_rain['rain_rate'] *= 24
        # add region name and track_PAL_id to the dataframe
        b_df_combined_rain['region'] = region_name  # Add region name for clarity
        b_df_combined_rain['ID'] = os.path.basename(b_file).split('.')[0]

        b_df_combined_rain.drop(columns=['quality_flag',
                                     'PLP_GPCP v3.2',], 
                                     inplace=True)

        region_buoy_sate_dfs.append(b_df_combined_rain)       

    # Combine all region PAL-GPCP dataframes into a single dataframe
    region_buoy_sate_df = pd.concat(region_buoy_sate_dfs)

    # calculate daily mean per track_PAL_id
    region_buoy_sate_df_daily_mean = region_buoy_sate_df.groupby(['ID'])[['rain_rate', 
                                                                        # 'GPCP_v1pt3', 
                                                                        'GPCP_v3pt2', 
                                                                        'GPCP_v3pt3', 
                                                                        # 'IMERG_v06',
                                                                        'IMERG v07',
                                                                        'ERA5']].mean().reset_index()  # , 'IMERG'IMERG']].mean().reset_index()
    region_buoy_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
    regional_buoy_sate_dfs_daily_mean[region_name] = region_buoy_sate_df_daily_mean

    # Append to the list for later processing
    regional_buoy_sate_dfs_daily_lst.append(region_buoy_sate_df)


gc.collect()  # Clean up memory