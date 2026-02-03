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

path_to_put_dfs = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/dfs_27Jan2026'
path_to_plots = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots_27Jan2026'

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
# print("\nStarting spatiotemporal matching of PAL and GPCP data...")
# resolution = 0.5  # 0.5 degree resolution

# # regional_PAL_sate_dfs_daily_mean = {}
# regional_PAL_sate_dfs_daily_mean = {}
# # regional_PAL_sate_dfs_daily_lst = []
# regional_PAL_sate_dfs_daily_lst = []
# for region_name, pal_files in list(pals_classed_by_region.items())[:-1]:
  
#     print(f"\nProcessing region: {region_name} with {len(pal_files)} PAL files")      

#     # store PAL and GPCP dataframes
#     region_pal_sate_dfs = []     

#     # LOAD PAL DATA
#     for i,pal_file in enumerate(pal_files):
#         pal_ds = xr.open_dataset(pal_file)

#         if i % 5 == 0:

#             print(f"Processing PAL file: {os.path.basename(pal_file)}")

#         pal_rain_df = grab_PAL_rain_and_wind_df(pal_ds)     
        
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
#         pal_rain_gpcpv3pt2_df = pal_rain_df.copy()           
        
#         pal_gpcpv3pt2_df_rain = process_gpcp_with_PAL_rain_and_wind(
#                                         pal_rain_gpcpv3pt2_df,
#                                         gpcp_ds_v3pt2_xr, 'GPCP v3.2') 

#         pal_gpcpv3pt2_df_rain.index = pd.to_datetime(pal_gpcpv3pt2_df_rain['time'])  # Ensure index is datetime

#         # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
#         pal_rain_gpcpv3pt3_df = pal_rain_df.copy()          
        
#         pal_gpcpv3pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
#                                         pal_rain_gpcpv3pt3_df,
#                                         gpcp_ds_v3pt3_xr, 'GPCP v3.3')  # , pal_wind_gpcpv3pt3_df

#         pal_gpcpv3pt3_df_rain.index = pd.to_datetime(pal_gpcpv3pt3_df_rain['time'])        

#         # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
#         # Process ERA5 data with PAL 
#         pal_rain_era5_df = pal_rain_df.copy() 
        
#         pal_era5_df_rain = process_era5_with_PAL_rain_and_wind_v1(pal_rain_era5_df, era5_ds_xr)        

#         pal_era5_df_rain.index = pd.to_datetime(pal_era5_df_rain['time'])

#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
#         # Process IMERG data with PAL 
#         # pal_rain_imerg_df = pal_rain_df.copy() 
#         # pal_imerg_v06_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v06_ds_xr, 'v06')
#         # pal_imerg_v06_df_rain.index = pd.to_datetime(pal_imerg_v06_df_rain['time'])
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
#         pal_rain_imerg_df = pal_rain_df.copy() 
#         pal_imerg_v07_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v07_ds_xr, 'v07')
#         pal_imerg_v07_df_rain.index = pd.to_datetime(pal_imerg_v07_df_rain['time'])
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        
#         # Process MERRA2 data with PAL
#         pal_rain_merra2_df = pal_rain_df.copy()
#         pal_merra2_df_rain = process_merra2_with_PAL_rain_and_wind_v1(pal_rain_merra2_df, mer2_ds_xr)
#         pal_merra2_df_rain.index = pd.to_datetime(pal_merra2_df_rain['time'])
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------

#         # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data         

#         # merge GPCP v3.2 data  
#         pal_df_combined_rain = pal_gpcpv3pt2_df_rain.copy()
#         pal_df_combined_rain = pal_df_combined_rain[['time','date','rain_rate', 
#                                                      'GPCP v3.2','PLP_GPCP v3.2']].copy()       

#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#         # merge GPCP v3.3 data
#         # bring the lquid precip data into gpcp v3.3 df
#         pal_df_combined_rain = pal_df_combined_rain.merge(
#             pal_gpcpv3pt3_df_rain[['date','GPCP v3.3']], 
#             left_index=True, right_index=True, how='left', suffixes=('', '_v3.3')
#         )

#         # Remove any duplicate columns from previous merges
#         pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
#                                                 ['GPCP v3.3_v3.3', 'date_v3.3']], 
#                                                 inplace=True)
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#         # ERA5 merge
        
#         pal_df_combined_rain = pal_df_combined_rain.merge(
#             pal_era5_df_rain[['date','ERA5']], 
#             left_index=True, right_index=True, how='left', suffixes=('', '_ERA5')
#         )
#         # # Remove any duplicate columns from previous merges
#         pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
#                                                 ['ERA5_ERA5', 'date_ERA5']], 
#                                                 inplace=True)
        
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#         # IMERG v06 merge
#         # pal_imerg_v06_daily = pal_imerg_v06_df_rain.copy()
        
#         # pal_df_combined_rain = pal_df_combined_rain.merge(
#         #     pal_imerg_v06_df_rain[['date','IMERG v06']], 
#         #     left_index=True, right_index=True, how='left', suffixes=('', '_IMERG v06')
#         # )
#         # # Remove any duplicate columns from previous merges
#         # pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
#         #                                         ['IMERG v06_IMERG v06', 'date_IMERG v06']], 
#         #                                         inplace=True)
        
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         # IMERG v07 merge
        
#         pal_df_combined_rain = pal_df_combined_rain.merge(
#             pal_imerg_v07_df_rain[['date','v07']], 
#             left_index=True, right_index=True, how='left', suffixes=('', '_v07')  
#         )
#         # # Remove any duplicate columns from previous merges
#         pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
#                                                 ['v07_v07', 'date_v07']], 
#                                                 inplace=True)       
        
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#         # MERRA2 merge
#         pal_df_combined_rain = pal_df_combined_rain.merge(
#             pal_merra2_df_rain[['date','MERRA2']], 
#             left_index=True, right_index=True, how='left', suffixes=('', '_MERRA2')
#         )
#         # Remove any duplicate columns from previous merges
#         pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
#                                                 ['MERRA2_MERRA2', 'date_MERRA2']], 
#                                                 inplace=True)
#         # 
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -        
#         pal_df_combined_rain = pal_df_combined_rain[pal_df_combined_rain['PLP_GPCP v3.2'] == 100]        

#         # groupby date and get mean of rain_rate and GPCP data 'GPCP_v1pt3',
#         grp = pal_df_combined_rain.groupby('date')
#         daily_avg_rain = grp.mean([['rain_rate', 
#                                     'GPCP v3.2', 
#                                     'GPCP v3.3', 
#                                     # 'IMERG v06',
#                                     'IMERG v07',
#                                     'ERA5',
#                                     'MERRA2']]) \
#         .join(pal_df_combined_rain.groupby('date')['time'] \
#         .count() \
#         .to_frame('n_min')
#         ).reset_index()

#         daily_avg_rain['cov_hr'] = daily_avg_rain['n_min'] / 60.0
#         daily_avg_rain = daily_avg_rain[daily_avg_rain['cov_hr'] >= 12] 
        
#         # # multiply PAL rain rate by 24 to get daily average
#         daily_avg_rain['rain_rate'] *= 24
#         # # add region name and track_PAL_id to the dataframe
#         daily_avg_rain['region'] = region_name  # Add region name for clarity
#         daily_avg_rain['track_PAL_id'] = os.path.basename(pal_file).split('.')[0]

#         daily_avg_rain.drop(columns=['n_min', 'cov_hr',
#                                      'PLP_GPCP v3.2',], 
#                                      inplace=True)
        
#         daily_avg_rain.rename(columns={'v07':'IMERG v07'}, 
#                                       inplace=True)

#         # region_pal_sate_dfs.append(daily_avg_rain)
#         region_pal_sate_dfs.append(daily_avg_rain)

#         pal_ds.close()

#     # Combine all region PAL-GPCP dataframes into a single dataframe
#     region_pal_sate_df = pd.concat(region_pal_sate_dfs)        
    
#     # calculate daily mean per track_PAL_id
#     region_pal_sate_df_daily_mean = region_pal_sate_df.groupby(['track_PAL_id'])[
#                                                                ['rain_rate', 
#                                                                 'GPCP v3.2', 
#                                                                 'GPCP v3.3',
#                                                                 # 'IMERG v06',
#                                                                 'IMERG v07',
#                                                                 'ERA5',
#                                                                 'MERRA2']] \
#                                                      .mean() \
#                                                      .reset_index()
#     region_pal_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
#     regional_PAL_sate_dfs_daily_mean[region_name] = region_pal_sate_df_daily_mean

#     # Append to the list for later processing
#     regional_PAL_sate_dfs_daily_lst.append(region_pal_sate_df)
# gc.collect()  # Clean up memory

# # save dfs to disk
# print("✅ PAL-GPCP matching complete!")
# print("Saving PAL-GPCP matched dataframes to disk...")
# pal_sate_daily_mean_df = pd.concat([df for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# pal_sate_daily_mean_df.to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_mean_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

# pal_sate_daily_rainfall_colasped_df = pd.concat(regional_PAL_sate_dfs_daily_lst, ignore_index=True)
# pal_sate_daily_rainfall_colasped_df.to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_rainfall_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

# pal_gpcv3_2_cmp = pd.concat([df[['rain_rate', 'GPCP v3.2']] for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# pal_gpcv3_3_cmp = pd.concat([df[['rain_rate', 'GPCP v3.3']] for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# pal_era5_cmp = pd.concat([df[['rain_rate', 'ERA5']] for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# pal_imergv07_cmp = pd.concat([df[['rain_rate', 'IMERG v07']] for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# pal_merra2_cmp = pd.concat([df[['rain_rate', 'MERRA2']] for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
# # save these files for later use
# pal_gpcv3_2_cmp.to_pickle(os.path.join(path_to_put_dfs, f'pal_gpcv3_2_daily_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))
# pal_gpcv3_3_cmp.to_pickle(os.path.join(path_to_put_dfs, f'pal_gpcv3_3_daily_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))
# pal_era5_cmp.to_pickle(os.path.join(path_to_put_dfs, f'pal_era5_daily_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))
# pal_imergv07_cmp.to_pickle(os.path.join(path_to_put_dfs, f'pal_imergv07_daily_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))
# pal_merra2_cmp.to_pickle(os.path.join(path_to_put_dfs, f'pal_merra2_daily_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# THE BUOY MATCHING
# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
print('Starting Buoy-GPCP matching...')
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
        era5_df.drop(columns=['number','spatial_ref'], inplace=True)  # drop redundant column
        era5_df.rename(columns={'tp': 'ERA5'}, inplace=True)
        
        if era5_df is None:
            print("Warning: Failed to extract ERA5 data, creating empty dataframe")
            era5_df = pd.DataFrame(columns=['date', 'ERA5'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process IMERG data with Buoy - Memory efficient version
        imerg_v07_df = extract_buoy_satellite_data_memory_efficient(
            imerg_v07_ds_xr, b_lat, b_lon, 'IMERG v07', None, chunk_size=300  # Reduced from 100 to 50 for IMERG
        )

        imerg_v07_df.drop(columns=['spatial_ref'], inplace=True)  # drop redundant column
        imerg_v07_df.rename(columns={'precipitation': 'IMERG v07'}, inplace=True)
        
        if imerg_v07_df is None:
            print("Warning: Failed to extract IMERG data, creating empty dataframe")
            imerg_v07_df = pd.DataFrame(columns=['date', 'IMERG v07'])
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -    

        # Process MERRA2 data with Buoy - Memory efficient version
        merra2_df = extract_buoy_satellite_data_memory_efficient(
            mer2_ds_xr, b_lat, b_lon, 'MERRA2', None, chunk_size=300
        )
        
        merra2_df.drop(columns=['spatial_ref'], inplace=True)
        merra2_df.rename(columns={'PRECTOTCORR':'MERRA2'}, inplace=True)
        
        if merra2_df is None:
            print("Warning: Failed to extract MERRA2 data, creating empty dataframe")
            merra2_df = pd.DataFrame(columns=['date', 'MERRA2'])

        # COMBINE BY RAINFALL RATE
        b_df_combined_rain = b_df[['date', 'rain_rate', 'quality_flag']].copy()        
        
         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.2 data
        b_df_combined_rain = b_df_combined_rain.merge(
            gpcpv3pt2_df[['date','GPCP v3.2', 'PLP_GPCP v3.2']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt2')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['GPCP v3.2_v3.2', 'date_v3pt2']], 
                                                inplace=True)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.3 data
        b_df_combined_rain = b_df_combined_rain.merge(
            gpcpv3pt3_df[['date','GPCP v3.3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3pt3')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['GPCP v3.3_v3.3', 'date_v3pt3']], 
                                                inplace=True)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # merge ERA5 data
        b_df_combined_rain = b_df_combined_rain.merge(
            era5_df[['date','ERA5']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_ERA5')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['ERA5_ERA5', 'date_ERA5']], 
                                                inplace=True)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge IMERG data - merge on 'date' column instead of index
        b_df_combined_rain = b_df_combined_rain.merge(
            imerg_v07_df[['date', 'IMERG v07']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_IMERG v07')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['IMERG v07_IMERG v07', 'date_IMERG v07']], 
                                                inplace=True)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # merge MERRA2 data
        b_df_combined_rain = b_df_combined_rain.merge(
            merra2_df[['date','MERRA2']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_MERRA2')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['MERRA2_MERRA2', 'date_MERRA2']], 
                                                inplace=True)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3. and PLP_GPCP v3.2 == 100
        b_df_combined_rain = b_df_combined_rain[(b_df_combined_rain['quality_flag'] >= 1) & \
                                                                                                    
                                                (b_df_combined_rain['quality_flag'] <= 3)]
        
        b_df_combined_rain = b_df_combined_rain[b_df_combined_rain['PLP_GPCP v3.2'] == 100]

        # multiply PAL rain rate by 24 to get daily average
        b_df_combined_rain['rain_rate'] *= 24
        
        # add region name and track_PAL_id to the dataframe
        b_df_combined_rain['region'] = region_name  # Add region name for clarity
        b_df_combined_rain['ID'] = os.path.basename(b_file).split('.')[0]

        b_df_combined_rain.drop(columns=['quality_flag', 'PLP_GPCP v3.2'], inplace=True)

        region_buoy_sate_dfs.append(b_df_combined_rain)       

    # Combine all region PAL-GPCP dataframes into a single dataframe
    region_buoy_sate_df = pd.concat(region_buoy_sate_dfs)

    # calculate daily mean per track_PAL_id
    region_buoy_sate_df_daily_mean = region_buoy_sate_df.groupby(['ID'])[['rain_rate',                                                                          
                                                                        'GPCP v3.2', 
                                                                        'GPCP v3.3', 
                                                                        # 'IMERG_v06',
                                                                        'IMERG v07',
                                                                        'ERA5',
                                                                        'MERRA2']].mean().reset_index()  # , 'IMERG'IMERG']].mean().reset_index()
    region_buoy_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
    regional_buoy_sate_dfs_daily_mean[region_name] = region_buoy_sate_df_daily_mean

    # Append to the list for later processing
    regional_buoy_sate_dfs_daily_lst.append(region_buoy_sate_df)


gc.collect()  # Clean up memory
print("✅ Buoy-GPCP matching complete!")
print("Saving Buoy-GPCP matched dataframes to disk...")
buoy_sate_daily_mean_df = pd.concat([df for df in regional_buoy_sate_dfs_daily_mean.values()], ignore_index=True)
buoy_sate_daily_mean_df.to_pickle(os.path.join(path_to_put_dfs, f'buoy_sate_daily_mean_from_all_regions_and_all_IDs_{cde_run_dte}.pkl'))

buoy_sate_daily_rainfall_colasped_df = pd.concat(regional_buoy_sate_dfs_daily_lst, ignore_index=True)
buoy_sate_daily_rainfall_colasped_df.to_pickle(os.path.join(path_to_put_dfs, f'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_{cde_run_dte}.pkl'))
print("✅ Buoy-GPCP matched dataframes saved to disk!")

#%% SCATTER PLOT
# PAL BASED ASSESSMENT

# calcute metrics for all regions combined
# rb_v1pt3, rmse_v1pt3, cc_v1pt3 = calculate_metrics(pal_gpcv1_3_cmp['rain_rate'], pal_gpcv1_3_cmp['GPCP_v1pt3'])
# rb_v3pt2, rmse_v3pt2, cc_v3pt2 = calculate_metrics(pal_gpcv3_2_cmp,'rain_rate', 'GPCP v3.2')
# rb_v3pt3, rmse_v3pt3, cc_v3pt3 = calculate_metrics(pal_gpcv3_3_cmp, 'rain_rate', 'GPCP v3.3')
# rb_era5, rmse_era5, cc_era5 = calculate_metrics(pal_era5_cmp, 'rain_rate', 'ERA5')
# rb_imergv07, rmse_imergv07, cc_imergv07 = calculate_metrics(pal_imergv07_cmp, 'rain_rate', 'IMERG v07')
# rb_merra2, rmse_merra2, cc_merra2 = calculate_metrics(pal_merra2_cmp, 'rain_rate', 'MERRA2')

# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# pal_sate_daily_mean_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'pal_sate_daily_mean_from_all_regions_and_all_tracks_20260128.pkl'))  # .to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_mean_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

# pal_sate_daily_rainfall_colasped_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'pal_sate_daily_rainfall_from_all_regions_and_all_tracks_20260128.pkl'))

# # Create scatter plots for PAL vs satellite products
# scatter_fig = plot_satellite_vs_groundtruth(pal_sate_daily_mean_df,
#                                             truth_col='rain_rate',
#                                             product_cols=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'],
#                                             product_labels=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'],
#                                             truth_label='PAL Observations',
#                                             max_val=18,
#                                             ticks=(0, 6, 12, 18),
#                                             figsize_per_col=6,
#                                             figsize_per_row=5,
#                                             savepath=os.path.join(path_to_plots, f'PAL_vs_Satellite_Comparison_{cde_run_dte}.png'))


#%% THE CATEGORICAL METRICS
# PAL BASED ASSESSMENT
# REGION BY REGION CAT METRICS

# region_based_cat_metrics = {}
# for region_name in pal_sate_daily_rainfall_colasped_df['region'].unique():

#     region_df = pal_sate_daily_rainfall_colasped_df[pal_sate_daily_rainfall_colasped_df['region'] == region_name]

#     # region_name = region_df['region'].unique()[0]

#     print(f"Processing region: {region_name}")

#     for product in ['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2']:
#         forcast = region_df[product]
#         observed = region_df['rain_rate']

#         # Calculate the categorical metrics
#         reg_cat_met = categorical_stats(forcast, observed, 1.0)

#         # Store the metrics in the dictionary
#         region_based_cat_metrics.setdefault(region_name, {})[product] = reg_cat_met



# products = ["GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5",  "MERRA2"]

# fig = plot_categorical_metrics_by_region(
#     metrics_dict=region_based_cat_metrics,
#     products=products,
#     product_colors=product_colors,
# )
# svnme = os.path.join(path_to_plots, 
#                      f'PAL_Satellite_Categorical_Metrics_by_Region_{cde_run_dte}.png')
# fig.savefig(svnme, dpi=300)


# # - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# # METRICS BY RAINFALL INTENSITY
# rainfall_bins = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# products = ['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2']
# ct_met = ['POD', 'FAR', 'Bias', 'HSS']
# qt_met = ['CC', 'RMSE', 'RB']
# # df_colapsed = pd.concat(regional_PAL_sate_dfs_daily_lst, ignore_index=True)

# qt_met_by_prdt = {}
# cat_met_by_prdt = {}
# for product in products:

#     cat_met_prdt = pd.DataFrame(index=rainfall_bins, columns=ct_met)
#     quant_met_prdt = pd.DataFrame(index=rainfall_bins, columns=qt_met)

#     forcast = pal_sate_daily_rainfall_colasped_df[product]
#     observed = pal_sate_daily_rainfall_colasped_df['rain_rate']

#     for r_bin in rainfall_bins:
#         # Calculate the categorical metrics
#         cat_mets = categorical_stats(forcast, observed, r_bin)
#         cat_met_prdt.loc[r_bin,'POD'] = cat_mets['POD']
#         cat_met_prdt.loc[r_bin,'FAR'] = cat_mets['FAR']
#         cat_met_prdt.loc[r_bin,'Bias'] = cat_mets['Bias']
#         cat_met_prdt.loc[r_bin,'HSS'] = cat_mets['HSS']

#         # Calculate the quantitative metrics
#         bin_df = pal_sate_daily_rainfall_colasped_df[pal_sate_daily_rainfall_colasped_df['rain_rate'] >= r_bin]
#         quant_mets = calculate_metrics(bin_df, 'rain_rate', product)
#         quant_met_prdt.loc[r_bin,'CC'] = quant_mets[2]
#         quant_met_prdt.loc[r_bin,'RMSE'] = quant_mets[1]
#         quant_met_prdt.loc[r_bin,'RB'] = quant_mets[0]   

#     cat_met_by_prdt[product] = cat_met_prdt
#     qt_met_by_prdt[product] = quant_met_prdt
    
# # - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# # THE PLOTTING
# # - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 


# # Metrics to plot
# cat_metrics = ['POD', 'FAR', 'HSS']
# qt_metrics  = ['CC', 'RMSE', 'RB']

# fig, axes = plt.subplots(
#     nrows=3, ncols=2,
#     figsize=(14, 12),
#     sharex=True
# )

# # ---- LEFT COLUMN: CATEGORICAL ----
# for i, met in enumerate(cat_metrics):
#     ax = axes[i, 0]

#     for product, df in cat_met_by_prdt.items():
#         ax.plot(
#             rainfall_bins,
#             df.loc[rainfall_bins, met].astype(float),
#             marker='o',
#             linewidth=2.2,
#             markersize=7,
#             color=product_colors[product],
#             label=product if i == 0 else None
#         )

#     ax.set_ylabel(met, fontsize=14, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_xscale('log')
#     ax.set_xticks(rainfall_bins)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#     ax.tick_params(labelsize=15)

# # ---- RIGHT COLUMN: QUANTITATIVE ----
# for i, met in enumerate(qt_metrics):
#     ax = axes[i, 1]

#     for product, df in qt_met_by_prdt.items():
#         ax.plot(
#             rainfall_bins,
#             df.loc[rainfall_bins, met].astype(float),
#             marker='o',
#             linewidth=2.2,
#             markersize=7,
#             color=product_colors[product],
#             label=product if i == 0 else None
#         )

#     if met == 'RB':
#         met_label = 'Bias [%]' 
#     elif met == 'RMSE':
#         met_label = 'RMSE [mm/day]'
#     else:
#         met_label = met

#     ax.set_ylabel(met_label, fontsize=15, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_xscale('log')
#     ax.set_xticks(rainfall_bins)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#     ax.tick_params(labelsize=15)

# # ---- X-axis labels (bottom row only) ----
# for ax in axes[-1, :]:
#     ax.set_xlabel('Rain Rate (mm/day)', fontsize=18, fontweight='bold')

# # ---- Column titles ----
# # axes[0, 0].set_title('Categorical Metrics', fontsize=16, fontweight='bold')
# # axes[0, 1].set_title('Quantitative Metrics', fontsize=16, fontweight='bold')

# # ---- Legend (single, clean) ----
# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.subplots_adjust(bottom=0.15)
# fig.legend(
#     handles, labels,
#     loc='lower center',
#     ncol=4,
#     fontsize=18,
#     frameon=False
# )

# plt.tight_layout(rect=[0, 0.08, 1, 1])
# svnme = os.path.join(path_to_plots, 
#                      f'PAL_Satellite_Metrics_by_Rainfall_Intensity_{cde_run_dte}.png')
# fig.savefig(svnme, dpi=300)
# # plt.show()

#%%  PDF
# bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
# bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']

# # Compute PDF elements for all datasets
# pal_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'rain_rate', bin_values)
# img_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'IMERG v07', bin_values)
# gpcp_v3pt2_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'GPCP v3.2', bin_values)
# gpcp_v3pt3_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'GPCP v3.3', bin_values)
# era5_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'ERA5', bin_values)
# merra2_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'MERRA2', bin_values)
# # ============================================================
# # PDFv / PDFc by Rainfall Intensity (Figure-5 style)
# # ============================================================

# fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=500)

# lw = 4

# # --- X axis: use actual bin values ---
# x = bin_values#[:-1]   # last edge has no PDF value

# # ------------------------------------------------------------
# # PDFv (Volume-based PDF)  —— ACTIVE
# # ------------------------------------------------------------
# ax.plot(x, pal_pdfc_pdfv['pdfv'], lw=lw, color='b', ls=':', label='PAL')
# ax.plot(x, merra2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['MERRA2'], label='MERRA2')
# ax.plot(x, era5_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['ERA5'], label='ERA5')
# ax.plot(x, img_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['IMERG v07'], label='IMERG v07')
# ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.2'], label='GPCP v3.2')
# ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.3'], label='GPCP v3.3')

# # ------------------------------------------------------------
# # PDFc (Count-based PDF)  —— OPTIONAL (commented)
# # ------------------------------------------------------------
# # ax.plot(x, pal_pdfc_pdfv['pdfc'], lw=lw, ls='--',
# #         color=product_colors['PAL'], label='PAL (PDFc)')
# # ax.plot(x, img_pdfc_pdfv['pdfc'], lw=lw, ls='--',
# #         color=product_colors['IMERG v07'], label='IMERG v07 (PDFc)')
# # ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfc'], lw=lw, ls='--',
# #         color=product_colors['GPCP v3.2'], label='GPCP v3.2 (PDFc)')
# # ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfc'], lw=lw, ls='--',
# #         color=product_colors['GPCP v3.3'], label='GPCP v3.3 (PDFc)')
# # ax.plot(x, era5_pdfc_pdfv['pdfc'], lw=lw, ls='--',
# #         color=product_colors['ERA5'], label='ERA5 (PDFc)')

# # ------------------------------------------------------------
# # Formatting (paper-quality)
# # ------------------------------------------------------------
# ax.set_xscale('log')
# ax.set_xlabel('Rain Rate [mm day$^{-1}$]', fontsize=18, fontweight='bold')
# ax.set_ylabel('PDF (%)', fontsize=18, fontweight='bold')

# ax.set_xticks(bin_values)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
# ax.tick_params(axis='both', which='major', labelsize=15, width=1.5, length=7)

# ax.grid(True, which='major', linestyle='--', alpha=0.6)
# ax.legend(fontsize=15, frameon=False)

# plt.tight_layout()
# svnme = os.path.join(path_to_plots, 
#                      f'PAL_Satellite_PDF_Comparison_{cde_run_dte}.png')
# fig.savefig(svnme, dpi=300)


#%%
print('Starting buoy-based assessment...')
# THE BUOY BASED ASSESSMENT
buoy_sate_daily_mean_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_mean_from_all_regions_and_all_IDs_20260128.pkl'))  

# Create scatter plots for Buoy vs satellite products
scatter_fig_buoy = plot_satellite_vs_groundtruth(buoy_sate_daily_mean_df,
                                            truth_col='rain_rate',
                                            product_cols=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07'], # , 'MERRA2'
                                            product_labels=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07'], # , 'MERRA2'
                                            truth_label='Buoy Observations',
                                            max_val=18,
                                            ticks=(0, 6, 12, 18),
                                            figsize_per_col=6,
                                            figsize_per_row=5,
                                            savepath=os.path.join(path_to_plots, f'Buoy_vs_Satellite_Comparison_{cde_run_dte}.png'))

print('Finished buoy-based assessment...')
print("-" * 30 + "\n")
#%%
print('Starting monthly timeseries analysis...')
# MONTHLY TIMESROES ANALYSIS
buoy_sate_daily_rainfall_colasped_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260128.pkl'))

df = buoy_sate_daily_rainfall_colasped_df.copy()

# Ensure datetime
df['date'] = pd.to_datetime(df['date'])

# Year–month
df['year_month'] = df['date'].dt.to_period('M')

products = [
    'rain_rate',     # Buoy (truth)
    'GPCP v3.2',
    'GPCP v3.3',
    'ERA5',
    'IMERG v07',
    'MERRA2'
]

monthly_by_region = {}

for region in df['region'].unique():
    dfr = df[df['region'] == region]

    monthly = (
        dfr
        .groupby('year_month')[products]
        .mean()
        .reset_index()
    )

    # Period → Timestamp
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

    monthly_by_region[region] = monthly

regions = list(monthly_by_region.keys())

fig, axes = plt.subplots(
    nrows=len(regions),
    ncols=1,
    figsize=(14, 3.5 * len(regions)),
    sharex=False
)

if len(regions) == 1:
    axes = [axes]

# Product colors (reuse your scheme if already defined)


for ax, region in zip(axes, regions):
    ts = monthly_by_region[region]

    # Buoy (reference)
    ax.plot(
        ts['year_month'],
        ts['rain_rate'],
        lw=2.5,
        color='b',
        label='Buoy'
    )

    # Satellite / reanalysis
    for prod in products[1:]:
        ax.plot(
            ts['year_month'],
            ts[prod],
            lw=2,
            color=product_colors[prod],
            alpha=0.9,
            label=prod
        )

    ax.set_title(region, fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Mean Rainfall (mm/day)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=11)

# Legend (single, clean)
axes[0].legend(
    ncol=3,
    fontsize=11,
    frameon=False
)

axes[-1].set_xlabel('Time', fontsize=13)

plt.tight_layout()
svnme = os.path.join(path_to_plots, 
                     f'Buoy_vs_Satellite_Monthly_Timeseries_Comparison_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# plt.show() 
print('Finished monthly timeseries analysis...')
print("-" * 30 + "\n")

#%% MONTHLY CLIMATOLOGY
df = buoy_sate_daily_rainfall_colasped_df.copy()

# Ensure datetime
df['date'] = pd.to_datetime(df['date'])

# Month index (1–12)
df['month'] = df['date'].dt.month

products = [
    'rain_rate',     # Buoy
    'GPCP v3.2',
    'GPCP v3.3',
    'ERA5',
    'IMERG v07',
    'MERRA2'
]

monthly_clim_by_region = {}

for region in df['region'].unique():
    dfr = df[df['region'] == region]

    clim = (
        dfr
        .groupby('month')[products]
        .mean()
        .reset_index()
    )

    monthly_clim_by_region[region] = clim


regions = list(monthly_clim_by_region.keys())

fig, axes = plt.subplots(
    nrows=len(regions),
    ncols=1,
    figsize=(14, 3.5 * len(regions)),
    sharex=True
)

if len(regions) == 1:
    axes = [axes]


month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

for ax, region in zip(axes, regions):
    clim = monthly_clim_by_region[region]

    # Buoy reference
    ax.plot(
        clim['month'],
        clim['rain_rate'],
        lw=4,
        color='b',
        label='Buoy'
    )

    # Satellite / reanalysis
    for prod in products[1:]:
        ax.plot(
            clim['month'],
            clim[prod],
            lw=4,
            color=product_colors[prod],
            label=prod
        )

    ax.set_title(region, fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Mean Rainfall (mm/day)', fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=11)

# Shared x-label
axes[-1].set_xlabel('Month', fontsize=13)

# One clean legend
axes[0].legend(
    ncol=3,
    fontsize=11,
    frameon=False
)

plt.tight_layout()
svnme = os.path.join(path_to_plots, 
                     f'Buoy_vs_Satellite_Monthly_Climatology_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# plt.show()
print('Finished monthly climatology analysis...')
print("-" * 30 + "\n")
print('Starting distribution of monthly means analysis...')


#%% DISTRIBUTION OF MONTHLY MEANS 
df = buoy_sate_daily_rainfall_colasped_df.copy()
PRODUCT_COLS = {
    "Buoy": "rain_rate",
    "GPCP v3.2": "GPCP v3.2",
    "GPCP v3.3": "GPCP v3.3",
    "ERA5": "ERA5",
    "IMERG v07": "IMERG v07",
    "MERRA2": "MERRA2",
}

df = df.copy()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

monthly = (
    df
    .groupby(["region", "year", "month"])[list(PRODUCT_COLS.values())]
    .mean()
    .reset_index()
)

region = 'ENP'
dfr = monthly[monthly['region'] == region]

products_to_plot = [
    'rain_rate',    # Buoy
    'GPCP v3.2',
    'ERA5',
    'IMERG v07'
]

labels = {
    'rain_rate': 'Buoy',
    'GPCP v3.2': 'GPCP v3.2',
    'ERA5': 'ERA5',
    'IMERG v07': 'IMERG v07'
}

# reshape to long format
long_df = dfr.melt(
    id_vars=['year', 'month'],
    value_vars=products_to_plot,
    var_name='product',
    value_name='monthly_mean'
)

long_df['product'] = long_df['product'].map(labels)

regions = monthly["region"].unique()
nrows = len(regions)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=1,
    figsize=(12, 3.8 * nrows),
    sharex=True
)

if nrows == 1:
    axes = [axes]

for ax, region in zip(axes, regions):
    dfr = monthly[monthly["region"] == region]

    # ---- Long format (NO renaming of original df) ----
    long_df = dfr.melt(
        value_vars=list(PRODUCT_COLS.values()),
        var_name="Product",
        value_name="Monthly Mean Rainfall"
    )

    # ---- Rename rain_rate → Buoy for plotting only ----
    long_df["Product"] = long_df["Product"].replace({
        "rain_rate": "Buoy"
    })

    # ---- Violin plot (distribution) ----
    sns.violinplot(
        data=long_df,
        x="Product",
        y="Monthly Mean Rainfall",
        ax=ax,
        palette=product_colors,
        inner=None,        # turn off internal quartiles
        cut=0,
        linewidth=1
    )

    # ---- Boxplot overlay (IQR + median) ----
    sns.boxplot(
        data=long_df,
        x="Product",
        y="Monthly Mean Rainfall",
        ax=ax,
        width=0.18,        # narrow box
        showcaps=True,
        boxprops=dict(facecolor="none", edgecolor="k", linewidth=1.2),
        whiskerprops=dict(color="k", linewidth=2.5),
        medianprops=dict(color="c", linewidth=2.5),
        showfliers=False
    )

    # ---- Mean marker ----
    sns.pointplot(
        data=long_df,
        x="Product",
        y="Monthly Mean Rainfall",
        ax=ax,
        estimator="mean",
        errorbar=None,
        color="r",
        markers="D",
        markersize=12,
        scale=0.6
    )

    ax.set_title(region, fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Mean Rainfall (mm/day)")
    ax.grid(True, linestyle="--", alpha=0.5)

# axes[-1].set_xlabel("Product")

plt.tight_layout()

svnme = os.path.join(path_to_plots, 
                     f'Distribution_of_Monthly_Means_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
print('Finished distribution of monthly means analysis...')
print("-" * 30 + "\n")
print('Starting distribution of daily means analysis...')
# plt.show()


#%% DISTRIBUTION OF DAILY MEANS
df = buoy_sate_daily_rainfall_colasped_df.copy()

# Ensure datetime
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

PRODUCT_COLS = {
    "Buoy": "rain_rate",
    "GPCP v3.2": "GPCP v3.2",
    "GPCP v3.3": "GPCP v3.3",
    "ERA5": "ERA5",
    "IMERG v07": "IMERG v07",
    "MERRA2": "MERRA2",
}

# Monthly means per region–year–month
monthly = (
    df
    .groupby(["region", "year", "month"])[list(PRODUCT_COLS.values())]
    .mean()
    .reset_index()
)

# Long format (key step)
long_df = monthly.melt(
    id_vars=["region", "year", "month"],
    value_vars=list(PRODUCT_COLS.values()),
    var_name="Product",
    value_name="Monthly Mean Rainfall"
)

# Rename rain_rate → Buoy (ONLY for plotting)
long_df["Product"] = long_df["Product"].replace({
    "rain_rate": "Buoy"
})


regions = long_df["region"].unique()
nrows = len(regions)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=1,
    figsize=(16, 3.8 * nrows),
    sharex=True
)

if nrows == 1:
    axes = [axes]

for ax, region in zip(axes, regions):
    dfr = long_df[long_df["region"] == region]

    sns.boxplot(
        data=dfr,
        x="year",
        y="Monthly Mean Rainfall",
        hue="Product",
        palette=product_colors,
        linewidth=0.8,
        fliersize=1.5,
        ax=ax
    )

    ax.set_title(region, fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Mean Rainfall (mm/day)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Reduce clutter
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=11)

    # One legend only
    if ax != axes[0]:
        ax.legend_.remove()

axes[0].legend(
    ncol=6,
    fontsize=11,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25)
)

axes[-1].set_xlabel("Year", fontsize=13)

plt.tight_layout()

svnme = os.path.join(path_to_plots, 
                     f'Distribution_of_Monthly_Means_per_Year_{cde_run_dte}.png')
# plt.show()
print('Finished distribution of daily means analysis...')
print("-" * 30 + "\n")
print('Starting year to year variability analysis...')

#%%  YEAR TO YEAR VARIABILITY
# Annual mean rainfall (mean of monthly means)
annual = (
    monthly
    .groupby(["region", "year"])[list(PRODUCT_COLS.values())]
    .mean()
    .reset_index()
)

# Long format for plotting
annual_long = annual.melt(
    id_vars=["region", "year"],
    value_vars=list(PRODUCT_COLS.values()),
    var_name="Product",
    value_name="Annual Mean Rainfall"
)

# Rename rain_rate → Buoy (plotting only)
annual_long["Product"] = annual_long["Product"].replace({
    "rain_rate": "Buoy"
})


regions = annual_long["region"].unique()
nrows = len(regions)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=1,
    figsize=(14, 3.5 * nrows),
    sharex=False
)

if nrows == 1:
    axes = [axes]

for ax, region in zip(axes, regions):
    dfr = annual_long[annual_long["region"] == region]

    for product, color in product_colors.items():
        dff = dfr[dfr["Product"] == product]

        ax.plot(
            dff["year"],
            dff["Annual Mean Rainfall"],
            lw=4 if product == "Buoy" else 4,
            color=color,
            alpha=0.95,
            label=product
        )

    ax.set_title(region, fontsize=14, fontweight="bold")
    ax.set_ylabel("Annual Mean Rainfall (mm/day)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=11)

# One clean legend
axes[0].legend(
    ncol=4,
    fontsize=11,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.5, 1.25)
)

axes[-1].set_xlabel("Year", fontsize=13)

plt.tight_layout()

svnme = os.path.join(path_to_plots, 
                     f'Annual_Mean_Rainfall_TimeSeries_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# plt.show()
print('Finished year to year variability analysis...')
print("-" * 30 + "\n")
print('All analyses completed successfully.')