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

path_to_ocRain = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/OceanRain'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_1998_2024'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2000_2020'

path_to_imerg_v06 = r'/ra1/pubdat/GPM/imerg6'

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
# ALIGN ALL DATASETS IN TIME
# mindate,maxdate = gpcp_ds_v3pt2_xr.time.min().values, gpcp_ds_v3pt2_xr.time.max().values
mindate,maxdate = imerg_v06_ds_xr.time.min().values, imerg_v06_ds_xr.time.max().values

# select time range for all datasets
gpcp_ds_v3pt2_al = gpcp_ds_v3pt2_xr.sel(time=slice(mindate, maxdate)).compute()#.chunk({'time': -1})
gpcp_ds_v3pt3_al = gpcp_ds_v3pt3_xr.sel(time=slice(mindate, maxdate)).compute()#.chunk({'time': -1})
era5_ds_al = era5_ds_xr.sel(valid_time=slice(mindate, maxdate))#.compute()#.chunk({'valid_time': -1})
imerg_v07_al = imerg_v07_ds_xr.sel(time=slice(mindate, maxdate))#.compute()#.chunk({'time': -1})
imerg_v06_al = imerg_v06_ds_xr.sel(time=slice(mindate, maxdate))#.compute()#.chunk({'time': -1})
mer2_ds_al = mer2_ds_xr.sel(time=slice(mindate, maxdate))#.chunk({'time': -1})

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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load OceanRain data file
# Load the .npz file using numpy
ocRain = np.load(os.path.join(path_to_ocRain, "OceanRAIN_MINUTE_coordinates_and_data_Kingsley_20260210.npz"))

# Access the keys in the .npz file
keys = ocRain.files
print("Keys in the .npz file:", keys)
# Create a DataFrame from the .npz file
ocRain_df = pd.DataFrame({key: ocRain[key] for key in keys})

print("Data files listed and datasets loaded.")

gc.collect()
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
dtmin,dtmax = pd.to_datetime(mindate).date(),pd.to_datetime(maxdate).date() # satellite data's min max date range
for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
        # store Buoy and GPCP dataframes
    region_buoy_sate_dfs = []  

    # LOAD Buoy DATA
    for b, b_file in enumerate(buoy_files):
        b_df, b_lat, b_lon = grab_Buoy_data_df(b_file)
        # subset b_df by time for wihtin dtmin,dtmax date range
        b_df = b_df[(b_df['time'].dt.date >= dtmin) & (b_df['time'].dt.date <= dtmax)]
        if b_df.empty:
            print(f"Warning: Buoy file {os.path.basename(b_file)} has no data within the date range {dtmin} to {dtmax}")
            continue
        df_t_min, df_t_max = b_df['time'].min().date(), b_df['time'].max().date()
        if b % 5 == 0:
            print(f"Processing Buoy file: {os.path.basename(b_file)}")      
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -

        # Process GPCP v3.2 - Memory efficient version
        # gpcpv3pt2_df = extract_buoy_satellite_data_memory_efficient(
        #     gpcp_ds_v3pt2_xr, b_lat, b_lon, 'GPCP v3.2', 'precip',chunk_size=300
        # )
        
        # if gpcpv3pt2_df is None:
        #     print("Warning: Failed to extract GPCP v3.2 data, creating empty dataframe")
        #     b_rain_gpcpv3pt2_df = pd.DataFrame(columns=['date', 'GPCP v3.2'])

        gpcpv3pt2_df = extract_point_timeseries_to_df(
                gpcp_ds_v3pt2_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=["precip",'probability_liquid_phase'],          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="lat",
                lon_name="lon",
            )
        
        if gpcpv3pt2_df.empty:
            print("Warning: Failed to extract GPCP v3.2 data, creating empty dataframe")
            gpcpv3pt2_df = pd.DataFrame(columns=['date', 'GPCP v3.2', 'PLP_GPCP v3.2'])
        else:
            gpcpv3pt2_df = gpcpv3pt2_df[['time','precip', 'probability_liquid_phase']].copy()
            gpcpv3pt2_df.columns = ['time', 'GPCP v3.2', 'PLP_GPCP v3.2']     
            gpcpv3pt2_df['date'] = gpcpv3pt2_df['time'].dt.date

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        # Process GPCP v3.3 - Memory efficient version
        # gpcpv3pt3_df = extract_buoy_satellite_data_memory_efficient(
        #     gpcp_ds_v3pt3_xr, b_lat, b_lon, 'GPCP v3.3', 'precip', chunk_size=300
        # )
        
        # if gpcpv3pt3_df is None:
        #     print("Warning: Failed to extract GPCP v3.3 data, creating empty dataframe")
        #     gpcpv3pt3_df = pd.DataFrame(columns=['date', 'GPCP v3.3'])

        
        gpcpv3pt3_df = extract_point_timeseries_to_df(
                gpcp_ds_v3pt3_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=["precip"],          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="lat",
                lon_name="lon",
            )
        
        if gpcpv3pt3_df.empty:
            print("Warning: Failed to extract GPCP v3.3 data, creating empty dataframe")
            gpcpv3pt3_df = pd.DataFrame(columns=['date', 'GPCP v3.3'])
        else:
            gpcpv3pt3_df = gpcpv3pt3_df[['time','precip']].copy()
            gpcpv3pt3_df.columns = ['time', 'GPCP v3.3']
            gpcpv3pt3_df['date'] = gpcpv3pt3_df['time'].dt.date


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process ERA5 data with Buoy - Memory efficient version 
        # era5_df = extract_buoy_satellite_data_memory_efficient(
        #     era5_ds_xr, b_lat, b_lon, 'ERA5', 'tp', chunk_size=300
        # )
        # era5_df.drop(columns=['number','spatial_ref'], inplace=True)  # drop redundant column
        # era5_df.rename(columns={'tp': 'ERA5'}, inplace=True)
        
        # if era5_df is None:
        #     print("Warning: Failed to extract ERA5 data, creating empty dataframe")
        #     era5_df = pd.DataFrame(columns=['date', 'ERA5'])

        era5_df = extract_point_timeseries_to_df(
                era5_ds_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=["tp"],          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="valid_time",
                lat_name="y",
                lon_name="x",
            )
        if era5_df.empty:
            print("Warning: Failed to extract ERA5 data, creating empty dataframe")
            era5_df = pd.DataFrame(columns=['date', 'ERA5'])
        else:
            era5_df = era5_df[['valid_time','tp']].copy()
            era5_df.columns = ['time', 'ERA5']
            era5_df['date'] = era5_df['time'].dt.date
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        
        # Process IMERG data with Buoy - Memory efficient version
        # imerg_v07_df = extract_buoy_satellite_data_memory_efficient(
        #     imerg_v07_ds_xr, b_lat, b_lon, 'IMERG v07', None, chunk_size=300  # Reduced from 100 to 50 for IMERG
        # )

        # imerg_v07_df.drop(columns=['spatial_ref'], inplace=True)  # drop redundant column
        # imerg_v07_df.rename(columns={'precipitation': 'IMERG v07'}, inplace=True)
        
        # if imerg_v07_df is None:
        #     print("Warning: Failed to extract IMERG data, creating empty dataframe")
        #     imerg_v07_df = pd.DataFrame(columns=['date', 'IMERG v07'])

        imerg_v07_df = extract_point_timeseries_to_df(
                imerg_v07_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=None,          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="lat",
                lon_name="lon",
            )
        if imerg_v07_df.empty:
            print("Warning: Failed to extract IMERG data, creating empty dataframe")
            imerg_v07_df = pd.DataFrame(columns=['date', 'IMERG v07'])
        else:
            imerg_v07_df = imerg_v07_df[['time','precipitation']].copy()
            imerg_v07_df.columns = ['time', 'IMERG v07']
            imerg_v07_df['date'] = imerg_v07_df['time'].dt.date
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -    
        imerg_v06_df = extract_point_timeseries_to_df(
                imerg_v06_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=None,          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="lat",
                lon_name="lon",
            )
        if imerg_v06_df.empty:
            print("Warning: Failed to extract IMERG v06 data, creating empty dataframe")
            imerg_v06_df = pd.DataFrame(columns=['date', 'IMERG v06'])
        else:
            imerg_v06_df = imerg_v06_df[['time','precipitation']].copy()
            imerg_v06_df.columns = ['time', 'IMERG v06']
            imerg_v06_df['date'] = imerg_v06_df['time'].dt.date
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        # Process MERRA2 data with Buoy - Memory efficient version
        # merra2_df = extract_buoy_satellite_data_memory_efficient(
        #     mer2_ds_xr, b_lat, b_lon, 'MERRA2', None, chunk_size=300
        # )
        
        # merra2_df.drop(columns=['spatial_ref'], inplace=True)
        # merra2_df.rename(columns={'PRECTOTCORR':'MERRA2'}, inplace=True)
        
        # if merra2_df is None:
        #     print("Warning: Failed to extract MERRA2 data, creating empty dataframe")
        #     merra2_df = pd.DataFrame(columns=['date', 'MERRA2'])

        merra2_df = extract_point_timeseries_to_df(
                mer2_ds_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=None,          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="y",
                lon_name="x",
            )
        if merra2_df.empty:
            print("Warning: Failed to extract MERRA2 data, creating empty dataframe")
            merra2_df = pd.DataFrame(columns=['date', 'MERRA2'])
        else:
            merra2_df = merra2_df[['time','PRECTOTCORR']].copy()
            merra2_df.columns = ['time', 'MERRA2']
            merra2_df['date'] = merra2_df['time'].dt.date
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

        # merge IMERG data - merge on 'date' column instead of index
        b_df_combined_rain = b_df_combined_rain.merge(
            imerg_v06_df[['date', 'IMERG v06']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_IMERG v06')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['IMERG v06_IMERG v06', 'date_IMERG v06']], 
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
                                                                        'IMERG_v06',
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
pal_sate_daily_mean_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'pal_sate_daily_mean_from_all_regions_and_all_tracks_20260128.pkl'))  # .to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_mean_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

pal_sate_daily_rainfall_colasped_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'pal_sate_daily_rainfall_from_all_regions_and_all_tracks_20260128.pkl'))

# Create scatter plots for PAL vs satellite products
scatter_fig = plot_satellite_vs_groundtruth(pal_sate_daily_mean_df,
                                            truth_col='rain_rate',
                                            product_cols=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'],
                                            product_labels=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'],
                                            truth_label='PAL Observations',
                                            max_val=18,
                                            ticks=(0, 6, 12, 18),
                                            figsize_per_col=6,
                                            figsize_per_row=5,
                                            region_markers=PAL_region_markers,
                                            region_colors=PAL_region_colors,
                                            region_labels=PAL_REGION_NAMES,
                                            savepath=os.path.join(path_to_plots, f'PAL_vs_Satellite_Comparison_{cde_run_dte}.png'))


#%% THE CATEGORICAL METRICS
# PAL BASED ASSESSMENT
# REGION BY REGION CAT METRICS

region_based_cat_metrics = {}
region_based_qt_metrics = {}

for region_name in pal_sate_daily_rainfall_colasped_df['region'].unique():

    region_df = pal_sate_daily_rainfall_colasped_df[pal_sate_daily_rainfall_colasped_df['region'] == region_name]

    # region_name = region_df['region'].unique()[0]

    print(f"Processing region: {region_name}")

    for product in ['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2']:
        forcast = region_df[product]
        observed = region_df['rain_rate']

        # Calculate the categorical metrics
        reg_cat_met = categorical_stats(forcast, observed, 1.0)
        # Calculate the quantitative metrics
        reg_qt_met = calculate_metrics(region_df, 'rain_rate', product)

        # Store the metrics in the dictionary
        region_based_cat_metrics.setdefault(region_name, {})[product] = reg_cat_met
        region_based_qt_metrics.setdefault(region_name, {})[product] = reg_qt_met


# The plot

products = ["GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5",  "MERRA2"]

# cate metrics
fig = plot_categorical_metrics_by_region(
    metrics_dict=region_based_cat_metrics,
    products=products,
    product_colors=product_colors,
    metrics=("POD", "FAR", "Bias", "HSS"),
    region_labels=PAL_REGION_NAMES,
)
svnme = os.path.join(path_to_plots, 
                     f'PAL_Satellite_Categorical_Metrics_by_Region_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)

# quant metrics
fig = plot_categorical_metrics_by_region(
    metrics_dict=region_based_qt_metrics,
    products=products,
    product_colors=product_colors,
    metrics=("CC", "RMSE", "MAE", "Bias"),
    region_labels=PAL_REGION_NAMES,
)
svnme = os.path.join(path_to_plots, 
                     f'PAL_Satellite_Quantitative_Metrics_by_Region_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - 
# METRICS BY RAINFALL INTENSITY
rainfall_bins = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] # 
products = ['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2']

cat_metrics = ['POD', 'FAR', 'Bias', 'HSS']
qt_metrics  = ['CC', 'RMSE', 'MAE', 'RB']   # <- as you want (Bias handled as RB in %)

# df_colapsed = pd.concat(regional_PAL_sate_dfs_daily_lst, ignore_index=True)
df_all = pal_sate_daily_rainfall_colasped_df.copy()

# ------------------------------------------------------------
# COMPUTE METRICS BY PRODUCT
# ------------------------------------------------------------
qt_met_by_prdt = {}
cat_met_by_prdt = {}

for product in products:

    cat_met_prdt = pd.DataFrame(index=rainfall_bins, columns=cat_metrics, dtype=float)
    qt_met_prdt  = pd.DataFrame(index=rainfall_bins, columns=qt_metrics,  dtype=float)

    forecast = df_all[product]
    observed = df_all['rain_rate']

    for r_bin in rainfall_bins:
        # ---- categorical ----
        cat_mets = categorical_stats(forecast, observed, r_bin)
        cat_met_prdt.loc[r_bin, 'POD']  = cat_mets.get('POD',  np.nan)
        cat_met_prdt.loc[r_bin, 'FAR']  = cat_mets.get('FAR',  np.nan)
        cat_met_prdt.loc[r_bin, 'Bias'] = cat_mets.get('Bias', np.nan)
        cat_met_prdt.loc[r_bin, 'HSS']  = cat_mets.get('HSS',  np.nan)

        # ---- quantitative (only days where observed >= bin) ----
        # bin_df = df_all[df_all['rain_rate'] >= r_bin]
        bin_df = df_all[(df_all['rain_rate'] >= r_bin) & (df_all[product] >=r_bin)]
        qt_mets = calculate_metrics(bin_df, 'rain_rate', product)

        qt_met_prdt.loc[r_bin, 'CC']   = qt_mets.get('CC',   np.nan)
        qt_met_prdt.loc[r_bin, 'RMSE'] = qt_mets.get('RMSE', np.nan)
        qt_met_prdt.loc[r_bin, 'MAE']  = qt_mets.get('MAE',  np.nan)
        qt_met_prdt.loc[r_bin, 'RB']   = qt_mets.get('Bias', np.nan)  # bias [%]

    cat_met_by_prdt[product] = cat_met_prdt
    qt_met_by_prdt[product]  = qt_met_prdt
    
# ------------------------------------------------------------
# PLOTTING: 4x2 layout (left=categorical, right=quantitative)
# ------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=4, ncols=2,
    figsize=(14, 14),
    sharex=True
)

# ---- LEFT COLUMN: CATEGORICAL ----
for i, met in enumerate(cat_metrics):
    ax = axes[i, 0]

    for product, dmet in cat_met_by_prdt.items():
        ax.plot(
            rainfall_bins,
            dmet.loc[rainfall_bins, met].astype(float),
            marker='o',
            linewidth=2.2,
            markersize=7,
            color=product_colors[product],
            label=product if i == 0 else None
        )

    ax.set_ylabel(met, fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xticks(rainfall_bins)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(labelsize=14)

# ---- RIGHT COLUMN: QUANTITATIVE ----
qt_labels = {
    'CC':   'CC',
    'RMSE': 'RMSE [mm/day]',
    'MAE':  'MAE [mm/day]',
    'RB':   'Bias [%]'
}

for i, met in enumerate(qt_metrics):
    ax = axes[i, 1]

    for product, dmet in qt_met_by_prdt.items():
        ax.plot(
            rainfall_bins,
            dmet.loc[rainfall_bins, met].astype(float),
            marker='o',
            linewidth=2.2,
            markersize=7,
            color=product_colors[product],
            label=product if i == 0 else None
        )

    ax.set_ylabel(qt_labels.get(met, met), fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xticks(rainfall_bins)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(labelsize=14)

# ---- X labels (bottom row only) ----
for ax in axes[-1, :]:
    ax.set_xlabel('Rain Rate (mm/day)', fontsize=18, fontweight='bold')

# ---- Optional column titles ----
axes[0, 0].set_title('Categorical Metrics', fontsize=18, fontweight='bold')
axes[0, 1].set_title('Quantitative Metrics', fontsize=18, fontweight='bold')

# ---- Single legend below figure ----
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.subplots_adjust(bottom=0.12)

fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    fontsize=18,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.show()
svnme = os.path.join(path_to_plots, 
                     f'PAL_Satellite_Metrics_by_Rainfall_Intensity_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# plt.show()

#%%  PDF Assessment relative to PAL
bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']

# Compute PDF elements for all datasets
pal_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'rain_rate', bin_values)
img_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'IMERG v07', bin_values)
gpcp_v3pt2_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'GPCP v3.2', bin_values)
gpcp_v3pt3_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'GPCP v3.3', bin_values)
era5_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'ERA5', bin_values)
merra2_pdfc_pdfv = compute_pdf_elements(pal_sate_daily_rainfall_colasped_df, 'MERRA2', bin_values)
# ============================================================
# PDFv / PDFc by Rainfall Intensity (Figure-5 style)
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=500)

lw = 4

# --- X axis: use actual bin values ---
x = pal_pdfc_pdfv['bin'].values#bin_values#[:-1]   # last edge has no PDF value

# ------------------------------------------------------------
# PDFv (Volume-based PDF)  —— ACTIVE
# ------------------------------------------------------------
ax.plot(x, pal_pdfc_pdfv['pdfv'], lw=lw, color='b', ls=':', label='PAL')
ax.plot(x, merra2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['MERRA2'], label='MERRA2')
ax.plot(x, era5_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['ERA5'], label='ERA5')
ax.plot(x, img_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['IMERG v07'], label='IMERG v07')
ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.2'], label='GPCP v3.2')
ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.3'], label='GPCP v3.3')

# ------------------------------------------------------------
# PDFc (Count-based PDF)  —— OPTIONAL (commented)
# ------------------------------------------------------------
# ax.plot(x, pal_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['PAL'], label='PAL (PDFc)')
# ax.plot(x, img_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['IMERG v07'], label='IMERG v07 (PDFc)')
# ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['GPCP v3.2'], label='GPCP v3.2 (PDFc)')
# ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['GPCP v3.3'], label='GPCP v3.3 (PDFc)')
# ax.plot(x, era5_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['ERA5'], label='ERA5 (PDFc)')

# ------------------------------------------------------------
# Formatting (paper-quality)
# ------------------------------------------------------------
ax.set_xscale('log')
ax.set_xlabel('Rain Rate [mm day$^{-1}$]', fontsize=18, fontweight='bold')
ax.set_ylabel('PDF (%)', fontsize=18, fontweight='bold')

# ax.set_xticks(bin_values)
# ax.set_xticklabels(bin_labels)

bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

ax.set_xscale('log')
ax.xaxis.set_major_locator(FixedLocator(bin_values))

ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, pos: "0.5" if abs(v-0.5) < 1e-12 else f"{int(round(v))}"
))

# optional: remove minor tick marks entirely (cleaner for paper)
ax.xaxis.set_minor_locator(FixedLocator([]))

# (optional) turn off minor tick labels so only these show
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='x', which='major', labelsize=15)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=15, width=1.5, length=7)

ax.grid(True, which='major', linestyle='--', alpha=0.6)
ax.legend(fontsize=15, frameon=False)

plt.tight_layout()
svnme = os.path.join(path_to_plots, 
                     f'PAL_Satellite_PDF_Comparison_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)


#%%
print('Starting buoy-based assessment...')
# THE BUOY BASED ASSESSMENT
buoy_sate_daily_mean_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_mean_from_all_regions_and_all_IDs_20260216.pkl'))  

# Create scatter plots for Buoy vs satellite products
scatter_fig_buoy = plot_satellite_vs_groundtruth(buoy_sate_daily_mean_df,
                                            truth_col='rain_rate',
                                            product_cols=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'], # 
                                            product_labels=['GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'], # 
                                            truth_label='Buoy Observations',
                                            max_val=15,
                                            ticks=(0, 5, 10, 15),
                                            figsize_per_col=6,
                                            figsize_per_row=5,
                                            region_markers=Buoy_region_markers,
                                            region_colors=Buoy_region_colors,
                                            region_labels=Buoy_REGION_NAMES,
                                            savepath=os.path.join(path_to_plots, f'Buoy_vs_Satellite_Comparison_{cde_run_dte}.png'))

print('Finished buoy-based assessment...')
print("-" * 30 + "\n")

gc.collect()
#%%
print('Starting monthly timeseries analysis...')
# MONTHLY TIMESERIES ANALYSIS
buoy_sate_daily_rainfall_colasped_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260216.pkl'))

df = buoy_sate_daily_rainfall_colasped_df.copy()

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

# df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()

# Ensure datetime
df['date'] = pd.to_datetime(df['date'])

# Year–month
df['year_month'] = df['date'].dt.to_period('M')


monthly_by_region = {}

for region in list(Buoy_region_markers.keys()):
    dfr = df[df['region'] == region].copy()

    monthly = (
        dfr
        .groupby('year_month')[products]
        .mean()
        .reset_index()
    )

    # Period → Timestamp
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

    # ---- NEW: enforce continuous monthly index ----
    full_index = pd.date_range(
        start=monthly['year_month'].min(),
        end=monthly['year_month'].max(),
        freq='MS'   # Month Start
    )

    monthly = (
        monthly
        .set_index('year_month')
        .reindex(full_index)
        .rename_axis('year_month')
        .reset_index()
    )

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

    ax.set_title(Buoy_REGION_NAMES[region], fontsize=16, fontweight='bold')#region, fontsize=14, fontweight='bold')
    ax.set_ylabel('Rainfall [mm day$^{-1}$]', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=11)

# Legend (single, clean)
axes[0].legend(
    ncol=3,
    fontsize=16,
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
gc.collect()
#%% TIME SEREIES ANALYSIS OF 6 MONTHS RUNNING MEAN
print('Starting 6 month running mean year to year variability analysis...')

# df = buoy_sate_daily_rainfall_colasped_df.copy()

# df["date"] = pd.to_datetime(df["date"])
# df["month"] = df["date"].dt.month

# df["year"] = df["date"].dt.year.astype(int)

# # df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()

# # ---- Define half-year flag and anchor month ----
# df["half"] = np.where(df["month"] <= 6, "H1", "H2")
# df["anchor_month"] = np.where(df["half"] == "H1", 6, 12)

# # ---- Semiannual mean ----
# semiannual = (
#     df
#     .groupby(["region", "year", "half", "anchor_month"])[products]
#     .mean()
#     .reset_index()
# )

# # ---- Create timestamp (June or December) ----
# semiannual["time"] = pd.to_datetime(
#     dict(
#         year=semiannual["year"],
#         month=semiannual["anchor_month"],
#         day=1
#     )
# )

# semiannual_by_region = {}

# for region in Buoy_region_markers.keys():#semiannual["region"].unique():
#     dfr = semiannual[semiannual["region"] == region].copy()

#     full_index = pd.date_range(
#         start=dfr["time"].min(),
#         end=dfr["time"].max(),
#         freq="6MS"   # 6-month frequency
#     )

#     dfr = (
#         dfr
#         .set_index("time")
#         .reindex(full_index)
#         .rename_axis("time")
#         .reset_index()
#     )

#     semiannual_by_region[region] = dfr

# regions = list(semiannual_by_region.keys())

# fig, axes = plt.subplots(
#     nrows=len(regions),
#     ncols=1,
#     figsize=(14, 3.5 * len(regions)),
#     sharex=False
# )

# if len(regions) == 1:
#     axes = [axes]

# for ax, region in zip(axes, regions):
#     ts = semiannual_by_region[region]

#     # Buoy
#     ax.plot(
#         ts["time"],
#         ts["rain_rate"],
#         lw=2.8,
#         color="blue",
#         label="Buoy"
#     )

#     # Satellite / reanalysis
#     for prod in products[1:]:
#         ax.plot(
#             ts["time"],
#             ts[prod],
#             lw=2,
#             color=product_colors[prod],
#             alpha=0.9,
#             label=prod
#         )

#     ax.set_title(region, fontsize=14, fontweight="bold")
#     ax.set_ylabel("6-month Mean Rainfall [mm day$^{-1}$]")
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.tick_params(axis="both", labelsize=11)

# # Single legend
# axes[0].legend(
#     ncol=3,
#     fontsize=11,
#     frameon=False
# )

# axes[-1].set_xlabel("Time (June / December anchors)", fontsize=13)

# plt.tight_layout()
# # plt.show()

# svnme = os.path.join(path_to_plots, 
#                      f'Buoy_vs_Satellite_Semiannual_Timeseries_Comparison_{cde_run_dte}.png')
# fig.savefig(svnme, dpi=300)
# print('Finished 6 month running mean analysis...')
# print("-" * 30 + "\n")

#--------------------------------------------------------------------------------
# calculate and and plot 6 months running mean
df = buoy_sate_daily_rainfall_colasped_df.copy()

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)
# Year–month
df['year_month'] = df['date'].dt.to_period('M')


seven_months_rolling_mean_by_region = {}

for region in list(Buoy_region_markers.keys()):
    dfr = df[df['region'] == region].copy()

    monthly = (
        dfr
        .groupby('year_month')[products]
        .mean()
        .reset_index()
    )

    mnth = monthly.copy()

    mnth = mnth.sort_values('year_month')

    for p in products:
        mnth[p + "_rm13"] = (
            mnth[p]
            .rolling(window=7, center=True, min_periods=7)
            .mean()
        )

    seven_months_rolling_mean_by_region[region] = mnth[['year_month'] + [p + "_rm13" for p in products]]


fig, axes = plt.subplots(
    nrows=len(regions),
    ncols=1,
    figsize=(14, 3.5 * len(regions)),
    sharex=False
)

if len(regions) == 1:
    axes = [axes]

for i, (ax, region) in enumerate(zip(axes, regions)):
    ts = seven_months_rolling_mean_by_region[region].copy()

    # Convert Period -> datetime (month start)
    x = ts["year_month"].dt.to_timestamp(how="start")

    # Buoy
    ax.plot(
        x,
        ts["rain_rate_rm13"],
        lw=2.8,
        color="blue",
        label="Buoy"
    )

    # Satellite / reanalysis
    for prod in products[1:]:
        ax.plot(
            x,
            ts[prod + "_rm13"],
            lw=2,
            color=product_colors[prod],
            alpha=0.9,
            label=prod
        )

    ax.set_title(Buoy_REGION_NAMES[region], fontsize=16, fontweight="bold")#region, fontsize=14, fontweight="bold")

    # y-axis label fontsize = 18
    ax.set_ylabel("Rainfall [mm day$^{-1}$]", fontsize=16)

    ax.grid(True, linestyle="--", alpha=0.6)

    # all tick labels fontsize = 15
    ax.tick_params(axis="both", labelsize=15)

    # ---- custom y ticks per panel ----
    if i == 0:
        ax.set_yticks([0, 3, 6, 9])
    elif i == 1:
        ax.set_yticks([4, 6, 8, 10, 12])
    elif i == 2:
        ax.set_yticks([4, 6, 8, 10, 12])
    elif i == 3:
        ax.set_yticks([0, 3, 6, 9])

# Single legend
axes[0].legend(ncol=3, fontsize=15, frameon=False)

plt.tight_layout()

svnme = os.path.join(
    path_to_plots,
    f"Buoy_vs_Satellite_6_Month_Running_Mean_{cde_run_dte}.png"
)
fig.savefig(svnme, dpi=300)

print('Finished 6 month running mean analysis...')
print("-" * 30 + "\n")
gc.collect()
#%% MONTHLY CLIMATOLOGY

df = buoy_sate_daily_rainfall_colasped_df.copy()
df["date"] = pd.to_datetime(df["date"])
# df['month'] = df['date'].dt.month.astype(int)
df["year"] = df["date"].dt.year.astype(int)
# df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()

products = [
    "rain_rate",
    "GPCP v3.2",
    "GPCP v3.3",
    "ERA5",
    "IMERG v07",
    "MERRA2",
]

monthly_clim_by_region = {}

for region in df["region"].unique():
    dfr = df[df["region"] == region].copy()

    # # 1) collapse multiple IDs -> ONE value per day for the region
    # daily_region = (
    #     dfr.groupby("date")[products]
    #        .mean()
    #        .reset_index()
    # )

    # 2) climatological monthly cycle (mean across years for each calendar month)
    # daily_region["month"] = daily_region["date"].dt.month
    dfr["month"] = dfr["date"].dt.month
    clim = (
        dfr.groupby("month")[products]
                    .mean()
                    .reset_index()
    )

    monthly_clim_by_region[region] = clim

regions = list(Buoy_region_markers.keys())#list(monthly_clim_by_region.keys())

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

# -----------------------------
# different method
# -----------------------------
# df = buoy_sate_daily_rainfall_colasped_df.copy()
# products = ["rain_rate", "GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"]

# # Compute equal-station-weight climatology
# monthly_clim_by_region, station_month = compute_monthly_climatology_equal_station_weight(
#     df=df,
#     products=products,
#     id_col="ID",
#     region_col="region",
#     date_col="date",
# )

regions = list(Buoy_region_markers.keys())

# Plot
svnme = os.path.join(
    path_to_plots,
    f"Buoy_vs_Satellite_Monthly_Climatology_equalStationWeight_{cde_run_dte}.png"
)

fig = plot_monthly_climatology_stack(
    monthly_clim_by_region=monthly_clim_by_region,
    regions=regions,
    products=products,
    product_colors=product_colors,
    figsize=(14, 3.5 * len(regions)),
    month_as_numbers=True,   # <-- month numbers as you prefer
    savepath=svnme
)

print('Finished monthly climatology analysis...')
print("-" * 30 + "\n")
print('Starting distribution of monthly means analysis...')


#-----------------------------------------------------------------------------
# A SIMILAR MONTHLY CLIMATOLY IN A 2BY2 SUBPLOTS
regions = list(Buoy_region_markers.keys())  # or whatever order you want


plot_monthly_climatology_2x2(
    monthly_clim_by_region=monthly_clim_by_region,
    regions=regions,
    region_labels=Buoy_REGION_NAMES,
    products=products,
    product_colors=product_colors,
    figsize=(12, 9),
    lw=3.5,
    ncol_legend=3
)
svnme = os.path.join(path_to_plots, 
                     f'Buoy_vs_Satellite_Monthly_Climatology_2x2_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)
# plt.show()

#-----------------------------------------------------------------------------
# AMP AND PHASE SCORES
scores = amp_and_phase_scores(monthly_clim_by_region)
scores.sort_values(["region","amp_ratio"])

fig = plot_amp_phase_heatmaps(scores)

# save if you want
svnme = os.path.join(path_to_plots, f"SeasonalCycle_AmpRatio_PhaseLag_Heatmaps_{cde_run_dte}.png")
fig.savefig(svnme, dpi=300, bbox_inches="tight")


#-----------------------------------------------------------------------------
# DIFFERENCES IN MONTHLY MEANS
regions = list(Buoy_region_markers.keys())

# Build anomalies
monthly_anom_by_region = make_monthly_clim_anoms(
    monthly_clim_by_region,
    buoy_col="rain_rate",
    products=products
)

# Plot + save (handles multiple figures if >4 regions)
for idx, fig in enumerate(
    plot_monthly_climatology_anoms_2x2(
        monthly_anom_by_region=monthly_anom_by_region,
        regions=regions,
        region_labels=Buoy_REGION_NAMES,
        products=products,
        product_colors=product_colors,
        figsize=(12, 9),
        lw=3.2,
        ncol_legend=3,
        ylim=None  # or e.g. (-3, 3)
    )
):
    svnme = os.path.join(
        path_to_plots,
        f"Buoy_vs_Satellite_Monthly_Climatology_ANOM_2x2_{cde_run_dte}_p{idx+1}.png"
    )
    fig.savefig(svnme, dpi=300)
    plt.close(fig)

#%% DISTRIBUTION OF MONTHLY MEANS 
df = buoy_sate_daily_rainfall_colasped_df.copy()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

# df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()
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

regions = Buoy_region_markers.keys()#monthly["region"].unique()
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

    ax.set_title(Buoy_REGION_NAMES[region], fontsize=14, fontweight="bold")#region, fontsize=14, fontweight="bold")
    ax.set_ylabel("Rainfall [mm day$^{-1}$]",fontsize=15)
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
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()
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


regions = Buoy_region_markers.keys()#long_df["region"].unique()
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
    ax.set_ylabel("Monthly Mean Rainfall [mm day$^{-1}$]")
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
# ---- Annual mean rainfall (mean of monthly means) ----
annual = (
    monthly
    .groupby(["region", "year"])[list(PRODUCT_COLS.values())]
    .mean()
    .reset_index()
)

# Long format
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

regions = Buoy_region_markers.keys()#annual_long["region"].unique()

fig, axes = plt.subplots(
    nrows=len(regions),
    ncols=1,
    figsize=(14, 3.5 * len(regions)),
    sharex=False
)

if len(regions) == 1:
    axes = [axes]

for ax, region in zip(axes, regions):

    dfr = annual_long[annual_long["region"] == region]

    # ---- FULL year range for this region ----
    year_min = dfr["year"].min()
    year_max = dfr["year"].max()
    full_years = pd.Index(range(year_min, year_max + 1), name="year")

    for product, color in product_colors.items():

        dff = (
            dfr[dfr["Product"] == product]
            .set_index("year")
            .reindex(full_years)          # <<< KEY STEP
            .reset_index()
        )

        ax.plot(
            dff["year"],
            dff["Annual Mean Rainfall"],
            lw=4 if product == "Buoy" else 3,
            color=color,
            alpha=0.95,
            label=product
        )

    ax.set_title(region, fontsize=14, fontweight="bold")
    ax.set_ylabel("Annual Mean Rainfall [mm day$^{-1}$]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=11)

# ---- Single clean legend ----
axes[0].legend(
    ncol=5,
    fontsize=11,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25)
)

axes[-1].set_xlabel("Year", fontsize=13)

plt.tight_layout()

svnme = os.path.join(
    path_to_plots,
    f'Annual_Mean_Rainfall_TimeSeries_GapAware_{cde_run_dte}.png'
)
fig.savefig(svnme, dpi=300)


#-----------------------------------------------------------------------------
# 2x2 subplots
df = buoy_sate_daily_rainfall_colasped_df.copy()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()
products = ["rain_rate","GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2"]

annual_df = pd.concat(
    [
        annual_mean_with_coverage(
            df,
            products=products,
            region=reg,
            min_days=250,          # products threshold
            min_days_ref=300,      # buoy threshold (ref only)
            ref="rain_rate",
        )
        for reg in list(Buoy_region_markers.keys())
    ],
    ignore_index=True
).sort_values(["region", "year"])

regions = ["ENP","WNP","IND","ATL"]  # make sure this is defined

fig = plot_interannual_variability_2x2_gapaware(
    annual_df=annual_df,
    regions=regions,
    products=products,
    product_colors=product_colors,
    ref="rain_rate",
    min_days_ref=300,         # matches what annual_df used for buoy
    max_gap_years=1,
    add_trend_band=True,
    n_boot=2000,
    ci=95
)

fig.savefig(os.path.join(path_to_plots, 
                         f"Interannual_variability_2x2_{cde_run_dte}.png"), dpi=300, bbox_inches="tight")

print("Finished gap-aware year-to-year variability analysis.")
print("-" * 30)

#-----------------------------------------------------------------------------
products = ["rain_rate","GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2"]

monthly_df = pd.concat(
    [
        monthly_mean_with_coverage(
            buoy_sate_daily_rainfall_colasped_df,
            products=products,
            region=reg,
            min_days=20,
            ref="rain_rate",
            min_days_ref=25
        )
        for reg in Buoy_region_markers.keys()
    ],
    ignore_index=True
)

# limit to 2000–2020 (monthly)
monthly_df["date"] = pd.to_datetime(monthly_df["date"])
monthly_df = monthly_df[
    (monthly_df["date"] >= "2000-01-01") & (monthly_df["date"] <= "2020-12-31")
].copy()


fig = plot_2x2_annual_rm13_with_trends(
    df_raw=buoy_sate_daily_rainfall_colasped_df,
    regions=["ENP","WNP","IND","ATL"],
    products=["rain_rate","GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2"],
    product_colors=product_colors,
    ref="rain_rate",
    region_names=Buoy_REGION_NAMES,
    year_min=2000,
    year_max=2020,
    xtick_step=2,
    show_ci_band=False  # set True if you want the band back
)



#%%
#%%  PDF Assessment relative to Buoy
df = buoy_sate_daily_rainfall_colasped_df.copy()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()
bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']

# Compute PDF elements for all datasets
buoy_pdfc_pdfv = compute_pdf_elements(df, 'rain_rate', bin_values)
img_pdfc_pdfv = compute_pdf_elements(df, 'IMERG v07', bin_values)
gpcp_v3pt2_pdfc_pdfv = compute_pdf_elements(df, 'GPCP v3.2', bin_values)
gpcp_v3pt3_pdfc_pdfv = compute_pdf_elements(df, 'GPCP v3.3', bin_values)
era5_pdfc_pdfv = compute_pdf_elements(df, 'ERA5', bin_values)
merra2_pdfc_pdfv = compute_pdf_elements(df, 'MERRA2', bin_values)
# ============================================================
# PDFv / PDFc by Rainfall Intensity (Figure-5 style)
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=500)

lw = 4

# --- X axis: use actual bin values ---
x = buoy_pdfc_pdfv['bin'].values#bin_values#[:-1]   # last edge has no PDF value

# ------------------------------------------------------------
# PDFv (Volume-based PDF)  —— ACTIVE
# ------------------------------------------------------------
ax.plot(x, buoy_pdfc_pdfv['pdfv'], lw=lw, color='b', ls='-', label='Buoy')
ax.plot(x, merra2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['MERRA2'], label='MERRA2')
ax.plot(x, era5_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['ERA5'], label='ERA5')
ax.plot(x, img_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['IMERG v07'], label='IMERG v07')
ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.2'], label='GPCP v3.2')
ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfv'], lw=lw, color=product_colors['GPCP v3.3'], label='GPCP v3.3')

# ------------------------------------------------------------
# PDFc (Count-based PDF)  —— OPTIONAL (commented)
# ------------------------------------------------------------
# ax.plot(x, pal_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['PAL'], label='PAL (PDFc)')
# ax.plot(x, img_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['IMERG v07'], label='IMERG v07 (PDFc)')
# ax.plot(x, gpcp_v3pt2_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['GPCP v3.2'], label='GPCP v3.2 (PDFc)')
# ax.plot(x, gpcp_v3pt3_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['GPCP v3.3'], label='GPCP v3.3 (PDFc)')
# ax.plot(x, era5_pdfc_pdfv['pdfc'], lw=lw, ls='--',
#         color=product_colors['ERA5'], label='ERA5 (PDFc)')

# ------------------------------------------------------------
# Formatting (paper-quality)
# ------------------------------------------------------------
ax.set_xscale('log')
ax.set_xlabel('Rain Rate [mm day$^{-1}$]', fontsize=18, fontweight='bold')
ax.set_ylabel('PDF (%)', fontsize=18, fontweight='bold')

# ax.set_xticks(bin_values)
# ax.set_xticklabels(bin_labels)
from matplotlib.ticker import FixedLocator, FuncFormatter

bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

ax.set_xscale('log')
ax.xaxis.set_major_locator(FixedLocator(bin_values))

ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, pos: "0.5" if abs(v-0.5) < 1e-12 else f"{int(round(v))}"
))

# optional: remove minor tick marks entirely (cleaner for paper)
ax.xaxis.set_minor_locator(FixedLocator([]))

# (optional) turn off minor tick labels so only these show
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='x', which='major', labelsize=15)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=15, width=1.5, length=7)

ax.grid(True, which='major', linestyle='--', alpha=0.6)
ax.legend(fontsize=15, frameon=False)

plt.tight_layout()
svnme = os.path.join(path_to_plots, 
                     f'Buoy_Satellite_PDF_Comparison_{cde_run_dte}.png')
fig.savefig(svnme, dpi=300)

#%% A regional pdf analysis based on Buoy data
df = buoy_sate_daily_rainfall_colasped_df.copy()

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

# df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].copy()

# Ensure datetime
df['date'] = pd.to_datetime(df['date'])

# Year–month
df['year_month'] = df['date'].dt.to_period('M')


monthly_by_region = {}

for region in list(Buoy_region_markers.keys()):
    dfr = df[df['region'] == region].copy()

    monthly = (
        dfr
        .groupby('year_month')[products]
        .mean()
        .reset_index()
    )

    # Period → Timestamp
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

    # ---- NEW: enforce continuous monthly index ----
    full_index = pd.date_range(
        start=monthly['year_month'].min(),
        end=monthly['year_month'].max(),
        freq='MS'   # Month Start
    )

    monthly = (
        monthly
        .set_index('year_month')
        .reindex(full_index)
        .rename_axis('year_month')
        .reset_index()
    )

    # set the region column back 
    monthly['region'] = region

    monthly_by_region[region] = monthly

df_monthly = pd.concat(monthly_by_region, ignore_index=True)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="ENP",
    product_cols=("rain_rate", "GPCP v3.3"),
    colors=("black", "red"),
)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="ENP",
    product_cols=("rain_rate", "GPCP v3.2"),
    colors=("black", "blue"),
)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="ENP",
    product_cols=("rain_rate", "ERA5"),
    colors=("black", "lime"),
)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="WNP",
    product_cols=("rain_rate", "GPCP v3.3", ),
    colors=("black", "red", ),
)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="WNP",
    product_cols=("rain_rate", "GPCP v3.2", ),
    colors=("black", "blue", ),
)

fig, axes = plot_global_and_regional_pdfs_regioncol(
    df_monthly,
    region_name="WNP",
    product_cols=("rain_rate", "ERA5", ),
    colors=("black", "lime", ),
)

fig, axes = plot_global_and_regional_pdfs_insitu_hist_only(
    df,
    region_name="ENP",
    insitu_col="rain_rate",
    line_cols=("rain_rate", "GPCP v3.2", "GPCP v3.3", "ERA5"),
    line_colors=("black", "blue", "red", "lime", ),
    bins=np.arange(0, 15.5, 0.5),
    smooth_window=3
)

# "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"
# "red", "lime", "orange", "purple"
#%% OceanRain based assessment for the 45 degree poleward
# write crs and resample to gpcp resolution

gpcp_32_pnt25 = resample_to_new_res(gpcp_ds_v3pt2_al['precip'],(720,1440),'lon','lat')
gpcp_33_pnt25 = resample_to_new_res(gpcp_ds_v3pt3_al['precip'],(720,1440),'lon','lat')
imergv7_pnt25 = resample_to_new_res(imerg_v07_al,(720,1440),'lon','lat')
era5_pnt25 = resample_to_new_res(era5_ds_al['tp'],(720,1440),'x','y')
mer2_pnt25 = resample_to_new_res(mer2_ds_al,(720,1440),'x','y')

df_qc = oceanrain_step0_qc_v2(ocRain_df, 
                              min_flag2=13, 
                              keep_spurious_flag2_11=False, 
                              qclip_hi=None)

# 0) QC
# df_qc = oceanrain_step0_qc(
#     ocRain_df,
#     min_flag2=None,   # set to 14 if you want >=0.1 mm/h (plus true_zero)
#     prob_thr=None,
#     wind_max=None,
#     qclip_hi=None, 
# )

# 1) Daily aggregation to GPCP pixels (includes poleward cut inside)
daily_or, rain_days, snow_days = oceanrain_daily_aggregate_to_gpcp_v2(
    df_qc,
    gpcp_ds_v3pt2_al.lat.values,
    gpcp_ds_v3pt2_al.lon.values,
    lat_abs_min=45,
    coverage_frac=0.10,
    phase_frac_thr=0.50,
    include_mixed_in_all=False,
)
#-------------------------------------------------------------
# cols = ["dsd_mmday_rain", "dsd_mmday_snow"]  # add "rate_gag_mmph" if you want too

# for c in cols:
#     top3 = (
#         snow_days[c]
#         .dropna()
#         .nlargest(3)
#     )
#     print(f"\n=== {c}: top 3 values ===")
#     print(top3)  # shows index + value

#     print("\nRows for these top values:")
#     print(snow_days.loc[top3.index, ["time_utc", "lat", "lon", "ship", "precip_flag", "precip_flag2", c]])
#------------------------------------------------------------

# NH/SH split is now easy
snow_days_NH = snow_days[snow_days["hemi"] == "NH"]
snow_days_SH = snow_days[snow_days["hemi"] == "SH"]

products = {
    "GPCP v3.2": (gpcp_ds_v3pt2_al, {"GPCP v3.2": 'precip'}), # , "gpcp_pliq": "probability_liquid_phase"
    "GPCP v3.3": (gpcp_ds_v3pt3_al, {"GPCP v3.3": 'precip'}), # , "gpcp_pliq": "probability_liquid_phase"
    "ERA5": (era5_ds_al, {"ERA5": "tp"}),
    "IMERG": (imerg_v07_al, {"IMERG v07": None}),
    "MERRA2": (mer2_ds_al, {"MERRA2": None})
}

snow_days2, rain_days2, (snow_NH, snow_SH, rain_NH, rain_SH) = step2_attach_and_split(
    snow_days=snow_days,
    rain_days=rain_days,
    gpcp_grid_ds=gpcp_ds_v3pt2_al,
    products=products
)

phase_based_cat_metrics_hemi = {}
phase_based_qt_metrics_hemi  = {}

phase_based_data = {"Snow": snow_days2, "Rain": rain_days2}

# ensure hemi exists (recommended)
# snow_days2 = add_pixel_coords_and_hemi(snow_days2, gpcp_ds_v3pt2_al["lat"].values, gpcp_ds_v3pt2_al["lon"].values)
# rain_days2 = add_pixel_coords_and_hemi(rain_days2, gpcp_ds_v3pt2_al["lat"].values, gpcp_ds_v3pt2_al["lon"].values)
phase_based_data = {"Snow": snow_days2, "Rain": rain_days2}

products = ['GPCP v3.2',  'GPCP v3.3','IMERG v07', 'ERA5', 'MERRA2'] # 

for hemi in ["NH", "SH"]:
    phase_based_cat_metrics_hemi.setdefault(hemi, {})
    phase_based_qt_metrics_hemi.setdefault(hemi, {})

    for phse, phse_dat_all in phase_based_data.items():
        phse_dat = phse_dat_all[phse_dat_all["hemi"] == hemi].copy()
        # if phse == 'Snow':
        #     phse_dat = phse_dat[phse_dat["gpcp_pliq"] <= 50]
        # elif phse == 'Rain':
        #     phse_dat = phse_dat[phse_dat["gpcp_pliq"] >= 80]

        # obs_lab = "dsd_mmday_snow" if phse == "Snow" else "dsd_mmday_rain"
        obs_lab = "dsd_mean_rain_mmph" if phse == "Snow" else "dsd_mean_snow_mmph"
        thr = 0.5 if phse == "Rain" else 0.25

        print(f"Processing Hemi={hemi} Phase={phse} (N={len(phse_dat)})")

        phase_based_cat_metrics_hemi[hemi].setdefault(phse, {})
        phase_based_qt_metrics_hemi[hemi].setdefault(phse, {})

        for product in products:
            forecast  = phse_dat[product]
            # observed  = phse_dat[obs_lab]
            observed  = phse_dat[obs_lab] * 24

            # categorical
            reg_cat_met = categorical_stats(forecast, observed, thr)

            # quantitative
            phse_dat_ = phse_dat.copy()
            phse_dat_ = phse_dat_[(phse_dat_[obs_lab] >= thr) & (phse_dat_[product] >=thr)]
            reg_qt_met  = calculate_metrics(phse_dat, obs_lab, product)

            phase_based_cat_metrics_hemi[hemi][phse][product] = reg_cat_met
            phase_based_qt_metrics_hemi[hemi][phse][product]  = reg_qt_met



ylims = {
    # "CC": (0, 0.7),
    # set others only if you want fixed ranges
    # "RMSE": (0, 20),
    # "MAE": (0, 10),
    # 'POD': (0,8)

}

# products = ["GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"]

fig_cat = plot_hemi_phase_metrics_barpanel(
    phase_based_cat_metrics_hemi,
    products,
    product_colors,
    metrics=("POD","FAR","Bias","HSS"),
    phases=("Rain","Snow"),
    hemis=("SH","NH"),
    ylims=None
)

fig_qt = plot_hemi_phase_quant_metrics_panel(
    phase_based_qt_metrics_hemi,
    products,
    product_colors,
    metrics=("CC","RMSE","MAE","Bias"),
    phases=("Rain","Snow"),
    hemis=("SH","NH"),
    ylims=None
)


#----------------------------------------------------------------------------------
products = ["GPCP v3.2","GPCP v3.3","ERA5","IMERG v07","MERRA2"]

snow_clim = make_doy_climatology_pooled(
    snow_days2, obs_col="dsd_mmday_snow", product_cols=products, min_pairs=1
)

rain_clim = make_doy_climatology_pooled(
    rain_days2, obs_col="dsd_mmday_rain", product_cols=products, min_pairs=1
)

fig_snow = plot_doy_climatology_scatter_panels_with_metrics(
    snow_clim,
    obs_col="dsd_mmday_snow",
    product_cols=products,
    product_colors=product_colors,
    title="Snow: DOY-mean climatology (pooled across years/ships) vs OceanRAIN",
    max_val=18,
    ticks=(0, 6, 12, 18),
)

fig_rain = plot_doy_climatology_scatter_panels_with_metrics(
    rain_clim,
    obs_col="dsd_mmday_rain",
    product_cols=products,
    product_colors=product_colors,
    title="Rain: DOY-mean climatology (pooled across years/ships) vs OceanRAIN",
    max_val=18,
    ticks=(0, 6, 12, 18),
)

#----------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

products = ['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2']

def monthly_climatology(df, obs_col, products, hemi):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[d["hemi"] == hemi].copy()
    d["month"] = d["date"].dt.month

    out = {"OceanRAIN": d.groupby("month")[obs_col].mean()}
    for p in products:
        out[p] = d.groupby("month")[p].mean()

    return pd.DataFrame(out).reindex(range(1, 13))

def plot_one(ax, clim_df, title, ylabel=None):
    # OceanRAIN bold to anchor
    ax.plot(clim_df.index, clim_df["OceanRAIN"], lw=3, label="OceanRAIN")

    # Products thinner
    for p in products:
        ax.plot(clim_df.index, clim_df[p], lw=1.8, label=p)

    ax.set_title(title)
    ax.set_xlabel("Month")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 12)

# ---- build climatologies ----
rain_nh = monthly_climatology(rain_days2, "dsd_mmday_rain", products, hemi="NH")
rain_sh = monthly_climatology(rain_days2, "dsd_mmday_rain", products, hemi="SH")

snow_nh = monthly_climatology(snow_days2, "dsd_mmday_snow", products, hemi="NH")
snow_sh = monthly_climatology(snow_days2, "dsd_mmday_snow", products, hemi="SH")

# ---- 2x2 plot: rows=hemi, cols=phase (Rain left, Snow right) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 6), dpi=200, sharex=True)

plot_one(axes[0, 0], rain_nh, "NH Rain: monthly climatology", ylabel="Daily intensity (mm/day)")
plot_one(axes[0, 1], snow_nh, "NH Snow: monthly climatology", ylabel="Daily intensity (mm/day)")
plot_one(axes[1, 0], rain_sh, "SH Rain: monthly climatology", ylabel="Daily intensity (mm/day)")
plot_one(axes[1, 1], snow_sh, "SH Snow: monthly climatology", ylabel="Daily intensity (mm/day)")

# One legend for all panels (outside)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
fig.subplots_adjust(bottom=0.18, wspace=0.25, hspace=0.30)

plt.show()
# plt.show()
#%% Data Visaulization plots

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))

# Log scales
ax.set_xscale("log")
ax.set_yscale("log")

# Axis limits (start at Daily now)
ax.set_xlim(1e-1, 1e4)
ax.set_ylim(1e0, 1e4)

# ---- X-axis: Spatial scale ----
x_ticks = [1e-1, 1e0, 1e2, 1e4]
x_labels = [
    "Point/Pixel",
    "Local",
    "Regional",
    "Global"
]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=13, fontweight="bold")

# ---- Y-axis: Temporal scale (Daily → Climatological) ----
y_ticks = [1e0, 1e1, 1e2, 1e3, 1e4]
y_labels = [
    "Daily",
    "Monthly",
    "Seasonal",
    "Annual",
    "Climatological"
]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=13, fontweight="bold")

# Grid
ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)

# Labels
ax.set_xlabel("Spatial Scale", fontsize=15, fontweight="bold")
ax.set_ylabel("Temporal Scale", fontsize=15, fontweight="bold")

# Frame styling
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------

# %%


# ------------------------------------------------------------
# Each dataset can have 1+ "segments" (start, end, resolution)
# Plot the longest segment as the thick solid bar
# Plot shorter segments as thinner dashed overlays on same row
# ------------------------------------------------------------

segments_by_dataset = {
    "PAL":       [(2010, 2021, "subhourly")],
    "Buoy":      [(1997, 2023, "hourly")],
    "Atolls":    [(1983, 2023, "monthly")],
    "OceanRAIN": [(2010, 2017, "subhourly")],

    # Example: monthly long window + daily subset window (overlay)
    "GPCP v3.2": [(1983, 2023, "monthly"), (2000, 2020, "daily")],
    "GPCP v3.3": [(1983, 2023, "monthly"), (2000, 2020, "daily")],
    "IMERG v07": [(1998, 2023, "monthly"), (2000, 2020, "daily")],  # adjust to your actual usage
    "ERA5":      [(1983, 2023, "monthly"), (2000, 2020, "daily")],
    "MERRA2":    [(1983, 2023, "monthly"), (2000, 2020, "daily")],
}

# color by resolution (keep your palette)
colors = {
    "subhourly": "#7B2CBF",
    "hourly":    "#1D4ED8",
    "daily":     "#16A34A",
    "monthly":   "#F59E0B",
}

# order top->bottom
names = list(segments_by_dataset.keys())
ypos = list(range(len(names)))[::-1]

fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

for name, y in zip(names, ypos):
    segs = segments_by_dataset[name]

    # sort longest first so it becomes the base bar
    segs = sorted(segs, key=lambda t: (t[1] - t[0]), reverse=True)

    # ---- base (longest) segment: thick solid ----
    x0, x1, res = segs[0]
    ax.hlines(y, x0, x1, lw=10, color=colors[res], alpha=0.9, zorder=1)
    ax.plot([x0, x1], [y, y], "o", ms=4, color=colors[res], zorder=2)

    # ---- overlays (shorter): thinner dashed ----
    for x0, x1, res in segs[1:]:
        ax.hlines(
            y, x0, x1,
            lw=4,
            color=colors[res],
            alpha=1.0,
            linestyle=(0, (4, 2)),   # dashed
            zorder=3
        )
        ax.plot([x0, x1], [y, y], "o", ms=3, color=colors[res], zorder=4)

ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlim(1983, 2023)
ax.set_xticks(np.arange(1983, 2028, 5))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# leave room for legend
fig.subplots_adjust(bottom=0.28)

# ---- Legend: solid = base availability, dashed = higher-res subset ----
res_handles = [Line2D([0],[0], color=colors[k], lw=8) for k in ["subhourly","hourly","daily","monthly"]]
res_labels  = ["Subhourly", "Hourly", "Daily", "Monthly"]

style_handles = [
    Line2D([0],[0], color="0.2", lw=8, linestyle="-"),
    Line2D([0],[0], color=colors["daily"], lw=6, linestyle=(0,(4,2))),
]
style_labels = ["Base availability", "Subset (higher-res window)"]

# two-row legend (same bottom center location)
fig.legend(res_handles, res_labels, loc="lower center", bbox_to_anchor=(0.5, 0.06),
           ncol=4, frameon=False, fontsize=12, handlelength=2.5, columnspacing=1.8)
# fig.legend(style_handles, style_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
#            ncol=2, frameon=False, fontsize=12, handlelength=2.5, columnspacing=2.0)

fig.tight_layout(rect=[0.02, 0.15, 0.98, 0.95])
plt.show()

