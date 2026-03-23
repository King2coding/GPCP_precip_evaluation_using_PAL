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
mindate,maxdate = gpcp_ds_v3pt2_xr.time.min().values, gpcp_ds_v3pt2_xr.time.max().values
# mindate,maxdate = imerg_v06_ds_xr.time.min().values, gpcp_ds_v3pt2_xr.time.max().values

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
        # Process ERA5 data with PAL 
        pal_rain_era5_df = pal_rain_df.copy() 
        
        pal_era5_df_rain = process_era5_with_PAL_rain_and_wind_v1(pal_rain_era5_df, era5_ds_xr)        

        pal_era5_df_rain.index = pd.to_datetime(pal_era5_df_rain['time'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        # Process IMERG data with PAL 
        # pal_rain_imerg_df = pal_rain_df.copy() 
        # pal_imerg_v06_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v06_ds_xr, 'v06')
        # pal_imerg_v06_df_rain.index = pd.to_datetime(pal_imerg_v06_df_rain['time'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        pal_rain_imerg_df = pal_rain_df.copy() 
        pal_imerg_v07_df_rain = process_imerg_with_PAL_rain_and_wind_v1(pal_rain_imerg_df, imerg_v07_ds_xr, 'v07')
        pal_imerg_v07_df_rain.index = pd.to_datetime(pal_imerg_v07_df_rain['time'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------
        
        # Process MERRA2 data with PAL
        pal_rain_merra2_df = pal_rain_df.copy()
        pal_merra2_df_rain = process_merra2_with_PAL_rain_and_wind_v1(pal_rain_merra2_df, mer2_ds_xr)
        pal_merra2_df_rain.index = pd.to_datetime(pal_merra2_df_rain['time'])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ---------------

        # combine all dfs into a single df, retaining only date, region, rain_rate, and GPCP data         

        # merge GPCP v3.2 data  
        pal_df_combined_rain = pal_gpcpv3pt2_df_rain.copy()
        pal_df_combined_rain = pal_df_combined_rain[['time','date','rain_rate', 
                                                     'GPCP v3.2','PLP_GPCP v3.2']].copy()       

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # merge GPCP v3.3 data
        # bring the lquid precip data into gpcp v3.3 df
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv3pt3_df_rain[['date','GPCP v3.3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v3.3')
        )

        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['GPCP v3.3_v3.3', 'date_v3.3']], 
                                                inplace=True)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # ERA5 merge
        
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
        
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_imerg_v07_df_rain[['date','v07']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v07')  
        )
        # # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['v07_v07', 'date_v07']], 
                                                inplace=True)       
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
        # MERRA2 merge
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_merra2_df_rain[['date','MERRA2']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_MERRA2')
        )
        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['MERRA2_MERRA2', 'date_MERRA2']], 
                                                inplace=True)
        # 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -        
        pal_df_combined_rain = pal_df_combined_rain[pal_df_combined_rain['PLP_GPCP v3.2'] == 100]        

        # groupby date and get mean of rain_rate and GPCP data 'GPCP_v1pt3',
        grp = pal_df_combined_rain.groupby('date')
        daily_avg_rain = grp.mean([['rain_rate', 
                                    'GPCP v3.2', 
                                    'GPCP v3.3', 
                                    # 'IMERG v06',
                                    'IMERG v07',
                                    'ERA5',
                                    'MERRA2']]) \
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
                                                                'ERA5',
                                                                'MERRA2']] \
                                                     .mean() \
                                                     .reset_index()
    region_pal_sate_df_daily_mean['region'] = region_name  # Add region name for clarity
    regional_PAL_sate_dfs_daily_mean[region_name] = region_pal_sate_df_daily_mean

    # Append to the list for later processing
    regional_PAL_sate_dfs_daily_lst.append(region_pal_sate_df)
gc.collect()  # Clean up memory

# # save dfs to disk
# print("✅ PAL-GPCP matching complete!")
# print("Saving PAL-GPCP matched dataframes to disk...")
pal_sate_daily_mean_df = pd.concat([df for df in regional_PAL_sate_dfs_daily_mean.values()], ignore_index=True)
pal_sate_daily_mean_df.to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_mean_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

pal_sate_daily_rainfall_colasped_df = pd.concat(regional_PAL_sate_dfs_daily_lst, ignore_index=True)
pal_sate_daily_rainfall_colasped_df.to_pickle(os.path.join(path_to_put_dfs, f'pal_sate_daily_rainfall_from_all_regions_and_all_tracks_{cde_run_dte}.pkl'))

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
                                                                        'IMERG v06',
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

#%% Daily-Scale Assessment: Detection and Intensity Skill (PAL BASED ASSESSMENT)
# SCATTER PLOT

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


#%% Daily-Scale Assessment: Detection and Intensity Skill (PAL BASED ASSESSMENT)
# THE CATEGORICAL METRICS
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

#%%  Daily-Scale Assessment: Detection and Intensity Skill (PAL BASED ASSESSMENT)
# PDF Assessment
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


#%% Daily-Scale Assessment: Detection and Intensity Skill (Buoy BASED ASSESSMENT)
# Scatter Plot
print('Starting buoy-based assessment...')
# THE BUOY BASED ASSESSMENT
buoy_sate_daily_mean_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_mean_from_all_regions_and_all_IDs_20260216.pkl'))  


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
#%%  Monthly to Interannual Variability: A Buoy-Supported Assessment
print('Starting monthly timeseries analysis...')
buoy_sate_daily_rainfall_colasped_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260216.pkl'))

df = buoy_sate_daily_rainfall_colasped_df.copy()

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year.astype(int)

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

# MONTHLY CLIMATOLOGY

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

#-----------------------------------------------------------------------------
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

#%%  Annual Cycle Comparison: products vs buoys
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

#%% Interannual Variability: Monthly Anomaly Scatterplots — Product vs Buoy
products_eval = ["rain_rate", "GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"]

monthly_region = make_monthly_region_series(
    df=buoy_sate_daily_rainfall_colasped_df,
    products=products_eval,
    date_col="date",
    region_col="region"
)

monthly_region_anom = deseasonalize_monthly(
    monthly_region,
    products=products_eval,
    time_col="month_start",
    region_col="region"
)

fig = plot_deseasonalized_anomaly_scatter(
    df_anom=monthly_region_anom,
    regions=["ENP", "WNP", "IND", "ATL"],
    ref_col="rain_rate",
    product_cols=["GPCP v3.2", "GPCP v3.3", "ERA5", "IMERG v07", "MERRA2"],
    region_labels=Buoy_REGION_NAMES,
    product_colors=product_colors,
    region_col="region",
    figsize=(20, 12),
    savepath=os.path.join(path_to_plots, f"deseasonalized_monthly_anomaly_scatter_{cde_run_dte}.png")
)
# plt.close(fig)
gc.collect()

#%% Poleward Assessment: OceanRAIN (≥45°)

df_qc = oceanrain_step0_qc_precip_main(
    ocRain_df,
    drop_harbor_inop=True,
    drop_spurious_flag2_11=True,
    keep_true_zero=True,
    keep_flag2_12_zero_precip=False,
    min_flag2_positive=13,
    prob_thr=None,
    wind_max=None,
    qclip_hi=None,
)

daily_or_all, daily_or_usable = oceanrain_daily_aggregate_to_gpcp_main(
    oc_df_minute=df_qc,
    gpcp_lat_1d=gpcp_ds_v3pt2_al.lat.values,
    gpcp_lon_1d=gpcp_ds_v3pt2_al.lon.values,
    lat_abs_min=45.0,
    coverage_frac=0.50,
)

#--------------------------------------------------------------
product_map = {
    "GPCP v3.2": (gpcp_ds_v3pt2_al, {"GPCP v3.2": "precip"}),
    "GPCP v3.3": (gpcp_ds_v3pt3_al, {"GPCP v3.3": "precip"}),
    "ERA5": (era5_ds_al, {"ERA5": "tp"}),
    "IMERG v07": (imerg_v07_al, {"IMERG v07": None}),
    "MERRA2": (mer2_ds_al, {"MERRA2": None}),
}

# print(type(product_map))

# daily_or_usable_common, common_start, common_end = subset_to_common_time_range(
#     daily_or_usable,
#     products,
#     date_col="date",
# )

daily_or_attached = step2_attach_products_oceanrain(
    daily_or_usable,
    gpcp_grid_ds=gpcp_ds_v3pt2_al,
    products=product_map,
    ref_col="main_mmday",
)

#--------------------------------------------------------------
products_eval = ['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2']

cat_metrics_hemi, qt_metrics_hemi = compute_hemi_metrics_oceanrain(
    daily_or_attached,
    products=products_eval,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
    cat_thr=0.3,
)

#--------------------------------------------------------------
fig1 = plot_hemi_cat_metrics_barpanel(
    cat_metrics_hemi,
    products_eval,
    product_colors,
    metrics=("POD", "FAR", "Bias", "HSS"),
    hemis=("SH", "NH"),
    figsize=(10, 10),
)

fig2 = plot_hemi_quant_metrics_barpanel(
    qt_metrics_hemi,
    products_eval,
    product_colors,
    metrics=("CC", "RMSE", "MAE", "Bias"),
    hemis=("SH", "NH"),
    figsize=(10, 10),
)
#-------------------------------------------------------------
products_eval = ['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2']

ship_month_clim = build_oceanrain_ship_month_climatology(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_eval,
    min_days_per_ship_month=1,
)

monthly_ship = build_oceanrain_ship_year_month_means(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_eval,
    min_days_per_month=1,   # you can later test 3 or 5
)

fig_sh = plot_oceanrain_monthly_ship_scatter_by_hemi(
    monthly_ship,
    hemi="NH",
    products=['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2'],
    figsize=(15, 7.5),
    xylim=(0, 15),
    add_titles=False,
)

fig_sh = plot_oceanrain_monthly_ship_scatter_by_hemi(
    monthly_ship,
    hemi="SH",
    products=['GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA2'],
    figsize=(15, 7.5),
    xylim=(0, 10),
    add_titles=False,
)

#-------------------------------------------------------------
ship_month_metrics = compute_metrics_from_ship_month_climatology(
    ship_month_clim,
    products=products_eval,
    obs_col="main_mmday",
)

#%% Data Visualization plots

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
    "Multi-year"
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

# %% Temporal Coverage, Resolution, and Assessment Windows of References and Products

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

