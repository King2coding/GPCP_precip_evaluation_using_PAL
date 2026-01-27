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

print("GPCP loading complete!...")
print("-" * 30 + "\n")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Use multiprocessing to process ERA5 files in parallel
era5_ds_xr_list = []
with Pool(processes=18) as pool:  # Adjust the number of processes as needed
    era5_ds_xr_list = pool.map(process_era5_file, enumerate(all_era5_tp_files))
# Combine all processed batches into a single xarray dataset - simple version
if era5_ds_xr_list:
    era5_ds_xr = xr.concat(era5_ds_xr_list, dim="valid_time")
    print("ERA5 loading complete")
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
    print("MERRA2 loading complete")
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
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, PAL_region_bounds)

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
print('PAL data loading and classification complete!')
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
print("BUOY CLASSIFICATION SUMMARY COMPLETED")
print("\n" + "="*50)
gc.collect()
print("Data files listed and datasets loaded.")


#%%

era5_xr = xr.open_dataset(r'/ra1/pubdat/ECMWF/ERA5/daily/era5_tp_2007_daily.nc', engine='netcdf4')
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