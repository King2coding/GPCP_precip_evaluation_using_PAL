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
path_to_ocean_rain_nc = r'/ra1/pubdat/OceanRain/nc'

path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_1998_2024'

path_to_gpcp_v3pt2 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_2_2000_2020'

path_to_gpcp_v2pt3 = r'/ra1/pubdat/GPCP/v2.3'

path_to_gpcp_v1pt3 = r'/ra1/pubdat/GPCP/V1.3'

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

all_gpcp_v2pt3_files = sorted([os.path.join(path_to_gpcp_v2pt3, f) for f in os.listdir(path_to_gpcp_v2pt3) if ('preliminary' not in f) and f.endswith('.nc')])

all_gpcp_v1pt3_files = sorted([os.path.join(path_to_gpcp_v1pt3, f) for f in os.listdir(path_to_gpcp_v1pt3) if f.endswith('.nc')])

all_gpcp_v1pt3_2000_2020_files = [f for f in all_gpcp_v1pt3_files if \
                                  2000 <= int(os.path.basename(f)
                                    .split('_')[3].replace('d','')[:4]) \
                                    <= 2020]

all_imerg_v06_files = sorted([os.path.join(path_to_imerg_v06, f) for f in os.listdir(path_to_imerg_v06) if f.endswith('.nc4')])

all_imerg_v07_files = sorted([os.path.join(path_to_imerg_v07, f) for f in os.listdir(path_to_imerg_v07) if f.endswith('.nc4')])

all_era5_tp_files = sorted([os.path.join(path_to_era5_tp, f) for f in os.listdir(path_to_era5_tp) if f'era5_tp_' in f and f.endswith('.nc')])

all_merra2_files = sorted([os.path.join(path_to_merra2, f) for f in os.listdir(path_to_merra2) if f.endswith('.nc4')])

#%%
# Load GPCP datasets using xarray with chunking for efficiency
gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcp_v3pt2_files,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, 
                                            "lon": 360},  # <<< important
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
                                    chunks={"time": 120, "lat": 180, 
                                            "lon": 360},  # <<< important
                                    cache=False
                                    )
gpcp_ds_v3pt3_xr = ds_swaplon(gpcp_ds_v3pt3_xr)

gpcp_ds_v1pt3_xr = xr.open_mfdataset(all_gpcp_v1pt3_2000_2020_files,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, 
                                    "lon": 360},  # <<< important
                                    cache=False
                                    )

gpcp_ds_v1pt3_xr = ds_swaplon(gpcp_ds_v1pt3_xr)

print("✅ GPCP loading complete!...")
print("-" * 30 + "\n")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Use multiprocessing to process ERA5 files in parallel
era5_ds_xr_list = []
with Pool(processes=5) as pool:  # Adjust the number of processes as needed
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
with Pool(processes=5) as pool:  # Adjust the number of processes as needed
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
# imerg_v06_ds_xr_list = []
# with Pool(processes=18) as pool:
#     imerg_v06_ds_xr_list = pool.map(
#         process_imerg_file,
#         [(idx, file_path, 'v06') for idx, file_path in enumerate(all_imerg_v06_files)]
#     )

# # Combine all processed batches into a single xarray dataset - simple version
# if imerg_v06_ds_xr_list:
#     imerg_v06_ds_xr = xr.concat(imerg_v06_ds_xr_list, dim="time")

imerg_v07_ds_xr_list = []
with Pool(processes=2) as pool:
    imerg_v07_ds_xr_list = pool.map(
        process_imerg_file,
        [(idx, file_path, 'v07') for idx, file_path in enumerate(all_imerg_v07_files)]
    )
# Combine all processed batches into a single xarray dataset - simple version
if imerg_v07_ds_xr_list:
    imerg_v07_ds_xr = xr.concat(imerg_v07_ds_xr_list, dim="time")


print("IMERG loading complete")
print("-" * 50 + "\n")
del(imerg_v07_ds_xr_list) # imerg_v06_ds_xr_list,
gc.collect() 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# ALIGN ALL DATASETS IN TIME
mindate,maxdate = gpcp_ds_v3pt2_xr.time.min().values, gpcp_ds_v3pt2_xr.time.max().values
# mindate,maxdate = imerg_v06_ds_xr.time.min().values, gpcp_ds_v3pt2_xr.time.max().values

# select time range for all datasets
gpcp_ds_v3pt2_al = gpcp_ds_v3pt2_xr.sel(time=slice(mindate, maxdate)).compute()#.chunk({'time': -1})
gpcp_ds_v3pt3_al = gpcp_ds_v3pt3_xr.sel(time=slice(mindate, maxdate)).compute()#.chunk({'time': -1})
gpcp_ds_v1pt3_al = gpcp_ds_v1pt3_xr.sel(time=slice(mindate, maxdate)).compute()#.chunk({'time': -1})
era5_ds_al = era5_ds_xr.sel(valid_time=slice(mindate, maxdate))#.compute()#.chunk({'valid_time': -1})
imerg_v07_al = imerg_v07_ds_xr.sel(time=slice(mindate, maxdate))#.compute()#.chunk({'time': -1})
# imerg_v06_al = imerg_v06_ds_xr.sel(time=slice(mindate, maxdate))#.compute()#.chunk({'time': -1})
mer2_ds_al = mer2_ds_xr.sel(time=slice(mindate, maxdate))#.chunk({'time': -1})

#%% The In situ data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
all_ocean_rain_files = sorted([os.path.join(path_to_ocean_rain_nc, f) for f in os.listdir(path_to_ocean_rain_nc) if f.endswith('.nc')])

# group ocean rain files by year
files_by_year = defaultdict(list)

for o in all_ocean_rain_files:
    # print(os.path.basename(o))
    with xr.open_dataset(o) as ds:
        years = pd.to_datetime(ds['time'].values).year
        unique_years = np.unique(years)
        for year in unique_years:
            files_by_year[year].append(o)

files_by_year = dict(sorted(files_by_year.items()))
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

#%% Generating spatial distribution plot of in-situ observations
print("\nGenerating spatial distribution plot of in-situ observations...")
# ------------------------------------------------------------
# figure/axes
# ------------------------------------------------------------

fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

# ------------------------------------------------------------
# OceanRain plotting: original NC files, but color by ACTUAL year in file
# ------------------------------------------------------------

# use each file only once
all_oceanrain_files = sorted(set([f for flist in files_by_year.values() for f in flist]))

stride = 10  # keep your current setting for speed

for f in all_oceanrain_files:
    ds = xr.open_dataset(f)

    time_var = infer_time_var(ds)

    # keep native longitude as stored, since that was closer to your original working version
    time_vals = pd.to_datetime(np.squeeze(ds[time_var].values))
    lat_vals  = np.squeeze(ds['latitude'].values)
    lon_vals  = np.squeeze(ds['longitude'].values)

    ds.close()

    # basic sanity: only continue if aligned 1D arrays
    if time_vals.ndim != 1 or lat_vals.ndim != 1 or lon_vals.ndim != 1:
        print(f"Skipping {os.path.basename(f)} because variables are not 1D:",
              time_vals.shape, lat_vals.shape, lon_vals.shape)
        continue

    n = min(len(time_vals), len(lat_vals), len(lon_vals))
    time_vals = time_vals[:n]
    lat_vals  = lat_vals[:n]
    lon_vals  = lon_vals[:n]

    good = (~pd.isna(time_vals)) & np.isfinite(lat_vals) & np.isfinite(lon_vals)
    time_vals = time_vals[good]
    lat_vals  = lat_vals[good]
    lon_vals  = lon_vals[good]

    years_in_file = pd.DatetimeIndex(time_vals).year

    for yr in sorted(year_colors.keys()):
        mask = (years_in_file == yr)
        if not np.any(mask):
            continue

        ax.scatter(
            lon_vals[mask][::stride],
            lat_vals[mask][::stride],
            color=year_colors[yr],
            s=2,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

    del time_vals, lat_vals, lon_vals, years_in_file, ds
    gc.collect()

# ------------------------------------------------------------
# PAL tracks
# ------------------------------------------------------------
for region, files in pals_classed_by_region.items():
    if region == "Unclassified" or len(files) == 0:
        continue

    color = PAL_region_colors[region]

    for file in files:
        ds = xr.open_dataset(file)
        lat = ds['lat'].values[::25]
        lon = wrap_lon(ds['lon'].values[::25])
        ds.close()

        ax.plot(
            lon, lat,
            transform=ccrs.PlateCarree(),
            color=color, linewidth=2.0, zorder=4
        )

# ------------------------------------------------------------
# Buoys
# ------------------------------------------------------------
buoy_counts = {
    "Eastern PACIFIC": len(buoy_files_by_region["ENP"]),
    "Western PACIFIC": len(buoy_files_by_region["WNP"]),
    "INDIAN": len(buoy_files_by_region["IND"]),
    "ATLANTIC": len(buoy_files_by_region["ATL"]),
}

buoy_specs = [
    ("Eastern PACIFIC", buoy_files_by_region["ENP"], '*'),
    ("Western PACIFIC", buoy_files_by_region["WNP"], 'P'),
    ("INDIAN",          buoy_files_by_region["IND"], 's'),
    ("ATLANTIC",        buoy_files_by_region["ATL"], 'd'),
]

for buoy_reg, buoy_files, marker in buoy_specs:
    for fl in buoy_files:
        xrfile = xr.open_dataset(fl)
        lat = xrfile['lat'].values[0]
        lon = wrap_lon(xrfile['lon'].values[0])
        xrfile.close()

        if lon == -180:
            lon = -179.8

        ax.scatter(
            lon, lat,
            color='k', s=35, linewidths=1.5, marker=marker,
            transform=ccrs.PlateCarree(), zorder=5
        )

# ------------------------------------------------------------
# Grid / ticks
# ------------------------------------------------------------
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.5, linestyle='--')

xticks = np.arange(-180, 181, 60)
yticks = np.arange(-90, 91, 30)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

ax.tick_params(labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# ax.set_title(
#     "Spatial Distribution of In Situ Observations Over Ocean Regions",
#     fontsize=22, fontweight='bold'
# )

# ------------------------------------------------------------
# Manual legend
# ------------------------------------------------------------
full_region_names = {
    "ETNP": "Extratropical North Pacific",
    "TNEP": "Tropical Northeastern Pacific",
    "TSEP": "Tropical Southeastern Pacific",
    "STNA": "Subtropical North Atlantic",
    "TNIO": "Tropical North Indian Ocean",
    "TNWP": "Tropical Northwestern Pacific"
}

legend_regions = [r for r in PAL_region_colors.keys() if r != "Unclassified"]

pal_handles = [
    mlines.Line2D([], [], color=PAL_region_colors[r], lw=2)
    for r in legend_regions
]
pal_labels = [
    f"{r}: ({full_region_names[r]} ({len(pals_classed_by_region.get(r, []))} PALs)"
    for r in legend_regions
]

# buoy handles: FIXED to four handles for four labels
buoy_handles = [
    mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=9),
    mlines.Line2D([], [], color='black', marker='P', linestyle='None', markersize=8),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=8),
    mlines.Line2D([], [], color='black', marker='d', linestyle='None', markersize=8),
]
buoy_labels = [
    f"Eastern Pacific ({buoy_counts['Eastern PACIFIC']} Buoys)",
    f"Western Pacific ({buoy_counts['Western PACIFIC']} Buoys)",
    f"Indian ({buoy_counts['INDIAN']} Buoys)",
    f"Atlantic ({buoy_counts['ATLANTIC']} Buoys)",
]

# OceanRain handles strictly from year_colors order
ocr_handles = [
    mlines.Line2D([], [], color=year_colors[yr], marker='o', linestyle='None', markersize=8)
    for yr in sorted(year_colors.keys())
]
ocr_labels = [f"OceanRain {yr}" for yr in sorted(year_colors.keys())]

handles = pal_handles + buoy_handles + ocr_handles
labels = pal_labels + buoy_labels + ocr_labels

leg = ax.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.3),
    fontsize=12,
    ncol=4,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.6
)

for text in leg.get_texts():
    text.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.30)

svname = os.path.join(
    path_to_plots,
    f'Fig02_Spatial_distribution_of_the_in_situ_ocean_precipitation_references_{cde_run_dte}.png'
)
plt.savefig(svname, bbox_inches='tight', dpi=250)
# plt.show()
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
        pal_rain_gpcpv1pt3_df = pal_rain_df.copy()          
        
        pal_gpcpv1pt3_df_rain = process_gpcp_with_PAL_rain_and_wind(
                                        pal_rain_gpcpv1pt3_df,
                                        gpcp_ds_v1pt3_xr, 'GPCP v1.3')  # , pal_wind_gpcpv1pt3_df

        pal_gpcpv1pt3_df_rain.index = pd.to_datetime(pal_gpcpv1pt3_df_rain['time'])        

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

        # merge GPCP v1.3 data
        pal_df_combined_rain = pal_df_combined_rain.merge(
            pal_gpcpv1pt3_df_rain[['date','GPCP v1.3']], 
            left_index=True, right_index=True, how='left', suffixes= ('', '_v1.3')
        )

        # Remove any duplicate columns from previous merges
        pal_df_combined_rain.drop(columns=[i for i in pal_df_combined_rain.columns if i in \
                                                ['GPCP v1.3_v1.3', 'date_v1.3']], 
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
                                    'GPCP v1.3',
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
                                                                'GPCP v1.3',
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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -------------
        # Process GPCP 1.3
        gpcpv1pt3_df = extract_point_timeseries_to_df(
                gpcp_ds_v1pt3_al,
                b_lat,
                b_lon,
                df_t_min,
                df_t_max,
                varnames=["precip"],          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
                method="nearest",
                time_name="time",
                lat_name="latitude",
                lon_name="longitude",
            )
        
        if gpcpv1pt3_df.empty:
            print("Warning: Failed to extract GPCP v1.3 data, creating empty dataframe")
            gpcpv1pt3_df = pd.DataFrame(columns=['date', 'GPCP v1.3'])
        else:
            gpcpv1pt3_df = gpcpv1pt3_df[['time','precip']].copy()
            gpcpv1pt3_df.columns = ['time', 'GPCP v1.3']
            gpcpv1pt3_df['date'] = gpcpv1pt3_df['time'].dt.date

        
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
        # imerg_v06_df = extract_point_timeseries_to_df(
        #         imerg_v06_al,
        #         b_lat,
        #         b_lon,
        #         df_t_min,
        #         df_t_max,
        #         varnames=None,          # <- can be "precip" or ["precip"] or ("precip", "probability_liquid_phase")
        #         method="nearest",
        #         time_name="time",
        #         lat_name="lat",
        #         lon_name="lon",
        #     )
        # if imerg_v06_df.empty:
        #     print("Warning: Failed to extract IMERG v06 data, creating empty dataframe")
        #     imerg_v06_df = pd.DataFrame(columns=['date', 'IMERG v06'])
        # else:
        #     imerg_v06_df = imerg_v06_df[['time','precipitation']].copy()
        #     imerg_v06_df.columns = ['time', 'IMERG v06']
        #     imerg_v06_df['date'] = imerg_v06_df['time'].dt.date
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
            merra2_df = merra2_df[['time','PRECTOT']].copy()
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
        # merge GPCP v1.3 data
        b_df_combined_rain = b_df_combined_rain.merge(
            gpcpv1pt3_df[['date','GPCP v1.3']], 
            left_index=True, right_index=True, how='left', suffixes=('', '_v1pt3')
        )
        # Remove any duplicate columns from previous merges
        b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
                                                ['GPCP v1.3_v1.3', 'date_v1pt3']], 
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

        # # merge IMERG data - merge on 'date' column instead of index
        # b_df_combined_rain = b_df_combined_rain.merge(
        #     imerg_v06_df[['date', 'IMERG v06']], 
        #     left_index=True, right_index=True, how='left', suffixes=('', '_IMERG v06')
        # )
        # # Remove any duplicate columns from previous merges
        # b_df_combined_rain.drop(columns=[i for i in b_df_combined_rain.columns if i in \
        #                                         ['IMERG v06_IMERG v06', 'date_IMERG v06']], 
        #                                         inplace=True)

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
                                                                        'GPCP v1.3',                                                                         
                                                                        'GPCP v3.2', 
                                                                        'GPCP v3.3', 
                                                                        # 'IMERG v06',
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

#%% When data is not available, read them from disk
pal_sate_daily_mean_df = globals().get("pal_sate_daily_mean_df", None)
pal_sate_daily_rainfall_colasped_df = globals().get("pal_sate_daily_rainfall_colasped_df", None)
buoy_sate_daily_mean_df = globals().get("buoy_sate_daily_mean_df", None)
buoy_sate_daily_rainfall_colasped_df = globals().get("buoy_sate_daily_rainfall_colasped_df", None)

if pal_sate_daily_mean_df is None:
    pal_sate_daily_mean_df = pd.read_pickle(
        os.path.join(path_to_put_dfs, "pal_sate_daily_mean_from_all_regions_and_all_tracks_20260401.pkl")
    )

if pal_sate_daily_rainfall_colasped_df is None:
    pal_sate_daily_rainfall_colasped_df = pd.read_pickle(
        os.path.join(path_to_put_dfs, "pal_sate_daily_rainfall_from_all_regions_and_all_tracks_20260401.pkl")
    )

if buoy_sate_daily_mean_df is None:
    buoy_sate_daily_mean_df = pd.read_pickle(
        os.path.join(path_to_put_dfs, "buoy_sate_daily_mean_from_all_regions_and_all_IDs_20260407.pkl")
    )

if buoy_sate_daily_rainfall_colasped_df is None:
    buoy_sate_daily_rainfall_colasped_df = pd.read_pickle(
        os.path.join(path_to_put_dfs, "buoy_sate_daily_rainfall_from_all_regions_and_all_IDs_20260407.pkl")
    )

pal_sate_daily_rainfall_colasped_df = pal_sate_daily_rainfall_colasped_df.rename(
    columns={'MERRA2':'MERRA-2'})  # Rename MERRA2 to MERRA-2 for consistency

buoy_sate_daily_rainfall_colasped_df = buoy_sate_daily_rainfall_colasped_df.rename(
    columns={'MERRA2':'MERRA-2'})  # Rename MERRA2 to MERRA-2 for consistency
#%% Daily-Scale Assessment: SCATTER PLOT of Multi-Year Means

# Create scatter plots for PAL vs satellite products
plot_prdtc = ['GPCP v1.3', 'GPCP v3.2', 'GPCP v3.3', 'IMERG v07','ERA5', 'MERRA-2']
scatter_fig = plot_satellite_vs_groundtruth(pal_sate_daily_mean_df,
                                            truth_col='rain_rate',
                                            product_cols=plot_prdtc,
                                            product_labels=plot_prdtc,
                                            truth_label='PAL Observations',
                                            max_val=18,
                                            ticks=(0, 6, 12, 18),
                                            figsize_per_col=6,
                                            figsize_per_row=5,
                                            region_markers=PAL_region_markers,
                                            region_colors=PAL_region_colors,
                                            region_labels=PAL_REGION_NAMES,
                                            savepath=os.path.join(path_to_plots, f'PAL_vs_Satellite_Comparison_{cde_run_dte}.png'))


# Scatter Plot
print('Starting buoy-based assessment...')
# THE BUOY BASED ASSESSMENT

#- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 
scatter_fig_buoy = plot_satellite_vs_groundtruth(buoy_sate_daily_mean_df,
                                            truth_col='rain_rate',
                                            product_cols=plot_prdtc, # ','GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA2'], # 
                                            product_labels=plot_prdtc, # 
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

#- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 

fig3 = plot_combined_daily_mean_pal_buoy_scatter_black(
    pal_df=pal_sate_daily_mean_df,
    buoy_df=buoy_sate_daily_mean_df,
    truth_col="rain_rate",
    product_cols=plot_prdtc,
    product_labels=[
        "GPCP v1.3",
        "GPCP v3.2",
        "GPCP v3.3",
        "IMERG v07",
        "ERA5",
        "MERRA-2",
    ],

    pal_truth_label="PAL Observations",
    buoy_truth_label="Buoy Observations",
    pal_max_val=18,
    buoy_max_val=15,
    pal_ticks=(0, 6, 12, 18),
    buoy_ticks=(0, 5, 10, 15),
    figsize=(18, 20),
    point_size=70,
    point_alpha=0.85,
    savepath=os.path.join(
        path_to_plots,
        f"Figure03_DailyMean_PAL_Buoy_ProductComparison_{cde_run_dte}.png"
    ),

)
gc.collect()
#%% Daily-Scale Assessment: PAL-Buoy Bar Plot of Regional Metrics

pal_region_based_cat_metrics = {}
pal_region_based_qt_metrics = {}

for region_name in pal_sate_daily_rainfall_colasped_df['region'].unique():

    region_df = pal_sate_daily_rainfall_colasped_df[pal_sate_daily_rainfall_colasped_df['region'] == region_name]
    region_df = region_df.rename(columns = {'MERRA2':'MERRA-2'})  # Rename MERRA2 to MERRA-2 for consistency

    # region_name = region_df['region'].unique()[0]

    print(f"Processing region: {region_name}")

    for product in ['GPCP v1.3','GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA-2']:
        forcast = region_df[product]
        observed = region_df['rain_rate']

        # Calculate the categorical metrics
        reg_cat_met = categorical_stats(forcast, observed, 1.0)
        # Calculate the quantitative metrics
        reg_qt_met = calculate_metrics(region_df, 'rain_rate', product)

        # Store the metrics in the dictionary
        pal_region_based_cat_metrics.setdefault(region_name, {})[product] = reg_cat_met
        pal_region_based_qt_metrics.setdefault(region_name, {})[product] = reg_qt_met

#- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 
pal_start = pal_sate_daily_rainfall_colasped_df["date"].min()
pal_end   = pal_sate_daily_rainfall_colasped_df["date"].max()

buoy_sate_daily_rainfall_colasped_df_overlap = buoy_sate_daily_rainfall_colasped_df.copy()
buoy_sate_daily_rainfall_colasped_df_overlap = (
    buoy_sate_daily_rainfall_colasped_df_overlap[
        (buoy_sate_daily_rainfall_colasped_df_overlap["date"] >= pal_start) &
        (buoy_sate_daily_rainfall_colasped_df_overlap["date"] <= pal_end)
    ]
    .copy()
)
buoy_region_based_cat_metrics = {}
buoy_region_based_qt_metrics = {}

for region_name in buoy_sate_daily_rainfall_colasped_df_overlap['region'].unique():

    region_df = buoy_sate_daily_rainfall_colasped_df_overlap[buoy_sate_daily_rainfall_colasped_df_overlap['region'] == region_name]
    region_df = region_df.rename(columns = {'MERRA2':'MERRA-2'})  # Rename MERRA2 to MERRA-2 for consistency

    # region_name = region_df['region'].unique()[0]

    print(f"Processing region: {region_name}")

    for product in ['GPCP v1.3','GPCP v3.2', 'GPCP v3.3', 'ERA5', 'IMERG v07', 'MERRA-2']:
        forcast = region_df[product]
        observed = region_df['rain_rate']

        # Calculate the categorical metrics
        reg_cat_met = categorical_stats(forcast, observed, 1.0)
        # Calculate the quantitative metrics
        reg_qt_met = calculate_metrics(region_df, 'rain_rate', product)

        # Store the metrics in the dictionary
        buoy_region_based_cat_metrics.setdefault(region_name, {})[product] = reg_cat_met
        buoy_region_based_qt_metrics.setdefault(region_name, {})[product] = reg_qt_met
#- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 

# df_pal_cat  = nested_metrics_to_tidy(pal_region_based_cat_metrics,  "PAL",  PAL_region_bounds)
# df_buoy_cat = nested_metrics_to_tidy(buoy_region_based_cat_metrics, "Buoy", Buoy_region_bounds)

# # quantitative
# df_pal_qnt  = nested_metrics_to_tidy(pal_region_based_qt_metrics,  "PAL",  PAL_region_bounds)
# df_buoy_qnt = nested_metrics_to_tidy(buoy_region_based_qt_metrics, "Buoy", Buoy_region_bounds)

# # combined
# df_cat = pd.concat([df_pal_cat, df_buoy_cat], ignore_index=True)
# df_cat = df_cat.copy()
# df_cat["metric"] = df_cat["metric"].replace({"Bias": "Bias_det"})

# df_qnt = pd.concat([df_pal_qnt, df_buoy_qnt], ignore_index=True)
# #- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 

# cat_metrics_to_plot = ("POD", "FAR", "Bias_det", "HSS")
# savepath = os.path.join(path_to_plots, f"categorical_4x2_pal_buoy_refined_{cde_run_dte}.png")
# fig, axes = plot_metric_bars_4x2_by_reference(
#     df=df_cat,
#     products=["GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"],
#     product_colors=product_colors,
#     metrics=cat_metrics_to_plot,
#     figsize=(24, 18),
#     bar_width=0.105,
#     group_gap=0.24,
#     max_yticks=4,
#     savepath=savepath
# )
# plt.show()
# gc.collect()
# #- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 

# quant_metrics_to_plot = ("CC", "RMSE", "MAE", "Bias")
# savepath = os.path.join(path_to_plots, f"quantitative_4x2_pal_buoy_refined_{cde_run_dte}.png")
# fig, axes = plot_metric_bars_4x2_by_reference(
#     df=df_qnt,
#     products=["GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA2"],
#     product_colors=product_colors,
#     metrics=quant_metrics_to_plot,
#     figsize=(24, 18),
#     bar_width=0.105,
#     group_gap=0.24,
#     max_yticks=4,
#     savepath="quantitative_4x2_pal_buoy_refined.png"
# )
# plt.show()

#- - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - --- - -- - - - - - - - 
# # A comparative spatial plot of metrics

# # Categorical Metrics Plot
# cat_metrics_to_plot = ["POD", "FAR", "Bias_det", "HSS"]
# savepath = os.path.join(path_to_plots, f"spatial_categorical_skill_refined_{cde_run_dte}.png")
# fig, axes = plot_spatial_metric_panels(
#     df=df_cat,
#     metrics=cat_metrics_to_plot,
#     reference_types=("PAL", "Buoy"),
#     figsize=(16, 14),
#     savepath="spatial_categorical_skill_refined.png"
# )
# plt.show()

# savepath = os.path.join(path_to_plots, f"spatial_quantitative_skill_refined_{cde_run_dte}.png")
# quant_metrics_to_plot = ["CC", "RMSE", "MAE", "Bias"]

# fig, axes = plot_spatial_metric_panels(
#     df=df_qnt,
#     metrics=quant_metrics_to_plot,
#     reference_types=("PAL", "Buoy"),
#     figsize=(16, 14),
#     savepath="spatial_quantitative_skill_refined.png"
# )
# plt.show()
# gc.collect()

#-- - --- - -- - --- - - -- - --- - -- - --- - -- - --- - -- - --- - -- - --- - -- - --- - -- - --- - -- - --- - -- - --
# another alternative spatial plot
obs_df = build_obs_df_for_representative_locations(
    pals_classed_by_region=pals_classed_by_region,
    buoy_files_by_region=buoy_files_by_region,
    pal_stride=25
)

rep_locs_df = compute_region_representative_locations(
    obs_df=obs_df,
    method="median"
)
#===================Categorical Metrics==========
df_pal_cat = nested_metrics_to_tidy_with_replocs(
    metrics_dict=pal_region_based_cat_metrics,
    reference_type="PAL",
    rep_locs_df=rep_locs_df
)

df_buoy_cat = nested_metrics_to_tidy_with_replocs(
    metrics_dict=buoy_region_based_cat_metrics,
    reference_type="Buoy",
    rep_locs_df=rep_locs_df
)

df_pal_cat["metric"] = df_pal_cat["metric"].replace({"Bias": "FreqBias"})
df_buoy_cat["metric"] = df_buoy_cat["metric"].replace({"Bias": "FreqBias"})

df_cat_plot = pd.concat([df_pal_cat, df_buoy_cat], ignore_index=True)

#===================Quantitative Metrics==========
df_pal_qnt = nested_metrics_to_tidy_with_replocs(
    metrics_dict=pal_region_based_qt_metrics,
    reference_type="PAL",
    rep_locs_df=rep_locs_df
)

df_buoy_qnt = nested_metrics_to_tidy_with_replocs(
    metrics_dict=buoy_region_based_qt_metrics,
    reference_type="Buoy",
    rep_locs_df=rep_locs_df
)

df_qnt_plot = pd.concat([df_pal_qnt, df_buoy_qnt], ignore_index=True)

#===================Plotting==========
# Categorical Metrics Plot with Context
cat_metrics = ["POD", "FAR", "FreqBias", "HSS"]
svname = os.path.join(path_to_plots, f"FigureS05_DailyDetectionSkillMaps_PAL_Buoy_AllMetrics_{cde_run_dte}.png")
metric_style = make_metric_style_dict()

fig, axes, ax_context = plot_spatial_skill_panels_with_context(
    df=df_cat_plot,   # your tidy dataframe
    metrics=["POD", "FAR", "FreqBias", "HSS"],   # use FreqBias if your metric column is named that
    pals_classed_by_region=pals_classed_by_region,
    buoy_files_by_region=buoy_files_by_region,
    PAL_region_colors=PAL_region_colors,
    metric_style=metric_style,
    figsize=(16, 15.2),
    savepath=None
)
# plt.show()
fig.savefig(svname, dpi=300, bbox_inches='tight')

gc.collect()

# Quantitative Metrics Plot with Context

svname = os.path.join(path_to_plots, f"FigureS06_DailyQuantitativeSkillMaps_PAL_Buoy_AllMetrics_{cde_run_dte}.png")
qnt_metrics = ["CC", "RMSE", "MAE", "Bias"]
metric_style = make_metric_style_dict()

fig, axes, ax_context = plot_spatial_skill_panels_with_context(
    df=df_qnt_plot,   # your tidy dataframe
    metrics=["CC", "RMSE", "MAE", "Bias"],   # use Bias if your metric column is named that
    pals_classed_by_region=pals_classed_by_region,
    buoy_files_by_region=buoy_files_by_region,
    PAL_region_colors=PAL_region_colors,
    metric_style=metric_style,
    figsize=(16, 15.2),
    savepath=None
)
fig.savefig(svname, dpi=300, bbox_inches='tight')
gc.collect()

#------------------------------------------------------------------------------
metric_style = make_metric_style_dict()

# If your categorical detection bias style was previously stored as "Bias",
# reuse it for FreqBias but relabel it.
# if "FreqBias" in metric_style:
#     metric_style["FreqBias"] = {
#         **metric_style["Bias"],
#         "label": "Frequency bias",
#     }

# Make sure relative bias is labeled clearly.
metric_style["Bias"] = {
    **metric_style["Bias"],
    "label": "Relative bias [%]",
}

fig4, axes4 = plot_main_daily_skill_maps_pal_buoy_six_metrics(
    cat_df=df_cat_plot,
    quant_df=df_qnt_plot,
    metric_style=metric_style,
    figsize=(18, 20),
    marker_size=135,
    savepath=os.path.join(
        path_to_plots,
        f"Figure05_DailySkill_Regional_PAL_Buoy_SixMetrics_{cde_run_dte}.png"
    ),
)

gc.collect()
#%% Daily Assessment: Metrics as a fucntion of intensity

rainfall_bins = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
products = ['GPCP v1.3', 'GPCP v3.2', 'GPCP v3.3', 'IMERG v07', 'ERA5', 'MERRA-2']

pal_cat, pal_qt = compute_metrics_by_intensity_for_df(
    pal_sate_daily_rainfall_colasped_df.copy(),
    products=products,
    rainfall_bins=rainfall_bins,
    obs_col="rain_rate"
)

buoy_cat, buoy_qt = compute_metrics_by_intensity_for_df(
    buoy_sate_daily_rainfall_colasped_df_overlap.copy(),
    products=products,
    rainfall_bins=rainfall_bins,
    obs_col="rain_rate"
)
savepath = os.path.join(path_to_plots, f"cat_metrics_by_intensity_pal_vs_buoy_{cde_run_dte}.png")
fig1, axes1 = plot_intensity_metrics_cat_4x2(
    pal_cat=pal_cat,
    buoy_cat=buoy_cat,
    rainfall_bins=rainfall_bins,
    products=products,
    product_colors=product_colors,
    figsize=(16, 16),
    savepath=savepath
)
gc.collect()

savepath = os.path.join(path_to_plots, f"qt_metrics_by_intensity_pal_vs_buoy_{cde_run_dte}.png")
fig2, axes2 = plot_intensity_metrics_qt_4x2(
    pal_qt=pal_qt,
    buoy_qt=buoy_qt,
    rainfall_bins=rainfall_bins,
    products=products,
    product_colors=product_colors,
    figsize=(16, 16),
    savepath=savepath
)

gc.collect()

#------------------------------------------------------------------------------
savepath = os.path.join(
    path_to_plots,
    f"Figure06_DailySkill_IntensityDependence_PAL_Buoy_SelectedMetrics_{cde_run_dte}.png"
)

fig6, axes6 = plot_intensity_metrics_selected_pal_buoy_4x2(
    pal_cat=pal_cat,
    buoy_cat=buoy_cat,
    pal_qt=pal_qt,
    buoy_qt=buoy_qt,
    rainfall_bins=[0.5, 1, 2, 4, 8, 16, 32],
    products=["GPCP v1.3", "GPCP v3.2", "GPCP v3.3", 
              "IMERG v07", "ERA5", "MERRA-2"],
    product_colors=product_colors,
    figsize=(17, 16),
    linewidth=3.2,
    markersize=7.5,
    savepath=savepath,
)
gc.collect()
#%% #%%  Daily-Scale Assessment: PDF Assessment

bin_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
savepath=os.path.join(path_to_plots, f"Figure04_DailyRainfall_VolumePDF_PAL_Buoy_ProductComparison_{cde_run_dte}.png")
fig, axes, pal_pdf_dict, buoy_pdf_dict = plot_pdf_comparison_pal_buoy(
    pal_df=pal_sate_daily_rainfall_colasped_df,
    buoy_df=buoy_sate_daily_rainfall_colasped_df,
    obs_col="rain_rate",
    products=("GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA-2"),
    product_colors=product_colors,
    bin_values=bin_values,
    pdf_kind="pdfv",                 # use "pdfc" if needed
    pal_year_range=None,
    buoy_year_range=(2000, 2020),
    figsize=(16, 6),
    dpi=150,
    lw=4,
    savepath=savepath
    
)
gc.collect()


#%% Monthly to Interannual Variability: A Buoy-Supported Assessment

gpcp_v2pt3_mnthly_ds_path = r'/ra1/pubdat/GPCP/v2.3'

gpcp_v3pt2_mnthly_ds_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_monthly_V3.2_1983_2023'

gpcp_v3pt3_mnthly_ds_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_monthly_1983_2024'

era5_mnhtly_file = r'/ra1/pubdat/GPCP/GPCP_Reproduce_GJ/era5_tp_198001202412_monthly.nc'

mer2_mnthly_files = r'/ra1/pubdat/MERRA/Monthly_complete'

imerg_mnthly_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7_monthly'

all_gpcp2pt3_mnthly_files = sorted([os.path.join(gpcp_v2pt3_mnthly_ds_path, f) for f in os.listdir(gpcp_v2pt3_mnthly_ds_path) if ('preliminary' not in f) and (f.endswith('.nc'))])
all_gpcp2pt3_mnthly_files_ = [f for f in all_gpcp2pt3_mnthly_files  if int(os.path.basename(f).split('_')[3].replace('d','')[:4]) >= 1998]

all_gpcp_v3pt2_mnthly_files = sorted([os.path.join(gpcp_v3pt2_mnthly_ds_path, f) for f in os.listdir(gpcp_v3pt2_mnthly_ds_path) if f.endswith('.nc4')])
all_gpcp_v3pt2_mnthly_files_ = [f for f in all_gpcp_v3pt2_mnthly_files  if int(os.path.basename(f).split('_')[2][:4]) >= 1998]

all_gpcp_v3pt3_mnthly_files = sorted([os.path.join(gpcp_v3pt3_mnthly_ds_path, f) for f in os.listdir(gpcp_v3pt3_mnthly_ds_path) if f.endswith('.nc4')])
all_gpcp_v3pt3_mnthly_files_ = [f for f in all_gpcp_v3pt3_mnthly_files  if int(os.path.basename(f).split('_')[2][:4]) >= 1998]

all_mer2_mnthly_files = sorted([os.path.join(mer2_mnthly_files, f) for f in os.listdir(mer2_mnthly_files) if f.endswith('.nc4')])
all_mer2_mnthly_files_ = [f for f in all_mer2_mnthly_files  if int(os.path.basename(f).split('.')[5][:4]) >= 1998]

all_imerg_mnthly_files = sorted([os.path.join(imerg_mnthly_files, f) for f in os.listdir(imerg_mnthly_files) if f.endswith('.HDF5')])

#%% Load monthly data files

gpcp_v2pt3_mnth_ds = xr.open_mfdataset(all_gpcp2pt3_mnthly_files_, 
                                        combine="nested",              # files are time-sequenced
                                        concat_dim="time",             # concatenate along time
                                        data_vars="minimal",           # don't unnecessarily align data_vars
                                        coords="minimal",
                                        compat="override",
                                        parallel=True,
                                        engine="netcdf4",
                                        chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                        cache=False)[['precip']]
gpcp_v2pt3_mnth_ds = ds_swaplon(gpcp_v2pt3_mnth_ds)

gpcp_v3pt2_mnth_ds  = xr.open_mfdataset(all_gpcp_v3pt2_mnthly_files_, 
                                        combine="nested",              # files are time-sequenced
                                        concat_dim="time",             # concatenate along time
                                        data_vars="minimal",           # don't unnecessarily align data_vars
                                        coords="minimal",
                                        compat="override",
                                        parallel=True,
                                        engine="netcdf4",
                                        chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                        cache=False)[['sat_gauge_precip']]

gpcp_v3pt3_mnth_ds  = xr.open_mfdataset(all_gpcp_v3pt3_mnthly_files_, 
                                        combine="nested",              # files are time-sequenced
                                        concat_dim="time",             # concatenate along time
                                        data_vars="minimal",           # don't unnecessarily align data_vars
                                        coords="minimal",
                                        compat="override",
                                        parallel=True,
                                        engine="netcdf4",
                                        chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                        cache=False)[['sat_gauge_precip']]

mer2_ds_mnth_ds = xr.open_mfdataset(all_mer2_mnthly_files_, 
                                        combine="nested",              # files are time-sequenced
                                        concat_dim="time",             # concatenate along time
                                        data_vars="minimal",           # don't unnecessarily align data_vars
                                        coords="minimal",
                                        compat="override",
                                        parallel=True,
                                        engine="netcdf4",
                                        chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                        cache=False)[['PRECTOT']]

mer2_ds_mnth_ds = mer2_ds_mnth_ds['PRECTOT'] * 3600
mer2_ds_mnth_ds = mer2_ds_mnth_ds * 24  # convert to mm/day
# Ensure time is monotonic (required for monthly resampling)
mer2_ds_mnth_ds = mer2_ds_mnth_ds.sortby("time")

# Compute monthly means
mer2_ds_mnth_ds = mer2_ds_mnth_ds.resample(time="MS").mean()

# Add a time dimension based on the file name or metadata
# resample to 0.5 degree resolution
cc = CRS.from_authority(code=4326, auth_name='EPSG')
mer2_ds_mnth_ds.rio.write_crs(cc.to_string(), inplace=True)
# Set spatial dimensions explicitly
mer2_ds_mnth_ds = mer2_ds_mnth_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
mer2_ds_mnth_ds = mer2_ds_mnth_ds.rio.reproject(
    mer2_ds_mnth_ds.rio.crs,
    shape=gpcp_v3pt2_mnth_ds['sat_gauge_precip'].shape[1:],  # (360, 720), set the shape as the GPCP data
    resampling=Resampling.average,
)

# era5_mnth_ds = era5_ds_xr.resample(valid_time='1M').mean()
era5_mnth_ds = xr.open_dataset(era5_mnhtly_file, engine='netcdf4')[['tp', 'latitude', 'longitude']]
era5_mnth_ds = ds_swaplon(era5_mnth_ds)
# data units are in m per day, convert to mm/day using 1000 factor
era5_mnth_ds['tp'] = era5_mnth_ds['tp'] * 1000  # mm/h
# resample to 0.5 degree resolution
cc = CRS.from_authority(code=4326, auth_name='EPSG')
era5_mnth_ds.rio.write_crs(cc.to_string(), inplace=True)
era5_mnth_ds = era5_mnth_ds.rio.reproject(
    era5_mnth_ds.rio.crs,
    shape=gpcp_v3pt2_mnth_ds['sat_gauge_precip'].shape[1:], # set the shape as the GPCP data
    resampling=Resampling.average,
)

era5_mnth_ds = era5_mnth_ds.sel(valid_time=slice('1998,01,01',None))['tp']

imerg_v07_mnth_ds_xr_list = []
with Pool(processes=18) as pool:
    imerg_v07_mnth_ds_xr_list = pool.map(
        process_imerg_hdf_file,
        [(idx, file_path) for idx, file_path in enumerate(all_imerg_mnthly_files)]
    )
# Combine all processed batches into a single xarray dataset - simple version
if imerg_v07_mnth_ds_xr_list:
    imerg_v07_mnthly_ds_xr = xr.concat(imerg_v07_mnth_ds_xr_list, dim="time")

imerg_v07_mnthly_ds_xr = imerg_v07_mnthly_ds_xr * 24

print("IMERG loading complete")
print("-" * 50 + "\n")
del(imerg_v07_mnth_ds_xr_list) # imerg_v06_ds_xr_list,
gc.collect()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# find min max dates in all dataset
min_date = pd.to_datetime(min(
    gpcp_v2pt3_mnth_ds['precip'].time.min(),
    gpcp_v3pt2_mnth_ds['sat_gauge_precip'].time.min(), 
    gpcp_v3pt3_mnth_ds['sat_gauge_precip'].time.min(), 
    era5_mnth_ds.valid_time.min(), 
    mer2_ds_mnth_ds.time.min(),
    imerg_v07_mnthly_ds_xr.time.min()).values)
max_date = pd.to_datetime(min(
    gpcp_v2pt3_mnth_ds['precip'].time.max(),    
    gpcp_v3pt2_mnth_ds['sat_gauge_precip'].time.max(), 
    gpcp_v3pt3_mnth_ds['sat_gauge_precip'].time.max(), 
    era5_mnth_ds.valid_time.max(), 
    mer2_ds_mnth_ds.time.max(),
    imerg_v07_mnthly_ds_xr.time.max()).values)
print(f"min_date: {min_date}, max_date: {max_date}")

#%% Spatio temporal monhtly collocation
print('Starting Buoy-GPCP matching...')
print("Starting Buoy-monthly product matching...")

regional_buoy_product_monthly_dict = {}
regional_buoy_product_monthly_list = []

for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
    region_tables = []

    for b, b_file in enumerate(buoy_files):
        b_df, b_lat, b_lon = grab_Buoy_data_df(b_file)

        b_df["time"] = pd.to_datetime(b_df["time"])
        b_df = b_df[
            (b_df["time"] >= min_date) &
            (b_df["time"] <= max_date)
        ].copy()

        if b_df.empty:
            print(f"Warning: {os.path.basename(b_file)} has no data in common date range")
            continue

        # quality filter
        b_df = b_df[(b_df["quality_flag"] >= 1) & (b_df["quality_flag"] <= 3)].copy()
        if b_df.empty:
            continue

        # daily mm/h -> daily mm/day
        b_df["rain_mm_day"] = b_df["rain_rate"] * 24.0
        b_df["month"] = b_df["time"].dt.to_period("M").dt.to_timestamp()

        # monthly buoy table
        b_monthly = (
            b_df.groupby("month", as_index=False)
                .agg(
                    Buoy=("rain_mm_day", "mean"),
                    n_days=("rain_mm_day", "count")
                )
        )

        # require enough daily coverage within month
        b_monthly = b_monthly[b_monthly["n_days"] >= 20].copy()
        if b_monthly.empty:
            continue

        # metadata
        buoy_id = os.path.basename(b_file).split(".")[0]
        loc_df = b_monthly.copy()
        loc_df["region"] = region_name
        loc_df["ID"] = buoy_id
        loc_df["lat"] = b_lat
        loc_df["lon"] = b_lon

        if b % 5 == 0:
            print(f"  Processing buoy file: {os.path.basename(b_file)}")

        # -----------------------------
        # Extract product monthly series
        # -----------------------------
        gpcp23_df = (
            gpcp_v2pt3_mnth_ds["precip"]
            .sel(latitude=float(b_lat), longitude=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp23_df["month"] = pd.to_datetime(gpcp23_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp23_df = gpcp23_df[["month", "precip"]].rename(columns={"precip": "GPCP v2.3"})

        gpcp32_df = (
            gpcp_v3pt2_mnth_ds["sat_gauge_precip"]
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp32_df["month"] = pd.to_datetime(gpcp32_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp32_df = gpcp32_df[["month", "sat_gauge_precip"]].rename(columns={"sat_gauge_precip": "GPCP v3.2"})

        gpcp33_df = (
            gpcp_v3pt3_mnth_ds["sat_gauge_precip"]
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp33_df["month"] = pd.to_datetime(gpcp33_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp33_df = gpcp33_df[["month", "sat_gauge_precip"]].rename(columns={"sat_gauge_precip": "GPCP v3.3"})

        era5_df = (
            era5_mnth_ds
            .sel(y=float(b_lat), x=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        era5_df["month"] = pd.to_datetime(era5_df["valid_time"]).dt.to_period("M").dt.to_timestamp()
        era5_df = era5_df[["month", "tp"]].rename(columns={"tp": "ERA5"})

        imerg_df = (
            imerg_v07_mnthly_ds_xr
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        imerg_df["month"] = pd.to_datetime(imerg_df["time"]).dt.to_period("M").dt.to_timestamp()
        imerg_df = imerg_df[["month", "precipitation"]].rename(columns={"precipitation": "IMERG v07"})

        merra2_df = (
            mer2_ds_mnth_ds
            .sel(y=float(b_lat), x=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        merra2_df["month"] = pd.to_datetime(merra2_df["time"]).dt.to_period("M").dt.to_timestamp()
        merra2_df = merra2_df[["month", "PRECTOT"]].rename(columns={"PRECTOT": "MERRA2"})

        # -----------------------------
        # Merge all products to one table
        # -----------------------------
        for prod_df in [gpcp23_df, gpcp32_df, gpcp33_df, era5_df, imerg_df, merra2_df]:
            loc_df = loc_df.merge(prod_df, on="month", how="left")

        # optional year/month columns
        loc_df["year"] = pd.to_datetime(loc_df["month"]).dt.year
        loc_df["month_num"] = pd.to_datetime(loc_df["month"]).dt.month

        region_tables.append(loc_df)
        regional_buoy_product_monthly_list.append(loc_df)

    regional_buoy_product_monthly_dict[region_name] = region_tables

# combine all buoy monthly tables
all_buoy_product_monthly_df = pd.concat(
    regional_buoy_product_monthly_list,
    ignore_index=True
) if regional_buoy_product_monthly_list else pd.DataFrame()

all_buoy_product_monthly_df = all_buoy_product_monthly_df.rename(columns={
    "MERRA2": "MERRA-2"
})
# save all_buoy_product_monthly_df to disk
all_buoy_product_monthly_df.to_pickle(os.path.join(path_to_put_dfs, f'buoy_monthly_df_{cde_run_dte}.pkl'))

gc.collect() 

#%% Monthly to Interannual Variability: Monthly Clim Cycles
all_buoy_product_monthly_df = pd.read_pickle(os.path.join(path_to_put_dfs, 'buoy_monthly_df_20260407.pkl'))
all_buoy_product_monthly_df = all_buoy_product_monthly_df.rename(columns={
    "MERRA2": "MERRA-2"
})
products = [
    "Buoy",
    "GPCP v2.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",   
    "MERRA-2",
]

monthly_clim_by_region = compute_monthly_climatology_from_monthly_buoy_df(
    all_buoy_product_monthly_df,
    products=products,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="Buoy",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=5,      # set None if you do not want this filter yet
    equal_weight_by_buoy=True   # recommended
)

monthly_clim_by_region_plot = rename_monthnum_for_plotting(monthly_clim_by_region)

regions = list(Buoy_region_markers.keys())
buoy_monthly_products = [
    "Buoy",
    "GPCP v2.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",    
    "MERRA-2",
]

fig = plot_monthly_climatology_2x2(
    monthly_clim_by_region=monthly_clim_by_region_plot,
    regions=regions,
    region_labels=Buoy_REGION_NAMES,
    products=buoy_monthly_products,
    product_colors=product_colors,
    figsize=(12, 9),
    lw=3.5,
    ncol_legend=4
)

svnme = os.path.join(
    path_to_plots,
    f'Figure07_BuoyMonthly_SeasonalCycle_ProductComparison_{cde_run_dte}.png'
)
fig.savefig(svnme, dpi=150, bbox_inches='tight')
gc.collect()
#%% Interanual Variability
# ============================================================
# BUILD ANNUAL SERIES FROM MONTHLY-SCREENED BUOY-PRODUCT TABLE
# ============================================================
buoy_monthly_products = [
    "Buoy",
    "GPCP v2.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

annual_by_region, annual_buoy_product_df = build_annual_from_monthly_buoy_df(
    all_buoy_product_monthly_df,
    products=buoy_monthly_products,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="Buoy",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=1, # 2
    min_months_per_year=10, # 4     # can change to 10 if you want stricter
    equal_weight_by_buoy=True
)

# ============================================================
# PLOT INTERANNUAL VARIABILITY
# ============================================================
regions = list(Buoy_region_markers.keys())
region_year_limits = {
    "ENP": (1998, 2013),
    "WNP": (1998, 2020),
    "IND": (2008, 2025),
    "ATL": (2003, 2025),   # or (2001, 2025) if you want to remove the early spike more aggressively
}

fig = plot_interannual_variability_2x2_from_monthly_df(
    annual_df=annual_buoy_product_df,
    regions=regions,
    products=buoy_monthly_products,
    product_colors=product_colors,
    ref="Buoy",
    region_labels=Buoy_REGION_NAMES,
    figsize=(17, 9.5),
    lw_ref=3.5,
    lw_prod=3.0,
    ncol_legend=5,
    year_min=1998, #min_date.year,
    year_max=2022, #max_date.year,
    region_year_limits=region_year_limits,
)

svnme = os.path.join(
    path_to_plots,
    f'Buoy_vs_Satellite_Annual_Interannual_Variability_2x2_{cde_run_dte}.png'
)
fig.savefig(svnme, dpi=300, bbox_inches='tight')

gc.collect()

#---------------------------------------------------------------------------
# PLOT INTERANNUAL VARIABILITY WITH SAMPLE COUNTS
#--------------------------------------------------------------------------
annual_by_region_buoy_sm, annual_df_buoy_sm = build_annual_from_monthly_buoy_df_with_sample_counts(
    df=all_buoy_product_monthly_df,
    products=buoy_monthly_products,
    buoy_col="Buoy",
    min_days_per_month=20,
    min_buoys_per_month=2,
    min_months_per_year=4,
    year_min=min_date.year,
    year_max=max_date.year,
    region_year_limits=region_year_limits,
)

fig = plot_interannual_variability_with_sample_counts_2x2(
    annual_df=annual_df_buoy_sm,
    regions=regions,
    products=buoy_monthly_products,
    product_colors=product_colors,
    ref="Buoy",
    count_col="n_valid_buoy_months",
    count_label="Sample Count",
    right_ylabel="Sample Count (Months)",
    count_ylim=None,   # auto-scale
    ncol_legend=5,
)

svnme = os.path.join(
    path_to_plots,
    f"Figure08_Buoy_AnnualRegionalPrecip_ProductComparison_{cde_run_dte}.png"
)
fig.savefig(svnme, dpi=150, bbox_inches='tight')

gc.collect()

#%% INTERANNUAL VARIABILITY: PRODUCT BASED COMPARISON
print('Starting Product-based matching...')

regional_product_monthly_dict = {}
regional_product_monthly_list = []

for region_name, buoy_files in buoy_files_by_region.items():
    print(f"Processing region: {region_name}")
    region_tables = []

    for b, b_file in enumerate(buoy_files):
        # define a unique buoy ID for THIS file
        buoy_id = os.path.splitext(os.path.basename(b_file))[0]
        b_df, b_lat, b_lon = grab_Buoy_data_df(b_file)           
        # -----------------------------
        # Extract product monthly series
        # -----------------------------
        gpcp23_df = (
            gpcp_v2pt3_mnth_ds["precip"]
            .sel(time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(latitude=float(b_lat), longitude=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp23_df["month"] = pd.to_datetime(gpcp23_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp23_df = gpcp23_df[["month", "precip"]].rename(columns={"precip": "GPCP v2.3"})

        gpcp32_df = (
            gpcp_v3pt2_mnth_ds["sat_gauge_precip"]
            .sel(time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp32_df["month"] = pd.to_datetime(gpcp32_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp32_df = gpcp32_df[["month", "sat_gauge_precip"]].rename(columns={"sat_gauge_precip": "GPCP v3.2"})

        gpcp33_df = (
            gpcp_v3pt3_mnth_ds["sat_gauge_precip"]
            .sel(time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        gpcp33_df["month"] = pd.to_datetime(gpcp33_df["time"]).dt.to_period("M").dt.to_timestamp()
        gpcp33_df = gpcp33_df[["month", "sat_gauge_precip"]].rename(columns={"sat_gauge_precip": "GPCP v3.3"})

        era5_df = (
            era5_mnth_ds
            .sel(valid_time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(y=float(b_lat), x=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        era5_df["month"] = pd.to_datetime(era5_df["valid_time"]).dt.to_period("M").dt.to_timestamp()
        era5_df = era5_df[["month", "tp"]].rename(columns={"tp": "ERA5"})

        imerg_df = (
            imerg_v07_mnthly_ds_xr
            .sel(time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(lat=float(b_lat), lon=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        imerg_df["month"] = pd.to_datetime(imerg_df["time"]).dt.to_period("M").dt.to_timestamp()
        imerg_df = imerg_df[["month", "precipitation"]].rename(columns={"precipitation": "IMERG v07"})

        merra2_df = (
            mer2_ds_mnth_ds
            .sel(time=slice(min_date, max_date))  # ensure we only select the time range that overlaps with buoy data
            .sel(y=float(b_lat), x=float(b_lon), method="nearest")
            .to_dataframe()
            .reset_index()
        )
        merra2_df["month"] = pd.to_datetime(merra2_df["time"]).dt.to_period("M").dt.to_timestamp()
        merra2_df = merra2_df[["month", "PRECTOT"]].rename(columns={"PRECTOT": "MERRA2"})

        # -----------------------------
        # Merge all products to one table
        # -----------------------------
        products_dfs = gpcp23_df.copy()
        for prod_df in [gpcp32_df, gpcp33_df, era5_df, imerg_df, merra2_df]:
            products_dfs = products_dfs.merge(prod_df, on="month", how="left")

        # optional year/month columns
        products_dfs["year"] = pd.to_datetime(products_dfs["month"]).dt.year
        products_dfs["month_num"] = pd.to_datetime(products_dfs["month"]).dt.month

        # add region column
        products_dfs["region"] = region_name

        # add buoy ID column
        products_dfs["ID"] = buoy_id

        region_tables.append(products_dfs)
        regional_product_monthly_list.append(products_dfs)

    regional_product_monthly_dict[region_name] = region_tables

# combine all product monthly tables
prodt_bsed_all_product_monthly_df = pd.concat(
    regional_product_monthly_list,
    ignore_index=True
) if regional_product_monthly_list else pd.DataFrame()

prodt_bsed_all_product_monthly_df = prodt_bsed_all_product_monthly_df.rename(columns={
    "MERRA2": "MERRA-2"})

# save all_product_monthly_df to disk
prodt_bsed_all_product_monthly_df.to_pickle(os.path.join(path_to_put_dfs, f'product_monthly_df_{cde_run_dte}.pkl'))

prodt_bsed_all_product_monthly_df = globals().get("prodt_bsed_all_product_monthly_df", None)

if prodt_bsed_all_product_monthly_df is None:
    prodt_bsed_all_product_monthly_df = pd.read_pickle(
        os.path.join(path_to_put_dfs, "product_monthly_df_20260423.pkl")
    )

gc.collect() 

monthly_products = [  
    "GPCP v2.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA2",
]
# prudtc_dfs = all_buoy_product_monthly_df.copy()
# prudtc_dfs.drop(columns = ['Buoy', 'n_days','ID',], inplace=True)
# reg_dfs = []
# for ke in regional_product_monthly_dict.keys():
#      reg_df = pd.concat(regional_product_monthly_dict[ke], ignore_index=True)
#      reg_df['region'] = ke
#      reg_dfs.append(reg_df)
# all_product_monthly_df = pd.concat(reg_dfs, ignore_index=True)



annual_by_region, annual_product_df = build_annual_from_monthly_buoy_df(
    prodt_bsed_all_product_monthly_df,
    products=monthly_products,
    region_col="region",
    id_col="ID",
    month_col="month",
    buoy_col="GPCP v3.3",
    n_days_col="n_days",
    min_days_per_month=20,
    min_buoys_per_month=None,
    min_months_per_year=12,      # can change to 10 if you want stricter
    equal_weight_by_buoy=False
)

# ann_df = all_product_monthly_df.groupby(['region', 'year'])[monthly_products].mean().reset_index()
region_year_limits = {
    "ENP": (1998, 2013),
    "WNP": (1998, 2020),
    "IND": (2008, 2025),
    "ATL": (2003, 2025),   # or (2001, 2025) if you want to remove the early spike more aggressively
}
fig = plot_interannual_variability_2x2_from_monthly_df(
    annual_df=annual_product_df,
    regions=regions,
    products=monthly_products,
    product_colors=product_colors,
    ref="GPCP v3.3",
    region_labels=Buoy_REGION_NAMES,
    figsize=(17, 9.5),
    lw_ref=3.5,
    lw_prod=3.5,
    ncol_legend=6,
    year_min=1998,
    year_max=2024,
    region_year_limits=region_year_limits,
)
gc.collect()


#---------------------------------------------------------------------------
# PLOT INTERANNUAL VARIABILITY WITH SAMPLE COUNTS
#--------------------------------------------------------------------------
annual_by_region_prdt_sm, annual_df_prdt_sm = build_annual_from_monthly_buoy_df_with_sample_counts(
    df=prodt_bsed_all_product_monthly_df,
    products=monthly_products,
    buoy_col="GPCP v3.3",
    min_days_per_month=None,
    min_buoys_per_month=None,
    min_months_per_year=12,
    year_min=min_date.year,
    year_max=max_date.year,
    region_year_limits=region_year_limits,
)

fig = plot_interannual_variability_with_sample_counts_2x2(
    annual_df=annual_df_prdt_sm,
    regions=regions,
    products=monthly_products,
    product_colors=product_colors,
    ref="GPCP v3.3",
    count_col="n_valid_buoy_months",
    count_label="Sample Count",
    right_ylabel="Sample Count (Months)",
    count_ylim=None,   # auto-scale
    ncol_legend=5,
)
svnme = os.path.join(path_to_plots, f"FigureS5_Buoy_AnnualRegionalPrecip_ProductOnlyAnalog_{cde_run_dte}.png")
fig.savefig(svnme, dpi=150, bbox_inches='tight')
gc.collect()

#%% Interannual Variability: Monthly Anomaly Scatterplots — Product vs Buoy
# ============================================================
# Monthly anomaly scatterplots — Product vs Buoy
# based on MONTHLY buoy-product dataframe
# ============================================================

buoy_monthly_products = [
    "Buoy",
    "GPCP v2.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",    
    "MERRA-2",
]

fig, monthly_region_buoy, monthly_region_buoy_anom = (
    plot_deseasonalized_anomaly_scatter_from_monthly_buoy_df(
        monthly_buoy_df=all_buoy_product_monthly_df,
        products=buoy_monthly_products,
        regions=["ENP", "WNP", "IND", "ATL"],
        ref_col="Buoy",
        product_cols=["GPCP v2.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA-2"],
        region_labels=Buoy_REGION_NAMES,
        product_colors=product_colors,
        region_col="region",
        month_col="month",
        id_col="ID",
        n_days_col="n_days",
        min_days_per_month=20,
        min_buoys_per_month=2,      # recommended main setting
        equal_weight_by_buoy=True,
        figsize=(20, 12),
        savepath=os.path.join(
            path_to_plots,
            f"Figure09_BuoyMonthlyAnomaly_Scatter_ProductComparison_{cde_run_dte}.png"
        )
    )
)

gc.collect()


#%% Poleward Assessment: OceanRAIN (≥45°)

# -------------------------------------------------------------
# OceanRAIN common 1° analysis grid
# -------------------------------------------------------------
# This is the grid used to assign OceanRAIN minute samples to daily
# ship–grid-cell records. It is also the grid used for product
# collocation after all products are harmonized to 1°.

or_grid_1deg = xr.Dataset(
    coords={
        "lat": np.arange(89.5, -90.0, -1.0, dtype=np.float32),
        "lon": np.arange(-179.5, 180.0, 1.0, dtype=np.float32),
    }
)

or_lat_1d = or_grid_1deg["lat"].values
or_lon_1d = or_grid_1deg["lon"].values

df_qc = oceanrain_step0_qc_precip_main(
    ocRain_df,
    drop_harbor_inop=True,
    drop_spurious_flag2_11=True,
    keep_true_zero=True,
    keep_flag2_12_zero_precip=False,
    min_flag2_positive=13,
    prob_thr=None,
    wind_max=None,
    qclip_hi=0.999,
)

# daily_or_all, daily_or_usable = oceanrain_daily_aggregate_to_gpcp_main(
#     oc_df_minute=df_qc,
#     gpcp_lat_1d=or_lat_1d,
#     gpcp_lon_1d=or_lon_1d, 
#     lat_abs_min=45.0,
#     coverage_frac=0.5,
# )
# gc.collect()


# -------------------------------------------------------------
# products harmonized to common 1° support
# -------------------------------------------------------------

gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al['precip'].copy()
gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al_res.where(gpcp_ds_v1pt3_al_res >= 0)

# gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al_res.where(gpcp_ds_v1pt3_al_res != -9999.0)

# gpcp_ds_v1pt3_al_res.rio.write_crs(cc.to_string(), inplace=True)
# gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al_res.rio.set_spatial_dims(x_dim="longitude", 
#                                                                  y_dim="latitude", 
#                                                                  inplace=True)
# gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al_res.rio.reproject(
#     gpcp_ds_v1pt3_al_res.rio.crs,
#     shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
#     resampling=Resampling.average,
# )
# # rename spatial dims back to latlon
# gpcp_ds_v1pt3_al_res = gpcp_ds_v1pt3_al_res.rename({'y': 'lat', 'x': 'lon'})

gpcp_ds_v3pt2_al_res = gpcp_ds_v3pt2_al['precip'].copy()
gpcp_ds_v3pt2_al_res.rio.write_crs(cc.to_string(), inplace=True)
gpcp_ds_v3pt2_al_res = gpcp_ds_v3pt2_al_res.rio.set_spatial_dims(x_dim="lon", 
                                                                 y_dim="lat", 
                                                                 inplace=True)
gpcp_ds_v3pt2_al_res = gpcp_ds_v3pt2_al_res.rio.reproject(
    gpcp_ds_v3pt2_al_res.rio.crs,
    shape=(180, 360),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    resampling=Resampling.average,
)
gpcp_ds_v3pt2_al_res = gpcp_ds_v3pt2_al_res.rename({'y': 'lat', 'x': 'lon'})

gpcp_ds_v3pt2_al_res = gpcp_ds_v3pt2_al_res.where(gpcp_ds_v3pt2_al_res >= 0)

gpcp_ds_v3pt3_al_res = gpcp_ds_v3pt3_al['precip'].copy()
gpcp_ds_v3pt3_al_res.rio.write_crs(cc.to_string(), inplace=True)
gpcp_ds_v3pt3_al_res = gpcp_ds_v3pt3_al_res.rio.set_spatial_dims(x_dim="lon", 
                                                                 y_dim="lat", 
                                                                 inplace=True)
gpcp_ds_v3pt3_al_res = gpcp_ds_v3pt3_al_res.rio.reproject(
    gpcp_ds_v3pt3_al_res.rio.crs,
    shape=(180, 360),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    resampling=Resampling.average,
)
# rename spatial dims back to latlon
gpcp_ds_v3pt3_al_res = gpcp_ds_v3pt3_al_res.rename({'y': 'lat', 'x': 'lon'})
gpcp_ds_v3pt3_al_res = gpcp_ds_v3pt3_al_res.where(gpcp_ds_v3pt3_al_res >= 0)

era5_ds_res = era5_ds_al['tp'].copy()
era5_ds_res.rio.write_crs(cc.to_string(), inplace=True)
era5_ds_res = era5_ds_res.rio.set_spatial_dims(x_dim="x", 
                                                y_dim="y", 
                                                inplace=True)
era5_ds_res = era5_ds_res.rio.reproject(
    era5_ds_res.rio.crs,
    shape=(180, 360),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    resampling=Resampling.average,
)
# rename spatial dims back to latlon
era5_ds_res = era5_ds_res.rename({'y': 'lat', 'x': 'lon'})
era5_ds_res = era5_ds_res.where(era5_ds_res >= 0)


imerg_ds_res = imerg_v07_al.copy()
imerg_ds_res.rio.write_crs(cc.to_string(), inplace=True)
imerg_ds_res = imerg_ds_res.rio.set_spatial_dims(x_dim="lon", 
                                                y_dim="lat", 
                                                inplace=True)
imerg_ds_res = imerg_ds_res.rio.reproject(
    imerg_ds_res.rio.crs,
    shape=(180, 360),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    resampling=Resampling.average,
)
# rename spatial dims back to latlon
imerg_ds_res = imerg_ds_res.rename({'y': 'lat', 'x': 'lon'})
imerg_ds_res = imerg_ds_res.where(imerg_ds_res >= 0)

merra_ds_res = mer2_ds_al.copy()
merra_ds_res.rio.write_crs(cc.to_string(), inplace=True)
merra_ds_res = merra_ds_res.rio.set_spatial_dims(x_dim="x", 
                                                y_dim="y", 
                                                inplace=True)
merra_ds_res = merra_ds_res.rio.reproject(
    merra_ds_res.rio.crs,
    shape=(180, 360),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    resampling=Resampling.average,
)
# rename spatial dims back to latlon
merra_ds_res = merra_ds_res.rename({'y': 'lat', 'x': 'lon'})
merra_ds_res = merra_ds_res.where(merra_ds_res >= 0)

# -------------------------------------------------------------
# Official manuscript OceanRAIN product map:
# products harmonized to common 1° support
# -------------------------------------------------------------

product_map_common1deg = {
    "GPCP v1.3": (
        gpcp_ds_v1pt3_al_res.where(gpcp_ds_v1pt3_al_res >= 0),
        {"GPCP v1.3": None},
    ),

    "GPCP v3.2": (
        gpcp_ds_v3pt2_al_res.where(gpcp_ds_v3pt2_al_res >= 0),
        {"GPCP v3.2": None},
    ),

    "GPCP v3.3": (
        gpcp_ds_v3pt3_al_res.where(gpcp_ds_v3pt3_al_res >= 0),
        {"GPCP v3.3": None},
    ),

    "IMERG v07": (
        imerg_ds_res.where(imerg_ds_res >= 0),
        {"IMERG v07": None},
    ),

    "ERA5": (
        era5_ds_res.where(era5_ds_res >= 0),
        {"ERA5": None},
    ),

    "MERRA-2": (
        merra_ds_res.where(merra_ds_res >= 0),
        {"MERRA-2": None},
    ),
}
# -------------------------------------------------------------
# Attach common-1° products to daily OceanRAIN ship–grid-cell samples
# -------------------------------------------------------------

# daily_or_attached = step2_attach_products_oceanrain(
#     daily_or_usable,
#     gpcp_grid_ds=or_grid_1deg,              # only used for lat_c/lon_c from ilat/ilon
#     products=product_map_common1deg,        # official common-1° product fields
#     ref_col="main_mmday",
#     method="nearest",
# )

# -------------------------------------------------------------
# OceanRAIN PAL-like common-1° matching:
# minute-level match to common 1° grid/products, then daily aggregation
# -------------------------------------------------------------

daily_or_all, daily_or_attached, minute_or_matched_common1deg = (
    oceanrain_minute_match_common1deg_then_daily_xr(
        df_qc,
        products=product_map_common1deg,
        grid_lat_1d=or_lat_1d,
        grid_lon_1d=or_lon_1d,
        time_col="time_utc",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        ship_col="ship",
        obs_rate_col="rate_main_mmph",
        precip_flag_col="precip_flag",
        lat_abs_min=45.0,
        coverage_frac=0.5,
        method="nearest",
        dropna_product_cols=False,
        chunk_size=200_000,
    )
)

daily_or_attached = daily_or_attached[
    daily_or_attached["main_mmday"] != -99999.0
].copy()

daily_or_attached.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"oceanrain_daily_common1deg_matched_{cde_run_dte}.csv"
    ),
    index=False,
)

#--------------------------------------------------------------
products_eval = ["GPCP v1.3", 'GPCP v3.2', 'GPCP v3.3',  'IMERG v07', 
                 'ERA5','MERRA-2']

cat_metrics_hemi, qt_metrics_hemi = compute_hemi_metrics_oceanrain(
    daily_or_attached,
    products=products_eval,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
    cat_thr=1.0,
)

#--------------------------------------------------------------
products_order = ["GPCP v1.3", "GPCP v3.2", "GPCP v3.3",  'IMERG v07', 
                  "ERA5", "MERRA-2"] 
#--------------------------------------------------------------
# 1) categorical table
#--------------------------------------------------------------

cat_table = categorical_dict_to_table(
    cat_metrics_hemi,   # replace with your categorical dict variable name
    products_order=products_order
)

cat_table = round_metric_table(cat_table, ["POD", "FAR", "Bias", "HSS"], ndigits=3)

cat_table_nh = cat_table[cat_table["hemi"] == "NH"].reset_index(drop=True)
cat_table_sh = cat_table[cat_table["hemi"] == "SH"].reset_index(drop=True)

print("\nNH categorical metrics")
print(cat_table_nh)

print("\nSH categorical metrics")
print(cat_table_sh)
# svve the categorical table
cat_table.to_csv(
    os.path.join(path_to_put_dfs, 
                f"oceanrain_cat_metrics_common1deg_{cde_run_dte}.csv"),
    index=False,    
)

#--------------------------------------------------------------
# Roebber / performance diagram for OceanRAIN categorical metrics
#--------------------------------------------------------------

products_roebber = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

fig, axes = plot_oceanrain_roebber_diagram(
    cat_metrics_hemi=cat_metrics_hemi,
    products=products_roebber,
    product_colors=product_colors,
    hemis=("NH", "SH"),
    figsize=(15, 6.2),
    marker_size=115,
    annotate=False,
    title=None,
    contour_label_fontsize=12,
    bias_label_fontsize=12,
    axis_label_fontsize=15,
    tick_fontsize=15,
    legend_fontsize=15,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"Fig10_OceanRAIN_occurrence_performance_roebber_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)


# plt.show()
gc.collect()

#--------------------------------------------------------------
# 4) quantitative summary plot
#--------------------------------------------------------------

qt_1deg_table = quantitative_dict_to_table(
    qt_metrics_hemi,
    products_order=products_eval,
)
qt_table_nh = qt_1deg_table[qt_1deg_table["hemi"] == "NH"].reset_index(drop=True)
qt_table_sh = qt_1deg_table[qt_1deg_table["hemi"] == "SH"].reset_index(drop=True)

print("\nNH quantitative metrics")
print(qt_table_nh)

print("\nSH quantitative metrics")
print(qt_table_sh)
# save the quantitative table
qt_1deg_table.to_csv(
    os.path.join(path_to_put_dfs, 
                f"oceanrain_quant_metrics_common1deg_{cde_run_dte}.csv"),
    index=False,    
)


products_plot = ["GPCP v1.3", "GPCP v3.2", "GPCP v3.3", "IMERG v07", "ERA5", "MERRA-2"]

fig, axes = plot_oceanrain_quant_summary_panel(
    qt_metrics_hemi,
    products=products_eval,
    product_colors=product_colors,
    hemis=("NH", "SH"),
    metrics=("CC", "Bias", "RMSE"),
    figsize=(12, 9),
    ylims={
        "CC": (0, 0.6),
        # "Bias": (-60, 15),
        "RMSE": (0, 8),
    },
    auto_ylim=True,
    ylabel_map={
        "CC": "CC",
        "Bias": "Bias [%]",
        "RMSE": "RMSE\n[mm day$^{-1}$]",
    },
    add_zero_line_for_bias=True,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"Fig11_OceanRAIN_quantitative_metrics_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)


gc.collect()


# 5) compact descriptive distribution table
dist_table = build_oceanrain_descriptive_stats_table(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_order,
    hemi_col="hemi",
    # hemis=("NH", "SH")
)

dist_table = round_metric_table(dist_table, ["median", "p90", "p95", "p99", "max"], ndigits=3)

#sve the distribution table
dist_table.to_csv(
    os.path.join(path_to_put_dfs, 
                 f"oceanrain_descriptive_stats_common1deg_{cde_run_dte}.csv"),
    index=False,    
)

dist_table_nh = dist_table[dist_table["hemi"] == "NH"].reset_index(drop=True)
dist_table_sh = dist_table[dist_table["hemi"] == "SH"].reset_index(drop=True)

print("\nNH distribution summary")
print(dist_table_nh)

print("\nSH distribution summary")
print(dist_table_sh)

# 5) Supplementary OceanRAIN daily distribution boxplot
#    This replaces the descriptive distribution table in the supplement.
#    Box = 25th–75th percentile, center line = median,
#    whiskers = 5th–95th percentile, black circle = mean.

fig, axes = plot_oceanrain_distribution_boxplot(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_order,
    product_colors=product_colors,
    hemi_col="hemi",
    hemis=("NH", "SH"),
    wet_only=False,
    wet_threshold=0.3,
    figsize=(13.5, 6),
    ylabel="Daily precipitation [mm day$^{-1}$]",
    ylimit=(0, 3.0),
    use_symlog=False,
    panel_label_fontsize=15,
    axis_label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=11,
    median_color="lime",
    median_linewidth=3.0,
    mean_marker_size=36,
    legend_ax_index=1,
     whisker_mode="p5_p95",
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"oceanrain_daily_distribution_boxplot_IQR_mean_median_common1deg_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)

gc.collect()

# -------------------------------------------------------------
# sample summary
# -------------------------------------------------------------

sample_summary_native_minute = compute_oceanrain_sample_distribution_summary(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_eval,
    hemi_col="hemi",
    hemis=("NH", "SH"),
    wet_only=False,
    wet_threshold=0,
)

sample_summary_native_minute_rounded = sample_summary_native_minute.copy()

round_cols = [
    "mean",
    "median",
    "std",
    "p95",
    "sum",
    "sum_ratio_to_OceanRAIN",
    "sum_bias_percent_vs_OceanRAIN",
]

sample_summary_native_minute_rounded[round_cols] = (
    sample_summary_native_minute_rounded[round_cols].round(2)
)

sample_summary_native_minute_rounded.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"TableC1_oceanrain_sample_summary_common1deg_rounded_{cde_run_dte}.csv"
    ),
    index=False,
)

print(sample_summary_native_minute_rounded)

#%% A Diagnostic Check
# ==============================================================
# OceanRAIN categorical threshold-sensitivity audit
#
# Required object already in memory:
#     daily_or_attached
#
# Expected columns:
#     hemi
#     main_mmday
#     GPCP v1.3
#     GPCP v3.2
#     GPCP v3.3
#     IMERG v07
#     ERA5
#     MERRA-2
#
# Optional columns used when available:
#     date
#     coverage_frac_day
# ==============================================================

from pathlib import Path


# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

OBS_COL = "main_mmday"

PRODUCTS = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

HEMISPHERES = ["NH", "SH"]

# Include the old 0.3 threshold, the current 1.0 threshold,
# and intermediate / heavier-rain thresholds.
THRESHOLDS = np.array([
    0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0
])

# Change this to your preferred folder.
OUTPUT_DIR = Path("./oceanrain_threshold_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# Input checks and cleanup
# --------------------------------------------------------------

required_columns = ["hemi", OBS_COL] + PRODUCTS
missing_columns = [
    col for col in required_columns
    if col not in daily_or_attached.columns
]

if missing_columns:
    raise KeyError(
        "daily_or_attached is missing required columns:\n"
        + "\n".join(missing_columns)
    )

df_or = daily_or_attached.copy()

# Standardize hemisphere labels.
df_or["hemi"] = (
    df_or["hemi"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Convert relevant fields to numeric.
for col in [OBS_COL] + PRODUCTS:
    df_or[col] = pd.to_numeric(df_or[col], errors="coerce")

# Convert date when present.
if "date" in df_or.columns:
    df_or["date"] = pd.to_datetime(
        df_or["date"],
        errors="coerce"
    )
    df_or["year"] = df_or["date"].dt.year

# Remove invalid OceanRAIN sentinel values and negative precipitation.
df_or = df_or[
    np.isfinite(df_or[OBS_COL])
    & (df_or[OBS_COL] >= 0)
].copy()

print("Rows available after OceanRAIN reference filtering:")
print(df_or.groupby("hemi").size())
print()


# --------------------------------------------------------------
# Calculate threshold sensitivity
# --------------------------------------------------------------

sensitivity_rows = []

for hemisphere in HEMISPHERES:

    hemi_df = df_or[
        df_or["hemi"] == hemisphere
    ].copy()

    for product in PRODUCTS:

        product_df = hemi_df[
            [OBS_COL, product]
        ].dropna()

        for threshold in THRESHOLDS:

            metrics = categorical_metrics_with_counts(
                forecast=product_df[product],
                observation=product_df[OBS_COL],
                threshold=threshold,
            )

            sensitivity_rows.append({
                "Hemisphere": hemisphere,
                "Product": product,
                "Threshold_mmday": threshold,
                **metrics,
            })

sensitivity_df = pd.DataFrame(sensitivity_rows)

sensitivity_df.to_csv(
    OUTPUT_DIR / "oceanrain_threshold_sensitivity_all_metrics.csv",
    index=False,
)

print("Threshold-sensitivity table:")
print(
    sensitivity_df[
        sensitivity_df["Threshold_mmday"].isin(
            [0.3, 1.0]
        )
    ][
        [
            "Hemisphere",
            "Product",
            "Threshold_mmday",
            "POD",
            "FAR",
            "Frequency_bias",
            "HSS",
            "Hits",
            "Misses",
            "False_alarms",
            "Observed_events",
        ]
    ].round(3)
)

print()


# --------------------------------------------------------------
# Direct 0.3 versus 1.0 comparison
# --------------------------------------------------------------

comparison_03_10 = (
    sensitivity_df[
        sensitivity_df["Threshold_mmday"].isin(
            [0.3, 1.0]
        )
    ]
    .pivot(
        index=["Hemisphere", "Product"],
        columns="Threshold_mmday",
        values=[
            "POD",
            "FAR",
            "Frequency_bias",
            "HSS",
            "Hits",
            "Misses",
        ],
    )
)

# Flatten multi-index columns.
comparison_03_10.columns = [
    f"{metric}_{threshold:g}"
    for metric, threshold
    in comparison_03_10.columns
]

comparison_03_10 = comparison_03_10.reset_index()

comparison_03_10["POD_change_0.3_to_1.0"] = (
    comparison_03_10["POD_1"]
    - comparison_03_10["POD_0.3"]
)

comparison_03_10["POD_relative_change_percent"] = (
    100
    * (
        comparison_03_10["POD_1"]
        - comparison_03_10["POD_0.3"]
    )
    / comparison_03_10["POD_0.3"]
)

comparison_03_10.to_csv(
    OUTPUT_DIR / "oceanrain_comparison_threshold_0p3_vs_1p0.csv",
    index=False,
)

print("Direct comparison: 0.3 versus 1.0 mm/day")
print(
    comparison_03_10[
        [
            "Hemisphere",
            "Product",
            "POD_0.3",
            "POD_1",
            "POD_change_0.3_to_1.0",
            "POD_relative_change_percent",
            "Hits_0.3",
            "Hits_1",
            "Misses_0.3",
            "Misses_1",
        ]
    ].round(3)
)
print()


# --------------------------------------------------------------
# Plot 1: POD sensitivity for all products
# --------------------------------------------------------------

for hemisphere in HEMISPHERES:

    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = sensitivity_df[
        sensitivity_df["Hemisphere"] == hemisphere
    ]

    for product in PRODUCTS:

        product_df = plot_df[
            plot_df["Product"] == product
        ]

        ax.plot(
            product_df["Threshold_mmday"],
            product_df["POD"],
            marker="o",
            linewidth=2,
            label=product,
        )

    ax.axvline(
        0.3,
        linestyle="--",
        linewidth=1,
        label="Old threshold: 0.3",
    )

    ax.axvline(
        1.0,
        linestyle=":",
        linewidth=1.5,
        label="Current threshold: 1.0",
    )

    ax.set_xlabel(
        "Rain-event threshold [mm day$^{-1}$]"
    )
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"OceanRAIN POD sensitivity — {hemisphere}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / f"pod_threshold_sensitivity_{hemisphere}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# --------------------------------------------------------------
# Plot 2: POD, FAR, frequency bias and HSS
#         ERA5 versus MERRA-2
# --------------------------------------------------------------

key_products = ["ERA5", "MERRA-2"]
key_metrics = [
    ("POD", "Probability of detection"),
    ("FAR", "False alarm ratio"),
    ("Frequency_bias", "Frequency bias"),
    ("HSS", "Heidke skill score"),
]

for hemisphere in HEMISPHERES:

    for metric, ylabel in key_metrics:

        fig, ax = plt.subplots(figsize=(8, 6))

        for product in key_products:

            product_df = sensitivity_df[
                (
                    sensitivity_df["Hemisphere"]
                    == hemisphere
                )
                & (
                    sensitivity_df["Product"]
                    == product
                )
            ]

            ax.plot(
                product_df["Threshold_mmday"],
                product_df[metric],
                marker="o",
                linewidth=2.2,
                label=product,
            )

        ax.axvline(
            0.3,
            linestyle="--",
            linewidth=1,
        )
        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1.5,
        )

        ax.set_xlabel(
            "Rain-event threshold [mm day$^{-1}$]"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel}: ERA5 versus MERRA-2 — "
            f"{hemisphere}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        if metric in ["POD", "FAR", "HSS"]:
            ax.set_ylim(0, 1.02)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR
            / (
                f"{metric.lower()}_ERA5_MERRA2_"
                f"{hemisphere}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


# --------------------------------------------------------------
# Plot 3: Fraction of OceanRAIN event days detected
#
# This is numerically equivalent to POD, but the event counts
# are shown explicitly to make the threshold effect intuitive.
# --------------------------------------------------------------

for hemisphere in HEMISPHERES:

    fig, ax = plt.subplots(figsize=(9, 6))

    for product in ["ERA5", "MERRA-2", "IMERG v07"]:

        product_df = sensitivity_df[
            (
                sensitivity_df["Hemisphere"]
                == hemisphere
            )
            & (
                sensitivity_df["Product"]
                == product
            )
        ]

        ax.plot(
            product_df["Threshold_mmday"],
            100 * product_df["POD"],
            marker="o",
            linewidth=2,
            label=product,
        )

    ax.axvline(
        0.3,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        1.0,
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Rain-event threshold [mm day$^{-1}$]"
    )
    ax.set_ylabel(
        "Observed OceanRAIN events detected [%]"
    )
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Fraction of OceanRAIN events detected — "
        f"{hemisphere}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / f"observed_events_detected_{hemisphere}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# --------------------------------------------------------------
# Diagnostic 4:
# Product rainfall distribution on OceanRAIN rainy days
#
# For each reference threshold, this isolates days on which
# OceanRAIN reports an event and examines the matched product
# rainfall values.
# --------------------------------------------------------------

rain_day_thresholds = [0.3, 1.0]

distribution_summary_rows = []

for hemisphere in HEMISPHERES:

    hemi_df = df_or[
        df_or["hemi"] == hemisphere
    ].copy()

    for obs_threshold in rain_day_thresholds:

        observed_event_df = hemi_df[
            hemi_df[OBS_COL] >= obs_threshold
        ].copy()

        for product in ["ERA5", "MERRA-2", "IMERG v07"]:

            values = (
                observed_event_df[product]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if values.empty:
                continue

            distribution_summary_rows.append({
                "Hemisphere": hemisphere,
                "OceanRAIN_event_threshold": obs_threshold,
                "Product": product,
                "N": len(values),
                "Mean": values.mean(),
                "Median": values.median(),
                "P10": values.quantile(0.10),
                "P25": values.quantile(0.25),
                "P75": values.quantile(0.75),
                "P90": values.quantile(0.90),
                "Fraction_below_0.3": (
                    values < 0.3
                ).mean(),
                "Fraction_below_1.0": (
                    values < 1.0
                ).mean(),
            })

distribution_summary_df = pd.DataFrame(
    distribution_summary_rows
)

distribution_summary_df.to_csv(
    OUTPUT_DIR
    / "product_distribution_on_oceanrain_event_days.csv",
    index=False,
)

print(
    "Product rainfall on OceanRAIN event days:"
)
print(
    distribution_summary_df.round(3)
)
print()


# --------------------------------------------------------------
# Plot 4: Histograms for ERA5 and MERRA-2 on OceanRAIN
#         rain-event days
#
# Separate plots are used for each hemisphere and reference
# event threshold.
# --------------------------------------------------------------

histogram_bins = np.array([
    0.0, 0.1, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0,
    8.0, 12.0, 20.0, 40.0,
])

for hemisphere in HEMISPHERES:

    hemi_df = df_or[
        df_or["hemi"] == hemisphere
    ].copy()

    for obs_threshold in rain_day_thresholds:

        event_df = hemi_df[
            hemi_df[OBS_COL] >= obs_threshold
        ].copy()

        fig, ax = plt.subplots(figsize=(9, 6))

        for product in ["ERA5", "MERRA-2"]:

            values = (
                event_df[product]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            ax.hist(
                values,
                bins=histogram_bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=product,
            )

        ax.axvline(
            0.3,
            linestyle="--",
            linewidth=1,
            label="0.3 mm/day",
        )

        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1.5,
            label="1.0 mm/day",
        )

        ax.set_xlabel(
            "Matched product precipitation "
            "[mm day$^{-1}$]"
        )
        ax.set_ylabel("Probability density")
        ax.set_title(
            f"{hemisphere}: product rainfall on "
            f"OceanRAIN ≥ {obs_threshold:g} mm/day days"
        )
        ax.set_xlim(
            0,
            np.nanpercentile(
                event_df[
                    ["ERA5", "MERRA-2"]
                ].to_numpy(),
                97.5,
            )
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR
            / (
                f"hist_ERA5_MERRA2_{hemisphere}_"
                f"OceanRAIN_ge_{obs_threshold:g}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


# --------------------------------------------------------------
# Diagnostic 5:
# Explicit count of OceanRAIN events migrating from hits at 0.3
# to misses at 1.0
#
# This directly tests the proposed explanation for ERA5 NH.
# --------------------------------------------------------------

migration_rows = []

for hemisphere in HEMISPHERES:

    hemi_df = df_or[
        df_or["hemi"] == hemisphere
    ].copy()

    for product in ["ERA5", "MERRA-2", "IMERG v07"]:

        pair = hemi_df[
            [OBS_COL, product]
        ].dropna()

        # OceanRAIN event under current 1.0 threshold.
        obs_event_1 = pair[OBS_COL] >= 1.0

        # Among those OceanRAIN events:
        # product detects at 0.3 but fails at 1.0.
        product_detects_03 = pair[product] >= 0.3
        product_detects_10 = pair[product] >= 1.0

        intermediate_detection = (
            obs_event_1
            & product_detects_03
            & ~product_detects_10
        )

        detected_at_10 = (
            obs_event_1
            & product_detects_10
        )

        missed_even_at_03 = (
            obs_event_1
            & ~product_detects_03
        )

        total_obs_events_1 = int(
            obs_event_1.sum()
        )

        migration_rows.append({
            "Hemisphere": hemisphere,
            "Product": product,
            "OceanRAIN_events_ge_1": total_obs_events_1,
            "Detected_product_ge_1": int(
                detected_at_10.sum()
            ),
            "Detected_at_0.3_but_not_1.0": int(
                intermediate_detection.sum()
            ),
            "Missed_below_0.3": int(
                missed_even_at_03.sum()
            ),
            "Percent_intermediate_0.3_to_1.0": (
                100
                * intermediate_detection.sum()
                / total_obs_events_1
                if total_obs_events_1 > 0
                else np.nan
            ),
        })

migration_df = pd.DataFrame(migration_rows)

migration_df.to_csv(
    OUTPUT_DIR
    / "event_migration_between_0p3_and_1p0.csv",
    index=False,
)

print(
    "OceanRAIN ≥1 mm/day events: detection migration "
    "between 0.3 and 1.0 thresholds"
)
print(migration_df.round(2))
print()


# --------------------------------------------------------------
# Optional diagnostic 6:
# Threshold sensitivity by daily OceanRAIN coverage
#
# This checks whether the result depends on the current
# coverage_frac_day criterion.
# --------------------------------------------------------------

if "coverage_frac_day" in df_or.columns:

    coverage_thresholds = [
        0.10,
        0.25,
        0.50,
        0.75,
    ]

    coverage_rows = []

    for minimum_coverage in coverage_thresholds:

        coverage_df = df_or[
            df_or["coverage_frac_day"]
            >= minimum_coverage
        ].copy()

        for hemisphere in HEMISPHERES:

            hemi_df = coverage_df[
                coverage_df["hemi"] == hemisphere
            ]

            for product in [
                "ERA5",
                "MERRA-2",
                "IMERG v07",
            ]:

                pair = hemi_df[
                    [OBS_COL, product]
                ].dropna()

                for threshold in [0.3, 1.0]:

                    metrics = (
                        categorical_metrics_with_counts(
                            forecast=pair[product],
                            observation=pair[OBS_COL],
                            threshold=threshold,
                        )
                    )

                    coverage_rows.append({
                        "Minimum_coverage_fraction":
                            minimum_coverage,
                        "Hemisphere": hemisphere,
                        "Product": product,
                        "Threshold_mmday": threshold,
                        **metrics,
                    })

    coverage_sensitivity_df = pd.DataFrame(
        coverage_rows
    )

    coverage_sensitivity_df.to_csv(
        OUTPUT_DIR
        / "coverage_threshold_sensitivity.csv",
        index=False,
    )

    for hemisphere in HEMISPHERES:

        for event_threshold in [0.3, 1.0]:

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            plot_df = coverage_sensitivity_df[
                (
                    coverage_sensitivity_df[
                        "Hemisphere"
                    ] == hemisphere
                )
                & (
                    coverage_sensitivity_df[
                        "Threshold_mmday"
                    ] == event_threshold
                )
            ]

            for product in [
                "ERA5",
                "MERRA-2",
                "IMERG v07",
            ]:

                product_df = plot_df[
                    plot_df["Product"]
                    == product
                ]

                ax.plot(
                    product_df[
                        "Minimum_coverage_fraction"
                    ],
                    product_df["POD"],
                    marker="o",
                    linewidth=2,
                    label=product,
                )

            ax.set_xlabel(
                "Minimum retained daily "
                "OceanRAIN coverage fraction"
            )
            ax.set_ylabel(
                "Probability of detection"
            )
            ax.set_ylim(0, 1.02)
            ax.set_title(
                f"{hemisphere}: POD sensitivity to "
                f"coverage at {event_threshold:g} mm/day"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)

            fig.tight_layout()
            fig.savefig(
                OUTPUT_DIR
                / (
                    f"coverage_POD_{hemisphere}_"
                    f"thr_{event_threshold:g}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

else:
    print(
        "coverage_frac_day is not present; "
        "coverage sensitivity was skipped."
    )


# --------------------------------------------------------------
# Optional diagnostic 7:
# Annual POD at 0.3 and 1.0 mm/day
#
# This checks whether a small number of years or cruises drives
# the apparent ERA5 NH reduction.
# --------------------------------------------------------------

if "year" in df_or.columns:

    annual_rows = []

    for hemisphere in HEMISPHERES:

        hemi_df = df_or[
            df_or["hemi"] == hemisphere
        ]

        valid_years = sorted(
            hemi_df["year"].dropna().unique()
        )

        for year in valid_years:

            year_df = hemi_df[
                hemi_df["year"] == year
            ]

            for product in [
                "ERA5",
                "MERRA-2",
                "IMERG v07",
            ]:

                pair = year_df[
                    [OBS_COL, product]
                ].dropna()

                for threshold in [0.3, 1.0]:

                    metrics = (
                        categorical_metrics_with_counts(
                            forecast=pair[product],
                            observation=pair[OBS_COL],
                            threshold=threshold,
                        )
                    )

                    annual_rows.append({
                        "Hemisphere": hemisphere,
                        "Year": int(year),
                        "Product": product,
                        "Threshold_mmday": threshold,
                        **metrics,
                    })

    annual_sensitivity_df = pd.DataFrame(
        annual_rows
    )

    # Avoid interpreting annual estimates with very few observed
    # events.
    annual_sensitivity_df[
        "Adequate_event_count"
    ] = (
        annual_sensitivity_df[
            "Observed_events"
        ] >= 20
    )

    annual_sensitivity_df.to_csv(
        OUTPUT_DIR
        / "annual_threshold_sensitivity.csv",
        index=False,
    )

    for hemisphere in HEMISPHERES:

        for event_threshold in [0.3, 1.0]:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            plot_df = annual_sensitivity_df[
                (
                    annual_sensitivity_df[
                        "Hemisphere"
                    ] == hemisphere
                )
                & (
                    annual_sensitivity_df[
                        "Threshold_mmday"
                    ] == event_threshold
                )
                & (
                    annual_sensitivity_df[
                        "Adequate_event_count"
                    ]
                )
            ]

            for product in [
                "ERA5",
                "MERRA-2",
                "IMERG v07",
            ]:

                product_df = plot_df[
                    plot_df["Product"]
                    == product
                ]

                ax.plot(
                    product_df["Year"],
                    product_df["POD"],
                    marker="o",
                    linewidth=1.8,
                    label=product,
                )

            ax.set_xlabel("Year")
            ax.set_ylabel(
                "Probability of detection"
            )
            ax.set_ylim(0, 1.02)
            ax.set_title(
                f"{hemisphere}: annual POD at "
                f"{event_threshold:g} mm/day"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)

            fig.tight_layout()
            fig.savefig(
                OUTPUT_DIR
                / (
                    f"annual_POD_{hemisphere}_"
                    f"thr_{event_threshold:g}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

else:
    print(
        "A usable date/year column is not present; "
        "annual sensitivity was skipped."
    )


print(
    "\nSensitivity audit complete. Outputs saved to:"
)
print(OUTPUT_DIR.resolve())


import numpy as np
import matplotlib.pyplot as plt

OBS = "main_mmday"
PRODUCTS = ["ERA5", "MERRA-2", "IMERG v07", "GPCP v1.3", "GPCP v3.2", "GPCP v3.3"]
THRESHOLD = 1.0

# Log-spaced rainfall bins, similar to your PAL/Buoy intensity plot
bins = np.array([0.1, 0.3, 0.5, 1, 2, 4, 8, 16, 32, 64])


# ============================================================
# 1. Rainfall-volume distribution by intensity class
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5),
    sharey=True
)

for ax, hemi in zip(axes, ["NH", "SH"]):

    df = daily_or_attached[
        daily_or_attached["hemi"] == hemi
    ].copy()

    columns = [OBS] + PRODUCTS
    df = df[columns].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    for col in columns:

        values = df[col].to_numpy()

        # Rainfall volume contributed by each intensity bin
        rainfall_volume, _ = np.histogram(
            values,
            bins=bins,
            weights=values
        )

        total_volume = np.sum(values)

        if total_volume > 0:
            rainfall_fraction = (
                100 * rainfall_volume / total_volume
            )
        else:
            rainfall_fraction = np.full(
                len(bins) - 1,
                np.nan
            )

        # Geometric centers are best for log-scaled bins
        bin_centers = np.sqrt(
            bins[:-1] * bins[1:]
        )

        ax.plot(
            bin_centers,
            rainfall_fraction,
            marker="o",
            linewidth=2.5,
            label=col
        )

    ax.axvline(
        0.3,
        linestyle="--",
        linewidth=1.2,
        label="0.3 mm/day"
    )

    ax.axvline(
        1.0,
        linestyle=":",
        linewidth=1.5,
        label="1.0 mm/day"
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(bins)
    ax.set_xticklabels(
        [str(x) for x in bins]
    )

    ax.set_xlabel(
        "Daily precipitation [mm day$^{-1}$]"
    )
    ax.set_title(hemi)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(
    "Contribution to total rainfall volume [%]"
)

handles, labels = axes[1].get_legend_handles_labels()

# Remove repeated threshold labels
unique = dict(zip(labels, handles))

fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
)

fig.suptitle(
    "OceanRAIN and reanalysis daily rainfall-volume distributions"
)

fig.tight_layout(
    rect=[0, 0.10, 1, 0.95]
)

plt.show()


# ============================================================
# 2. OceanRAIN versus ERA5 scatter plot
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5.5),
    sharex=True,
    sharey=True
)

for ax, hemi in zip(axes, ["NH", "SH"]):

    df = daily_or_attached[
        daily_or_attached["hemi"] == hemi
    ][[OBS, "ERA5"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    ax.scatter(
        df[OBS],
        df["ERA5"],
        alpha=0.55,
        s=28
    )

    max_value = np.nanpercentile(
        df[[OBS, "ERA5"]].to_numpy(),
        98
    )

    max_value = max(max_value, 5)

    ax.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
        linewidth=1.2,
        label="1:1 line"
    )

    ax.axvline(
        THRESHOLD,
        linestyle=":",
        linewidth=1.5
    )

    ax.axhline(
        THRESHOLD,
        linestyle=":",
        linewidth=1.5
    )

    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)

    ax.set_xlabel(
        "OceanRAIN [mm day$^{-1}$]"
    )
    ax.set_title(hemi)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(
    "ERA5 [mm day$^{-1}$]"
)

fig.suptitle(
    "OceanRAIN versus ERA5 daily precipitation"
)

fig.tight_layout()
plt.show()


# ============================================================
# 3. Simple threshold-quadrant counts
# ============================================================

for hemi in ["NH", "SH"]:

    df = daily_or_attached[
        daily_or_attached["hemi"] == hemi
    ][[OBS, "ERA5"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    hits = (
        (df[OBS] >= THRESHOLD)
        & (df["ERA5"] >= THRESHOLD)
    ).sum()

    misses = (
        (df[OBS] >= THRESHOLD)
        & (df["ERA5"] < THRESHOLD)
    ).sum()

    false_alarms = (
        (df[OBS] < THRESHOLD)
        & (df["ERA5"] >= THRESHOLD)
    ).sum()

    correct_negatives = (
        (df[OBS] < THRESHOLD)
        & (df["ERA5"] < THRESHOLD)
    ).sum()

    print(f"\n{hemi} at {THRESHOLD} mm/day")
    print(f"Hits:              {hits}")
    print(f"Misses:            {misses}")
    print(f"False alarms:      {false_alarms}")
    print(f"Correct negatives: {correct_negatives}")

    observed_events = hits + misses

    if observed_events > 0:
        print(
            "Observed events represented below "
            f"{THRESHOLD} mm/day by ERA5: "
            f"{100 * misses / observed_events:.1f}%"
        )

#%%
# -------------------------------------------------------------
# OceanRAIN PAL-like native-grid sensitivity:
# attach products at minute locations first, then daily aggregate
# -------------------------------------------------------------


products_eval = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

product_map_native_minute = {
    "GPCP v1.3": (
        gpcp_ds_v1pt3_al["precip"].where(gpcp_ds_v1pt3_al["precip"] >= 0),
        {"GPCP v1.3": None},
    ),
    "GPCP v3.2": (
        gpcp_ds_v3pt2_al["precip"].where(gpcp_ds_v3pt2_al["precip"] >= 0),
        {"GPCP v3.2": None},
    ),
    "GPCP v3.3": (
        gpcp_ds_v3pt3_al["precip"].where(gpcp_ds_v3pt3_al["precip"] >= 0),
        {"GPCP v3.3": None},
    ),
    "IMERG v07": (
        imerg_v07_al.where(imerg_v07_al >= 0),
        {"IMERG v07": None},
    ),
    "ERA5": (
        era5_ds_al["tp"].where(era5_ds_al["tp"] >= 0),
        {"ERA5": None},
    ),
    "MERRA-2": (
        mer2_ds_al.where(mer2_ds_al >= 0),
        {"MERRA-2": None},
    ),
}

daily_or_attached_native_minute, minute_or_attached_native = (
    oceanrain_attach_products_minute_native_then_daily(
        df_qc,
        products=product_map_native_minute,
        time_col="time_utc",
        lat_col="lat",
        lon_col="lon",
        ship_col="ship",
        obs_rate_col="rate_main_mmph",
        precip_flag_col="precip_flag",
        gpcp_lat_1d=np.arange(89.5, -90.0, -1.0, dtype=np.float32),
        gpcp_lon_1d=np.arange(-179.5, 180.0, 1.0, dtype=np.float32),
        lat_abs_min=45.0,
        coverage_frac=0.5,
        group_mode="ship_grid_day",
        method="nearest",
    )
)

products_eval = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

cat_native_minute, qt_native_minute = compute_hemi_metrics_oceanrain(
    daily_or_attached_native_minute,
    products=products_eval,
    obs_col="main_mmday",
    hemis=("NH", "SH"),
    cat_thr=0.3,
)

cat_native_minute_table = categorical_dict_to_table(
    cat_native_minute,
    products_order=products_eval,
)
cat_native_minute_table["sampling_method"] = "minute_native_then_daily"

qt_native_minute_table = quantitative_dict_to_table(
    qt_native_minute,
    products_order=products_eval,
)
qt_native_minute_table["sampling_method"] = "minute_native_then_daily"

cat_native_minute_table.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"oceanrain_cat_metrics_minute_native_then_daily_{cde_run_dte}.csv",
    ),
    index=False,
)

qt_native_minute_table.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"oceanrain_qt_metrics_minute_native_then_daily_{cde_run_dte}.csv",
    ),
    index=False,
)

print(cat_native_minute_table)
print(qt_native_minute_table)

# -------------------------------------------------------------
# Plot native-minute OceanRAIN categorical metrics:
# Roebber diagram + categorical bar panel
# -------------------------------------------------------------

cat_native_minute_for_plot = (
    cat_native_minute_table
    .drop(columns=["sampling_method"], errors="ignore")
    .copy()
)

cat_metrics_hemi_native_minute = {
    hemi: (
        cat_native_minute_for_plot[cat_native_minute_for_plot["hemi"] == hemi]
        .set_index("product")
        .drop(columns=["hemi"], errors="ignore")
        .to_dict(orient="index")
    )
    for hemi in ["NH", "SH"]
}

products_roebber = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA2",
]

fig, axes = plot_oceanrain_roebber_diagram(
    cat_metrics_hemi=cat_metrics_hemi_native_minute,
    products=products_roebber,
    product_colors=product_colors,
    hemis=("NH", "SH"),
    figsize=(15, 6.2), 
    marker_size=95,
    annotate=False,
    title=None,
    contour_label_fontsize=10,
    bias_label_fontsize=10,
    axis_label_fontsize=14,
    tick_fontsize=14,
    legend_fontsize=14,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"oceanrain_roebber_native_minute_then_daily_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)

gc.collect()


# -------------------------------------------------------------
# Quantitative bar panel for native-minute OceanRAIN sensitivity
# using existing plot_oceanrain_quant_summary_panel()
# -------------------------------------------------------------

products_eval = [
    "GPCP v1.3",
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
    "ERA5",
    "MERRA-2",
]

qt_native_minute_for_plot = (
    qt_native_minute_table
    .drop(columns=["sampling_method"], errors="ignore")
    .copy()
)

# Convert flat table back to the dictionary structure expected by
# plot_oceanrain_quant_summary_panel()
qt_metrics_hemi_native_minute = {
    hemi: (
        qt_native_minute_for_plot[qt_native_minute_for_plot["hemi"] == hemi]
        .set_index("product")
        .drop(columns=["hemi"], errors="ignore")
        .to_dict(orient="index")
    )
    for hemi in ["NH", "SH"]
}

fig, axes = plot_oceanrain_quant_summary_panel(
    qt_metrics_hemi_native_minute,
    products=products_eval,
    product_colors=product_colors,
    hemis=("NH", "SH"),
    metrics=("CC", "Bias", "RMSE"),
    figsize=(12, 9),
    ylims={
        "CC": (0, 0.7),
        "Bias": (-60, 40),
        "RMSE": (0, 15),
    },
    ylabel_map={
        "CC": "CC",
        "Bias": "Bias [%]",
        "RMSE": "RMSE\n[mm day$^{-1}$]",
    },
    add_zero_line_for_bias=True,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"oceanrain_quant_summary_native_minute_then_daily_{cde_run_dte}.png"
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.show()


sample_summary_native_minute = compute_oceanrain_sample_distribution_summary(
    daily_or_attached_native_minute,
    obs_col="main_mmday",
    product_cols=products_eval,
    hemi_col="hemi",
    hemis=("NH", "SH"),
    wet_only=False,
    wet_threshold=0.3,
)

sample_summary_native_minute.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"oceanrain_sample_summary_native_minute_then_daily_{cde_run_dte}.csv"
    ),
    index=False,
)

print(sample_summary_native_minute)

sample_summary_native_minute_rounded = sample_summary_native_minute.copy()

round_cols = [
    "mean",
    "median",
    "std",
    "p95",
    "sum",
    "sum_ratio_to_OceanRAIN",
    "sum_bias_percent_vs_OceanRAIN",
]

sample_summary_native_minute_rounded[round_cols] = (
    sample_summary_native_minute_rounded[round_cols].round(2)
)

sample_summary_native_minute_rounded.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"oceanrain_sample_summary_native_minute_then_daily_rounded_{cde_run_dte}.csv"
    ),
    index=False,
)

print(sample_summary_native_minute_rounded)

#%%
# -------------------------------------------------------------
# Common 0.5°-resolution daily spatial-mean product time series
# Separate NH and SH diagnostics
# Independent diagnostic: not sampled by OceanRAIN
# -------------------------------------------------------------
product_map_common05deg = {
    "GPCP v3.2": (
        gpcp_ds_v3pt2_al["precip"]
        .where(gpcp_ds_v3pt2_al["precip"] >= 0),
        {"GPCP v3.2": None},
    ),
    "GPCP v3.3": (
        gpcp_ds_v3pt3_al["precip"]
        .where(gpcp_ds_v3pt3_al["precip"] >= 0),
        {"GPCP v3.3": None},
    ),
    "IMERG v07": (
        imerg_v07_al.where(imerg_v07_al >= 0),
        {"IMERG v07": None},
    ),
}

ts_common05deg_nh = compute_native_daily_spatial_mean_timeseries(
    product_map_common05deg,
    region="NH",
    lat_abs_min=45.0,
    start_date="2010-01-01",
    end_date="2020-12-31",
    join="inner",
    mask_negative=True,
)

ts_common05deg_sh = compute_native_daily_spatial_mean_timeseries(
    product_map_common05deg,
    region="SH",
    lat_abs_min=45.0,
    start_date="2010-01-01",
    end_date="2020-12-31",
    join="inner",
    mask_negative=True,
)

ts_common05deg_nh.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"common05deg_product_daily_spatial_mean_NH_poleward45_{cde_run_dte}.csv"
    ),
    index=False,
)

ts_common05deg_sh.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"common05deg_product_daily_spatial_mean_SH_poleward45_{cde_run_dte}.csv"
    ),
    index=False,
)

ts_common05deg_highlat = compute_native_daily_spatial_mean_timeseries(
    product_map_common05deg,
    region="highlat_all",
    lat_abs_min=45.0,
    start_date="2010-01-01",
    end_date="2020-12-31",
    join="inner",
    mask_negative=True,
)

ts_common05deg_highlat.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"common05deg_product_daily_spatial_mean_highlat_all_{cde_run_dte}.csv"
    ),
    index=False,
)

fig, ax = plot_native_daily_spatial_mean_timeseries(
    ts_common05deg_highlat,
    products=['GPCP v3.2', 'GPCP v3.3', 'IMERG v07',],
    product_colors=product_colors,
    rolling=30,
    figsize=(13.5, 5.5),
    ylabel="mm day$^{-1}$",
    title="Common 0.5° daily spatial means poleward of 45°",
    tick_fontsize=12,
    axis_label_fontsize=13,
    legend_fontsize=12,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"common05deg_product_daily_spatial_mean_highlat_all_30day_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)
# plt.close()

gc.collect()

fig, ax = plot_native_daily_spatial_mean_timeseries(
    ts_common05deg_nh,
    products=['GPCP v3.2', 'GPCP v3.3', 'IMERG v07',],
    product_colors=product_colors,
    rolling=30,
    figsize=(13.5, 5.5),
    ylabel="mm day$^{-1}$",
    title="Common 0.5° daily spatial means poleward of 45°N",
    tick_fontsize=12,
    axis_label_fontsize=13,
    legend_fontsize=12,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"common05deg_product_daily_spatial_mean_NH_poleward45_30day_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)
gc.collect()

fig, ax = plot_native_daily_spatial_mean_timeseries(
    ts_common05deg_sh,
    products=['GPCP v3.2', 'GPCP v3.3', 'IMERG v07',],
    product_colors=product_colors,
    rolling=30,
    figsize=(13.5, 5.5),
    ylabel="mm day$^{-1}$",
    title="Common 0.5° daily spatial means poleward of 45°S",
    tick_fontsize=12,
    axis_label_fontsize=13,
    legend_fontsize=12,
)

fig.savefig(
    os.path.join(
        path_to_plots,
        f"common05deg_product_daily_spatial_mean_SH_poleward45_30day_{cde_run_dte}.png"
    ),
    dpi=150,
    bbox_inches="tight",
)
gc.collect()


# -------------------------------------------------------------
# Common 0.5° binary land-sea mask
# 1 = ocean, 0 = land
# -------------------------------------------------------------

land_sea_mask_path = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/"
    "ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4"
)

lsm_ds = xr.open_dataset(land_sea_mask_path)
lsm_arr = lsm_ds["landseamask"]

# Put longitude on x-axis
lsm_transposed = lsm_arr.transpose("lat", "lon")

# Preserve the orientation used in the previously working code
lsm_flipped = lsm_transposed.isel(lat=slice(None, None, -1))

# Binary classification:
# 1 = ocean, 0 = land
lsm = xr.where(lsm_flipped < 25, 1, 0).astype("uint8")

cc = CRS.from_authority(
    code=4326,
    auth_name="EPSG",
)

lsm.rio.write_crs(
    cc.to_string(),
    inplace=True,
)

lsm.rio.set_spatial_dims(
    x_dim="lon",
    y_dim="lat",
    inplace=True,
)

lsm_common05deg = lsm.rio.reproject(
    lsm.rio.crs,
    shape=(360, 720),
    resampling=Resampling.mode,
)

lsm_common05deg = lsm_common05deg.rename(
    {"x": "lon", "y": "lat"}
)

# Final masks
ocean_mask_common05deg = lsm_common05deg == 0
land_mask_common05deg = lsm_common05deg == 1

print("Common 0.5° mask shape:", lsm_common05deg.shape)
print(
    "Ocean fraction:",
    float(ocean_mask_common05deg.mean().values),
)
print(
    "Land fraction:",
    float(land_mask_common05deg.mean().values),
)

print(
    "Mask fraction sum:",
    float(
        ocean_mask_common05deg.mean().values
        + land_mask_common05deg.mean().values
    ),
)

product_map_common05deg_ocean = mask_common05deg_product_map(
    product_map_common05deg,
    ocean_mask_common05deg,
)

product_map_common05deg_land = mask_common05deg_product_map(
    product_map_common05deg,
    land_mask_common05deg,
)

common05deg_surface_results = {}

for region in ["NH", "SH"]:
    for surface, product_map_surface in [
        ("ocean", product_map_common05deg_ocean),
        ("land", product_map_common05deg_land),
    ]:

        result_key = f"{region}_{surface}"

        print(
            f"\nCalculating common 0.5° daily spatial means: "
            f"{region}, {surface}"
        )

        ts_surface = compute_native_daily_spatial_mean_timeseries(
            product_map_surface,
            region=region,
            lat_abs_min=45.0,
            start_date="2010-01-01",
            end_date="2020-12-31",
            join="inner",
            mask_negative=True,
        )

        ts_surface["hemisphere"] = region
        ts_surface["surface"] = surface

        common05deg_surface_results[result_key] = ts_surface

        ts_surface.to_csv(
            os.path.join(
                path_to_put_dfs,
                f"common05deg_product_daily_spatial_mean_"
                f"{region}_{surface}_poleward45_"
                f"{cde_run_dte}.csv"
            ),
            index=False,
        )

        gc.collect()


products_common05deg = [
    "GPCP v3.2",
    "GPCP v3.3",
    "IMERG v07",
]

for region in ["NH", "SH"]:
    for surface in ["ocean", "land"]:

        result_key = f"{region}_{surface}"
        ts_surface = common05deg_surface_results[result_key]

        latitude_label = "45°N" if region == "NH" else "45°S"
        surface_title = surface.capitalize()

        fig, ax = plot_native_daily_spatial_mean_timeseries(
            ts_surface,
            products=products_common05deg,
            product_colors=product_colors,
            rolling=30,
            figsize=(13.5, 5.5),
            ylabel="mm day$^{-1}$",
            title=(
                f"Common 0.5° daily {surface} spatial means "
                f"poleward of {latitude_label}"
            ),
            tick_fontsize=12,
            axis_label_fontsize=13,
            legend_fontsize=12,
        )

        fig.savefig(
            os.path.join(
                path_to_plots,
                f"common05deg_product_daily_spatial_mean_"
                f"{region}_{surface}_poleward45_30day_"
                f"{cde_run_dte}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()
        # plt.close(fig)
        gc.collect()


surface_summary_rows = []

for result_key, ts_df in common05deg_surface_results.items():

    region, surface = result_key.split("_", 1)

    for product in products_common05deg:

        values = pd.to_numeric(
            ts_df[product],
            errors="coerce",
        ).dropna()

        surface_summary_rows.append(
            {
                "hemisphere": region,
                "surface": surface,
                "product": product,
                "N_days": len(values),
                "mean_mm_day": values.mean(),
                "median_mm_day": values.median(),
                "std_mm_day": values.std(),
            }
        )

common05deg_surface_summary = pd.DataFrame(
    surface_summary_rows
)

common05deg_surface_summary.to_csv(
    os.path.join(
        path_to_put_dfs,
        f"common05deg_GPCP_IMERG_land_ocean_summary_"
        f"{cde_run_dte}.csv"
    ),
    index=False,
)

print(
    common05deg_surface_summary.round(3)
)
#%%

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
products_eval = ["GPCP v1.3",'GPCP v3.2', 'GPCP v3.3', 
                 'IMERG v07', 'ERA5', 'MERRA2']

ship_month_clim = build_oceanrain_ship_month_climatology(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_eval,
    min_days_per_ship_month=5,
)

monthly_ship = build_oceanrain_ship_year_month_means(
    daily_or_attached,
    obs_col="main_mmday",
    product_cols=products_eval,
    min_days_per_month=5,   # you can later test 3 or 5
)

fig_nh = plot_oceanrain_monthly_ship_scatter_by_hemi(
    monthly_ship,
    hemi="NH",
    products=["GPCP v1.3",'GPCP v3.2', 'GPCP v3.3', 
              'IMERG v07', 'ERA5', 'MERRA2'],
    figsize=(15, 7.5),
    xylim=(0, 10),
    add_titles=False,
)

fig_sh = plot_oceanrain_monthly_ship_scatter_by_hemi(
    monthly_ship,
    hemi="SH",
    products=["GPCP v1.3",'GPCP v3.2', 'GPCP v3.3', 
              'IMERG v07', 'ERA5', 'MERRA2'],
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
    "Buoy":      [(1997, 2025, "hourly")],
    "Atolls":    [(1983, 2023, "monthly")],
    "OceanRAIN": [(2010, 2017, "subhourly")],

    # Example: monthly long window + daily subset window (overlay)
    "GPCP v1.3": [(2000, 2020, "daily")],
    "GPCP v2.3": [(1983, 2025, "monthly")],
    "GPCP v3.2": [(1983, 2025, "monthly"), (2000, 2020, "daily")],
    "GPCP v3.3": [(1983, 2025, "monthly"), (2000, 2020, "daily")],
    "IMERG v07": [(1998, 2025, "monthly"), (2000, 2020, "daily")],  # adjust to your actual usage
    "ERA5":      [(1983, 2025, "monthly"), (2000, 2020, "daily")],
    "MERRA2":    [(1983, 2025, "monthly"), (2000, 2020, "daily")],
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

fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

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
ax.set_yticklabels(names, fontsize=15)
ax.set_xlim(1983, 2025)
ax.set_xticks(np.arange(1983, 2028, 6))

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
fig.legend(res_handles, res_labels, loc="lower center", bbox_to_anchor=(0.5, 0.08),
           ncol=4, frameon=False, fontsize=18, handlelength=2.5, columnspacing=1.8)
# fig.legend(style_handles, style_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
#            ncol=2, frameon=False, fontsize=12, handlelength=2.5, columnspacing=2.0)

fig.tight_layout(rect=[0.02, 0.15, 0.98, 0.95])
svnme = os.path.join(
    path_to_plots,
    f"Figure01_DatasetTemporalCoverage_Resolution_{cde_run_dte}.png"
)
fig.savefig(svnme, dpi=150, bbox_inches='tight')
# plt.show()
gc.collect()

