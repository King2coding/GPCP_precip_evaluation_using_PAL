#%%
# important links 
# https://data.pmel.noaa.gov/generic/erddap/info/pmelTaoDyRain/index.html
# buoy data download link: https://www.pmel.noaa.gov/tao/drupal/disdel/

# OceanRain article: https://www.nature.com/articles/sdata2018122

'''
This script explores ocean in-situ observations data from varying sources including:
moored buoys, Pacific Aquatic Listeners (PAL), and OceanRain products
Here, we summarize the available data spatiotemporally to inform usage in 
satellite product evaluation.

'''

#%%
import importlib
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])

from util_functions import *

#%%
# define the path to the data
path_to_pal_data = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'

moored_bouys_paf = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/Moored_Buoys'

path_to_ocean_rain = r'/ra1/pubdat/OceanRain/nc'

path_to_put_plts = r'/home/kkumah/Projects/Satellite_eval_over_Oceans/Results/plots'
#%%
# define global variables
cde_run_dte = str(date.today().strftime('%Y%m%d'))

all_pal_files = sorted([os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')])

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# read buoys data
# Define directories for each region
# List all directories in the parent folder
all_buoy_dirs = [os.path.join(moored_bouys_paf, d) for d in os.listdir(moored_bouys_paf) if os.path.isdir(os.path.join(moored_bouys_paf, d))]

# Filter directories based on region names
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
indian_buoy_files = sorted([os.path.join(indian_buoy_dir, f) for f in os.listdir(indian_buoy_dir) if f.endswith('.cdf')])
atlantic_buoy_files = sorted([os.path.join(atlantic_buoy_dir, f) for f in os.listdir(atlantic_buoy_dir) if f.endswith('.cdf')])

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
all_ocean_rain_files = sorted([os.path.join(path_to_ocean_rain, f) for f in os.listdir(path_to_ocean_rain) if f.endswith('.nc')])

# group ocean rain files by year
files_by_year = defaultdict(list)

for o in all_ocean_rain_files:
    print(os.path.basename(o))
    with xr.open_dataset(o) as ds:
        years = pd.to_datetime(ds['time'].values).year
        unique_years = np.unique(years)
        for year in unique_years:
            files_by_year[year].append(o)

files_by_year = dict(sorted(files_by_year.items()))

# OceanRain data file
# Load the .npz file using numpy
ocRain = np.load(r'/ra1/pubdat/OceanRain/output/OceanRAIN_MINUTE_coordinates_and_data_Kingsley.npz')

# Access the keys in the .npz file
keys = ocRain.files
print("Keys in the .npz file:", keys)

# Create a DataFrame from the .npz file
ocRain_df = pd.DataFrame({key: ocRain[key] for key in keys})

# Display the first few rows of the DataFrame
# print(ocRain_df.head())
# Subset ocRain_df to capture Atlantic Ocean data based on Buoy Atlantic Ocean data
atlantic_ocean_bounds = {
    "lat_min": -30,
    "lat_max": 30,
    "lon_min": -60,
    "lon_max": 20
}

atl_ocRain = ocRain_df[
    (ocRain_df['lat'] >= atlantic_ocean_bounds['lat_min']) &
    (ocRain_df['lat'] <= atlantic_ocean_bounds['lat_max']) &
    (ocRain_df['lon'] >= atlantic_ocean_bounds['lon_min']) &
    (ocRain_df['lon'] <= atlantic_ocean_bounds['lon_max'])
]

atl_ocRain.sort_values(by=['time_utc'], inplace=True)

print(f"Subset Atlantic Ocean data contains {atl_ocRain.shape[0]} records.")

gc.collect()
#%% CLASSIFY AND GROUP PAL FILES
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

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

gc.collect()

#%% VISUALIZING PAL, BUOY, AND OCEAN RAIN LOCATIONS
# Set font to Times New Roman and bold for all texts

# Define colors for each year
year_colors = {
    2010: 'black',
    2011: 'magenta',
    2012: 'green',
    2013: 'lime',
    2014: 'orange',
    2015: 'cyan',
    2016: 'blue',
    2017: 'red'
}

# Remove unavailable Times New Roman to avoid findfont warnings

print("\nGenerating spatial distribution plot of in-situ observations...")
fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-181, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot OceanRain data
# for yr, files in files_by_year.items():
#     color = year_colors.get(yr, 'gray')  # Default to gray if year not in dictionary
#     for f in files:
#         ds = xr.open_dataset(f, drop_variables=[v for v in xr.open_dataset(f).data_vars if v not in ['latitude', 'longitude']])
#         lat = ds['latitude'].values[::500]
#         lon = ds['longitude'].values[::500]
#         ax.scatter(lon, lat, color=color, s=3, label=f'OceanRain {yr}', transform=ccrs.PlateCarree())
#         ds.close()

for yr in np.unique(ocRain_df['time_utc'].dt.year):
    color = year_colors.get(yr, 'gray')  # Default to gray if year not in dictionary
    yr_data = ocRain_df[ocRain_df['time_utc'].dt.year == yr]
    ax.scatter(yr_data['lon'], yr_data['lat'], color=color, s=3, label=f'OceanRain {yr}', transform=ccrs.PlateCarree())
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 
for region, files in pals_classed_by_region.items():
    if region == "Unclassified" or len(files) == 0:
        continue  # Skip unclassified and empty regions for plotting bounds
    
    color = region_colors[region]
    for file in files:
        # Load only lat and lon efficiently, downsample by slicing
        ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
        lat = ds['lat'].values[::25]
        lon = ds['lon'].values[::25]
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=2)
        ds.close()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# Initialize buoy counts
buoy_counts = {"PACIFIC": len(pacific_buoy_files), "INDIAN": len(indian_buoy_files), "ATLANTIC": len(atlantic_buoy_files)}

for buoy_reg, buoy_files, marker in [("PACIFIC", pacific_buoy_files, '*'), 
                                     ("INDIAN", indian_buoy_files, 's'), 
                                     ("ATLANTIC", atlantic_buoy_files, 'd')]:
    for fl in buoy_files: 
        xrfile = xr.open_dataset(fl)       
        # Plot moored buoy data
        lat = xrfile['lat'].values[0]
        lon = xrfile['lon'].values[0]
        # make lon between 180 and -180
        lon = (lon + 180) % 360 - 180

        if lon == -180:
            # shift lon slightly for plotting
            lon = -178

        ax.scatter(lon, lat, color='black', s=25, marker=marker, label=f'{buoy_reg} Buoys', transform=ccrs.PlateCarree())

# Add grid lines for major ticks
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.7, linestyle='--')

# Create legend with full region names and PAL counts
legend_regions = [r for r in region_colors.keys() if r != "Unclassified"]
handles = [plt.Line2D([0], [0], color=region_colors[r], lw=2) for r in legend_regions]

# Full region names mapping
full_region_names = {
    "ETNP": "Extratropical North Pacific",
    "TNEP": "Tropical Northeastern Pacific", 
    "TSEP": "Tropical Southeastern Pacific",
    "STNA": "Subtropical North Atlantic",
    "TNIO": "Tropical North Indian Ocean",
    "TNWP": "Tropical Northwestern Pacific"
}

# Add full names and PAL counts to legend labels
labels = [f"{region}: ({full_region_names[region]} ({len(pals_classed_by_region.get(region, []))} PALs)" for region in legend_regions]

# Add buoy markers and counts to the legend
handles.extend([
    plt.Line2D([0], [0], color='black', marker='*', markersize=10, linestyle='None'),
    plt.Line2D([0], [0], color='black', marker='s', markersize=10, linestyle='None'),
    plt.Line2D([0], [0], color='black', marker='d', markersize=10, linestyle='None')
])
labels.extend([
    f"PACIFIC ({buoy_counts['PACIFIC']} Buoys)",
    f"INDIAN ({buoy_counts['INDIAN']} Buoys)",
    f"ATLANTIC ({buoy_counts['ATLANTIC']} Buoys)"
])

# Add OceanRain year markers to the legend
handles.extend([plt.Line2D([0], [0], color=year_colors[yr], marker='o', markersize=10, linestyle='None') for yr in year_colors.keys()])
labels.extend([f"OceanRain {yr}" for yr in year_colors.keys()])

leg = plt.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.5), 
    fontsize=12, ncol=4, frameon=False
)
# Set legend fontweight to bold
for text in leg.get_texts():
    text.set_fontweight('bold')
if leg.get_title() is not None:
    leg.get_title().set_fontweight('bold')

# Set ticks and format them with degree symbols and N/S/E/W
xticks = range(-180, 181, 60)
yticks = range(-30, 61, 15)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

ax.tick_params(labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
ax.set_title("Spatial Distribution of In Situ Observations Over Ocean Regions", fontsize=20, 
             fontweight='bold', fontname='Times New Roman')

plt.tight_layout()
plt.subplots_adjust(bottom=0.4)  # Add extra space at the bottom for legend

svname = os.path.join(path_to_put_plts, f'insitu_distribution_over_oceans_{cde_run_dte}.png')
# plt.savefig(svname, bbox_inches='tight', dpi=500)
# plt.show()
gc.collect()

print(f"Plot saved as: {svname}")
#%%
# NOW WE WILL COLLECT ALL INSTANCES AND DATA FOR WHERE PAL AND BUOY OCCUR IN THE SAME GPCP V3.3 PIXEL
focus_regions = ['TNEP', 'TNWP']

# FIRST GROUP BUOY FILES BY REGION BASED ON THEIR LONGITUDE
# Define longitude bounds for TNEP and TNWP
region_bounds = {
    'TNEP': (-180, -60),  # Longitude range for Tropical Northeastern Pacific in [-180, 180]
    'TNWP': (120, 180)      # Longitude range for Tropical Northwestern Pacific in [-180, 180]
}

# Group buoy files by region
buoy_files_by_region = {'TNEP': [], 'TNWP': []}
for buoy_file in pacific_buoy_files:
    with xr.open_dataset(buoy_file) as ds:
        buoy_lon = ds['lon'].values[0]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]
    
    for region, bounds in region_bounds.items():
        if bounds[0] <= buoy_lon <= bounds[1]:
            buoy_files_by_region[region].append(buoy_file)
            break

#%% OLD METHOD
# Initialize an empty list to store matches
# matches = []

# # Define the GPCP resolution
# gpcp_resolution = 2.5

# # Iterate over the focus regions
# for region in focus_regions:
#     print(f"Processing region: {region}")
#     pal_files = pals_classed_by_region.get(region, [])
#     buoy_files = buoy_files_by_region.get(region, [])
#     print(f"Number of PAL files in {region}: {len(pal_files)}")
#     print(f"Number of Buoy files in {region}: {len(buoy_files)}")
    
#     # Iterate over PAL files
#     for pal_file in pal_files:
#         print(f"Processing PAL file: {os.path.basename(pal_file)}")
#         # Extract PAL data into a DataFrame
#         with xr.open_dataset(pal_file) as ds:
#             pal_time = ds['time'].values
#             pal_lat = ds['lat'].values
#             pal_lon = ds['lon'].values
#             pal_rain_rate = ds['rain_rate'].values  # Assuming 'rain_rate' is the variable name
            
#         # Normalize longitude to [-180, 180]
#         pal_lon = (pal_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180] for compatibility with GPCP grid
        
#         # Create a DataFrame
#         pal_df = pd.DataFrame({
#             "time": pal_time,
#             "lat": pal_lat,
#             "lon": pal_lon,
#             "rain_rate": pal_rain_rate
#         })

#         # Filter out rows with NaN values or negative rain_rate
#         pal_df = pal_df.dropna(subset=['lat', 'lon', 'rain_rate'])
#         pal_df = pal_df[pal_df['rain_rate'] >= 0]

#         # Compute date from time and group by date to calculate daily means
#         pal_df['date'] = pd.to_datetime(pal_df['time']).dt.date
#         # daily_pal_df = pal_df.groupby('date').mean().reset_index()
        
#         pal_id = os.path.basename(pal_file)  # Extract PAL file ID from filename
        
#         # Iterate over Buoy files in the same region
#         for buoy_file in buoy_files:
#             # print(f"Comparing with Buoy file: {os.path.basename(buoy_file)}")
#             # Extract Buoy lat/lon
#             with xr.open_dataset(buoy_file) as ds:
#                 buoy_lat = ds['lat'].values[0]
#                 buoy_lon = ds['lon'].values[0]
#                 # Normalize longitude to [-180, 180]
#             buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

#             buoy_row, buoy_col = assign_to_gpcp_grid(buoy_lat, buoy_lon, gpcp_resolution)
#             buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            
    
#             # Check if any daily PAL lat/lon falls in the same GPCP pixel as the Buoy
#             pal_df['row_col'] = pal_df.apply(
#                 lambda row: assign_to_gpcp_grid(row['lat'], row['lon'], gpcp_resolution), axis=1
#             )
            
#             # Check for matches
#             match_indices = pal_df['row_col'].apply(
#                 lambda rc: rc == (buoy_row, buoy_col)
#             )
            
#             if match_indices.any():
#                 print(f"Match found between PAL file {pal_id} and Buoy file {buoy_id} in region {region} at row {buoy_row}, col {buoy_col}")
#                 # Store the match details
#                 match_df = pal_df[match_indices].copy()
#                 match_df['PAL_File'] = pal_file
#                 match_df['Buoy_File'] = buoy_file
#                 match_df['Buoy_Row'] = buoy_row
#                 match_df['Buoy_Col'] = buoy_col
#                 match_df['PAL_ID'] = pal_id
#                 match_df['Buoy_ID'] = buoy_id
#                 match_df['PAL_Region'] = region
#                 match_df['Buoy_Region'] = region
                
#                 # Append the match DataFrame to the matches list
#                 matches.append(match_df)
#             # else:
#             #     print(f"No match found for PAL file {pal_id} with Buoy file {buoy_id}")

# # Concatenate all match DataFrames into a single DataFrame
# if matches:
#     pal_buoy_matches = pd.concat(matches, ignore_index=True)
#     print("Matching process completed. Here are the first few matches:")
#     print(pal_buoy_matches.head())
# else:
#     print("No matches found between PAL and Buoy files.")

# gc.collect()


#%%
# NEW METHOD
'''
grid_resolution = 2.5
dfs_by_region = {}
for region in focus_regions:
    print(f"Processing region: {region}")
    
    pal_files = pals_classed_by_region.get(region, [])
    # buoy_files = buoy_files_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    # print(f"Number of Buoy files in {region}: {len(buoy_files)}")

    # Initialize a list to store DataFrames for this region
    region_dfs = []
    
    # Iterate over PAL files
    for pal_file in pal_files:
        # print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL data into a DataFrame
        with xr.open_dataset(pal_file) as ds:
            pal_time = ds['time'].values
            pal_lat = ds['lat'].values
            pal_lon = ds['lon'].values
            pal_rain_rate = ds['rain_rate'].values  # Assuming 'rain_rate' is the variable name
            
        # Normalize longitude to [-180, 180]
        pal_lon = (pal_lon + 180) % 360 - 180
        
        # Create a DataFrame
        pal_df = pd.DataFrame({
            "lat": pal_lat,
            "lon": pal_lon,
        })
        # Filter out rows with NaN values or negative rain_rate
        pal_df = pal_df.dropna(subset=['lat', 'lon'])


        # Original code for reference
        # pal_df[['PAL_row', 'PAL_col']] = pal_df.apply(
        #     lambda row: pd.Series(find_grid_indices(row['lat'], row['lon'], grid_resolution)), axis=1
        # )

        # find latlon grid indices using vectorized operations
        pal_df['PAL_row'] = ((90 - pal_df['lat']) // grid_resolution).astype(int)
        pal_df['PAL_col'] = ((pal_df['lon'] + 180) // grid_resolution).astype(int)

        pal_df['PAL_File'] = pal_file
        pal_df['PAL_ID'] = os.path.basename(pal_file)
        pal_df['PAL_Region'] = region

        # Append the DataFrame for this PAL file to the list for the region
        region_dfs.append(pal_df)

    # Concatenate all DataFrames for this region into a single DataFrame
    if region_dfs:
       dfs_by_region[region] = pd.concat(region_dfs, ignore_index=True)

gc.collect()
'''
#%%
# Now, iterate over Buoy files and check for matches with PAL DataFrames
'''
region_match_buoy_pal_df = {}
for region, pal_df in dfs_by_region.items():
    print(f"Processing Buoy files for region: {region}")    
    
    buoy_files = buoy_files_by_region.get(region, [])

    match_buoy_pal_df = []
    
    # Iterate over Buoy files in the same region
    for buoy_file in buoy_files:
        
        # Extract Buoy lat/lon
        # with xr.open_dataset(buoy_file) as ds:
        ds  = xr.open_dataset(buoy_file) #as ds
        buoy_lat = ds['lat'].values[0]
        buoy_lon = ds['lon'].values[0]
            # Normalize longitude to [-180, 180]
        buoy_lon = (buoy_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180] for consistency

        # buoy_row, buoy_col = find_grid_indices(buoy_lat, buoy_lon, grid_resolution)
        # find latlon grid indices using vectorized operations
        buoy_row = ((90-buoy_lat) // grid_resolution).astype(int)
        buoy_col = ((buoy_lon + 180) // grid_resolution).astype(int)

        buoy_id = os.path.basename(buoy_file)  # Extract Buoy file ID from filename            
        
        # Check if any PAL lat/lon falls in the same GPCP pixel as the Buoy
        match_pal_df = pal_df.copy()
        match_pal_df = match_pal_df.loc[(match_pal_df['PAL_row'] == buoy_row) & (match_pal_df['PAL_col'] == buoy_col)]
        if match_pal_df.shape[0] > 0:
            print(f"Match found between PAL file {match_pal_df['PAL_ID'].iloc[0]} and Buoy file {buoy_id} "
                  f"in region {region} at row {buoy_row}, col {buoy_col}")
            # Store the match details
            
            match_pal_df['Buoy_File'] = buoy_file
            match_pal_df['Buoy_Lat'] = buoy_lat
            match_pal_df['Buoy_Lon'] = buoy_lon
            match_pal_df['Buoy_Row'] = buoy_row
            match_pal_df['Buoy_Col'] = buoy_col
            match_pal_df['Buoy_ID'] = buoy_id
            match_pal_df['Buoy_Region'] = region

            # Append the match DataFrame to the matches list
            match_buoy_pal_df.append(match_pal_df)
    region_match_buoy_pal_df[region] = pd.concat(match_buoy_pal_df, ignore_index=True)

gc.collect()

# plot the buoy and pal locations for the matches found
fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-181, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')
dfplot = region_match_buoy_pal_df['TNWP'].copy()
ax.scatter(dfplot['lon'], dfplot['lat'], transform=ccrs.PlateCarree(), color='blue', s=2.5, label='PAL Locations')
ax.scatter(dfplot['Buoy_Lon'], dfplot['Buoy_Lat'], 
           transform=ccrs.PlateCarree(), color='red', s=5, label='Buoy Locations')

dfplot = region_match_buoy_pal_df['TNEP'].copy()
ax.scatter(dfplot['lon'], dfplot['lat'], transform=ccrs.PlateCarree(), color='blue', s=2.5)
ax.scatter(dfplot['Buoy_Lon'], dfplot['Buoy_Lat'], 
           transform=ccrs.PlateCarree(), color='red', s=5)

# Show all tick marks and labels on all sides
ax.tick_params(axis='both', which='both', direction='in', 
               length=6, width=1.5, labelsize=12, top=True, 
               bottom=True, left=True, right=True, 
               labeltop=True, labelright=True)

# Add grid lines for major ticks
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.7, linestyle='--')

# Set ticks and format them with degree symbols and N/S/E/W
xticks = range(-180, 181, 60)
yticks = range(-30, 61, 15)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# Add legend
ax.legend(loc='lower left', fontsize=12, frameon=False)
'''
#%% DO A MORE FOCUSED ANALYSIS OF IN SITU RAINFALL RATES
# CONSIDERING PAL AND BUOYS IN 2.5 DEGREE GRID CELLS
'''
regional_pal_buoy_data = {'TNEP': [], 'TNWP': []}
for region, df in region_match_buoy_pal_df.items():

    pal_ids = df['PAL_ID'].unique()

    for pID in pal_ids:
        pal_df_by_pID = df[df['PAL_ID'] == pID]

        pal_fle = pal_df_by_pID['PAL_File'].values[0]

        # Extract PAL data
        with xr.open_dataset(pal_fle) as ds:
            pal_time = ds['time'].values
            pal_lat = ds['lat'].values
            pal_lon = ds['lon'].values
            pal_rain_rate = ds['rain_rate'].values  # Assuming 'rain_rate' is the variable name
            
        # Normalize longitude to [-180, 180]
        pal_lon = (pal_lon + 180) % 360 - 180
        
        # Create a DataFrame
        pal_df = pd.DataFrame({
            "time": pal_time,
            "pal_rain_rate": pal_rain_rate,
            "lat": pal_lat,
            "lon": pal_lon,
        })
        # Filter out rows with NaN values or negative rain_rate
        pal_df = pal_df.dropna(subset=['pal_rain_rate','lat', 'lon'])

        pal_df['PAL_row'] = ((90 - pal_df['lat']) // grid_resolution).astype(int)
        pal_df['PAL_col'] = ((pal_df['lon'] + 180) // grid_resolution).astype(int)
        pal_df['date'] = pd.to_datetime(pal_df['time']).dt.date  # Extract date from time
        pal_df = pal_df.dropna(axis=0, how='any')

        pal_df = pal_df[pal_df['pal_rain_rate'] >= 0]

        # groupby date and get mean of rain_rate and GPCP data
        daily_avg_rain = pal_df.groupby(['date', 'PAL_row', 'PAL_col']).mean(['pal_rain_rate',]).reset_index()

        # multiply PAL rain rate by 24 to get daily average
        daily_avg_rain['pal_rain_rate'] *= 24

        buoy_ids = pal_df_by_pID['Buoy_ID'].unique()

        b_elem = []

        for bID in buoy_ids:
            buoy_df_by_bID = pal_df_by_pID[pal_df_by_pID['Buoy_ID'] == bID]

            buoy_fle = buoy_df_by_bID['Buoy_File'].values[0]

            b_ds = xr.open_dataset(buoy_fle)
            b_lat = b_ds['lat'].values[0]
            b_lon = b_ds['lon'].values[0]
            b_lon = (b_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]

            b_df = b_ds[['time', 'RN_485', 'QRN_5485']].to_dataframe().reset_index()
            b_df.rename(columns={'RN_485': 'buoy_rain_rate', 'QRN_5485': 'quality_flag'}, inplace=True)
            b_df['date'] = pd.to_datetime(b_df['time']).dt.date  # Extract date from time
            # Filter out rows with negative rain_rate
            b_df = b_df[b_df['buoy_rain_rate'] >= 0]

            # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
            b_df = b_df[(b_df['quality_flag'] >= 1) & (b_df['quality_flag'] <= 3)]

            # multiply buoy rain rate by 24 to get daily average
            b_df['buoy_rain_rate'] *= 24

            # find latlon grid indices using vectorized operations
            buoy_row = ((90-b_lat) // grid_resolution).astype(int)
            buoy_col = ((b_lon + 180) // grid_resolution).astype(int)
            b_df['Buoy_row'] = buoy_row
            b_df['Buoy_col'] = buoy_col
            b_df['Buoy_ID'] = bID

            pal_df_sub = daily_avg_rain.copy()
            pal_df_sub = pal_df_sub[(pal_df_sub['PAL_row'] == buoy_row) & (pal_df_sub['PAL_col'] == buoy_col)]

            # merge pal and buoy data on date
            merged_df = pd.merge(pal_df_sub, b_df, left_on=['date'], 
                                 right_on=['date'], 
                                 suffixes=('_PAL', '_Buoy'))
            
            if merged_df.shape[0] > 0:
                print(f"Merged data for PAL ID {pID} with Buoy data, resulting in {merged_df.shape[0]} records.")
                regional_pal_buoy_data[region].append((merged_df))# , pal_df_by_pID, b_df_comb


            # b_elem.append(b_df)

        # b_df_comb = pd.concat(b_elem, ignore_index=True)
        # b_df_comb = b_df_comb.groupby(['date','Buoy_row', 'Buoy_col'])['buoy_rain_rate'].mean().reset_index()

        # merge pal and buoy data on date
        # merged_df = pd.merge(daily_avg_rain, b_df_comb, left_on=['date'], 
        #                      right_on=['date'], 
        #                      suffixes=('_PAL', '_Buoy'))
        # if merged_df.shape[0] > 0:
        #     print(f"Merged data for PAL ID {pID} with Buoy data, resulting in {merged_df.shape[0]} records.")
        #     regional_pal_buoy_data[region].append((merged_df))# , pal_df_by_pID, b_df_comb

        # print(pID)
  
fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-181, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')
# dfplot = pal_df.copy()
# ax.scatter(dfplot['lon'], dfplot['lat'], transform=ccrs.PlateCarree(), 
#            color='blue', s=2.5, label='PAL Locations')
dfplot = b_elem[0].copy()
ax.scatter(dfplot['lon'], dfplot['lat'], 
           transform=ccrs.PlateCarree(), color='red', s=1.5, 
           label='Buoy Locations')

dfplot = b_elem[1].copy()
ax.scatter(dfplot['lon'], dfplot['lat'], 
           transform=ccrs.PlateCarree(), 
           color='blue', s=1.5)
# ax.scatter(dfplot['Buoy_Lon'], dfplot['Buoy_Lat'], 
#            transform=ccrs.PlateCarree(), color='red', s=5)

# Show all tick marks and labels on all sides
ax.tick_params(axis='both', which='both', direction='in', 
               length=6, width=1.5, labelsize=12, top=True, 
               bottom=True, left=True, right=True, 
               labeltop=True, labelright=True)

# Add grid lines for major ticks
ax.grid(True, which='major', linewidth=0.55, color='grey', alpha=0.7, linestyle='--')

# Set ticks and format them with degree symbols and N/S/E/W
xticks = range(-180, 181, 60)
yticks = range(-30, 61, 15)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# Add legend
ax.legend(loc='lower left', fontsize=12, frameon=False)
'''
#%% PAL BUOY RAINFALL RATE COMPARISON BY OCEAN REGION
# lets compare rainfall rates between PAL and Buoy data for the two focus regions
# make pal and buoy dfs by region
pal_dfs_by_region = {}
buoy_dfs_by_region = {}

for region in focus_regions:
    print(f"Processing region: {region}")
    
    pal_files = pals_classed_by_region.get(region, [])
    buoy_files = buoy_files_by_region.get(region, [])
    print(f"Number of PAL files in {region}: {len(pal_files)}")
    print(f"Number of Buoy files in {region}: {len(buoy_files)}")

    # Initialize lists to store DataFrames for this region
    pal_region_dfs = []
    buoy_region_dfs = []
    
    # Iterate over PAL files
    for pal_file in pal_files:
        # print(f"Processing PAL file: {os.path.basename(pal_file)}")
        # Extract PAL data into a DataFrame
        # with xr.open_dataset(pal_file) as ds:
        pal_ds = xr.open_dataset(pal_file)
        pal_df = pd.DataFrame({
            'time': pd.to_datetime(pal_ds['time'].values),            
            'rain_rate': pal_ds['rain_rate'].values,
            'wind_speed': pal_ds['wind_speed'].values,
        })

        # Filter out rows with negative rain_rate
        pal_df = pal_df[pal_df['rain_rate'] >= 0]

        # drop rows where wind speed is >= 15 m/s
        pal_df = pal_df.drop(pal_df[pal_df['wind_speed'] >= 15].index)

        pal_df['date'] = pal_df['time'].dt.date  # Extract date from time

        pal_df = pal_df.dropna(axis=0, how='any')

        # groupby date and get mean of rain_rate and GPCP data
        daily_avg_rain = pal_df.groupby('date').mean(['rain_rate',]).reset_index()

        # multiply PAL rain rate by 24 to get daily average
        daily_avg_rain['rain_rate'] *= 24
        daily_avg_rain['ID'] = os.path.basename(os.path.basename(pal_file))  # Extract PAL file ID from filename

        # append the DataFrame for this PAL file to the list for the region
        pal_region_dfs.append(daily_avg_rain)

    # Concatenate all DataFrames for this region into a single DataFrame
    pal_dfs_by_region[region] = pd.concat(pal_region_dfs)

    # Iterate over Buoy files in the same region
    for buoy_file in buoy_files:
        buoy_ds = xr.open_dataset(buoy_file)
        buoy_df = pd.DataFrame({
            'time': pd.to_datetime(buoy_ds['time'].values),            
            'rain_rate': buoy_ds['RN_485'].values.flatten(),
            'quality_code': buoy_ds['QRN_5485'].values.flatten(),
        })

        # Filter out rows with negative rain_rate
        buoy_df = buoy_df[buoy_df['rain_rate'] >= 0]

        # buoys cover longer time period than pal so select date time period covering pal datetime period range
        pal_start_date = pal_dfs_by_region[region]['date'].min()
        pal_end_date = pal_dfs_by_region[region]['date'].max()
        buoy_df = buoy_df[(buoy_df['time'] >= pd.to_datetime(pal_start_date)) & (buoy_df['time'] <= pd.to_datetime(pal_end_date))]

        # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
        buoy_df = buoy_df[(buoy_df['quality_code'] >= 1) & (buoy_df['quality_code'] <= 3)]

        # daily rain rate
        buoy_df['rain_rate'] *= 24
        buoy_df['ID'] = os.path.basename(os.path.basename(buoy_file))  # Extract Buoy file ID from filenamebuoy_df['time'].dt.date  # Extract date from time

        # append the DataFrame for this Buoy file to the list for the region
        buoy_region_dfs.append(buoy_df)

    # Concatenate all DataFrames for this region into a single DataFrame
    buoy_dfs_by_region[region] = pd.concat(buoy_region_dfs)

gc.collect()


#%% PDFc PDFv CALCULATION AND PLOTTING

# tnep_dfs = pd.concat(regional_pal_buoy_data['TNEP'])
# tnep_pal_pdfc_pdfv = compute_pdf_elements(tnep_dfs,'pal_rain_rate', bin_values)
# tnep_buoy_pdfc_pdfv = compute_pdf_elements(tnep_dfs,'buoy_rain_rate', bin_values)

tnep_pal_pdfc_pdfv = compute_pdf_elements(pal_dfs_by_region['TNEP'], bin_values)
tnwp_pal_pdfc_pdfv = compute_pdf_elements(pal_dfs_by_region['TNWP'], bin_values)

tnep_buoy_pdfc_pdfv = compute_pdf_elements(buoy_dfs_by_region['TNEP'], bin_values)
tnwp_buoy_pdfc_pdfv = compute_pdf_elements(buoy_dfs_by_region['TNWP'], bin_values)


# make a 2by 2 line plot of the pdfc and pdfv for pal and buoy data by region
fig, axs = plt.subplots(2, 2, figsize=(16, 10), 
                        sharex=False, sharey=False, dpi=1000)

# Set common x-axis ticks and labels
bin_labels = ['0.5', '1', '2', '4', '8', '16', '32', ] # '64', '128', '256'
bin_positions = range(len(bin_labels))
lw = 2
# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Line Plot TNEP PDFc
axs[0, 0].plot(bin_positions, tnep_pal_pdfc_pdfv['pdfc'], label='PAL', marker='o',lw=lw)
axs[0, 0].plot(bin_positions, tnep_buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='x',lw=lw)
axs[0, 0].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0, 0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[0, 0].set_xticks(bin_positions)
axs[0, 0].set_xticklabels(bin_labels)
axs[0, 0].legend(fontsize=18, frameon=False)

# Line Plot TNEP PDFv
axs[0, 1].plot(bin_positions, tnep_pal_pdfc_pdfv['pdfv'], label='PAL', marker='o', lw=lw)
axs[0, 1].plot(bin_positions, tnep_buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='x', lw=lw)
axs[0, 1].set_title('TNEP', fontsize=18, fontweight='bold')
axs[0, 1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[0, 1].set_xticks(bin_positions)
axs[0, 1].set_xticklabels(bin_labels)
axs[0, 1].legend(fontsize=18, frameon=False)

# Line Plot TNWP PDFc
axs[1, 0].plot(bin_positions, tnwp_pal_pdfc_pdfv['pdfc'], label='PAL', marker='o')
axs[1, 0].plot(bin_positions, tnwp_buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='x')
axs[1, 0].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1, 0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[1, 0].set_xticks(bin_positions)
axs[1, 0].set_xticklabels(bin_labels)
axs[1, 0].legend(fontsize=18, frameon=False)

# Line Plot TNWP PDFv
axs[1, 1].plot(bin_positions, tnwp_pal_pdfc_pdfv['pdfv'], label='PAL', marker='o', lw=lw)
axs[1, 1].plot(bin_positions, tnwp_buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='x', lw=lw)
axs[1, 1].set_title('TNWP', fontsize=18, fontweight='bold')
axs[1, 1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[1, 1].set_xticks(bin_positions)
axs[1, 1].set_xticklabels(bin_labels)
axs[1, 1].legend(fontsize=18, frameon=False)

# Adjust layout
plt.tight_layout()

# # save the figure
# svnme = os.path.join(path_to_put_plts, f'pdfc_pdfv_comparison_{cde_run_dte}.png')
# plt.savefig(svnme, bbox_inches='tight')
# tnep_dfs['year'] = pd.to_datetime(tnep_dfs['date']).dt.year
# tnep_dfs['month'] = pd.to_datetime(tnep_dfs['date']).dt.month

# compute monthly means for pal and buoy data
# tnep_monthly_means = tnep_dfs.groupby(['year', 'month'])[['pal_rain_rate', 'buoy_rain_rate']].sum().reset_index()
# tnep_monthly_means = tnep_monthly_means.groupby('month')[['pal_rain_rate', 'buoy_rain_rate']].mean().reset_index()

# Plotting comparison of multiyear monthly mean rainfall rate for PAL and Buoy data
# Create a 1x1 plot for multiyear monthly mean rainfall rate comparison
# fig, axs = plt.subplots(1, 1, figsize=(16, 10), sharex=False, sharey=True, dpi=100)
# # Update matplotlib parameters for consistent styling
# axs.plot(tnep_monthly_means['month'], tnep_monthly_means['pal_rain_rate'], label='PAL', marker='o', lw=lw)
# axs.plot(tnep_monthly_means['month'], tnep_monthly_means['buoy_rain_rate'], label='Buoy', marker='x', lw=lw)
# axs.set_title('ENP', fontsize=18, fontweight='bold')
# axs.set_ylabel('Rainfall Rate (mm)', fontsize=18, fontweight='bold')
# axs.set_xticks(month_positions)
# axs.set_xticklabels(month_labels)
# axs.legend(fontsize=18, frameon=False)
#%%
# calculate multiyear monthly mean rainfall rate for PAL and Buoy data

# Call the function
pal_monthly_means_by_region,pal_yr_by_yr = calculate_multiyear_monthly_mean_rainfall_by_region(pal_dfs_by_region,'date')
buoy_monthly_means_by_region,buoy_yr_by_yr = calculate_multiyear_monthly_mean_rainfall_by_region(buoy_dfs_by_region,'time')


# Plotting the multiyear monthly mean rainfall rate for PAL and Buoy data
# Create a 2x1 plot for multiyear monthly mean rainfall rate comparison
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=False, sharey=True, dpi=100)

# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Plot TNEP data
axs[0].plot(month_positions, pal_monthly_means_by_region['TNEP']['rain_rate'], label='PAL', marker='o', lw=lw)
axs[0].plot(month_positions, buoy_monthly_means_by_region['TNEP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[0].set_title('ENP', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')
axs[0].set_xticks(month_positions)
axs[0].set_xticklabels(month_labels)
axs[0].legend(fontsize=18, frameon=False)

# Plot TNWP data
axs[1].plot(month_positions, pal_monthly_means_by_region['TNWP']['rain_rate'], label='PAL', marker='o', lw=lw)
axs[1].plot(month_positions, buoy_monthly_means_by_region['TNWP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[1].set_title('WNP', fontsize=18, fontweight='bold')
axs[1].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')
axs[1].set_xticks(month_positions)
axs[1].set_xticklabels(month_labels)
axs[1].legend(fontsize=18, frameon=False)

# Adjust layout
plt.tight_layout()
gc.collect()

# Save the figure
svnme = os.path.join(path_to_put_plts, f'multiyear_monthly_mean_comparison_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight')

# - - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  - - - --  

fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=False, sharey=True, dpi=100)

# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Plot TNEP data
axs[0].plot(pal_yr_by_yr['TNEP']['date'], pal_yr_by_yr['TNEP']['rain_rate'], label='PAL', marker='o', lw=lw)
axs[0].plot(buoy_yr_by_yr['TNEP']['date'], buoy_yr_by_yr['TNEP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[0].set_title('ENP', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')
axs[0].legend(fontsize=18, frameon=False)

# X-axis ticks and labels for axs[0]
# Fill missing data in the WNP dataframe with NaN for the gap between 2016 and 2020
wnp_full_date_range = pd.date_range(start='2012-04-01', end='2021-08-31', freq='MS')  # Monthly start frequency
wnp_df = pal_yr_by_yr['TNWP'].set_index('date').reindex(wnp_full_date_range).reset_index()[['index', 'rain_rate']]
wnp_df.columns = ['date', 'rain_rate']  # Rename columns after reindexing

# Plot TNEP data
custom_ticks = pd.date_range(start=pal_yr_by_yr['TNEP']['date'].min(), 
                             end=np.datetime64('2021-08-31'), 
                             freq='YS')  # Year start frequency
tick_labels = [f"{t.strftime('%b')}\n{t.strftime('%Y')}" for t in custom_ticks]

axs[0].set_xticks(custom_ticks)
axs[0].set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=16, fontweight='bold', linespacing=1.5)

# Add minor ticks on x-axis
axs[0].xaxis.set_minor_locator(AutoMinorLocator(2))

# X-axis limits for axs[0]
start = pal_yr_by_yr['TNEP']['date'].min()
end = np.datetime64('2021-08-31')
axs[0].set_xlim(start, end)
axs[0].set_ylim(0, 20)

# Y-axis: reduce ticks and add minor ticks
axs[0].set_ylabel('Rainfall [mm/day]', fontsize=18, fontweight='bold')
axs[0].yaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between each major

# Make major and minor ticks more visible
axs[0].tick_params(axis='x', which='major', length=12, width=2)
axs[0].tick_params(axis='x', which='minor', length=6, width=1)
axs[0].tick_params(axis='y', which='major', length=12, width=2)
axs[0].tick_params(axis='y', which='minor', length=6, width=1)

# Title and labels
axs[0].set_xlabel('Month, Year', fontsize=18, fontweight='bold', labelpad=10)

# Y-tick label style
for label in axs[0].get_yticklabels():
    label.set_fontsize(15)
    label.set_fontweight('bold')

# Add legend
axs[0].legend(fontsize=14, loc='upper right', frameon=False)

# Plot TNWP data with the gap filled
axs[1].plot(wnp_df['date'], wnp_df['rain_rate'], label='PAL', marker='o', lw=lw)
axs[1].plot(buoy_yr_by_yr['TNWP']['date'], buoy_yr_by_yr['TNWP']['rain_rate'], label='Buoy', marker='x', lw=lw)
axs[1].set_title('WNP', fontsize=18, fontweight='bold')
axs[1].set_ylabel('Rainfall [mm]', fontsize=18, fontweight='bold')

# X-axis ticks and labels for axs[1]
custom_ticks = pd.date_range(start=wnp_df['date'].min(), 
                             end=wnp_df['date'].max(), 
                             freq='YS')  # Year start frequency
tick_labels = [f"{t.strftime('%b')}\n{t.strftime('%Y')}" for t in custom_ticks]

axs[1].set_xticks(custom_ticks)
axs[1].set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=16, fontweight='bold', linespacing=1.5)

# Add minor ticks on x-axis
axs[1].xaxis.set_minor_locator(AutoMinorLocator(2))

# X-axis limits for axs[1]
start = wnp_df['date'].min()
end = wnp_df['date'].max()
axs[1].set_xlim(start, end)
axs[1].set_ylim(0, 20)

# Y-axis: reduce ticks and add minor ticks
axs[1].set_ylabel('Rainfall [mm/day]', fontsize=18, fontweight='bold')
axs[1].yaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between each major

# Make major and minor ticks more visible
axs[1].tick_params(axis='x', which='major', length=12, width=2)
axs[1].tick_params(axis='x', which='minor', length=6, width=1)
axs[1].tick_params(axis='y', which='major', length=12, width=2)
axs[1].tick_params(axis='y', which='minor', length=6, width=1)

# Title and labels
axs[1].set_xlabel('Month, Year', fontsize=18, fontweight='bold', labelpad=10)

# Y-tick label style
for label in axs[1].get_yticklabels():
    label.set_fontsize(15)
    label.set_fontweight('bold')

# Add legend
axs[1].legend(fontsize=14, loc='upper right', frameon=False)

# Adjust layout
plt.tight_layout()
svnme = os.path.join(path_to_put_plts, f'multiyear_yr_by_yr_comparison_{cde_run_dte}.png')
plt.savefig(svnme, bbox_inches='tight',dpi=500)
plt.close()
gc.collect()


#%%
# scatterplot of PAL and Buoy data for TNEP and TNWP using the monthly means
# Calculate metrics for TNEP and TNWP
# tnep_rb, tnep_rmse, tnep_cc = calculate_metrics(pal_monthly_means_by_region['TNEP']['rain_rate'], 
#                                                 buoy_monthly_means_by_region['TNEP']['rain_rate'])

# tnwp_rb, tnwp_rmse, tnwp_cc = calculate_metrics(pal_monthly_means_by_region['TNWP']['rain_rate'], 
#                                                 buoy_monthly_means_by_region['TNWP']['rain_rate'])
# # plot scatter plot of PAL and Buoy data for TNEP and TNWP using the monthly means
# fig, axs = plt.subplots(1, 2, figsize=(18, 10), sharex=False, sharey=False, dpi=1000)

# # Removed 'Times New Roman' to avoid findfont warnings
# mpl.rcParams['ytick.labelsize'] = 18    

# # Add grid lines and customize ticks
# for ax in axs.flat:
#     ax.grid(True, which='major', linestyle='--', alpha=0.7)
#     ax.tick_params(axis='both', which='major', length=8, width=1.5)

# # Scatter Plot TNEP
# axs[0].scatter(pal_monthly_means_by_region['TNEP']['rain_rate'], buoy_monthly_means_by_region['TNEP']['rain_rate'], 
#                label='TNEP', color='blue', alpha=0.7, edgecolors='w', s=100)
# axs[0].set_title('TNEP', fontsize=18, fontweight='bold')
# axs[0].set_xlabel('PAL Rainfall [mm]', fontsize=18, fontweight='bold')
# axs[0].set_ylabel('Buoy Rainfall [mm]', fontsize=18, fontweight='bold')
# axs[0].set_xlim(0, 250)
# axs[0].set_ylim(0, 250)
# # Add 1:1 line
# axs[0].plot([0, 250], [0, 250], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
# # axs[0].legend(fontsize=18, frameon=False)

# axs[0].text(
#     0.05, 0.95,
#     f'RB: {tnep_rb:.2f}%\nRMSE: {tnep_rmse:.2f} mm\nCC: {tnep_cc:.2f}',
#     transform=axs[0].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
#     bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
# )

# # Scatter Plot TNWP
# axs[1].scatter(pal_monthly_means_by_region['TNWP']['rain_rate'], buoy_monthly_means_by_region['TNWP']['rain_rate'], 
#                label='TNWP', color='red', alpha=0.7, edgecolors='w', s=100)
# axs[1].set_title('TNWP', fontsize=18, fontweight='bold')
# axs[1].set_xlabel('PAL Rainfall [mm]', fontsize=18, fontweight='bold')
# axs[1].set_ylabel('Buoy Rainfall [mm]', fontsize=18, fontweight='bold')
# axs[1].set_xlim(0, 250)
# axs[1].set_ylim(0, 250)
# # Add 1:1 line
# axs[1].plot([0, 250], [0, 250], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
# # axs[1].legend(fontsize=18, frameon=False)

# axs[1].text(
#     0.05, 0.95,
#     f'RB: {tnwp_rb:.2f}%\nRMSE: {tnwp_rmse:.2f} mm\nCC: {tnwp_cc:.2f}',
#     transform=axs[1].transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
#     bbox=dict(facecolor='none', alpha=0.8, edgecolor='none')
# )

# # Adjust layout
# plt.tight_layout()
# gc.collect()

# # Save the figure
# svnme = os.path.join(path_to_put_plts, f'multiyear_monthly_mean_scatter_{cde_run_dte}.png')
# plt.savefig(svnme, bbox_inches='tight')

# %% A DEDICTAED COMPARATIVE ANALYSIS OF BUOY AND OCEANRAIN DATA
atl_buoy_dfs = []
for bfle in atlantic_buoy_files:
    buoy_ds = xr.open_dataset(bfle)
    buoy_df = pd.DataFrame({
        'time': pd.to_datetime(buoy_ds['time'].values),            
        'rain_rate': buoy_ds['RN_485'].values.flatten(),
        'quality_code': buoy_ds['QRN_5485'].values.flatten(),
    })

    # Filter out rows with negative rain_rate
    buoy_df = buoy_df[buoy_df['rain_rate'] >= 0]
    buoy_df = buoy_df.dropna(subset=['rain_rate'], axis=0, how='any')

    # buoys cover longer time period than pal so select date time period covering pal datetime period range
    ocR_yr_start_date = atl_ocRain['time_utc'].min()
    ocR_yr_end_date = atl_ocRain['time_utc'].max()
    buoy_df = buoy_df[(buoy_df['time'] >= ocR_yr_start_date) & (buoy_df['time'] <= ocR_yr_end_date)]

    # To get probably valid data only, request QRN_5485>=1 and QRN_5485<=3.
    buoy_df = buoy_df[(buoy_df['quality_code'] >= 1) & (buoy_df['quality_code'] <= 3)]

    # daily rain rate
    buoy_df['rain_rate'] *= 24
    buoy_df['ID'] = os.path.basename(os.path.basename(bfle))  # Extract Buoy file ID from filename

    atl_buoy_dfs.append(buoy_df)

# - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - -
# compute and plot monthly and year means
atl_buoy_df = pd.concat(atl_buoy_dfs, ignore_index=True)
atl_buoy_df['month'] = atl_buoy_df['time'].dt.month
atl_buoy_df['year'] = atl_buoy_df['time'].dt.year
atl_buoy_df_mnth_means_by_year = atl_buoy_df.groupby(['year','month'])['rain_rate'].mean().reset_index()
atl_buoy_df_mnth_means_by_year['date'] = pd.to_datetime(atl_buoy_df_mnth_means_by_year[['year', 'month']].assign(day=1))
# atl_buoy_monthly_sums = atl_buoy_df.groupby(['year', 'month'])['rain_rate'].mean().reset_index()
atl_buoy_monthly_means = atl_buoy_df_mnth_means_by_year.groupby('month')['rain_rate'].mean().reset_index()

# do similar computations for ocRain data
atl_ocR_df = atl_ocRain.copy()
atl_ocR_df.dropna(subset=['rate_dsd_mmph','rate_gag_mmph'], axis=0, how='any', inplace=True)
atl_ocR_df['date'] = atl_ocR_df['time_utc'].dt.date  # Extract date from time
atl_ocR_df_daily = atl_ocR_df.groupby(atl_ocR_df['date'])[['rate_dsd_mmph','rate_gag_mmph']].mean().reset_index()
atl_ocR_df_daily['rate_dsd_mmph'] *= 24
atl_ocR_df_daily['rate_gag_mmph'] *= 24
atl_ocR_df_daily['date'] = pd.to_datetime(atl_ocR_df_daily['date'])
atl_ocR_df_daily['month'] = atl_ocR_df_daily['date'].dt.month
atl_ocR_df_daily['year'] = atl_ocR_df_daily['date'].dt.year
atl_ocR_df_month_means_by_year = atl_ocR_df_daily.groupby(['year','month'])[['rate_dsd_mmph','rate_gag_mmph']].mean().reset_index()
atl_ocR_df_month_means_by_year['date'] = pd.to_datetime(atl_ocR_df_month_means_by_year[['year', 'month']].assign(day=1))

# replace annoying spikes in the data with NAN
atl_ocR_df_month_means_by_year.loc[atl_ocR_df_month_means_by_year['rate_dsd_mmph'] > 10, 'rate_dsd_mmph'] = np.nan
atl_ocR_df_month_means_by_year.loc[atl_ocR_df_month_means_by_year['rate_gag_mmph'] > 10, 'rate_gag_mmph'] = np.nan

atl_ocR_monthly_means = atl_ocR_df_month_means_by_year.groupby('month')[['rate_dsd_mmph','rate_gag_mmph']].mean().reset_index()

# plot the monthly means
fig, ax = plt.subplots(figsize=(16, 6), dpi=100)

# Plot monthly rainfall time series
ax.plot(atl_buoy_df_mnth_means_by_year['date'], atl_buoy_df_mnth_means_by_year['rain_rate'], 
    marker='x', lw=lw, color='k', label='Buoy')
ax.plot(atl_ocR_df_month_means_by_year['date'], atl_ocR_df_month_means_by_year['rate_dsd_mmph'], 
    marker='o', lw=lw, color='orange', label='OceanRain_dsd')
ax.plot(atl_ocR_df_month_means_by_year['date'], atl_ocR_df_month_means_by_year['rate_gag_mmph'], 
    marker='o', lw=lw, color='blue', label='OceanRain_gag')

# X-axis: show Jan and Jun of each year, format as 'MMM' (month) above and 'YYYY' (year) below
months = atl_buoy_df_mnth_means_by_year['date'].dt.month
years = atl_buoy_df_mnth_means_by_year['date'].dt.year
mask_jan_jun = (months == 1) | (months == 6)
tick_times = atl_buoy_df_mnth_means_by_year['date'].values[mask_jan_jun]

# Prepare two lines of tick labels: month above, year below
month_labels_ = [pd.to_datetime(str(t)).strftime('%b') for t in tick_times]
year_labels = [pd.to_datetime(str(t)).strftime('%Y') for t in tick_times]
custom_ticks = [
    (2010, 10), (2012, 7), (2014, 1), (2015, 7), (2016, 1),
]
tick_labels = [f"{m}\n{y}" for m, y in zip(month_labels_, year_labels)]

ax.set_xticks(tick_times)
ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=16, fontweight='bold', linespacing=1.5)

# X-axis limits: 2010-Jan to 2024-Sep
start = np.datetime64('2010-11-01')
end = np.datetime64('2016-01-01')  # np.datetime64('2024-09')
ax.set_xlim(start, end)
ax.set_ylim(0, 10)

# Y-axis: reduce ticks and add minor ticks
ax.set_ylabel('Rainfall [mm/day]', fontsize=18, fontweight='bold')
ax.yaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between each major

# Make major and minor ticks more visible
ax.tick_params(axis='x', which='major', length=12, width=2)
ax.tick_params(axis='x', which='minor', length=6, width=1)
ax.tick_params(axis='y', which='major', length=12, width=2)
ax.tick_params(axis='y', which='minor', length=6, width=1)

# Title and labels
ax.set_xlabel('Month, Year', fontsize=18, fontweight='bold', labelpad=10)
# ax.set_title("Mean Rainfall", fontsize=20, fontweight='bold', pad=15)

# Grid
ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)
ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.5)

# Y-tick label style
for label in ax.get_yticklabels():
    label.set_fontsize(15)
    label.set_fontweight('bold')

# Add legend
ax.legend(fontsize=14, loc='upper right', frameon=False)

plt.tight_layout()
# - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - -
# plot the monthly mean as well

fig, ax = plt.subplots(figsize=(16, 6), dpi=100)

# Plot monthly rainfall time series
ax.plot(month_positions, atl_buoy_monthly_means['rain_rate'].values, 
    marker='x', lw=lw, color='k', label='Buoy')
ax.plot(month_positions, atl_ocR_monthly_means['rate_dsd_mmph'].values, 
    marker='o', lw=lw, color='orange', label='OceanRain_dsd')
ax.plot(month_positions, atl_ocR_monthly_means['rate_gag_mmph'].values, 
    marker='o', lw=lw, color='blue', label='OceanRain_gag')
ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels)
ax.legend(fontsize=18, frameon=False)

ax.set_ylabel('Rainfall [mm/day]', fontsize=18, fontweight='bold')
ax.set_xlabel('Month', fontsize=18, fontweight='bold', labelpad=10)
ax.set_title("ATLANTIC Mean Monthly Rainfall", fontsize=20, fontweight='bold', pad=15)
ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)
ax.yaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between each major
ax.tick_params(axis='x', which='major', length=12, width=2)
ax.tick_params(axis='x', which='minor', length=6, width=1)
ax.tick_params(axis='y', which='major', length=12, width=2)
ax.tick_params(axis='y', which='minor', length=6, width=1)
for label in ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontweight('bold')

plt.tight_layout()
# - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - -
# compute and plot pdfc and pdfv for ocRain and Buoy data
ind_ocR_pdfc_pdfv_dsd = compute_pdf_elements(atl_ocR_df_daily, 'rate_dsd_mmph', bin_values)
ind_ocR_pdfc_pdfv_gg = compute_pdf_elements(atl_ocR_df_daily, 'rate_gag_mmph', bin_values)

ind_buoy_pdfc_pdfv = compute_pdf_elements(atl_buoy_df, 'rain_rate', bin_values)

fig, axs = plt.subplots(1, 2, figsize=(16, 10), 
                        sharex=False, sharey=False, dpi=100)

# Set common x-axis ticks and labels
bin_positions = range(len(bin_labels))
# Add grid lines and customize ticks
for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

# Line Plot TNEP PDFc
axs[0].plot(bin_positions, ind_ocR_pdfc_pdfv_dsd['pdfc'], label='OceanRain_dsd', marker='o',markersize=8,lw=lw, c='orange')
axs[0].plot(bin_positions, ind_ocR_pdfc_pdfv_gg['pdfc'], label='OceanRain_rg', marker='o',markersize=8,lw=lw, c='b')
axs[0].plot(bin_positions, ind_buoy_pdfc_pdfv['pdfc'], label='Buoy', marker='x',markersize=8,lw=lw, c='k')
axs[0].set_title('ATLANTIC', fontsize=18, fontweight='bold')
axs[0].set_ylabel('PDFc (%)', fontsize=18, fontweight='bold')
axs[0].set_xticks(bin_positions)
axs[0].set_xticklabels(bin_labels)
axs[0].legend(fontsize=18, frameon=False)

# Line Plot TNEP PDFv
axs[1].plot(bin_positions, ind_ocR_pdfc_pdfv_dsd['pdfv'], label='OceanRain_dsd', marker='o', lw=lw, c='orange')
axs[1].plot(bin_positions, ind_ocR_pdfc_pdfv_gg['pdfv'], label='OceanRain_rg', marker='o', lw=lw, c='b')
axs[1].plot(bin_positions, ind_buoy_pdfc_pdfv['pdfv'], label='Buoy', marker='x', lw=lw, c='k')
axs[1].set_title('ATLANTIC', fontsize=18, fontweight='bold')
axs[1].set_ylabel('PDFv (%)', fontsize=18, fontweight='bold')
axs[1].set_xticks(bin_positions)
axs[1].set_xticklabels(bin_labels)
axs[1].legend(fontsize=18, frameon=False)
# Adjust layout
plt.tight_layout()
