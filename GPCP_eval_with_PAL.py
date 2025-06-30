#%% IMPORT LIBRARIES
import importlib
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    importlib.reload(sys.modules['util_functions'])

from util_functions import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

#%% DEBUG: Check current region bounds and test overlap function
print("CURRENT REGION BOUNDS:")
print("="*50)
for region, bounds in region_bounds.items():
    print(f"{region}: {bounds}")

print("\nTEST OVERLAP FUNCTION:")
print("-" * 30)
# Test PAL 19412: Lat 1.04-3.09, Lon 164.90-174.91 vs TNWP
pal_lat_min, pal_lat_max = 1.04, 3.09
pal_lon_min, pal_lon_max = 164.90, 174.91

tnwp_bounds = region_bounds["TNWP"]
print(f"PAL 19412: Lat {pal_lat_min}-{pal_lat_max}, Lon {pal_lon_min}-{pal_lon_max}")
print(f"TNWP: {tnwp_bounds}")

overlap_result = simple_box_check(pal_lat_min, pal_lat_max, pal_lon_min, pal_lon_max,
                                 tnwp_bounds["lat_min"], tnwp_bounds["lat_max"],
                                 tnwp_bounds["lon_min"], tnwp_bounds["lon_max"])
print(f"Should overlap with TNWP: {overlap_result}")

# Test PAL 6874 vs TNEP  
pal2_lat_min, pal2_lat_max = 1.50, 4.59
pal2_lon_min, pal2_lon_max = -165.53, -140.20

tnep_bounds = region_bounds["TNEP"]
print(f"\nPAL 6874: Lat {pal2_lat_min}-{pal2_lat_max}, Lon {pal2_lon_min}-{pal2_lon_max}")
print(f"TNEP: {tnep_bounds}")

overlap_result2 = simple_box_check(pal2_lat_min, pal2_lat_max, pal2_lon_min, pal2_lon_max,
                                  tnep_bounds["lat_min"], tnep_bounds["lat_max"],
                                  tnep_bounds["lon_min"], tnep_bounds["lon_max"])
print(f"Should overlap with TNEP: {overlap_result2}")
print("="*50)

#%% DEFINE PATH TO DATA
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'


#%% DEFINE GLOBAL VARIABLES
all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]

#%% CLASSIFY AND GROUP PAL FILES
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

#%% PLOT
# === Plot ===
fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for region, files in pals_classed_by_region.items():
    if region == "Unclassified" or len(files) == 0:
        continue  # Skip unclassified and empty regions for plotting bounds
    
    color = region_colors[region]
    for file in files:
        # Load only lat and lon efficiently, downsample by slicing
        ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
        lat = ds['lat'].values[::10]  # every 10th point
        lon = ds['lon'].values[::10]
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)
        ds.close()

    bounds = region_bounds[region]
    # Comment out dashed rectangular boundaries for now - difficult to achieve programmatically
    # rect = Rectangle(
    #     (bounds["lon_min"], bounds["lat_min"]),
    #     bounds["lon_max"] - bounds["lon_min"],
    #     bounds["lat_max"] - bounds["lat_min"],
    #     linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--',
    #     transform=ccrs.PlateCarree()
    # )
    # ax.add_patch(rect)

    # Remove region name and count labels from the plot
    # label_lon = bounds["lon_min"] + 2
    # label_lat = bounds["lat_max"] - 5
    # ax.text(label_lon, label_lat, f"{region}: {len(files)}", fontsize=15, color=color, transform=ccrs.PlateCarree())

# Remove unclassified files plotting since there are none
# if "Unclassified" in pals_classed_by_region and len(pals_classed_by_region["Unclassified"]) > 0:
#     color = region_colors["Unclassified"]
#     for file in pals_classed_by_region["Unclassified"]:
#         ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
#         lat = ds['lat'].values[::10]
#         lon = ds['lon'].values[::10]
#         ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)
#         ds.close()
#     
#     # Add text annotation for unclassified
#     ax.text(-170, -25, f"Unclassified: {len(pals_classed_by_region['Unclassified'])}", 
#             fontsize=15, color=color, transform=ccrs.PlateCarree())

# Create legend with PAL counts per region at the bottom, single row
legend_regions = [r for r in region_colors.keys() if r != "Unclassified"]
handles = [plt.Line2D([0], [0], color=region_colors[r], lw=2) for r in legend_regions]
# Add PAL counts to legend labels
labels = [f"{region} ({len(pals_classed_by_region.get(region, []))})" for region in legend_regions]
plt.legend(handles, labels, title="Regions", loc="lower center", bbox_to_anchor=(0.5, -0.1), 
          fontsize=14, title_fontsize=16, ncol=6)

ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(range(-30, 61, 15), crs=ccrs.PlateCarree())  # Changed to 15 degree intervals
ax.tick_params(labelsize=12)
ax.set_title("PAL Trajectories by Ocean Region", fontsize=20)

plt.tight_layout()
plt.show()
gc.collect()  # Clean up memory

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