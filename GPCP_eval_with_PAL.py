#%% IMPORT LIBRARIES

from util_functions import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

#%% DEFINE PATH TO DATA
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'


#%% DEFINE GLOBAL VARIABLES
all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]

#%% CLASSIFY AND GROUP PAL FILES
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

# === Plot ===
fig = plt.figure(figsize=(18, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -30, 60], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for region, files in pals_classed_by_region.items():
    color = region_colors[region]
    for file in files:
        # Load only lat and lon efficiently, downsample by slicing
        ds = xr.open_dataset(file, drop_variables=[v for v in xr.open_dataset(file).data_vars if v not in ['lat', 'lon']])
        lat = ds['lat'].values[::10]  # every 10th point
        lon = ds['lon'].values[::10]
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)

    bounds = region_bounds[region]
    rect = Rectangle(
        (bounds["lon_min"], bounds["lat_min"]),
        bounds["lon_max"] - bounds["lon_min"],
        bounds["lat_max"] - bounds["lat_min"],
        linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--',
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(rect)

    label_lon = bounds["lon_min"] + 2
    label_lat = bounds["lat_max"] - 5
    ax.text(label_lon, label_lat, f"{region}: {len(files)}", fontsize=15, color=color, transform=ccrs.PlateCarree())

handles = [plt.Line2D([0], [0], color=region_colors[r], lw=2) for r in region_colors]
labels = list(region_colors.keys())
plt.legend(handles, labels, title="Regions", loc="lower left", fontsize=18, title_fontsize=18, ncol=2)

ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(range(-30, 61, 30), crs=ccrs.PlateCarree())
ax.tick_params(labelsize=12)
ax.set_title("PAL Trajectories by Ocean Region", fontsize=20)

plt.tight_layout()
plt.show()
gc.collect()  # Clean up memory