#%%
# Import packages
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

import xarray as xr

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

pal_dat = xr.open_dataset(all_pal_files[0])

#%%
import gc
import numpy as np
import pandas as pd
from datetime import datetime

import os
import matplotlib.pyplot as plt

import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation


import re
from matplotlib.colors import ListedColormap, BoundaryNorm
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_to_seviri_data = r'/home/kkumah/miscellaneous/misc_dat/seviri'
path_to_clm_data = r'/home/kkumah/miscellaneous/misc_dat/clm'
path_to_plt = r'/home/kkumah/miscellaneous/misc_dat/plt'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_bt(channel, radiance):
    # Constants
    c1 = 1.19104e-5  # W·m²·sr⁻¹·(cm⁻¹)⁴
    c2 = 1.43877     # K·cm

    # Central wavenumber lookup (cm⁻¹)
    wavenumbers = {
        4: 2569.094,
        5: 1598.566,
        6: 1362.142,
        9: 930.659,
        10: 839.661,
        11: 752.381,
    }

    ν = wavenumbers.get(channel)
    if ν is None:
        raise ValueError(f"BT computation not valid for channel {channel}")

    return (c2 * ν) / np.log(1 + (c1 * ν**3) / radiance)

def compute_reflectance(channel, radiance, solar_zenith_deg, earth_sun_dist_au=1.0):
    # Solar TOA irradiance (W/m²/µm)
    E_sun = {
        1: 682.82,
        2: 539.59,
        3: 253.10,
    }

    E = E_sun.get(channel)
    if E is None:
        raise ValueError(f"Reflectance computation not valid for channel {channel}")

    theta_rad = np.deg2rad(solar_zenith_deg)
    return (np.pi * radiance * earth_sun_dist_au**2) / (E * np.cos(theta_rad))


def parse_time_key(pass_time_key):
    dt = datetime.strptime(pass_time_key, "%Y%m%dT%H%M%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_all_clm_to_xr(all_clm_files):
    datasets = []
    times = []
    for f in all_clm_files:
        filename = os.path.basename(f)
        match = re.search(r'_(\d{8}T\d{6}Z)_', filename)
        if not match:
            continue
        time_str = match.group(1)
        # Remove 'T' and parse to datetime64 for xarray
        time_str_clean = time_str.replace('T', '')
        time_val = np.datetime64(datetime.strptime(time_str_clean, "%Y%m%d%H%M%SZ"))
        ds = xr.open_dataset(f)
        # Add time as a new coordinate
        ds = ds.expand_dims({'time': [time_val]})
        datasets.append(ds)
        times.append(time_val)
    # Concatenate along time dimension
    if datasets:
        combined = xr.concat(datasets, dim='time')
        return combined
    else:
        return None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

all_seviri_files = sorted([os.path.join(path_to_seviri_data, f) for f in os.listdir(path_to_seviri_data) if f.endswith('.nc')])
all_clm_files = sorted([os.path.join(path_to_clm_data, f) for f in os.listdir(path_to_clm_data) if f.endswith('.nc')])
# (Cloud Mask) CML contains binary pixel Value Meaning 
# 0 Clear sky over water
# 1 Clear sky over land
# 2 Cloud
# 3 No data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# process clm data 

clm_xr = read_all_clm_to_xr(all_clm_files)

# get cloudy scene/areas
clm_mask = xr.where(clm_xr['cloud_mask'] == 2, 1, 0)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# process seviri data
read_all_seviri_chan9 = []

for f in all_seviri_files:
    filename = os.path.basename(f)
    # Extract start time from filename
    match = re.search(r'HRSEVIRI_(\d{8}T\d{6}Z)_', filename)
    if not match:
        continue
    time_str = match.group(1)
    # Parse to datetime and round to nearest minute
    dt = pd.to_datetime(time_str, format="%Y%m%dT%H%M%SZ").round('min')
    # Open dataset and extract channel_9
    ds = xr.open_dataset(f)
    if "channel_9" not in ds:
        continue
    radiance = ds["channel_9"]
    bt = compute_bt(9, radiance)
    # Add time dimension
    bt = bt.expand_dims(time=[dt])
    read_all_seviri_chan9.append(bt)

if read_all_seviri_chan9:
     seviri_chan9_bt = xr.concat(read_all_seviri_chan9, dim="time")
else:
    seviri_chan9_bt = None

cloud_chan9 = seviri_chan9_bt.where(clm_mask == 1, drop=True)
# make some movies
gc.collect()  # Clean up memory before plotting

fig = plt.figure(figsize=(8, 6), dpi=1000)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

vmin = float(cloud_chan9.min())
vmax = float(cloud_chan9.max())

im = ax.imshow(
    cloud_chan9.isel(time=0),
    origin='lower',
    extent=[
        float(cloud_chan9.lon.min()), float(cloud_chan9.lon.max()),
        float(cloud_chan9.lat.min()), float(cloud_chan9.lat.max())
    ],
    transform=ccrs.PlateCarree(),
    cmap='jet',
    vmin=vmin,
    vmax=vmax
)

# Add latitude and longitude gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                  color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}
cb = plt.colorbar(im, ax=ax, orientation='vertical', label='BT (K)')
title = ax.set_title(f"SEVIRI 10.8 (µm) BT\n{str(cloud_chan9.time.values[0])[:16]} UTC")

def update(frame):
    im.set_data(cloud_chan9.isel(time=frame))
    title.set_text(f"SEVIRI 10.8 (µm) BT\n{str(cloud_chan9.time.values[frame])[:16]} UTC")
    return [im, title]

ani = animation.FuncAnimation(
    fig, update, frames=len(cloud_chan9.time), interval=400, blit=False
)

plt.close(fig)  # Prevents duplicate display in some environments

# Try to save as mp4 if ffmpeg is available, else fallback to gif
mp4_path = os.path.join(path_to_plt, "seviri_chan9_bt_movie.mp4")
gif_path = os.path.join(path_to_plt, "seviri_chan9_bt_movie.gif")

try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(mp4_path, writer=writer, dpi=1000)
    print(f"Animation saved as mp4: {mp4_path}")
except (KeyError, RuntimeError, ValueError) as e:
    print("ffmpeg not available or failed, saving as gif instead.")
    ani.save(gif_path, writer='pillow', dpi=1000)
    print(f"Animation saved as gif: {gif_path}")

gc.collect()  # Clean up memory after plotting

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# make a plot of cloud mask data, cloudy areas and a IR BT scenes for 13:30

# Find the index for 13:30
target_time = np.datetime64('2025-06-24T13:30:00')
if target_time in clm_xr['time'].values:
    idx = np.where(clm_xr['time'].values == target_time)[0][0]
else:
    raise ValueError("13:30 time not found in clm_xr['time']")

# Prepare data for plotting
cloud_mask_1330 = clm_xr['cloud_mask'].isel(time=idx)
cloudy_mask_1330 = clm_mask.isel(time=idx)
bt_1330 = seviri_chan9_bt.isel(time=idx)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), 
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        dpi=1000)

# Cloud mask colormap and norm
cloud_mask_cmap = ListedColormap(['#1f77b4', '#2ca02c', '#d62728', '#7f7f7f'])
cloud_mask_bounds = [0, 1, 2, 3, 4]
cloud_mask_norm = BoundaryNorm(cloud_mask_bounds, cloud_mask_cmap.N)

# Cloud mask
im0 = axs[0].pcolormesh(clm_xr['lon'], clm_xr['lat'], cloud_mask_1330, cmap=cloud_mask_cmap, norm=cloud_mask_norm)
axs[0].set_title('Cloud Mask\n2025-06-24 13:30')
axs[0].coastlines()
axs[0].add_feature(cfeature.BORDERS, linewidth=0.5)
gl0 = axs[0].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl0.top_labels = False
gl0.right_labels = False
gl0.xlabel_style = {'size': 11}
gl0.ylabel_style = {'size': 11}

# Cloudy mask (binary, invert cmap: 1=cloud=black, 0=white)
cloudy_cmap = ListedColormap(['white', 'black'])
im1 = axs[1].pcolormesh(clm_xr['lon'], clm_xr['lat'], cloudy_mask_1330, cmap=cloudy_cmap, vmin=0, vmax=1)
axs[1].set_title('Cloudy Areas Mask (2=cloud)\n2025-06-24 13:30')
axs[1].coastlines()
axs[1].add_feature(cfeature.BORDERS, linewidth=0.5)
gl1 = axs[1].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': 11}
gl1.ylabel_style = {'size': 11}

# IR BT
im2 = axs[2].pcolormesh(seviri_chan9_bt['lon'], 
                        seviri_chan9_bt['lat'], 
                        bt_1330, cmap='jet')
axs[2].set_title('SEVIRI 10.8 µm IR BT\n2025-06-24 13:30')
axs[2].coastlines()
axs[2].add_feature(cfeature.BORDERS, linewidth=0.5)
gl2 = axs[2].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 11}
gl2.ylabel_style = {'size': 11}

for ax in axs:
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Colorbars below plots
cbar0 = plt.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.18, fraction=0.045, aspect=30)
cbar0.set_label('Mask Value', fontsize=11)
cbar0.set_ticks([0.5, 1.5, 2.5, 3.5])
cbar0.set_ticklabels([
    '0: Clear sky\n over water',
    '1: Clear sky\n over land',
    '2: Cloud',
    '3: No data'
])
cbar0.ax.tick_params(labelsize=11)

cbar1 = plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.12, fraction=0.045, aspect=30)
cbar1.set_label('Cloudy (1=cloud, black)', fontsize=11)
cbar1.set_ticks([0.25, 0.75])
cbar1.set_ticklabels(['0: No cloud', '1: Cloud'])
cbar1.ax.tick_params(labelsize=11)

cbar2 = plt.colorbar(im2, ax=axs[2], orientation='horizontal', pad=0.18, fraction=0.045, aspect=30)
cbar2.set_label('BT (K)', fontsize=11)
cbar2.ax.tick_params(labelsize=11)

plt.show()

gc.collect()  # Clean up memory after plotting






# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for f in all_seviri_files:
    filename = os.path.basename(f)
    match = re.search(r'HRSEVIRI_(\d{8}T\d{6}Z)_', filename)
    if not match:
        continue
    time_key = match.group(1)
    time_key = parse_time_key(time_key)

    ds = xr.open_dataset(f)

    time_data = {}

    for var_name in ds.data_vars:
        if not var_name.startswith("channel_"):
            continue
        chan_num = int(var_name.split("_")[1])
        # raw = ds[var_name]
        radiance = ds[var_name]

        # Apply offset + slope to get radiance
        # attr_name = f"ch{chan_num:02}_cal"
        # offset, slope = map(float, ds.attrs[attr_name].split())
        # radiance = offset + slope * raw

        # if chan_num in [1, 2, 3]:
        #     result = compute_reflectance(chan_num, radiance, solar_zenith_deg)
        if chan_num in [4, 5, 6, 9, 10, 11]:
            result = compute_bt(chan_num, radiance)
        else:
            continue

        time_data[var_name] = result

    output_dict[time_key] = time_data

# Mask areas where values are <= 253 for channel_9 at '2025-06-24T08:00:10Z'
msk_r = output_dict['2025-06-24T08:00:10Z']['channel_9'].where(
    output_dict['2025-06-24T08:00:10Z']['channel_9'] < 255
)
# Constants for SEVIRI IR 10.8 µm (example)
c1 = 1.191e-5
c2 = 1.439
wavenumber = 927.0  # cm⁻¹

# Convert to brightness temperature
micro108 = (c2 * wavenumber) / np.log(1 + ((c1 * wavenumber ** 3) / seviri_data['channel_9']))