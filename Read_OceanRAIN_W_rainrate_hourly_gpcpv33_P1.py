# -*- coding: utf-8 -*-
# Part 1: Data Processing & Saving
# Run this once to create the collocated data file.
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr

# ----------------------------- HELPERS ----------------------------------------
def _safe_var(ds, *names, required=True):
    for nm in names:
        if nm in ds.variables:
            v = ds[nm][:]
            if np.ma.isMaskedArray(v):
                v = np.ma.filled(v, np.nan)
            return np.asarray(v)
    if required:
        raise KeyError(f"None of variables {names} found.")
    return None

def _parse_oceanrain_timestamp(date_ut, time_ut):
    date_str = pd.Series(date_ut).astype(str).str.zfill(8)
    if time_ut is None:
        time_str = pd.Series(['1200'] * len(date_str))
        fmt = '%d%m%Y %H%M'
    else:
        raw = pd.Series(time_ut).astype(str)
        L = raw.str.len().max()
        if L <= 4:
            time_str = raw.str.zfill(4)
            fmt = '%d%m%Y %H%M'
        else:
            time_str = raw.str.zfill(6)
            fmt = '%d%m%Y %H%M%S'
    dt = pd.to_datetime(date_str + ' ' + time_str, format=fmt, errors='coerce', utc=True)
    return dt

def _wrap_lon_180(lon):
    lon = np.asarray(lon, dtype='float64')
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    lon_wrapped[lon_wrapped == 180.0] = -180.0
    return lon_wrapped

def _nearest_index(arr1d, values):
    a = np.asarray(arr1d)
    v = np.asarray(values)
    if a.ndim != 1: a = a.ravel()
    if v.ndim != 1: v = v.ravel()
    v_finite = np.where(np.isfinite(v), v, np.nan)
    asc = bool(a[0] <= a[-1])
    if not asc: a_work = a[::-1]
    else: a_work = a
    idx = np.searchsorted(a_work, v_finite)
    idx0 = np.clip(idx - 1, 0, a_work.size - 1)
    idx1 = np.clip(idx, 0, a_work.size - 1)
    choose_left = (np.abs(v_finite - a_work[idx0]) <= np.abs(v_finite - a_work[idx1]))
    out_rev = np.where(choose_left, idx0, idx1)
    if not asc: out = (a_work.size - 1) - out_rev
    else: out = out_rev
    if np.issubdtype(v.dtype, np.floating):
        nanmask = ~np.isfinite(v)
        if nanmask.any():
            out = out.astype('int64')
            out[nanmask] = 0
    return out

def coverage_stats(df):
    n = len(df)
    return pd.Series({
        'n_samples': n,
        'rate_dsd_mean_mmph': df['rate_dsd_mmph'].mean(skipna=True),
        'rate_gag_mean_mmph': df['rate_gag_mmph'].mean(skipna=True),
        # 'snow_prob_mean': df['snow_prob'].mean(skipna=True),
        # 't2m_k_mean': df['t2m_k'].mean(skipna=True),
    })

def _fix_lon(ds, lon_name):
    lon_vals = ds[lon_name].values
    lon_wrap = _wrap_lon_180(lon_vals)
    ds2 = ds.assign_coords({lon_name: (lon_name, lon_wrap)}).sortby(lon_name)
    return ds2

# ----------------------------- USER PATHS & SETTINGS --------------------------
ocn_dir = r'C:\E-Disk\Research_Scientist\Research\1-SnowDepth\Arctic\OceanRain\nc\temp'
ocn_head, ocn_mid, ocn_tail = 'OceanRAIN-W_', '_v1', '.nc'
gpcp_dir = r'C:\E-Disk\Research_Scientist\Research\2-Glacier\datasets\gpcpv33\daily'
gpcp_plp_glob = os.path.join(gpcp_dir, 'GPCPDAY_L3_20220703_V3.3.nc4')
save_dir = r'C:\E-Disk\Research_Scientist\Research\1-SnowDepth\Arctic\OceanRain\output\temp'
os.makedirs(save_dir, exist_ok=True)
FILL_THRESH = -99.0

# ----------------------------- 1) READ OCEANRAIN -------------------------------
ocn_files = sorted([f for f in os.listdir(ocn_dir) if f.startswith(ocn_head) and f.endswith(ocn_tail)])
if not ocn_files: raise RuntimeError(f"No OceanRAIN files found in {ocn_dir}")
ocn_rows = []
for i, fname in enumerate(ocn_files):
    fpath = os.path.join(ocn_dir, fname)
    print(f"[OceanRAIN] Reading {i+1}/{len(ocn_files)}: {fpath}")
    try:
        with Dataset(fpath) as ds:
            lat = _safe_var(ds, 'latitude')
            lon = _safe_var(ds, 'longitude')
            date_ut = _safe_var(ds, 'date_UT', 'date_UTC', 'date')
            time_ut = _safe_var(ds, 'time_UT', 'time_UTC', 'time', required=False)
            rate_dsd = _safe_var(ds, 'ODM470_precipitation_rate_R', 'odm470_precipitation_rate_R')
            rate_gag = _safe_var(ds, 'rain_gauge_precipitation_rate', 'rain_gauge_rate', required=False)
            precip_flag = _safe_var(ds, 'precip_flag', 'precip_flag', required=False)
            precip_flag2 = _safe_var(ds, 'precip_flag2', 'precip_flag2', required=False)
            mixed_prob = _safe_var(ds, 'probability_for_mixed_phase', 'mixed_probability', required=False)
            rain_prob = _safe_var(ds, 'probability_for_rain', 'rain_probability', required=False)
            snow_prob = _safe_var(ds, 'probability_for_snow', 'snow_probability', required=False)
            wind_speed = _safe_var(ds, 'true_wind_speed', 'wind_speed', required=False)

            # t_air_c = _safe_var(ds, 'air_temperature', 'air_temperature_C', required=False)
            # if snow_prob is not None: snow_prob[snow_prob <= FILL_THRESH] = np.nan
            # rate_dsd[rate_dsd <= FILL_THRESH] = np.nan
            # if rate_gag is not None: rate_gag[rate_gag <= FILL_THRESH] = np.nan
            lon = _wrap_lon_180(lon)
            mask = np.isfinite(lat) & np.isfinite(lon) #& np.isfinite(rate_dsd)
            # if rate_gag is not None: mask &= np.isfinite(rate_gag)
            lat, lon = lat[mask], lon[mask]
            rate_dsd, rate_gag = rate_dsd[mask], rate_gag[mask]
            precip_flag, precip_flag2 = precip_flag[mask], precip_flag2[mask]
            mixed_prob, rain_prob, snow_prob = mixed_prob[mask], rain_prob[mask], snow_prob[mask]
            # snow_prob = snow_prob[mask] if snow_prob is not None else None
            # t_air_k = (t_air_c[mask] + 273.15).astype('float32') if t_air_c is not None else np.full(lat.shape, np.nan, dtype='float32')
            dt = _parse_oceanrain_timestamp(date_ut[mask], time_ut[mask] if time_ut is not None else None)
            ship = fname.split(ocn_head)[1].split(ocn_mid)[0] if ocn_mid in fname else 'unknown'
            df_file = pd.DataFrame({
                'time_utc': pd.to_datetime(dt), 'lat': lat.astype('float32'), 'lon': lon.astype('float32'),
                'rate_dsd_mmph': rate_dsd.astype('float32'), 'rate_gag_mmph': rate_gag.astype('float32'),
                'precip_flag': precip_flag.astype('float32'), 'precip_flag2': precip_flag2.astype('float32'),
                'mixed_prob': mixed_prob.astype('float32'), 'rain_prob': rain_prob.astype('float32'),
                'snow_prob': snow_prob.astype('float32'), 'wind_speed': wind_speed.astype('float32'),
                # 't2m_k': t_air_k,
                'ship': str(ship),
                # 'file': fname,
            })
            df_file = df_file[df_file['time_utc'].notna()]
            ocn_rows.append(df_file)
    except Exception as e:
        print(f"   ERROR reading {fname}: {e}")

if not ocn_rows: raise RuntimeError("No OceanRAIN data could be read after QC.")
ocn_df = pd.concat(ocn_rows, ignore_index=True).sort_values('time_utc')
# ocn_df['hour_utc'] = ocn_df['time_utc'].dt.floor('h')

# # ---------------- 1.1) MINUTE OCEANRAIN DATA FOR KINGSLEY --------------------------
npz_out = os.path.join(save_dir, 'OceanRAIN_MINUTE_coordinates_and_data_Kingsley_new.npz')
np.savez(npz_out,
         time_utc=ocn_df['time_utc'].values.astype('datetime64'),
         lat=ocn_df['lat'].values.astype('float32'),
         lon=ocn_df['lon'].values.astype('float32'),
         rate_dsd_mmph=ocn_df['rate_dsd_mmph'].values.astype('float32'),
         rate_gag_mmph=ocn_df['rate_gag_mmph'].values.astype('float32'),
		 precip_flag = ocn_df['precip_flag'].values.astype('float32'),
		 precip_flag2 = ocn_df['precip_flag2'].values.astype('float32'),
		 mixed_prob = ocn_df['mixed_prob'].values.astype('float32'),
		 rain_prob = ocn_df['rain_prob'].values.astype('float32'),
		 snow_prob = ocn_df['snow_prob'].values.astype('float32'),
		 wind_speed = ocn_df['wind_speed'].values.astype('float32'),
         ship = ocn_df['ship'].values.astype('str')
         )
print(f"[Save] Final OCEANRAIN MINUT data saved to NPZ → {npz_out}")

data = np.load(npz_out, allow_pickle=True)
ocn_df = pd.DataFrame({key: data[key] for key in data.files})

# ---------------- 2) OPEN GPCPV33 & NORMALIZE LONGITUDE --------------------------
print("[GPCPV33] Opening tp/sf datasets...")
try:
    ds_plp = xr.open_mfdataset(gpcp_plp_glob, combine='by_coords')
except Exception:
    ds_plp = xr.open_mfdataset(gpcp_plp_glob, combine='by_coords')

var_plp = ds_plp['probability_liquid_phase']
time_dim = 'time' if 'time' in var_plp.dims else ('valid_time' if 'valid_time' in var_plp.dims else var_plp.dims[0])
lat_dim  = 'latitude' if 'latitude' in var_plp.dims else ('lat' if 'lat' in var_plp.dims else var_plp.dims[1])
lon_dim  = 'longitude' if 'longitude' in var_plp.dims else ('lon' if 'lon' in var_plp.dims else var_plp.dims[2])
ds_plp = _fix_lon(ds_plp, lon_dim)

try:
    ds_plp = ds_plp.reindex_like(ds_plp, method='nearest')
except Exception:
    ds_plp = ds_plp.transpose(*var_plp.dims).reindex({time_dim: ds_plp[time_dim], lat_dim: ds_plp[lat_dim], lon_dim: ds_plp[lon_dim]}, method='nearest')

gpcp_lats = ds_plp[lat_dim].values
gpcp_lons = ds_plp[lon_dim].values
gpcp_times = pd.to_datetime(ds_plp[time_dim].values, utc=True)

# ---------------- 3) BIN OCEANRAIN TO GPCPV33 GRID + HOURLY ----------------------
print("[Bin] Assigning 1-min samples to nearest GPCPV33 grid cell...")
ocn_df['lat_idx'] = _nearest_index(gpcp_lats, ocn_df['lat'].values)
ocn_df['lon_idx'] = _nearest_index(gpcp_lons, ocn_df['lon'].values)
print("[Bin] Aggregating to hourly cell means with coverage...")
keys = ['time_utc', 'lat_idx', 'lon_idx']
ocn_cell_hour = ocn_df.groupby(keys, sort=True).apply(coverage_stats).reset_index()
ocn_cell_hour['gpcp_lat'] = gpcp_lats[ocn_cell_hour['lat_idx'].values]
ocn_cell_hour['gpcp_lon'] = gpcp_lons[ocn_cell_hour['lon_idx'].values]
# ---------------- 5) FINAL MERGE & SAVE --------------------------------------
print("[Save] Merging and saving collocated data...")
colloc = ocn_cell_hour.copy()

# Save to NPZ
npz_out = os.path.join(save_dir, 'OceanRAIN_GPCPV33_coordinates_and_data_Kingsley.npz')
np.savez(npz_out,
         hour_utc=colloc['time_utc'].values.astype('datetime64[ns]'),
         gpcp_lat=colloc['gpcp_lat'].values.astype('float32'),
         gpcp_lon=colloc['gpcp_lon'].values.astype('float32'),
         n_samples=colloc['n_samples'].values.astype('int16'),
         rate_dsd_mean_mmph=colloc['rate_dsd_mean_mmph'].values.astype('float32'),
         rate_gag_mean_mmph=colloc['rate_gag_mean_mmph'].values.astype('float32'),
         
         snow_prob_mean=colloc['snow_prob_mean'].values.astype('float32'),
         # t2m_k_mean=colloc['t2m_k_mean'].values.astype('float32'),
         # era_tp_mm=colloc['era_tp_mm'].values.astype('float32'),
         # era_sf_mm=colloc['era_sf_mm'].values.astype('float32'),
         # era_sf_frac=colloc['era_sf_frac'].values.astype('float32')
         )
print(f"[Save] Final collocated data saved to NPZ → {npz_out}")

# Close original datasets
ds_plp.close()
