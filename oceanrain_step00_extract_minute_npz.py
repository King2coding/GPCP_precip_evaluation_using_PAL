import os
from datetime import date
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ----------------------------- HELPERS ----------------------------------------
cde_run_dte = str(date.today().strftime('%Y%m%d'))
def _safe_var(ds, *names, required=True, fill_to_nan=True, as_float=None):
    """
    Safe variable reader that won't crash on int32 masked arrays.

    as_float:
      - None  -> auto: ints stay ints unless we need NaN, floats stay floats
      - True  -> always return float64 (allows NaNs)
      - False -> keep native dtype where possible
    """
    for nm in names:
        if nm in ds.variables:
            v = ds.variables[nm][:]

            # Decide casting behavior
            if as_float is None:
                # auto: if int and we might need NaN -> cast to float
                want_float = np.issubdtype(getattr(v, "dtype", np.asarray(v).dtype), np.integer)
            else:
                want_float = bool(as_float)

            # Handle masked arrays safely
            if np.ma.isMaskedArray(v):
                if want_float:
                    v = v.astype("float64")
                    v = np.ma.filled(v, np.nan)
                else:
                    # keep int, fill with sentinel
                    v = np.ma.filled(v, -999999)

            v = np.asarray(v)

            # Optional: convert common negative fill values to NaN (only if float)
            if fill_to_nan and np.issubdtype(v.dtype, np.floating):
                v = np.where(v <= -90, np.nan, v)  # conservative rule for OceanRAIN

            return v

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


# ----------------------------- USER PATHS & SETTINGS --------------------------
ocn_dir = '/ra1/pubdat/OceanRain/nc'#r'C:\E-Disk\Research_Scientist\Research\1-SnowDepth\Arctic\OceanRain\nc\temp'
ocn_head, ocn_mid, ocn_tail = 'OceanRAIN-W_', '_v1', '.nc'
save_dir = '/ra1/pubdat/Satellite_eval_over_Oceans/data/OceanRain'#r'C:\E-Disk\Research_Scientist\Research\1-SnowDepth\Arctic\OceanRain\output\temp'
os.makedirs(save_dir, exist_ok=True)

# ----------------------------- 1) READ OCEANRAIN -------------------------------
ocn_files = sorted([f for f in os.listdir(ocn_dir) if f.startswith(ocn_head) and f.endswith(ocn_tail)])
if not ocn_files:
    raise RuntimeError(f"No OceanRAIN files found in {ocn_dir}")

ocn_rows = []

for i, fname in enumerate(ocn_files):
    fpath = os.path.join(ocn_dir, fname)
    print(f"[OceanRAIN] Reading {i+1}/{len(ocn_files)}: {fpath}")

    try:
        with Dataset(fpath) as ds:

            # --- core geo/time ---
            lat = _safe_var(ds, 'latitude')
            lon = _safe_var(ds, 'longitude')
            lon = _wrap_lon_180(lon)

            date_ut = _safe_var(ds, "date_UT", "date_UTC", "date", as_float=False, fill_to_nan=False)
            time_ut = _safe_var(ds, "time_UT", "time_UTC", "time", required=False, as_float=False, fill_to_nan=False)

            dt = _parse_oceanrain_timestamp(date_ut, time_ut)

            # --- phase + QC flags ---
            precip_flag  = _safe_var(ds, "precip_flag", required=False, as_float=True)
            precip_flag2 = _safe_var(ds, "precip_flag2", required=False, as_float=True)

            rain_prob  = _safe_var(ds, 'probability_for_rain', required=False)
            snow_prob  = _safe_var(ds, 'probability_for_snow', required=False)
            mixed_prob = _safe_var(ds, 'probability_for_mixed_phase', required=False)

            # --- precipitation rates ---
            # Primary: phase-specific disdrometer "theoretical" rates
            rate_rain_dsd = _safe_var(ds, 'theoretical_rain_rate_disdrometer', required=True)
            rate_snow_dsd = _safe_var(ds, 'theoretical_snow_rate_disdrometer', required=True)

            # Diagnostic: rain gauge + generic ODM470 rate
            rate_gag = _safe_var(ds, 'rain_gauge_precipitation_rate', required=False)
            rate_odm = _safe_var(ds, 'ODM470_precipitation_rate_R', required=False)

            # --- wind (context / optional QC) ---
            true_wind_speed = _safe_var(ds, 'true_wind_speed', required=False)
            true_wind_dir   = _safe_var(ds, 'true_wind_direction', required=False)
            u10             = _safe_var(ds, 'wind_speed_in_10m_height', required=False)
            rel_wind_speed  = _safe_var(ds, 'relative_wind_speed', required=False)

            # --- build a safe mask (only require lat/lon/time) ---
            mask = np.isfinite(lat) & np.isfinite(lon) & pd.Series(dt).notna().values

            # apply mask to required arrays
            lat = lat[mask].astype('float32')
            lon = lon[mask].astype('float32')
            dtm = pd.to_datetime(dt[mask])

            # apply mask to optional arrays safely
            def _m(v):
                if v is None:
                    return np.full(lat.shape, np.nan, dtype='float32')
                return np.asarray(v)[mask].astype('float32')

            df_file = pd.DataFrame({
                "time_utc": dtm,
                "lat": lat,
                "lon": lon,

                # primary reference rates
                "rate_rain_dsd_mmph": _m(rate_rain_dsd),
                "rate_snow_dsd_mmph": _m(rate_snow_dsd),

                # diagnostics
                "rate_gag_mmph": _m(rate_gag),
                "rate_odm_mmph": _m(rate_odm),

                # flags + phase probs
                "precip_flag": _m(precip_flag),
                "precip_flag2": _m(precip_flag2),
                "rain_prob": _m(rain_prob),
                "snow_prob": _m(snow_prob),
                "mixed_prob": _m(mixed_prob),

                # wind context
                "true_wind_speed": _m(true_wind_speed),
                "true_wind_dir": _m(true_wind_dir),
                "u10": _m(u10),
                "rel_wind_speed": _m(rel_wind_speed),
            })

            ship = fname.split(ocn_head)[1].split(ocn_mid)[0] if ocn_mid in fname else "unknown"
            df_file["ship"] = ship  # repeated string column

            ocn_rows.append(df_file)

    except Exception as e:
        print(f"   ERROR reading {fname}: {e}")

if not ocn_rows:
    raise RuntimeError("No OceanRAIN data could be read after QC.")

ocn_df = pd.concat(ocn_rows, ignore_index=True).sort_values("time_utc")

# ---------------- 2) SAVE NPZ --------------------------
npz_out = os.path.join(save_dir, f"OceanRAIN_MINUTE_coordinates_and_data_Kingsley_{cde_run_dte}.npz")

np.savez(
    npz_out,
    time_utc=ocn_df["time_utc"].values.astype("datetime64[ns]"),
    lat=ocn_df["lat"].values.astype("float32"),
    lon=ocn_df["lon"].values.astype("float32"),

    rate_rain_dsd_mmph=ocn_df["rate_rain_dsd_mmph"].values.astype("float32"),
    rate_snow_dsd_mmph=ocn_df["rate_snow_dsd_mmph"].values.astype("float32"),
    rate_gag_mmph=ocn_df["rate_gag_mmph"].values.astype("float32"),
    rate_odm_mmph=ocn_df["rate_odm_mmph"].values.astype("float32"),

    precip_flag=ocn_df["precip_flag"].values.astype("float32"),
    precip_flag2=ocn_df["precip_flag2"].values.astype("float32"),
    rain_prob=ocn_df["rain_prob"].values.astype("float32"),
    snow_prob=ocn_df["snow_prob"].values.astype("float32"),
    mixed_prob=ocn_df["mixed_prob"].values.astype("float32"),

    true_wind_speed=ocn_df["true_wind_speed"].values.astype("float32"),
    true_wind_dir=ocn_df["true_wind_dir"].values.astype("float32"),
    u10=ocn_df["u10"].values.astype("float32"),
    rel_wind_speed=ocn_df["rel_wind_speed"].values.astype("float32"),

    ship=ocn_df["ship"].values.astype("U")
)

print(f"[Save] OceanRAIN minute data saved to NPZ → {npz_out}")

# Reload to confirm
data = np.load(npz_out, allow_pickle=True)
ocn_df2 = pd.DataFrame({k: data[k] for k in data.files})
print(ocn_df2.head())