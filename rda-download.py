#!/usr/bin/env python
""" 
Python script to download selected files from rda.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
#%%
import sys, os
from urllib.request import build_opener

opener = build_opener()

filelist = [
  'https://data-osdf.rda.ucar.edu/ncar-rda/d484000/Y93678',
  'https://data-osdf.rda.ucar.edu/ncar-rda/d484000/Y93679'
]

download_location = '/ra1/pubdat/Satellite_eval_over_Oceans/data/PACRAIN-atoll'

# Ensure the download location exists
os.makedirs(download_location, exist_ok=True)

for file in filelist:
    ofile = os.path.basename(file)
    # Add .gz extension to the file name
    if not ofile.endswith('.gz'):
        ofile += '.gz'
    destination = os.path.join(download_location, ofile)
    sys.stdout.write("downloading " + ofile + " to " + destination + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    with open(destination, "wb") as outfile:
        outfile.write(infile.read())
    sys.stdout.write("done\n")


#%%
# PACRAIN daily + monthly parser
# - Parses daily .dat files (station+YYYYMM followed by 31 daily tokens, each token may have flags appended)
# - Parses monthly summary file (e.g., Y93679) to extract station metadata (name, lat, lon)
# - Joins lat/lon/name onto daily records
# - Saves a tidy CSV
#
# Notes on flags seen in daily tokens (per PACRAIN/CPRDB docs and samples):
#   - Values are in tenths of mm; divide by 10 to get mm
#   - '-99' indicates missing; '-99X' is padding for non-existent days
#   - 'T' trace; 'A' accumulation included; 'S' span of accumulation; 'E' estimated; '?' questionable
#   - Keep flags in a column; expose booleans for common flags
#
# You can adapt DAILY_DIR and MONTHLY_FILE to your paths.

from pathlib import Path
import re
import calendar
import pandas as pd
import numpy as np

# ======= CONFIG =======
DAILY_DIR = Path("/ra1/pubdat/Satellite_eval_over_Oceans/data/PACRAIN-atoll/Y93678_files/daily")
MONTHLY_PATH = Path("/ra1/pubdat/Satellite_eval_over_Oceans/data/PACRAIN-atoll/Y93679_files/monthly")  # file OR directory
OUT_CSV = "/ra1/pubdat/Satellite_eval_over_Oceans/data/PACRAIN-atoll/pacrain_daily_joined.csv"

# How to treat trace ("T") values: None = leave as NaN; or set e.g. 0.05 to treat trace as 0.05 mm
TRACE_VALUE = None
# ======================

def parse_monthly_metadata(monthly_path: Path) -> pd.DataFrame:
    """
    Parse PACRAIN monthly summary content to extract station metadata (ID, name, lat, lon).
    Works with either:
      - a single text file (e.g., the monolithic Y93679 file), OR
      - a directory containing many monthly *.dat files (e.g., 1874_01.dat, ...)
    Returns DataFrame[station_id, station_name, lat, lon] with unique stations.
    """
    meta = {}
    # Regex tolerates an optional rainfall flag between rainfall and #obs
    # Example line:
    #   'US03987 KEALAKEKUA 4 74.8              1931   -15556   123.4 M      30'
    line_re = re.compile(
        r"\s*([A-Z]{2}\d{5})\s+(.{1,30}?)\s+(-?\d+)\s+(-?\d+)\s+([-\d\.]+)(?:\s+[A-Z])?\s+(\d+)"
    )

    def scan_file(fp: Path):
        try:
            with open(fp, "r", errors="replace") as f:
                for line in f:
                    m = line_re.match(line)
                    if m:
                        sid = m.group(1).strip()
                        name = m.group(2).strip()
                        lat_hund = int(m.group(3))
                        lon_hund = int(m.group(4))
                        lat = lat_hund / 100.0
                        lon = lon_hund / 100.0
                        # Preserve first occurrence
                        if sid not in meta:
                            meta[sid] = (name, lat, lon)
        except Exception:
            # Skip files that aren't text (or unexpected content)
            pass

    if monthly_path.is_dir():
        # Scan all .dat files in that directory tree
        for fp in sorted(monthly_path.rglob("*.dat")):
            scan_file(fp)
    else:
        scan_file(monthly_path)

    df_meta = pd.DataFrame(
        [{"station_id": k, "station_name": v[0], "lat": v[1], "lon": v[2]} for k, v in meta.items()]
    )
    return df_meta

def split_token(token: str):
    """
    Split a daily token like '1753A', '0T', '-99X', '13', '551A', '0', '15E', '188A', '-99S'.
    Returns (value_mm, flags, is_missing, is_trace).
    - Values are tenths of mm; divide by 10 to get mm.
    - Any -99 indicates missing (includes -99X, -99S, etc.)
    - 'T' indicates trace.
    """
    token = token.strip()
    m = re.match(r"^(-?\d+)([A-Z\?]*)$", token)
    if not m:
        return (np.nan, "", True, False)
    num_str, flags = m.groups()

    # Missing
    if num_str.startswith("-99"):
        return (np.nan, flags, True, False)

    # Trace
    is_trace = ("T" in flags)

    # Convert tenths-mm -> mm
    try:
        val_mm = int(num_str) / 10.0
    except Exception:
        val_mm = np.nan

    # Optional override for trace
    if is_trace and TRACE_VALUE is not None:
        val_mm = float(TRACE_VALUE)

    return (val_mm, flags, False, is_trace)

def parse_daily_file(path: Path) -> pd.DataFrame:
    """
    Parse a single PACRAIN daily .dat file into tidy rows.
    Columns: station_id, date, value_mm, flags, is_missing, is_trace,
             is_estimated, is_accumulated, is_span, is_questionable, source_prefix
    """
    rows = []
    header_re = re.compile(r"^([A-Z]{2}\d{5})(\d{4})(\d{2})\s+(.*)$")
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = header_re.match(line)
            if not m:
                continue
            sid, year_str, month_str, rest = m.groups()
            year = int(year_str)
            month = int(month_str)
            max_days = calendar.monthrange(year, month)[1]
            tokens = rest.split()
            for i, tok in enumerate(tokens, start=1):
                if i > 31:  # safety guard
                    break
                # Only keep valid days in the month
                if i > max_days:
                    continue
                value_mm, flags, is_missing, is_trace = split_token(tok)
                date = pd.Timestamp(year=year, month=month, day=i)
                rows.append({
                    "station_id": sid,
                    "date": date,
                    "value_mm": value_mm,
                    "flags": flags,
                    "is_missing": bool(is_missing),
                    "is_trace": bool(is_trace),
                    "is_estimated": ("E" in flags),
                    "is_accumulated": ("A" in flags),
                    "is_span": ("S" in flags),          # days covered by an accumulation
                    "is_questionable": ("?" in flags),
                    "source_prefix": sid[:2]
                })
    return pd.DataFrame(rows)

def parse_daily_folder(daily_dir: Path, pattern: str = "*.dat") -> pd.DataFrame:
    """
    Parse all daily .dat files under daily_dir and concatenate.
    """
    dfs = []
    for p in sorted(daily_dir.rglob(pattern)):
        # Defensive filter: PACRAIN daily station files start with a 2-letter prefix
        if p.suffix.lower() == ".dat" and p.name.upper().startswith(("FR","NZ","SP","US","TA")):
            dfs.append(parse_daily_file(p))
    if not dfs:
        return pd.DataFrame(columns=[
            "station_id","date","value_mm","flags","is_missing","is_trace",
            "is_estimated","is_accumulated","is_span","is_questionable","source_prefix"
        ])
    return pd.concat(dfs, ignore_index=True)

# ---------- RUN ----------
df_daily = parse_daily_folder(DAILY_DIR, pattern="*.dat")
df_meta  = parse_monthly_metadata(MONTHLY_PATH)

# Join metadata onto daily
df = df_daily.merge(df_meta, on="station_id", how="left")

# Final tidy order
cols = ["station_id","station_name","lat","lon","date","value_mm","flags",
        "is_missing","is_trace","is_estimated","is_accumulated","is_span","is_questionable","source_prefix"]
df = df[cols].sort_values(["station_id","date"]).reset_index(drop=True)

# Save
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("Wrote:", OUT_CSV)
print("Rows:", len(df))
print("Stations:", df['station_id'].nunique())
print("Date range:", df['date'].min(), "→", df['date'].max())


df_atoll_name = df[df['station_name'].str.contains("atoll", case=False, na=False)]

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Extract unique lat/lon points for atoll stations
unique_stations = df_atoll_name[['lat', 'lon']].drop_duplicates()

# Set up the map
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# Add features to the map
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Scatter plot of station locations
scatter = ax.scatter(
  unique_stations['lon'], 
  unique_stations['lat'], 
  color='red', 
  s=50, 
  transform=ccrs.PlateCarree(),
  label='Atoll Stations'
)

# Set map extent to 20N to -20S latitude range
ax.set_extent([-180, 180, -20, 20], crs=ccrs.PlateCarree())

# Add title and legend
plt.title("Global Map of Atoll Stations", fontsize=16)
# plt.legend(loc='upper right')

# Save the map
plt.savefig("/ra1/pubdat/Satellite_eval_over_Oceans/data/PACRAIN-atoll/atoll_stations_map.png", dpi=300)
plt.show()

#%%
import pandas as pd, geopandas as gpd
from shapely.geometry import Point

# 1) unique stations from your parsed df
stn = (df[['station_id','station_name','lat','lon']]
       .dropna(subset=['lat','lon'])
       .drop_duplicates())
# normalize longitudes to [-180,180]
stn['lon'] = ((stn['lon'] + 180) % 360) - 180
gstn = gpd.GeoDataFrame(stn, geometry=gpd.points_from_xy(stn['lon'], stn['lat']), crs="EPSG:4326")

# 2) read atoll polygons (download the shapefile/geojson first)
atolls = gpd.read_file("Atolls_of_the_World.shp").to_crs("EPSG:4326")

# optional: restrict to tropical Pacific domain in PACRAIN
atolls = atolls.cx[-180:180, -47:22]

# small buffer so points on rims count as “within”
atolls_buf = atolls.copy()
atolls_buf['geometry'] = atolls_buf.buffer(0.02)  # ~2 km at equator

# 3) spatial join to tag atoll membership
tagged = gpd.sjoin(gstn, atolls_buf[['geometry']], how='left', predicate='within')
tagged['is_atoll'] = tagged['index_right'].notna()
stn_atoll = tagged.loc[tagged['is_atoll'], ['station_id']]

# 4) bring the tag back to your daily dataframe
df = df.merge(stn_atoll.assign(is_atoll=True), on='station_id', how='left')
df['is_atoll'] = df['is_atoll'].fillna(False)

# pacific atolls subset, daily:
df_atoll_daily = df[df['is_atoll']].copy()


#%%
from pydap.client import open_url
import pandas as pd, numpy as np

BASE = "https://apdrc.soest.hawaii.edu/dapper/insitu/rainfall_time_pac.cdp"
ds = open_url(BASE)

loc = ds["location"]  # this is a Sequence

# --- 1) Build station metadata from the location sequence ---
# Read the simple columns first
lon = np.array(loc["lon"][:], dtype=float)
lat = np.array(loc["lat"][:], dtype=float)

# The station metadata live under the nested attributes Structure, which is
# repeated once per location row. Pull each field as a column vector:
SITE_ID      = np.array(loc["attributes"]["SITE_ID"][:], dtype=object)
STATION_NAME = np.array(loc["attributes"]["STATION_NAME"][:], dtype=object)
TERRAIN      = np.array(loc["attributes"]["TERRAIN"][:], dtype=object)

meta = pd.DataFrame({
    "station_id": SITE_ID,
    "station_name": STATION_NAME,
    "terrain": TERRAIN,
    "lon": lon,
    "lat": lat,
})
# Clean/standardize
meta["terrain"] = meta["terrain"].astype(str).str.upper()
meta = meta.dropna(subset=["lon","lat"]).reset_index(drop=True)

# --- 2) Filter to ATOLL in the tropical Pacific domain ---
is_atoll  = meta["terrain"].eq("ATOLL")
in_domain = meta["lon"].between(120, 270) & meta["lat"].between(-45, 28)
atoll_meta = meta[is_atoll & in_domain].reset_index(drop=True)
print(f"ATOLL stations in domain: {len(atoll_meta)}")

# Optional: a helper to convert epoch ms to timezone-aware UTC
def ms_to_utc(ms):
    return pd.to_datetime(ms, unit="ms", utc=True)

# --- 3) Fetch daily time series for *each* station by indexing location rows ---
# We can locate the row index by matching lon/lat+id; to be robust, precompute a map.
# (If station IDs are unique, you can also match on station_id.)
lon_list = lon.tolist()
lat_list = lat.tolist()

def fetch_station_daily(station_row_idx, t1="2000-06-01", t2="2017-05-31"):
    """
    Pull one station's daily records by indexing the 'location' sequence at a row,
    then reading its nested 'time_series' sequence.
    """
    # Slice the location Sequence to a single row, then drill to time_series
    ts = ds["location"][station_row_idx]["time_series"]
    rain  = np.array(ts["Rainfall"][:], dtype=float)
    flags = np.array(ts["flags"][:], dtype=float)
    tz    = np.array(ts["tz"][:], dtype=float)
    time  = np.array(ts["time"][:], dtype=np.float64)

    df = pd.DataFrame({
        "rain_mm": rain,
        "flags": flags,
        "tz_hr": tz,
        "time": ms_to_utc(time),
    })
    # Restrict window
    t1 = pd.Timestamp(t1, tz="UTC")
    t2 = pd.Timestamp(t2, tz="UTC")
    df = df[(df["time"] >= t1) & (df["time"] <= t2)].copy()

    # Attach station metadata
    sid   = SITE_ID[station_row_idx]
    sname = STATION_NAME[station_row_idx]
    slon  = lon[station_row_idx]
    slat  = lat[station_row_idx]

    df.insert(0, "station_id", sid)
    df.insert(1, "station_name", sname)
    df.insert(2, "lat", slat)
    df.insert(3, "lon", slon)
    return df

# Example: pull the first 3 atoll stations (adjust the date window as needed)
rows = []
for idx in atoll_meta.index[:3]:
    # map back to the original row index in 'loc'
    # (here, atoll_meta index already corresponds to loc row since we created it directly from loc arrays)
    rows.append(fetch_station_daily(idx, "2000-06-01", "2017-05-31"))

atoll_daily = pd.concat(rows, ignore_index=True)
print(atoll_daily.head())

#%%
import pandas as pd

url = "https://apdrc.soest.hawaii.edu/erddap/tabledap/pacrain_dly.csv"
params = {
    "time>=": "1970-01-01T00:00:00Z",
    "time<=": "2017-05-31T00:00:00Z"
}

df = pd.read_csv(
    url,
    parse_dates=["time"]
)

print(df.head())
