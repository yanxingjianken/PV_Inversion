"""
02_prep_clim.py
===============
Prepare the climatological basic state for piecewise PV inversion.

Source : existing 9-level Jan 1990–2020 hourly-clim NetCDFs in era/clim/
Target : NH domain (0–90°N, 0–358.5°E), 8 pressure levels that are native
         to the ERA5 hourly-clim files (1000, 850, 700, 500, 400, 300, 250,
         200 hPa).  No log-p interpolation needed.

Steps
-----
1. Load Jan hourly clim (z, t, u, v) from era/clim/
2. Select day=8, hour=0  (Jan-8 00Z climatological equivalent)
3. Subset to NH domain  0–90°N  (lat ascending S→N, lon 0–358.5°E)
4. Select 8 native pressure levels
5. Average across 31 years (1990–2020) using dask for parallelism
6. Merge all variables and save data/clim/clim_2025-01-08_00Z_NH.nc

Run with: micromamba run -n fourcastnetv2 python 02_prep_clim.py

Output file variables: z (m²/s²), t (K), u (m/s), v (m/s)
Output dimensions: pressure_level (8), latitude (61 ascending), longitude (240)
"""

import os
import numpy as np
import xarray as xr

# ── paths ────────────────────────────────────────────────────────────────────
CLIM_DIR = "/net/flood/data2/users/x_yan/era/clim"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "clim")
os.makedirs(OUT_DIR, exist_ok=True)

# ── NH domain (lat ascending 0→90°N, lon 0→358.5°E) ────────────────────────
LAT_S, LAT_N = 0.0, 90.0     # NY=61 at 1.5° spacing
LON_W, LON_E = 0.0, 358.5    # NX=240 at 1.5° spacing

# ── 8 native pressure levels (925 and 600 dropped) ──────────────────────────
TARGET_LEVS = [1000, 850, 700, 500, 400, 300, 250, 200]

# ── date/time selection ──────────────────────────────────────────────────────
SELECT_DAY  = 8    # Jan-8
SELECT_HOUR = 0    # 00Z

# ── variable short names ─────────────────────────────────────────────────────
VARS = ["z", "t", "u", "v"]

OUT_FILE = os.path.join(OUT_DIR, "clim_2025-01-08_00Z_NH.nc")


def _probe_dims(ds):
    """Return (lat_name, lon_name, lev_name, year_name) from dataset."""
    dims = list(ds.dims)
    lat  = next(d for d in dims if "lat"  in d.lower())
    lon  = next(d for d in dims if "lon"  in d.lower())
    lev  = next(d for d in dims if "lev" in d.lower()
                                 or "press" in d.lower()
                                 or "level" in d.lower())
    yr   = next((d for d in dims if "year" in d.lower()), None)
    return lat, lon, lev, yr


def _sel_day_hour(ds, lat_name, lon_name, lev_name):
    """Select day=8 hour=0, squeeze time-like dims, return DataArray."""
    data_var = next(v for v in ds.data_vars if "time" not in v.lower())
    da = ds[data_var]
    for cand in ["hour", "time", "valid_time"]:
        if cand in da.dims:
            da = da.sel({cand: SELECT_HOUR})
            break
    for cand in ["day", "dayofyear"]:
        if cand in da.dims:
            da = da.sel({cand: SELECT_DAY})
            break
    if "month" in da.dims:
        da = da.sel(month=1)
    return da.squeeze()


def process_var(var_short):
    clim_file = os.path.join(CLIM_DIR, f"era5_hourly_clim_1990-2020_jan_{var_short}.nc")
    print(f"  Loading {clim_file} …")
    ds = xr.open_dataset(clim_file)
    lat_name, lon_name, lev_name, yr_name = _probe_dims(ds)

    # select day=8, hour=0 from climatological file
    da = _sel_day_hour(ds, lat_name, lon_name, lev_name)

    # ── restrict to 8 native levels ──────────────────────────────────────────
    levs = da[lev_name].values.astype(float)
    keep = np.array([lv for lv in TARGET_LEVS if any(abs(levs - lv) < 0.5)])
    if len(keep) < len(TARGET_LEVS):
        missing = set(TARGET_LEVS) - set(int(k) for k in keep)
        raise ValueError(f"{var_short}: missing levels {missing} in clim file. "
                         f"Available: {sorted(levs)}")
    da = da.sel({lev_name: [float(k) for k in keep]})

    # ── roll longitudes from [-180, 180) → [0, 360) if needed ───────────────
    lon_vals = da[lon_name].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
        order = np.argsort(new_lon)
        da = da.isel({lon_name: order})
        da = da.assign_coords({lon_name: new_lon[order]})

    # ── subset NH domain ─────────────────────────────────────────────────────
    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values
    lat_mask = (lat_vals >= LAT_S - 0.01) & (lat_vals <= LAT_N + 0.01)
    lon_mask = (lon_vals >= LON_W - 0.01) & (lon_vals <= LON_E + 0.01)
    da = da.isel({lat_name: np.where(lat_mask)[0],
                  lon_name: np.where(lon_mask)[0]})

    # ── ensure lat ascending (S→N) ───────────────────────────────────────────
    lv = da[lat_name].values
    if lv[0] > lv[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    # ── year-mean if year dim exists (parallel via dask) ─────────────────────
    if yr_name and yr_name in da.dims:
        da = da.mean(dim=yr_name)

    return da.rename({
        lat_name: "latitude",
        lon_name: "longitude",
        lev_name: "pressure_level",
    }).rename(var_short)


if __name__ == "__main__":
    if os.path.exists(OUT_FILE):
        print(f"Clim file already exists: {OUT_FILE}")
    else:
        das_dict = {}
        for v in VARS:
            da = process_var(v)
            print(f"  Computing year-mean for {v} …")
            das_dict[v] = da.compute() if hasattr(da.data, "compute") else da

        ds_out = xr.Dataset(das_dict)
        ds_out = ds_out.assign_coords(pressure_level=TARGET_LEVS)
        ds_out.attrs["description"] = (
            "ERA5 Jan 1990-2020 climatological mean at Jan-8 00Z, "
            "NH domain (0-90N, 0-358.5E), 8 native pressure levels."
        )
        ds_out.to_netcdf(OUT_FILE)

        print(f"  → saved {OUT_FILE}")
        print(f"  dims: {dict(ds_out.dims)}")
        print(f"  lat: {ds_out.latitude.values[0]:.1f} → {ds_out.latitude.values[-1]:.1f}")
        print(f"  lev: {list(ds_out.pressure_level.values)}")

    print("\n02_prep_clim.py complete.")
