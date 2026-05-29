"""
01_download_era5.py
===================
Download ERA5 on 1.5°×1.5° grid for the 2025 Jan CA blocking event
on the full Northern-Hemisphere domain (for piecewise PV inversion).

Climatology (Jan 8 00Z mean, 1990-2020) is taken from the existing
hourly-clim NetCDFs in /net/flood/data2/users/x_yan/era/clim/ and
prepared separately by 02_prep_clim.py.

Run with: micromamba run -n fourcastnetv2 python 01_download_era5.py

Outputs
-------
../data/era5/era5_event_2025-01-08_00Z.nc  – event snapshot (2025-01-08 00Z)
                                              NH domain, 8 pressure levels
"""

import cdsapi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── NH domain (full hemisphere, snapped to 1.5° ERA5 grid) ──────────────────
# CDS convention: area = [N, W, S, E]
AREA = [90, 0, 0, 358.5]
GRID = ["1.5", "1.5"]

OUT_ERA5 = os.path.join(os.path.dirname(__file__), "..", "data", "era5")
os.makedirs(OUT_ERA5, exist_ok=True)

# ── 8 pressure levels (drop 925 and 600 hPa from old 10-level set) ───────────
LEVELS = ["1000", "850", "700", "500", "400", "300", "250", "200"]

# ── variables ────────────────────────────────────────────────────────────────
VARS = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}

EVENT_FILE = os.path.join(OUT_ERA5, "era5_event_2025-01-08_00Z.nc")

# ── download helper ──────────────────────────────────────────────────────────

def _download_var(var_short, var_long, outpath):
    """Download a single variable for the event time and return the path."""
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [var_long],
            "pressure_level": LEVELS,
            "year": "2025",
            "month": "01",
            "day": "08",
            "time": "00:00",
            "area": AREA,
            "grid": GRID,
            "format": "netcdf",
        },
        outpath,
    )
    return outpath


# ── 1. Event snapshot: 2025-01-08 00Z ───────────────────────────────────────
if os.path.exists(EVENT_FILE):
    print(f"Event file already exists: {EVENT_FILE}")
else:
    import xarray as xr
    import tempfile, shutil

    tmp_dir = "/net/flood/data2/users/x_yan/tmp/era5_dl_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_files = {k: os.path.join(tmp_dir, f"event_{k}.nc") for k in VARS}

    # parallel CDS requests — one per variable (respects CDS rate limit)
    print("Downloading ERA5 event snapshot 2025-01-08 00Z (NH, 8 levels) …")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_download_var, k, v, tmp_files[k]): k
                for k, v in VARS.items()
                if not os.path.exists(tmp_files[k])}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                fut.result()
                print(f"  downloaded: {k}")
            except Exception as e:
                print(f"  ERROR downloading {k}: {e}")
                raise

    # merge all variables into one NetCDF
    ds_list = [xr.open_dataset(tmp_files[k]) for k in VARS]
    ds_merged = xr.merge(ds_list)
    # Ensure latitude is ascending (S→N) for pvtend.sh_ops
    if "latitude" in ds_merged.dims:
        lat = ds_merged["latitude"].values
        if lat[0] > lat[-1]:
            ds_merged = ds_merged.isel(latitude=slice(None, None, -1))
    ds_merged.to_netcdf(EVENT_FILE)
    print(f"  → merged & saved: {EVENT_FILE}")
    print(f"  grid: {dict(ds_merged.dims)}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("\nDownload step complete.")
print(f"Event file: {EVENT_FILE}")
