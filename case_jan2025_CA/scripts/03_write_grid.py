"""
03_write_grid.py
================
Convert ERA5 NetCDF files to Wu-code ASCII *.grid files.

The Wu pvpialln_94UV.f expects one *.grid file per analysis time.
Format (fixed-width, Fortran unformatted):
  - Header (8 reals): lat_S lon_W lat_N lon_E dlat dlon NX NY
  - NX * NW block of geopotential Z (m) laid out level-by-level, E→W fastest
  - NX * NW block of potential temperature θ (K)
  - NX * NW block of u-wind (m/s)
  - NX * NW block of v-wind (m/s)

Each block: NW levels, each level has NY rows × NX columns.
Values written as (5F12.4) per row (Fortran F77 convention used by the test data).

This script writes:
  - 20250108_00.grid   → event snapshot (perturbation + basic state)
  - meanstate.grid     → climatological basic state only
    (pvpialln_94UV reads the mean-state .grid files separately — but
     the mean-state is embedded in the event grid file with the same format)

Run with: micromamba run -n fourcastnetv2 python 03_write_grid.py
"""

import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
ROOT        = SCRIPTS_DIR.parent
ERA5_DIR    = ROOT / "data" / "era5"
CLIM_DIR    = ROOT / "data" / "clim"
WU_DIR_BASE = ROOT / "wu_inv"
INV3D_EXE   = Path("/net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025")

# ── domain / grid params ─────────────────────────────────────────────────────
LAT_S =  10.5
LON_W = -169.5
LAT_N =  85.5
LON_E = -40.5
DLAT  =  1.5
DLON  =  1.5
NX    =  87
NY    =  51
NW    =  10
TARGET_LEVS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]   # hPa

# Gravitational acceleration (geopotential → geopotential height)
G = 9.80665   # m/s²

# Poisson exponent for potential temperature: θ = T * (1000/p)^(R/Cp)
KAPPA = 0.2854   # R/Cp for dry air


def theta_from_T_p(T_K, p_hPa):
    """Compute potential temperature from temperature T (K) and pressure p (hPa)."""
    return T_K * (1000.0 / p_hPa) ** KAPPA


def write_grid_file(outpath, Z_m, theta_K, u_ms, v_ms,
                    lat_s=LAT_S, lon_w=LON_W, lat_n=LAT_N, lon_e=LON_E,
                    dlat=DLAT, dlon=DLON, nx=NX, ny=NY):
    """
    Write one Wu-format *.grid ASCII file.

    Parameters
    ----------
    outpath : str or Path
    Z_m     : ndarray shape (NW, NY, NX) – geopotential height in metres
    theta_K : ndarray shape (NW, NY, NX) – potential temperature in K
    u_ms    : ndarray shape (NW, NY, NX) – u-wind in m/s
    v_ms    : ndarray shape (NW, NY, NX) – v-wind in m/s
    """
    with open(outpath, "w") as f:
        # Header: lat_S lon_W lat_N lon_E dlat dlon NX NY
        hdr = f"{lat_s:8.2f}{lon_w:8.2f}{lat_n:8.2f}{lon_e:8.2f}" \
              f"{dlat:8.2f}{dlon:8.2f}{float(nx):8.1f}{float(ny):8.1f}\n"
        f.write(hdr)

        # Write each variable: NW levels, NY rows, NX cols; 10 values per line
        # Wu pvpialln_94UV.f reads with FORMAT(10F8.1)
        for var_name, arr in [("Z", Z_m), ("theta", theta_K), ("u", u_ms), ("v", v_ms)]:
            for k in range(arr.shape[0]):        # level index
                for j in range(arr.shape[1]):    # lat index
                    row = arr[k, j, :]           # NX values
                    # Write in groups of 10 per line  (10F8.1)
                    for i_start in range(0, len(row), 10):
                        chunk = row[i_start:i_start + 10]
                        line  = "".join(f"{v:8.1f}" for v in chunk)
                        f.write(line + "\n")

    print(f"  Written: {outpath}")


def load_clim():
    """
    Load the 10-level climatological mean state prepared by 02_prep_clim.py.
    Returns dict var_short -> ndarray(NW, NY, NX)
    """
    clim = {}
    for var_short in ["z", "t", "u", "v"]:
        nc = CLIM_DIR / f"clim_jan08_00z_{var_short}.nc"
        if not nc.exists():
            print(f"ERROR: clim file missing: {nc}")
            print("Run 02_prep_clim.py first.")
            sys.exit(1)
        ds = xr.open_dataset(nc)
        # The DataArray should have dims (pressure_level, lat, lon) in S→N order
        # from the preprocessing step.
        vname = [v for v in ds.data_vars][0]
        arr = ds[vname].values
        # Ensure shape is (NW=10, NY=51, NX=87)
        assert arr.shape == (NW, NY, NX), \
            f"Unexpected clim shape {arr.shape} for {nc}"
        clim[var_short] = arr
    return clim


def load_event():
    """
    Load the 2025-01-08 00Z ERA5 event snapshot downloaded by 01_download_era5.py.
    Returns dict var_short -> ndarray(NW, NY, NX)
    """
    nc = ERA5_DIR / "era5_event_20250108_1p5deg.nc"
    if not nc.exists():
        print(f"ERROR: event file missing: {nc}")
        print("Run 01_download_era5.py first.")
        sys.exit(1)
    ds = xr.open_dataset(nc)
    print(f"  Event dataset dims: {dict(ds.dims)}")
    print(f"  Event variables: {list(ds.data_vars)}")

    # Squeeze time dimension (should be single time step)
    ds = ds.squeeze()

    lat_name = [d for d in ds.dims if "lat" in d.lower()][0]
    lon_name = [d for d in ds.dims if "lon" in d.lower()][0]
    lev_name = [d for d in ds.dims if "lev" in d.lower() or "press" in d.lower()
                or "level" in d.lower()][0]

    # Reorder: lat S→N, correct pressure level order (surface→top)
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    lev_vals = ds[lev_name].values.astype(float)
    # Reorder levels to match TARGET_LEVS (surface → top)
    lev_order = [np.argmin(np.abs(lev_vals - lv)) for lv in TARGET_LEVS]
    ds = ds.isel({lev_name: lev_order})

    # Map CDS variable names to short names
    # CDS names: 'z' or 'geopotential'; 't' or 'temperature'; 'u'/'v'
    name_map = {}
    for vname in ds.data_vars:
        vl = vname.lower()
        if vl in ("z", "geopotential"):
            name_map["z"] = vname
        elif vl in ("t", "temperature"):
            name_map["t"] = vname
        elif vl == "u":
            name_map["u"] = vname
        elif vl == "v":
            name_map["v"] = vname

    event = {}
    for short, cds_name in name_map.items():
        arr = ds[cds_name].values.astype(float)
        assert arr.shape == (NW, NY, NX), \
            f"Unexpected event shape {arr.shape} for {short}"
        event[short] = arr
    return event


# ── main ──────────────────────────────────────────────────────────────────────
print("Loading climatological mean state …")
clim = load_clim()

# Convert clim to grid units: Z in metres (geopotential / g), θ from T
clim_Z_m     = clim["z"] / G
clim_theta_K = np.stack([
    theta_from_T_p(clim["t"][k], TARGET_LEVS[k])
    for k in range(NW)
])
clim_u = clim["u"]
clim_v = clim["v"]

print("\nLoading ERA5 event snapshot …")
event = load_event()

event_Z_m     = event["z"] / G
event_theta_K = np.stack([
    theta_from_T_p(event["t"][k], TARGET_LEVS[k])
    for k in range(NW)
])
event_u = event["u"]
event_v = event["v"]

# ── write mean-state .grid file ───────────────────────────────────────────────
# pvpialln_94UV reads one mean-state .grid file per half-day.
# We write the climatological state as the "mean" (basic) state file.
mean_outdir = WU_DIR_BASE
mean_outdir.mkdir(parents=True, exist_ok=True)

print("\nWriting mean-state .grid file …")
write_grid_file(
    mean_outdir / "mean_jan08_00z.grid",
    clim_Z_m, clim_theta_K, clim_u, clim_v
)

# ── write event .grid file ────────────────────────────────────────────────────
print("\nWriting event (total state) .grid file …")
write_grid_file(
    mean_outdir / "20250108_00.grid",
    event_Z_m, event_theta_K, event_u, event_v
)

print("\n03_write_grid.py complete.")
print("Grid files written to:", mean_outdir)
print("Next: run 04_run_wu.sh")
