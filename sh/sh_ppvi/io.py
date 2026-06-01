"""sh_ppvi.io — ERA5 data loading and output helpers.

Mirrors the input/output conventions of the Wu/Davis Fortran programs.
Grid convention inside all sh_ppvi modules:
  * latitude  : ascending (south→north), 0°S .. 90°N for NH
  * longitude : ascending 0 → 360° (or 0 → 358.5°)
  * level axis: axis 0, surface (k=0, 1000 hPa) → top (k=NW-1, 200 hPa)
  * all arrays float64

ERA5 input file (global):
  dims: (number | valid_time, pressure_level, latitude, longitude)
  vars: t [K], u [m/s], v [m/s], z [m² s⁻²]
  latitudes: descending (90°N → 90°S) in file; reversed here to ascending.
  pressure_level: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
                  → reordered here to surface-first [1000..200].
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional

from .coords import PLEVS_PA, NW

__all__ = ["load_global_era5", "prep_event", "write_step_output"]

# Expected ERA5 pressure levels in hPa (any order — will be sorted)
_ERA5_PLEVS_HPA = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200],
                             dtype=float)


def load_global_era5(path: str | Path) -> xr.Dataset:
    """Load the global ERA5 single-event NetCDF file.

    Returns an xr.Dataset with dims (pressure_level, latitude, longitude),
    pressure levels ordered surface-first (1000 → 200 hPa) and latitude
    ascending (south → north).  The 'number' / 'valid_time' dimensions are
    squeezed out if singleton.

    Variables returned: t, u, v, z   (all float64).
    """
    ds = xr.open_dataset(path, engine="netcdf4")

    # --- squeeze ensemble/time dims if singleton ---
    for dim in ("number", "valid_time"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0})

    # --- latitude: ensure ascending (south→north) ---
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # --- pressure_level: ensure surface-first (1000 → 200 hPa) ---
    plev = ds["pressure_level"].values
    if plev[0] < plev[-1]:
        # currently top-first; reverse
        ds = ds.isel(pressure_level=slice(None, None, -1))

    # --- select only the 10 levels we use ---
    target_hpa = _ERA5_PLEVS_HPA
    plev_vals = ds["pressure_level"].values
    idx = [int(np.argmin(np.abs(plev_vals - p))) for p in target_hpa]
    ds = ds.isel(pressure_level=idx)

    # --- cast to float64 ---
    for var in ("t", "u", "v", "z"):
        if var in ds:
            ds[var] = ds[var].astype(np.float64)

    return ds


def prep_event(
    ds: xr.Dataset,
    hemisphere: str = "NH",
    lat_min: float = 0.0,
    lat_max: float = 90.0,
) -> Dict[str, np.ndarray]:
    """Extract a dict of NH numpy arrays from a global ERA5 Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        As returned by load_global_era5 (lat ascending, level surface-first).
    hemisphere : str
        Only "NH" is supported (latitudes > lat_min). Kept as parameter
        for future extension.
    lat_min, lat_max : float
        Latitude bounds for the NH subdomain (default 0–90°N).

    Returns
    -------
    dict with keys:
        u, v, T   : (NW, nlat, nlon) float64  [m/s, m/s, K]
        Phi       : (NW, nlat, nlon) float64  geopotential [m² s⁻²]
        lat       : (nlat,)   float64  [degrees N]
        lon       : (nlon,)   float64  [degrees E]
        plev_hpa  : (NW,)     float64  pressure levels [hPa]
    """
    if hemisphere != "NH":
        raise NotImplementedError("Only hemisphere='NH' is implemented.")

    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    plev_hpa = ds["pressure_level"].values.astype(np.float64)

    # Slice latitude to NH
    mask = (lat >= lat_min) & (lat <= lat_max)
    lat_nh = lat[mask]

    u   = ds["u"].values[:, mask, :].astype(np.float64)   # (NW, nlat, nlon)
    v   = ds["v"].values[:, mask, :].astype(np.float64)
    T   = ds["t"].values[:, mask, :].astype(np.float64)
    Phi = ds["z"].values[:, mask, :].astype(np.float64)   # geopotential m²/s²

    return {
        "u": u,
        "v": v,
        "T": T,
        "Phi": Phi,
        "lat": lat_nh,
        "lon": lon,
        "plev_hpa": plev_hpa,
    }


def write_step_output(
    path: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    plev_hpa: Optional[np.ndarray] = None,
    **arrays: np.ndarray,
) -> xr.Dataset:
    """Write a dict of (NW, nlat, nlon) arrays to NetCDF.

    Parameters
    ----------
    path : str or Path
        Output file path.
    lat, lon : 1-D arrays
        Latitude (ascending) and longitude [degrees].
    plev_hpa : 1-D array, optional
        Pressure levels [hPa]. Defaults to PLEVS_PA / 100.
    **arrays
        Named (NW, nlat, nlon) or (nlat, nlon) arrays to write.

    Returns
    -------
    xr.Dataset written to *path*.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if plev_hpa is None:
        plev_hpa = PLEVS_PA / 100.0

    coords = {
        "pressure_level": ("pressure_level", plev_hpa,
                            {"units": "hPa", "long_name": "pressure level"}),
        "latitude":  ("latitude",  lat,
                      {"units": "degrees_north"}),
        "longitude": ("longitude", lon,
                      {"units": "degrees_east"}),
    }

    data_vars = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            data_vars[name] = (["pressure_level", "latitude", "longitude"], arr)
        elif arr.ndim == 2:
            data_vars[name] = (["latitude", "longitude"], arr)
        else:
            data_vars[name] = (["pressure_level"], arr)

    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_netcdf(path)
    return ds
