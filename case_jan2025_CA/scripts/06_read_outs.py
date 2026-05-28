#!/usr/bin/env python
"""
06_read_outs.py  — PPVI advection post-processing.

Computes the true 250-hPa PV tendency and per-piece PV advections from the
PPVI inversion output, using **pole-stable** spherical-harmonic operators
from pvtend.sh_ops (gradient_sh, vortdiv_sh).  Runs under the `blocking`
micromamba env (which has pyspharm + pvtend on PYTHONPATH).

Inputs
------
data/ppvi/method_{linear,nonlinear}.nc  – PPVI streamfunction / induced winds
data/era5/era5_event_2025-01-08_00Z.nc – ERA5 event (NH, 8 levels)
data/clim/clim_2025-01-08_00Z_NH.nc    – ERA5 climatology (NH, 8 levels)

Outputs
-------
data/ppvi/method_{linear,nonlinear}_advection.nc
  with q_pert_250, true_pv_tend, PVadv[3], U_250, V_250, U_ind[3], V_ind[3],
  mismatch_mask
"""

from __future__ import annotations
import os
import sys
import numpy as np
import xarray as xr

# Make pvtend importable in the blocking env (it isn't pip-installed here)
sys.path.insert(0, "/net/flood/data2/users/x_yan/pvtend/src")
import pvtend.sh_ops as sh   # noqa: E402

CASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPVI_DIR  = os.path.join(CASE_DIR, "data", "ppvi")
ERA5_NC   = os.path.join(CASE_DIR, "data", "era5",  "era5_event_2025-01-08_00Z.nc")
CLIM_NC   = os.path.join(CASE_DIR, "data", "clim",  "clim_2025-01-08_00Z_NH.nc")

G       = 9.80665
OMEGA   = 7.2921e-5
KAPPA   = 0.2854
R_EARTH = 6.371e6

PLEVS_HPA = np.array([1000., 850., 700., 500., 400., 300., 250., 200.])
PLEVS_PA  = PLEVS_HPA * 100.0
IDX_250   = int(np.argmin(np.abs(PLEVS_HPA - 250.0)))
NPIECES   = 3


# ─────────────────────────────────────────────────────────────────────────────
def _ensure_asc_lat(ds):
    lat_dim = next(d for d in ds.dims if "lat" in d.lower())
    if ds[lat_dim].values[0] > ds[lat_dim].values[-1]:
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    return ds


def _sort_plev(ds):
    lev_dim = next(d for d in ds.dims if "pressure" in d.lower()
                   or "level" in d.lower() or "lev" in d.lower())
    lv = ds[lev_dim].values.astype(float)
    if lv[0] > lv[-1]:
        ds = ds.isel({lev_dim: slice(None, None, -1)})
    return ds


def _load_era5():
    ev = xr.open_dataset(ERA5_NC).squeeze()
    cl = xr.open_dataset(CLIM_NC).squeeze()
    ev = _sort_plev(_ensure_asc_lat(ev))
    cl = _sort_plev(_ensure_asc_lat(cl))
    return ev, cl


def _theta(T_arr):
    return T_arr * (1e5 / PLEVS_PA[:, None, None]) ** KAPPA


def _ertel_pv_sh(T_arr, u_arr, v_arr, lat, lon):
    """Ertel PV using SH-based vorticity (pole-stable).

    Q = -g (f + ζ) ∂θ/∂p in PVU.
    """
    theta = _theta(T_arr)
    dtheta_dp = np.gradient(theta, PLEVS_PA, axis=0)
    f2d = (2.0 * OMEGA * np.sin(np.radians(lat)))[:, None]
    q = np.empty_like(T_arr)
    for k in range(T_arr.shape[0]):
        zeta, _ = sh.vortdiv_sh(u_arr[k], v_arr[k], lat, lon, R_earth=R_EARTH)
        q[k] = -G * (f2d + zeta) * dtheta_dp[k] * 1.0e6   # PVU
    return q


def _grad_sh(field2d, lat, lon):
    """SH gradient of a 2-D field; returns (df/dx, df/dy)."""
    return sh.gradient_sh(field2d, lat, lon, R_earth=R_EARTH)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("06_read_outs.py — PPVI advection (SH gradients, pole-stable)")
    print("=" * 60)

    ev, cl = _load_era5()
    lat_dim = next(d for d in ev.dims if "lat" in d.lower())
    lon_dim = next(d for d in ev.dims if "lon" in d.lower())
    lat = ev[lat_dim].values.astype(float)
    lon = ev[lon_dim].values.astype(float)
    print(f"  lat: {lat[0]:.1f}→{lat[-1]:.1f}N  lon: {lon[0]:.1f}→{lon[-1]:.1f}E")

    ev_t = ev["t"].values.astype(float)
    ev_u = ev["u"].values.astype(float)
    ev_v = ev["v"].values.astype(float)
    cl_t = cl["t"].values.astype(float)
    cl_u = cl["u"].values.astype(float)
    cl_v = cl["v"].values.astype(float)

    print("Computing Ertel PV via SH-vorticity (event & clim) …")
    q_event = _ertel_pv_sh(ev_t, ev_u, ev_v, lat, lon)
    q_clim  = _ertel_pv_sh(cl_t, cl_u, cl_v, lat, lon)
    q_pert  = q_event - q_clim
    q_pert_250 = q_pert[IDX_250]
    print(f"  Q' at 250 hPa: {q_pert_250.min():.2f} … {q_pert_250.max():.2f} PVU")

    u250_era5  = ev_u[IDX_250]
    v250_era5  = ev_v[IDX_250]

    print("Computing ∂Q'/∂x, ∂Q'/∂y via gradient_sh …")
    dqdx, dqdy = _grad_sh(q_pert_250, lat, lon)
    true_pv_tend = -(u250_era5 * dqdx + v250_era5 * dqdy) * 86400.0
    ref_scale    = 0.01 * float(np.nanpercentile(np.abs(true_pv_tend), 99))
    print(f"  true PV tend: {true_pv_tend.min():.2f} … {true_pv_tend.max():.2f} PVU/day")

    for method in ("linear", "nonlinear"):
        ppvi_nc = os.path.join(PPVI_DIR, f"method_{method}.nc")
        out_nc  = os.path.join(PPVI_DIR, f"method_{method}_advection.nc")
        if not os.path.exists(ppvi_nc):
            print(f"[SKIP] {ppvi_nc} missing"); continue

        print(f"\nProcessing method={method} …")
        ppvi = xr.open_dataset(ppvi_nc)
        PVadv = np.zeros((NPIECES, lat.size, lon.size))
        u_ind = np.zeros((NPIECES, lat.size, lon.size))
        v_ind = np.zeros((NPIECES, lat.size, lon.size))
        for i in range(NPIECES):
            u_pi = ppvi[f"u_p{i}"].values.astype(float)
            v_pi = ppvi[f"v_p{i}"].values.astype(float)
            u_ind[i] = u_pi;  v_ind[i] = v_pi
            PVadv[i] = -(u_pi * dqdx + v_pi * dqdy) * 86400.0
            print(f"  piece {i}: |U_ind| max={np.abs(u_pi).max():.1f} m/s, "
                  f"PVadv range {PVadv[i].min():.2f}…{PVadv[i].max():.2f} PVU/day")

        pv_sum   = PVadv.sum(axis=0)
        mismatch = np.abs(pv_sum - true_pv_tend) > ref_scale

        coords = {
            "piece":     (["piece"],     np.arange(NPIECES)),
            "latitude":  (["latitude"],  lat,  {"units": "degrees_north"}),
            "longitude": (["longitude"], lon,  {"units": "degrees_east"}),
        }
        ds_out = xr.Dataset({
            "q_pert_250":    (["latitude","longitude"], q_pert_250,
                              {"units": "PVU"}),
            "true_pv_tend":  (["latitude","longitude"], true_pv_tend,
                              {"units": "PVU/day"}),
            "PVadv":         (["piece","latitude","longitude"], PVadv,
                              {"units": "PVU/day"}),
            "U_250":         (["latitude","longitude"], u250_era5,
                              {"units": "m/s"}),
            "V_250":         (["latitude","longitude"], v250_era5,
                              {"units": "m/s"}),
            "U_ind":         (["piece","latitude","longitude"], u_ind,
                              {"units": "m/s"}),
            "V_ind":         (["piece","latitude","longitude"], v_ind,
                              {"units": "m/s"}),
            "mismatch_mask": (["latitude","longitude"], mismatch.astype("i1"),
                              {"units": "flag"}),
        }, coords=coords)
        ds_out.attrs["method"] = method
        ds_out.attrs["gradient_backend"] = "pvtend.sh_ops (pyspharm)"
        ds_out.to_netcdf(out_nc)
        print(f"  Saved → {out_nc}")

    print("\n06_read_outs.py complete.")
