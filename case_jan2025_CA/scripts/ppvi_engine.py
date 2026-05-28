#!/usr/bin/env python
"""
ppvi_engine.py
==============
Piecewise Potential Vorticity Inversion (PPVI) engine for the 2025-01-08
Northern-Hemisphere blocking case.

Run with:
    micromamba run -n blocking python scripts/ppvi_engine.py

Environment requirement: the 'blocking' micromamba environment (has pyspharm
installed).  All imports of pvtend.sh_ops are via sys.path.insert so the
pvtend source tree is used read-only — nothing in the pvtend package is
modified.

Algorithm
---------
Two inversion methods are computed (saved as separate NetCDF files):

Method 1 – "linear" (per-level barotropic)
  At each pressure level k independently:
      ∇²ψ'_k = ζ'_k   where  ζ'_k = Q'_k * 1e-6 / (g * σ̄_k)
  σ̄_k = |∂θ̄/∂p|_k is the static stability from the basic state.
  Each 2D Poisson solve uses pvtend.sh_ops.invert_laplacian_sh (pyspharm
  back-end, pole-stable via parity mirroring).

Method 2 – "nonlinear" (3D QG Gauss-Seidel)
  Same per-level solve but augmented with the QG vertical stretching term:
      ∇²ψ'_k^(n+1) = Q'_k * 1e-6 / (g * σ̄_k)
                     − f₀² * d/dp(σ̄⁻¹ * dψ'/dp)|_k^(n)  /  g
  Outer Gauss-Seidel loop, max_iter=30, tol=1e-4, under-relaxation α=0.7
  if residual diverges.

Three PV pieces (vertical masking):
  piece 0 (lower):  Q' nonzero at 850, 700 hPa
  piece 1 (mid):    Q' nonzero at 500, 400 hPa
  piece 2 (upper):  Q' nonzero at 300, 250 hPa
  (BC levels 1000 and 200 hPa are excluded from all pieces)

Parallelization: ProcessPoolExecutor(max_workers=3) for the 3 pieces within
each method; methods are run sequentially.

Inputs
------
data/era5/era5_event_2025-01-08_00Z.nc  – ERA5 event (NH, 8 levels, lat asc)
data/clim/clim_2025-01-08_00Z_NH.nc    – ERA5 climatology (same grid)

Outputs
-------
data/ppvi/method_linear.nc     – linear inversion results
data/ppvi/method_nonlinear.nc  – 3D QG inversion results

Both output files contain per-piece (3 pieces × NL=8 levels):
  psi_p{i}   – streamfunction perturbation ψ' (m²/s)
  u_p{i}     – induced zonal wind u' at 250 hPa (m/s)
  v_p{i}     – induced meridional wind v' at 250 hPa (m/s)
Plus diagnostics:
  q_pert     – Ertel PV anomaly Q' on all levels (PVU)
  q_event    – Ertel PV event (PVU)
  q_clim     – Ertel PV climatology (PVU)
  theta_clim – basic state potential temperature θ̄ (K)
  sigma_clim – static stability σ̄ = |∂θ̄/∂p| (K/Pa)
"""

from __future__ import annotations
import os, sys, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

# ── pvtend import (read-only, no package modification) ───────────────────────
_PVTEND_SRC = "/net/flood/data2/users/x_yan/pvtend/src"
if _PVTEND_SRC not in sys.path:
    sys.path.insert(0, _PVTEND_SRC)
import pvtend.sh_ops as sh

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_NC  = os.path.join(_ROOT, "data", "era5",  "era5_event_2025-01-08_00Z.nc")
CLIM_NC   = os.path.join(_ROOT, "data", "clim",  "clim_2025-01-08_00Z_NH.nc")
PPVI_DIR  = os.path.join(_ROOT, "data", "ppvi")
os.makedirs(PPVI_DIR, exist_ok=True)

# ── physical constants ────────────────────────────────────────────────────────
G      = 9.80665       # m/s²
OMEGA  = 7.2921e-5     # s⁻¹
KAPPA  = 0.2854        # R/Cp (dry air)
F0     = 2 * OMEGA * np.sin(np.radians(45.0))   # reference Coriolis at 45°N

# ── grid / level constants ────────────────────────────────────────────────────
PLEVS_HPA = np.array([1000., 850., 700., 500., 400., 300., 250., 200.])
PLEVS_PA  = PLEVS_HPA * 100.0   # Pa
NL        = len(PLEVS_HPA)       # 8 total levels

# 250 hPa level index (0-based)
IDX_250 = int(np.argmin(np.abs(PLEVS_HPA - 250.0)))   # = 6

# PV pieces: indices into PLEVS_HPA
PIECES = {
    0: [1, 2],   # lower: 850, 700 hPa
    1: [3, 4],   # mid:   500, 400 hPa
    2: [5, 6],   # upper: 300, 250 hPa
}
PIECE_NAMES = ["lower_850_700", "mid_500_400", "upper_300_250"]

# ── R_EARTH ────────────────────────────────────────────────────────────────────
R_EARTH = 6.371e6   # m  (consistent with pvtend default)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """Load ERA5 event and clim NetCDFs; return (event, clim, lat, lon)."""
    ev  = xr.open_dataset(EVENT_NC)
    cl  = xr.open_dataset(CLIM_NC)

    # Standardise coordinate names
    for ds in (ev, cl):
        if "valid_time" in ds.dims:
            ds = ds.squeeze("valid_time")
        elif "time" in ds.dims:
            ds = ds.squeeze("time")

    # Identify lat/lon/lev dimension names robustly
    def _dim(ds, hints):
        dims = list(ds.dims)
        for h in hints:
            match = [d for d in dims if h in d.lower()]
            if match:
                return match[0]
        raise KeyError(f"Cannot find dim matching {hints} in {dims}")

    ev_lat = _dim(ev, ["lat"])
    ev_lon = _dim(ev, ["lon"])
    ev_lev = _dim(ev, ["pressure", "level", "lev"])

    cl_lat = _dim(cl, ["lat"])
    cl_lon = _dim(cl, ["lon"])
    cl_lev = _dim(cl, ["pressure", "level", "lev"])

    # Extract to numpy, squeezing any leftover time-like dims
    def _get(ds, var, lat_d, lev_d):
        da = ds[var].squeeze()
        # ensure pressure ascending → descending ordering matches PLEVS_HPA
        plev = da[lev_d].values.astype(float)
        if plev[0] > plev[-1]:
            da = da.isel({lev_d: slice(None, None, -1)})
        # lat: ensure ascending (S→N)
        la = da[lat_d].values
        if la[0] > la[-1]:
            da = da.isel({lat_d: slice(None, None, -1)})
        return da.values  # (NL, NY, NX)

    ev_data = {v: _get(ev, v, ev_lat, ev_lev) for v in ("z", "t", "u", "v")}
    cl_data = {v: _get(cl, v, cl_lat, cl_lev) for v in ("z", "t", "u", "v")}

    lat = np.sort(ev[ev_lat].values.astype(float))
    lon = np.sort(ev[ev_lon].values.astype(float))
    return ev_data, cl_data, lat, lon


def compute_theta(T_K: np.ndarray, plev_Pa: np.ndarray) -> np.ndarray:
    """Potential temperature θ = T * (1e5/p)^κ."""
    return T_K * (1e5 / plev_Pa[:, None, None]) ** KAPPA


def compute_static_stability(theta: np.ndarray) -> np.ndarray:
    """σ̄ = |∂θ̄/∂p| [K/Pa], positive in stable atmosphere.

    Uses centered differences in the interior and one-sided at boundaries.
    Clipped below at a minimum value to avoid division by zero.
    """
    sigma = np.abs(np.gradient(theta, PLEVS_PA, axis=0))
    sigma = np.clip(sigma, 1e-5, None)  # floor: ≈ extremely weakly stratified
    return sigma


def compute_ertel_pv(T: np.ndarray, u: np.ndarray, v: np.ndarray,
                     lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Ertel PV = −g*(f+ζ)*∂θ/∂p in PVU (10⁻⁶ m² K s⁻¹ kg⁻¹).

    Uses pvtend.sh_ops.vortdiv_sh for spectral-quality ζ on the sphere.
    Loop over levels (vortdiv_sh operates on 2D slices).
    """
    theta = compute_theta(T, PLEVS_PA)
    dtheta_dp = np.gradient(theta, PLEVS_PA, axis=0)   # K/Pa (negative in troposphere)
    f_2d = (2 * OMEGA * np.sin(np.radians(lat)))[None, :, None]  # (1,NY,1)

    q_ertel = np.zeros_like(T)
    for k in range(NL):
        zeta_k, _ = sh.vortdiv_sh(u[k], v[k], lat, lon, R_EARTH)
        abs_vort = f_2d[0] + zeta_k            # (NY, NX)
        q_ertel[k] = -G * abs_vort * dtheta_dp[k] * 1e6   # convert to PVU
    return q_ertel   # (NL, NY, NX) in PVU


# ─────────────────────────────────────────────────────────────────────────────
# Per-piece inversion workers (called in separate processes)
# ─────────────────────────────────────────────────────────────────────────────

def _invert_piece_linear(
    q_pert_masked: np.ndarray,
    sigma: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Method 1: independent per-level 2D Poisson solve.

    ∇²ψ'_k = Q'_k * 1e-6 / (g * σ̄_k)

    Returns psi (NL, NY, NX) in m²/s.
    """
    psi = np.zeros_like(q_pert_masked)
    for k in range(NL):
        rhs = q_pert_masked[k] * 1e-6 / (G * sigma[k])   # s⁻¹
        psi[k] = sh.invert_laplacian_sh(rhs, lat, lon, R_EARTH, parity="scalar")
    return psi


def _invert_piece_nonlinear(
    q_pert_masked: np.ndarray,
    sigma: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-4,
    alpha: float = 0.7,
) -> np.ndarray:
    """Method 2: 3D QG Gauss-Seidel iteration.

    ∇²ψ'_k^(n+1) = Q'_k*1e-6/(g*σ̄_k) − f₀²/g * d/dp(σ̄⁻¹ dψ'/dp)|_k^n

    Vertical stretching term uses centered FD at interior levels;
    zero-Neumann BC at top and bottom (perturbation confined to NH interior).
    Under-relaxation ω=alpha when residual increases.

    Returns psi (NL, NY, NX) in m²/s.
    """
    # Initialise with the barotropic solution (good starting point)
    psi = _invert_piece_linear(q_pert_masked, sigma, lat, lon)

    # Horizontal RHS (static): Q'_k * 1e-6 / (g * σ̄_k)
    rhs0 = np.zeros_like(q_pert_masked)
    for k in range(NL):
        rhs0[k] = q_pert_masked[k] * 1e-6 / (G * sigma[k])

    prev_res = np.inf
    for it in range(max_iter):
        psi_old = psi.copy()

        # vertical stretching term: Δ_vert[k] = f₀²/g * d/dp(σ̄⁻¹ dψ/dp)|_k
        # discretised with centered differences
        vert = np.zeros_like(psi)
        for k in range(NL):
            if k == 0:
                # bottom BC: dψ/dp ≈ (psi[1]-psi[0])/(Δp₊)
                dp_plus  = PLEVS_PA[1] - PLEVS_PA[0]
                sig_plus = 0.5 * (sigma[0] + sigma[1])
                dpsidp_plus  = (psi[1] - psi[0])   / dp_plus
                dp_bar   = 0.5 * (PLEVS_PA[1] - PLEVS_PA[0])
                vert[k]  = F0**2 / G * (dpsidp_plus / sig_plus) / dp_bar
            elif k == NL - 1:
                # top BC: dψ/dp ≈ (psi[NL-1]-psi[NL-2])/(Δp₋)
                dp_minus = PLEVS_PA[k] - PLEVS_PA[k-1]
                sig_minus = 0.5 * (sigma[k] + sigma[k-1])
                dpsidp_minus = (psi[k] - psi[k-1]) / dp_minus
                dp_bar   = 0.5 * (PLEVS_PA[k] - PLEVS_PA[k-1])
                vert[k]  = -F0**2 / G * (dpsidp_minus / sig_minus) / dp_bar
            else:
                dp_plus  = PLEVS_PA[k+1] - PLEVS_PA[k]
                dp_minus = PLEVS_PA[k]   - PLEVS_PA[k-1]
                dp_bar   = 0.5 * (PLEVS_PA[k+1] - PLEVS_PA[k-1])
                sig_plus  = 0.5 * (sigma[k+1] + sigma[k])
                sig_minus = 0.5 * (sigma[k]   + sigma[k-1])
                dpsidp_plus  = (psi[k+1] - psi[k])   / dp_plus
                dpsidp_minus = (psi[k]   - psi[k-1]) / dp_minus
                vert[k] = F0**2 / G * (
                    dpsidp_plus / sig_plus - dpsidp_minus / sig_minus
                ) / dp_bar

        # Solve ∇²ψ'_k = rhs0_k - vert_k
        psi_new = np.zeros_like(psi)
        for k in range(NL):
            psi_new[k] = sh.invert_laplacian_sh(
                rhs0[k] - vert[k], lat, lon, R_EARTH, parity="scalar"
            )

        # Under-relaxation
        psi = psi_old + alpha * (psi_new - psi_old)

        # Convergence check
        denom = np.nanmax(np.abs(psi))
        res   = np.nanmax(np.abs(psi - psi_old)) / max(denom, 1e-30)
        if res > prev_res * 1.5 and it > 2:
            alpha = max(alpha * 0.7, 0.1)   # slow down if diverging
        prev_res = res
        if res < tol:
            print(f"      3D QG converged in {it+1} iterations (res={res:.2e})")
            break
    else:
        print(f"      3D QG did NOT converge (final res={res:.2e})")

    return psi


def _invert_one_piece(args):
    """Worker function for ProcessPoolExecutor: invert one piece."""
    piece_idx, method, q_pert_full, sigma, lat, lon = args

    # Re-import pvtend in each worker process
    import sys
    if _PVTEND_SRC not in sys.path:
        sys.path.insert(0, _PVTEND_SRC)
    import pvtend.sh_ops as _sh  # noqa: F401 (resets Spharmt cache in this process)

    # Mask PV to this piece's levels only
    q_masked = np.zeros_like(q_pert_full)
    for ki in PIECES[piece_idx]:
        q_masked[ki] = q_pert_full[ki]

    print(f"    Inverting piece {piece_idx} ({PIECE_NAMES[piece_idx]}) "
          f"method={method} | max|Q'| in piece = "
          f"{np.nanmax(np.abs(q_masked)):.2f} PVU")

    if method == "linear":
        psi = _invert_piece_linear(q_masked, sigma, lat, lon)
    else:
        psi = _invert_piece_nonlinear(q_masked, sigma, lat, lon)

    return piece_idx, psi


# ─────────────────────────────────────────────────────────────────────────────
# Wind recovery from streamfunction
# ─────────────────────────────────────────────────────────────────────────────

def psi_to_uv_250(psi_3d: np.ndarray, lat: np.ndarray, lon: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Geostrophic wind at 250 hPa from streamfunction ψ' (m²/s).

    u' = −∂ψ'/∂y,  v' = +∂ψ'/∂x
    Uses pvtend.sh_ops.gradient_sh for spectral-quality gradients.
    Returns (u250, v250) each (NY, NX) in m/s.
    """
    dpsi_dx, dpsi_dy = sh.gradient_sh(psi_3d[IDX_250], lat, lon, R_EARTH)
    u250 = -dpsi_dy   # [m²/s / m] = [m/s]
    v250 =  dpsi_dx
    return u250, v250


# ─────────────────────────────────────────────────────────────────────────────
# Main PPVI runner
# ─────────────────────────────────────────────────────────────────────────────

def run_method(method: str, q_pert: np.ndarray, sigma: np.ndarray,
               lat: np.ndarray, lon: np.ndarray) -> Dict:
    """Run one method (linear or nonlinear) for all 3 pieces in parallel.

    Returns dict of {piece_idx: psi_3d}.
    """
    print(f"\n  Running method={method} …")
    tasks = [
        (i, method, q_pert, sigma, lat, lon)
        for i in range(len(PIECES))
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_invert_one_piece, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            piece_idx, psi = fut.result()
            results[piece_idx] = psi
            print(f"    piece {piece_idx} done  "
                  f"max|ψ'|={np.nanmax(np.abs(psi)):.3e} m²/s")
    return results


def build_output(method: str, piece_psi_dict: Dict, q_pert: np.ndarray,
                 q_event: np.ndarray, q_clim: np.ndarray,
                 theta_clim: np.ndarray, sigma: np.ndarray,
                 lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Assemble xr.Dataset from inversion results."""
    data_vars = {
        "q_pert":      (["pressure_level", "latitude", "longitude"], q_pert,
                        {"long_name": "Ertel PV anomaly Q'", "units": "PVU"}),
        "q_event":     (["pressure_level", "latitude", "longitude"], q_event,
                        {"long_name": "Ertel PV event",     "units": "PVU"}),
        "q_clim":      (["pressure_level", "latitude", "longitude"], q_clim,
                        {"long_name": "Ertel PV climatology", "units": "PVU"}),
        "theta_clim":  (["pressure_level", "latitude", "longitude"], theta_clim,
                        {"long_name": "Basic-state theta θ̄", "units": "K"}),
        "sigma_clim":  (["pressure_level", "latitude", "longitude"], sigma,
                        {"long_name": "Static stability |∂θ̄/∂p|", "units": "K/Pa"}),
    }
    for i in range(len(PIECES)):
        psi = piece_psi_dict[i]
        u250, v250 = psi_to_uv_250(psi, lat, lon)
        data_vars[f"psi_p{i}"]  = (["pressure_level", "latitude", "longitude"], psi,
                                    {"long_name": f"ψ' piece {PIECE_NAMES[i]}",
                                     "units": "m2 s-1"})
        data_vars[f"u_p{i}"]    = (["latitude", "longitude"], u250,
                                    {"long_name": f"u' 250 hPa piece {PIECE_NAMES[i]}",
                                     "units": "m s-1"})
        data_vars[f"v_p{i}"]    = (["latitude", "longitude"], v250,
                                    {"long_name": f"v' 250 hPa piece {PIECE_NAMES[i]}",
                                     "units": "m s-1"})

    coords = {
        "pressure_level": (["pressure_level"], PLEVS_HPA, {"units": "hPa"}),
        "latitude":       (["latitude"],        lat,       {"units": "degrees_north"}),
        "longitude":      (["longitude"],       lon,       {"units": "degrees_east"}),
    }
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "title":       f"PPVI output — method={method}",
        "case":        "2025-01-08 00Z California blocking",
        "balance":     method,
        "pieces":      str(PIECES),
        "piece_names": str(PIECE_NAMES),
    }
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows safety (no-op on Linux)

    print("=" * 60)
    print("PPVI engine — 2025-01-08 00Z CA blocking")
    print("=" * 60)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\nLoading ERA5 data …")
    ev_data, cl_data, lat, lon = load_data()
    print(f"  lat: {lat[0]:.1f} → {lat[-1]:.1f} N  (NY={lat.size})")
    print(f"  lon: {lon[0]:.1f} → {lon[-1]:.1f} E  (NX={lon.size})")
    print(f"  levels: {PLEVS_HPA}")

    # ── basic-state quantities ─────────────────────────────────────────────────
    print("\nComputing basic-state theta and stability …")
    theta_clim  = compute_theta(cl_data["t"], PLEVS_PA)
    sigma_clim  = compute_static_stability(theta_clim)
    print(f"  θ̄ range: {theta_clim.min():.1f} – {theta_clim.max():.1f} K")
    print(f"  σ̄ range: {sigma_clim.min():.2e} – {sigma_clim.max():.2e} K/Pa")

    # ── Ertel PV ───────────────────────────────────────────────────────────────
    print("\nComputing Ertel PV (event and climatology) …")
    q_event = compute_ertel_pv(ev_data["t"], ev_data["u"], ev_data["v"], lat, lon)
    q_clim  = compute_ertel_pv(cl_data["t"], cl_data["u"], cl_data["v"], lat, lon)
    q_pert  = q_event - q_clim
    print(f"  Q_event 250 hPa: {q_event[IDX_250].min():.2f} – "
          f"{q_event[IDX_250].max():.2f} PVU")
    print(f"  Q' 250 hPa:      {q_pert[IDX_250].min():.2f} – "
          f"{q_pert[IDX_250].max():.2f} PVU")

    # ── run both methods ──────────────────────────────────────────────────────
    for method in ("linear", "nonlinear"):
        out_nc = os.path.join(PPVI_DIR, f"method_{method}.nc")
        if os.path.exists(out_nc):
            print(f"\nSkipping {method} (output exists: {out_nc})")
            continue

        piece_psi = run_method(method, q_pert, sigma_clim, lat, lon)
        ds_out    = build_output(method, piece_psi,  q_pert,
                                 q_event, q_clim, theta_clim, sigma_clim,
                                 lat, lon)
        ds_out.to_netcdf(out_nc)
        print(f"  Saved → {out_nc}")

    print("\nPPVI engine complete.")
