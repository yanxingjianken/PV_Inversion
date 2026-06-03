"""sh_ppvi.sor — Spectral-residual SOR solvers faithful to Wu/Davis Fortran.

Design
------
* The Fortran SOR update is:  ψ(i,j) ← ψ - ω · R / A(I,3)
  where  R = (1/DL²) ∑ A(I,m) ψ_neighbour - RHS  is the Laplacian residual.
  A(I,3) = −(2 + σ²·AP(I)·[APM(I)+APP(I)]) / AP(I)²  (always < 0).

* Here we replace the finite-difference Laplacian with the spherical-harmonic
  Laplacian from operators.laplacian(), and scale the residual by
      scale = mean(|A(I,3)|)  over all interior latitudes.
  This matches Wu's "Option A" (scalar scale).

* psi_sor_level  — 2-D SOR for one level of ψ (mirrors pvpialln L383–420
  and qinvert21 L800–822 and qinvertp21 L800–828).
* H_sor_3d       — 3-D coupled SOR for H with BB/BH/BL vertical coupling
  (mirrors qinvert21 L701–680 and qinvertp21 L701–745).
* outer_total_iteration — outer MAXT loop with PART underrelaxation
  (mirrors qinvert21 L900 and qinvertp21 L900).

Coordinate conventions:
  * All arrays (NW, nlat, nlon), NW=10 levels, nlat ascending (S→N).
  * k=0: surface (1000 hPa),  k=NW-1: top (200 hPa).
  * Homogeneous Dirichlet BCs at lateral boundaries enforced by zeroing
    the outermost row/column after each SOR sweep (IBC=0 case).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .coords import PI_VALS, NW
from .operators import laplacian
from .nondim import PI_WU

__all__ = [
    "compute_scale_A_I3",
    "psi_sor_level",
    "H_sor_3d",
    "outer_total_iteration",
    "build_BB_BH_BL",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Scale factor (mirrors Wu A(I,3) centre stencil weight)
# ──────────────────────────────────────────────────────────────────────────────

def compute_scale_A_I3(lat: np.ndarray, dlon_deg: float, dlat_deg: float) -> float:
    """Mean |A(I,3)| over all interior latitudes.

    Wu Fortran (pvpialln_94UV.f, L99–110):
        SIGM = HDR(5)/HDR(6)     (Δlon/Δlat, dimensionless)
        AP(I)  = cos(lat_I)
        APM(I) = cos(lat_I - Δlat/2)
        APP(I) = cos(lat_I + Δlat/2)
        A(I,3) = -(2 + SIGM²·AP(I)·(APM(I)+APP(I))) / AP(I)²

    Parameters
    ----------
    lat : (nlat,) ascending latitudes [degrees].
    dlon_deg, dlat_deg : float   grid spacing in degrees.

    Returns
    -------
    scale : positive float   mean |A(I,3)|
    """
    sigma = dlon_deg / dlat_deg
    lat_rad = np.deg2rad(lat)
    dlat_rad = np.deg2rad(dlat_deg)
    AP  = np.cos(lat_rad)
    APM = np.cos(lat_rad - 0.5 * dlat_rad)
    APP = np.cos(lat_rad + 0.5 * dlat_rad)
    AP_safe = np.where(AP == 0, 1e-10, AP)
    A3 = -(2.0 + sigma**2 * AP * (APM + APP)) / AP_safe**2
    # Use mid-latitude (45°) representative value, NOT mean: mean is dominated
    # by the 1/cos(lat)² singularity near the pole, whereas the spectral
    # Laplacian has no such polar blow-up.  Use lat closest to 45°.
    idx = int(np.argmin(np.abs(lat - 45.0)))
    return float(np.abs(A3[idx]))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Vertical operator coefficients BB, BH, BL, DPI2
# ──────────────────────────────────────────────────────────────────────────────

def build_BB_BH_BL(
    pi: np.ndarray = PI_VALS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build Wu's vertical second-derivative coefficients (qinvert21 L202–209).

    For interior levels k=1..NW-2 (0-indexed):
        BB(k) = -2 / [(π(k+1)-π(k))·(π(k)-π(k-1))]
        BH(k) =  2 / [(π(k+1)-π(k))·(π(k+1)-π(k-1))]   ← upper coeff
        BL(k) =  2 / [(π(k+1)-π(k-1))·(π(k)-π(k-1))]   ← lower coeff
        DPI2(k) = (π(k+1)-π(k-1)) / 2

    Returns arrays of shape (NW,); values at k=0 and k=NW-1 are set to 0
    (those rows handled by boundary conditions).
    """
    N = len(pi)
    BB   = np.zeros(N)
    BH   = np.zeros(N)
    BL   = np.zeros(N)
    DPI2 = np.zeros(N)
    for k in range(1, N - 1):
        dkp = pi[k + 1] - pi[k]
        dkm = pi[k] - pi[k - 1]
        BB[k]   = -2.0 / (dkp * dkm)
        BH[k]   =  2.0 / (dkp * (dkp + dkm))
        BL[k]   =  2.0 / ((dkp + dkm) * dkm)
        DPI2[k] = (pi[k + 1] - pi[k - 1]) / 2.0
    return BB, BH, BL, DPI2


# pre-compute for the ERA5 grid using Wu's non-dim π (Δπ_wu ≈ 1 per level)
# Using PI_VALS (Δπ_phys ≈ 0.05) gave BB ≈ -3886 which blew up STB.
# With PI_WU = CP·PI_VALS/DPI (Δπ_wu ≈ 1) BB ≈ O(-10) — matches Fortran.
BB, BH, BL, DPI2 = build_BB_BH_BL(PI_WU)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  2-D SOR for ψ at one level
# ──────────────────────────────────────────────────────────────────────────────

def psi_sor_level(
    psi_k: np.ndarray,
    rhs_k: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    scale: float,
    LL: float = 1.0,
    omega: float = 1.4,
    thr: float = 0.01,
    max_it: int = 200,
    dirichlet_bc: bool = False,
) -> Tuple[np.ndarray, int, bool]:
    """One level of the Wu ψ SOR.

    Solves  ∇²ψ = rhs  by SOR using the spherical-harmonic Laplacian.

    Wu's FD Laplacian in non-dim coords is (1/Δλ_nd²)§·A(I,m)§·ψ.  Spectrally,
    `∇²_nondim = LL² · ∇²_phys`.  Passing `LL = R·Δλ` makes the spectral residual
    match Wu's non-dim convention where `scale = mean|A(I,3)| ≈ 6`.
    Pass `LL=1` to stay in physical units (then `scale` must also be in m⁻²).

    SOR update (mirrors pvpialln L395–403, qinvert21 L800–822):
        R     = LL² · ∇²_phys ψ − rhs
        Δψ    = −ω · R / scale
        ψ    += Δψ
        converged when max|Δψ| < thr

    Parameters
    ----------
    psi_k  : (nlat, nlon) current ψ field.
    rhs_k  : (nlat, nlon) RHS field.
    lat, lon : coordinate arrays.
    scale  : normalisation denominator (= mean |A(I,3)|).
    LL     : zonal grid spacing in metres (Wu's LL); default 1 = physical units.
    omega  : SOR overrelaxation parameter.
    thr    : convergence threshold (same units as ψ).
    max_it : maximum iterations.
    dirichlet_bc : if True, zero boundary values after each sweep.

    Returns
    -------
    psi_k  : updated field.
    n_it   : number of iterations taken.
    converged : bool.
    """
    psi_k = psi_k.copy()
    LL2   = LL * LL

    for it in range(1, max_it + 1):
        # spectral Laplacian residual: R = LL² · ∇²_phys ψ − rhs  (Wu non-dim)
        lap = LL2 * laplacian(psi_k[np.newaxis], lat, lon)[0]
        R   = lap - rhs_k
        dpsi = -omega * R / scale
        psi_k += dpsi

        if dirichlet_bc:
            psi_k[0,  :] = 0.0
            psi_k[-1, :] = 0.0
            psi_k[:,  0] = 0.0
            psi_k[:, -1] = 0.0

        if np.max(np.abs(dpsi)) < thr:
            return psi_k, it, True

    return psi_k, max_it, False


# ──────────────────────────────────────────────────────────────────────────────
# 4.  3-D coupled SOR for H (geopotential)
# ──────────────────────────────────────────────────────────────────────────────

def H_sor_3d(
    H: np.ndarray,
    ASI: np.ndarray,
    rh: np.ndarray,
    theta_B: np.ndarray,
    theta_T: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    scale: float,
    LL: float = 1.0,
    omega: float = 1.4,
    thr: float = 0.01,
    max_it: int = 200,
    dirichlet_bc: bool = False,
) -> Tuple[np.ndarray, int, bool]:
    """3-D SOR for geopotential H (mirrors qinvert21 L701–688).

    Solves the combined balance + PV equation for H given current ψ and ASI:

        ∇²H + ASI·[BB·H + BH·H(k+1) + BL·H(k-1)] = rh(i,j,k)

    with boundary conditions (qinvert21 L260–265):
        H(k=0)   = H(k=1)   + θ_B · (π(2)−π(1))
        H(k=NW-1)= H(k=NW-2)− θ_T · (π(NW)−π(NW-1))

    The SOR update at each interior grid point uses the diagonal coefficient
    as the normalisation denominator:
        pivot = scale + ASI[k] · BB[k]          (interior levels)
        pivot = scale + ASI[k] · (BB[k]+BL[k])  (bottom interior level k=1)
        pivot = scale + ASI[k] · (BB[k]+BH[k])  (top interior level k=NW-2)

    Parameters
    ----------
    H      : (NW, nlat, nlon) current H field (modified in-place copy).
    ASI    : (NW, nlat, nlon)  = (f + ∇²ψ) / (FR·STB·|A(I,3)|)  diagonal coeff.
    rh     : (NW, nlat, nlon)  RHS from balnc_rhs / balp_rhs.
    theta_B: (nlat, nlon)  lower boundary θ [K] (half level between k=0 and k=1).
    theta_T: (nlat, nlon)  upper boundary θ [K] (half level between k=NW-2 and NW-1).
    lat, lon : coordinate arrays.
    scale  : denominator (mean |A(I,3)|, same as psi SOR).
    omega  : SOR overrelaxation parameter.
    thr    : convergence threshold (m or non-dim).
    max_it : maximum iterations.
    dirichlet_bc : zero lateral boundaries after each sweep.

    Returns
    -------
    H      : updated (NW, nlat, nlon).
    n_it   : int.
    converged : bool.
    """
    NL  = NW
    H   = H.copy()
    LL2 = LL * LL

    for it in range(1, max_it + 1):
        converged_flag = True

        # --- Apply boundary theta (qinvert21 L260–265) ---
        _apply_theta_bc(H, theta_B, theta_T)

        for k in range(1, NL - 1):
            # vertical coupling terms
            vert = (ASI[k]
                    * (BH[k] * H[k + 1] + BL[k] * H[k - 1] + BB[k] * H[k]))

            # horizontal Laplacian in Wu non-dim spatial units
            lap_H_k = LL2 * laplacian(H[k][np.newaxis], lat, lon)[0]

            R = lap_H_k + vert - rh[k]

            # diagonal pivot
            if k == 1:
                pivot = scale + ASI[k] * (BB[k] + BL[k])
            elif k == NL - 2:
                pivot = scale + ASI[k] * (BB[k] + BH[k])
            else:
                pivot = scale + ASI[k] * BB[k]

            # guard against near-zero pivot
            pivot = np.where(np.abs(pivot) < 1e-30, np.sign(pivot + 1e-31) * 1e-30, pivot)

            DH = -omega * R / pivot
            H[k] += DH

            if dirichlet_bc:
                H[k, 0,  :] = 0.0
                H[k, -1, :] = 0.0
                H[k, :,  0] = 0.0
                H[k, :, -1] = 0.0

            if np.max(np.abs(DH)) > thr:
                converged_flag = False

        if converged_flag:
            _apply_theta_bc(H, theta_B, theta_T)
            return H, it, True

    _apply_theta_bc(H, theta_B, theta_T)
    return H, max_it, False


def _apply_theta_bc(
    H: np.ndarray,
    theta_B: np.ndarray,
    theta_T: np.ndarray,
) -> None:
    """Update ghost levels of H from θ boundary conditions (in-place).

    (qinvert21 L260–265):
        H[0]     = H[1]     + θ_B · (π[1] − π[0])
        H[NW-1]  = H[NW-2]  − θ_T · (π[NW-1] − π[NW-2])
    """
    H[0]      = H[1]      + theta_B * (PI_VALS[1]      - PI_VALS[0])
    H[NW - 1] = H[NW - 2] - theta_T * (PI_VALS[NW - 1] - PI_VALS[NW - 2])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Outer total-iteration loop
# ──────────────────────────────────────────────────────────────────────────────

def outer_total_iteration(
    state: Dict[str, np.ndarray],
    rhs_builders: Dict[str, Callable],
    params: Dict,
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """Run the Wu outer MAXT loop (underrelaxation + joint convergence).

    Mirrors qinvert21_94.f L900 and qinvertp21_94.f L900.

    The outer loop alternates:
      1. Build ψ-RHS from current H.
      2. SOR for ψ until convergence.
      3. Underrelax: ψ ← PART·ψ_new + (1−PART)·ψ_old.
      4. Build H-RHS from updated ψ.
      5. 3-D SOR for H until convergence.
      6. Update boundary θ BCs.
      7. Underrelax: H ← PART·H_new + (1−PART)·H_old.
      8. Check total convergence: both SOR loops converged in 1 iteration.
         If so, stop; otherwise repeat up to MAXT times.

    Parameters
    ----------
    state : dict with keys 'psi', 'H'  — (NW, nlat, nlon) each.
    rhs_builders : dict with keys
        'psi_rhs'  : callable(H, psi, state, params) → (NW, nlat, nlon)
        'H_rhs'    : callable(H, psi, state, params) → (NW, nlat, nlon)
        'ASI'      : callable(psi, state, params) → (NW, nlat, nlon)
        'theta_B'  : callable(state) → (nlat, nlon)
        'theta_T'  : callable(state) → (nlat, nlon)
    params : dict with keys
        'lat', 'lon', 'scale', 'omegs', 'omegh', 'thr', 'part',
        'max_psi', 'max_H', 'max_total'

    Returns
    -------
    state : updated dict.
    hist  : list of per-iteration dicts with convergence info.
    """
    lat     = params["lat"]
    lon     = params["lon"]
    scale   = params["scale"]
    omegs   = params.get("omegs", 1.4)
    omegh   = params.get("omegh", 1.4)
    thr     = params.get("thr", 0.01)
    part    = params.get("part", 0.5)
    max_psi = params.get("max_psi", 200)
    max_H   = params.get("max_H", 200)
    max_tot = params.get("max_total", 200)
    dirichlet = params.get("dirichlet_bc", True)

    psi = state["psi"].copy()
    H   = state["H"].copy()
    hist = []

    for iitot in range(max_tot):
        psi_old = psi.copy()
        H_old   = H.copy()

        # --- Step 1: build ψ RHS and solve ---
        psi_rhs_3d = rhs_builders["psi_rhs"](H, psi, state, params)

        psi_new = psi.copy()
        n_psi_tot = 0
        psi_conv = True
        for k in range(1, NW - 1):
            psi_new[k], n_it, conv = psi_sor_level(
                psi_new[k], psi_rhs_3d[k], lat, lon,
                scale=scale, omega=omegs, thr=thr, max_it=max_psi,
                dirichlet_bc=dirichlet,
            )
            n_psi_tot += n_it
            if not conv:
                psi_conv = False

        # --- Step 2: underrelax ψ ---
        if iitot > 0:
            psi = part * psi_new + (1.0 - part) * psi_old
        else:
            psi = psi_new  # first iteration: no underrelax

        # --- Step 3: build H RHS and ASI ---
        ASI    = rhs_builders["ASI"](psi, state, params)        # (NW, nlat, nlon)
        H_rhs  = rhs_builders["H_rhs"](H, psi, state, params)  # (NW, nlat, nlon)
        th_B   = rhs_builders["theta_B"](state)                 # (nlat, nlon)
        th_T   = rhs_builders["theta_T"](state)                 # (nlat, nlon)

        # --- Step 4: 3-D H SOR ---
        H_new, n_H, H_conv = H_sor_3d(
            H, ASI, H_rhs, th_B, th_T, lat, lon,
            scale=scale, omega=omegh, thr=thr, max_it=max_H,
            dirichlet_bc=dirichlet,
        )

        # --- Step 5: underrelax H ---
        if iitot > 0:
            H = part * H_new + (1.0 - part) * H_old
        else:
            H = H_new

        hist.append({
            "iitot": iitot,
            "n_psi": n_psi_tot,
            "n_H": n_H,
            "psi_conv": psi_conv,
            "H_conv": H_conv,
        })

        # --- Step 6: check total convergence ---
        # Mirrors qinvert21 L704-720: stop when both H and all ψ converge in 1 iter
        if psi_conv and H_conv and n_psi_tot <= (NW - 2) and n_H == 1:
            break

    state = dict(state)
    state["psi"] = psi
    state["H"]   = H
    return state, hist
