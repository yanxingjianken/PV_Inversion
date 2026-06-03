"""2-D Poisson solver benchmark: ∇²ψ = ζ.

Compares Wu SOR, xinvert, spectral SH, and pvtend FFT on three test cases:
  A — Spherical harmonic Y_5^3 (global 181×360, smooth)
  B — Gaussian bump (Wu NH 51×87, localised)
  C — ERA5 500 hPa vorticity snapshot (Wu NH 51×87, real data)
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
#  Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════


def _l2_error(computed: np.ndarray, exact: np.ndarray) -> float:
    """Relative L² error: ‖c − e‖₂ / ‖e‖₂."""
    denom = float(np.linalg.norm(exact))
    if denom < 1e-30:
        return float(np.linalg.norm(computed))
    return float(np.linalg.norm(computed - exact)) / denom


def _linf_error(computed: np.ndarray, exact: np.ndarray) -> float:
    """L∞ error: max|computed − exact|."""
    return float(np.abs(computed - exact).max())


def _pole_error(
    computed: np.ndarray, exact: np.ndarray, lat: np.ndarray, pole_thresh: float = 80.0
) -> float:
    """L² error restricted to |lat| > pole_thresh degrees."""
    lat = np.asarray(lat, dtype=float).ravel()
    nlat = len(lat)
    pole_mask = np.abs(lat) > pole_thresh  # (nlat,) boolean
    if not pole_mask.any():
        return 0.0
    diff = (computed - exact)[pole_mask, :].ravel()
    exact_m = exact[pole_mask, :].ravel()
    denom = float(np.linalg.norm(exact_m))
    if denom < 1e-30:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff)) / denom


# ═══════════════════════════════════════════════════════════════════════════
#  Solver implementations (unified interface)
# ═══════════════════════════════════════════════════════════════════════════


def _make_laplacian_coeffs(lat: np.ndarray, nlon: int) -> np.ndarray:
    """Build 5-pt Laplacian coefficients A[i, 0..4] for a lat-lon grid.

    Mirrors the Wu Fortran coefficient computation (qinvert21_94.f lines
    222-227), allowing the SOR solver to work on arbitrary grid sizes.
    """
    nlat = len(lat)
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    dl = np.deg2rad(abs(lat[1] - lat[0])) if nlat > 1 else np.deg2rad(1.5)
    dp = 2.0 * np.pi / nlon
    sigm = dl / dp
    A = np.zeros((nlat, 5), dtype=np.float64)
    for i in range(nlat):
        ap = cos_lat[i]
        apm = cos_lat[i - 1] if i > 0 else cos_lat[i]
        app = cos_lat[i + 1] if i < nlat - 1 else cos_lat[i]
        A[i, 0] = sigm * sigm * apm / ap
        A[i, 1] = 1.0 / (ap * ap)
        A[i, 2] = -(2.0 + sigm * sigm * ap * (apm + app)) / (ap * ap)
        A[i, 3] = 1.0 / (ap * ap)
        A[i, 4] = sigm * sigm * app / ap
    return A


def _solve_wu_sor(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    psi_exact: np.ndarray | None = None,
    omega: float = 1.75,
    max_iter: int = 5000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int, float]:
    """Wu numba red-black SOR (2-D Poisson).

    Uses the production ``sor_poisson_2d`` for the native 51×87 grid,
    or a dynamically-built Laplacian for other grid sizes.
    """
    nlat, nlon = rhs.shape
    is_wu_grid = (nlat == 51 and nlon == 87)

    if is_wu_grid:
        # Use production solver (hardcoded to 51×87 Wu grid)
        from wu_python.core.sor_solver import sor_poisson_2d

        bc = np.zeros_like(rhs)
        if psi_exact is not None:
            bc[0, :] = psi_exact[0, :]
            bc[-1, :] = psi_exact[-1, :]
            bc[:, 0] = psi_exact[:, 0]
            bc[:, -1] = psi_exact[:, -1]

        psi, n_iter, max_err = sor_poisson_2d(
            rhs, bc, omega=omega, max_iter=max_iter, tol=tol,
        )
        return psi, n_iter, float(max_err)
    else:
        # Generic grid: build Laplacian coefficients dynamically
        from wu_python.core.sor_solver import _sor_red_black_numba

        A_arr = _make_laplacian_coeffs(np.asarray(lat, dtype=np.float64), nlon)
        psi = np.zeros_like(rhs, dtype=np.float64)
        if psi_exact is not None:
            psi[0, :] = psi_exact[0, :]
            psi[-1, :] = psi_exact[-1, :]
            psi[:, 0] = psi_exact[:, 0]
            psi[:, -1] = psi_exact[:, -1]

        n_iter, max_err = _sor_red_black_numba(
            psi, rhs.astype(np.float64), omega, max_iter, tol, A_arr,
        )
        return psi, n_iter, float(max_err)


def _solve_xinvert(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    with_mg: bool = False,
) -> tuple[np.ndarray, int, float]:
    """xinvert SOR (2-D Poisson) on lat-lon grid.

    Uses xarray wrapping; periodic in longitude, fixed in latitude.
    When *with_mg* is True, applies multi-grid acceleration.
    """
    import xarray as xr
    from xinvert import invert_Poisson

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # xinvert expects ascending lat for cartesian, or lat-lon coords.
    # For NH-only grids, use 'cartesian' with proper dx,dy to avoid
    # spherical-harmonic assumptions.
    use_cartesian = bool(lat[0] < lat[-1] and np.min(np.abs(lat)) < 5.0)
    coord_type = "cartesian" if use_cartesian else "lat-lon"

    da = xr.DataArray(
        rhs,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )

    iParams = {
        "BCs": ["fixed", "periodic"],
        "undef": float("nan"),
        "mxLoop": 5000,
        "tolerance": 1e-6,
        "optArg": None,
    }

    if with_mg:
        from xinvert import invert_MultiGrid
        result = invert_MultiGrid(
            invert_Poisson, da, dims=["lat", "lon"],
            coords=coord_type, iParams=iParams,
            ratio=4, gridNo=3,
        )
    else:
        result = invert_Poisson(
            da, dims=["lat", "lon"], coords=coord_type, iParams=iParams,
        )

    psi = np.asarray(result.values, dtype=np.float64)
    return psi, 5000, 0.0


def _solve_spectral_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """pvtend spectral SH Poisson (2-D).  Single-shot, no iteration."""
    from .sh.solver import solve_poisson_2d_sh

    t0 = time.perf_counter()
    psi = solve_poisson_2d_sh(rhs, lat, lon)
    dt = time.perf_counter() - t0
    return psi, 1, 0.0  # spectral methods are direct


def _solve_pvtend_fft(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """pvtend FFT + tridiagonal Poisson (2-D)."""
    from pvtend.helmholtz import solve_poisson_spherical_fft

    lat = np.asarray(lat, dtype=np.float64)
    dy = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6
    dlon_rad = np.deg2rad(lon[1] - lon[0])

    # Reorder to descending lat (pvtend convention) and use Dirichlet BC
    rhs_desc = rhs[::-1, :].copy()
    chi_desc = solve_poisson_spherical_fft(
        rhs_desc, lat[::-1], dy=dy, dlon_rad=dlon_rad,
        bc_type="dirichlet",
    )
    chi = chi_desc[::-1, :]
    return chi, 1, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

SOLVER_NAMES = [
    "wu_sor",
    "xinvert",
    "spectral_sh",
    "pvtend_fft",
]


def _get_solver(name: str) -> Callable:
    return {
        "wu_sor": _solve_wu_sor,
        "xinvert": lambda r, la, lo, pe=None, **kw: _solve_xinvert(r, la, lo, with_mg=False),
        "spectral_sh": lambda r, la, lo, pe=None, **kw: _solve_spectral_sh(r, la, lo),
        "pvtend_fft": lambda r, la, lo, pe=None, **kw: _solve_pvtend_fft(r, la, lo),
    }[name]


class Test2DPoisson:
    """2-D Poisson solver comparison suite."""

    # ── Case A: spherical harmonic (global grid) ──────────────────────────

    @pytest.mark.parametrize("solver_name", SOLVER_NAMES)
    def test_sph_harm_l5m3(self, solver_name, sph_harm_l5m3):
        """Y_5^3 on 181×360 global grid — tests pole closure."""
        lat, lon, psi_exact, zeta = sph_harm_l5m3
        solver = _get_solver(solver_name)

        t0 = time.perf_counter()
        psi, n_iter, _ = solver(zeta, lat, lon, psi_exact=psi_exact)
        dt = time.perf_counter() - t0

        l2 = _l2_error(psi, psi_exact)
        pole = _pole_error(psi, psi_exact, lat)

        # Direct solvers should recover the smooth SH near-exactly.
        # Iterative solvers have gauge/BC differences — check stability.
        if solver_name in ("spectral_sh",):
            assert l2 < 1e-6, f"{solver_name}: L² error {l2:.4e}"
        else:
            assert np.all(np.isfinite(psi)), f"{solver_name}: NaN/Inf"

        print(f"\n  {solver_name:14s}  L²={l2:.4e}  pole_err={pole:.4e}  "
              f"iters={n_iter}  dt={dt:.3f}s")

    # ── Case B: Gaussian bump (Wu NH grid) ────────────────────────────────

    @pytest.mark.parametrize("solver_name", SOLVER_NAMES)
    def test_gaussian_wu(self, solver_name, gaussian_wu, wu_lat, wu_lon):
        """Gaussian bump on 51×87 NH grid."""
        psi_exact, zeta = gaussian_wu
        solver = _get_solver(solver_name)

        t0 = time.perf_counter()
        psi, n_iter, _ = solver(zeta, wu_lat, wu_lon, psi_exact=psi_exact)
        dt = time.perf_counter() - t0

        l2 = _l2_error(psi, psi_exact)
        linf = _linf_error(psi, psi_exact)

        # Stability check: no NaN/Inf, finite values
        assert np.all(np.isfinite(psi)), f"{solver_name}: NaN/Inf"

        print(f"\n  {solver_name:14s}  L²={l2:.4e}  L∞={linf:.4e}  "
              f"iters={n_iter}  dt={dt:.3f}s")

    # ── Case C: ERA5 500 hPa (real data, no exact solution) ───────────────

    @pytest.mark.parametrize("solver_name", SOLVER_NAMES)
    def test_era5_500hpa(self, solver_name, era5_event_vorticity, wu_lat, wu_lon):
        """ERA5 500 hPa relative vorticity — compare residual norms."""
        # Extract 500-hPa level (index 5 in σ-coords: 1.0, 0.925, 0.85, 0.7, 0.6, 0.5...)
        zeta = era5_event_vorticity[5]  # σ=0.5 ≈ 500 hPa
        solver = _get_solver(solver_name)

        t0 = time.perf_counter()
        psi, n_iter, _ = solver(zeta, wu_lat, wu_lon, psi_exact=None)
        dt = time.perf_counter() - t0

        # Compute residual ‖∇²ψ − ζ‖₂
        from wu_python.core.grid import A, NY, NX

        lap_psi = np.zeros_like(psi)
        for i in range(1, NY - 1):
            for j in range(1, NX - 1):
                lap_psi[i, j] = (
                    A[i, 0] * psi[i - 1, j]
                    + A[i, 1] * psi[i, j - 1]
                    + A[i, 2] * psi[i, j]
                    + A[i, 3] * psi[i, j + 1]
                    + A[i, 4] * psi[i + 1, j]
                )
        residual = float(np.linalg.norm(lap_psi[1:-1, 1:-1] - zeta[1:-1, 1:-1]))
        norm_zeta = float(np.linalg.norm(zeta[1:-1, 1:-1]))
        rel_res = residual / (norm_zeta + 1e-30)

        assert np.all(np.isfinite(psi)), f"{solver_name}: NaN/Inf in solution"

        print(f"\n  {solver_name:14s}  residual={residual:.4e}  rel_res={rel_res:.4e}  "
              f"dt={dt:.3f}s")
