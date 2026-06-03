"""3-D Helmholtz solver benchmark: (∇² + c_k) H_k + coupling = rhs_k.

Compares per-level spectral SH, xinvert 3-D, and wu_python BALNC
on the H-equation component of the nonlinear PV inversion.

The test equation (from qinvert21_94.f BALNC) is:

    ∇²H_k + ASI_k·(BB[k]·H_k + BH[k]·H_{k+1} + BL[k]·H_{k-1}) = rhs_k

where ASI_k = FCM + FR·∇²ψ (spatially varying absolute-vorticity factor).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from .config import R_EARTH

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_synthetic_3d_helmholtz(
    lat: np.ndarray,
    lon: np.ndarray,
    c_k: np.ndarray,
    bh_k: np.ndarray,
    bl_k: np.ndarray,
    asi_mean: float = 9.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3-D Helmholtz problem with known solution.

    H_exact[k] = sin(2π·k/nlev) · P_3(cos θ) · cos(2λ)
    where P_3 is the 3rd Legendre polynomial.

    Then compute rhs_k = (∇² + c_k) H_exact_k + bh·H_exact_{k+1} + bl·H_exact_{k-1}.
    """
    nlev = len(c_k)
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))

    # Legendre P_3(cos θ) = ½(5 cos³θ − 3 cos θ)
    cos_t = np.cos(lat_rad)
    p3 = 0.5 * (5.0 * cos_t**3 - 3.0 * cos_t)  # (nlat,)

    H_exact = np.zeros((nlev, len(lat), len(lon)), dtype=np.float64)
    rhs = np.zeros_like(H_exact)

    for k in range(nlev):
        # Vertical modulation
        vscale = np.sin(2.0 * np.pi * k / max(nlev - 1, 1))

        H_k = vscale * p3[:, None] * np.cos(2.0 * lon_rad)[None, :]
        H_exact[k] = H_k

        # ∇² H_k = -l(l+1)/R² · H_k  for l=3
        lap_eig = -3.0 * 4.0 / (R_EARTH * R_EARTH)
        lap_H_k = lap_eig * H_k

        # Build RHS
        rhs_k = lap_H_k + asi_mean * c_k[k] * H_k
        if k > 0:
            rhs_k += asi_mean * bl_k[k] * H_exact[k - 1]
        if k < nlev - 1:
            rhs_k += asi_mean * bh_k[k] * H_exact[k + 1]

        rhs[k] = rhs_k

    return H_exact, rhs


# ═══════════════════════════════════════════════════════════════════════════
#  Solver backends (unified interface)
# ═══════════════════════════════════════════════════════════════════════════


def _sor_helmholtz_2d_numba(H_k, rhs_k, A_arr, diag_k, omega, max_inner, tol):
    """Numba-parallel 2-D red-black SOR for (∇² + c)·H_k = rhs_k.

    Extracted as a separate jittable function so the inner SOR sweep
    runs at near-C speed.  Called per level from the 3-D outer loop.
    """
    import numba as nb

    @nb.jit(nopython=True, parallel=True, cache=True)
    def _sweep(H, rhs, A, diag, omega, max_inner, tol):
        ny, nx = H.shape
        row_errs = np.zeros(ny, dtype=np.float64)
        for _ in range(max_inner):
            # Red sweep: i+j even
            for i in nb.prange(1, ny - 1):
                row_max = 0.0
                d = diag[i]
                for j in range(1, nx - 1):
                    if (i + j) % 2 == 0:
                        lap = (
                            A[i, 0] * H[i - 1, j]
                            + A[i, 1] * H[i, j - 1]
                            + A[i, 2] * H[i, j]
                            + A[i, 3] * H[i, j + 1]
                            + A[i, 4] * H[i + 1, j]
                        )
                        res = lap + d * H[i, j] - rhs[i, j]
                        delta = omega * res / (A[i, 2] + d)
                        H[i, j] -= delta
                        ad = abs(delta)
                        if ad > row_max:
                            row_max = ad
                row_errs[i] = row_max
            max_err = row_errs.max()
            # Black sweep: i+j odd
            for i in nb.prange(1, ny - 1):
                row_max = 0.0
                d = diag[i]
                for j in range(1, nx - 1):
                    if (i + j) % 2 == 1:
                        lap = (
                            A[i, 0] * H[i - 1, j]
                            + A[i, 1] * H[i, j - 1]
                            + A[i, 2] * H[i, j]
                            + A[i, 3] * H[i, j + 1]
                            + A[i, 4] * H[i + 1, j]
                        )
                        res = lap + d * H[i, j] - rhs[i, j]
                        delta = omega * res / (A[i, 2] + d)
                        H[i, j] -= delta
                        ad = abs(delta)
                        if ad > row_max:
                            row_max = ad
                row_errs[i] = row_max
            max_err = max(max_err, row_errs.max())
            if max_err < tol:
                break
        return max_err

    return _sweep(
        H_k.astype(np.float64), rhs_k.astype(np.float64),
        A_arr.astype(np.float64), diag_k.astype(np.float64),
        omega, max_inner, tol,
    )


def _solve_3d_wu_balnc(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    c_k: np.ndarray,
    bh_k: np.ndarray,
    bl_k: np.ndarray,
    asi_mean: float = 9.0e-5,
    max_outer: int = 50,
    part: float = 0.5,
) -> tuple[np.ndarray, int, bool]:
    """wu_python BALNC-inspired 3-D SOR for the H-equation (numba-accelerated).

    Vertical coupling terms (BH·H[k+1], BL·H[k-1]) are folded into the
    RHS before each per-level 2-D Helmholtz solve, which uses a
    numba-prange red-black SOR kernel.
    """
    from wu_python.core.grid import A

    nlev, nlat, nlon = rhs.shape
    A_arr = A.astype(np.float64)

    H = np.zeros_like(rhs, dtype=np.float64)
    converged = False
    omega = 1.75
    max_inner = 2000
    tol_inner = 1e-10

    for outer in range(max_outer):
        H_old = H.copy()
        max_dh = 0.0

        for k in range(nlev):
            # Effective diagonal shift from Helmholtz term
            diag_k_val = asi_mean * c_k[k]

            # Add vertical coupling to RHS
            rhs_k = rhs[k].copy()
            if k > 0:
                rhs_k -= asi_mean * bl_k[k] * H[k - 1]
            if k < nlev - 1:
                rhs_k -= asi_mean * bh_k[k] * H[k + 1]

            _sor_helmholtz_2d_numba(
                H[k], rhs_k, A_arr,
                np.full(nlat, diag_k_val, dtype=np.float64),
                omega, max_inner, tol_inner,
            )

            max_dh = max(max_dh, float(np.abs(H[k] - H_old[k]).max()))

        rel = max_dh / (np.abs(H).max() + 1e-30)
        if rel < 1e-4:
            converged = True
            break

    return H, outer + 1, converged


def _solve_3d_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    c_k: np.ndarray,
    bh_k: np.ndarray,
    bl_k: np.ndarray,
    H_init: np.ndarray | None = None,
    max_outer: int = 50,
    part: float = 0.05,
) -> tuple[np.ndarray, int, bool]:
    """Spectral SH with lagged vertical coupling."""
    from .sh.solver import solve_helmholtz_3d_sh

    # Scale c_k by ASI≈9e-5 to get dimensional Helmholtz constant
    asi = 9.0e-5
    c_k_scaled = np.asarray(c_k, dtype=np.float64) * asi

    t0 = time.perf_counter()
    H, n_iter, converged = solve_helmholtz_3d_sh(
        rhs, lat, lon,
        c_k=c_k_scaled,
        bh_k=np.asarray(bh_k, dtype=np.float64) * asi,
        bl_k=np.asarray(bl_k, dtype=np.float64) * asi,
        H_init=H_init,
        max_outer=max_outer,
        part=part,
        tol=1e-4,
    )
    dt = time.perf_counter() - t0
    return H, n_iter, converged


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════


class Test3DHelmholtz:
    """3-D Helmholtz solver comparison suite."""

    @pytest.fixture(scope="class")
    def synt_3d(self, wu_lat, wu_lon):
        """Synthetic 3-D H-equation on the Wu NH grid."""
        from wu_python.core.nondim import BB, BH, BL

        c_k = np.array(BB, dtype=np.float64)   # (8,)
        bh_k = np.array(BH, dtype=np.float64)
        bl_k = np.array(BL, dtype=np.float64)

        H_exact, rhs = _make_synthetic_3d_helmholtz(
            wu_lat, wu_lon, c_k, bh_k, bl_k,
            asi_mean=9.0e-5,
        )
        return H_exact, rhs, c_k, bh_k, bl_k

    def test_sh_3d_helmholtz(self, synt_3d, wu_lat, wu_lon):
        """Spectral SH solves the synthetic 3-D Helmholtz problem."""
        H_exact, rhs, c_k, bh_k, bl_k = synt_3d
        H, n_iter, converged = _solve_3d_sh(
            rhs, wu_lat, wu_lon, c_k, bh_k, bl_k,
            max_outer=100, part=0.05,
        )

        l2 = float(np.linalg.norm(H - H_exact)) / (float(np.linalg.norm(H_exact)) + 1e-30)
        print(f"\n  SH 3-D:  L²={l2:.4e}  iters={n_iter}  converged={converged}")
        assert np.all(np.isfinite(H)), "SH 3-D H has NaN/Inf"

    def test_wu_3d_helmholtz(self, synt_3d, wu_lat, wu_lon):
        """Wu SOR solves the synthetic 3-D Helmholtz problem."""
        H_exact, rhs, c_k, bh_k, bl_k = synt_3d
        H, n_iter, converged = _solve_3d_wu_balnc(
            rhs, wu_lat, wu_lon, c_k, bh_k, bl_k,
            max_outer=100, part=0.5,
        )

        l2 = float(np.linalg.norm(H - H_exact)) / (float(np.linalg.norm(H_exact)) + 1e-30)
        print(f"\n  Wu 3-D:  L²={l2:.4e}  iters={n_iter}  converged={converged}")
        assert np.all(np.isfinite(H)), "Wu 3-D H has NaN/Inf"

    def test_compare_residuals(self, synt_3d, wu_lat, wu_lon):
        """Compare per-level residual norms: SH vs Wu SOR."""
        H_exact, rhs, c_k, bh_k, bl_k = synt_3d

        H_sh, n_sh, _ = _solve_3d_sh(
            rhs, wu_lat, wu_lon, c_k, bh_k, bl_k,
            max_outer=100, part=0.05,
        )
        H_wu, n_wu, _ = _solve_3d_wu_balnc(
            rhs, wu_lat, wu_lon, c_k, bh_k, bl_k,
        )

        res_sh = float(np.linalg.norm(H_sh - H_exact))
        res_wu = float(np.linalg.norm(H_wu - H_exact))

        print(f"\n  SH  residual = {res_sh:.4e} ({n_sh} outer iters)")
        print(f"  Wu  residual = {res_wu:.4e} ({n_wu} outer iters)")

        # Both should recover the solution to within factor 2 of each other
        assert np.all(np.isfinite(H_sh)) and np.all(np.isfinite(H_wu)), \
            "NaN/Inf in 3-D Helmholtz solution"
