"""Ladder step 1: the float64 spectral core.

These tests gate algebra, not physics.  They are deliberately strict -- the
Krylov solver downstream needs transforms good to machine precision, so anything
softer than ~1e-12 here would show up later as a residual floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.sht import (
    SHT,
    gaussian_grid,
    grid_from_axes,
    legendre_derivative_table,
    legendre_table,
)

R = 6.371e6


def era5_mirrored_grid():
    """ERA5 1.5 deg NH mirrored onto the sphere: 121 rows, both poles included."""
    nh = np.arange(0.0, 90.0 + 1e-9, 1.5)
    lat = np.concatenate([-nh[::-1][:-1], nh])
    lon = np.arange(240, dtype=float) * 1.5
    return grid_from_axes(lat, lon)


def f09_mirrored_grid():
    """CESM f09 NH mirrored: 192 rows, cell-centred about the equator, poles on."""
    nh = np.linspace(0.47120418848167, 90.0, 96)
    lat = np.concatenate([-nh[::-1], nh])
    lon = np.arange(288, dtype=float) * (360.0 / 288)
    return grid_from_axes(lat, lon)


def test_legendre_orthonormal():
    """(1/2) int Pbar_n^m^2 dmu == 1 and distinct degrees are orthogonal."""
    lmax = 24
    mu, w = np.polynomial.legendre.leggauss(64)
    p = legendre_table(lmax, mu)
    for m in (0, 1, 5, 12):
        block = p[m : lmax + 1, m]  # (n_count, nj)
        gram = 0.5 * (block * w[None, :]) @ block.T
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12)


def test_legendre_matches_known_values():
    """Low-degree terms against closed forms.

    Normalisation is fixed by ``(1/2) int Pbar^2 dmu == 1`` per term, so the
    sectoral constants carry the double-factorial ratio rather than the plain
    ``sqrt(2n+1)`` of the zonal ones.
    """
    mu = np.array([-0.7, -0.1, 0.0, 0.35, 0.9])
    p = legendre_table(3, mu)
    sin_theta = np.sqrt(1 - mu**2)
    assert np.allclose(p[0, 0], 1.0)
    assert np.allclose(p[1, 0], np.sqrt(3.0) * mu)
    assert np.allclose(p[1, 1], np.sqrt(1.5) * sin_theta)
    assert np.allclose(p[2, 0], np.sqrt(5.0) * 0.5 * (3 * mu**2 - 1))
    assert np.allclose(p[2, 1], np.sqrt(7.5) * mu * sin_theta)
    assert np.allclose(p[2, 2], np.sqrt(15.0 / 8.0) * sin_theta**2)
    for n in range(4):
        for m in range(n + 1, 4):
            assert np.all(p[n, m] == 0.0), f"P[{n},{m}] should be unused"


def test_legendre_derivative_table_matches_finite_difference():
    """H = (1-mu^2) dP/dmu, checked against a centred difference in mu."""
    lmax = 12
    mu0 = np.array([-0.62, -0.2, 0.13, 0.55, 0.88])
    h_step = 1e-6
    h_tab = legendre_derivative_table(legendre_table(lmax + 1, mu0), lmax)
    p_plus = legendre_table(lmax, mu0 + h_step)
    p_minus = legendre_table(lmax, mu0 - h_step)
    fd = (p_plus - p_minus) / (2 * h_step) * (1 - mu0**2)[None, None, :]
    for m in (0, 1, 4, 9):
        for n in range(m, lmax + 1):
            assert np.allclose(h_tab[n, m], fd[n, m], atol=1e-6, rtol=1e-5)


def test_lmax_guard():
    with pytest.raises(ValueError, match="512"):
        legendre_table(600, np.array([0.0]))
    with pytest.raises(ValueError, match="not resolved"):
        SHT(gaussian_grid(32, 64), lmax=40)


@pytest.mark.parametrize("nm", [(0, 0), (1, 0), (1, 1), (7, 3), (20, 20), (25, 4)])
def test_roundtrip_single_harmonic_gaussian(nm):
    """One harmonic in, the same one out, to machine precision."""
    n, m = nm
    sht = SHT(gaussian_grid(48, 96), lmax=32)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    spec[m, n] = 1.0 + (0.5j if m > 0 else 0.0)
    field = sht.synthesize(spec)
    back = sht.analyze(field)
    assert np.abs(back[m, n] - spec[m, n]) < 1e-13
    assert np.abs(back - spec).max() < 1e-13


def test_roundtrip_random_field_gaussian():
    rng = np.random.default_rng(0)
    sht = SHT(gaussian_grid(64, 128), lmax=40)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    for m in range(sht.lmax + 1):
        spec[m, m:] = rng.normal(size=sht.lmax + 1 - m) / (1 + np.arange(m, sht.lmax + 1))
        if m > 0:
            spec[m, m:] = spec[m, m:] + 1j * rng.normal(size=sht.lmax + 1 - m) / (
                1 + np.arange(m, sht.lmax + 1)
            )
    field = sht.synthesize(spec)
    assert np.abs(sht.analyze(field) - spec).max() < 1e-13


@pytest.mark.parametrize("grid_fn", [era5_mirrored_grid, f09_mirrored_grid])
def test_roundtrip_on_pole_inclusive_data_grid(grid_fn):
    """Least-squares analysis recovers a resolved field on a grid with poles."""
    grid = grid_fn()
    assert grid.has_poles
    sht = SHT(grid, lmax=min(40, grid.nlat - 1))
    rng = np.random.default_rng(3)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    for m in range(sht.lmax + 1):
        spec[m, m:] = rng.normal(size=sht.lmax + 1 - m)
        if m > 0:
            spec[m, m:] = spec[m, m:] + 1j * rng.normal(size=sht.lmax + 1 - m)
    field = sht.synthesize(spec)
    assert np.isfinite(field).all()
    assert np.abs(sht.analyze(field) - spec).max() < 1e-10


def test_pole_row_is_single_valued():
    """A scalar synthesised onto a pole row must not vary with longitude."""
    grid = era5_mirrored_grid()
    sht = SHT(grid, lmax=30)
    rng = np.random.default_rng(7)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    for m in range(sht.lmax + 1):
        spec[m, m:] = rng.normal(size=sht.lmax + 1 - m) + 1j * rng.normal(
            size=sht.lmax + 1 - m
        )
    field = sht.synthesize(spec)
    for row in (0, -1):
        assert np.ptp(field[row]) < 1e-12


def test_laplacian_eigenvalue():
    sht = SHT(gaussian_grid(48, 96), lmax=32, radius=R)
    for n, m in [(1, 0), (5, 2), (14, 14)]:
        spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
        spec[m, n] = 1.0
        lap = sht.laplacian_spec(spec)
        assert np.isclose(lap[m, n].real, -n * (n + 1) / R**2, rtol=1e-14)
        back = sht.invert_laplacian_spec(lap)
        assert np.abs(back - spec).max() < 1e-13


def test_gradient_against_analytic():
    """Gradient of Y_1^0 ~ sin(lat): d/dy = sqrt(3) cos(lat)/a, d/dx = 0."""
    sht = SHT(gaussian_grid(48, 96), lmax=32, radius=R)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    spec[0, 1] = 1.0
    dfdx, dfdy = sht.gradient(spec)
    cos_lat = sht.grid.cos_lat[:, None]
    assert np.abs(dfdx).max() < 1e-16
    assert np.abs(dfdy - np.sqrt(3.0) * cos_lat / R).max() < 1e-16


def test_gradient_refused_on_pole_grid():
    sht = SHT(era5_mirrored_grid(), lmax=20)
    spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    with pytest.raises(ValueError, match="pole"):
        sht.gradient(spec)


def test_vorticity_divergence_of_rotational_wind():
    """From a streamfunction: vorticity is its Laplacian, divergence is zero."""
    sht = SHT(gaussian_grid(64, 128), lmax=42, radius=R)
    rng = np.random.default_rng(11)
    psi_spec = np.zeros((sht.lmax + 1, sht.lmax + 1), dtype=complex)
    for m in range(6):
        psi_spec[m, max(m, 1) : 12] = 1e7 * (
            rng.normal(size=12 - max(m, 1)) + 1j * rng.normal(size=12 - max(m, 1))
        )
    psi_spec[0] = psi_spec[0].real
    dpsi_dx, dpsi_dy = sht.gradient(psi_spec)
    u, v = -dpsi_dy, dpsi_dx

    zeta = sht.vorticity(u, v)
    div = sht.divergence(u, v)
    zeta_expected = sht.laplacian_spec(psi_spec)

    scale = np.abs(zeta_expected).max()
    assert np.abs(zeta - zeta_expected).max() / scale < 1e-10
    assert np.abs(div).max() / scale < 1e-10


def test_regrid_between_grids_is_exact():
    """Band-limited fields move between grids without loss."""
    data = SHT(era5_mirrored_grid(), lmax=30)
    solver = SHT(gaussian_grid(64, 128), lmax=30)
    rng = np.random.default_rng(5)
    spec = np.zeros((data.lmax + 1, data.lmax + 1), dtype=complex)
    for m in range(data.lmax + 1):
        spec[m, m:] = rng.normal(size=data.lmax + 1 - m) + 1j * rng.normal(
            size=data.lmax + 1 - m
        )
    on_data = data.synthesize(spec)
    on_solver = data.regrid_to(solver, on_data)
    back = solver.regrid_to(data, on_solver)
    assert np.abs(back - on_data).max() / np.abs(on_data).max() < 1e-11


def test_level_stack_matches_per_level():
    """Leading axes are transformed independently (the batching contract)."""
    sht = SHT(gaussian_grid(48, 96), lmax=24)
    rng = np.random.default_rng(13)
    stack = rng.normal(size=(9, sht.grid.nlat, sht.grid.nlon))
    batched = sht.analyze(stack)
    for k in range(stack.shape[0]):
        assert np.abs(batched[k] - sht.analyze(stack[k])).max() < 1e-15


def test_pole_crossing_blob_winds_match_analytic():
    """The reason the package exists: a vortex straddling the pole.

    A Gaussian streamfunction centred at 85N covers the pole.  Its induced wind is
    purely tangential with speed ``2 r psi / L^2`` in great-circle distance ``r``;
    the spectral gradient must reproduce that everywhere, including the rows
    nearest the pole, with nothing special done for them.
    """
    solver = gaussian_grid(128, 256)
    sht = SHT(solver, lmax=80, radius=R)
    lat0, lon0, length = 85.0, 30.0, 1.5e6
    lon2d, lat2d = np.meshgrid(solver.lon, solver.lat)
    cos_c = np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0)) + np.cos(
        np.radians(lat2d)
    ) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2d - lon0))
    gcd = np.arccos(np.clip(cos_c, -1, 1)) * R
    psi = 1e7 * np.exp(-((gcd / length) ** 2))

    spec = sht.analyze(psi)
    dpsi_dx, dpsi_dy = sht.gradient(spec)
    speed = np.hypot(dpsi_dx, dpsi_dy)
    analytic = 2 * gcd / length**2 * psi

    assert np.isfinite(speed).all()
    near_pole = np.abs(solver.lat) > 80.0
    err = np.abs(speed - analytic)[near_pole].max() / analytic.max()
    assert err < 1e-3, f"pole-region wind error {err:.2e}"
    assert np.abs(speed - analytic).max() / analytic.max() < 1e-3
