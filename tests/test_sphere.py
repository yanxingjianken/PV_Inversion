"""Ladder step 2: invariant horizontal operators.

The headline test is the three-way agreement on the nonlinear balance term.  A
flat finite difference drops the ``tan(lat)`` metric terms and still looks
plausible in mid-latitudes, so the check is made against two independent
covariant forms rather than against intuition.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.mirror import coriolis_star
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps

R = 6.371e6


def ops_and_spec(lmax=32, nlat=72, nlon=144, seed=1, nmax=10):
    sht = SHT(gaussian_grid(nlat, nlon), lmax=lmax, radius=R)
    ops = SphereOps(sht)
    rng = np.random.default_rng(seed)
    spec = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    for m in range(min(6, nmax)):
        lo = max(m, 1)
        size = nmax - lo
        if size <= 0:
            continue
        spec[m, lo:nmax] = rng.normal(size=size) + 1j * rng.normal(size=size)
    spec[0] = spec[0].real
    return ops, spec


def test_requires_gaussian_grid():
    from pvinv_sph.sht import grid_from_axes

    lat = np.linspace(-90, 90, 61)
    sht = SHT(grid_from_axes(lat, np.arange(120) * 3.0), lmax=20)
    with pytest.raises(ValueError, match="Gaussian"):
        SphereOps(sht)


def test_div_grad_is_laplacian():
    ops, spec = ops_and_spec()
    gx, gy = ops.grad(spec)
    lap_from_div = ops.div(gx, gy)
    lap_direct = ops.lap(spec)
    scale = np.abs(lap_direct).max()
    assert np.abs(lap_from_div - lap_direct).max() / scale < 1e-11


def test_dot_grad_is_symmetric_and_matches_product_rule():
    """``grad(a).grad(b) = (lap(ab) - a lap b - b lap a)/2`` -- an identity."""
    ops, a = ops_and_spec(seed=2)
    _, b = ops_and_spec(seed=3)
    direct = ops.dot_grad(a, b)
    assert np.allclose(direct, ops.dot_grad(b, a))

    a_g, b_g = ops.synth(a), ops.synth(b)
    lap_ab = ops.synth(ops.lap(ops.analyze(a_g * b_g)))
    identity = 0.5 * (lap_ab - a_g * ops.synth(ops.lap(b)) - b_g * ops.synth(ops.lap(a)))
    assert np.abs(direct - identity).max() / np.abs(direct).max() < 1e-8


def geodesic_blob(ops, lat0, lon0, radius_m=1.2e6, amplitude=1.0e7):
    """A streamfunction depending only on great-circle distance from a centre.

    Two such blobs at different centres are related by a rotation of the sphere,
    which is what makes them useful: any covariantly built quantity must take the
    same values around each of them.
    """
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    cos_c = np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0)) + np.cos(
        np.radians(lat2d)
    ) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2d - lon0))
    gcd = np.arccos(np.clip(cos_c, -1, 1)) * ops.sht.radius
    return amplitude * np.exp(-((gcd / radius_m) ** 2)), gcd


@pytest.mark.parametrize("lat0", [60.0, 80.0, 89.0, 90.0])
def test_balance_nonlinear_is_rotation_invariant(lat0):
    """The nonlinear term must not know where on the sphere the vortex sits.

    The same geodesic vortex is moved to 60N, 80N, 89N and onto the pole itself.
    With the Coriolis term switched off everything left is built covariantly, so
    the integral of its square over a cap around the centre is a rotation
    invariant and has to reproduce the mid-latitude value.  This is the property
    the package exists for, and a formulation that drops metric terms loses it at
    high latitude.

    The comparison uses the cap integral rather than the peak: the peak is a
    maximum over grid points, and where those fall relative to the centre changes
    with latitude, which moves it by a fraction of a percent for reasons that have
    nothing to do with the operator.
    """
    ops = SphereOps(SHT(gaussian_grid(160, 320), lmax=100, radius=R))
    zero_f = np.zeros((ops.grid.nlat, ops.grid.nlon))
    weights = ops.grid.weights[:, None]

    def cap_integral(lat_c, lon_c):
        psi_grid, gcd = geodesic_blob(ops, lat_c, lon_c)
        field = ops.synth(ops.balance_nonlinear(ops.analyze(psi_grid), zero_f))
        assert np.isfinite(field).all()
        return float(np.sum((field**2 * weights)[gcd < 3.0e6]))

    reference = cap_integral(45.0, 120.0)
    moved = cap_integral(lat0, 250.0)
    assert abs(moved - reference) / reference < 1e-5, (
        f"cap integral drifted to {moved:.6e} from {reference:.6e} at {lat0}N"
    )


def test_deformation_identity_holds_for_a_resolved_field():
    """Cross-check on the production path through the deformation identity.

    ``div(f grad psi) + 2 det(Hess) - |grad psi|^2/a^2`` must reproduce
    :meth:`balance_nonlinear`.  The check carries a floor of a few parts in a
    thousand and no better: it re-expands the wind components as scalars, and on
    a sphere those are not band limited even when the streamfunction is.  That is
    loose, but a dropped metric term shows up here at order one, so the two cases
    are never in doubt.
    """
    ops = SphereOps(SHT(gaussian_grid(192, 384), lmax=120, radius=R))
    psi = np.zeros((ops.sht.lmax + 1, ops.sht.lmax + 1), dtype=complex)
    psi[0, 1] = 1.4e7
    psi[0, 2] = -0.6e7
    psi[2, 3] = 3.0e6 + 1.5e6j
    lat = ops.grid.lat[:, None]
    f = coriolis_star(np.broadcast_to(lat, (ops.grid.nlat, ops.grid.nlon)))

    production = ops.synth(ops.balance_nonlinear(psi, f))
    reconstructed = (
        ops.synth(ops.div_c_grad(f, psi))
        + ops.hessian_determinant_from_deformation(psi)
        - ops.curvature_term(psi)
    )
    band = np.abs(ops.grid.lat) < 70.0
    scale = np.abs(production[band]).max()
    err = np.abs(production[band] - reconstructed[band]).max() / scale
    assert err < 1e-2, f"deformation cross-check {err:.2e}"

    # What this test does pin is the metric terms in D1 and D2: without them the
    # reconstruction misses by order one.  The curvature term is pinned elsewhere,
    # by test_solid_body_rotation_against_closed_form, where dropping it changes
    # the closed-form answer outright.


def test_curvature_term_is_the_gap_to_the_plane_form():
    """Size and sign of what a limited-area code drops, on a realistic jet.

    The curvature term is ``|V|^2/a^2``.  For a 30 m/s flow that is a couple of
    percent of the balance terms -- small, systematic, and one-signed, which is
    why it belongs in the record of how this solver differs from the Fortran
    rather than in a list of rounding differences.
    """
    ops = SphereOps(SHT(gaussian_grid(160, 320), lmax=100, radius=R))
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    psi_grid = -3.0e7 * np.sin(np.radians(lat2d)) - 8.0e6 * np.exp(
        -(((lat2d - 45.0) / 10.0) ** 2 + ((lon2d - 200.0) / 25.0) ** 2)
    )
    psi = ops.analyze(psi_grid)
    u, v = ops.rotational_wind(psi)
    speed = np.hypot(u, v)

    curvature = ops.curvature_term(psi)
    assert np.all(curvature >= 0.0)
    assert np.allclose(curvature, speed**2 / R**2, rtol=1e-10)

    lat = ops.grid.lat[:, None]
    f = coriolis_star(np.broadcast_to(lat, (ops.grid.nlat, ops.grid.nlon)))
    balance = ops.synth(ops.balance_nonlinear(psi, f))
    band = (ops.grid.lat > 25) & (ops.grid.lat < 65)
    ratio = np.abs(curvature[band]).max() / np.abs(balance[band]).max()
    assert 1e-3 < ratio < 0.1, f"curvature share {ratio:.3f} outside the expected band"


def test_tangent_reproduces_quadratic_increment_exactly():
    """The INLIN=1 freeze: N(mean+pert) - N(mean) == DN(mean + pert/2)[pert].

    Exact because N is quadratic.  This identity is the whole reason piecewise
    solutions superpose, so it is asserted at solver tolerance rather than
    approximately.
    """
    ops, mean = ops_and_spec(lmax=42, nlat=96, nlon=192, seed=7)
    _, pert = ops_and_spec(lmax=42, nlat=96, nlon=192, seed=8)
    mean, pert = mean * 1e7, pert * 2e6
    lat = ops.grid.lat[:, None]
    f = coriolis_star(np.broadcast_to(lat, (ops.grid.nlat, ops.grid.nlon)))

    increment = ops.balance_nonlinear(mean + pert, f) - ops.balance_nonlinear(mean, f)
    tangent = ops.balance_nonlinear_tangent(mean + 0.5 * pert, pert, f)
    assert np.abs(increment - tangent).max() / np.abs(increment).max() < 1e-10


def test_tangent_is_linear_in_the_perturbation():
    """Pieces add: DN(ref)[a] + DN(ref)[b] == DN(ref)[a+b] for a frozen reference."""
    ops, ref = ops_and_spec(lmax=32, seed=9)
    _, a = ops_and_spec(lmax=32, seed=10)
    _, b = ops_and_spec(lmax=32, seed=11)
    ref, a, b = ref * 1e7, a * 1e6, b * 3e6
    lat = ops.grid.lat[:, None]
    f = coriolis_star(np.broadcast_to(lat, (ops.grid.nlat, ops.grid.nlon)))

    sum_of_parts = ops.balance_nonlinear_tangent(
        ref, a, f
    ) + ops.balance_nonlinear_tangent(ref, b, f)
    whole = ops.balance_nonlinear_tangent(ref, a + b, f)
    assert np.abs(sum_of_parts - whole).max() / np.abs(whole).max() < 1e-12


def test_solid_body_rotation_against_closed_form():
    """Solid-body rotation, worked out by hand end to end.

    For ``psi = C sin(lat)`` the wind is a rigid rotation, so the deformation
    vanishes and ``2 det(Hess) = zeta^2/2``.  The spherical balance term is then
    ``zeta^2/2 - |grad psi|^2/a^2``, which reduces to ``C^2 (3 sin^2 - 1)/a^4`` --
    a closed form that fixes both the curvature correction and its sign.
    """
    ops = SphereOps(SHT(gaussian_grid(96, 192), lmax=60, radius=R))
    amp = 1e7
    psi = np.zeros((ops.sht.lmax + 1, ops.sht.lmax + 1), dtype=complex)
    psi[0, 1] = amp
    zeta = ops.synth(ops.lap(psi))
    zero_f = np.zeros((ops.grid.nlat, ops.grid.nlon))

    nonlinear = ops.synth(ops.balance_nonlinear(psi, zero_f))
    # psi on the grid is amp * sqrt(3) * sin(lat), so C = amp * sqrt(3).
    c = amp * np.sqrt(3.0)
    sin2 = np.sin(np.radians(ops.grid.lat))[:, None] ** 2
    closed_form = c**2 * (3.0 * sin2 - 1.0) / R**4
    scale = np.abs(closed_form).max()
    assert np.abs(nonlinear - closed_form).max() / scale < 1e-9

    identity = 0.5 * zeta**2 - ops.curvature_term(psi)
    assert np.abs(nonlinear - identity).max() / scale < 1e-9

    # Deformation vanishes for a rigid rotation.  The shearing component carries
    # the truncation floor described in test_deformation_identity_holds_for_a_
    # resolved_field -- a few parts in a thousand, against order one if the
    # tan(lat) term were dropped.
    d1, d2 = ops.deformation(psi)
    band = np.abs(ops.grid.lat) < 80.0
    assert np.abs(d1[band]).max() / np.abs(zeta).max() < 1e-12
    assert np.abs(d2[band]).max() / np.abs(zeta).max() < 1e-2


def test_rotational_wind_is_nondivergent():
    ops, psi = ops_and_spec(lmax=42, nlat=96, nlon=192, seed=12)
    u, v = ops.rotational_wind(psi * 1e7)
    div = ops.div(u, v)
    zeta = ops.lap(psi * 1e7)
    assert np.abs(div).max() / np.abs(zeta).max() < 1e-10
