"""Ladder step 5: the diagnostic pass and the nonlinear total inversion.

The test that matters here is the round trip through the physics rather than
through the algebra: build a balanced state, read off the potential vorticity the
inversion would be given, and check that the inversion returns the state it came
from.  Nothing in that loop is a manufactured right-hand side, so it exercises
the residual, the Jacobian, the boundary handling and the gauge together.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.config import KrylovConfig, MirrorConfig, NewtonConfig
from pvinv_sph.levels import G, build_levels
from pvinv_sph.mirror import coriolis_star
from pvinv_sph.passab import (
    check_pv_units,
    diagnose,
    ertel_pv_pvu,
    potential_temperature,
)
from pvinv_sph.passc import BalancedInversion
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps

R = 6.371e6


def make_ops(lmax=24, nlat=48, nlon=96):
    return SphereOps(SHT(gaussian_grid(nlat, nlon), lmax=lmax, radius=R))


def synthetic_atmosphere(
    ops, levels, vortex_lat=45.0, vortex_amp=-1.2e7, theta_amp=0.0
):
    """A stably stratified westerly flow with one vortex, on the solver grid.

    Returned as the raw model output the diagnostic pass expects: geopotential,
    temperature, and wind.  ``theta_amp`` adds a surface-confined warm anomaly,
    which is what drives the boundary pieces -- without it the two boundary
    temperatures are identical between two states and those pieces are correctly
    zero.
    """
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    nlev = levels.nlev

    psi_grid = np.zeros((nlev, ops.grid.nlat, ops.grid.nlon))
    for k in range(nlev):
        amp = 1.0 + 0.5 * k
        psi_grid[k] = amp * (
            -2.4e7 * np.sin(np.radians(lat2d))
            + vortex_amp
            * np.exp(-(((lat2d - vortex_lat) / 12.0) ** 2 + ((lon2d - 200.0) / 25.0) ** 2))
        )

    theta = 290.0 + 9.0 * np.arange(nlev)[:, None, None] + 5.0 * np.cos(
        np.radians(lat2d)
    )
    if theta_amp:
        blob = np.exp(
            -(((lat2d - vortex_lat) / 12.0) ** 2 + ((lon2d - 200.0) / 25.0) ** 2)
        )
        decay = np.exp(-np.arange(nlev) / 1.5)[:, None, None]
        theta = theta + theta_amp * decay * blob[None, :, :]
    phi = np.zeros_like(psi_grid)
    for k in range(1, nlev):
        phi[k] = phi[k - 1] - 0.5 * (theta[k] + theta[k - 1]) * (
            levels.pi[k] - levels.pi[k - 1]
        )
    phi += 5.5e4  # a plausible offset; the inversion is blind to it

    temperature = theta / (
        (1.0e5 / (levels.p_hpa[:, None, None] * 100.0)) ** (2.0 / 7.0)
    )

    u = np.empty_like(psi_grid)
    v = np.empty_like(psi_grid)
    for k in range(nlev):
        u[k], v[k] = ops.rotational_wind(ops.analyze(psi_grid[k]))
    return phi, temperature, u, v, psi_grid, theta


def test_streamfunction_is_recovered_exactly():
    """No perimeter integral, no relaxation: psi is the inverse Laplacian.

    The comparison is against the *resolved* streamfunction.  The synthetic field
    is a Gaussian in latitude and longitude, which has no finite spherical-harmonic
    expansion, so the truncated projection is the only thing any spectral method
    can return -- comparing against the raw field would measure the truncation
    tail and call it a solver error.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, psi_grid, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    for k in range(levels.nlev):
        recovered = ops.synth(state.psi_spec[k])
        resolved = ops.synth(ops.analyze(psi_grid[k]))
        scale = np.abs(resolved - resolved.mean()).max()
        error = np.abs(
            (recovered - recovered.mean()) - (resolved - resolved.mean())
        ).max()
        assert error / scale < 1e-9, f"level {k}: {error / scale:.2e}"


def test_diagnosed_wind_matches_the_input_wind():
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    for k in (0, 4, levels.nlev - 1):
        uu, vv = ops.rotational_wind(state.psi_spec[k])
        assert np.abs(uu - u[k]).max() / np.abs(u[k]).max() < 1e-8
        assert np.abs(vv - v[k]).max() / np.abs(v[k]).max() < 1e-8


def test_potential_vorticity_is_physical():
    """Values in PVU, rising through the troposphere -- the Fortran's units check."""
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    summary = check_pv_units(state.q_hat, levels)
    assert 0.0 < summary["median"] < 5.0, summary
    assert 0.5 < summary["top_level_median"] < 20.0, summary

    pvu = ertel_pv_pvu(state.q_hat, levels)
    medians = np.median(pvu, axis=(1, 2))
    # Upper troposphere above lower, which is the robust statement.  The detailed
    # profile follows the synthetic lapse rate against the uneven Exner spacing
    # and says nothing about the solver.
    assert medians[-2] > 2.0 * medians[0], f"PV should rise upward: {medians}"
    assert np.all(medians > 0)


def test_double_exner_shows_up_in_the_units():
    """Passing theta where temperature belongs is visible, not silent.

    With the limited-area source the Exner factor gets applied twice, so the
    stratification -- and with it the potential vorticity -- comes out several
    times too large; the summary in PVU is what shows it.  With the operator
    source the potential vorticity comes from the geopotential and would not
    change at all, so the diagnosis refuses the pair instead: the boundary
    temperature the geopotential implies no longer matches the one the
    temperature gives.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, theta = synthetic_atmosphere(ops, levels)
    doubled = potential_temperature(theta, levels.p_hpa)
    good = check_pv_units(
        diagnose(ops, levels, phi, temperature, u, v, pv_source="data").q_hat, levels
    )
    bad = check_pv_units(
        diagnose(ops, levels, phi, doubled, u, v, pv_source="data").q_hat, levels
    )
    assert good["median"] < 5.0
    assert bad["median"] > 5.0
    assert bad["median"] > 5.0 * good["median"]
    consistent = diagnose(ops, levels, phi, temperature, u, v)
    inconsistent = diagnose(ops, levels, phi, doubled, u, v)
    assert abs(consistent.boundary_theta_ratio[1] - 1.0) < 0.05
    assert inconsistent.boundary_theta_ratio[1] < 0.7, inconsistent.boundary_theta_ratio
    assert check_pv_units(consistent.q_hat, levels)["median"] < 5.0


def test_inverted_level_axis_is_rejected():
    """A top-down level axis is caught before it becomes a wrong answer.

    Static stability cannot catch this on its own -- a reversed profile still
    looks stratified -- so the guard is on the geopotential, which rises with
    height whatever the weather.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    with pytest.raises(ValueError, match="geopotential decreases with height"):
        diagnose(ops, levels, phi[::-1], temperature[::-1], u[::-1], v[::-1])


def test_level_count_is_checked():
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    with pytest.raises(ValueError, match="levels"):
        diagnose(ops, levels, phi[:-1], temperature[:-1], u[:-1], v[:-1])


def test_balanced_state_reproduces_a_balanced_input():
    """The physics round trip.

    The synthetic state is built from a streamfunction and a hydrostatic
    geopotential, so it is close to -- but not exactly -- balanced.  Inverting its
    potential vorticity has to return a state whose wind matches the input to well
    inside the unbalanced part, and whose residual is far smaller than the one it
    started from.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)

    inv = BalancedInversion(
        ops,
        levels,
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-11, maxiter=300),
        newton=NewtonConfig(phi_tol=G * 0.01, max_steps=12),
    )
    before = inv.residual_norms(
        state.psi_spec, state.phi_spec, state.q_hat, state.theta_bot, state.theta_top
    )
    psi_bal, phi_bal, report = inv.solve(
        state.psi_spec, state.phi_spec, state.q_hat, state.theta_bot, state.theta_top
    )
    assert report.converged, str(report)
    assert report.steps <= 8, str(report)

    # Quadratic convergence: with the Jacobian consistent with the residual, each
    # step should square the error until it hits the solver tolerance.
    drops = np.array(report.residuals)
    # The first residual is the physical one (the gauge constant is removed
    # before the first step), and one Newton step from a nearly balanced input
    # removes most of it; the later steps are quadratic.
    assert drops[1] < 0.1 * drops[0], f"first step barely moved: {drops}"
    assert np.all(np.diff(drops) < 0), f"residual did not fall monotonically: {drops}"
    assert drops[-1] < 1e-5 * drops[0], f"did not converge: {drops}"

    after = inv.residual_norms(
        psi_bal, phi_bal, state.q_hat, state.theta_bot, state.theta_top
    )
    # The synthetic state already satisfies the potential-vorticity equation by
    # construction -- it is where q_hat came from -- so the work is all in the
    # balance equation, which must end well inside the Fortran's 0.1 m threshold.
    assert after["balance_m"] < 0.1, after
    assert after["balance_m"] < 1e-2 * before["balance_m"], (before, after)
    assert after["pv_pvu"] < 1e-3, after

    # The balanced wind differs from the input by the unbalanced part it removed:
    # present, but a small fraction of the flow.
    k = levels.nlev // 2
    ub, vb = ops.rotational_wind(psi_bal[k])
    speed = np.hypot(u[k], v[k]).max()
    assert 1e-4 < np.abs(ub - u[k]).max() / speed < 0.05
    assert np.abs(vb - v[k]).max() / speed < 0.05


def test_equatorial_blending_is_the_only_thing_between_us_and_an_exact_solve():
    """Quantify what the taper costs, so the number is on the record.

    With the taper off the nonlinear solve is exact to rounding: the residual of
    the equations as posed falls from over a kilometre of equivalent geopotential
    height to nothing.  With it on, the converged state satisfies the *tapered*
    equations instead, and the difference reads as a fraction of a metre -- almost
    all of it sourced at the equator, spread globally by the inverse Laplacian
    used to express it as a height.

    That is the price of taming the mirror's kink, and it is why the taper is off
    by default for states that are already smooth across the equator.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    args = (state.psi_spec, state.phi_spec, state.q_hat, state.theta_bot, state.theta_top)

    results = {}
    for blend in (False, True):
        inv = BalancedInversion(
            ops,
            levels,
            mirror=MirrorConfig(blend=blend),
            krylov=KrylovConfig(rtol=1e-12, maxiter=600),
            newton=NewtonConfig(phi_tol=G * 0.001, max_steps=12),
        )
        psi_bal, phi_bal, report = inv.solve(*args)
        assert report.converged, f"blend={blend}: {report}"
        # Whatever system it was handed, the solver drives its own residual down
        # to the rounding floor rather than stalling part way.
        assert report.residuals[-1] < 1e-8 * report.residuals[0], report.residuals
        assert inv.residual_via_operator(psi_bal, phi_bal, *args[2:])[1].frozen.report.worst() == 0.0
        results[blend] = inv.residual_norms(psi_bal, phi_bal, *args[2:])["balance_m"]

    assert results[False] < 1e-6, f"unblended solve should be exact: {results}"
    assert 0.05 < results[True] < 5.0, f"blending cost outside the known range: {results}"


def test_balanced_boundary_levels_stay_hydrostatic():
    """The returned state must satisfy the ghost relation the operator assumed."""
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    inv = BalancedInversion(
        ops, levels, mirror=MirrorConfig(blend=False), newton=NewtonConfig(max_steps=4)
    )
    psi_bal, phi_bal, _ = inv.solve(
        state.psi_spec, state.phi_spec, state.q_hat, state.theta_bot, state.theta_top
    )
    pi = levels.pi
    implied_bot = -(ops.synth(phi_bal[1] - phi_bal[0])) / (pi[1] - pi[0])
    implied_top = -(ops.synth(phi_bal[-1] - phi_bal[-2])) / (pi[-1] - pi[-2])
    # Compared against the resolved boundary temperature: the operator only ever
    # sees the spectral projection of it, so that is what it can reproduce.
    resolved_bot = ops.synth(ops.analyze(state.theta_bot))
    resolved_top = ops.synth(ops.analyze(state.theta_top))
    scale = np.abs(resolved_bot).max()
    assert np.abs(implied_bot - resolved_bot).max() / scale < 1e-10
    assert np.abs(implied_top - resolved_top).max() / scale < 1e-10


def test_polar_vortex_inverts_without_special_treatment():
    """The same case with the vortex over the pole must behave identically.

    No masking, no truncation, no band limit: the only difference from the
    mid-latitude case is where the vortex sits.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    results = {}
    for lat0 in (45.0, 88.0):
        phi, temperature, u, v, _, _ = synthetic_atmosphere(
            ops, levels, vortex_lat=lat0
        )
        state = diagnose(ops, levels, phi, temperature, u, v)
        inv = BalancedInversion(
            ops,
            levels,
            mirror=MirrorConfig(blend=False),
            krylov=KrylovConfig(rtol=1e-11, maxiter=300),
            newton=NewtonConfig(phi_tol=G * 0.01, max_steps=12),
        )
        psi_bal, phi_bal, report = inv.solve(
            state.psi_spec,
            state.phi_spec,
            state.q_hat,
            state.theta_bot,
            state.theta_top,
        )
        assert report.converged, f"{lat0}N: {report}"
        assert np.isfinite(psi_bal).all() and np.isfinite(phi_bal).all()
        results[lat0] = report.steps

    assert results[88.0] <= results[45.0] + 3, results
