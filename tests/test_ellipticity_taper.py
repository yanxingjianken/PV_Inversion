"""The two properties the nonlinear pass depends on and could not check before.

Pass C is Newton on ``F(x) = J_{x/2}[x] + F(0)``.  That is Newton only if the
operator is the derivative of the residual, which needs the bilinear part of the
frozen operator to be symmetric in the reference and the perturbation -- with the
equatorial taper on as well as off -- and it converges only if the linearised
balance row is elliptic, which the deformation of a strong anticyclone can break.
Both were found wanting on real events: the taper cost a factor of two per Newton
step everywhere, and the unlimited deformation left a fold that no inner solver
could get past.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.config import ClampConfig, MirrorConfig
from pvinv_sph.levels import build_levels
from pvinv_sph.operator import FrozenState, PiecewiseOperator
from pvinv_sph.passab import diagnose
from pvinv_sph.passc import BalancedInversion
from pvinv_sph.passd import invert_pieces
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps
from pvinv_sph.vertical import streamfunction_ghost
from tests.test_operator import random_state
from tests.test_passes import make_ops, synthetic_atmosphere


def _full_state(op, rng, scale_phi, scale_psi):
    """A perturbation on all levels, boundary levels as homogeneous ghosts."""
    phi_int, psi_int = random_state(op, rng, scale_phi=scale_phi, scale_psi=scale_psi)
    levels = op.levels
    nlev = levels.nlev
    shape = (nlev,) + phi_int.shape[1:]
    phi = np.zeros(shape, dtype=complex)
    psi = np.zeros(shape, dtype=complex)
    phi[1:-1] = phi_int
    psi[1:-1] = psi_int
    phi[0], phi[-1] = phi[1], phi[-2]
    psi[0], psi[-1] = psi[1], psi[-2]
    return phi, psi


def _residual_rows(frozen_ref, op_template, x_phi, x_psi):
    """``J_{ref}[x]`` on the interior levels, as unscaled rows."""
    op = PiecewiseOperator(frozen_ref)
    interior = op.levels.interior
    return op.apply(x_phi[interior], x_psi[interior])


@pytest.mark.parametrize("blend", [False, True])
def test_midpoint_identity_holds_with_the_taper(blend):
    """``F(a) - F(b) = J_{(a+b)/2}[a - b]`` to rounding, taper on or off.

    The identity is what makes the pieces the exact decomposition of the
    balanced perturbation and what makes the Newton step a Newton step.  It
    holds for any exactly quadratic system with a symmetric bilinear form.  The
    two states share the boundary temperature and carry its ghosts, as the
    nonlinear pass maintains; no floor and no limiter is active, which are the
    only things allowed to break it.
    """
    ops = make_ops(lmax=20, nlat=40, nlon=80)
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels, vortex_amp=-3.0e6)
    state = diagnose(ops, levels, phi, temperature, u, v)
    mirror = MirrorConfig(blend=blend)
    clamps = ClampConfig()
    inv = BalancedInversion(ops, levels, clamps=clamps, mirror=mirror)
    rng = np.random.default_rng(5)
    dummy = PiecewiseOperator(
        FrozenState(ops, levels, state.psi_spec, state.phi_spec, clamps=clamps, mirror=mirror)
    )
    dphi, dpsi = _full_state(dummy, rng, scale_phi=50.0, scale_psi=5.0e5)
    theta_b, theta_t = state.theta_bot, state.theta_top
    b_psi, b_phi = inv._rebuild_boundaries(
        state.psi_spec.copy(), state.phi_spec.copy(), theta_b, theta_t
    )
    a_psi, a_phi = inv._rebuild_boundaries(b_psi + dpsi, b_phi + dphi, theta_b, theta_t)
    interior = levels.interior
    zero_q = np.zeros((interior.size, ops.grid.nlat, ops.grid.nlon))
    unit = np.ones_like(dummy.frozen.deform_limit)

    def frozen(phi_s, psi_s):
        return FrozenState(
            ops, levels, psi_s, phi_s, clamps=clamps, mirror=mirror, deformation_limit=unit
        )

    def residual_rows(phi_s, psi_s):
        op = PiecewiseOperator(frozen(0.5 * phi_s, 0.5 * psi_s))
        r1, r2 = op.apply(phi_s[interior], psi_s[interior])
        b1, b2 = op.rhs_rows(zero_q, theta_b, theta_t)
        return r1 - b1, r2 - b2

    mid = frozen(0.5 * (a_phi + b_phi), 0.5 * (a_psi + b_psi))
    for state_ in (mid, frozen(0.5 * a_phi, 0.5 * a_psi), frozen(0.5 * b_phi, 0.5 * b_psi)):
        assert state_.report.worst() == 0.0, "the case must not touch a floor"
    assert FrozenState(
        ops, levels, a_psi, a_phi, clamps=clamps, mirror=mirror
    ).deform_limit.min() == 1.0, "the case must not need the limiter"

    fa1, fa2 = residual_rows(a_phi, a_psi)
    fb1, fb2 = residual_rows(b_phi, b_psi)
    j1, j2 = PiecewiseOperator(mid).apply((a_phi - b_phi)[interior], (a_psi - b_psi)[interior])
    for lhs, rhs in (((fa1 - fb1), j1), ((fa2 - fb2), j2)):
        scale = np.abs(rhs).max()
        assert np.abs(lhs - rhs).max() / scale < 1e-9, (blend, np.abs(lhs - rhs).max() / scale)


def test_deformation_limiter_keeps_the_balance_row_elliptic():
    """Where the reference deformation beats the vorticity, the limiter acts.

    A strong, narrow vortex has a flank where the deformation exceeds the
    absolute vorticity; there the symbol of the linearised balance row has a
    negative eigenvalue and the operator is indefinite.  The limiter scales the
    deformation part down until ``AVO - s w D`` keeps the margin, and leaves
    everything else alone.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    lat = ops.grid.lat[:, None]
    lon = ops.grid.lon[None, :]
    nlev = levels.nlev
    psi_grid = np.zeros((nlev, ops.grid.nlat, ops.grid.nlon))
    for k in range(nlev):
        psi_grid[k] = -2.0e7 * np.sin(np.radians(lat)) + 6.0e7 * np.exp(
            -(((lat - 45.0) / 4.0) ** 2 + ((lon - 180.0) / 6.0) ** 2)
        )
    theta = 300.0 + 8.0 * np.arange(nlev)[:, None, None] + 0.0 * lat
    phi_grid = np.zeros_like(psi_grid)
    for k in range(1, nlev):
        phi_grid[k] = phi_grid[k - 1] - 0.5 * (theta[k] + theta[k - 1]) * (
            levels.pi[k] - levels.pi[k - 1]
        )
    psi_spec = np.stack([ops.analyze(psi_grid[k]) for k in range(nlev)])
    phi_spec = np.stack([ops.analyze(phi_grid[k]) for k in range(nlev)])
    margin = 0.05
    frozen = FrozenState(
        ops,
        levels,
        psi_spec,
        phi_spec,
        clamps=ClampConfig(deformation_margin=margin),
        mirror=MirrorConfig(),
    )
    limit = frozen.deform_limit
    assert limit.min() < 1.0, "the vortex flank must trigger the limiter"
    assert np.all(limit <= 1.0) and np.all(limit > 0.0)
    far = np.abs(ops.grid.lat - 45.0) > 20.0
    assert np.all(limit[:, far, :] == 1.0), "the limiter must be local to the vortex"

    deform = np.stack(
        [frozen._deformation_magnitude(psi_spec[k]) for k in levels.interior]
    )
    w = frozen.weight[None]
    # The symbol of the limited row is AVO -/+ s w D; the limiter keeps the
    # smaller eigenvalue at least the margin times AVO.
    smallest_eigen = frozen.avo - limit * w * deform
    assert np.all(smallest_eigen >= (margin - 1e-9) * frozen.avo), (
        "the limited row must keep its ellipticity margin"
    )


def test_jacobian_is_the_derivative_of_the_residual_where_the_limiter_acts():
    """Central differences of the code's own residual against the operator.

    With the limiter frozen and shared by the residual and the Jacobian, the
    residual is exactly quadratic and the operator its derivative.  Checked on a
    state where the limiter is active, so the property is tested where it was
    previously lost, and on a state whose boundary levels are the operator's
    ghosts, which is what the nonlinear pass maintains after its first step.
    """
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels)
    state = diagnose(ops, levels, phi, temperature, u, v)
    lat = ops.grid.lat[:, None]
    lon = ops.grid.lon[None, :]
    # A saddle: strong deformation with little vorticity, so the limiter acts
    # while the absolute-vorticity floor -- a separate, non-smooth device that
    # is not under test here -- stays clear.
    x, y = (lon - 180.0) / 8.0, (lat - 45.0) / 8.0
    saddle = 0.8e8 * x * y * np.exp(-(x * x + y * y))
    psi = state.psi_spec + ops.analyze(saddle)[None]
    phi_s = state.phi_spec.copy()
    inv = BalancedInversion(ops, levels, clamps=ClampConfig(), mirror=MirrorConfig())
    theta_b, theta_t = state.theta_bot, state.theta_top
    psi, phi_s = inv._rebuild_boundaries(psi, phi_s, theta_b, theta_t)
    full = FrozenState(ops, levels, psi, phi_s)
    limiter = full.deform_limit
    assert limiter.min() < 1.0, "the saddle must activate the limiter"
    assert full.report.worst() == 0.0, "the saddle must not reach a floor"
    assert FrozenState(ops, levels, 0.5 * psi, 0.5 * phi_s).report.worst() == 0.0
    interior = levels.interior

    def rows(psi_s, phi_s_):
        half = PiecewiseOperator(
            FrozenState(ops, levels, 0.5 * psi_s, 0.5 * phi_s_, deformation_limit=limiter)
        )
        r1, r2 = half.apply(phi_s_[interior], psi_s[interior])
        b1, b2 = half.rhs_rows(state.q_hat, theta_b, theta_t)
        return np.concatenate([(r1 - b1).ravel(), (r2 - b2).ravel()])

    jac = PiecewiseOperator(FrozenState(ops, levels, psi, phi_s, deformation_limit=limiter))
    rng = np.random.default_rng(3)
    dphi, dpsi = _full_state(jac, rng, scale_phi=20.0, scale_psi=2.0e5)
    j1, j2 = jac.apply(dphi[interior], dpsi[interior])
    predicted = np.concatenate([j1.ravel(), j2.ravel()])
    eps = 1.0e-3
    finite = (rows(psi + eps * dpsi, phi_s + eps * dphi) - rows(psi - eps * dpsi, phi_s - eps * dphi)) / (2.0 * eps)
    # The floors switch with the state and are the only non-quadratic thing
    # left; a smooth state keeps them inactive, so the agreement is to rounding.
    assert np.linalg.norm(finite - predicted) / np.linalg.norm(predicted) < 1e-8


def test_piece_boundary_levels_carry_the_thermal_wind_ghost():
    """A piece's 1000 hPa streamfunction is its 850 hPa one plus the psi ghost."""
    ops = make_ops()
    levels = build_levels("NL9")
    phi, temperature, u, v, _, _ = synthetic_atmosphere(ops, levels, theta_amp=8.0)
    event = diagnose(ops, levels, phi, temperature, u, v)
    phi_m, temperature_m, u_m, v_m, _, _ = synthetic_atmosphere(ops, levels, vortex_amp=-0.8e7)
    mean = diagnose(ops, levels, phi_m, temperature_m, u_m, v_m)
    from pvinv_sph.config import InversionConfig, NewtonConfig

    result = invert_pieces(
        ops, levels, mean, event, cfg=InversionConfig(newton=NewtonConfig(max_steps=3)),
        pieces={"1000": [1], "300": [6]},
    )
    surface = result.pieces["1000"]
    f = np.broadcast_to(
        MirrorConfig().f_star(ops.grid.lat)[:, None], (ops.grid.nlat, ops.grid.nlon)
    )
    ghost = ops.analyze(
        streamfunction_ghost(event.theta_bot - mean.theta_bot, f, ops.grid.weights)
    ) * (levels.pi[1] - levels.pi[0])
    implied = surface.psi_spec[0] - surface.psi_spec[1]
    assert np.abs(implied - ghost).max() / np.abs(ghost).max() < 1e-9
    interior_piece = result.pieces["300"]
    assert np.abs(interior_piece.psi_spec[0] - interior_piece.psi_spec[1]).max() == 0.0
