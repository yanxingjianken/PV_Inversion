"""Ladder step 6: the piecewise decomposition and its closure.

The closure property is the one the whole method rests on -- if the pieces do not
add up, attributing a flow to a level means nothing.  On a window it holds only
once the boundary response is accounted for; here it should hold to the solver
tolerance, because there is no boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.config import ClampConfig, InversionConfig, KrylovConfig, MirrorConfig, NewtonConfig
from pvinv_sph.levels import build_levels
from pvinv_sph.passab import diagnose
from pvinv_sph.passd import all_sources_inversion, default_pieces, invert_pieces
from pvinv_sph.qmin import floor_pv
from pvinv_sph.sphere import SphereOps

from test_passes import make_ops, synthetic_atmosphere


def make_case(lmax=20, nlat=40, nlon=80):
    ops = make_ops(lmax=lmax, nlat=nlat, nlon=nlon)
    levels = build_levels("NL9")
    mean_fields = synthetic_atmosphere(
        ops, levels, vortex_lat=45.0, vortex_amp=-0.3e7, theta_amp=0.0
    )
    event_fields = synthetic_atmosphere(
        ops, levels, vortex_lat=45.0, vortex_amp=-1.5e7, theta_amp=4.0
    )
    mean = diagnose(ops, levels, *mean_fields[:4])
    event = diagnose(ops, levels, *event_fields[:4])
    cfg = InversionConfig(
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-11, maxiter=400),
        newton=NewtonConfig(max_steps=8),
    )
    return ops, levels, mean, event, cfg


def test_default_pieces_are_labelled_by_pressure():
    levels = build_levels("NL9")
    pieces = default_pieces(levels)
    assert list(pieces) == ["1000", "850", "700", "500", "400", "300", "250", "200", "100"]
    assert pieces["1000"] == [1] and pieces["100"] == [9]


def test_pieces_sum_to_the_all_sources_inversion():
    """Closure, with no wall term to account for first."""
    ops, levels, mean, event, cfg = make_case()
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    total, report = all_sources_inversion(ops, levels, mean, event, cfg=cfg)
    assert report.converged

    summed = result.summed_psi()
    scale = np.abs(total).max()
    assert np.abs(summed - total).max() / scale < 1e-6, "pieces do not close"


def test_every_source_is_used_exactly_once():
    """No source counted twice, none dropped -- the precondition for closure."""
    levels = build_levels("NL9")
    used = sorted(i for lst in default_pieces(levels).values() for i in lst)
    assert used == list(range(1, levels.nlev + 1))


def _profile(ops, piece, nlev):
    """Amplitude of a piece's induced streamfunction level by level, on the grid.

    Taken on the grid rather than from the spectral coefficients: the spectrum is
    dominated by its lowest degrees, so its magnitude profile says little about
    where the induced flow actually sits.
    """
    return np.array(
        [np.abs(ops.synth(piece.psi_spec[k])).max() for k in range(nlev)]
    )


def test_boundary_pieces_carry_the_boundary_temperature():
    """The surface piece responds to surface temperature and nothing else."""
    ops, levels, mean, event, cfg = make_case()
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    surface = result.pieces["1000"]
    assert np.abs(surface.psi_spec).max() > 0

    # A warm surface anomaly acts like a positive potential-vorticity sheet at the
    # boundary, so its induced flow is strongest at the bottom and weakens upward.
    # How fast it weakens is set by the Rossby depth of the anomaly, which for a
    # feature this broad is comparable to the depth of the domain -- so the test
    # is on the shape of the profile, not on a decay rate.
    strength = _profile(ops, surface, levels.nlev)
    assert strength[0] > strength[-1], strength
    assert np.all(np.diff(strength) <= 1e-9 * strength.max()), strength


def test_boundary_piece_vanishes_without_a_temperature_anomaly():
    """Two states with the same boundary temperature leave those pieces empty."""
    ops = make_ops(lmax=20, nlat=40, nlon=80)
    levels = build_levels("NL9")
    cfg = InversionConfig(
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-11, maxiter=400),
        newton=NewtonConfig(max_steps=8),
    )
    mean = diagnose(ops, levels, *synthetic_atmosphere(ops, levels, 45.0, -0.3e7)[:4])
    event = diagnose(ops, levels, *synthetic_atmosphere(ops, levels, 45.0, -1.5e7)[:4])
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    assert np.abs(result.pieces["1000"].psi_spec).max() == 0.0
    assert np.abs(result.pieces["100"].psi_spec).max() == 0.0
    assert np.abs(result.pieces["300"].psi_spec).max() > 0.0


def test_interior_piece_is_strongest_near_its_own_level():
    ops, levels, mean, event, cfg = make_case()
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    piece = result.pieces["300"]
    level_index = int(np.where(levels.p_hpa == 300.0)[0][0])
    strength = _profile(ops, piece, levels.nlev)
    assert strength.argmax() in (level_index - 1, level_index, level_index + 1), strength


def test_piece_sources_are_additive_through_overrides():
    """A scale split has to go through the source, and then it sums exactly."""
    ops, levels, mean, event, cfg = make_case()
    base = invert_pieces(ops, levels, mean, event, cfg=cfg)
    upper = base.pieces["300"]

    q_anom = _piece_source(ops, levels, mean, event, cfg, "300")
    part_a = 0.4 * q_anom
    part_b = q_anom - part_a
    split = invert_pieces(
        ops,
        levels,
        mean,
        event,
        cfg=cfg,
        pieces={"a": [6], "b": [6]},
        qp_overrides={"a": part_a, "b": part_b},
    )
    combined = split.pieces["a"].psi_spec + split.pieces["b"].psi_spec
    scale = np.abs(upper.psi_spec).max()
    assert np.abs(combined - upper.psi_spec).max() / scale < 1e-6


def _piece_source(ops, levels, mean, event, cfg, name):
    """The interior potential-vorticity anomaly one piece would be handed."""
    weights = ops.grid.weights
    q_event, _ = floor_pv(
        event.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_mean, _ = floor_pv(
        mean.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_anom = q_event - q_mean
    index = int(np.where(levels.p_hpa == float(name))[0][0])
    position = int(np.where(levels.interior == index)[0][0])
    source = np.zeros_like(q_anom)
    source[position] = q_anom[position]
    return source


def test_high_latitude_event_needs_no_special_handling():
    """The same decomposition with the anomaly over the pole.

    Nothing is masked, truncated or band-limited; the only difference is where
    the anomaly sits.  Closure and convergence have to be unaffected.
    """
    ops = make_ops(lmax=20, nlat=40, nlon=80)
    levels = build_levels("NL9")
    cfg = InversionConfig(
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-11, maxiter=400),
        newton=NewtonConfig(max_steps=8),
    )
    mean = diagnose(
        ops, levels, *synthetic_atmosphere(ops, levels, 88.0, -0.3e7)[:4]
    )
    event = diagnose(
        ops, levels, *synthetic_atmosphere(ops, levels, 88.0, -1.5e7)[:4]
    )
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    total, _ = all_sources_inversion(ops, levels, mean, event, cfg=cfg)

    for piece in result.pieces.values():
        assert piece.report.converged, f"{piece.name}: {piece.report}"
        assert np.isfinite(piece.psi_spec).all()
    assert np.abs(result.summed_psi() - total).max() / np.abs(total).max() < 1e-6

    # And the induced flow really is over the pole, not pushed away from it.
    upper = result.pieces["300"]
    k = int(np.where(levels.p_hpa == 300.0)[0][0])
    u, v = ops.rotational_wind(upper.psi_spec[k])
    speed = np.hypot(u, v)
    polar = ops.grid.lat > 80.0
    assert speed[polar].max() > 0.5 * speed.max()


def test_pv_floor_conserves_the_area_integral():
    ops, levels, _, event, _ = make_case()
    weights = ops.grid.weights
    q = event.q_hat.copy()
    # Flip the lowest tenth negative.  The correction has to come from somewhere,
    # so most of the level must stay above the floor -- push everything under it
    # and no redistribution can conserve anything, which is a property of the
    # situation rather than of the code.
    tail = q[2] <= np.percentile(q[2], 10)
    q[2] = np.where(tail, -np.abs(q[2]), q[2])

    area = np.broadcast_to(weights[:, None], q.shape[1:])
    before = float(np.sum(q[2] * area))
    floored, report = floor_pv(q, levels, weights, qmin_pvu=0.01, mode="clean")
    after = float(np.sum(floored[2] * area))
    assert 0.05 < report.fraction[2] < 0.5, report.fraction
    assert abs(after - before) / abs(before) < 1e-9


def test_parity_floor_reports_its_slack():
    """The Fortran's non-iterating guard leaves a little unconserved; say so."""
    ops, levels, _, event, _ = make_case()
    weights = ops.grid.weights
    q = event.q_hat.copy()
    tail = q[1] <= np.percentile(q[1], 10)
    q[1] = np.where(tail, -np.abs(q[1]), q[1])
    _, parity = floor_pv(q, levels, weights, qmin_pvu=0.01, mode="parity")
    _, clean = floor_pv(q, levels, weights, qmin_pvu=0.01, mode="clean")
    assert 0.05 < parity.fraction[1] < 0.5
    assert abs(clean.conservation_slack_pvu[1]) <= abs(
        parity.conservation_slack_pvu[1]
    ) + 1e-12
    assert abs(clean.conservation_slack_pvu[1]) < 1e-6 * abs(clean.added_pvu[1])
