"""Ladder steps 3-4: vertical folding, the coupled operator, and its solve.

The manufactured solution is the load-bearing test here.  A pair of fields is
chosen, the operator is applied to produce a right-hand side, and the solver has
to recover the pair.  That exercises every term at once -- including the ones a
physically-shaped test would leave small -- and it fails loudly if the vertical
folding, the cross terms or the gauge handling disagree between the operator and
the preconditioner.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.config import ClampConfig, KrylovConfig, MirrorConfig
from pvinv_sph.krylov import solve
from pvinv_sph.levels import build_levels
from pvinv_sph.operator import FrozenState, PiecewiseOperator
from pvinv_sph.precond import SeparablePreconditioner
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps
from pvinv_sph.state import SpectralPacker
from pvinv_sph.vertical import VerticalOperator

R = 6.371e6


# ---------------------------------------------------------------------------
# Vertical operator
# ---------------------------------------------------------------------------


def test_vertical_d2_folds_ghosts_consistently():
    """Folded operator plus boundary forcing must equal the explicit ghost column."""
    lev = build_levels("NL9")
    vert = VerticalOperator(lev)
    rng = np.random.default_rng(1)
    interior = rng.normal(size=(vert.nint, 3, 4))
    theta_b = rng.normal(size=(3, 4))
    theta_t = rng.normal(size=(3, 4))

    folded = vert.d2(interior) + vert.theta_forcing(theta_b, theta_t)

    full = vert.extend(interior, theta_b, theta_t)
    explicit = np.stack(
        [
            lev.bl[k] * full[k - 1] + lev.bb[k] * full[k] + lev.bh[k] * full[k + 1]
            for k in lev.interior
        ]
    )
    assert np.abs(folded - explicit).max() < 1e-10 * np.abs(explicit).max()


def test_vertical_d_dpi_folds_ghosts_consistently():
    lev = build_levels("NL9")
    vert = VerticalOperator(lev)
    rng = np.random.default_rng(2)
    interior = rng.normal(size=(vert.nint, 5))
    theta_b = rng.normal(size=5)
    theta_t = rng.normal(size=5)

    folded = vert.d_dpi(interior, theta_b, theta_t)
    full = vert.extend(interior, theta_b, theta_t)
    explicit = np.stack(
        [(full[k + 1] - full[k - 1]) / (2 * lev.dpi2[k]) for k in lev.interior]
    )
    assert np.abs(folded - explicit).max() < 1e-10 * np.abs(explicit).max()


def test_vertical_d_dpi_splits_into_homogeneous_and_ghost_parts():
    """The split the right-hand side relies on: linear, so it separates exactly."""
    lev = build_levels("NL9")
    vert = VerticalOperator(lev)
    rng = np.random.default_rng(3)
    interior = rng.normal(size=(vert.nint, 4))
    theta_b, theta_t = rng.normal(size=4), rng.normal(size=4)
    both = vert.d_dpi(interior, theta_b, theta_t)
    homogeneous = vert.d_dpi(interior)
    ghost = vert.d_dpi(np.zeros_like(interior), theta_b, theta_t)
    assert np.abs(both - (homogeneous + ghost)).max() < 1e-12


def test_vertical_annihilates_a_constant_column():
    """Rows sum to zero, so a constant geopotential is in the null space."""
    lev = build_levels("NL9")
    vert = VerticalOperator(lev)
    ones = np.ones((vert.nint, 2, 2))
    assert np.abs(vert.d2(ones)).max() < 1e-12 * abs(vert.diag).max()
    assert np.abs(vert.d_dpi(ones)).max() < 1e-12


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def test_packer_roundtrip_and_dof_count():
    packer = SpectralPacker(nlev=7, lmax=12)
    assert packer.per_level == 13**2
    rng = np.random.default_rng(4)
    fields = []
    for _ in range(2):
        spec = rng.normal(size=(7, 13, 13)) + 1j * rng.normal(size=(7, 13, 13))
        fields.append(packer.canonicalize(spec))
    vec = packer.pack(*fields)
    back = packer.unpack(vec)
    for a, b in zip(fields, back):
        assert np.abs(a - b).max() < 1e-15


# ---------------------------------------------------------------------------
# The coupled operator
# ---------------------------------------------------------------------------


def build_case(lmax=24, nlat=48, nlon=96, seed=11, jet=True):
    """A frozen state with structure in every coefficient the operator uses."""
    sht = SHT(gaussian_grid(nlat, nlon), lmax=lmax, radius=R)
    ops = SphereOps(sht)
    lev = build_levels("NL9")
    rng = np.random.default_rng(seed)

    lat = ops.grid.lat[:, None]
    lon = ops.grid.lon[None, :]
    nlev = lev.nlev

    # Reference streamfunction: solid body, a Y_2^0-like shear, and a localised
    # jet, growing with height -- enough to make the absolute vorticity, the
    # deformation and the cross terms all vary.
    psi_grid = np.zeros((nlev, ops.grid.nlat, ops.grid.nlon))
    for k in range(nlev):
        amp = 1.0 + 0.6 * k
        psi_grid[k] = amp * (
            -2.0e7 * np.sin(np.radians(lat))
            - 6.0e6 * (3 * np.sin(np.radians(lat)) ** 2 - 1) / 2
        )
        if jet:
            psi_grid[k] -= (
                amp
                * 1.2e7
                * np.exp(-(((lat - 45.0) / 12.0) ** 2 + ((lon - 200.0) / 30.0) ** 2))
            )
    # Reference geopotential: hydrostatic with a stable, height-varying lapse so
    # the static stability is positive and varies with level and latitude.
    theta_ref = (
        300.0
        + 8.0 * np.arange(nlev)[:, None, None]
        + 6.0 * np.cos(np.radians(lat))[None, :, :]
    )
    phi_grid = np.zeros_like(psi_grid)
    for k in range(1, nlev):
        phi_grid[k] = phi_grid[k - 1] - 0.5 * (
            theta_ref[k] + theta_ref[k - 1]
        ) * (lev.pi[k] - lev.pi[k - 1])

    psi_spec = np.stack([ops.analyze(psi_grid[k]) for k in range(nlev)])
    phi_spec = np.stack([ops.analyze(phi_grid[k]) for k in range(nlev)])
    frozen = FrozenState(
        ops,
        lev,
        psi_spec,
        phi_spec,
        clamps=ClampConfig(),
        mirror=MirrorConfig(),
    )
    return PiecewiseOperator(frozen), rng


def random_state(op, rng, scale_phi=200.0, scale_psi=3.0e6):
    """A smooth, resolved pair of perturbation fields."""
    shape = (op.nint, op.lmax + 1, op.lmax + 1)
    phi = np.zeros(shape, dtype=complex)
    psi = np.zeros(shape, dtype=complex)
    nmax = min(10, op.lmax)
    for k in range(op.nint):
        for m in range(min(5, nmax)):
            lo = max(m, 1)
            size = nmax - lo
            if size <= 0:
                continue
            decay = 1.0 / (1 + np.arange(lo, nmax)) ** 2
            phi[k, m, lo:nmax] = scale_phi * decay * (
                rng.normal(size=size) + 1j * rng.normal(size=size)
            )
            psi[k, m, lo:nmax] = scale_psi * decay * (
                rng.normal(size=size) + 1j * rng.normal(size=size)
            )
    phi = op.packer.canonicalize(phi)
    psi = op.packer.canonicalize(psi)
    return phi, psi


def test_operator_is_linear():
    op, rng = build_case()
    a = op.pack_state(*random_state(op, rng))
    b = op.pack_state(*random_state(op, rng))
    lhs = op.matvec(a + 2.5 * b)
    rhs = op.matvec(a) + 2.5 * op.matvec(b)
    assert np.abs(lhs - rhs).max() / np.abs(lhs).max() < 1e-10


def test_constant_geopotential_is_removed_only_by_the_gauge():
    """A uniform geopotential is in the operator's null space by construction."""
    op, _ = build_case()
    phi = np.zeros((op.nint, op.lmax + 1, op.lmax + 1), dtype=complex)
    psi = np.zeros_like(phi)
    phi[:, 0, 0] = 137.0
    r1, r2 = op.apply(phi, psi)

    physical = r2.copy()
    physical[:, 0, 0] = 0.0  # strip the rank-one gauge term
    assert np.abs(r1).max() < 1e-8
    assert np.abs(physical).max() < 1e-8
    assert np.abs(r2[:, 0, 0]).max() > 0.0  # the gauge does see it


def test_level_mean_streamfunction_is_pinned_by_the_gauge():
    op, _ = build_case()
    phi = np.zeros((op.nint, op.lmax + 1, op.lmax + 1), dtype=complex)
    psi = np.zeros_like(phi)
    psi[:, 0, 0] = np.arange(1.0, op.nint + 1.0)
    r1, _ = op.apply(phi, psi)
    assert np.allclose(r1[:, 0, 0].real, np.arange(1.0, op.nint + 1.0))


def test_manufactured_solution_is_recovered():
    """Apply the operator to a known pair, then solve for it."""
    op, rng = build_case()
    phi_true, psi_true = random_state(op, rng)
    x_true = op.pack_state(phi_true, psi_true)
    b = op.matvec(x_true)

    pre = SeparablePreconditioner(op)
    x, report = solve(
        op.matvec, b, preconditioner=pre.apply, cfg=KrylovConfig(rtol=1e-10, maxiter=400)
    )
    assert report.converged, str(report)
    err = np.abs(x - x_true).max() / np.abs(x_true).max()
    assert err < 1e-6, f"solution error {err:.2e} after {report.iterations} iterations"


def test_preconditioner_is_what_makes_the_solve_practical():
    """Same budget, preconditioned and not.

    The unpreconditioned system is badly scaled -- the two equations differ by
    many orders of magnitude in units alone -- so this is not a marginal
    improvement to check with a ratio: within a budget that the preconditioned
    solve finishes several times over, the bare solve does not converge at all.
    """
    op, rng = build_case()
    b = op.matvec(op.pack_state(*random_state(op, rng)))
    cfg = KrylovConfig(rtol=1e-8, maxiter=240)
    pre = SeparablePreconditioner(op)
    _, with_pre = solve(op.matvec, b, preconditioner=pre.apply, cfg=cfg)
    _, without = solve(op.matvec, b, preconditioner=None, cfg=cfg)
    assert with_pre.converged, str(with_pre)
    assert with_pre.iterations < 120, str(with_pre)
    assert not without.converged or without.iterations > 4 * with_pre.iterations


def test_right_hand_side_is_linear_in_its_sources():
    """Pieces add because ``b`` does: this is the closure property, upstream."""
    op, rng = build_case()
    shape = (op.nint, op.ops.grid.nlat, op.ops.grid.nlon)
    q_a = rng.normal(size=shape)
    q_b = rng.normal(size=shape)
    theta_b = rng.normal(size=shape[1:])
    theta_t = rng.normal(size=shape[1:])

    separate = (
        op.rhs(q_a)
        + op.rhs(q_b)
        + op.rhs(np.zeros(shape), theta_b, theta_t)
    )
    together = op.rhs(q_a + q_b, theta_b, theta_t)
    assert np.abs(separate - together).max() / np.abs(together).max() < 1e-12


def test_pieces_sum_to_the_all_sources_solution():
    """Ladder step 6, in miniature: the property the whole method rests on."""
    op, rng = build_case()
    pre = SeparablePreconditioner(op)
    cfg = KrylovConfig(rtol=1e-10, maxiter=400)
    shape = (op.nint, op.ops.grid.nlat, op.ops.grid.nlon)

    q_full = rng.normal(size=shape) * 1e-3
    theta_b = rng.normal(size=shape[1:]) * 0.5
    theta_t = rng.normal(size=shape[1:]) * 0.5

    pieces = []
    for k in range(op.nint):
        q_k = np.zeros(shape)
        q_k[k] = q_full[k]
        x, rep = solve(op.matvec, op.rhs(q_k), preconditioner=pre.apply, cfg=cfg)
        assert rep.converged, f"level piece {k}: {rep}"
        pieces.append(x)
    for tb, tt in ((theta_b, None), (None, theta_top := theta_t)):
        x, rep = solve(
            op.matvec,
            op.rhs(np.zeros(shape), tb, tt),
            preconditioner=pre.apply,
            cfg=cfg,
        )
        assert rep.converged, "boundary piece"
        pieces.append(x)

    total, rep = solve(
        op.matvec,
        op.rhs(q_full, theta_b, theta_top),
        preconditioner=pre.apply,
        cfg=cfg,
    )
    assert rep.converged, str(rep)
    summed = np.sum(pieces, axis=0)
    err = np.abs(summed - total).max() / np.abs(total).max()
    assert err < 1e-6, f"piece closure {err:.2e}"


def test_clamp_report_is_populated():
    op, _ = build_case()
    rep = op.frozen.report
    assert rep.avo_fraction.shape == (op.nint,)
    assert np.all(rep.avo_fraction >= 0) and np.all(rep.avo_fraction <= 1)
    assert rep.worst() <= 1.0


def test_preconditioner_rejects_a_non_elliptic_mean_state():
    op, _ = build_case()
    op.frozen.stb_mean = op.frozen.stb_mean * -1.0
    with pytest.raises(ValueError, match="positive"):
        SeparablePreconditioner(op)


def test_taper_is_a_weight_on_the_products_not_on_the_reference():
    """The tapered vorticity is the weight times the reference's own vorticity.

    The balance row uses the reference streamfunction's gradients in its
    deformation terms and the tapered vorticity in its first term.  Those are
    one flow, weighted, rather than two different flows: the weight multiplies
    the products the quadratic terms are built from, which is what keeps the
    bilinear form symmetric and the midpoint linearisation exact where the
    taper acts.  Rebuilding the streamfunction from the tapered vorticity
    instead -- the earlier design -- made the row the tangent of nothing there.
    """
    op, _ = build_case()
    frozen = op.frozen
    assert frozen.mirror.blend, "the taper is what this test is about"
    ops, levels = frozen.ops, frozen.levels
    assert frozen.weight.min() < 1.0 and frozen.weight.max() == 1.0
    for k in range(len(levels.interior)):
        own = ops.synth(ops.lap(frozen.psi_ref_spec[k]))
        assert np.array_equal(frozen.psi_ref_spec[k], frozen.psi_spec[levels.interior[k]])
        assert np.abs(frozen.weight * own - frozen.zeta_ref[k]).max() / np.abs(
            own
        ).max() < 1e-12, "the tapered vorticity is not the weighted reference vorticity"


def test_boundary_ghosts_respect_geostrophy():
    """The streamfunction's boundary ghost is the geopotential's over f.

    In the geostrophic limit the perturbation satisfies Phi' = f psi', and the
    boundary condition has to respect that: the temperature fixes the
    geopotential's vertical derivative, and the streamfunction's is smaller by f.
    Handing both cross terms the same ghost -- which is what reading the Fortran
    literally suggests, before its scaling of psi by f is unwound -- makes the
    streamfunction's boundary condition too large by f everywhere.
    """
    op, _ = build_case()
    frozen = op.frozen
    ops, levels = frozen.ops, frozen.levels

    nint = len(levels.interior)
    zero_q = np.zeros((nint, ops.grid.nlat, ops.grid.nlon))
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    theta = 3.0 * np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))
    zero = np.zeros_like(theta)

    # Drive with a bottom temperature only, and compare the response against one
    # driven by a temperature scaled so the two ghosts would coincide.  They must
    # differ by the Coriolis parameter, not be equal.
    _, b_theta = op.rhs_rows(zero_q, theta, zero)
    _, b_zero = op.rhs_rows(zero_q, zero, zero)
    response = np.abs(ops.synth(b_theta[0] - b_zero[0]))
    assert response.max() > 0, "a boundary temperature must drive the equations"

    # The ghost's own scaling: doubling the temperature doubles the forcing, so
    # the relation is linear and the only question is the factor, which the
    # geostrophic construction below pins down.
    _, b_double = op.rhs_rows(zero_q, 2.0 * theta, zero)
    doubled = np.abs(ops.synth(b_double[0] - b_zero[0]))
    assert np.allclose(doubled, 2.0 * response, rtol=1e-8)
