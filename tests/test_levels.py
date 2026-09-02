"""Ladder step 3a: level tables, Exner stencil and the PV unit conversion."""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.levels import (
    CP,
    G,
    KAPPA,
    P0,
    build_levels,
    exner,
    pv_rhs_scale,
)


def test_presets():
    nl9 = build_levels("NL9")
    nl10 = build_levels("NL10")
    assert nl9.nlev == 9 and nl10.nlev == 10
    assert nl9.p_hpa[0] == 1000.0 and nl9.p_hpa[-1] == 100.0
    assert 150.0 in set(nl10.p_hpa) and 150.0 not in set(nl9.p_hpa)
    assert np.array_equal(nl9.interior, np.arange(1, 8))
    with pytest.raises(ValueError, match="unknown level preset"):
        build_levels("NL7")


def test_exner_matches_definition():
    p = np.array([1000.0, 500.0, 100.0])
    assert np.allclose(exner(p), CP * (p * 100.0 / P0) ** KAPPA)
    assert np.isclose(exner(np.array([1000.0]))[0], CP)


def test_vertical_stencil_differentiates_exactly():
    """BB/BH/BL must reproduce d2/dPi2 exactly for a quadratic in Pi."""
    lev = build_levels("NL9")
    pi = lev.pi
    f = 3.0 - 1.7 * pi + 0.45 * pi**2  # second derivative is 0.90 everywhere
    for k in lev.interior:
        d2 = lev.bl[k] * f[k - 1] + lev.bb[k] * f[k] + lev.bh[k] * f[k + 1]
        assert np.isclose(d2, 0.90, rtol=1e-10)


def test_vertical_stencil_rows_sum_to_zero():
    """A pure second derivative annihilates constants -- no zeroth-order part.

    This is the property that made the earlier explicit-lagging solver fragile:
    all vertical coupling sits off the diagonal, so nothing damps it.  The new
    solver treats the whole column implicitly, but the property is still what the
    preconditioner's tridiagonal is built from, so assert it.
    """
    lev = build_levels("NL9")
    for k in lev.interior:
        assert abs(lev.bl[k] + lev.bb[k] + lev.bh[k]) < 1e-12 * abs(lev.bb[k])
    assert np.all(lev.bb[lev.interior] < 0)
    assert np.all(lev.bh[lev.interior] > 0)
    assert np.all(lev.bl[lev.interior] > 0)


def test_dpi2_folds_boundary_theta():
    """``bl[k] * (Pi_k - Pi_{k-1}) == 1/dpi2[k]`` -- the theta-folding identity.

    Substituting the hydrostatic ghost level into the ``k = 1`` row turns the
    boundary potential temperature into a forcing divided by ``dpi2``; this is
    that step's algebra, and it is where a stray factor would silently rescale
    the surface piece.
    """
    lev = build_levels("NL9")
    for k in lev.interior:
        assert np.isclose(lev.bl[k] * (lev.pi[k] - lev.pi[k - 1]), 1.0 / lev.dpi2[k])
        assert np.isclose(lev.bh[k] * (lev.pi[k + 1] - lev.pi[k]), 1.0 / lev.dpi2[k])


def test_pv_rhs_scale_matches_derivation():
    lev = build_levels("NL9")
    scale = pv_rhs_scale(lev.p_hpa)
    p_pa = lev.p_hpa * 100.0
    assert np.allclose(scale, p_pa / (G * KAPPA * lev.pi))
    # kappa = 2/7 makes the two exponent forms coincide; if that ever changes the
    # physical form above is the one to keep.
    assert np.allclose(scale, (p_pa / P0) ** (1 - KAPPA) * P0 / (KAPPA * G * CP))
    assert np.allclose(scale, (p_pa / P0) ** (2.5 * KAPPA) * P0 / (KAPPA * G * CP))
    assert np.isclose(scale[0], P0 / (KAPPA * G * CP))
    assert np.all(np.diff(scale) < 0)  # decreasing upward
