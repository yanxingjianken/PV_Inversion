"""The reusable engine and the hemisphere-level output.

A batch builds the spectral tables once and hands the same engine to every
event; the result has to be the one a fresh call produces, to the bit, and the
pieces on the data's own grid have to be the values the cropped patch holds.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.config import InversionConfig, KrylovConfig, MirrorConfig, NewtonConfig
from pvinv_sph.levels import G, RD, build_levels
from pvinv_sph.pipeline import SphereEngine, invert_event, invert_hemisphere

LAT_NH = np.linspace(0.9375, 89.0625, 48)
LON = np.arange(64) * (360.0 / 64)


def _fields():
    levels = build_levels("NL9")
    lon2d, lat2d = np.meshgrid(LON, LAT_NH)
    nlev = levels.nlev
    temperature = np.stack(
        [288.0 - 6.5 * k + 4.0 * np.cos(np.radians(lat2d)) for k in range(nlev)]
    )
    height = np.empty_like(temperature)
    height[0] = 100.0 + 150.0 * np.cos(np.radians(lat2d)) + 60.0 * np.sin(np.radians(lon2d))
    for k in range(1, nlev):
        t_bar = 0.5 * (temperature[k] + temperature[k - 1])
        height[k] = height[k - 1] + (RD * t_bar / G) * np.log(
            levels.p_hpa[k - 1] / levels.p_hpa[k]
        )
    u = np.stack([(6.0 + 2.0 * k) * np.cos(np.radians(lat2d)) for k in range(nlev)])
    v = np.zeros_like(u)
    mean = (height, temperature, u, v)
    event = (height, temperature * 1.01, u * 1.1, v)
    return mean, event


def _cfg():
    return InversionConfig(
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-8, maxiter=200),
        newton=NewtonConfig(max_steps=3),
    )


def _same(a: dict, b: dict) -> list[str]:
    assert set(a) == set(b)
    return [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]


def test_engine_reuse_is_byte_identical_to_a_fresh_call():
    mean, event = _fields()
    cfg = _cfg()
    centre = (45.0, 100.0)
    fresh = invert_event(
        mean, event, LAT_NH, LON, centre, cfg=cfg, lat_half=30.0, lon_half=30.0,
        solver_nlat=32, solver_nlon=64, lmax=20,
    )
    engine = SphereEngine.build(LAT_NH, LON, cfg=cfg, solver_nlat=32, solver_nlon=64, lmax=20)
    first = invert_event(
        mean, event, LAT_NH, LON, centre, engine=engine, lat_half=30.0, lon_half=30.0
    )
    second = invert_event(
        mean, event, LAT_NH, LON, (60.0, 200.0), engine=engine, lat_half=30.0, lon_half=30.0
    )
    third = invert_event(
        mean, event, LAT_NH, LON, centre, engine=engine, lat_half=30.0, lon_half=30.0
    )
    assert _same(fresh.arrays, first.arrays) == []
    assert _same(first.arrays, third.arrays) == []
    assert fresh.meta["newton_steps"] == first.meta["newton_steps"]
    assert second.meta["centre_lat"] == 60.0


def test_hemisphere_rows_are_what_the_patch_holds():
    mean, event = _fields()
    engine = SphereEngine.build(LAT_NH, LON, cfg=_cfg(), solver_nlat=32, solver_nlon=64, lmax=20)
    centre = (70.0, 350.0)
    hi = invert_hemisphere(engine, mean, event, centre, pieces_mode="per_level")
    out = invert_event(
        mean, event, LAT_NH, LON, centre, engine=engine, lat_half=30.0, lon_half=30.0
    )
    lat_vec = out.arrays["lat_vec"]
    lon_vec = out.arrays["lon_vec"]
    assert np.isnan(lat_vec).any(), "a centre at 70N with a 30-degree box passes the pole"
    for name in ("500", "1000", "100"):
        u_nh = hi.northern(hi.piece_u[name])
        assert u_nh.shape == (engine.levels.nlev, LAT_NH.size, LON.size)
        patch = out.arrays[f"u_rot_anom_ppvi_{name}_3d"]
        for j, plat in enumerate(lat_vec):
            if not np.isfinite(plat):
                assert np.isnan(patch[:, j, :]).all()
                continue
            r = int(np.argmin(np.abs(LAT_NH - plat)))
            for i, plon in enumerate(lon_vec):
                c = int(np.argmin(np.abs((LON - plon + 180.0) % 360.0 - 180.0)))
                np.testing.assert_array_equal(
                    patch[:, j, i], u_nh[:, r, c].astype(np.float32)
                )
    # The residual is the observed anomaly minus the pieces, in the order solved.
    summed = np.zeros_like(hi.u_obs)
    for name in hi.piece_names():
        summed += hi.piece_u[name]
    np.testing.assert_array_equal(hi.u_residual, hi.u_obs - summed)
    assert hi.meta["piece_names"] == hi.piece_names()


def test_engine_refuses_other_axes_and_conflicting_settings():
    mean, event = _fields()
    engine = SphereEngine.build(LAT_NH, LON, cfg=_cfg(), solver_nlat=32, solver_nlon=64, lmax=20)
    assert engine.fits(LAT_NH, LON)
    assert not engine.fits(LAT_NH[:-1], LON[:-1])
    with pytest.raises(ValueError, match="other latitude"):
        invert_event(mean, event, LAT_NH * 0.5, LON, (45.0, 0.0), engine=engine)
    with pytest.raises(ValueError, match="cfg was given"):
        invert_event(mean, event, LAT_NH, LON, (45.0, 0.0), cfg=_cfg(), engine=engine)
    with pytest.raises(ValueError, match="lmax"):
        invert_event(mean, event, LAT_NH, LON, (45.0, 0.0), engine=engine, lmax=15)
