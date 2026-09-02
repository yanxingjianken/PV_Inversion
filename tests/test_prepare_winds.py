"""State preparation and the two croppings.

The mirror is checked by the symmetry it is supposed to produce, exactly, rather
than by inspecting the code path that produces it; and the rotated frame is
checked against the geographic one where the two must agree, which is the only
place a rotation convention can hide a sign.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.levels import build_levels
from pvinv_sph.mirror import mirror_even, mirror_odd, mirrored_latitudes, restrict_to_nh
from pvinv_sph.prepare import (
    fill_below_ground,
    prepare_state,
    solver_grid_is_symmetric,
    symmetrize_even,
)
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps
from pvinv_sph.winds import (
    frame_rotation_angle,
    geographic_patch,
    rotate_to_pole,
    rotated_patch,
)

CESM_LAT_NH = np.linspace(0.47120418848167, 90.0, 96)
CESM_LON = np.arange(288) * (360.0 / 288)


def hemisphere_fields(nlev=9, lat_nh=CESM_LAT_NH, lon=CESM_LON):
    """A smooth hemisphere of data whose height and temperature are hydrostatic.

    The pipeline refuses a pair that is not, as real data always are to within a
    few percent, so the height is integrated from the temperature rather than
    written down independently.
    """
    from pvinv_sph.levels import G, RD, build_levels

    lon2d, lat2d = np.meshgrid(lon, lat_nh)
    temperature = np.stack(
        [290.0 - 6.0 * k + 5.0 * np.cos(np.radians(lat2d)) for k in range(nlev)]
    )
    p_hpa = (
        build_levels("NL9").p_hpa
        if nlev == 9
        else 1000.0 * (0.1 ** (np.arange(nlev) / (nlev - 1)))
    )
    height = np.empty_like(temperature)
    height[0] = 100.0 + 200.0 * np.cos(np.radians(lat2d)) + 80.0 * np.sin(np.radians(lon2d))
    for k in range(1, nlev):
        t_bar = 0.5 * (temperature[k] + temperature[k - 1])
        height[k] = height[k - 1] + (RD * t_bar / G) * np.log(p_hpa[k - 1] / p_hpa[k])
    u = np.stack([(5.0 + 3.0 * k) * np.cos(np.radians(lat2d)) for k in range(nlev)])
    v = np.stack(
        [
            2.0 * np.sin(np.radians(2 * lon2d)) * np.cos(np.radians(lat2d))
            for _ in range(nlev)
        ]
    )
    return height, temperature, u, v


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------



def tiny_inversion(centre, rotated_track=False, lat_half=30.0, lon_half=30.0):
    """One inversion of a smooth synthetic state, small enough to run in a test.

    The state is a zonal jet with a little longitudinal structure: enough for the
    balance to have something to do, smooth enough that a truncation of twenty
    resolves it, and cheap enough that a test can afford several.

    Returns the output, the data's own latitude axis, and its longitudes -- the
    axis matters to callers because everything south of its first row on the
    output grid is the mirror's invention rather than data.
    """
    from pvinv_sph.config import (
        InversionConfig,
        KrylovConfig,
        MirrorConfig,
        NewtonConfig,
    )
    from pvinv_sph.levels import build_levels
    from pvinv_sph.pipeline import invert_event

    levels = build_levels("NL9")
    lat_nh = np.linspace(0.9375, 89.0625, 48)
    lon = np.arange(64) * (360.0 / 64)
    lon2d, lat2d = np.meshgrid(lon, lat_nh)
    from pvinv_sph.levels import G, RD

    nlev = levels.nlev
    temperature = np.stack(
        [288.0 - 6.5 * k + 4.0 * np.cos(np.radians(lat2d)) for k in range(nlev)]
    )
    # Hydrostatic height, as real data are and as the pipeline requires.
    height = np.empty_like(temperature)
    height[0] = 100.0 + 150.0 * np.cos(np.radians(lat2d)) + 60.0 * np.sin(np.radians(lon2d))
    for k in range(1, nlev):
        t_bar = 0.5 * (temperature[k] + temperature[k - 1])
        height[k] = height[k - 1] + (RD * t_bar / G) * np.log(
            levels.p_hpa[k - 1] / levels.p_hpa[k]
        )
    u = np.stack([(6.0 + 2.0 * k) * np.cos(np.radians(lat2d)) for k in range(nlev)])
    v = np.zeros_like(u)
    cfg = InversionConfig(
        mirror=MirrorConfig(blend=False),
        krylov=KrylovConfig(rtol=1e-8, maxiter=200),
        newton=NewtonConfig(max_steps=3),
    )
    out = invert_event(
        (height, temperature, u, v),
        (height, temperature * 1.01, u * 1.1, v),
        lat_nh,
        lon,
        centre=centre,
        cfg=cfg,
        lat_half=lat_half,
        lon_half=lon_half,
        solver_nlat=32,
        solver_nlon=64,
        lmax=20,
        rotated_track=rotated_track,
    )
    return out, lat_nh, lon


def rotated_geographic_latitudes(centre, lat_half=30.0, lon_half=30.0, step=1.0):
    """Where each point of a rotated crop actually landed, in geographic latitude.

    The crop stores its own frame's coordinates, not these, so a test that wants
    to know which points came from the mirror has to rebuild them the way the crop
    did.
    """
    from pvinv_sph.winds import rotate_to_pole

    lat_rel = np.arange(-lat_half, lat_half + 0.5 * step, step)
    lon_rel = np.arange(-lon_half, lon_half + 0.5 * step, step)
    lon_mesh, lat_mesh = np.meshgrid(lon_rel, lat_rel)
    lat_geo, _ = rotate_to_pole(lat_mesh, lon_mesh, centre[0], centre[1])
    return lat_geo


def test_mirror_parities_round_trip():
    lat_nh = CESM_LAT_NH
    rng = np.random.default_rng(0)
    field = rng.normal(size=(3, lat_nh.size, 8))
    even = mirror_even(field, lat_nh)
    odd = mirror_odd(field, lat_nh)
    assert even.shape[-2] == 2 * lat_nh.size
    assert np.allclose(restrict_to_nh(even, lat_nh), field)
    assert np.allclose(restrict_to_nh(odd, lat_nh), field)
    n = lat_nh.size
    assert np.allclose(even[..., :n, :], field[..., ::-1, :])
    assert np.allclose(odd[..., :n, :], -field[..., ::-1, :])
    assert np.allclose(mirrored_latitudes(lat_nh), np.concatenate([-lat_nh[::-1], lat_nh]))


def test_pole_centred_layout_zeroes_the_equator_for_odd_fields():
    lat_nh = np.arange(0.0, 90.1, 1.5)
    field = np.ones((2, lat_nh.size, 4))
    odd = mirror_odd(field, lat_nh)
    assert odd.shape[-2] == 2 * lat_nh.size - 1
    assert np.all(odd[..., lat_nh.size - 1, :] == 0.0)


def test_a_band_that_is_not_a_hemisphere_is_refused():
    lat = np.arange(20.0, 80.1, 1.5)
    with pytest.raises(ValueError, match="not a hemisphere"):
        mirror_even(np.zeros((1, lat.size, 4)), lat)


def test_below_ground_fill_is_a_no_op_without_gaps():
    height, temperature, u, v = hemisphere_fields()
    out = fill_below_ground(height, temperature, u, v, build_levels("NL9").p_hpa)
    assert out[0] is height and out[1] is temperature


def test_below_ground_fill_is_hydrostatic():
    levels = build_levels("NL9")
    height, temperature, u, v = hemisphere_fields()
    height = height.copy()
    height[0, :10, :5] = np.nan
    temperature = temperature.copy()
    temperature[0, :10, :5] = np.nan
    filled_z, filled_t, _, _ = fill_below_ground(
        height, temperature, u, v, levels.p_hpa
    )
    assert np.isfinite(filled_z).all()
    # Continued downward, so the filled level sits below the one above it.
    assert np.all(filled_z[0, :10, :5] < filled_z[1, :10, :5])
    assert np.allclose(filled_t[0, :10, :5], filled_t[1, :10, :5])


# ---------------------------------------------------------------------------
# State preparation
# ---------------------------------------------------------------------------


def test_prepared_state_is_exactly_even():
    """The property the whole scaffold argument rests on."""
    ops = SphereOps(SHT(gaussian_grid(64, 128), lmax=40))
    levels = build_levels("NL9")
    assert solver_grid_is_symmetric(ops.grid)
    state = prepare_state(*hemisphere_fields(), CESM_LAT_NH, CESM_LON, levels, ops)

    half = ops.grid.nlat // 2
    for name, field in (
        ("vorticity", state.zeta),
        ("theta", state.theta),
        ("q_hat", state.q_hat),
    ):
        north = field[..., half:, :]
        south = field[..., :half, :][..., ::-1, :]
        scale = max(np.abs(field).max(), 1e-300)
        assert np.abs(north - south).max() / scale < 1e-12, name

    psi = np.stack([ops.synth(state.psi_spec[k]) for k in range(levels.nlev)])
    north, south = psi[:, half:, :], psi[:, :half, :][:, ::-1, :]
    assert np.abs(north - south).max() / np.abs(psi).max() < 1e-12


def test_prepared_state_keeps_the_hemisphere_it_was_given():
    """The northern half must still be the data, not an average with a mirror."""
    ops = SphereOps(SHT(gaussian_grid(96, 192), lmax=60))
    levels = build_levels("NL9")
    height, temperature, u, v = hemisphere_fields()
    state = prepare_state(height, temperature, u, v, CESM_LAT_NH, CESM_LON, levels, ops)

    half = ops.grid.nlat // 2
    lat_solver = ops.grid.lat[half:]
    band = (lat_solver > 20.0) & (lat_solver < 80.0)
    theta_north = state.theta[:, half:, :][:, band, :]
    # Compare against the analytic potential temperature of the input.
    lon2d, lat2d = np.meshgrid(ops.grid.lon, lat_solver[band])
    expected_t = np.stack(
        [290.0 - 6.0 * k + 5.0 * np.cos(np.radians(lat2d)) for k in range(levels.nlev)]
    )
    expected_theta = expected_t * (
        1.0e5 / (levels.p_hpa[:, None, None] * 100.0)
    ) ** (2.0 / 7.0)
    assert np.abs(theta_north - expected_theta).max() / expected_theta.max() < 1e-3


def test_symmetrize_even_needs_paired_rows():
    with pytest.raises(ValueError, match="even number"):
        symmetrize_even(np.zeros((3, 5, 4)))


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def test_geographic_patch_wraps_longitude_and_nans_past_the_pole():
    lat = np.arange(-90.0, 90.1, 1.5)
    lon = np.arange(0.0, 360.0, 1.5)
    field = np.broadcast_to(lat[:, None], (lat.size, lon.size)).copy()

    patch = geographic_patch(field, lat, lon, 45.0, 4.5, lat_half=15.0, lon_half=30.0)
    assert patch.values.shape == (21, 41)
    assert np.isfinite(patch.values).all()
    assert np.isclose(patch.values[10, 20], 45.0)  # centre holds the centre value
    assert patch.lon[0] == pytest.approx(334.5)  # wrapped westward past zero

    polar = geographic_patch(field, lat, lon, 84.0, 0.0, lat_half=15.0, lon_half=30.0)
    # Past 90 N is a real place -- 90 + d north is 90 - d on the opposite
    # meridian -- but a box with one longitude per column cannot label it.
    assert np.isnan(polar.values[-1]).all()
    assert np.isfinite(polar.values[0]).all()


def test_rotation_is_identity_for_a_centre_at_the_origin():
    """A frame centred where the equator meets the prime meridian changes nothing.

    Note that a frame centred at the *pole* is not the identity: its equator then
    runs through the pole, which is exactly the point of the construction.
    """
    lat_q = np.array([[10.0, 40.0], [-25.0, 80.0]])
    lon_q = np.array([[0.0, 90.0], [200.0, 355.0]])
    lat_geo, lon_geo = rotate_to_pole(lat_q, lon_q, 0.0, 0.0)
    assert np.allclose(lat_geo, lat_q)
    assert np.allclose(lon_geo % 360.0, lon_q % 360.0)
    assert np.allclose(frame_rotation_angle(lat_geo, lon_geo, 0.0, 0.0), 0.0, atol=1e-9)


def test_rotated_north_points_along_the_meridian_through_the_centre():
    """Moving north in the rotated frame moves north along the centre's meridian."""
    lat0, lon0 = 40.0, 210.0
    lat_geo, lon_geo = rotate_to_pole(np.array([5.0]), np.array([0.0]), lat0, lon0)
    assert lat_geo[0] == pytest.approx(lat0 + 5.0, abs=1e-9)
    assert lon_geo[0] == pytest.approx(lon0, abs=1e-9)


def test_rotated_frame_centre_lands_on_the_event():
    for lat0, lon0 in ((45.0, 200.0), (85.0, 30.0), (89.9, 100.0)):
        lat_geo, lon_geo = rotate_to_pole(np.array([0.0]), np.array([0.0]), lat0, lon0)
        assert lat_geo[0] == pytest.approx(lat0, abs=1e-9)
        assert lon_geo[0] % 360.0 == pytest.approx(lon0 % 360.0, abs=1e-9)


def test_rotated_patch_matches_the_geographic_one_at_the_equator():
    """With the centre on the equator the two frames coincide, so must the winds."""
    lat = np.arange(-90.0, 90.1, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = 20.0 * np.exp(-(((lat2d - 0.0) / 10.0) ** 2 + ((lon2d - 180.0) / 20.0) ** 2))
    v = -8.0 * np.exp(-(((lat2d - 0.0) / 10.0) ** 2 + ((lon2d - 180.0) / 20.0) ** 2))

    u_rot, v_rot = rotated_patch(
        u, v, lat, lon, 0.0, 180.0, lat_half=10.0, lon_half=10.0
    )
    u_geo = geographic_patch(u, lat, lon, 0.0, 180.0, lat_half=10.0, lon_half=10.0)
    v_geo = geographic_patch(v, lat, lon, 0.0, 180.0, lat_half=10.0, lon_half=10.0)
    assert np.abs(u_rot.values - u_geo.values).max() < 1e-6 * np.abs(u).max()
    assert np.abs(v_rot.values - v_geo.values).max() < 1e-6 * np.abs(v).max()


def _geodesic_vortex_wind(lat, lon, lat0, lon0, radius_m=1.5e6, amp=1.0e7):
    """Wind of a geodesic streamfunction vortex, and its analytic speed.

    A smooth vector field: its speed depends only on distance from the centre, so
    in a frame centred there it is a known function of the rotated coordinates.
    Fields built by giving the eastward and northward *components* a shape are not
    smooth vectors near a pole -- the local basis turns with longitude there -- and
    cannot be interpolated across it by anything.
    """
    R = 6.371e6
    lon2d, lat2d = np.meshgrid(lon, lat)
    cos_c = np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0)) + np.cos(
        np.radians(lat2d)
    ) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2d - lon0))
    gcd = np.arccos(np.clip(cos_c, -1, 1)) * R
    psi = amp * np.exp(-((gcd / radius_m) ** 2))
    speed = 2.0 * gcd / radius_m**2 * psi

    # Tangential, so perpendicular to the bearing towards the centre.
    bearing = frame_rotation_angle(lat2d, lon2d, lat0, lon0)
    u = speed * np.cos(bearing)
    v = -speed * np.sin(bearing)
    return u, v, psi


def test_rotated_patch_is_complete_across_the_pole():
    """The case the geographic box cannot represent at all."""
    lat = np.arange(-90.0, 90.1, 0.5)
    lon = np.arange(0.0, 360.0, 0.5)
    lat0, lon0 = 88.0, 40.0
    u, v, _ = _geodesic_vortex_wind(lat, lon, lat0, lon0)

    geo = geographic_patch(u, lat, lon, lat0, lon0, lat_half=20.0, lon_half=40.0)
    assert np.isnan(geo.values).any(), "a geographic box at 88N must run off the pole"

    u_rot, v_rot = rotated_patch(
        u, v, lat, lon, lat0, lon0, lat_half=20.0, lon_half=40.0
    )
    assert np.isfinite(u_rot.values).all()
    assert np.isfinite(v_rot.values).all()
    assert u_rot.values.shape == (41, 81)

    # In the rotated frame the speed depends only on distance from the centre.
    R, radius_m, amp = 6.371e6, 1.5e6, 1.0e7
    lat_r = np.radians(u_rot.lat_rel)[:, None]
    lon_r = np.radians(u_rot.lon_rel)[None, :]
    gcd = np.arccos(np.clip(np.cos(lat_r) * np.cos(lon_r), -1, 1)) * R
    exact = 2.0 * gcd / radius_m**2 * amp * np.exp(-((gcd / radius_m) ** 2))
    speed = np.hypot(u_rot.values, v_rot.values)
    assert np.abs(speed - exact).max() / exact.max() < 5e-3


def test_rotated_patch_accuracy_does_not_depend_on_latitude():
    """The whole point: the same vortex reads the same wherever it sits."""
    lat = np.arange(-90.0, 90.1, 0.5)
    lon = np.arange(0.0, 360.0, 0.5)
    R, radius_m, amp = 6.371e6, 1.5e6, 1.0e7
    errors = {}
    for lat0 in (45.0, 80.0, 89.0, 90.0):
        u, v, _ = _geodesic_vortex_wind(lat, lon, lat0, 40.0)
        u_rot, v_rot = rotated_patch(
            u, v, lat, lon, lat0, 40.0, lat_half=15.0, lon_half=15.0
        )
        lat_r = np.radians(u_rot.lat_rel)[:, None]
        lon_r = np.radians(u_rot.lon_rel)[None, :]
        gcd = np.arccos(np.clip(np.cos(lat_r) * np.cos(lon_r), -1, 1)) * R
        exact = 2.0 * gcd / radius_m**2 * amp * np.exp(-((gcd / radius_m) ** 2))
        speed = np.hypot(u_rot.values, v_rot.values)
        errors[lat0] = float(np.abs(speed - exact).max() / exact.max())
    assert max(errors.values()) < 1e-3, errors
    # The claim is that the pole is not a special place, not that the error is
    # constant -- it wanders with where the grid points happen to fall relative to
    # the centre.  What must not happen is the polar case degrading.
    assert errors[90.0] < 5.0 * errors[45.0], errors
    assert errors[89.0] < 5.0 * errors[45.0], errors


def test_rotated_patch_preserves_wind_speed():
    """A frame rotation turns the components; it must not change the magnitude."""
    lat = np.arange(-90.0, 90.1, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = 12.0 * np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))
    v = 7.0 * np.sin(np.radians(2 * lon2d))

    from pvinv_sph.winds import _sample, from_cartesian, to_cartesian

    u_rot, v_rot = rotated_patch(u, v, lat, lon, 75.0, 210.0, lat_half=20.0, lon_half=30.0)
    speed_rot = np.hypot(u_rot.values, v_rot.values)
    cart = to_cartesian(u, v, lat, lon)
    sampled = tuple(_sample(c, lat, lon, u_rot.lat, u_rot.lon) for c in cart)
    speed_geo = np.hypot(*from_cartesian(sampled, u_rot.lat, u_rot.lon))
    assert np.abs(speed_rot - speed_geo).max() < 1e-9 * speed_geo.max()


def test_grid_refuses_a_descending_latitude_axis():
    """Reversing the axis without the data turns every field upside down.

    Least-squares analysis fits an upside-down field perfectly well and returns a
    clean equator-mirrored solution with the eastward wind sign-flipped, so this
    has to be refused rather than accommodated.
    """
    from pvinv_sph.sht import grid_from_axes

    lat = np.arange(90.0, -90.1, -1.5)
    lon = np.arange(0.0, 360.0, 1.5)
    with pytest.raises(ValueError, match="latitudes must ascend"):
        grid_from_axes(lat, lon)
    grid_from_axes(lat[::-1], lon)  # the same axis the right way round is fine


def test_grid_refuses_a_longitude_axis_that_does_not_start_at_zero():
    """The transform is an FFT over the column index and takes column zero as 0E.

    A -180..180 axis therefore rotates the solver's frame by half a turn.  The
    per-level path would never notice -- the shift cancels between analysis and
    synthesis -- but anything using an absolute longitude, such as seeding a scale
    split at the event centre, would then be working at the antipode.
    """
    from pvinv_sph.sht import grid_from_axes

    lat = np.arange(-90.0, 90.1, 1.5)
    with pytest.raises(ValueError, match="longitudes must start at 0"):
        grid_from_axes(lat, np.arange(-180.0, 180.0, 1.5))
    with pytest.raises(ValueError, match="longitudes must start at 0"):
        grid_from_axes(lat, np.arange(0.0, 180.0, 1.5))  # half the circle
    grid_from_axes(lat, np.arange(0.0, 360.0, 1.5))


def test_pipeline_blanks_the_mirror_scaffold():
    """A patch reaching south of the data must not fill with reflected weather.

    The output grid spans both hemispheres; south of the data's own first row every
    value is a reflection of the north, and a centre below about forty degrees
    reaches into it.  Those rows have to come back empty, or they enter a composite
    as though they were observations.
    """
    out, lat_nh, _ = tiny_inversion(centre=(25.0, 100.0))

    patch_lat = out.arrays["lat_vec"]
    field = out.arrays["u_rot_anom_ppvi_300_3d"]
    southern = patch_lat < lat_nh[0]
    assert southern.any(), "this centre should reach south of the data"
    assert np.isnan(field[:, southern, :]).all(), "scaffold rows leaked into the patch"
    assert np.isfinite(field[:, ~southern & np.isfinite(patch_lat), :]).any()


def test_residual_key_follows_the_windowed_contract():
    """The residual sits outside the piece namespace, as it does in the old files."""
    from pvinv_sph.pipeline import _key

    assert _key("u", "residual", "_3d") == "u_rot_anom_residual_ppvi_3d"
    assert _key("v", "residual", "") == "v_rot_anom_residual_ppvi"
    assert _key("u", "300", "_3d") == "u_rot_anom_ppvi_300_3d"


def test_rotated_track_erases_the_mirror_scaffold_too():
    """The rotated crop is handed the Cartesian components, and those are not blanked.

    ``rotated_patch`` ignores the eastward and northward fields when it is given
    Cartesian ones, and it has to be given them: a NaN spreads through the
    sampler's spline prefilter and empties the whole patch, so the erasure cannot
    happen before sampling.  It therefore has to happen after, by where each
    sampled point landed -- and this is the test that says so, because the
    geographic crop's own scaffold test passes either way.
    """
    centre = (25.0, 100.0)
    out, lat_nh, _ = tiny_inversion(centre, rotated_track=True)
    keys = [k for k in out.arrays if k.endswith("_rot") and k.startswith("u_")]
    assert keys, "the rotated track wrote nothing"

    lat_geo = rotated_geographic_latitudes(centre)
    invented = lat_geo < lat_nh[0]
    assert invented.any(), "this centre should reach south of the data"

    for key in keys:
        field = out.arrays[key]
        assert np.isnan(field[..., invented]).all(), (
            f"{key} delivered the mirror's reflection as data"
        )
        assert np.isfinite(field[..., ~invented]).all(), (
            f"{key} is empty where it should have values"
        )


def test_rotated_track_is_complete_for_a_polar_centre():
    """The reason the rotated track exists: a polar event with nothing missing.

    A latitude-longitude box centred at eighty degrees asks for rows past the pole
    and cannot label them, and its pole row has no eastward direction to resolve.
    The rotated crop has neither problem, and a composite over polar events needs
    that.
    """
    centre = (80.0, 100.0)
    out, lat_nh, _ = tiny_inversion(centre, rotated_track=True)

    geographic = out.arrays["u_rot_anom_ppvi_300_3d"]
    assert np.isnan(geographic).any(), (
        "a box centred at eighty degrees should have rows it cannot label"
    )

    lat_geo = rotated_geographic_latitudes(centre)
    assert (lat_geo >= lat_nh[0]).all(), "this centre should not reach the mirror"
    for key in [k for k in out.arrays if k.endswith("_rot") and k[0] in "uv"]:
        assert np.isfinite(out.arrays[key]).all(), f"{key} has holes at the pole"


def test_rotated_track_writes_every_piece_and_the_residual():
    """Whatever the geographic track carries, the rotated one carries too.

    A composite built on the rotated track has to be able to close the same
    identity as one built on the geographic track, which it cannot do if a piece
    is missing from it.
    """
    out, _, _ = tiny_inversion((70.0, 100.0), rotated_track=True)
    plain = {
        k[:-4] for k in out.arrays if k.endswith("_rot") and k[0] in "uv"
    }
    for name in plain:
        assert name in out.arrays, f"{name}_rot exists with no geographic counterpart"
    for name in [k for k in out.arrays if k.endswith("_3d") and k[0] in "uv"]:
        assert f"{name}_rot" in out.arrays, f"{name} has no rotated counterpart"
