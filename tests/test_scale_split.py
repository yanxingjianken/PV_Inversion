"""Splitting a source by scale, on the convention the windowed pipeline uses.

The property that matters is additivity: whatever the split does, the parts must
sum to what went in, because that is what keeps the pieces they drive summing to
the piece the whole source would have driven. Beyond that, the object has to be
one connected body through the depth, found inside the event's own box, and the
planetary part has to still be planetary after it has been masked.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvinv_sph.scale_split import (
    KMAX,
    KMIN,
    SEED_HALO_LAT,
    SEED_HALO_LON,
    component_containing,
    event_box,
    great_circle_degrees,
    scale_pieces,
    centre_indices,
    seed_near_centre,
    split_planetary_eddy,
    zonal_filter,
)
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps


def make_ops(nlat=64, nlon=128, lmax=40):
    return SphereOps(SHT(gaussian_grid(nlat, nlon), lmax=lmax))


def wavenumber_field(ops, orders, amplitude=1.0, seed=0):
    """A field built from exactly the given zonal orders.

    Assembled in spectral space, so it is band limited by construction. Built on
    the grid instead -- ``cos(lat)**2 cos(m lon)``, say -- it would not be: for
    order three that shape needs an infinite series in the associated Legendre
    functions of that order, and the truncation would show up as a filter error
    belonging to the test field.
    """
    rng = np.random.default_rng(seed)
    spec = np.zeros((ops.sht.lmax + 1, ops.sht.lmax + 1), dtype=complex)
    for m in orders:
        for n in range(max(m, 1), min(m + 6, ops.sht.lmax + 1)):
            spec[m, n] = amplitude * (rng.normal() + 1j * rng.normal())
    spec[0] = spec[0].real
    return ops.synth(spec)


def blob_source(ops, lat0, lon0, nint=7, upper=(3, 4, 5, 6), amplitude=-1.0):
    """A negative anomaly on the upper levels, plus small-scale noise."""
    distance = great_circle_degrees(ops, lat0, lon0)
    blob = amplitude * np.exp(-((distance / 18.0) ** 2))
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    noise = 0.15 * np.cos(np.radians(11 * lon2d)) * np.cos(np.radians(lat2d)) ** 2
    field = np.zeros((nint, ops.grid.nlat, ops.grid.nlon))
    for k in range(nint):
        field[k] = noise + (blob if k in upper else 0.3 * blob)
    return field


def test_zonal_filter_keeps_exactly_the_requested_orders():
    ops = make_ops()
    kept = zonal_filter(ops, wavenumber_field(ops, [1, 3, 7, 12]))
    spec = ops.analyze(kept)
    inside = np.abs(spec[KMIN : KMAX + 1]).max()
    outside = max(np.abs(spec[0]).max(), np.abs(spec[KMAX + 1 :]).max())
    assert inside > 0
    assert outside / inside < 1e-12

    unchanged = wavenumber_field(ops, [2, 3], seed=5)
    assert np.abs(zonal_filter(ops, unchanged) - unchanged).max() / np.abs(
        unchanged
    ).max() < 1e-10


def test_zonal_filter_drops_the_zonal_mean():
    """Order zero is excluded on purpose: it carries no wave structure."""
    ops = make_ops()
    lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)[1]
    zonal_only = np.cos(np.radians(lat2d)) ** 2
    assert np.abs(zonal_filter(ops, zonal_only)).max() / np.abs(zonal_only).max() < 1e-12


def test_the_object_is_one_body_through_the_depth():
    """Six-connected in three dimensions, not a component per level."""
    mask = np.zeros((5, 9, 9), dtype=bool)
    mask[1:4, 4, 4] = True          # a column, connected level to level
    mask[0, 0, 0] = True            # a separate speck
    got = component_containing(mask, (2, 4, 4))
    assert got.sum() == 3
    assert not got[0, 0, 0]

    with pytest.raises(ValueError, match="not inside any component"):
        component_containing(mask, (4, 4, 4))
    with pytest.raises(ValueError, match="empty"):
        component_containing(np.zeros((3, 3, 3), bool), (1, 1, 1))


def test_the_box_takes_longitude_the_short_way():
    ops = make_ops()
    rows, cols = event_box(ops, 55.0, 5.0, lat_half=20.0, lon_half=30.0)
    taken = ops.grid.lon[cols]
    assert taken.min() < 10.0 and taken.max() > 330.0, "the box stops at the seam"
    assert cols.size < ops.grid.nlon


def test_the_box_stops_at_the_pole():
    ops = make_ops()
    rows, _ = event_box(ops, 80.0, 200.0, lat_half=30.0)
    assert ops.grid.lat[rows].max() <= 90.0
    assert rows.size > 0


def seed_here(ops, filtered, lat0, lon0):
    rows, cols = event_box(ops, lat0, lon0)
    row, col = centre_indices(ops, lat0, lon0)
    dlat = float(np.mean(np.diff(ops.grid.lat)))
    dlon = 360.0 / ops.grid.nlon
    return seed_near_centre(
        filtered, rows, cols, row, col,
        int(round(SEED_HALO_LAT / dlat)), int(round(SEED_HALO_LON / dlon)),
    )


def test_the_seed_is_near_the_tracked_centre():
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    filtered = zonal_filter(ops, q[[3, 4, 5, 6]])
    rows, cols = event_box(ops, 55.0, 200.0)
    seed, q_min = seed_here(ops, filtered, 55.0, 200.0)
    assert q_min < 0
    assert np.isclose(filtered[seed], q_min)
    assert seed[1] in rows and seed[2] in cols
    assert great_circle_degrees(ops, 55.0, 200.0)[seed[1], seed[2]] <= SEED_HALO_LON


def test_the_seed_ignores_a_deeper_system_elsewhere_in_the_box():
    """The box is wide enough to hold something deeper than the event.

    Keying the seed off the box minimum would invert that other system instead,
    and nothing downstream would complain: the eddy part is defined as the
    remainder, so the decomposition still sums to the anomaly either way.
    """
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0) + 2.5 * blob_source(ops, 40.0, 250.0)
    filtered = zonal_filter(ops, q[[3, 4, 5, 6]])
    rows, cols = event_box(ops, 55.0, 200.0)
    seed, _ = seed_here(ops, filtered, 55.0, 200.0)
    deepest = np.unravel_index(int(np.nanargmin(filtered[:, rows, :][:, :, cols])),
                               (4, rows.size, cols.size))
    assert great_circle_degrees(ops, 55.0, 200.0)[rows[deepest[1]], cols[deepest[2]]] > 20.0, (
        "the test is vacuous unless the box minimum really is the other system"
    )
    assert great_circle_degrees(ops, 55.0, 200.0)[seed[1], seed[2]] <= SEED_HALO_LON


def test_the_box_columns_run_around_the_circle():
    """Consecutive columns must be neighbours on the globe, seam or no seam.

    Sorted indices are not: a box spanning the prime meridian comes back as two
    runs at opposite ends of the array, which severs the object at the seam and
    joins the box's two edges instead.
    """
    ops = make_ops()
    for lon0 in (2.0, 180.0, 358.0):
        _, cols = event_box(ops, 55.0, lon0, lat_half=20.0, lon_half=30.0)
        step = (np.diff(ops.grid.lon[cols]) + 180.0) % 360.0 - 180.0
        assert np.allclose(step, 360.0 / ops.grid.nlon), (
            f"the box at {lon0} is out of circular order"
        )
        assert np.unique(cols).size == cols.size, "a column is listed twice"


def test_the_split_is_exactly_additive():
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    theta = blob_source(ops, 55.0, 200.0, nint=1)[0] * 4.0
    upper = [3, 4, 5, 6]
    out = split_planetary_eddy(ops, q, theta, upper, 55.0, 200.0)
    # Both parts cover the upper levels only; the rest belongs to the lower piece.
    assert np.abs((out["q_p"] + out["q_e"])[upper] - q[upper]).max() / np.abs(
        q
    ).max() < 1e-13
    other = [k for k in range(q.shape[0]) if k not in upper]
    assert np.abs(out["q_p"][other]).max() == 0.0
    assert np.abs(out["q_e"][other]).max() == 0.0
    assert np.abs(out["theta_p"] + out["theta_e"] - theta).max() / np.abs(
        theta
    ).max() < 1e-13


def test_the_planetary_part_is_still_planetary_after_masking():
    """A hard mask edge puts power back into every order, so it is filtered again.

    Without the second filter the piece carries small-scale structure under a
    name that says it does not.
    """
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    theta = blob_source(ops, 55.0, 200.0, nint=1)[0] * 4.0
    out = split_planetary_eddy(ops, q, theta, [3, 4, 5, 6], 55.0, 200.0)

    spec = ops.analyze(out["q_p"][4])
    inside = np.abs(spec[KMIN : KMAX + 1]).max()
    outside = max(np.abs(spec[0]).max(), np.abs(spec[KMAX + 1 :]).max())
    assert outside / inside < 1e-12

    # and the merely-masked field is not: the second filter does real work
    filtered = zonal_filter(ops, q[[3, 4, 5, 6]])
    merely = ops.analyze((filtered * out["mask"][[3, 4, 5, 6]])[1])
    leak = max(np.abs(merely[0]).max(), np.abs(merely[KMAX + 1 :]).max())
    assert leak / np.abs(merely[KMIN : KMAX + 1]).max() > 0.1


def test_the_object_is_the_event_and_not_the_hemisphere():
    """The box is what bounds it; a seed alone would not."""
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    theta = blob_source(ops, 55.0, 200.0, nint=1)[0] * 4.0
    out = split_planetary_eddy(ops, q, theta, [3, 4, 5, 6], 55.0, 200.0)
    assert 0.0 < out["object_fraction"] < 0.25
    rows, cols = event_box(ops, 55.0, 200.0)
    outside = np.ones(out["mask"].shape[1:], bool)
    outside[np.ix_(rows, cols)] = False
    assert not out["mask"][:, outside].any(), "the object leaked outside its box"


def test_the_top_boundary_inherits_the_highest_level_mask():
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    # A wave, not a constant: the filter keeps orders one to four, so a uniform
    # temperature has nothing for the planetary part to take.
    theta = wavenumber_field(ops, [2, 3], amplitude=3.0, seed=7)
    out = split_planetary_eddy(ops, q, theta, [3, 4, 5, 6], 55.0, 200.0)
    assert 0.0 < out["top_fraction"] < 0.25
    assert np.abs(out["theta_p"]).max() > 0


def test_it_works_with_the_event_over_the_pole():
    """The reason the filter needs no window: it is a coefficient selection."""
    ops = make_ops()
    q = blob_source(ops, 88.0, 200.0)
    theta = blob_source(ops, 88.0, 200.0, nint=1)[0] * 4.0
    upper = [3, 4, 5, 6]
    out = split_planetary_eddy(ops, q, theta, upper, 88.0, 200.0)
    assert np.isfinite(out["q_p"]).all() and np.isfinite(out["q_e"]).all()
    assert np.abs((out["q_p"] + out["q_e"])[upper] - q[upper]).max() / np.abs(
        q
    ).max() < 1e-13
    assert out["object_fraction"] > 0


def test_a_box_with_no_anomaly_is_refused():
    ops = make_ops()
    lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)[1]
    q = np.ones((7, ops.grid.nlat, ops.grid.nlon)) * np.cos(np.radians(lat2d))
    theta = np.zeros((ops.grid.nlat, ops.grid.nlon))
    with pytest.raises(ValueError, match="no negative planetary-scale anomaly"):
        split_planetary_eddy(ops, q, theta, [3, 4, 5, 6], 55.0, 200.0)


def test_scale_pieces_partition_the_source():
    ops = make_ops()
    q = blob_source(ops, 55.0, 200.0)
    theta = blob_source(ops, 55.0, 200.0, nint=1)[0] * 4.0
    upper = [3, 4, 5, 6]
    sources, split = scale_pieces(ops, q, theta, upper, 55.0, 200.0)
    total = sources["lower"] + sources["upper_p"] + sources["upper_e"]
    assert np.abs(total - q).max() / np.abs(q).max() < 1e-13
    assert np.abs(sources["lower"][upper]).max() == 0.0
    assert np.abs(sources["lower"][0] - q[0]).max() == 0.0
