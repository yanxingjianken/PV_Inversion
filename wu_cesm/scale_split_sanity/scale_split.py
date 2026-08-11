"""Planetary / eddy split of the upper-level PV anomaly, for piecewise inversion.

One mask, defined once on the 3-D PV anomaly, used everywhere:

    q'          Wu PV anomaly (event - mean), GLOBAL in longitude
    q'_k4       zonal Fourier filter of q', wavenumbers 1..4
    M           the 3-D connected component of {q'_k4 < -thresh} that CONTAINS the
                seed, searched inside the event box only, over the upper INTERIOR
                levels.  The seed is the local minimum of q'_k4 within SEED_RADIUS
                of the tracked centroid, not the centroid itself: on a 2.2e13 m^2
                Z500 blob the centroid sat 10 deg south and 19 deg east of the
                nearest negative PV centre and was at the 92nd percentile of the
                anomaly, i.e. firmly positive.
    M[top]      the 200 hPa slice of M projected onto the 100 hPa boundary, because
                Wu's Q is identically zero at K=1 and K=NL -- those are boundary-θ
                sources, not PV, so there is no PV field at 100 hPa to flood-fill
    q'_p        = F(q'_k4 * M)     on the interior upper levels, F = the same k<=4 filter
    q'_e        = q' - q'_p
    th'_p       = F(th'_k4 * M[top])
    th'_e       = th' - th'_p

The second filter is not redundant.  Multiplying by a spatial mask is a
convolution in wavenumber, so `q'_k4 * M` is broadband again: measured on a
synthetic k=3 blob it came back with only 64 % of its power at k=1..4, 14 % above
k=4 and 22 % in the zonal mean.  Re-applying F makes `q'_p` exactly k<=4 by
construction, and `q'_e = q' - q'_p` still sums exactly.  The price is that
`q'_p` is no longer confined to the object's footprint -- a band-limited field
cannot have compact support, so support and bandwidth cannot both be imposed.
What survives is the intended meaning: `q'_p` is the planetary-scale projection
OF that 3-D connected object.

Including the top level in the flood-fill is what lets the object decide for
itself whether it reaches the boundary.  If the connected component stops below
it, ``M[top]`` is empty and the planetary piece simply has no boundary
contribution -- an outcome, not an assumption.  It also avoids inventing a
second criterion: no separate threshold, sign convention or 2-D connectivity
analysis for theta, because the mask is the same PV object's footprint.

`q'_p + q'_e = q'` and `th'_p + th'_e = th'` exactly, and pass D is linear in
both sources, so the pieces sum exactly to the unsplit upper piece.  That
additivity is an identity; the check worth running is `upper + lower + surface`
against the Helmholtz-decomposed rotational-wind anomaly.

**The filter has to be global.**  Zonal wavenumber is defined on the full 360-deg
circle.  The inversion box is ~120 deg wide, whose own fundamental is global k=3,
so filtering inside the box cannot separate k<=4 from k>4 at all.  Pass A/B
(`pvpialln_core`) is a local diagnostic, not an elliptic solve, and its NY/NX/NW
are f2py-hidden dimensions taken from the input shape -- so it runs once on the
full NH band, the split is done there, and only the anomaly arrays are cropped.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from pvtend.ppvi import _ext
from pvtend.ppvi.solver import (PR, PassABParams, _from_core, _to_core,
                                fill_below_ground)

KMIN, KMAX = 1, 4          # zonal wavenumbers kept in the planetary piece
SEED_RADIUS_DEG = 20.0     # search radius around the tracked centroid for the seed
THRESH = 0.3               # |Wu PV anomaly| threshold for "negative"


def wu_state_band(H, T, U, V, lat_n, lat_s, dlat, dlon, nx, pab=None):
    """Wu PV and boundary theta on an arbitrary band, pass A/B only.

    Returns ``(q, thb, tht)`` -- PV ``(NL, NY, NX)`` and the bottom/top boundary
    potential temperatures ``(NY, NX)``.
    """
    # `invert_piecewise` fills below-ground NaN internally; this is a direct
    # pass-A/B call, so it has to do the same.  Without it Q comes back NaN at
    # every level that is underground anywhere (850/700/500 over terrain), and a
    # NaN there silently poisons the whole mask.
    H, T, U, V = fill_below_ground(H, T, U, V)
    p = pab or PassABParams()
    zhdr10 = np.zeros(10, dtype=np.float32)
    zhdr10[0], zhdr10[1], zhdr10[2] = lat_s, 0.0, lat_n
    zhdr10[3] = (nx - 1) * dlon
    zhdr10[4], zhdr10[5] = dlon, dlat
    zhdr10[6], zhdr10[7] = nx, H.shape[1]
    _, q, thb, tht = _ext.load_ext().pvpialln_core(
        _to_core(H), _to_core(T), _to_core(U), _to_core(V),
        zhdr10, PR, int(p.imax), np.float32(p.omegs), np.float32(p.thrs),
    )
    return _from_core(q), np.asarray(thb), np.asarray(tht)


def zonal_filter(field, kmin=KMIN, kmax=KMAX):
    """Keep zonal wavenumbers ``kmin..kmax``; last axis must span 360 deg.

    k=0 is dropped: it is the zonal mean, and putting it in the planetary piece
    would give that piece a meridional-only offset with no wave structure.
    """
    F = np.fft.rfft(np.asarray(field, dtype=float), axis=-1)
    G = np.zeros_like(F)
    G[..., kmin:kmax + 1] = F[..., kmin:kmax + 1]
    return np.fft.irfft(G, n=np.shape(field)[-1], axis=-1)


def component_containing(mask, seed_lev, seed_lat, seed_lon):
    """3-D 6-connected component of `mask` containing the seed point.

    The selector is the tracked anticyclone's own centroid, not "the largest
    component".  On a k<=4-filtered field the largest component is useless: the
    filtered field's zero line is a smooth planetary-scale curve, so its negative
    side is a few very large patches, and with a periodic longitude they link
    through the weakly negative subtropics into ONE object covering ~38 % of the
    hemisphere.  That is every negative PV anomaly in the NH, not this block.

    Raises if the seed is not inside the mask -- which is the useful failure: it
    says the threshold or the level range is wrong, instead of silently handing
    back a hemispheric blob.
    """
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        raise ValueError("mask is empty at this threshold")
    lid = int(lab[seed_lev, seed_lat, seed_lon])
    if lid == 0:
        raise ValueError(
            f"the tracked centroid (lev {seed_lev}, lat {seed_lat}, lon {seed_lon}) "
            f"is not inside any negative component at this threshold")
    return lab == lid


def largest_component_periodic(mask):
    """Largest 3-D 6-connected component of a boolean mask, periodic in the last axis."""
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    left, right = lab[..., 0], lab[..., -1]
    both = (left > 0) & (right > 0)
    for a, b in zip(left[both].ravel(), right[both].ravel()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(n + 1)])
    merged = roots[lab]
    sizes = np.bincount(merged.ravel())
    sizes[0] = 0
    return merged == int(np.argmax(sizes))


def seed_from_centroid(q_k4_level, lat, lon, clat, clon, radius=SEED_RADIUS_DEG):
    """Local minimum of the filtered anomaly within `radius` of the tracked centroid.

    Returns global ``(j, i)``.  Using the centroid itself fails on large blobs:
    the Z500 blob centroid and the upper-level PV minimum are different points,
    and the gap grows with blob size.
    """
    jj = np.nonzero(np.abs(lat - clat) <= radius)[0]
    ii = np.nonzero(np.abs(((lon - clon + 180.0) % 360.0) - 180.0) <= radius)[0]
    sub = q_k4_level[np.ix_(jj, ii)]
    a, b = np.unravel_index(int(np.argmin(sub)), sub.shape)
    return int(jj[a]), int(ii[b])


def split_planetary_eddy(q_anom, th_anom_top, upper_idx, top_idx,
                         box_lat, box_lon, seed_lev, seed_lat, seed_lon,
                         kmin=KMIN, kmax=KMAX, thresh=THRESH):
    """Split the upper PV anomaly (and its top boundary theta) into planetary/eddy.

    Args:
        q_anom: ``(NL, NY, NX_global)`` Wu PV anomaly, last axis spanning 360 deg.
        th_anom_top: ``(NY, NX_global)`` top boundary-theta anomaly.
        upper_idx: 0-based level indices of the upper INTERIOR levels.
        top_idx: 0-based index of the top boundary level (must be in the
            flood-fill domain, i.e. ``upper_idx + [top_idx]``).
        box_lat, box_lon: index arrays of the event box.  The filter is global,
            the object search is confined to the box.
        seed_lev, seed_lat, seed_lon: global indices of the tracked centroid; the
            connected component containing this point is the object.

    Returns:
        ``dict`` with ``q_p, q_e`` ``(NL, NY, NX)``; ``th_p, th_e`` ``(NY, NX)``;
        ``mask`` ``(NL, NY, NX)`` bool; and ``reaches_top`` bool.
    """
    upper_idx = np.asarray(upper_idx, dtype=int)
    dom = np.concatenate([upper_idx, [top_idx]])
    box_lat = np.asarray(box_lat, dtype=int)
    box_lon = np.asarray(box_lon, dtype=int)

    # 1. filter GLOBALLY -- zonal wavenumber only exists on the full circle.
    #    The flood-fill domain is the upper INTERIOR levels only: Wu's Q is
    #    identically zero at the two boundary levels.
    q_k_int = zonal_filter(q_anom[upper_idx], kmin, kmax)

    # 2. crop to the event box and take the component holding the seed, so the
    #    object is THIS anticyclone and not every negative patch in the NH
    sub = q_k_int[:, box_lat, :][:, :, box_lon] < -abs(thresh)
    j = int(np.nonzero(box_lat == seed_lat)[0][0])
    i = int(np.nonzero(box_lon == seed_lon)[0][0])
    k = int(np.nonzero(upper_idx == seed_lev)[0][0])
    m_box = component_containing(sub, k, j, i)

    # 3. embed back into the global array; outside the box is not the object
    mask_int = np.zeros_like(q_k_int, dtype=bool)
    mask_int[np.ix_(np.arange(len(upper_idx)), box_lat, box_lon)] = m_box

    mask = np.zeros_like(q_anom, dtype=bool)
    mask[upper_idx] = mask_int
    # 4. the top boundary inherits the mask of the highest interior level (200 hPa)
    mask[top_idx] = mask_int[-1]

    q_k = q_k_int
    mask_dom = mask_int

    q_p = np.zeros_like(q_anom)
    q_e = np.zeros_like(q_anom)
    # interior levels only: the top level's PV is not an inversion source
    q_masked = q_k[:len(upper_idx)] * mask_dom[:len(upper_idx)]
    q_p_upper = zonal_filter(q_masked, kmin, kmax)     # re-filter: see module docstring
    for n, k in enumerate(upper_idx):
        q_p[k] = q_p_upper[n]
        q_e[k] = q_anom[k] - q_p[k]

    th_k = zonal_filter(th_anom_top, kmin, kmax)
    m_top = mask[top_idx]
    th_p = zonal_filter(th_k * m_top, kmin, kmax)
    th_e = th_anom_top - th_p

    return {"q_p": q_p, "q_e": q_e, "th_p": th_p, "th_e": th_e,
            "mask": mask, "top_frac": float(m_top.mean()),
            "seed": (int(seed_lev), int(seed_lat), int(seed_lon))}


def wavenumber_spectrum(field):
    """Fractional zonal-wavenumber power, for checking the k<=4 claim."""
    F = np.fft.rfft(np.asarray(field, dtype=float), axis=-1)
    p = (np.abs(F) ** 2).sum(axis=tuple(range(np.ndim(field) - 1)))
    return p / max(p.sum(), 1e-300)
