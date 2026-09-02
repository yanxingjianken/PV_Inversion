"""The windowed inversion, driven exactly as production drives it.

Calls ``pvtend.ppvi.invert_piecewise`` -- the compiled Wu/Davis Fortran chain --
without modifying anything in that package.  What has to be reproduced here is the
*geometry*, because the Fortran is handed cubes and a header and reconstructs its
own latitudes from them:

* the solve band is a fixed 10.5 to 85.5 N, the same for every event, and its rows
  must run north to south (CESM is stored south to north, so they are reversed);
* longitude is a wrapped window of plus and minus ninety degrees about the event;
* the header carries spacing as ``(dlon, dlat)`` where the public argument order
  is ``(dlat, dlon)``, which is a no-op on an isotropic grid and silently lays the
  rows at the wrong spacing on f09.

Pieces are per level and the lateral condition is ``ibc=1``: each piece is solved
with the full perturbation on its walls.

That last word matters.  The Fortran accumulates what it has already attributed
and puts only the *remainder* on the next piece's walls -- but only within one
call.  Here every piece is its own call, so the accumulator starts at zero each
time and all nine pieces carry the same wall response.  Summing them counts that
response nine times, which showed up as an induced flow of 250 m/s where the
anomaly itself is 51.

The response is therefore solved for on its own -- one more inversion, with the
sources zeroed, so that only the walls drive it -- and removed from each piece.
The pieces then carry what their own source induced, and adding the wall response
back once gives the full balanced perturbation, which is what the global solve
produces without any of this.
"""
from __future__ import annotations

import numpy as np

BAND_SOUTH, BAND_NORTH = 10.5, 85.5
LON_HALF_INVERSION = 90.0


def band_rows(lat: np.ndarray) -> np.ndarray:
    """Row indices of the solve band, ordered north to south."""
    inside = np.flatnonzero((lat >= BAND_SOUTH) & (lat <= BAND_NORTH))
    if lat[inside[0]] < lat[inside[-1]]:
        inside = inside[::-1]
    return inside


def window_columns(lon: np.ndarray, lon0: float, half: float = LON_HALF_INVERSION):
    """Column indices of the longitude window, wrapped, centred on the event."""
    dlon = float(np.mean(np.diff(lon)))
    pad = int(round(half / dlon))
    centre = int(np.argmin(np.abs((lon - lon0 + 180.0) % 360.0 - 180.0)))
    return (np.arange(2 * pad + 1) + centre - pad) % lon.size


def invert_windowed(mean_fields, event_fields, lat, lon, lat0, lon0, ibc=1):
    """Run the Fortran chain on the production window; return per-level winds.

    Returns ``(u, v, rows, cols, parts)``.  The winds are the full balanced
    perturbation -- every piece's own contribution plus the wall response counted
    once -- shaped ``(nlev, nrow, ncol)`` on the window.  ``parts`` holds the
    corrected pieces and the wall separately, for attribution.
    """
    from pvtend.ppvi import PIECES, PassDParams, invert_piecewise, psi_to_winds

    rows = band_rows(lat)
    cols = window_columns(lon, lon0)
    dlat = float(np.mean(np.abs(np.diff(lat))))
    dlon = float(np.mean(np.diff(lon)))
    band_lats = lat[rows]

    def window(fields):
        return [np.ascontiguousarray(f[:, rows, :][:, :, cols]) for f in fields]

    mean_w = window(mean_fields)
    event_w = window(event_fields)
    header = np.array(
        [
            band_lats[-1],
            0.0,
            band_lats[0],
            (cols.size - 1) * dlon,
            dlat,
            dlon,
            cols.size,
            rows.size,
        ],
        dtype=np.float32,
    )
    # One extra inversion whose sources are the mean state itself, so its
    # perturbation is identically zero and only the walls drive it.  The overrides
    # are additive, so zeros mean "use the mean", which is what makes this the
    # wall response alone.
    zero_q = np.zeros((len(PIECES), rows.size, cols.size), dtype=np.float64)
    zero_theta = np.zeros((rows.size, cols.size, 2), dtype=np.float64)
    pieces = dict(PIECES)
    pieces["wall"] = [2]
    result = invert_piecewise(
        *mean_w,
        *event_w,
        header,
        pieces=pieces,
        qp_anoms={"wall": zero_q},
        th_anoms={"wall": zero_theta},
        pd=PassDParams(ibc=ibc),
    )

    def winds(psi):
        return psi_to_winds(psi, band_lats, dlat, dlon)

    wall_u, wall_v = winds(result["psi_pieces"]["wall"])
    parts = {"wall": (wall_u, wall_v)}
    total_u, total_v = wall_u.copy(), wall_v.copy()
    for name, psi in result["psi_pieces"].items():
        if name == "wall":
            continue
        u, v = winds(psi)
        parts[name] = (u - wall_u, v - wall_v)
        total_u += parts[name][0]
        total_v += parts[name][1]
    return total_u, total_v, rows, cols, parts


def scatter_to_globe(field, rows, cols, nlat, nlon):
    """Place a window's field back on the global grid, empty everywhere else."""
    out = np.full((field.shape[0], nlat, nlon), np.nan)
    out[:, rows[:, None], cols[None, :]] = field
    return out
