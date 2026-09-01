"""Prepare .grid input files for the Fortran PV-inversion code from
CESM2-LENS2 6-hourly pressure-level data.

For one ensemble member and one event time, extracts a window of 12-hourly
snapshots, fills below-ground NaN on the native grid (the pressure-level
archive masks levels under terrain / high surface pressure), bilinearly
interpolates Z3/T/U/V to two grids centered on the event (an inner inversion
grid and an outer background grid), and writes one fixed-format .grid file
per snapshot per grid, plus a meta.json manifest.

Example
-------
python prep_event.py --member m91 --time '1996-01-15 12:00' \\
    --lat0 50.0 --lon0 -20.0 --outdir /path/to/event
"""

import argparse
import datetime
import json
import os
import re

import cftime
import numpy as np
import xarray as xr

import talia_io

DATA_DIR = "/net/flood/data2/users/x_yan/cesm-blocking/cesm_6hourly/cesm2_lens2_wu9_nh"
VARS = ["Z3", "T", "U", "V"]
#: Latitude margin (deg) kept around the outer grid for interpolation.
LAT_MARGIN = 3.0


def normalize_member(s):
    """Map 'm91', 'm091', '91', '091', 'm100' ... to the file token 'm91'."""
    m = re.fullmatch(r"[mM]?0*(\d+)", s.strip())
    if not m:
        raise ValueError(f"cannot parse member {s!r}")
    return f"m{int(m.group(1))}"


def parse_time(s):
    """Parse 'YYYY-MM-DD HH:MM' into a DatetimeNoLeap."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"cannot parse time {s!r} (want 'YYYY-MM-DD HH:MM')")
    y, mo, d, h, mi = (int(g) for g in m.groups())
    return cftime.DatetimeNoLeap(y, mo, d, h, mi)


def halfday_times(t0, half_days):
    """12-hourly window t_k = t0 + (k - half_days//2)*12h for k=1..half_days.

    The event itself is halfday index ``half_days // 2`` (1-based).
    """
    event_k = half_days // 2
    step = datetime.timedelta(hours=12)
    return [t0 + (k - event_k) * step for k in range(1, half_days + 1)]


def fill_below_ground(ds):
    """Fill below-ground NaN in a (time, plev, lat, lon) dataset, in place.

    The pressure-level archive stores NaN where an isobaric level lies below
    ground.  The Fortran inversion code is not NaN-aware (and F8.1 fields
    cannot encode NaN), so columns are completed before interpolation using
    the same convention as ``pvtend.ppvi.solver.fill_below_ground`` (the
    ERA5-style treatment): T/U/V by constant downward extrapolation from the
    lowest valid level, Z3 hydrostatically,
    ``H_k = H_{k+1} + (R_d*Tbar/g)*ln(p_{k+1}/p_k)``.

    Requires plev ordered bottom-up (index 0 = 1000 hPa) and a finite top
    level in every column.
    """
    rd, g = 287.05, 9.80665
    p = ds.plev.values.astype(np.float64)
    H = ds["Z3"].values
    T = ds["T"].values
    fields = [T, ds["U"].values, ds["V"].values]
    for k in range(len(p) - 2, -1, -1):     # downward, filling k from k+1
        for a in fields:
            m = ~np.isfinite(a[:, k])
            a[:, k][m] = a[:, k + 1][m]
        mz = ~np.isfinite(H[:, k])
        dz = (rd * 0.5 * (T[:, k] + T[:, k + 1]) / g) * np.log(p[k + 1] / p[k])
        H[:, k][mz] = (H[:, k + 1] + dz)[mz]
    return ds


def to_pm180(lon):
    """Map a longitude to the [-180, 180) convention."""
    return ((lon + 180.0) % 360.0) - 180.0


def target_grid(lat0, lon0, ny, nx, dlat, dlon):
    """Build target lat/lon axes centered on (lat0, lon0).

    Latitudes run north -> south (row 0 is the north edge), longitudes run
    west -> east.  ny and nx must be odd so (lat0, lon0) is a gridpoint.
    Returned lons are continuous (may exceed [-180, 180]); take them modulo
    360 before interpolating against a 0..360 source axis.
    """
    if ny % 2 == 0 or nx % 2 == 0:
        raise ValueError(f"grid dims must be odd, got ny={ny}, nx={nx}")
    half_lat = (ny - 1) // 2 * dlat
    half_lon = (nx - 1) // 2 * dlon
    lats = lat0 + half_lat - dlat * np.arange(ny)   # north -> south
    lons = lon0 - half_lon + dlon * np.arange(nx)   # west -> east
    return lats, lons


def open_window(member, times):
    """Open the member-year file(s) covering ``times`` and return
    (dataset restricted to VARS and the window's time span, source paths).

    The window may straddle Jan 1 and thus span two consecutive year files.
    """
    years = sorted({t.year for t in times})
    tmin, tmax = min(times), max(times)
    parts, paths = [], []
    for year in years:
        path = os.path.join(DATA_DIR, f"lens2_smbb_{member}_{year}_plev.nc")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds = xr.open_dataset(path)[VARS]
        parts.append(ds.sel(time=slice(tmin, tmax)))
        paths.append(path)
    ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")

    have = {t.isoformat() for t in ds.time.values}
    missing = [t.isoformat() for t in times if t.isoformat() not in have]
    if missing:
        raise ValueError(
            f"times not on the 6-hourly axis of {paths}: {missing}")
    return ds, paths


def interp_snapshot(snap, lats, lons):
    """Bilinearly interpolate one time snapshot to a target grid.

    ``snap`` must already have a longitude axis extended past 360 so that a
    wrapped target range stays in-range.  Returns arrays shaped
    (nlev, ny, nx) with rows north -> south.
    """
    out = snap.interp(lat=lats, lon=lons, method="linear")
    fields = {}
    for v in VARS:
        a = out[v].values
        if not np.isfinite(a).all():
            raise ValueError(f"{v}: NaNs after interpolation")
        fields[v] = a
    return fields


def prep_event(member, t0, lat0, lon0, outdir, half_days=20,
               dlat=2.5, dlon=2.5, inner_ny=21, inner_nx=45,
               outer_ny=25, outer_nx=89):
    member = normalize_member(member)
    if not -180.0 <= lon0 <= 180.0:
        raise ValueError(f"lon0 must be in [-180, 180], got {lon0}")

    grids = {
        "inner": target_grid(lat0, lon0, inner_ny, inner_nx, dlat, dlon),
        "outer": target_grid(lat0, lon0, outer_ny, outer_nx, dlat, dlon),
    }
    outer_lats = grids["outer"][0]
    if not (2.0 <= outer_lats.min() and outer_lats.max() <= 88.0):
        raise ValueError(
            f"outer grid latitude range [{outer_lats.min()}, "
            f"{outer_lats.max()}] leaves [2, 88] degN")

    times = halfday_times(t0, half_days)
    ds, paths = open_window(member, times)

    # Load only the latitude band needed, then extend the 0..360 longitude
    # axis by one wrap copy so any target window (span < 360 deg) is
    # in-range for bilinear interpolation without seam handling.
    lat_lo = outer_lats.min() - LAT_MARGIN
    lat_hi = outer_lats.max() + LAT_MARGIN
    ds = ds.sel(lat=slice(lat_lo, lat_hi)).load()
    ds = fill_below_ground(ds)
    ds = xr.concat([ds, ds.assign_coords(lon=ds.lon + 360.0)], dim="lon")

    for name in grids:
        os.makedirs(os.path.join(outdir, name), exist_ok=True)

    meta_grids = {}
    for name, (lats, lons) in grids.items():
        ny, nx = lats.size, lons.size
        header = (lats.min(), to_pm180(lons[0]), lats.max(),
                  to_pm180(lons[-1]), dlon, dlat, nx, ny)
        meta_grids[name] = {
            "slat": float(lats.min()), "nlat": float(lats.max()),
            "wlon": float(to_pm180(lons[0])),
            "elon": float(to_pm180(lons[-1])),
            "dlat": dlat, "dlon": dlon, "ny": ny, "nx": nx,
            "rows": "north->south", "cols": "west->east",
        }
        # Interpolation targets: mod-360 west edge, increasing eastward.
        lons_mod = lons[0] % 360.0 + (lons - lons[0])
        for t in times:
            snap = ds.sel(time=t)
            fields = interp_snapshot(snap, lats, lons_mod)
            stamp = f"{t.year:04d}{t.month:02d}{t.day:02d}{t.hour:02d}"
            path = os.path.join(outdir, name, f"{stamp}.grid")
            talia_io.write_grid(path, header, fields["Z3"], fields["T"],
                                fields["U"], fields["V"])

    meta = {
        "member": member,
        "center": {"lat0": lat0, "lon0": lon0},
        "times": [t.isoformat() for t in times],
        "event_index": half_days // 2,
        "event_index_base": 1,
        "half_days": half_days,
        "grids": meta_grids,
        "plev_hPa": talia_io.PLEV,
        "variables": {"H": "Z3 (m)", "T": "T (K)",
                      "U": "U (m/s)", "V": "V (m/s)"},
        "calendar": "noleap",
        "below_ground_fill": "T/U/V persistence, Z3 hydrostatic "
                             "(pvtend.ppvi.solver.fill_below_ground convention)",
        "source_files": paths,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    p = argparse.ArgumentParser(
        description="Prepare Fortran PV-inversion .grid inputs from "
                    "CESM2-LENS2 6-hourly pressure-level data.")
    p.add_argument("--member", required=True,
                   help="ensemble member: m91|m091|91 ... m100")
    p.add_argument("--time", required=True,
                   help="event time 'YYYY-MM-DD HH:MM' (noleap calendar)")
    p.add_argument("--lat0", type=float, required=True,
                   help="event center latitude (degN)")
    p.add_argument("--lon0", type=float, required=True,
                   help="event center longitude in -180..180")
    p.add_argument("--outdir", required=True)
    p.add_argument("--half-days", type=int, default=20,
                   help="number of 12-hourly snapshots (event at index n//2)")
    p.add_argument("--dlat", type=float, default=2.5)
    p.add_argument("--dlon", type=float, default=2.5)
    p.add_argument("--inner-ny", type=int, default=21)
    p.add_argument("--inner-nx", type=int, default=45)
    p.add_argument("--outer-ny", type=int, default=25)
    p.add_argument("--outer-nx", type=int, default=89)
    args = p.parse_args()

    meta = prep_event(args.member, parse_time(args.time), args.lat0,
                      args.lon0, args.outdir, half_days=args.half_days,
                      dlat=args.dlat, dlon=args.dlon,
                      inner_ny=args.inner_ny, inner_nx=args.inner_nx,
                      outer_ny=args.outer_ny, outer_nx=args.outer_nx)
    n = len(meta["times"]) * len(meta["grids"])
    print(f"wrote {n} grid files under {args.outdir} "
          f"({meta['member']}, event {meta['times'][meta['event_index'] - 1]})")


if __name__ == "__main__":
    main()
