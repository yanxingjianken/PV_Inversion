"""What the windowed chain can reach, and what it cannot.

Two real blocking peaks -- one at 55N, one at 80N -- inverted twice each: by the
windowed Wu/Davis Fortran chain as production drives it, and by the global solve.
Both are asked the same question and the answer is drawn the same way:

    the flow at 300 hPa induced by every piece of the decomposition, summed.

Summed rather than piece by piece, because that is the quantity the two methods
can be held to.  Each decomposes the same balanced perturbation; the windowed
chain additionally has a wall response, solved for separately and added back once
(see :mod:`windowed`).  Beside them sits the anomaly itself, which is what both
are trying to account for, and which sets the colour scale -- the windowed field
runs outside it near its walls, and saturating there is the honest way to show
that rather than letting it flatten everything else.

**The projection.**  A latitude-longitude box and a polar cap each work for one
kind of event and fail for the other: a box degenerates towards the pole, where
the meridians meet, and a cap is centred correctly only if the event is polar.
Neither difficulty is really about projection -- it is that the coordinate system
is tied to the earth's axis rather than to the event.  So the frame is rotated to
put the event at its own pole and the figure drawn in an azimuthal projection
about it.  Every event then has identical geometry, radius is great-circle
distance from the centre, and the earth's pole is an ordinary interior point with
nothing special about it.  For the 80N event it lies inside the disc, which is
exactly the case a windowed chain cannot reach and what this figure is for.

    PYTHONPATH=src micromamba run -n blocking python examples/02_polar_event/run.py
    PYTHONPATH=src micromamba run -n blocking python examples/02_polar_event/run.py --from-cache
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pvinv_sph.config import InversionConfig, MirrorConfig  # noqa: E402
from pvinv_sph.io import climatology_slot, load_cesm_state  # noqa: E402
from pvinv_sph.levels import build_levels  # noqa: E402
from pvinv_sph.passd import invert_pieces  # noqa: E402
from pvinv_sph.prepare import prepare_state  # noqa: E402
from pvinv_sph.sht import SHT, gaussian_grid, grid_from_axes  # noqa: E402
from pvinv_sph.sphere import SphereOps  # noqa: E402
from pvinv_sph.winds import rotational_wind_stack  # noqa: E402
from windowed import invert_windowed, scatter_to_globe  # noqa: E402

CESM = "/net/flood/data2/users/x_yan/cesm-blocking/cesm_6hourly"
HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL_HPA = 300.0

#: Radius of the disc drawn about each event, in degrees of great-circle arc.
DISC_RADIUS = 40.0

#: A mid-latitude control and one whose event sits above the windowed solve band.
EVENTS = (
    ("55N control", "m091_t00201", 91, 1992, 1, 10, 12, 55.471138, 157.139292),
    ("80N event", "m094_t00227", 94, 1993, 1, 19, 0, 80.088000, 54.671474),
)


def load_pair(member, year, month, day, hour):
    """The event state and the matching climatology slot."""
    import cftime
    import xarray as xr

    path = f"{CESM}/cesm2_lens2_wu9_nh/lens2_smbb_m{member}_{year}_plev.nc"
    with xr.open_dataset(path) as ds:
        want = cftime.DatetimeNoLeap(year, month, day, hour)
        index = int(np.flatnonzero(ds["time"].values == want)[0])
    state = load_cesm_state(path, index)
    clim_path = f"{CESM}/clim/LENS2_smbb91_100_wu9_clim_6hourly_1985_2014.nc"
    clim = load_cesm_state(
        clim_path, climatology_slot(month, day, hour, clim_path), state.p_hpa
    )
    return state, clim


def to_data_grid(ops, out_sht, u, v):
    """Move a wind pair to the data grid, carried as the band-limited product.

    The components are not band limited on a sphere; ``u cos(lat)`` and
    ``v cos(lat)`` are, so the transfer is exact for those and loses several
    digits for the components themselves.
    """
    cos_solver = ops.grid.cos_lat[:, None]
    cos_out = out_sht.grid.cos_lat[:, None]
    usable = np.abs(cos_out) > 1e-8
    moved = []
    for field in (u, v):
        transferred = ops.sht.regrid_to(out_sht, field * cos_solver)
        result = np.full_like(transferred, np.nan)
        np.divide(transferred, cos_out, out=result, where=usable)
        moved.append(result)
    return moved


def disc_mask(lat, lon, lat0, lon0, radius=DISC_RADIUS):
    """Points within ``radius`` degrees of great-circle arc from the centre."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    cos_c = np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0)) + np.cos(
        np.radians(lat2d)
    ) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2d - lon0))
    return np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0))) <= radius


def compute(args) -> dict:
    levels = build_levels("NL9")
    level = int(np.where(levels.p_hpa == LEVEL_HPA)[0][0])
    ops = SphereOps(
        SHT(gaussian_grid(args.solver_nlat, args.solver_nlon), lmax=args.lmax)
    )
    cache = {}

    for label, track, member, year, month, day, hour, lat0, lon0 in EVENTS:
        state, clim = load_pair(member, year, month, day, hour)
        lon0 = lon0 % 360.0
        print(f"\n{label}  {track}  ({lat0:.2f}N, {lon0:.2f}E)  "
              f"{year}-{month:02d}-{day:02d}", flush=True)

        lat_full = np.concatenate([-state.lat[::-1], state.lat])
        out_sht = SHT(grid_from_axes(lat_full, state.lon), lmax=args.lmax)
        southern = lat_full < state.lat.min()

        started = time.time()
        mean = prepare_state(*clim.as_tuple(), clim.lat, clim.lon, levels, ops)
        event = prepare_state(*state.as_tuple(), state.lat, state.lon, levels, ops)
        result = invert_pieces(
            ops, levels, mean, event,
            cfg=InversionConfig(mirror=MirrorConfig(blend=True)),
        )
        print(f"  global    {time.time() - started:5.0f}s  newton "
              f"{result.newton_steps} converged "
              f"{result.diagnostics['newton_converged']}", flush=True)

        u_sum, v_sum = to_data_grid(
            ops, out_sht, *rotational_wind_stack(ops, result.summed_psi())
        )
        u_obs, v_obs = to_data_grid(
            ops, out_sht, *rotational_wind_stack(ops, event.psi_spec - mean.psi_spec)
        )
        for field in (u_sum, v_sum, u_obs, v_obs):
            field[:, southern, :] = np.nan  # the mirror's invented hemisphere

        started = time.time()
        u_win, v_win, rows, cols, parts = invert_windowed(
            clim.as_tuple(), state.as_tuple(), state.lat, state.lon, lat0, lon0
        )
        wall = float(
            np.sqrt(
                np.nanmean(
                    parts["wall"][0][level] ** 2 + parts["wall"][1][level] ** 2
                )
            )
        )
        print(f"  windowed  {time.time() - started:5.0f}s  band "
              f"{state.lat[rows].min():.1f}-{state.lat[rows].max():.1f}N, "
              f"{cols.size} columns; wall response {wall:.1f} m/s", flush=True)

        windowed = {}
        for name, field in (("u", u_win), ("v", v_win)):
            full = np.full((levels.nlev, lat_full.size, state.lon.size), np.nan)
            full[:, state.lat.size :, :] = scatter_to_globe(
                field, rows, cols, state.lat.size, state.lon.size
            )
            windowed[name] = full

        key = label.replace(" ", "_")
        cache[f"{key}_lat"] = lat_full
        cache[f"{key}_lon"] = state.lon
        cache[f"{key}_centre"] = np.array([lat0, lon0])
        cache[f"{key}_wall_rms"] = np.array(wall)
        cache[f"{key}_band_north"] = np.array(state.lat[rows].max())
        disc = disc_mask(lat_full, state.lon, lat0, lon0)
        for name, (u, v) in (
            ("windowed", (windowed["u"][level], windowed["v"][level])),
            ("global", (u_sum[level], v_sum[level])),
            ("observed", (u_obs[level], v_obs[level])),
        ):
            cache[f"{key}_{name}_u"] = u
            cache[f"{key}_{name}_v"] = v
            speed = np.hypot(u, v)[disc]
            print(f"    {name:9s} disc: {np.isnan(speed).mean():5.1%} empty, "
                  f"peak {np.nanmax(speed):6.1f} m/s, "
                  f"98th pct {np.nanpercentile(speed, 98):5.1f} m/s", flush=True)
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=HERE)
    parser.add_argument("--solver-nlat", type=int, default=96)
    parser.add_argument("--solver-nlon", type=int, default=192)
    parser.add_argument("--lmax", type=int, default=63)
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args(argv)

    path = os.path.join(args.out_dir, "fields.npz")
    if args.from_cache:
        cache = dict(np.load(path))
        print(f"redrawing from {path}")
    else:
        cache = compute(args)
        np.savez_compressed(path, **cache)
        print(f"\nwrote {path}")
    _plot(cache, os.path.join(args.out_dir, "polar_cap.png"))
    return 0


def _plot(cache: dict, path: str) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt

    # The two reconstructions, the anomaly they are reconstructing, and what each
    # of them missed.  The differences carry their own scale, shared between the
    # two so that "which method missed more" is readable off the colour.
    columns = (
        ("windowed", "windowed chain, ibc=1"),
        ("global", "global inversion"),
        ("observed", "observed anomaly"),
        ("windowed-observed", "windowed - observed"),
        ("global-observed", "global - observed"),
    )
    labels = [label for label, *_ in EVENTS]
    keys = [label.replace(" ", "_") for label in labels]

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.c_[np.sin(theta), np.cos(theta)] * 0.5 + 0.5)

    fig = plt.figure(figsize=(25, 10.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)

    for row, key in enumerate(keys):
        lat, lon = cache[f"{key}_lat"], cache[f"{key}_lon"]
        lat0, lon0 = cache[f"{key}_centre"]
        # The frame is the event's own: it becomes the pole of the projection, so
        # both rows are the same picture of the same disc.
        projection = ccrs.AzimuthalEquidistant(
            central_longitude=float(lon0), central_latitude=float(lat0)
        )
        disc = disc_mask(lat, lon, lat0, lon0)
        speed_obs = np.hypot(cache[f"{key}_observed_u"], cache[f"{key}_observed_v"])
        vmax = float(np.nanpercentile(speed_obs[disc], 98))
        lon_closed = np.append(lon, 360.0)

        def field_for(name):
            """The three stored fields, and the two differences from the anomaly."""
            if "-" in name:
                first, second = name.split("-")
                return (
                    cache[f"{key}_{first}_u"] - cache[f"{key}_{second}_u"],
                    cache[f"{key}_{first}_v"] - cache[f"{key}_{second}_v"],
                )
            return cache[f"{key}_{name}_u"], cache[f"{key}_{name}_v"]

        diff_max = max(
            float(np.nanpercentile(np.hypot(*field_for(name))[disc], 98))
            for name in ("windowed-observed", "global-observed")
        )

        for col, (name, title) in enumerate(columns):
            axis = fig.add_subplot(grid[row, col], projection=projection)
            axis.set_boundary(circle, transform=axis.transAxes)
            radius = DISC_RADIUS * 111.195e3  # degrees of arc to metres
            axis.set_xlim(-radius, radius)
            axis.set_ylim(-radius, radius)

            u, v = field_for(name)
            speed = np.hypot(u, v)
            is_difference = "-" in name
            limit = diff_max if is_difference else vmax
            image = axis.pcolormesh(
                lon_closed,
                lat,
                np.c_[speed, speed[:, :1]],
                transform=ccrs.PlateCarree(),
                cmap="viridis_r" if is_difference else "magma_r",
                vmin=0.0,
                vmax=limit,
                shading="auto",
            )
            axis.quiver(
                lon,
                lat,
                u,
                v,
                transform=ccrs.PlateCarree(),
                regrid_shape=20,
                scale=650,
                width=0.005,
                color="0.12",
            )
            axis.plot(
                lon0, lat0, "c*", markersize=18, markeredgecolor="k",
                markeredgewidth=0.9, transform=ccrs.PlateCarree(), zorder=6,
            )
            # The earth's pole, marked so it can be seen to be an ordinary point.
            axis.plot(
                0.0, 90.0, "o", color="white", markersize=7, markeredgecolor="k",
                markeredgewidth=1.0, transform=ccrs.PlateCarree(), zorder=6,
            )
            axis.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            axis.gridlines(
                linewidth=0.4, color="0.55", alpha=0.8,
                ylocs=range(0, 91, 15), xlocs=range(-180, 181, 30),
            )
            empty = float(np.isnan(speed[disc]).mean())
            above = float(np.nanmean(speed[disc] > limit))
            inside = float(np.sqrt(np.nanmean(speed[disc] ** 2)))
            note = f"{empty:.0%} of the disc empty" if empty > 0.005 else "complete"
            if above > 0.005:
                note += f", {above:.0%} above the scale"
            axis.set_title(
                f"{title}\n{note};  rms {inside:.1f} m/s", fontsize=9.5
            )
            if col == 0:
                axis.text(
                    -0.06, 0.5,
                    f"{labels[row]}\ncentre {float(lat0):.1f}N "
                    f"{float(lon0):.0f}E\nwall response "
                    f"{float(cache[f'{key}_wall_rms']):.1f} m/s",
                    transform=axis.transAxes, rotation=90,
                    va="center", ha="center", fontsize=10,
                )
            fig.colorbar(
                image, ax=axis, shrink=0.72, pad=0.04,
                label=(
                    "|missed| at 300 hPa [m s$^{-1}$]"
                    if is_difference
                    else "induced |v| at 300 hPa [m s$^{-1}$]"
                ),
            )
    fig.suptitle(
        "Flow at 300 hPa induced by all pieces summed, against the anomaly itself\n"
        f"azimuthal equidistant about each event, {DISC_RADIUS:.0f}-degree disc; "
        "star the event, circle the north pole; colour scale set by the anomaly",
        fontsize=12,
    )
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
