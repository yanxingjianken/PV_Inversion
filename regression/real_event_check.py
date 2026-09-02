"""Run one real event and print the numbers that say whether it worked.

Not a unit test: it reads model output, so it belongs outside the suite that has
to run in a minute.  What it checks is the property the method rests on -- that
the pieces sum to the all-sources inversion -- together with the physical
attribution, which is the part a passing algebra test cannot vouch for.

    PYTHONPATH=src micromamba run -n blocking python regression/real_event_check.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from pvinv_sph.config import InversionConfig, MirrorConfig
from pvinv_sph.io import climatology_slot, load_cesm_state
from pvinv_sph.levels import build_levels
from pvinv_sph.passd import all_sources_inversion, invert_pieces
from pvinv_sph.prepare import prepare_state
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps
from pvinv_sph.winds import rotational_wind_stack

CESM_ROOT = "/net/flood/data2/users/x_yan/cesm-blocking/cesm_6hourly"
DEFAULT_EVENT = f"{CESM_ROOT}/cesm2_lens2_wu9_nh/lens2_smbb_m100_1990_plev.nc"
DEFAULT_CLIM = (
    f"{CESM_ROOT}/clim/LENS2_smbb91_100_wu9_clim_6hourly_1985_2014.nc"
)


def event_timestamp(path: str, index: int):
    """The timestamp of one step, read from the file."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return ds["time"].values[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", default=DEFAULT_EVENT)
    parser.add_argument("--clim-file", default=DEFAULT_CLIM)
    parser.add_argument("--time-index", type=int, default=120)
    parser.add_argument("--nlat", type=int, default=96)
    parser.add_argument("--nlon", type=int, default=192)
    parser.add_argument("--lmax", type=int, default=63)
    args = parser.parse_args(argv)

    state = load_cesm_state(args.event_file, args.time_index)
    # Read the date off the event file rather than taking it as an argument: two
    # independent arguments let the event and its climatology drift apart, and a
    # run against the wrong day looks exactly like a converged run against the
    # right one.
    stamp = event_timestamp(args.event_file, args.time_index)
    slot = climatology_slot(stamp.month, stamp.day, stamp.hour, args.clim_file)
    clim = load_cesm_state(args.clim_file, slot, state.p_hpa)
    print(
        f"event {stamp.isoformat()} (index {args.time_index})  "
        f"climatology slot {slot}"
    )

    levels = build_levels("NL9")
    ops = SphereOps(SHT(gaussian_grid(args.nlat, args.nlon), lmax=args.lmax))
    cfg = InversionConfig(mirror=MirrorConfig(blend=True))

    started = time.time()
    mean = prepare_state(*clim.as_tuple(), clim.lat, clim.lon, levels, ops)
    event = prepare_state(*state.as_tuple(), state.lat, state.lon, levels, ops)
    prepared = time.time()
    result = invert_pieces(ops, levels, mean, event, cfg=cfg)
    pieces_done = time.time()
    total, report = all_sources_inversion(ops, levels, mean, event, cfg=cfg)
    finished = time.time()

    north = ops.grid.lat > 0
    summed = result.summed_psi()
    u_sum, v_sum = rotational_wind_stack(ops, summed)
    u_all, v_all = rotational_wind_stack(ops, total)
    u_obs, v_obs = rotational_wind_stack(ops, event.psi_spec - mean.psi_spec)

    def rms(field):
        return float(np.sqrt(np.mean(field[:, north, :] ** 2)))

    print(f"prepare {prepared - started:.0f}s   pieces {pieces_done - prepared:.0f}s"
          f"   all-sources {finished - pieces_done:.0f}s")
    print(f"newton steps {result.newton_steps}, converged "
          f"{result.diagnostics['newton_converged']}")
    print(f"linear iterations {result.diagnostics['linear_iterations']}")
    print(f"clamp worst fraction {result.clamp_worst:.4f}")
    print()
    print("closure (the property the decomposition rests on)")
    print(f"  max|sum - all_sources| / max|all_sources| = "
          f"{np.abs(summed - total).max() / np.abs(total).max():.2e}")
    print(f"  rms wind difference {rms(u_sum - u_all):.2e} m/s against a signal of "
          f"{rms(u_all):.2f} m/s")
    print(f"  all-sources solve converged: {report.converged}")
    print()
    print("method residual: what balance cannot represent")
    print(f"  |observed - all_sources| / |observed| = "
          f"{rms(u_obs - u_all) / rms(u_obs):.3f}")
    print()
    print("attribution: projection of each piece on the observed anomaly, 400-200 hPa")
    upper = [3, 4, 5, 6, 7]
    denominator = float(
        np.sum((u_obs * u_obs + v_obs * v_obs)[upper][:, north, :])
    )
    for name, piece in result.pieces.items():
        u_piece, v_piece = rotational_wind_stack(ops, piece.psi_spec)
        numerator = float(
            np.sum((u_piece * u_obs + v_piece * v_obs)[upper][:, north, :])
        )
        print(f"  {name:>5s}  projection {numerator / denominator:+.3f}   "
              f"amplitude {rms(u_piece):6.2f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
