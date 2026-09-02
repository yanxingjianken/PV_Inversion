"""Resolve exp02's fifty events into a catalogue this package can run.

The events are taken from exp02 unchanged, so the windowed row of this comparison
is the very same population -- and in fact the very same files: the production
store is read, never rewritten.

Each row needs the state file and the step within it, and the climatology slot.
Both are resolved by matching the event's own timestamp against the file's time
coordinate rather than by arithmetic on a calendar: the plev files carry a noleap
axis wearing Gregorian labels, and a formula that disagrees by one day produces a
run that looks entirely healthy.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvinv_sph.io import climatology_slot  # noqa: E402

ROOT = "/net/flood/data2/users/x_yan/cesm-blocking/cesm_6hourly"
STATE = f"{ROOT}/cesm2_lens2_wu9_nh"
CLIM = f"{ROOT}/clim/LENS2_smbb91_100_wu9_clim_6hourly_1985_2014.nc"
EXP = Path(__file__).resolve().parent


def main() -> int:
    events = pd.read_csv(EXP / "events_50.csv")
    events["ts"] = pd.to_datetime(events.base_ts)
    rows, missing = [], []
    cache: dict[str, np.ndarray] = {}

    for _, r in events.iterrows():
        stamp = r.ts
        path = f"{STATE}/lens2_smbb_m{int(r.member)}_{stamp.year}_plev.nc"
        if path not in cache:
            if not Path(path).exists():
                missing.append((r.track_id, path))
                continue
            with xr.open_dataset(path) as ds:
                cache[path] = ds["time"].values
        want = cftime.DatetimeNoLeap(stamp.year, stamp.month, stamp.day, stamp.hour)
        hits = np.flatnonzero(cache[path] == want)
        if hits.size != 1:
            missing.append((r.track_id, f"{stamp} not in {Path(path).name}"))
            continue
        rows.append(
            {
                "event_id": f"track_{r.track_id}_{stamp.strftime('%Y%m%d%H')}_dh+0",
                "state_path": path,
                "state_index": int(hits[0]),
                "clim_path": CLIM,
                "clim_index": climatology_slot(stamp.month, stamp.day, stamp.hour, CLIM),
                "lat": float(r.lat0),
                # The centre longitude is converted, never the axis: the transform
                # takes column zero to be the prime meridian, so rolling the data
                # would rotate the solver's frame instead.
                "lon": float(r.lon0) % 360.0,
            }
        )

    out = EXP / "catalogue.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} events -> {out}")
    if missing:
        print(f"unresolved: {missing}")
    lats = [r["lat"] for r in rows]
    print(f"latitudes {min(lats):.2f} to {max(lats):.2f} N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
