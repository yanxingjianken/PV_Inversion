#!/usr/bin/env python3
"""00 — tune the object threshold f, the prerequisite for every figure in this folder.

THE OBJECT (final spec, 2026-08-06)
-----------------------------------
    q'_k4 = Pi_{k<=4}[pv_anom]        filtered GLOBALLY (a wavenumber needs the full ring),
                                      then cropped to the block-relative window
    binary = q'_k4 < f * q'_k4[seed]  seed = the tracked centre at 250 hPa
    M      = the 6-connected 3-D component of `binary` containing the seed,
             over 400/300/250/200 (100 hPa EXCLUDED — the archive PV there is ~47% missing,
             worst at high latitude), used as a HARD 0/1 object (no smoothing)

The object itself is `_evt.build_object`, imported rather than reimplemented — this script and
figures 01-04 must never disagree about what the object is.

WHY "MAXIMISE THE PASS RATE" WOULD BE THE WRONG TARGET
------------------------------------------------------
All three acceptance criteria get EASIER as f loosens, so maximising the pass rate drives f -> 0,
where the object swallows every negative-PV point in the window and the criteria pass vacuously.
With the area cap dropped, the non-degenerate reading of "tune it to work for all cases" is:

    take the STRICTEST f whose pass rate is still >= PASS_TARGET.

That is what this script reports. The full pass-rate curve is printed so the choice is visible
rather than asserted.

ACCEPTANCE CRITERIA (user-specified)
    1. object non-empty AND contains the tracked centre at 250 hPa
    2. 3-D connected, 6-connectivity   (satisfied by construction — one labelled component)
    3. spans >= 2 pressure levels

Validation set: onset/peak/decay x dh=0 x prp/block, sampled across members and decades.

Run:  micromamba run -n blocking python 00_threshold_scan.py [--per-combo 30]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import xarray as xr

sys.path.insert(0, "/net/flood/data2/users/x_yan/cesm-blocking/08_triad_resonance")
import build_pvbudget_15deg as Bg                                    # noqa: E402
from wave_diagnostic import zonal_bandpass                           # noqa: E402
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import _evt                                                          # noqa: E402

WU = list(Bg.WU)                                    # [1000,850,700,500,400,300,250,200,100]
OBJ_HPA = _evt.OBJ_HPA                              # [400,300,250,200] — 100 hPa excluded
OBJ_IDX = _evt.OBJ_IDX
SEED_LEV = _evt.SEED_LEV_IN_OBJ                     # row of 250 hPa WITHIN the object subset
KMAX = 4


def k4_upper_window(pv9, pvbar9, jb, iw):
    """Pi_{k<=4}[pv_anom] on the OBJECT levels, filtered globally then cropped. (4, NYb, NXw)."""
    anom = (pv9 - pvbar9)[:, jb, :].astype(np.float64)      # (9, NYb, nlon) GLOBAL longitudes
    bad = ~np.isfinite(anom)
    k4 = zonal_bandpass(np.where(bad, 0.0, anom), m_min=0, m_max=KMAX, lon_axis=-1)
    k4 = np.where(bad, np.nan, k4)
    return k4[OBJ_IDX][..., iw]


def object_3d(q4, jcen, icen, f, search_rad=8):
    """The SAME object as figures 01-04 (`_evt.build_object`), plus 00's failure-reason contract."""
    sub_ok = np.isfinite(q4[SEED_LEV]).any() and np.nanmin(q4[SEED_LEV]) < 0
    if not sub_ok:
        return None, None, {"reason": "no negative PV at 250 hPa in the window"}
    M, obj, info = _evt.build_object(q4, jcen, icen, f=f, search_rad=search_rad)
    if not obj.any():
        return None, None, {"reason": f"seed not inside any object at f={f}"}
    return M, obj, info


def accepts(info):
    """The three user-specified criteria. (2) holds by construction — one labelled component."""
    if info.get("reason"):
        return False
    return info["nlev"] >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-combo", type=int, default=30)
    ap.add_argument("--pass-target", type=float, default=0.99)
    ap.add_argument("--f", type=float, nargs="+",
                    default=[0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10])
    args = ap.parse_args()

    Bg._load_clim15()
    allrows, _ = Bg.build_tasks(["block", "prp"])
    rng = np.random.default_rng(0)
    sel = []
    for grp in ("block", "prp"):
        for st in ("onset", "peak", "decay"):
            d = allrows[(allrows["group"] == grp) & (allrows["stage"] == st)]
            take = d.iloc[rng.choice(len(d), size=min(args.per_combo, len(d)), replace=False)]
            sel.append((grp, st, take))
    print(f"validation set: {sum(len(t) for _,_,t in sel)} events "
          f"({args.per_combo} per group x stage, dh=0)\n")

    res = defaultdict(lambda: defaultdict(list))          # f -> (grp,stage) -> [pass?]
    area = defaultdict(list)
    for grp, st, take in sel:
        for (mem, dec), g in take.groupby(["member", "decade"]):
            fp = Bg.PVDIR / f"m{mem}" / f"cesm2_lens2_pv9_15deg_m{mem:02d}_d{dec}.nc"
            if not fp.exists():
                continue
            with xr.open_dataset(fp) as ds:
                lat, lon = ds["lat"].values, ds["lon"].values
                nlon = lon.size
                didx = {(int(t.year), int(t.month), int(t.day)): i
                        for i, t in enumerate(ds["time"].values)}
                jb = np.where((lat >= Bg.BAND_S) & (lat <= Bg.BAND_N))[0][::-1]
                band = lat[jb]
                for r in g.to_dict("records"):
                    ti = didx.get((int(r["year"]), int(r["month"]), int(r["day"])))
                    if ti is None:
                        continue
                    pv9 = ds["pv"].sel(lev=WU).isel(time=ti).load().values
                    mo, di = int(r["month"]), int(r["day"]) - 1
                    clat, clon = float(r["lat"]), float(r["lon180"]) % 360.0
                    ic = int(np.abs(((lon - clon + 180) % 360) - 180).argmin())
                    iw = (np.arange(-Bg.WIN_PAD, Bg.WIN_PAD + 1) + ic) % nlon
                    jcen = int(np.abs(band - clat).argmin())
                    q4 = k4_upper_window(pv9, Bg._CLIM15[mo]["pv"][di], jb, iw)
                    for f in args.f:
                        _, _, info = object_3d(q4, jcen, Bg.WIN_PAD, f)
                        ok = accepts(info)
                        res[f][(grp, st)].append(ok)
                        if ok:
                            area[f].append(info["area_frac"])

    print("  f      pass rate by group x stage                                   overall   "
          "median area")
    print("         blk-ons blk-pk blk-dec prp-ons prp-pk prp-dec")
    best = None
    for f in args.f:
        cells, allv = [], []
        for grp in ("block", "prp"):
            for st in ("onset", "peak", "decay"):
                v = res[f][(grp, st)]
                cells.append(np.mean(v) if v else np.nan)
                allv += v
        overall = float(np.mean(allv)) if allv else np.nan
        med = float(np.median(area[f])) if area[f] else np.nan
        mark = ""
        if overall >= args.pass_target and best is None:
            best = f; mark = "  <== strictest f meeting the target"
        print(f"  {f:<5.2f}  " + " ".join(f"{c:6.2f} " for c in cells)
              + f"  {overall:6.3f}    {med:8.3f}{mark}")
    print()
    if best is None:
        print(f"  NO f reaches the {args.pass_target:.0%} target — loosen the sweep or revisit "
              f"the criteria.")
    else:
        print(f"  CHOSEN f = {best}  (strictest value with pass rate >= {args.pass_target:.0%})")
        print(f"  median object occupies {np.median(area[best]):.1%} of the "
              f"{len(OBJ_HPA)}-level object window")


if __name__ == "__main__":
    main()
