#!/usr/bin/env python3
"""01 — where the block object is: pv_anom with the `_p` object drawn on top.

Two panels for one BLOCK peak event at dh=0:
  (a) y-p section through the block centre longitude — 1000 to 200 hPa
  (b) x-y at 250 hPa

Both shade the ARCHIVE `pv_anom` and outline the object. The object is defined on the k<=4 field
over 400/300/250/200, so on (a) it can only appear above 400 hPa — that is the definition, not a
plotting artefact, and the piece boundary is drawn to make it obvious. The y-p panel stops at
200 hPa: 100 hPa is not an object level and its archive PV is ~47% missing, so plotting it would
show mostly white space that reads as a physical absence rather than a data gap.

Also caches the event + inversion to `_cache.npz` so 02-04 do not repeat the ~5 min PPVI.

Run:  micromamba run -n blocking python 01_pv_anom_sections.py [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _evt

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache.npz"
GRID = 1.5
# NOTE: `_evt.load` crops in LONGITUDE only — the window keeps all 51 Wu-band rows
# (10.5-85.5N). Relative latitude is therefore band - clat, 51 values, NOT the 55-row
# +-40.5 ordinate the pvtend budget patch uses.


def get_event(seed, rebuild=False):
    """Load + invert once, cache to disk. 02-04 call this and get the cached copy."""
    rec = _evt.pick_event(seed=seed)
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        if int(z["seed"]) == seed:
            return rec, z
    e = _evt.load(rec)
    p, q = _evt.split_pv(e)
    psi = _evt.invert(e)
    up, vp = _evt.winds(e, psi["p"])
    ue, ve = _evt.winds(e, psi["e"])
    ut, vt = _evt.winds(e, psi["tot"])
    ua, va = _evt.rot_anom_reference(e, rec)
    np.savez_compressed(
        CACHE, seed=seed, meta=np.array(str(e.meta)), band=e.band, jcen=e.jcen, icen=e.icen,
        pv_anom=e.pv_anom, k4=e.k4, M=e.M, obj=e.obj, info=np.array(str(e.info)),
        pv_anom_p=p, pv_anom_e=q,
        psi_p=psi["p"], psi_e=psi["e"], psi_tot=psi["tot"],
        u_p=up, v_p=vp, u_e=ue, v_e=ve, u_tot=ut, v_tot=vt,
        u_rot_anom=ua, v_rot_anom=va,
        ev_z=e.ev[0], ev_t=e.ev[1], ev_u=e.ev[2], ev_v=e.ev[3],
        cl_z=e.cl[0], cl_t=e.cl[1], cl_u=e.cl[2], cl_v=e.cl[3])
    return rec, np.load(CACHE, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--lon-half", type=float, default=0.0,
                    help="average the y-p section over +-this many deg of rel. lon (0 = single column)")
    args = ap.parse_args()

    rec, z = get_event(args.seed, args.rebuild)
    meta = eval(str(z["meta"]))
    info = eval(str(z["info"]))
    pv, k4, M, obj = z["pv_anom"], z["k4"], z["M"], z["obj"]
    jcen, icen = int(z["jcen"]), int(z["icen"])
    WU = np.array(_evt.WU, float)
    rel_lon = np.arange(-icen, icen + 1) * GRID
    band = z["band"]                      # N->S absolute band latitudes, 51 rows
    rel_lat = band - float(meta["clat"])  # N->S

    # M lives on the OBJECT levels only -> lift to all 9 for plotting, NaN elsewhere
    M9 = np.full(pv.shape, np.nan)
    M9[_evt.OBJ_IDX] = M
    O9 = np.zeros(pv.shape, bool)
    O9[_evt.OBJ_IDX] = obj
    NP = len(WU) - 1                       # y-p panel: 1000..200, drop 100 hPa (see the docstring)
    WUP = WU[:NP]

    h = max(1, int(round(args.lon_half / GRID)))
    csl = slice(icen - h + 1, icen + h) if args.lon_half > 0 else slice(icen, icen + 1)
    sec = lambda a: np.nanmean(a[..., csl], axis=-1)[..., ::-1]          # -> ascending lat
    y = rel_lat[::-1]
    L250 = _evt.WU.index(250)

    vmax = float(np.nanpercentile(np.abs(pv[_evt.OBJ_IDX]), 99.5))
    lev = np.linspace(-vmax, vmax, 21)

    # width_ratios: panel (b) is aspect-equal over 180 deg lon x 75 deg lat = 2.4:1, so it needs
    # a wider box than the y-p panel or matplotlib shrinks it and leaves the row mostly white.
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.0),
                             gridspec_kw=dict(width_ratios=[1.0, 1.85]))

    # ── (a) y-p ────────────────────────────────────────────────────────────────
    ax = axes[0]
    cf = ax.contourf(y, WUP, sec(pv)[:NP], levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(y, WUP, sec(M9)[:NP], levels=[0.5], colors="k", linewidths=2.0)
    with np.errstate(invalid="ignore"):
        ax.contourf(y, WUP, sec(O9.astype(float))[:NP], levels=[0.5, 1.5],
                    colors="none", hatches=["//"], zorder=3)
    # the object is cut from the k<=4 field, NOT from the shading — draw it so the reader is not
    # invited to conclude the threshold is too tight when the two are simply different fields
    _k = sec(k4)[:NP]
    _kl = np.nanpercentile(np.abs(_k), 97) * np.array([0.35, 0.7])
    ax.contour(y, WUP, _k, levels=-_kl[::-1], colors="#1a6b3a", linewidths=1.0)
    ax.contour(y, WUP, _k, levels=_kl, colors="#8a5a00", linewidths=1.0, linestyles=":")
    ax.axhline(450, color="0.25", lw=1.2, ls="--")
    # The line is the SEARCH limit, not the object. That the object stops at 300 while the search
    # allowed 400 is a result about this event, so the two must not be conflated in the label.
    ax.text(0.03, 0.045, "object was searched over 400–200 hPa only",
            transform=ax.transAxes, fontsize=8, color="0.25")
    ax.set_yscale("log"); ax.set_ylim(1000, 200); ax.set_yticks(WUP)
    ax.set_yticklabels([f"{int(p)}" for p in WUP], fontsize=8); ax.minorticks_off()
    ax.set_xlabel("relative latitude [deg]"); ax.set_ylabel("pressure [hPa]")
    ax.set_title(f"$\\mathbf{{(a)}}$ $y$–$p$ at rel. lon "
                 f"{'0' if args.lon_half == 0 else f'±{args.lon_half:g}°'}", fontsize=11)

    # ── (b) x-y at 250 hPa ─────────────────────────────────────────────────────
    ax = axes[1]
    cf = ax.contourf(rel_lon, y, pv[L250][::-1], levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(rel_lon, y, M9[L250][::-1], levels=[0.5], colors="k", linewidths=2.0)
    ax.contourf(rel_lon, y, O9[L250][::-1].astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["//"], zorder=3)
    _k2 = k4[L250][::-1]
    _kl2 = np.nanpercentile(np.abs(_k2), 97) * np.array([0.35, 0.7])
    ax.contour(rel_lon, y, _k2, levels=-_kl2[::-1], colors="#1a6b3a", linewidths=1.0)
    ax.contour(rel_lon, y, _k2, levels=_kl2, colors="#8a5a00", linewidths=1.0, linestyles=":")
    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.set_aspect("equal"); ax.set_xlabel("relative longitude [deg]")
    ax.set_ylabel("relative latitude [deg]")
    ax.set_title(r"$\mathbf{(b)}$ 250 hPa", fontsize=11)

    fig.suptitle(
        f"01 · pv_anom and the block object — m{meta['member']} "
        f"{meta['year']}-{meta['month']:02d}-{meta['day']:02d} {meta['stage']} "
        f"{meta['label']} @ {meta['clat']:.1f}°N {meta['clon']:.0f}°E\n"
        f"object: 3-D 6-connected on the $k\\leq4$ field, 400–200 hPa, f={info['f']}, "
        f"spans {info['levels']} hPa, {100*info['area_frac']:.1f}% of the object window.  "
        f"Black outline / hatching = the object (hard 0/1, no smoothing).\n"
        f"Green solid / orange dotted = the $k\\leq4$ field the object was CUT FROM (−/+); the "
        f"shading is the FULL spectrum, which is why the object looks small against it.",
        fontsize=9.5)
    fig.subplots_adjust(top=0.815, left=0.05, right=0.90, bottom=0.105, wspace=0.16)
    # ONE colourbar (both panels share `lev`), placed explicitly so it cannot cover panel (b)
    _evt.side_colorbar(fig, cf, axes, "pv_anom [PVU]", shrink=0.86)
    out = HERE / "01_pv_anom_sections.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out.name}")
    print(f"  event  : m{meta['member']} {meta['year']}-{meta['month']:02d}-{meta['day']:02d} "
          f"{meta['stage']} {meta['label']}")
    print(f"  object : f={info['f']} levels={info['levels']} area={100*info['area_frac']:.1f}% "
          f"seed={info['seed_val']:.2f} PVU")
    print(f"  250 hPa: p99|pv_anom|={np.nanpercentile(np.abs(pv[L250]),99):.2f}  "
          f"|_p|={np.nanpercentile(np.abs(z['pv_anom_p'][L250]),99):.2f}  "
          f"|_e|={np.nanpercentile(np.abs(z['pv_anom_e'][L250]),99):.2f} PVU")


if __name__ == "__main__":
    main()
