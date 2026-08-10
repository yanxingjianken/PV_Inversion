#!/usr/bin/env python3
"""04 — do the Wu-PV split and the Ertel-PV split agree, piece by piece?

Same spirit as `wu_cesm/closure_check.png` (Wu balanced PV vs the input Ertel PV), but resolved
into the two PIECES rather than only the total.

WHY THE LITERAL CHECK WOULD BE VACUOUS
--------------------------------------
`Q_block + Q_eddy - Q_total` is identically zero: the eddy piece is DEFINED as the total minus the
block piece. Checking it proves nothing. It is reported as a number and not plotted.

WHAT ACTUALLY NEEDS CHECKING BEFORE THE MASS LAUNCH
---------------------------------------------------
Two different PV variables are in play, deliberately:

  * the ARCHIVE Ertel PV  — what the budget differentiates in grad q'_p and grad q'_e
  * the WU CORE's own PV  — what the inversion is fed, because the two are not pointwise
                            interchangeable (the core re-differentiates theta over only 9 coarse
                            pressure levels; measured slope 79-94 rather than the exact 100)

The SAME mask M is applied to both. If the two splits disagree, then `-V_p . grad q_e` is built
from a wind and a gradient that do not refer to the same object, and the cross terms are
inconsistent — which would not show up anywhere downstream. That is what this figure measures.

Columns: total / block / eddy.  Top row: the Ertel piece shaded with the Wu piece contoured on it.
Bottom row: Wu vs Ertel scatter with the fitted slope, r, and NRMS.

Comparison is on the INTERIOR levels only (850-200). Levels 1000 and 100 carry boundary theta, not
interior PV, in the Wu formulation — including them would compare a quantity against nothing. All
three pieces are scored on the SAME valid points (see VALID below) so the columns are comparable.

Reads `_cache.npz` from 01, and recomputes the Wu PV directly (cheap, no inversion).

Run:  micromamba run -n blocking python 04_pv_closure.py
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
INTERIOR = [850, 700, 500, 400, 300, 250, 200]          # Wu levels 2..8 — where interior PV exists


def stats(a, b):
    """b (Wu) against a (Ertel): slope through the origin, r, and NRMS of b-a scaled by std(a)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return dict(slope=np.nan, r=np.nan, nrms=np.nan, n=int(m.sum()))
    x, y = a[m], b[m]
    return dict(slope=float((x * y).sum() / (x * x).sum()),
                r=float(np.corrcoef(x, y)[0, 1]),
                nrms=float(np.sqrt(np.mean((y - x) ** 2)) / np.std(x)),
                n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lev", type=int, default=250, help="pressure level for the map row")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit("no _cache.npz — run 01_pv_anom_sections.py first")
    z = np.load(CACHE, allow_pickle=True)
    meta, info = eval(str(z["meta"])), eval(str(z["info"]))

    # rebuild the Event to get the Wu PV — the load is cheap, the inversion is what was expensive
    rec = _evt.pick_event(seed=args.seed)
    e = _evt.load(rec)
    if int(z["seed"]) != args.seed or not np.allclose(e.M, z["M"], equal_nan=True):
        raise SystemExit("cache and freshly-loaded event disagree — rerun 01 --rebuild")
    wu_tot, wu_p, wu_e = _evt.wu_pv_split(e)

    ER = {"total": z["pv_anom"], "block": z["pv_anom_p"], "eddy": z["pv_anom_e"]}
    WUq = {"total": wu_tot, "block": wu_p, "eddy": wu_e}

    ident = float(np.nanmax(np.abs(wu_p + wu_e - wu_tot)))       # the vacuous check, as a number
    kint = [_evt.WU.index(p) for p in INTERIOR]
    #: Score all three pieces on the SAME points. `pv_anom_p` is 0 (finite) wherever the archive PV
    #: is missing, because it is built as `zeros + M*k4`, and `wu_p` has the same property on the
    #: Wu side. Without this the block column would be scored on ~1k extra exact (0,0) pairs the
    #: other two columns never see, and the three n would not be comparable.
    VALID = (np.isfinite(ER["total"][kint]) & np.isfinite(WUq["total"][kint])).ravel()
    ST = {k: stats(np.where(VALID, ER[k][kint].ravel(), np.nan),
                   WUq[k][kint].ravel()) for k in ER}

    kL = _evt.WU.index(args.lev)
    icen = int(z["icen"])
    y = (z["band"] - float(meta["clat"]))[::-1]
    x = np.arange(-icen, icen + 1) * GRID
    ix = np.abs(x) <= 60.0
    xc = x[ix]
    M250 = z["M"][_evt.OBJ_HPA.index(args.lev)][::-1] if args.lev in _evt.OBJ_HPA else None

    vmax = float(np.nanpercentile(np.abs(ER["total"][kL]), 99.5))
    lev = np.linspace(-vmax, vmax, 21)
    cl = vmax * np.array([0.25, 0.5, 0.75])

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6),
                             gridspec_kw=dict(height_ratios=[0.80, 1.0]))
    lbl = {"total": r"total  $q'$", "block": r"block  $q'_p$", "eddy": r"eddy  $q'_e$"}

    for j, key in enumerate(("total", "block", "eddy")):
        # ── map: Ertel shaded, Wu contoured on top ───────────────────────────────────────
        ax = axes[0, j]
        a = ER[key][kL][::-1][:, ix]
        b = WUq[key][kL][::-1][:, ix]
        cf = ax.contourf(xc, y, a, levels=lev, cmap="RdBu_r", extend="both")
        with np.errstate(invalid="ignore"):
            ax.contour(xc, y, b, levels=-cl[::-1], colors="k", linewidths=1.1)
            ax.contour(xc, y, b, levels=cl, colors="k", linewidths=1.1, linestyles=":")
        if M250 is not None:
            ax.contour(xc, y, M250[:, ix], levels=[0.5], colors="#00c853", linewidths=1.8)
        ax.plot(0, 0, "k+", ms=12, mew=2)
        ax.set_aspect("equal")
        ax.set_title(f"$\\mathbf{{({chr(97+j)})}}$  {lbl[key]}  @ {args.lev} hPa", fontsize=11)
        ax.set_xlabel("rel. lon [deg]")
        if j == 0:
            ax.set_ylabel("rel. lat [deg]")

        # ── scatter over ALL interior levels ─────────────────────────────────────────────
        ax = axes[1, j]
        A, B = ER[key][kint].ravel(), WUq[key][kint].ravel()
        m = VALID & np.isfinite(A) & np.isfinite(B)      # identical support in all three columns
        ax.hexbin(A[m], B[m], gridsize=55, bins="log", cmap="Greys", mincnt=1, linewidths=0)
        lo, hi = np.nanpercentile(A[m], [0.05, 99.95])
        pad = 0.08 * (hi - lo)
        t = np.linspace(lo - pad, hi + pad, 5)
        ax.plot(t, t, color="#0b5394", lw=1.4, label="1:1")
        ax.plot(t, ST[key]["slope"] * t, color="#c0392b", lw=1.4, ls="--",
                label=f"slope {ST[key]['slope']:.3f}")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(t[0], t[-1])
        ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("archive Ertel  [PVU]")
        if j == 0:
            ax.set_ylabel("Wu core PV  [PVU]")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_title(f"$\\mathbf{{({chr(100+j)})}}$  r = {ST[key]['r']:.4f},  "
                     f"NRMS = {ST[key]['nrms']:.3f},  n = {ST[key]['n']:,}", fontsize=10)

    fig.suptitle(
        f"04 · does the Wu-PV split match the Ertel-PV split? — m{meta['member']} "
        f"{meta['year']}-{meta['month']:02d}-{meta['day']:02d} {meta['stage']} {meta['label']} "
        f"@ {meta['clat']:.1f}°N {meta['clon']:.0f}°E   ·   green = the object at {args.lev} hPa\n"
        f"Shading = archive Ertel (what the budget differentiates); black contours = Wu core PV "
        f"(what the inversion was fed), $\\pm$25/50/75% of the total's range, same mask M.\n"
        f"Scatter is over interior levels 850–200, on identical valid points for all three columns.  "
        f"$Q_p+Q_e-Q_{{tot}}$ = {ident:.1e} PVU — exact by construction, hence not plotted.",
        fontsize=9.5)
    fig.subplots_adjust(top=0.845, left=0.05, right=0.90, bottom=0.070, hspace=0.34, wspace=0.20)
    # top row only — the scatter row carries no shading, so a full-height bar would imply otherwise
    _evt.side_colorbar(fig, cf, axes[0, :], "archive Ertel $q'$ [PVU] (shading)", shrink=0.88)
    out = HERE / "04_pv_closure.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out.name}")
    print(f"  event : m{meta['member']} {meta['year']}-{meta['month']:02d}-{meta['day']:02d} "
          f"{meta['stage']} {meta['label']}  object {info['levels']} hPa")
    print(f"  interior levels {INTERIOR}")
    print(f"  {'piece':7s} {'slope':>8s} {'r':>8s} {'NRMS':>8s}   {'n':>10s}")
    for k in ("total", "block", "eddy"):
        s = ST[k]
        print(f"  {k:7s} {s['slope']:8.4f} {s['r']:8.4f} {s['nrms']:8.4f}   {s['n']:10,d}")
    print(f"  identity |Q_p+Q_e-Q_tot|max = {ident:.3e} PVU")


if __name__ == "__main__":
    main()
